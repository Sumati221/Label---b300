#!/usr/bin/env python3
"""Import controlled label inputs exported from the Databricks Workspace.

The application deliberately preserves the original binary source artwork and
specification documents.  It records only facts that can be read locally from a
DOCX; a missing printed-size requirement is explicitly marked for review rather
than guessed from the artwork pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


IMAGE_SUFFIXES = {".png", ".svg", ".jpg", ".jpeg"}


def canonical_id(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^LS[-_]", "", stem, flags=re.I)
    stem = re.sub(r"(?:[-_ ]SPEC(?:IFICATION)?|[-_ ]\d{4}-\d{2}-\d{2}.*)$", "", stem, flags=re.I)
    match = re.match(r"([A-Za-z0-9]+)", stem)
    return match.group(1).upper() if match else ""


def clean_source_files(folder: Path, suffixes: set[str]):
    for source in sorted(folder.iterdir()):
        if not source.is_file() or source.suffix.lower() not in suffixes:
            continue
        # The Workspace UI produced timestamped copies of the same guide.  The
        # canonical file is retained once; byte-level de-duplication is applied
        # later too.
        if re.search(r"\s20\d\d-\d\d-\d\d\s", source.name):
            continue
        yield source


def copy_unique(sources, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    copied, skipped, hashes = [], [], set()
    for existing in destination.iterdir():
        if existing.is_file():
            hashes.add(hashlib.sha256(existing.read_bytes()).hexdigest())
    for source in sources:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target = destination / source.name
        if digest in hashes or target.exists():
            skipped.append(source.name)
            continue
        shutil.copy2(source, target)
        hashes.add(digest)
        copied.append(target.name)
    return copied, skipped


def docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
        return " ".join((node.text or "") for node in root.iter() if node.tag.endswith("}t"))
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return ""


def capture(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return " ".join(match.group(1).split())
    return None


def extract_record(path: Path, symbol_id: str) -> dict:
    text = docx_text(path)
    title = capture(text, [
        r"Symbol\s+Title\s*:?\s*(.+?)(?=\s+(?:Reference|Sample Specification|Lifetime|Document Properties)\b)",
        r"Title\s*:?\s*(.+?)(?=\s+(?:Reference|Sample Specification|Lifetime|Document Properties)\b)",
    ])
    reference = capture(text, [r"Reference\s*:?\s*(.+?)(?=\s+(?:Sample Specification|Lifetime|Document Properties)\b)"])
    size_phrase = capture(text, [
        r"(Sample Specification\s*:?\s*(?:Min(?:imum)?\.?\s*)?\d+(?:\.\d+)?\s*mm[^.]{0,140})",
        r"((?:Min(?:imum)?\.?\s*)?\d+(?:\.\d+)?\s*mm[^.]{0,140})",
    ])
    minimum = None
    if size_phrase:
        found = re.search(r"(?:min(?:imum)?\.?\s*)?(\d+(?:\.\d+)?)\s*mm", size_phrase, flags=re.I)
        if found and re.search(r"min", size_phrase, flags=re.I):
            minimum = float(found.group(1))
    return {
        "symbol_id": symbol_id,
        "symbol_name": title or f"Symbol {symbol_id}",
        "reference": reference,
        "required_width_mm": None,
        "required_height_mm": None,
        "minimum_size_mm": minimum,
        "size_specification": size_phrase or "No explicit printed size found in the source specification; use the reference-label layout and require review.",
        "source_document": path.name,
        "approved": True,
        "layout_review_required": size_phrase is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Exported Workspace data folder")
    parser.add_argument("--target", type=Path, required=True, help="App repository root")
    args = parser.parse_args()
    source, target = args.source, args.target
    symbols_source = source / "symbols"
    guides_source = source / "label guides"
    cdlm_source = source / "CDLM information"
    assets_dir = target / "data" / "symbols" / "assets"
    specs_dir = target / "data" / "symbols" / "specifications"
    guides_dir = target / "data" / "label_guides"
    cdlm_dir = target / "data" / "cdlm"

    copied_assets, skipped_assets = copy_unique(clean_source_files(symbols_source, IMAGE_SUFFIXES), assets_dir)
    copied_specs, skipped_specs = copy_unique(clean_source_files(symbols_source, {".docx"}), specs_dir)
    copied_guides, skipped_guides = copy_unique(clean_source_files(guides_source, {".pdf"}), guides_dir)
    copied_cdlm, skipped_cdlm = copy_unique(clean_source_files(cdlm_source, {".xlsx"}), cdlm_dir)

    catalog_path = target / "data" / "symbols" / "symbol_specifications.json"
    current = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"symbols": []}
    existing = {canonical_id(str(item.get("symbol_id", ""))): item for item in current.get("symbols", []) if isinstance(item, dict)}
    image_ids = {canonical_id(item.name) for item in assets_dir.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES}
    for spec in clean_source_files(specs_dir, {".docx"}):
        symbol_id = canonical_id(spec.name)
        if symbol_id and symbol_id in image_ids and symbol_id not in existing:
            existing[symbol_id] = extract_record(spec, symbol_id)
    catalog_path.write_text(json.dumps({"symbols": sorted(existing.values(), key=lambda item: str(item["symbol_id"]))}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "source": str(source),
        "assets": {"copied": copied_assets, "skipped": skipped_assets, "total": len(image_ids)},
        "specifications": {"copied": copied_specs, "skipped": skipped_specs},
        "label_guides": {"copied": copied_guides, "skipped": skipped_guides},
        "cdlm": {"copied": copied_cdlm, "skipped": skipped_cdlm},
        "catalog_symbol_count": len(existing),
        "review_note": "Records without an explicit source size remain layout-review-required; no artwork pixel size is treated as a regulatory size.",
    }
    (target / "data" / "ingestion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
