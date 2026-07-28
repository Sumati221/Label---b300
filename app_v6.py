"""
B300 Label Generator v6

Pipeline:
1. PyMuPDF (fitz): rasterize PDF page + extract text spans with bounding boxes
2. OpenCV: multi-scale template matching of each PNG/SVG symbol against the
   rendered PDF (transparency-aware binary masks)
3. Output: normalized coords for symbol placement + editable text fields
4. No hard-coded positions — everything detected from the selected PDF
"""
import os
import re
import io
import base64
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

try:
    from openpyxl import load_workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import cairosvg
    HAS_CAIRO = True
except (ImportError, OSError):
    HAS_CAIRO = False

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("label-v6")


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy int32/float32 from OpenCV in JSON responses."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def sanitize(obj):
    """Recursively convert numpy types to native Python for JSON.
    Also handles inf/nan which are not valid JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if v != v:  # nan
            return 0.0
        if v == float('inf') or v == float('-inf'):
            return 1.0
        return v
    if isinstance(obj, float):
        if obj != obj:  # nan
            return 0.0
        if obj == float('inf') or obj == float('-inf'):
            return 1.0
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


app = FastAPI(title="B300 Label Generator v6", docs_url="/docs")

_APP = Path(__file__).parent
_SYMBOLS = _APP / "data" / "symbols"
RENDER_DPI = 300  # DPI for PDF rasterization during matching
MATCH_THRESHOLD = 0.15  # IoU-based scoring yields lower values than template correlation
LABEL_EDGE_MARGIN_MM = 1.0  # Keep all placed symbols clear of the blank label border

# Extracted from the controlled specification documents. These values are kept
# with the app so label generation does not depend on a live AI/SQL request.
SYMBOL_SPECIFICATIONS = {
    "100025": {
        "symbol_id": "100025",
        "symbol_name": "INMETRO Brazilian National Institute of Metrology symbol",
        "reference": "INMETRO Ordinance 54/2016",
        "required_width_mm": 20.0,
        "minimum_size_mm": None,
        "size_specification": "Width 20 mm.",
        "applicable_for": "Product",
        "color_specification": None,
        "usage_notes": "Required symbol; no required combination with other symbols.",
        "source_document": "LS_100025_SPEC.docx",
    },
    "100183": {
        "symbol_id": "100183",
        "symbol_name": "Thai FDA logo",
        "reference": "Thailand Ministry of Public Health, Volume 137, Special Section 260, November 5, 2020",
        "required_width_mm": None,
        "minimum_size_mm": 5.0,
        "size_specification": "Minimum size 5 mm; text within the symbol must remain legible.",
        "applicable_for": "Product and packaging of medical devices sold in Thailand",
        "color_specification": "Not specified internally or by regulation.",
        "usage_notes": "Must contain the license number, detailed notification number, or notification receipt number in Arabic numerals within the Thai FDA logo frame.",
        "source_document": "LS-100183.docx",
    },
}

_cache = None
_cache_t = 0
_cdlm_cache = None


# ════════════════════════════════════════════════════════════
# SYMBOL ASSET LOADING
# ════════════════════════════════════════════════════════════

def load_symbol_as_image(path: str) -> Optional[np.ndarray]:
    """Load a PNG or SVG symbol as an RGBA numpy array, trimmed of padding."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.svg':
            if HAS_CAIRO:
                png_bytes = cairosvg.svg2png(url=path, dpi=300)
                img = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
            else:
                # Fallback: try rendering SVG with PyMuPDF
                if HAS_FITZ:
                    doc = fitz.open(path)
                    page = doc[0]
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
                    doc.close()
                else:
                    log.warning(f"No SVG renderer for {path}")
                    return None
        else:
            img = Image.open(path).convert('RGBA')

        arr = np.array(img)
        # Trim transparent/white padding
        arr = trim_padding(arr)
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            log.warning(f"Symbol too small after trim: {path}")
            return None
        return arr
    except Exception as e:
        log.warning(f"Cannot load symbol {path}: {e}")
        return None


def trim_padding(img_rgba: np.ndarray) -> np.ndarray:
    """Remove transparent or near-white padding from all sides."""
    alpha = img_rgba[:, :, 3]
    # Consider a pixel as content if alpha > 10 AND not pure white
    rgb = img_rgba[:, :, :3]
    is_white = np.all(rgb > 240, axis=2)
    content_mask = (alpha > 10) & ~is_white
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    if not rows.any() or not cols.any():
        return img_rgba
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img_rgba[r0:r1+1, c0:c1+1]


def make_match_mask(img_rgba: np.ndarray) -> np.ndarray:
    """Create a binary mask for template matching (content pixels = 255)."""
    alpha = img_rgba[:, :, 3]
    rgb = img_rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Content = has alpha AND is dark enough (not white background)
    content = (alpha > 30) & (gray < 200)
    mask = np.zeros_like(gray)
    mask[content] = 255
    return mask


def encode_image_b64(img_rgba: np.ndarray) -> str:
    """Encode an RGBA numpy array to base64 PNG string."""
    img = Image.fromarray(img_rgba)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def get_symbol_specification(symbol_id: str) -> Optional[Dict]:
    """Return controlled metadata for a symbol ID, if a specification exists."""
    normalized_id = re.sub(r"\D", "", str(symbol_id))
    specification = SYMBOL_SPECIFICATIONS.get(normalized_id)
    return dict(specification) if specification else None


def attach_symbol_specifications(symbols: List[Dict]) -> List[Dict]:
    """Add controlled specification metadata to matched symbols at generation time."""
    enriched = []
    for symbol in symbols:
        item = dict(symbol)
        specification = get_symbol_specification(item.get("code", ""))
        if specification:
            item["specification"] = specification
        enriched.append(item)
    return enriched


def load_country_label_matrix() -> Dict:
    """Read country/product label text from the bundled CDLM workbook.

    The workbook identifies product applicability with an ``x`` in a product
    column.  Only the literal Text-column value is returned; values beginning
    with ``see`` are document references, not printable label text.
    """
    global _cdlm_cache
    if _cdlm_cache is not None:
        return _cdlm_cache

    empty = {"countries": [], "products": [], "entries": [], "error": None}
    if not HAS_OPENPYXL:
        empty["error"] = "The CDLM reader dependency is unavailable."
        _cdlm_cache = empty
        return _cdlm_cache

    files = sorted(_SYMBOLS.glob("LS-200004_CDLM_Avalon_Family_RevP*.xlsx"))
    if not files:
        empty["error"] = "No CDLM workbook was found in data/symbols."
        _cdlm_cache = empty
        return _cdlm_cache

    # Prefer the canonical filename if both a duplicate and original are present.
    source = next((f for f in files if " (" not in f.name), files[0])
    try:
        workbook = load_workbook(source, read_only=True, data_only=True)
        worksheet = workbook["Country Label"]
        product_columns = {
            column: str(worksheet.cell(2, column).value).strip()
            for column in range(11, worksheet.max_column + 1)
            if worksheet.cell(2, column).value
        }
        entries = []
        for row_number, row in enumerate(worksheet.iter_rows(min_row=3, values_only=True), start=3):
            country = str(row[3]).strip() if len(row) > 3 and row[3] else ""
            text = str(row[8]).strip() if len(row) > 8 and row[8] else ""
            location = str(row[9]).strip() if len(row) > 9 and row[9] else ""
            if not country or not text:
                continue
            for column, product in product_columns.items():
                cell = row[column - 1] if len(row) >= column else None
                if str(cell).strip().lower() != "x":
                    continue
                entries.append({
                    "country": country,
                    "product": product,
                    "text": text,
                    "location": location,
                    "is_reference": text.lower().startswith("see "),
                    "source_row": row_number,
                })
        workbook.close()
        _cdlm_cache = {
            "countries": sorted({entry["country"] for entry in entries}),
            "products": sorted(product_columns.values()),
            "entries": entries,
            "error": None,
        }
    except Exception as error:
        log.error(f"CDLM read error: {error}", exc_info=True)
        empty["error"] = "Unable to read the CDLM workbook."
        _cdlm_cache = empty
    return _cdlm_cache


def scan_symbol_assets():
    """Find all PNG/SVG symbol files. Pre-encode to base64 for fast responses."""
    assets = []
    sdir = str(_SYMBOLS)
    if not os.path.isdir(sdir):
        log.warning(f"Symbols dir not found: {sdir}")
        return assets
    for f in sorted(os.listdir(sdir)):
        low = f.lower()
        if not (low.endswith('.png') or low.endswith('.svg') or low.endswith('.jpg')):
            continue
        path = os.path.join(sdir, f)
        img = load_symbol_as_image(path)
        if img is None:
            continue
        code_m = re.match(r'(\d+)', f)
        # Pre-encode to base64 so we never need to serialize numpy later
        b64 = encode_image_b64(img)
        assets.append({
            'code': code_m.group(1) if code_m else os.path.splitext(f)[0],
            'file': f,
            'path': path,
            'image': img,          # numpy array for template matching
            'image_b64': b64,      # pre-encoded for API responses
            'h': int(img.shape[0]),
            'w': int(img.shape[1]),
            'is_svg': low.endswith('.svg')
        })
        log.info(f"  Loaded: {f} ({img.shape[1]}x{img.shape[0]}, b64={len(b64)} chars)")
    # Prefer SVG over PNG for same code
    seen_codes = {}
    deduped = []
    for a in assets:
        if a['code'] in seen_codes:
            if a['is_svg'] and not seen_codes[a['code']]['is_svg']:
                deduped = [x for x in deduped if x['code'] != a['code']]
                deduped.append(a)
                seen_codes[a['code']] = a
        else:
            deduped.append(a)
            seen_codes[a['code']] = a
    log.info(f"Total symbol assets loaded: {len(deduped)}")
    return deduped


# ════════════════════════════════════════════════════════════
# PDF ANALYSIS
# ════════════════════════════════════════════════════════════

def detect_label_boundary(page, rendered_gray, w_mm=85, h_mm=50):
    """
    Detect the inner label rectangle using PyMuPDF vector paths.
    Uses page.get_drawings() to find the rounded/rectangular path whose
    aspect ratio matches the known label dimensions (85/50 = 1.7).

    Returns (x, y, w, h) in rendered-pixel coords, or None if not found.
    """
    expected_aspect = max(w_mm, h_mm) / min(w_mm, h_mm)  # 1.7
    scale = RENDER_DPI / 72  # PDF points -> rendered pixels
    page_w_pt = page.rect.width
    page_h_pt = page.rect.height
    page_area_pt = page_w_pt * page_h_pt

    # Expected label size in PDF points
    exp_w_pt = w_mm / 25.4 * 72  # ~240 pt
    exp_h_pt = h_mm / 25.4 * 72  # ~141 pt

    candidates = []

    # Strategy 1: page.get_drawings() — vector paths
    try:
        drawings = page.get_drawings()
        for d in drawings:
            rect = d.get('rect')  # fitz.Rect bounding box of the path
            if rect is None:
                continue
            rx, ry, rx1, ry1 = rect
            rw = rx1 - rx
            rh = ry1 - ry
            if rw < 20 or rh < 20:
                continue  # skip tiny marks, arrows, dimension ticks
            area = rw * rh
            # Reject paths that span the full page (border)
            if area > page_area_pt * 0.7:
                continue
            # Reject very small paths (< 3% of page)
            if area < page_area_pt * 0.02:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            candidates.append((rx, ry, rw, rh, area, aspect))
    except Exception as e:
        log.warning(f"get_drawings() failed: {e}")

    # Strategy 2: also try page rects from annotations or explicit rect paths
    # (some PDFs encode the label outline as a simple rect, not a drawing)
    # We already have candidates from drawings; if empty, check page.rect

    if not candidates:
        log.warning("No vector paths found; cannot detect label boundary")
        return None

    # Score candidates: prefer aspect ratio close to 1.7 AND size close to expected
    best = None
    best_score = float('inf')

    for (rx, ry, rw, rh, area, aspect) in candidates:
        # Aspect error (try both landscape and portrait)
        asp_err = min(abs(aspect - expected_aspect),
                      abs(aspect - 1.0 / expected_aspect))
        if asp_err > 0.4:  # more than 40% off -> skip
            continue

        # Dimensional error: how close to expected 240x141 pt?
        dim_err1 = (abs(rw - exp_w_pt) / exp_w_pt +
                    abs(rh - exp_h_pt) / exp_h_pt)
        dim_err2 = (abs(rw - exp_h_pt) / exp_h_pt +
                    abs(rh - exp_w_pt) / exp_w_pt)
        dim_err = min(dim_err1, dim_err2)

        # Combined score: weight aspect more heavily
        score = asp_err * 2.0 + dim_err * 1.0
        if score < best_score:
            best_score = score
            best = (rx, ry, rw, rh)

    if best is None:
        log.warning("No candidate matched label aspect ratio 1.7")
        return None

    rx, ry, rw, rh = best
    log.info(f"  Label rect (PDF pts): ({rx:.1f}, {ry:.1f}) {rw:.1f}x{rh:.1f} "
             f"(score={best_score:.3f})")

    # Convert PDF points -> rendered pixel coords
    px = int(rx * scale)
    py = int(ry * scale)
    pw = int(rw * scale)
    ph = int(rh * scale)
    return (px, py, pw, ph)


def extract_text_region(page, label_bounds_pt, matched_symbols):
    """
    Extract text spans inside the label, exclude those overlapping symbols,
    and return ONE union bounding box as {x, y, w, h} in normalized 0-1 coords.
    Returns None if no text found.
    """
    lx, ly, lw, lh = label_bounds_pt
    if lw < 1 or lh < 1:
        return None

    text_dict = page.get_text("dict")
    span_boxes = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox", [0, 0, 0, 0])
                sx0, sy0, sx1, sy1 = bbox
                if not (sx0 >= lx - 1 and sx1 <= lx + lw + 1 and
                        sy0 >= ly - 1 and sy1 <= ly + lh + 1):
                    continue
                nx = max(0.0, (sx0 - lx) / lw)
                ny = max(0.0, (sy0 - ly) / lh)
                nw = min(1.0, (sx1 - sx0) / lw)
                nh = min(1.0, (sy1 - sy0) / lh)
                # Exclude spans overlapping matched symbols
                overlaps = False
                for s in matched_symbols:
                    ssx = float(s.get('x', 0))
                    ssy = float(s.get('y', 0))
                    ssw = float(s.get('w', 0))
                    ssh = float(s.get('h', 0))
                    if (nx < ssx + ssw + 0.03 and nx + nw > ssx - 0.03 and
                        ny < ssy + ssh + 0.03 and ny + nh > ssy - 0.03):
                        overlaps = True
                        break
                if not overlaps and nw > 0.001 and nh > 0.001:
                    span_boxes.append((nx, ny, nw, nh))

    if not span_boxes:
        return None

    # Union bounding box
    x0 = min(b[0] for b in span_boxes)
    y0 = min(b[1] for b in span_boxes)
    x1 = max(b[0] + b[2] for b in span_boxes)
    y1 = max(b[1] + b[3] for b in span_boxes)
    return {'x': float(x0), 'y': float(y0), 'w': float(x1 - x0), 'h': float(y1 - y0)}


def build_text_mask(page, label_bounds_px, render_dpi):
    """Build binary mask of text regions inside the label crop."""
    bx, by, bw, bh = label_bounds_px
    inv_scale = 72 / render_dpi
    lx_pt, ly_pt = bx * inv_scale, by * inv_scale
    lw_pt, lh_pt = bw * inv_scale, bh * inv_scale
    scale = render_dpi / 72
    mask = np.zeros((bh, bw), dtype=np.uint8)
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span.get("bbox", [0, 0, 0, 0])
                if sx1 < lx_pt or sx0 > lx_pt + lw_pt:
                    continue
                if sy1 < ly_pt or sy0 > ly_pt + lh_pt:
                    continue
                px0 = max(0, int((sx0 - lx_pt) * scale))
                py0 = max(0, int((sy0 - ly_pt) * scale))
                px1 = min(bw, int((sx1 - lx_pt) * scale))
                py1 = min(bh, int((sy1 - ly_pt) * scale))
                if px1 > px0 and py1 > py0:
                    mask[py0:py1, px0:px1] = 255
    return mask


def find_graphic_components(label_crop, text_mask, min_area_pct=0.002):
    """Find connected graphic blobs after removing text from label."""
    h, w = label_crop.shape[:2]
    min_area = int(h * w * min_area_pct)
    _, binary = cv2.threshold(label_crop, 180, 255, cv2.THRESH_BINARY_INV)
    binary[text_mask > 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for i in range(1, num_labels):
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if cw > w * 0.95 and ch > h * 0.95:
            continue
        comp_mask = (labels[cy:cy+ch, cx:cx+cw] == i).astype(np.uint8) * 255
        components.append({'x': cx, 'y': cy, 'w': cw, 'h': ch,
                           'area': area, 'mask': comp_mask})
    components.sort(key=lambda c: c['area'], reverse=True)
    return components


def match_asset_to_component(symbol_asset, component):
    """Score how well an asset matches a component (IoU + aspect)."""
    comp_mask = component['mask']
    ch, cw = comp_mask.shape
    sym_rgba = symbol_asset['image']
    sym_match_mask = make_match_mask(sym_rgba)
    if np.sum(sym_match_mask > 0) == 0:
        return 0.0
    mask_resized = cv2.resize(sym_match_mask, (cw, ch), interpolation=cv2.INTER_AREA)
    _, sym_bin = cv2.threshold(mask_resized, 50, 255, cv2.THRESH_BINARY)
    sym_fg = sym_bin > 0
    comp_fg = comp_mask > 0
    intersection = int(np.sum(sym_fg & comp_fg))
    union = int(np.sum(sym_fg | comp_fg))
    if union == 0:
        return 0.0
    iou = intersection / union
    sym_h, sym_w = sym_rgba.shape[:2]
    sym_asp = sym_w / max(sym_h, 1)
    comp_asp = cw / max(ch, 1)
    asp_sim = min(sym_asp, comp_asp) / max(sym_asp, comp_asp)
    return iou * (0.5 + 0.5 * asp_sim)


def component_match_pipeline(page, label_crop, label_bounds_px, symbol_assets,
                             w_mm=85, h_mm=50):
    """Component-based matching: mask text, find blobs, match assets.
    Keep detected component dimensions unless a controlled specification overrides them."""
    bx, by, bw, bh = label_bounds_px
    text_mask = build_text_mask(page, label_bounds_px, RENDER_DPI)
    components = find_graphic_components(label_crop, text_mask)
    log.info(f"  {len(components)} graphic components found")
    for i, c in enumerate(components[:10]):
        log.info(f"    comp[{i}]: ({c['x']},{c['y']}) {c['w']}x{c['h']} area={c['area']}")

    matched = []
    unmatched = []
    used = set()
    for asset in symbol_assets:
        best_score = 0.0
        best_idx = -1
        for i, comp in enumerate(components):
            if i in used:
                continue
            score = match_asset_to_component(asset, comp)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score >= MATCH_THRESHOLD and best_idx >= 0:
            comp = components[best_idx]
            used.add(best_idx)
            # Start with the size detected on the source label.
            norm_w = float(comp['w']) / bw
            norm_h = float(comp['h']) / bh
            sym_w_mm = norm_w * w_mm
            sym_h_mm = norm_h * h_mm

            # Apply only a specification that explicitly controls size.
            specification = get_symbol_specification(asset['code'])
            if specification and specification.get('required_width_mm'):
                sym_w_mm = specification['required_width_mm']
                sym_h_mm = sym_w_mm / (asset['w'] / max(asset['h'], 1))
                norm_w = sym_w_mm / w_mm
                norm_h = sym_h_mm / h_mm
            elif specification and specification.get('minimum_size_mm'):
                min_size_mm = specification['minimum_size_mm']
                detected_max_mm = max(sym_w_mm, sym_h_mm)
                if detected_max_mm < min_size_mm:
                    scale = min_size_mm / max(detected_max_mm, 0.001)
                    sym_w_mm *= scale
                    sym_h_mm *= scale
                    norm_w = sym_w_mm / w_mm
                    norm_h = sym_h_mm / h_mm

            # Center on the detected component, while preserving a 1 mm clear
            # margin on every side of the blank label.
            cx = (comp['x'] + comp['w'] / 2) / bw
            cy = (comp['y'] + comp['h'] / 2) / bh
            margin_x = LABEL_EDGE_MARGIN_MM / w_mm
            margin_y = LABEL_EDGE_MARGIN_MM / h_mm
            min_x, min_y = margin_x, margin_y
            max_x = max(min_x, 1.0 - margin_x - norm_w)
            max_y = max(min_y, 1.0 - margin_y - norm_h)
            norm_x = min(max(min_x, cx - norm_w / 2), max_x)
            norm_y = min(max(min_y, cy - norm_h / 2), max_y)
            matched.append({
                'asset': asset['file'], 'code': asset['code'],
                'x': round(norm_x, 4),
                'y': round(norm_y, 4),
                'w': round(norm_w, 4),
                'h': round(norm_h, 4),
                'confidence': round(best_score, 4)
            })
            log.info(f"  MATCH {asset['file']} -> comp[{best_idx}] "
                     f"score={best_score:.4f} size={sym_w_mm:.1f}x{sym_h_mm:.1f}mm")
        else:
            unmatched.append({
                'asset': asset['file'],
                'reason': 'not present in this label' if best_score < 0.05
                          else f'below threshold ({best_score:.3f})',
                'confidence': round(best_score, 4)
            })
            log.info(f"  SKIP {asset['file']}: best={best_score:.4f}")

    # Text region: normalized union of text mask extent
    text_coords = np.where(text_mask > 0)
    text_region = None
    if len(text_coords[0]) > 0:
        ty0 = int(text_coords[0].min())
        ty1 = int(text_coords[0].max())
        tx0 = int(text_coords[1].min())
        tx1 = int(text_coords[1].max())
        text_region = {
            'x': float(tx0) / bw, 'y': float(ty0) / bh,
            'w': float(tx1 - tx0) / bw, 'h': float(ty1 - ty0) / bh
        }

    return matched, unmatched, len(components), text_region


def template_match_symbol(rendered_gray, symbol_asset, label_bounds_px):
    """LEGACY fallback: multi-scale template matching. Not used in new pipeline."""
    lx, ly, lw, lh = label_bounds_px
    # Crop rendered image to label area
    label_region = rendered_gray[ly:ly+lh, lx:lx+lw]
    if label_region.size == 0:
        return None

    # Create grayscale template from symbol
    sym_rgba = symbol_asset['image']
    sym_gray = cv2.cvtColor(sym_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    sym_mask = make_match_mask(sym_rgba)

    # Diagnostic: verify mask has foreground pixels
    mask_fg_pixels = int(np.sum(sym_mask > 0))
    if mask_fg_pixels == 0:
        log.warning(f"  {symbol_asset['file']}: mask has ZERO fg pixels, skipping")
        return None

    # Invert: in PDF, symbols are dark on light; template should match dark content
    _, sym_bin = cv2.threshold(sym_gray, 128, 255, cv2.THRESH_BINARY_INV)
    # Also threshold the label region
    _, label_bin = cv2.threshold(label_region, 180, 255, cv2.THRESH_BINARY_INV)

    best_match = None
    best_val = 0

    # Multi-scale matching
    template_h, template_w = sym_bin.shape
    label_h_px, label_w_px = label_bin.shape

    # Scale range: symbol could be 5% to 60% of label width
    min_w = int(label_w_px * 0.03)
    max_w = int(label_w_px * 0.70)

    # Generate scales
    scales = []
    current = min_w
    while current <= max_w:
        scale_factor = current / template_w
        scaled_h = int(template_h * scale_factor)
        if scaled_h > 5 and scaled_h < label_h_px - 2 and current < label_w_px - 2:
            scales.append((current, scaled_h))
        current = int(current * 1.15)  # 15% increments

    for (tw, th) in scales:
        resized_template = cv2.resize(sym_bin, (tw, th), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(sym_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Skip if template is larger than search region
        if tw >= label_w_px or th >= label_h_px:
            continue

        try:
            result = cv2.matchTemplate(label_bin, resized_template,
                                       cv2.TM_CCORR_NORMED, mask=resized_mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Reject inf/nan (mask normalization error)
            if max_val != max_val or max_val == float('inf'):
                continue
            if max_val > best_val and max_val <= 1.0:
                best_val = max_val
                best_match = {
                    'x': float(max_loc[0]) / label_w_px,
                    'y': float(max_loc[1]) / label_h_px,
                    'w': float(tw) / label_w_px,
                    'h': float(th) / label_h_px,
                    'confidence': float(max_val)
                }
        except cv2.error:
            continue

    if best_match and best_match['confidence'] >= MATCH_THRESHOLD:
        return best_match
    elif best_match:
        log.info(f"  Low confidence ({best_match['confidence']:.3f}) for "
                 f"{symbol_asset['file']} — not placed")
    return None


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════

MIN_SYMBOL_SIZE = 0.02  # reject matches < 2% of label dimension


def process_pdf_label(pdf_path, symbol_assets, output_dpi=600):
    """
    Crop-first pipeline:
    1. Rasterize full PDF page
    2. Extract text to get label dimensions (mm)
    3. Detect inner label rectangle via vector paths
    4. CROP rendered image to label boundary only
    5. Template-match symbols within cropped label image
    6. Validate: reject <2% matches, require >=2 symbols
    7. Compute text region (one union box excluding symbols)
    """
    if not HAS_FITZ or not HAS_CV2:
        return None

    fname = os.path.basename(pdf_path)
    log.info(f"Processing: {fname}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error(f"Cannot open {fname}: {e}")
        return None

    page = doc[0]

    # 1. Extract text first to get dimensions
    full_text = page.get_text()
    w_mm, h_mm = 85, 50
    m = re.search(r'\(?(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\)?', full_text)
    if m:
        d1, d2 = int(m.group(1)), int(m.group(2))
        w_mm, h_mm = max(d1, d2), min(d1, d2)
    log.info(f"  Dimensions: {w_mm}x{h_mm}mm")

    # 2. Rasterize full page
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    full_rendered = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img_h, img_w = full_rendered.shape
    log.info(f"  Full page: {img_w}x{img_h}px")

    # 3. Detect inner label boundary using vector paths
    label_bounds_px = detect_label_boundary(page, full_rendered, w_mm, h_mm)
    if label_bounds_px is None:
        log.error(f"  FAILED: no label boundary in {fname}")
        doc.close()
        return {
            'id': fname.replace('.pdf', ''),
            'title': fname.replace('.pdf', ''),
            'error': 'No label boundary detected via vector paths',
            'w_mm': w_mm, 'h_mm': h_mm,
            'symbols': [], 'text_region': None,
            'debug': {'error': 'no vector path matched aspect 1.7'}
        }

    bx, by, bw, bh = label_bounds_px
    log.info(f"  Label rect: ({bx},{by}) {bw}x{bh}px")

    # 4. CROP rendered image to label only
    bx = max(0, min(bx, img_w - 1))
    by = max(0, min(by, img_h - 1))
    bw = min(bw, img_w - bx)
    bh = min(bh, img_h - by)
    if bw < 20 or bh < 20:
        log.error(f"  Crop too small: {bw}x{bh}")
        doc.close()
        return {
            'id': fname.replace('.pdf', ''),
            'title': fname.replace('.pdf', ''),
            'error': f'Label crop too small ({bw}x{bh}px)',
            'w_mm': w_mm, 'h_mm': h_mm,
            'symbols': [], 'text_region': None,
            'debug': {'label_bounds_px': [bx, by, bw, bh]}
        }

    label_crop = full_rendered[by:by+bh, bx:bx+bw]
    log.info(f"  Cropped label: {label_crop.shape[1]}x{label_crop.shape[0]}px")

    # 5. Component-based matching (text masked, graphic blobs isolated, IoU scored)
    matched_symbols, failed_symbols, n_components, text_region = \
        component_match_pipeline(page, label_crop, (bx, by, bw, bh), symbol_assets,
                                 w_mm=w_mm, h_mm=h_mm)
    n_matched = len(matched_symbols)
    log.info(f"  Result: {n_matched} matched, {len(failed_symbols)} skipped, "
             f"{n_components} components")

    # Title
    title = fname.replace('.pdf', '')
    tm_match = re.search(r'TITLE\n(.+)', full_text)
    if tm_match:
        title = tm_match.group(1).strip()

    doc.close()

    return {
        'id': fname.replace('.pdf', ''),
        'title': title,
        'w_mm': w_mm, 'h_mm': h_mm,
        'symbols': matched_symbols,
        'text_region': text_region,
        'debug': {
            'render_dpi': RENDER_DPI,
            'page_size_px': f"{img_w}x{img_h}",
            'label_bounds_px': [bx, by, bw, bh],
            'label_crop_px': f"{bw}x{bh}",
            'pipeline': 'component-match',
            'assets_tested': len(symbol_assets),
            'assets_matched': n_matched,
            'graphic_components_found': n_components,
            'unmatched_assets': failed_symbols,
            'has_text_region': text_region is not None
        }
    }


def _dead_code_start():  # pragma: no cover
    """Everything below until _dead_code_end was the OLD matching logic."""

    for asset in symbol_assets:
        match = template_match_symbol(label_crop, asset, crop_bounds)
        if match:
            # Reject too-small matches (<2% of label)
            if match['w'] < MIN_SYMBOL_SIZE or match['h'] < MIN_SYMBOL_SIZE:
                reason = f"too small ({match['w']:.4f}x{match['h']:.4f})"
                log.warning(f"  REJECT {asset['file']}: {reason}")
                failed_symbols.append({'asset': asset['file'],
                                       'reason': reason,
                                       'confidence': match['confidence']})
                continue
            # Reject outside bounds
            if (match['x'] < -0.01 or match['y'] < -0.01 or
                match['x'] + match['w'] > 1.05 or
                match['y'] + match['h'] > 1.05):
                reason = f"outside label (x={match['x']:.3f} y={match['y']:.3f})"
                log.warning(f"  REJECT {asset['file']}: {reason}")
                failed_symbols.append({'asset': asset['file'],
                                       'reason': reason,
                                       'confidence': match['confidence']})
                continue
            matched_symbols.append({
                'asset': asset['file'], 'code': asset['code'],
                'x': match['x'], 'y': match['y'],
                'w': match['w'], 'h': match['h'],
                'confidence': match['confidence']
            })
            log.info(f"  OK {asset['file']}: "
                     f"({match['x']:.3f},{match['y']:.3f}) "
                     f"{match['w']:.3f}x{match['h']:.3f} "
                     f"conf={match['confidence']:.3f}")
        else:
            failed_symbols.append({'asset': asset['file'],
                                   'reason': 'not present in this label',
                                   'confidence': 0})
            log.info(f"  SKIP {asset['file']}: not in this PDF")

    # 6. Each PDF uses its own subset of symbols — do NOT require all
    n_matched = len(matched_symbols)
    n_not_present = len([f for f in failed_symbols if f['confidence'] == 0])
    n_rejected = len([f for f in failed_symbols if f['confidence'] > 0])
    log.info(f"  Result: {n_matched} matched, {n_not_present} not in this PDF, "
             f"{n_rejected} rejected (size/bounds)")

    # Only fail if ZERO symbols matched AND label has visible content
    if n_matched == 0:
        dark_pix = int(np.sum(label_crop < 128))
        total_pix = label_crop.shape[0] * label_crop.shape[1]
        has_content = dark_pix > total_pix * 0.01
        if has_content:
            log.error(f"  FAIL: 0 matched but label has content ({dark_pix/total_pix*100:.1f}% dark)")
            doc.close()
            return {
                'id': fname.replace('.pdf', ''),
                'title': fname.replace('.pdf', ''),
                'error': f'No symbols matched (0/{len(symbol_assets)}). Label has visible content but no asset correlated.',
                'w_mm': w_mm, 'h_mm': h_mm,
                'symbols': [],
                'failed_symbols': failed_symbols,
                'text_region': None,
                'debug': {
                    'label_bounds_px': [bx, by, bw, bh],
                    'label_crop_px': f"{bw}x{bh}",
                    'dark_pixel_pct': f"{dark_pix/total_pix*100:.1f}%",
                    'failed_symbols': failed_symbols
                }
            }
        else:
            log.warning(f"  Label crop appears blank")
            doc.close()
            return None

    # 7. Text region (union box excluding symbol areas)
    scale = 72 / RENDER_DPI
    label_bounds_pt = (bx * scale, by * scale, bw * scale, bh * scale)
    text_region = extract_text_region(page, label_bounds_pt, matched_symbols)
    log.info(f"  Text region: {text_region}")

    # Title
    title = fname.replace('.pdf', '')
    tm_match = re.search(r'TITLE\n(.+)', full_text)
    if tm_match:
        title = tm_match.group(1).strip()

    doc.close()

    return {
        'id': fname.replace('.pdf', ''),
        'title': title,
        'w_mm': w_mm, 'h_mm': h_mm,
        'symbols': matched_symbols,
        'text_region': text_region,
        'debug': {
            'render_dpi': RENDER_DPI,
            'page_size_px': f"{img_w}x{img_h}",
            'label_bounds_px': [bx, by, bw, bh],
            'label_crop_px': f"{bw}x{bh}",
            'assets_tested': len(symbol_assets),
            'assets_matched': len(matched_symbols),
            'failed_symbols': failed_symbols,
            'has_text_region': text_region is not None
        }
    }


def _dead_code_end():  # pragma: no cover
    pass


# ════════════════════════════════════════════════════════════
# CATALOG
# ════════════════════════════════════════════════════════════

def load_catalog():
    global _cache, _cache_t
    now = time.time()
    if _cache and (now - _cache_t) < 300:
        return _cache

    assets = scan_symbol_assets()
    log.info(f"Symbol assets: {[a['file'] for a in assets]}")

    labels = []
    sdir = str(_SYMBOLS)
    if os.path.isdir(sdir):
        for f in sorted(os.listdir(sdir)):
            if f.lower().endswith('.pdf') and 'mart' in f.lower():
                result = process_pdf_label(os.path.join(sdir, f), assets)
                if result:
                    labels.append(result)

    _cache = {'labels': labels, 'assets': assets}
    _cache_t = now
    return _cache


# ════════════════════════════════════════════════════════════
# API
# ════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home():
    p = _APP / "index.html"
    return HTMLResponse(p.read_text()) if p.exists() else HTMLResponse("<h1>B300 v6</h1>")


@app.get("/api/catalog")
async def api_catalog():
    c = load_catalog()
    return {
        "products": [{
            'product_code': lab['id'].split('-M-')[0],
            'product_desc': lab['title'],
            'label_size': f"{lab['h_mm']} X {lab['w_mm']} mm",
            'sheet_name': lab['id'],
            'symbol_count': len(lab['symbols'])
        } for lab in c['labels']],
        "count": len(c['labels'])
    }


@app.get("/api/country-labels")
async def api_country_labels(country: Optional[str] = None, product: Optional[str] = None):
    """Expose CDLM options and safe product-label text for the selected pair."""
    matrix = load_country_label_matrix()
    if matrix["error"]:
        return JSONResponse(content={"error": matrix["error"]}, status_code=503)

    response = {"countries": matrix["countries"], "products": matrix["products"]}
    if country and product:
        response["entries"] = [
            entry for entry in matrix["entries"]
            if entry["country"] == country
            and entry["product"] == product
            and "product label" in entry["location"].lower()
        ]
    return response


@app.get("/api/symbols/{symbol_id}")
async def api_symbol_specification(symbol_id: str):
    """Return the controlled specification for one symbol."""
    specification = get_symbol_specification(symbol_id)
    if not specification:
        raise HTTPException(404, f"No specification found for symbol {symbol_id}")
    return specification


@app.get("/api/generate/{label_id}")
async def api_generate(label_id: str, dpi: int = 600):
    """Generate label: returns normalized symbol positions + base64 images."""
    try:
        c = load_catalog()
        lab = next((l for l in c['labels'] if l['id'] == label_id), None)
        if not lab:
            avail = [l['id'] for l in c['labels']]
            return JSONResponse(content={"error": f"Not found: {label_id}", "available": avail}, status_code=404)

        # Check if label processing returned an error
        if lab.get('error'):
            return JSONResponse(content={
                "error": lab['error'],
                "label_id": lab['id'],
                "failed_symbols": sanitize(lab.get('failed_symbols', [])),
                "matched_symbols": sanitize(lab.get('symbols', [])),
                "debug": sanitize(lab.get('debug', {}))
            }, status_code=422)

        # Use pre-encoded base64 (no numpy conversion at request time)
        sym_images = {}
        for sym in lab['symbols']:
            asset = next((a for a in c['assets'] if a['file'] == sym['asset']), None)
            if asset and asset.get('image_b64'):
                sym_images[sym['asset']] = asset['image_b64']

        symbols = attach_symbol_specifications(lab['symbols'])
        result = {
            "label_id": lab['id'],
            "title": lab['title'],
            "label_size": f"{lab['h_mm']} X {lab['w_mm']} mm",
            "w_mm": int(lab['w_mm']),
            "h_mm": int(lab['h_mm']),
            "symbols": sanitize(symbols),
            "symbol_specifications": [s['specification'] for s in symbols if 'specification' in s],
            "text_region": sanitize(lab.get('text_region')),
            "symbol_images": sym_images,
            "symbols_placed": len(lab['symbols']),
            "convention": "template-matched",
            "debug": sanitize(lab.get('debug', {}))
        }
        return JSONResponse(content=result)
    except Exception as e:
        log.error(f"Generate error for {label_id}: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/generate/{label_id}/image")
async def api_generate_image(label_id: str, dpi: int = 600, debug_overlay: bool = False):
    """Render label as a single PNG image with symbols composited."""
    c = load_catalog()
    lab = next((l for l in c['labels'] if l['id'] == label_id), None)
    if not lab:
        raise HTTPException(404, f"Not found: {label_id}")

    w_px = int(lab['w_mm'] / 25.4 * dpi)
    h_px = int(lab['h_mm'] / 25.4 * dpi)

    # White canvas
    canvas = Image.new('RGB', (w_px, h_px), 'white')

    # Place symbols
    for sym in lab['symbols']:
        asset = next((a for a in c['assets'] if a['file'] == sym['asset']), None)
        if not asset:
            continue
        sx = int(sym['x'] * w_px)
        sy = int(sym['y'] * h_px)
        sw = int(sym['w'] * w_px)
        sh = int(sym['h'] * h_px)
        if sw < 1 or sh < 1:
            continue
        sym_img = Image.fromarray(asset['image']).resize((sw, sh), Image.LANCZOS)
        # Paste with alpha
        if sym_img.mode == 'RGBA':
            canvas.paste(sym_img, (sx, sy), sym_img)
        else:
            canvas.paste(sym_img, (sx, sy))

    # Debug overlay
    if debug_overlay:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        # Label border
        draw.rectangle([2, 2, w_px-3, h_px-3], outline='blue', width=2)
        # Symbol boxes
        for sym in lab['symbols']:
            sx = int(sym['x'] * w_px)
            sy = int(sym['y'] * h_px)
            sw = int(sym['w'] * w_px)
            sh = int(sym['h'] * h_px)
            draw.rectangle([sx, sy, sx+sw, sy+sh], outline='red', width=1)
            draw.text((sx, sy-10),
                      f"{sym['asset']} ({sym['confidence']:.2f})",
                      fill='red')
        # Text region box
        tr = lab.get('text_region')
        if tr:
            tx = int(float(tr['x']) * w_px)
            ty = int(float(tr['y']) * h_px)
            tw = int(float(tr['w']) * w_px)
            th = int(float(tr['h']) * h_px)
            draw.rectangle([tx, ty, tx+tw, ty+th], outline='green', width=1)

    buf = io.BytesIO()
    canvas.save(buf, format='PNG', dpi=(dpi, dpi))
    buf.seek(0)

    if debug_overlay:
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image": f"data:image/png;base64,{img_b64}",
                "product_code": lab['id'].split('-M-')[0],
                "product_desc": lab['title'],
                "label_size": f"{lab['h_mm']} X {lab['w_mm']} mm",
                "symbols_placed": len(lab['symbols']),
                "convention": "template-matched"}

    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/download/{label_id}")
async def api_download(label_id: str, dpi: int = 600):
    resp = await api_generate_image(label_id, dpi, debug_overlay=False)
    if isinstance(resp, Response):
        resp.headers["Content-Disposition"] = f'attachment; filename="{label_id}.png"'
        return resp
    raise HTTPException(500, "Unexpected response type")


@app.get("/api/health")
async def api_health():
    try:
        c = load_catalog()
        data = {
            "status": "ok",
            "dependencies": {
                "PyMuPDF": HAS_FITZ, "OpenCV": HAS_CV2, "CairoSVG": HAS_CAIRO
            },
            "symbols_dir": str(_SYMBOLS),
            "dir_exists": os.path.isdir(str(_SYMBOLS)),
            "all_files": os.listdir(str(_SYMBOLS)) if os.path.isdir(str(_SYMBOLS)) else [],
            "assets_loaded": [{'file': a['file'], 'size': f"{a['w']}x{a['h']}"}
                              for a in c['assets']],
            "labels": [{
                'id': l['id'],
                'symbols_matched': len(l['symbols']),
                'matched': [{'asset': s['asset'],
                             'confidence': float(s['confidence']),
                             'pos': f"({float(s['x']):.2f},{float(s['y']):.2f})",
                             'size': f"{float(s['w']):.2f}x{float(s['h']):.2f}"}
                            for s in l['symbols']],
                'error': l.get('error'),
                'has_text_region': l.get('text_region') is not None,
                'debug': l.get('debug', {})
            } for l in c['labels']]
        }
        return JSONResponse(content=json.loads(json.dumps(data, cls=NumpyEncoder)))
    except Exception as e:
        log.error(f"Health check error: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.on_event("startup")
async def startup():
    log.info("=== B300 Label Generator v6 (PyMuPDF + OpenCV) ===")
    log.info(f"PyMuPDF: {HAS_FITZ}, OpenCV: {HAS_CV2}, CairoSVG: {HAS_CAIRO}")
    log.info(f"Symbols dir: {_SYMBOLS} (exists={os.path.isdir(str(_SYMBOLS))})")
    if os.path.isdir(str(_SYMBOLS)):
        log.info(f"Contents: {os.listdir(str(_SYMBOLS))}")
    # Pre-warm cache
    try:
        load_catalog()
    except Exception as e:
        log.error(f"Startup cache error: {e}", exc_info=True)


@app.get("/api/test")
async def api_test():
    """Simple test: returns OK if app is running."""
    return JSONResponse(content={"ok": True, "msg": "App is alive"})


@app.get("/api/test_generate/{label_id}")
async def api_test_generate(label_id: str):
    """Debug generate: returns step-by-step trace to find crash point."""
    steps = []
    try:
        steps.append("1. Loading catalog...")
        c = load_catalog()
        steps.append(f"2. Labels: {[l['id'] for l in c['labels']]}")
        steps.append(f"3. Assets: {[a['file'] for a in c['assets']]}")

        lab = next((l for l in c['labels'] if l['id'] == label_id), None)
        if not lab:
            steps.append(f"4. FAIL: '{label_id}' not in labels")
            return JSONResponse(content={"steps": steps})

        steps.append(f"4. Found label: {lab['id']}")
        steps.append(f"5. Symbols: {len(lab['symbols'])}")
        steps.append(f"6. Text spans: {len(lab['text_spans'])}")

        # Check symbol types
        for i, sym in enumerate(lab['symbols']):
            steps.append(f"7.{i} sym keys={list(sym.keys())} types={{k: type(v).__name__ for k,v in sym.items()}}")

        # Try building sym_images
        steps.append("8. Building sym_images...")
        sym_images = {}
        for sym in lab['symbols']:
            asset = next((a for a in c['assets'] if a['file'] == sym['asset']), None)
            if asset and asset.get('image_b64'):
                sym_images[sym['asset']] = asset['image_b64'][:50] + "..."
                steps.append(f"   OK: {sym['asset']} ({len(asset['image_b64'])} chars)")
            else:
                steps.append(f"   MISS: {sym['asset']} (asset={asset is not None})")

        # Try sanitize
        steps.append("9. Sanitizing symbols...")
        clean_syms = sanitize(lab['symbols'])
        steps.append(f"10. Sanitized type: {type(clean_syms).__name__}")
        if clean_syms:
            steps.append(f"11. First sym: {clean_syms[0]}")

        steps.append("12. Building result dict...")
        result = {
            "label_id": str(lab['id']),
            "w_mm": int(lab['w_mm']),
            "h_mm": int(lab['h_mm']),
            "symbols": clean_syms,
            "symbol_images": sym_images,
            "symbols_placed": len(lab['symbols'])
        }
        steps.append("13. Returning JSONResponse...")
        return JSONResponse(content={"steps": steps, "result_keys": list(result.keys())})
    except Exception as e:
        steps.append(f"CRASH: {type(e).__name__}: {e}")
        import traceback
        steps.append(traceback.format_exc())
        return JSONResponse(content={"steps": steps}, status_code=500)
