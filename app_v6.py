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
MATCH_THRESHOLD = 0.55  # Minimum confidence for template match

_cache = None
_cache_t = 0


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


def template_match_symbol(rendered_gray, symbol_asset, label_bounds_px):
    """
    Multi-scale template matching of a symbol against the rendered PDF.
    Returns best match: {x, y, w, h, confidence} in normalized label coords,
    or None if below threshold.
    """
    lx, ly, lw, lh = label_bounds_px
    # Crop rendered image to label area
    label_region = rendered_gray[ly:ly+lh, lx:lx+lw]
    if label_region.size == 0:
        return None

    # Create grayscale template from symbol
    sym_rgba = symbol_asset['image']
    sym_gray = cv2.cvtColor(sym_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    sym_mask = make_match_mask(sym_rgba)

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

    # 5. Template-match within cropped label (coords are 0,0-relative)
    crop_bounds = (0, 0, bw, bh)
    matched_symbols = []
    failed_symbols = []

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
                                   'reason': 'no match above threshold',
                                   'confidence': 0})
            log.warning(f"  MISS {asset['file']}")

    # 6. VALIDATE: require at least 2 symbols
    if len(matched_symbols) < 2:
        log.error(f"  FAIL: only {len(matched_symbols)}/{len(symbol_assets)} symbols")
        doc.close()
        return {
            'id': fname.replace('.pdf', ''),
            'title': fname.replace('.pdf', ''),
            'error': (f'Symbol matching failed: {len(matched_symbols)}/'
                      f'{len(symbol_assets)} matched'),
            'w_mm': w_mm, 'h_mm': h_mm,
            'symbols': matched_symbols,
            'failed_symbols': failed_symbols,
            'text_region': None,
            'debug': {
                'label_bounds_px': [bx, by, bw, bh],
                'label_crop_px': f"{bw}x{bh}",
                'failed_symbols': failed_symbols
            }
        }

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

        result = {
            "label_id": lab['id'],
            "title": lab['title'],
            "label_size": f"{lab['h_mm']} X {lab['w_mm']} mm",
            "w_mm": int(lab['w_mm']),
            "h_mm": int(lab['h_mm']),
            "symbols": sanitize(lab['symbols']),
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
