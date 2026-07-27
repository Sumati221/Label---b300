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
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
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


def scan_symbol_assets():
    """Find all PNG/SVG symbol files (not PDFs, not Excel, not .gitkeep)."""
    assets = []
    sdir = str(_SYMBOLS)
    if not os.path.isdir(sdir):
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
        assets.append({
            'code': code_m.group(1) if code_m else os.path.splitext(f)[0],
            'file': f,
            'path': path,
            'image': img,
            'h': img.shape[0],
            'w': img.shape[1],
            'is_svg': low.endswith('.svg')
        })
    # Prefer SVG over PNG for same code
    seen_codes = {}
    deduped = []
    for a in assets:
        if a['code'] in seen_codes:
            if a['is_svg'] and not seen_codes[a['code']]['is_svg']:
                # Replace PNG with SVG
                deduped = [x for x in deduped if x['code'] != a['code']]
                deduped.append(a)
                seen_codes[a['code']] = a
        else:
            deduped.append(a)
            seen_codes[a['code']] = a
    return deduped


# ════════════════════════════════════════════════════════════
# PDF ANALYSIS
# ════════════════════════════════════════════════════════════

def detect_label_boundary(page, rendered_gray):
    """
    Detect the actual label boundary within the engineering drawing.
    Uses edge detection + contour finding on the rendered image.
    Returns (x, y, w, h) in pixel coords of the rendered image.
    """
    # Threshold to get strong edges
    _, thresh = cv2.threshold(rendered_gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = rendered_gray.shape
        return (0, 0, w, h)

    # Find the largest rectangular contour that isn't the full page
    img_h, img_w = rendered_gray.shape
    img_area = img_h * img_w
    best_rect = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Must be substantial but not the full page
        if area > img_area * 0.02 and area < img_area * 0.85:
            if area > best_area:
                best_area = area
                best_rect = (x, y, w, h)

    if best_rect:
        return best_rect

    # Fallback: use the page text extents from PyMuPDF
    text_dict = page.get_text("dict")
    blocks = text_dict.get("blocks", [])
    if blocks:
        x0s = [b["bbox"][0] for b in blocks if "bbox" in b]
        y0s = [b["bbox"][1] for b in blocks if "bbox" in b]
        x1s = [b["bbox"][2] for b in blocks if "bbox" in b]
        y1s = [b["bbox"][3] for b in blocks if "bbox" in b]
        if x0s:
            # Scale from PDF points to rendered pixels
            scale = RENDER_DPI / 72
            return (int(min(x0s)*scale), int(min(y0s)*scale),
                    int((max(x1s)-min(x0s))*scale),
                    int((max(y1s)-min(y0s))*scale))

    return (0, 0, img_w, img_h)


def extract_text_spans(page, label_bounds_pt):
    """
    Extract text spans from the PDF page within the label boundary.
    Returns list of {text, x, y, w, h, font_size} all in normalized coords (0-1).
    label_bounds_pt: (x0, y0, w, h) in PDF points.
    """
    lx, ly, lw, lh = label_bounds_pt
    spans = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox", [0, 0, 0, 0])
                sx0, sy0, sx1, sy1 = bbox
                # Check if inside label boundary
                if (sx0 >= lx - 1 and sx1 <= lx + lw + 1 and
                    sy0 >= ly - 1 and sy1 <= ly + lh + 1):
                    # Normalize to label coords
                    nx = (sx0 - lx) / lw
                    ny = (sy0 - ly) / lh
                    nw = (sx1 - sx0) / lw
                    nh = (sy1 - sy0) / lh
                    spans.append({
                        'text': span.get('text', ''),
                        'x': max(0, nx),
                        'y': max(0, ny),
                        'w': min(1, nw),
                        'h': min(1, nh),
                        'font_size': span.get('size', 8),
                        'font': span.get('font', '')
                    })
    return spans


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

            if max_val > best_val:
                best_val = max_val
                best_match = {
                    'x': max_loc[0] / label_w_px,
                    'y': max_loc[1] / label_h_px,
                    'w': tw / label_w_px,
                    'h': th / label_h_px,
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

def process_pdf_label(pdf_path, symbol_assets, output_dpi=600):
    """
    Full pipeline for one PDF label:
    1. Rasterize PDF at RENDER_DPI for template matching
    2. Detect label boundary
    3. Template-match each symbol asset
    4. Extract text spans
    5. Return normalized layout
    """
    if not HAS_FITZ or not HAS_CV2:
        return None

    fname = os.path.basename(pdf_path)
    log.info(f"Processing: {fname}")

    doc = fitz.open(pdf_path)
    page = doc[0]
    page_rect = page.rect  # PDF points

    # 1. Rasterize at RENDER_DPI
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    rendered = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    rendered_color = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_h, img_w = rendered.shape
    log.info(f"  Rendered: {img_w}x{img_h}px at {RENDER_DPI}dpi")

    # 2. Detect label boundary (in rendered pixel coords)
    label_bounds_px = detect_label_boundary(page, rendered)
    bx, by, bw, bh = label_bounds_px
    log.info(f"  Label boundary: ({bx},{by}) {bw}x{bh}px")

    # Convert to PDF points for text extraction
    scale = 72 / RENDER_DPI
    label_bounds_pt = (bx * scale, by * scale, bw * scale, bh * scale)

    # 3. Template-match each symbol
    matched_symbols = []
    for asset in symbol_assets:
        match = template_match_symbol(rendered, asset, label_bounds_px)
        if match:
            matched_symbols.append({
                'asset': asset['file'],
                'code': asset['code'],
                'x': match['x'],
                'y': match['y'],
                'w': match['w'],
                'h': match['h'],
                'confidence': match['confidence']
            })
            log.info(f"  Matched {asset['file']}: "
                     f"pos=({match['x']:.3f},{match['y']:.3f}) "
                     f"size=({match['w']:.3f}x{match['h']:.3f}) "
                     f"conf={match['confidence']:.3f}")
        else:
            log.info(f"  No match for {asset['file']}")

    # 4. Extract text spans
    text_spans = extract_text_spans(page, label_bounds_pt)
    log.info(f"  Text spans: {len(text_spans)}")

    # 5. Get label dimensions in mm
    full_text = page.get_text()
    w_mm, h_mm = 85, 50
    m = re.search(r'\(?(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\)?', full_text)
    if m:
        d1, d2 = int(m.group(1)), int(m.group(2))
        w_mm, h_mm = max(d1, d2), min(d1, d2)

    # Title
    title = fname.replace('.pdf', '')
    tm = re.search(r'TITLE\n(.+)', full_text)
    if tm:
        title = tm.group(1).strip()

    doc.close()

    return {
        'id': fname.replace('.pdf', ''),
        'title': title,
        'w_mm': w_mm, 'h_mm': h_mm,
        'symbols': matched_symbols,
        'text_spans': text_spans,
        'debug': {
            'render_dpi': RENDER_DPI,
            'page_size_px': f"{img_w}x{img_h}",
            'label_bounds_px': [int(x) for x in label_bounds_px],
            'assets_tested': len(symbol_assets),
            'assets_matched': len(matched_symbols),
            'text_spans': len(text_spans)
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

        # Encode matched symbol assets as base64 for frontend rendering
        sym_images = {}
        for sym in lab['symbols']:
            asset = next((a for a in c['assets'] if a['file'] == sym['asset']), None)
            if asset is not None and asset.get('image') is not None:
                try:
                    img = Image.fromarray(asset['image'])
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    sym_images[sym['asset']] = base64.b64encode(buf.getvalue()).decode()
                except Exception as enc_err:
                    log.error(f"Image encode error {sym['asset']}: {enc_err}")

        result = {
            "label_id": lab['id'],
            "title": lab['title'],
            "label_size": f"{lab['h_mm']} X {lab['w_mm']} mm",
            "w_mm": int(lab['w_mm']),
            "h_mm": int(lab['h_mm']),
            "symbols": sanitize(lab['symbols']),
            "text_fields": sanitize(lab['text_spans']),
            "symbol_images": sym_images,
            "symbols_placed": len(lab['symbols']),
            "convention": "template-matched",
            "debug": sanitize(lab['debug'])
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
        # Text boxes
        for tf in lab['text_spans']:
            tx = int(tf['x'] * w_px)
            ty = int(tf['y'] * h_px)
            tw = int(tf['w'] * w_px)
            th = int(tf['h'] * h_px)
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
                'text_spans': len(l['text_spans']),
                'debug': l['debug']
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
    load_catalog()
