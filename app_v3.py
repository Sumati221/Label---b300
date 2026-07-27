"""
B300 Label Generator v3
Extracts label boundary, symbol locations/sizes, and text-field positions
directly from PDF MART drawings. Recreates the label with matched PNG symbols
at exact positions. Text areas are empty editable placeholders.
"""
import os
import re
import io
import base64
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("b300-label-gen")

app = FastAPI(title="B300 Label Generator", docs_url="/docs")

_APP_DIR = Path(__file__).parent
SYMBOLS_DIR = _APP_DIR / "data" / "symbols"
DPI = 600
MM_TO_INCH = 1 / 25.4
PT_TO_MM = 25.4 / 72  # PDF points to mm
SYMBOL_EXTENSIONS = ['.png', '.jpg', '.svg']

_cache = None
_cache_time = 0


# ─── PDF PARSING ─────────────────────────────────────────────────────────────

def parse_label_pdf(pdf_path):
    """
    Parse a MART PDF drawing to extract:
    - Label boundary (dimensions)
    - Symbol bounding boxes (position + size within label)
    - Text field locations
    
    Strategy:
    1. Get all rectangles from the PDF page
    2. Identify the title block (bottom portion of A4 landscape) and exclude it
    3. Among remaining rects, the label outline is the one closest to expected
       dimensions (from text or largest non-page rect)
    4. All smaller rects INSIDE the label boundary are symbol regions
    5. All text words INSIDE the label boundary are text fields
    """
    if not HAS_PDFPLUMBER:
        log.error("pdfplumber not available")
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            pw, ph = page.width, page.height  # points

            # ── Step 1: Extract label dimensions from text ──
            text = page.extract_text() or ''
            label_w_mm, label_h_mm = 85, 50  # defaults
            size_match = re.search(r'\(?(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\)?', text)
            if size_match:
                d1, d2 = int(size_match.group(1)), int(size_match.group(2))
                label_w_mm = max(d1, d2)
                label_h_mm = min(d1, d2)

            # ── Step 2: Get all rectangles ──
            all_rects = page.rects or []
            rects = []
            for r in all_rects:
                x0 = min(r.get('x0', 0), r.get('x1', 0))
                y0 = min(r.get('top', 0), r.get('bottom', 0))
                x1 = max(r.get('x0', 0), r.get('x1', 0))
                y1 = max(r.get('top', 0), r.get('bottom', 0))
                w = x1 - x0
                h = y1 - y0
                if w > 3 and h > 3:  # skip hairlines
                    rects.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                                  'w': w, 'h': h})

            # ── Step 3: Identify title block ──
            # On A4 landscape (841x595 pts), title block is typically
            # a large rect in the lower portion (y > 60% of page height)
            title_block_y = ph * 0.55

            # Filter: rects above title block area
            drawing_rects = [r for r in rects if r['y0'] < title_block_y]

            # ── Step 4: Find the label boundary ──
            # Expected label size in points
            expected_w_pt = label_w_mm / PT_TO_MM
            expected_h_pt = label_h_mm / PT_TO_MM

            label_boundary = None
            best_score = float('inf')

            for r in drawing_rects:
                # Score: how close is this rect to expected label dims?
                w_err = abs(r['w'] - expected_w_pt) / expected_w_pt
                h_err = abs(r['h'] - expected_h_pt) / expected_h_pt
                score = w_err + h_err
                if score < best_score and score < 0.5:  # within 50% tolerance
                    best_score = score
                    label_boundary = r

            # Fallback: use largest rect in drawing area
            if not label_boundary and drawing_rects:
                label_boundary = max(drawing_rects, key=lambda r: r['w'] * r['h'])

            if not label_boundary:
                log.warning(f"Could not find label boundary in {pdf_path}")
                # Use entire drawing area
                label_boundary = {'x0': 50, 'y0': 50,
                                  'x1': 50 + expected_w_pt,
                                  'y1': 50 + expected_h_pt,
                                  'w': expected_w_pt, 'h': expected_h_pt}

            lx0 = label_boundary['x0']
            ly0 = label_boundary['y0']
            lw = label_boundary['w']
            lh = label_boundary['h']

            # ── Step 5: Find symbol regions inside label ──
            symbol_regions = []
            for r in drawing_rects:
                if r is label_boundary:
                    continue
                # Check if fully contained within label boundary (with tolerance)
                tol = 2  # points
                if (r['x0'] >= lx0 - tol and r['x1'] <= lx0 + lw + tol and
                    r['y0'] >= ly0 - tol and r['y1'] <= ly0 + lh + tol):
                    # Relative position within label (0 to 1)
                    rx = (r['x0'] - lx0) / lw
                    ry = (r['y0'] - ly0) / lh
                    rw = r['w'] / lw
                    rh = r['h'] / lh
                    # In PDF, y=0 is top. For our canvas, flip Y.
                    symbol_regions.append({
                        'rel_x': max(0, rx),
                        'rel_y': max(0, 1.0 - ry - rh),  # flip Y
                        'rel_w': min(1, rw),
                        'rel_h': min(1, rh),
                        'aspect': r['w'] / r['h'] if r['h'] > 0 else 1,
                        'area_pt2': r['w'] * r['h']
                    })

            # ── Step 6: Extract text fields inside label ──
            text_fields = []
            words = page.extract_words(keep_blank_chars=True) or []
            for word in words:
                wx = word.get('x0', 0)
                wy = word.get('top', 0)
                wx1 = word.get('x1', 0)
                wy1 = word.get('bottom', 0)
                # Inside label boundary?
                if (wx >= lx0 - 2 and wx1 <= lx0 + lw + 2 and
                    wy >= ly0 - 2 and wy1 <= ly0 + lh + 2):
                    rx = (wx - lx0) / lw
                    ry = (wy - ly0) / lh
                    rw = (wx1 - wx) / lw
                    rh = (wy1 - wy) / lh
                    text_fields.append({
                        'original_text': word.get('text', ''),
                        'rel_x': max(0, rx),
                        'rel_y': max(0, 1.0 - ry - rh),
                        'rel_w': rw,
                        'rel_h': rh,
                        'font_size_pt': word.get('size', 8)
                    })

            # ── Extract title from drawing title block ──
            title = os.path.basename(pdf_path).replace('.pdf', '')
            title_match = re.search(r'TITLE\n(.+?)\n', text)
            if title_match:
                title = title_match.group(1).strip()

            drawing_num = title.split('-M-')[0] if '-M-' in title else title

            return {
                'width_mm': label_w_mm,
                'height_mm': label_h_mm,
                'title': title,
                'drawing_number': drawing_num,
                'symbol_regions': symbol_regions,
                'text_fields': text_fields,
                'debug': {
                    'total_rects': len(all_rects),
                    'drawing_rects': len(drawing_rects),
                    'label_boundary_pts': [lx0, ly0, lw, lh],
                    'symbols_found': len(symbol_regions),
                    'texts_found': len(text_fields)
                }
            }

    except Exception as e:
        log.error(f"PDF parse error ({pdf_path}): {e}")
        return None


# ─── SYMBOL MATCHING ─────────────────────────────────────────────────────────

def get_available_pngs():
    """Scan for available symbol PNG/JPG/SVG files."""
    symbols = []
    sdir = str(SYMBOLS_DIR)
    if not os.path.exists(sdir):
        return symbols
    for f in sorted(os.listdir(sdir)):
        if any(f.lower().endswith(e) for e in SYMBOL_EXTENSIONS):
            path = os.path.join(sdir, f)
            try:
                img = Image.open(path)
                code = re.match(r'(\d+)', f)
                symbols.append({
                    'code': code.group(1) if code else f.split('.')[0],
                    'filename': f,
                    'path': path,
                    'width_px': img.size[0],
                    'height_px': img.size[1],
                    'aspect': img.size[0] / max(img.size[1], 1)
                })
            except Exception:
                pass
    return symbols


def match_pngs_to_regions(symbol_regions, available_pngs):
    """
    Match available PNG symbols to detected PDF regions.
    Uses aspect ratio as primary matching criterion.
    Each PNG is used at most once; best aspect-ratio match wins.
    """
    if not symbol_regions or not available_pngs:
        return []

    matches = []
    used_pngs = set()

    # Sort regions by area (largest first = most important)
    sorted_regions = sorted(symbol_regions, key=lambda r: r['area_pt2'], reverse=True)

    for region in sorted_regions:
        best_idx = None
        best_err = float('inf')

        for i, png in enumerate(available_pngs):
            if i in used_pngs:
                continue
            err = abs(png['aspect'] - region['aspect'])
            if err < best_err:
                best_err = err
                best_idx = i

        if best_idx is not None:
            matches.append({
                'region': region,
                'png': available_pngs[best_idx]
            })
            used_pngs.add(best_idx)

    return matches


# ─── CATALOG & GENERATION ────────────────────────────────────────────────────

def load_catalog():
    global _cache, _cache_time
    now = time.time()
    if _cache and (now - _cache_time) < 300:
        return _cache

    available_pngs = get_available_pngs()
    products = []

    sdir = str(SYMBOLS_DIR)
    if os.path.exists(sdir):
        for f in sorted(os.listdir(sdir)):
            if f.lower().endswith('.pdf') and 'mart' in f.lower():
                pdf_path = os.path.join(sdir, f)
                layout = parse_label_pdf(pdf_path)
                if not layout:
                    continue

                matches = match_pngs_to_regions(
                    layout['symbol_regions'], available_pngs)

                label_id = f.replace('.pdf', '')
                products.append({
                    'product_code': layout['drawing_number'],
                    'product_desc': layout['title'],
                    'label_size': f"{layout['height_mm']} X {layout['width_mm']} mm",
                    'sheet_name': label_id,
                    'symbol_count': len(matches),
                    'width_mm': layout['width_mm'],
                    'height_mm': layout['height_mm'],
                    'matches': matches,
                    'text_fields': layout['text_fields'],
                    'debug': layout['debug']
                })

    _cache = {'products': products, 'pngs': available_pngs}
    _cache_time = now
    return _cache


def generate_label(label_id, dpi=600):
    """Render label: place PNGs at exact PDF positions, text as editable boxes."""
    catalog = load_catalog()
    product = next((p for p in catalog['products'] if p['sheet_name'] == label_id), None)
    if not product:
        raise ValueError(f"Label '{label_id}' not found. Available: "
                         f"{[p['sheet_name'] for p in catalog['products']]}")

    w_in = product['width_mm'] * MM_TO_INCH
    h_in = product['height_mm'] * MM_TO_INCH

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    ax.set_xlim(0, w_in)
    ax.set_ylim(0, h_in)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Label border
    ax.add_patch(Rectangle((0.02, 0.02), w_in - 0.04, h_in - 0.04,
                            linewidth=0.8, edgecolor='black',
                            facecolor='none', zorder=1))

    # ── Place symbols ──
    placed = 0
    for m in product['matches']:
        rgn = m['region']
        png = m['png']
        x = rgn['rel_x'] * w_in
        y = rgn['rel_y'] * h_in
        w = rgn['rel_w'] * w_in
        h = rgn['rel_h'] * h_in

        try:
            img = Image.open(png['path'])
            if img.mode == 'RGBA':
                bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            img = img.convert('RGB')
            ax.imshow(img, extent=[x, x + w, y, y + h],
                      aspect='auto', interpolation='lanczos', zorder=3)
            placed += 1
        except Exception as e:
            log.warning(f"Symbol render error {png['filename']}: {e}")

    # ── Text placeholders (empty editable boxes) ──
    for tf in product['text_fields']:
        tx = tf['rel_x'] * w_in
        ty = tf['rel_y'] * h_in
        tw = max(tf['rel_w'] * w_in, 0.3)
        th = max(tf['rel_h'] * h_in, 0.08)
        ax.add_patch(Rectangle((tx, ty), tw, th,
                               linewidth=0.3, edgecolor='#bbb',
                               facecolor='#fafafa', linestyle='--',
                               alpha=0.6, zorder=2))

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), placed, product


# ─── API ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    p = _APP_DIR / "index.html"
    if p.exists():
        return HTMLResponse(content=p.read_text())
    return HTMLResponse("<h1>B300 Label Generator</h1>")


@app.get("/api/catalog")
async def api_catalog():
    c = load_catalog()
    return {
        "products": [{
            'product_code': p['product_code'],
            'product_desc': p['product_desc'],
            'label_size': p['label_size'],
            'sheet_name': p['sheet_name'],
            'symbol_count': p['symbol_count']
        } for p in c['products']],
        "count": len(c['products'])
    }


@app.get("/api/generate/{label_id}")
async def api_generate(label_id: str, dpi: int = 600):
    try:
        img_bytes, placed, product = generate_label(label_id, dpi=dpi)
        return {
            "image": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}",
            "product_code": product['product_code'],
            "product_desc": product['product_desc'],
            "label_size": product['label_size'],
            "symbols_placed": placed,
            "convention": "pdf-extracted",
            "text_placeholders": len(product['text_fields'])
        }
    except Exception as e:
        log.error(f"Generate error: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/api/download/{label_id}")
async def api_download(label_id: str, dpi: int = 600):
    try:
        img_bytes, _, product = generate_label(label_id, dpi=dpi)
        return Response(content=img_bytes, media_type="image/png",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{label_id}.png"'})
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/api/health")
async def api_health():
    """Debug endpoint showing what was extracted from PDFs."""
    c = load_catalog()
    return {
        "status": "ok",
        "pdfplumber": HAS_PDFPLUMBER,
        "available_pngs": [p['filename'] for p in c['pngs']],
        "labels": [{
            'id': p['sheet_name'],
            'title': p['product_desc'],
            'size_mm': f"{p['width_mm']}x{p['height_mm']}",
            'symbols_matched': p['symbol_count'],
            'text_fields': len(p['text_fields']),
            'debug': p['debug']
        } for p in c['products']]
    }
