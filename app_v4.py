"""
B300 Label Generator v4
Extracts label boundary, symbol locations/sizes from PDF MART drawings using
spatial clustering of all graphical elements (lines, curves, rects).
Matches detected symbol regions to PNG files and renders at exact positions.
Text areas become empty editable fields preserving placement.
"""
import os
import re
import io
import base64
import logging
import time
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
log = logging.getLogger("b300-label")

app = FastAPI(title="B300 Label Generator", docs_url="/docs")

_APP_DIR = Path(__file__).parent
SYMBOLS_DIR = _APP_DIR / "data" / "symbols"
DPI = 600
MM_TO_INCH = 1 / 25.4
PT_TO_MM = 25.4 / 72
SYMBOL_EXT = ['.png', '.jpg', '.svg']

_cache = None
_cache_time = 0


# ======== PDF EXTRACTION ========

def find_label_boundary(page, expected_w_mm, expected_h_mm):
    """
    Find the label outline rectangle on the page.
    Strategy: look at ALL rects and lines to find a closed region matching
    the expected label dimensions.
    """
    pw, ph = page.width, page.height
    exp_w_pt = expected_w_mm / PT_TO_MM
    exp_h_pt = expected_h_mm / PT_TO_MM

    # Collect all rectangles
    all_rects = page.rects or []
    candidates = []
    for r in all_rects:
        x0 = min(r.get('x0', 0), r.get('x1', 0))
        y0 = min(r.get('top', 0), r.get('bottom', 0))
        x1 = max(r.get('x0', 0), r.get('x1', 0))
        y1 = max(r.get('top', 0), r.get('bottom', 0))
        w, h = x1 - x0, y1 - y0
        if w < 10 or h < 10:
            continue
        candidates.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w': w, 'h': h})

    # Also check for rectangles formed by 4 connected lines
    lines = page.lines or []
    # Group horizontal and vertical lines
    h_lines = [l for l in lines if abs(l.get('top', 0) - l.get('bottom', 0)) < 2]
    v_lines = [l for l in lines if abs(l.get('x0', 0) - l.get('x1', 0)) < 2]

    # Find the rectangle that best matches expected label dimensions
    best = None
    best_score = float('inf')
    for c in candidates:
        # Try both orientations
        score1 = abs(c['w'] - exp_w_pt)/exp_w_pt + abs(c['h'] - exp_h_pt)/exp_h_pt
        score2 = abs(c['w'] - exp_h_pt)/exp_h_pt + abs(c['h'] - exp_w_pt)/exp_w_pt
        score = min(score1, score2)
        if score < best_score:
            best_score = score
            best = c

    # Accept if within 60% tolerance; otherwise use largest rect in upper half
    if best and best_score < 0.6:
        return best

    # Fallback: largest rect not spanning the whole page
    non_page = [c for c in candidates if c['w'] < pw * 0.9 and c['h'] < ph * 0.9]
    if non_page:
        return max(non_page, key=lambda c: c['w'] * c['h'])

    # Last resort: infer from line endpoints in the drawing area
    if lines:
        xs = [l.get('x0', 0) for l in lines] + [l.get('x1', 0) for l in lines]
        ys = [l.get('top', 0) for l in lines] + [l.get('bottom', 0) for l in lines]
        # Filter to upper 60% of page (exclude title block)
        upper_ys = [y for y in ys if y < ph * 0.6]
        upper_xs = xs  # use all x
        if upper_ys and upper_xs:
            return {
                'x0': min(upper_xs), 'y0': min(upper_ys),
                'x1': max(upper_xs), 'y1': max(upper_ys),
                'w': max(upper_xs) - min(upper_xs),
                'h': max(upper_ys) - min(upper_ys)
            }

    return {'x0': 50, 'y0': 50, 'x1': 50 + exp_w_pt, 'y1': 50 + exp_h_pt,
            'w': exp_w_pt, 'h': exp_h_pt}


def cluster_elements(points, threshold):
    """
    Simple spatial clustering: group points that are within threshold distance.
    Returns list of clusters, each cluster is a list of (x, y) points.
    """
    if not points:
        return []
    clusters = []
    used = set()
    for i, p in enumerate(points):
        if i in used:
            continue
        cluster = [p]
        used.add(i)
        for j, q in enumerate(points):
            if j in used:
                continue
            if abs(p[0] - q[0]) < threshold and abs(p[1] - q[1]) < threshold:
                cluster.append(q)
                used.add(j)
        clusters.append(cluster)
    return clusters


def extract_symbol_regions(page, label_boundary):
    """
    Detect symbol regions inside the label by clustering all graphical elements.
    
    Symbols in engineering PDFs are drawn as vector art (lines, curves, arcs).
    We find all line/curve endpoints inside the label boundary, cluster them
    spatially, and each cluster's bounding box = one symbol region.
    """
    lx0, ly0 = label_boundary['x0'], label_boundary['y0']
    lx1, ly1 = label_boundary['x1'], label_boundary['y1']
    lw, lh = label_boundary['w'], label_boundary['h']

    # Collect all graphical element points inside the label
    inside_points = []

    # From lines
    for line in (page.lines or []):
        x0 = line.get('x0', 0)
        y0 = line.get('top', 0)
        x1 = line.get('x1', 0)
        y1 = line.get('bottom', 0)
        # Check if line is inside label boundary (with small margin)
        margin = 3
        if (x0 >= lx0 - margin and x1 <= lx1 + margin and
            y0 >= ly0 - margin and y1 <= ly1 + margin):
            # Exclude lines that span the full label (borders)
            line_w = abs(x1 - x0)
            line_h = abs(y1 - y0)
            if line_w > lw * 0.9 or line_h > lh * 0.9:
                continue
            inside_points.append((x0, y0))
            inside_points.append((x1, y1))

    # From rects inside label (excluding the label boundary itself)
    for r in (page.rects or []):
        rx0 = min(r.get('x0', 0), r.get('x1', 0))
        ry0 = min(r.get('top', 0), r.get('bottom', 0))
        rx1 = max(r.get('x0', 0), r.get('x1', 0))
        ry1 = max(r.get('top', 0), r.get('bottom', 0))
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < 5 or rh < 5:
            continue
        # Inside label but not the boundary itself
        if (rx0 >= lx0 + 2 and rx1 <= lx1 - 2 and
            ry0 >= ly0 + 2 and ry1 <= ly1 - 2):
            inside_points.append((rx0, ry0))
            inside_points.append((rx1, ry1))
            inside_points.append((rx0, ry1))
            inside_points.append((rx1, ry0))

    # From curves if available
    for curve in (page.curves or []):
        pts = curve.get('pts', []) or curve.get('points', [])
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                px, py = pt[0], pt[1]
                if lx0 <= px <= lx1 and ly0 <= py <= ly1:
                    inside_points.append((px, py))

    if not inside_points:
        return []

    # Cluster points spatially
    # Threshold: symbols are typically at least 10pt apart
    threshold = min(lw, lh) * 0.15  # 15% of label dimension
    clusters = cluster_elements(inside_points, threshold)

    # Convert clusters to bounding box regions
    regions = []
    min_size = 8  # minimum 8pt to be a symbol
    for cluster in clusters:
        if len(cluster) < 3:  # need at least 3 points for a meaningful shape
            continue
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)
        bw, bh = bx1 - bx0, by1 - by0
        if bw < min_size or bh < min_size:
            continue
        # Convert to relative coords within label (0-1)
        regions.append({
            'rel_x': (bx0 - lx0) / lw,
            'rel_y': 1.0 - (by1 - ly0) / lh,  # flip Y (PDF top-down → canvas bottom-up)
            'rel_w': bw / lw,
            'rel_h': bh / lh,
            'aspect': bw / max(bh, 1),
            'area': bw * bh,
            'num_points': len(cluster)
        })

    # Sort by area descending (most prominent symbols first)
    regions.sort(key=lambda r: r['area'], reverse=True)
    return regions


def extract_text_fields(page, label_boundary):
    """Extract text positions inside the label boundary."""
    lx0, ly0 = label_boundary['x0'], label_boundary['y0']
    lx1, ly1 = label_boundary['x1'], label_boundary['y1']
    lw, lh = label_boundary['w'], label_boundary['h']

    fields = []
    for word in (page.extract_words() or []):
        wx0 = word.get('x0', 0)
        wy0 = word.get('top', 0)
        wx1 = word.get('x1', 0)
        wy1 = word.get('bottom', 0)
        if (wx0 >= lx0 and wx1 <= lx1 and wy0 >= ly0 and wy1 <= ly1):
            fields.append({
                'text': word.get('text', ''),
                'rel_x': (wx0 - lx0) / lw,
                'rel_y': 1.0 - (wy1 - ly0) / lh,
                'rel_w': (wx1 - wx0) / lw,
                'rel_h': (wy1 - wy0) / lh
            })
    return fields


def parse_pdf_label(pdf_path):
    """Full PDF parse: dimensions, symbol regions, text fields."""
    if not HAS_PDFPLUMBER:
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            text = page.extract_text() or ''

            # Get label dimensions from text
            w_mm, h_mm = 85, 50
            m = re.search(r'\(?(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\)?', text)
            if m:
                d1, d2 = int(m.group(1)), int(m.group(2))
                w_mm, h_mm = max(d1, d2), min(d1, d2)

            # Find label boundary
            boundary = find_label_boundary(page, w_mm, h_mm)

            # Extract symbol regions
            symbol_regions = extract_symbol_regions(page, boundary)

            # Extract text fields
            text_fields = extract_text_fields(page, boundary)

            # Title
            title = os.path.basename(pdf_path).replace('.pdf', '')
            tm = re.search(r'TITLE\n(.+)', text)
            if tm:
                title = tm.group(1).strip()

            return {
                'width_mm': w_mm,
                'height_mm': h_mm,
                'title': title,
                'drawing_number': title.split('-M-')[0] if '-M-' in title else title,
                'symbol_regions': symbol_regions,
                'text_fields': text_fields,
                'boundary_pts': [boundary['x0'], boundary['y0'],
                                 boundary['w'], boundary['h']],
                'debug': {
                    'page_size': [page.width, page.height],
                    'total_lines': len(page.lines or []),
                    'total_rects': len(page.rects or []),
                    'total_curves': len(page.curves or []),
                    'regions_found': len(symbol_regions),
                    'texts_found': len(text_fields),
                    'boundary': boundary
                }
            }
    except Exception as e:
        log.error(f"PDF parse error: {e}")
        return None


# ======== SYMBOL MATCHING ========

def get_available_pngs():
    """Get available PNG symbol images with metadata."""
    pngs = []
    sdir = str(SYMBOLS_DIR)
    if not os.path.exists(sdir):
        return pngs
    for f in sorted(os.listdir(sdir)):
        if any(f.lower().endswith(e) for e in SYMBOL_EXT):
            path = os.path.join(sdir, f)
            try:
                img = Image.open(path)
                code = re.match(r'(\d+)', f)
                pngs.append({
                    'code': code.group(1) if code else f.split('.')[0],
                    'filename': f,
                    'path': path,
                    'w': img.size[0],
                    'h': img.size[1],
                    'aspect': img.size[0] / max(img.size[1], 1)
                })
            except Exception:
                pass
    return pngs


def match_pngs_to_regions(regions, pngs):
    """Match PNGs to detected regions. Best aspect-ratio match, each PNG used once."""
    if not regions or not pngs:
        # FALLBACK: if no regions detected, distribute PNGs evenly across label
        return []

    matches = []
    used = set()

    # Only consider regions up to the number of available PNGs
    top_regions = regions[:len(pngs) * 2]  # candidates = 2x PNGs

    for png_idx, png in enumerate(pngs):
        best_region_idx = None
        best_err = float('inf')
        for r_idx, region in enumerate(top_regions):
            if r_idx in used:
                continue
            err = abs(png['aspect'] - region['aspect'])
            if err < best_err:
                best_err = err
                best_region_idx = r_idx
        if best_region_idx is not None:
            matches.append({
                'region': top_regions[best_region_idx],
                'png': png
            })
            used.add(best_region_idx)

    return matches


def fallback_layout(pngs, w_in, h_in):
    """If PDF extraction finds no regions, distribute PNGs across the label."""
    matches = []
    n = len(pngs)
    if n == 0:
        return matches

    margin = 0.1
    usable_w = w_in - 2 * margin
    usable_h = h_in - 2 * margin

    # Layout: wide symbols at bottom, tall ones on right, square in center
    # Sort by aspect ratio
    sorted_pngs = sorted(pngs, key=lambda p: p['aspect'], reverse=True)

    cols = min(n, 3)
    cell_w = usable_w / cols

    for i, png in enumerate(sorted_pngs):
        col = i % cols
        row = i // cols
        cell_h = usable_h / max(1, (n + cols - 1) // cols)

        # Scale to fit cell
        png_w_in = png['w'] / DPI
        png_h_in = png['h'] / DPI
        scale = min((cell_w * 0.9) / png_w_in, (cell_h * 0.9) / png_h_in, 1.0)
        sw = png_w_in * scale
        sh = png_h_in * scale

        x = margin + col * cell_w + (cell_w - sw) / 2
        y = margin + row * cell_h + (cell_h - sh) / 2

        matches.append({
            'region': {'rel_x': x / w_in, 'rel_y': y / h_in,
                       'rel_w': sw / w_in, 'rel_h': sh / h_in},
            'png': png
        })
    return matches


# ======== CATALOG & RENDER ========

def load_catalog():
    global _cache, _cache_time
    now = time.time()
    if _cache and (now - _cache_time) < 300:
        return _cache

    pngs = get_available_pngs()
    products = []
    sdir = str(SYMBOLS_DIR)

    if os.path.exists(sdir):
        for f in sorted(os.listdir(sdir)):
            if f.lower().endswith('.pdf') and 'mart' in f.lower():
                layout = parse_pdf_label(os.path.join(sdir, f))
                if not layout:
                    continue
                matches = match_pngs_to_regions(layout['symbol_regions'], pngs)
                # Fallback if no regions detected
                if not matches:
                    w_in = layout['width_mm'] * MM_TO_INCH
                    h_in = layout['height_mm'] * MM_TO_INCH
                    matches = fallback_layout(pngs, w_in, h_in)

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

    _cache = {'products': products, 'pngs': pngs}
    _cache_time = now
    return _cache


def render_label(label_id, dpi=600):
    """Render label with PNG symbols at positions and editable text boxes."""
    catalog = load_catalog()
    product = next((p for p in catalog['products'] if p['sheet_name'] == label_id), None)
    if not product:
        raise ValueError(f"Label not found: {label_id}")

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
                            lw=0.8, edgecolor='black', facecolor='none', zorder=1))

    # Place symbols
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
            log.warning(f"Render error {png['filename']}: {e}")

    # Editable text placeholders
    for tf in product.get('text_fields', []):
        tx = tf['rel_x'] * w_in
        ty = tf['rel_y'] * h_in
        tw = max(tf['rel_w'] * w_in, 0.2)
        th = max(tf['rel_h'] * h_in, 0.06)
        ax.add_patch(Rectangle((tx, ty), tw, th,
                               lw=0.25, edgecolor='#ccc', facecolor='#fafafa',
                               linestyle=':', alpha=0.5, zorder=2))

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), placed, product


# ======== API ========

@app.get("/", response_class=HTMLResponse)
async def home():
    p = _APP_DIR / "index.html"
    return HTMLResponse(p.read_text()) if p.exists() else HTMLResponse("<h1>B300</h1>")


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
        img_bytes, placed, product = render_label(label_id, dpi)
        return {
            "image": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}",
            "product_code": product['product_code'],
            "product_desc": product['product_desc'],
            "label_size": product['label_size'],
            "symbols_placed": placed,
            "convention": "pdf-extracted"
        }
    except Exception as e:
        log.error(f"Generate error: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/api/download/{label_id}")
async def api_download(label_id: str, dpi: int = 600):
    try:
        img_bytes, _, _ = render_label(label_id, dpi)
        return Response(content=img_bytes, media_type="image/png",
                        headers={"Content-Disposition": f'attachment; filename="{label_id}.png"'})
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/api/health")
async def api_health():
    """Debug: shows what was extracted from each PDF."""
    c = load_catalog()
    return {
        "status": "ok",
        "pdfplumber": HAS_PDFPLUMBER,
        "pngs": [{"file": p['filename'], "aspect": round(p['aspect'], 2),
                  "size": f"{p['w']}x{p['h']}"} for p in c['pngs']],
        "labels": [{
            'id': p['sheet_name'],
            'size': p['label_size'],
            'symbols_matched': p['symbol_count'],
            'text_fields': len(p.get('text_fields', [])),
            'debug': p.get('debug', {})
        } for p in c['products']]
    }
