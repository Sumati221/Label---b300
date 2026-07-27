"""
B300 Label Generator v5

For each source PDF:
1. Detect label dimensions and boundary
2. Detect the PHILIPS logo and all graphical symbols
3. Match each graphic to the corresponding PNG in data/symbols/
4. Preserve relative visual order (e.g. logo top-left, cert mark below text)
5. Text regions -> empty editable text fields (same size/position/alignment)
6. Works dynamically for multiple label PDFs with different layouts
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
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from PIL import Image

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("label-gen")

app = FastAPI(title="B300 Label Generator", docs_url="/docs")

_APP = Path(__file__).parent
_SYMBOLS = _APP / "data" / "symbols"
MM2IN = 1 / 25.4
PT2MM = 25.4 / 72
_cache = None
_cache_t = 0


# ════════════════════════════════════════════════════════════════════
# PNG SYMBOL INVENTORY
# ════════════════════════════════════════════════════════════════════

def scan_pngs():
    """Find all PNG/JPG/SVG symbol images. Returns list with metadata."""
    pngs = []
    d = str(_SYMBOLS)
    if not os.path.isdir(d):
        log.warning(f"Symbols dir not found: {d}")
        return pngs
    for f in sorted(os.listdir(d)):
        low = f.lower()
        if not (low.endswith('.png') or low.endswith('.jpg') or low.endswith('.svg')):
            continue
        path = os.path.join(d, f)
        try:
            img = Image.open(path)
            w, h = img.size
            code_m = re.match(r'(\d+)', f)
            pngs.append({
                'code': code_m.group(1) if code_m else f.split('.')[0],
                'file': f, 'path': path,
                'w': w, 'h': h,
                'aspect': w / max(h, 1)
            })
            log.info(f"  PNG: {f} ({w}x{h}, aspect={w/max(h,1):.2f})")
        except Exception as e:
            log.warning(f"Cannot open {f}: {e}")
    log.info(f"Total PNGs found: {len(pngs)}")
    return pngs


# ════════════════════════════════════════════════════════════════════
# PDF LABEL EXTRACTION
# ════════════════════════════════════════════════════════════════════

def extract_label(pdf_path, available_pngs):
    """
    Parse a single MART PDF and extract:
    - Label dimensions (from embedded text like '50mm x 85mm')
    - Label boundary on the page
    - All graphical symbol regions with position + size
    - All text regions with position + size
    - Match symbols to PNGs by aspect ratio
    """
    if not HAS_PDF:
        return None

    fname = os.path.basename(pdf_path)
    log.info(f"Parsing PDF: {fname}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            pw, ph = page.width, page.height
            full_text = page.extract_text() or ''

            # ── 1. LABEL DIMENSIONS ──
            w_mm, h_mm = 85, 50  # sensible default
            m = re.search(r'\(?(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\)?', full_text)
            if m:
                d1, d2 = int(m.group(1)), int(m.group(2))
                w_mm, h_mm = max(d1, d2), min(d1, d2)

            # ── 2. GATHER ALL GRAPHICAL ELEMENTS ──
            all_rects = page.rects or []
            all_lines = page.lines or []
            all_curves = page.curves or []

            log.info(f"  Page: {pw:.0f}x{ph:.0f}pt | "
                     f"rects={len(all_rects)} lines={len(all_lines)} "
                     f"curves={len(all_curves)}")

            # ── 3. FIND LABEL BOUNDARY ──
            # Normalize rects
            norm_rects = []
            for r in all_rects:
                x0 = min(r.get('x0', 0), r.get('x1', 0))
                y0 = min(r.get('top', 0), r.get('bottom', 0))
                x1 = max(r.get('x0', 0), r.get('x1', 0))
                y1 = max(r.get('top', 0), r.get('bottom', 0))
                w, h = x1 - x0, y1 - y0
                if w > 5 and h > 5:
                    norm_rects.append({'x0':x0,'y0':y0,'x1':x1,'y1':y1,'w':w,'h':h})

            # Expected label size in points
            exp_w = w_mm / PT2MM
            exp_h = h_mm / PT2MM

            # Score each rect against expected label dims (try both orientations)
            label_rect = None
            best = 999
            for r in norm_rects:
                s1 = abs(r['w']-exp_w)/exp_w + abs(r['h']-exp_h)/exp_h
                s2 = abs(r['w']-exp_h)/exp_h + abs(r['h']-exp_w)/exp_w
                s = min(s1, s2)
                if s < best and r['w'] < pw*0.95 and r['h'] < ph*0.95:
                    best = s
                    label_rect = r

            if not label_rect or best > 1.0:
                # Fallback: use all lines to determine the drawing area bounds
                # excluding the title block (bottom ~35% of page)
                cutoff_y = ph * 0.6
                pts_x, pts_y = [], []
                for ln in all_lines:
                    for key_x in ['x0','x1']:
                        pts_x.append(ln.get(key_x, 0))
                    for key_y in ['top','bottom']:
                        y = ln.get(key_y, 0)
                        if y < cutoff_y:
                            pts_y.append(y)
                if pts_x and pts_y:
                    label_rect = {
                        'x0': min(pts_x), 'y0': min(pts_y),
                        'x1': max(pts_x), 'y1': max(pts_y),
                        'w': max(pts_x)-min(pts_x), 'h': max(pts_y)-min(pts_y)
                    }
                else:
                    label_rect = {'x0':40,'y0':40,'x1':40+exp_w,'y1':40+exp_h,
                                  'w':exp_w,'h':exp_h}

            lx, ly, lw, lh = label_rect['x0'], label_rect['y0'], label_rect['w'], label_rect['h']
            log.info(f"  Label boundary: ({lx:.0f},{ly:.0f}) {lw:.0f}x{lh:.0f}pt")

            # ── 4. FIND SYMBOL REGIONS ──
            # Strategy A: rects inside label boundary (not the boundary itself)
            inner_rects = []
            for r in norm_rects:
                if r is label_rect:
                    continue
                # Must be meaningfully inside the label
                if (r['x0'] >= lx + 1 and r['x1'] <= lx + lw - 1 and
                    r['y0'] >= ly + 1 and r['y1'] <= ly + lh - 1):
                    # Skip very thin lines masquerading as rects
                    if r['w'] > 8 and r['h'] > 8:
                        inner_rects.append(r)

            # Strategy B: cluster line endpoints inside label
            inner_line_pts = []
            for ln in all_lines:
                x0 = ln.get('x0', 0)
                y0_l = ln.get('top', 0)
                x1 = ln.get('x1', 0)
                y1_l = ln.get('bottom', 0)
                # Inside label?
                if (min(x0,x1) >= lx and max(x0,x1) <= lx+lw and
                    min(y0_l,y1_l) >= ly and max(y0_l,y1_l) <= ly+lh):
                    # Exclude full-width/height lines (borders)
                    span_x = abs(x1 - x0)
                    span_y = abs(y1_l - y0_l)
                    if span_x > lw * 0.85 or span_y > lh * 0.85:
                        continue
                    inner_line_pts.append((x0, y0_l))
                    inner_line_pts.append((x1, y1_l))

            # Strategy C: curves inside label
            for curve in all_curves:
                pts = curve.get('pts', []) or curve.get('points', [])
                for pt in (pts if pts else []):
                    if isinstance(pt, (list,tuple)) and len(pt)>=2:
                        px, py = float(pt[0]), float(pt[1])
                        if lx <= px <= lx+lw and ly <= py <= ly+lh:
                            inner_line_pts.append((px, py))

            # Build symbol regions from inner rects
            symbol_regions = []
            for r in inner_rects:
                symbol_regions.append({
                    'x_pt': r['x0'] - lx,
                    'y_pt': r['y0'] - ly,
                    'w_pt': r['w'],
                    'h_pt': r['h'],
                    'aspect': r['w'] / max(r['h'], 1),
                    'source': 'rect'
                })

            # If few/no rects found, cluster line points
            if len(symbol_regions) < len(available_pngs) and inner_line_pts:
                # Grid-based clustering
                grid_size = min(lw, lh) * 0.12
                grid = {}
                for (px, py) in inner_line_pts:
                    gx = int((px - lx) / grid_size)
                    gy = int((py - ly) / grid_size)
                    key = (gx, gy)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append((px, py))

                # Merge adjacent grid cells
                visited = set()
                clusters = []
                for key in grid:
                    if key in visited:
                        continue
                    # BFS to find connected cells
                    cluster_pts = []
                    queue = [key]
                    while queue:
                        k = queue.pop(0)
                        if k in visited:
                            continue
                        visited.add(k)
                        if k in grid:
                            cluster_pts.extend(grid[k])
                            # Check neighbors
                            for dx in [-1,0,1]:
                                for dy in [-1,0,1]:
                                    nk = (k[0]+dx, k[1]+dy)
                                    if nk in grid and nk not in visited:
                                        queue.append(nk)
                    if len(cluster_pts) >= 4:
                        clusters.append(cluster_pts)

                # Convert clusters to regions
                for cl in clusters:
                    xs = [p[0] for p in cl]
                    ys = [p[1] for p in cl]
                    bx0, bx1 = min(xs), max(xs)
                    by0, by1 = min(ys), max(ys)
                    bw, bh = bx1-bx0, by1-by0
                    if bw > 8 and bh > 8:
                        symbol_regions.append({
                            'x_pt': bx0 - lx,
                            'y_pt': by0 - ly,
                            'w_pt': bw,
                            'h_pt': bh,
                            'aspect': bw / max(bh, 1),
                            'source': 'cluster'
                        })

            log.info(f"  Symbol regions found: {len(symbol_regions)}")

            # ── 5. MATCH SYMBOLS TO PNGs ──
            # Sort regions top-to-bottom, left-to-right (preserves visual order)
            symbol_regions.sort(key=lambda r: (r['y_pt'], r['x_pt']))

            matches = []
            used_pngs = set()
            for region in symbol_regions:
                best_i = None
                best_err = float('inf')
                for i, png in enumerate(available_pngs):
                    if i in used_pngs:
                        continue
                    err = abs(png['aspect'] - region['aspect'])
                    if err < best_err:
                        best_err = err
                        best_i = i
                if best_i is not None and best_err < 15:  # very generous
                    matches.append({
                        'png': available_pngs[best_i],
                        'rel_x': region['x_pt'] / lw,
                        'rel_y': 1.0 - (region['y_pt'] + region['h_pt']) / lh,
                        'rel_w': region['w_pt'] / lw,
                        'rel_h': region['h_pt'] / lh
                    })
                    used_pngs.add(best_i)
                    if len(used_pngs) >= len(available_pngs):
                        break

            # FALLBACK: if no matches, use positional heuristic
            if not matches and available_pngs:
                log.warning(f"  No regions matched, using heuristic layout")
                # Sort PNGs: widest first (logo at top), then by aspect
                sorted_pngs = sorted(available_pngs, key=lambda p: -p['aspect'])
                n = len(sorted_pngs)
                for i, png in enumerate(sorted_pngs):
                    # Stack vertically with proportional sizing
                    total_h = sum(1/max(p['aspect'],0.5) for p in sorted_pngs)
                    rel_h = (1/max(png['aspect'],0.5)) / total_h * 0.85
                    rel_w = min(0.9, rel_h * png['aspect'] * (h_mm/w_mm))
                    y_offset = sum(
                        (1/max(sorted_pngs[j]['aspect'],0.5))/total_h*0.85
                        for j in range(i)
                    )
                    matches.append({
                        'png': png,
                        'rel_x': 0.05,
                        'rel_y': 0.90 - y_offset - rel_h,
                        'rel_w': rel_w,
                        'rel_h': rel_h
                    })

            log.info(f"  Matches: {len(matches)} "
                     f"({[m['png']['file'] for m in matches]})")

            # ── 6. TEXT FIELDS ──
            text_fields = []
            words = page.extract_words() or []
            for word in words:
                wx0 = word.get('x0', 0)
                wy0 = word.get('top', 0)
                wx1 = word.get('x1', 0)
                wy1 = word.get('bottom', 0)
                # Inside label?
                if (wx0 >= lx and wx1 <= lx+lw and wy0 >= ly and wy1 <= ly+lh):
                    text_fields.append({
                        'rel_x': (wx0 - lx) / lw,
                        'rel_y': 1.0 - (wy1 - ly) / lh,
                        'rel_w': (wx1 - wx0) / lw,
                        'rel_h': (wy1 - wy0) / lh,
                        'original': word.get('text', '')
                    })

            # Group adjacent words into text blocks
            text_blocks = merge_text_fields(text_fields)

            # ── 7. BUILD TITLE ──
            title = fname.replace('.pdf', '')
            tm = re.search(r'TITLE\n(.+)', full_text)
            if tm:
                title = tm.group(1).strip()

            return {
                'id': fname.replace('.pdf', ''),
                'title': title,
                'drawing_num': title.split('-M-')[0] if '-M-' in title else title,
                'w_mm': w_mm, 'h_mm': h_mm,
                'matches': matches,
                'text_blocks': text_blocks,
                'debug': {
                    'page': f"{pw:.0f}x{ph:.0f}",
                    'rects': len(all_rects),
                    'lines': len(all_lines),
                    'curves': len(all_curves),
                    'inner_rects': len(inner_rects),
                    'inner_line_pts': len(inner_line_pts),
                    'symbol_regions': len(symbol_regions),
                    'matches': len(matches),
                    'text_blocks': len(text_blocks)
                }
            }

    except Exception as e:
        log.error(f"Error parsing {fname}: {e}", exc_info=True)
        return None


def merge_text_fields(fields):
    """Merge individual words into text blocks (same line = one block)."""
    if not fields:
        return []
    # Group by approximate Y position (same line)
    fields_sorted = sorted(fields, key=lambda f: (-f['rel_y'], f['rel_x']))
    blocks = []
    current = None
    for f in fields_sorted:
        if current is None:
            current = dict(f)
            continue
        # Same line if Y difference is small
        if abs(f['rel_y'] - current['rel_y']) < 0.03:
            # Extend block
            current['rel_w'] = (f['rel_x'] + f['rel_w']) - current['rel_x']
            current['original'] += ' ' + f['original']
        else:
            blocks.append(current)
            current = dict(f)
    if current:
        blocks.append(current)
    return blocks


# ════════════════════════════════════════════════════════════════════
# CATALOG & RENDER
# ════════════════════════════════════════════════════════════════════

def load_catalog():
    global _cache, _cache_t
    now = time.time()
    if _cache and (now - _cache_t) < 300:
        return _cache

    pngs = scan_pngs()
    products = []

    sdir = str(_SYMBOLS)
    if os.path.isdir(sdir):
        for f in sorted(os.listdir(sdir)):
            if f.lower().endswith('.pdf') and 'mart' in f.lower():
                result = extract_label(os.path.join(sdir, f), pngs)
                if result:
                    products.append(result)

    _cache = {'products': products, 'pngs': pngs}
    _cache_t = now
    log.info(f"Catalog: {len(products)} labels, {len(pngs)} PNGs")
    return _cache


def render(label_id, dpi=600):
    """Render label with matched PNGs and editable text placeholders."""
    cat = load_catalog()
    prod = next((p for p in cat['products'] if p['id'] == label_id), None)
    if not prod:
        avail = [p['id'] for p in cat['products']]
        raise ValueError(f"'{label_id}' not found. Available: {avail}")

    w_in = prod['w_mm'] * MM2IN
    h_in = prod['h_mm'] * MM2IN

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    ax.set_xlim(0, w_in)
    ax.set_ylim(0, h_in)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Border
    ax.add_patch(Rectangle((0.015, 0.015), w_in-0.03, h_in-0.03,
                            lw=0.6, ec='black', fc='none', zorder=1))

    # ── Place symbols ──
    placed = 0
    for m in prod['matches']:
        x = m['rel_x'] * w_in
        y = m['rel_y'] * h_in
        w = m['rel_w'] * w_in
        h = m['rel_h'] * h_in
        try:
            img = Image.open(m['png']['path'])
            if img.mode == 'RGBA':
                bg = Image.new('RGBA', img.size, (255,255,255,255))
                img = Image.alpha_composite(bg, img)
            img = img.convert('RGB')
            ax.imshow(img, extent=[x, x+w, y, y+h],
                      aspect='auto', interpolation='lanczos', zorder=3)
            placed += 1
        except Exception as e:
            log.warning(f"Render fail {m['png']['file']}: {e}")

    # ── Editable text placeholders ──
    for tb in prod.get('text_blocks', []):
        tx = tb['rel_x'] * w_in
        ty = tb['rel_y'] * h_in
        tw = max(tb['rel_w'] * w_in, 0.15)
        th = max(tb['rel_h'] * h_in, 0.05)
        ax.add_patch(Rectangle((tx, ty), tw, th,
                               lw=0.2, ec='#bbb', fc='#f9f9f9',
                               ls=':', alpha=0.5, zorder=2))

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), placed, prod


# ════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home():
    p = _APP / "index.html"
    return HTMLResponse(p.read_text()) if p.exists() else HTMLResponse("<h1>B300</h1>")


@app.get("/api/catalog")
async def api_catalog():
    c = load_catalog()
    return {
        "products": [{
            'product_code': p['drawing_num'],
            'product_desc': p['title'],
            'label_size': f"{p['h_mm']} X {p['w_mm']} mm",
            'sheet_name': p['id'],
            'symbol_count': len(p['matches'])
        } for p in c['products']],
        "count": len(c['products'])
    }


@app.get("/api/generate/{label_id}")
async def api_generate(label_id: str, dpi: int = 600):
    try:
        img_bytes, placed, prod = render(label_id, dpi)
        return {
            "image": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}",
            "product_code": prod['drawing_num'],
            "product_desc": prod['title'],
            "label_size": f"{prod['h_mm']} X {prod['w_mm']} mm",
            "symbols_placed": placed,
            "convention": "pdf-extracted"
        }
    except Exception as e:
        log.error(f"Generate: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/api/download/{label_id}")
async def api_download(label_id: str, dpi: int = 600):
    try:
        img_bytes, _, _ = render(label_id, dpi)
        return Response(content=img_bytes, media_type="image/png",
                        headers={"Content-Disposition": f'attachment; filename="{label_id}.png"'})
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/api/health")
async def api_health():
    c = load_catalog()
    return {
        "status": "ok",
        "pdfplumber": HAS_PDF,
        "symbols_dir": str(_SYMBOLS),
        "symbols_dir_exists": os.path.isdir(str(_SYMBOLS)),
        "files_in_symbols_dir": os.listdir(str(_SYMBOLS)) if os.path.isdir(str(_SYMBOLS)) else [],
        "pngs": [{'file': p['file'], 'size': f"{p['w']}x{p['h']}",
                  'aspect': round(p['aspect'],2)} for p in c['pngs']],
        "labels": [{
            'id': p['id'], 'size': f"{p['w_mm']}x{p['h_mm']}mm",
            'symbols_matched': len(p['matches']),
            'matched_files': [m['png']['file'] for m in p['matches']],
            'text_blocks': len(p.get('text_blocks', [])),
            'debug': p.get('debug', {})
        } for p in c['products']]
    }


@app.on_event("startup")
async def startup():
    """Pre-load catalog on startup so first request is fast."""
    log.info("=== B300 Label Generator v5 starting ===")
    log.info(f"Symbols dir: {_SYMBOLS}")
    log.info(f"Dir exists: {os.path.isdir(str(_SYMBOLS))}")
    if os.path.isdir(str(_SYMBOLS)):
        log.info(f"Contents: {os.listdir(str(_SYMBOLS))}")
    load_catalog()
