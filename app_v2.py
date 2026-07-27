"""
B300 Label Generator App v2
Extracts exact symbol placement from PDF label specs (MART drawings).
Each PDF = one country-specific label. Symbols are matched to PNG files
and positioned precisely. Text areas become empty editable placeholders.
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
import numpy as np
import pandas as pd
from PIL import Image

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("b300-label-generator")

app = FastAPI(title="B300 Label Generator", docs_url="/docs")

# === CONFIGURATION ===
_APP_DIR = Path(__file__).parent
DATA_DIR = _APP_DIR / "data"
SYMBOLS_DIR = DATA_DIR / "symbols"
DPI = 600
MM_TO_INCH = 1 / 25.4
PT_TO_INCH = 1 / 72
SYMBOL_IMAGE_EXTENSIONS = ['.png', '.jpg', '.svg']

# === CACHED STATE ===
_catalog_cache = None
_catalog_cache_time = 0
CACHE_TTL = 300


# === PDF PARSING ===

def find_pdf_labels():
    """Find all MART PDF label specs in the symbols directory."""
    pdfs = []
    symbols_dir = str(SYMBOLS_DIR)
    if not os.path.exists(symbols_dir):
        return pdfs
    for f in sorted(os.listdir(symbols_dir)):
        if f.lower().endswith('.pdf') and 'MART' in f.upper():
            pdfs.append({
                'filename': f,
                'path': os.path.join(symbols_dir, f)
            })
    return pdfs


def extract_label_from_pdf(pdf_path):
    """Extract label layout from a PDF engineering drawing.
    
    Returns label dimensions, symbol regions (rectangles), and text positions.
    The PDF is an A4 engineering drawing with the label drawn at 1:1 scale.
    """
    if not HAS_PDFPLUMBER:
        return None

    result = {
        'width_mm': 85,
        'height_mm': 50,
        'title': '',
        'drawing_number': '',
        'symbols': [],
        'texts': []
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            page_w = page.width   # in points
            page_h = page.height  # in points
            text = page.extract_text() or ''

            # Extract label dimensions from text
            size_match = re.search(r'\(?(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\)?', text)
            if size_match:
                result['width_mm'] = int(size_match.group(2))  # wider dim
                result['height_mm'] = int(size_match.group(1))  # shorter dim
                # Ensure width > height
                if result['width_mm'] < result['height_mm']:
                    result['width_mm'], result['height_mm'] = result['height_mm'], result['width_mm']

            # Extract title
            title_match = re.search(r'TITLE\s*\n(.+?)\n', text)
            if title_match:
                result['title'] = title_match.group(1).strip()
            else:
                # Try drawing number from filename
                fname = os.path.basename(pdf_path)
                result['title'] = fname.replace('.pdf', '').replace('-M-MART_', ' Rev ')

            # Extract drawing number
            num_match = re.search(r'DRAWING NUMBER.*?\n(\S+)', text)
            if num_match:
                result['drawing_number'] = num_match.group(1)
            else:
                result['drawing_number'] = os.path.basename(pdf_path).split('-M-')[0]

            # Get all rectangles on the page
            rects = page.rects or []
            lines = page.lines or []

            # Find the label boundary: look for a rectangle that matches
            # the expected label dimensions (within the drawing area)
            label_w_pt = result['width_mm'] / 25.4 * 72  # mm to points
            label_h_pt = result['height_mm'] / 25.4 * 72

            # Title block is typically in the bottom-right corner
            # The label drawing area is in the upper/left portion
            # Find rectangles that could be symbol bounding boxes
            # (smaller than the label, within the drawing area)

            # Identify the title block boundary (largest rect near bottom)
            # and exclude it
            title_block_top = page_h * 0.6  # Title block is usually bottom 40%

            # Get all rectangles in the drawing area (above title block)
            drawing_rects = []
            for r in rects:
                x0 = r.get('x0', 0)
                y0 = r.get('top', 0)
                x1 = r.get('x1', 0)
                y1 = r.get('bottom', 0)
                w = abs(x1 - x0)
                h = abs(y1 - y0)
                # Skip very tiny rects (decorative) and page-sized rects
                if w < 5 or h < 5:
                    continue
                if w > page_w * 0.8 and h > page_h * 0.8:
                    continue
                # Skip rects in the title block area
                if y0 > title_block_top:
                    continue
                drawing_rects.append({
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                    'width': w, 'height': h,
                    'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2
                })

            # Find the label outline (should be close to expected dimensions)
            label_rect = None
            best_match = float('inf')
            for r in drawing_rects:
                # Check if this rect matches label dimensions (tolerance 20%)
                w_ratio = r['width'] / label_w_pt if label_w_pt > 0 else 0
                h_ratio = r['height'] / label_h_pt if label_h_pt > 0 else 0
                if 0.7 < w_ratio < 1.3 and 0.7 < h_ratio < 1.3:
                    err = abs(w_ratio - 1) + abs(h_ratio - 1)
                    if err < best_match:
                        best_match = err
                        label_rect = r

            # If no label rect found, use the drawing area bounds
            if not label_rect:
                if drawing_rects:
                    all_x0 = min(r['x0'] for r in drawing_rects)
                    all_y0 = min(r['y0'] for r in drawing_rects)
                    all_x1 = max(r['x1'] for r in drawing_rects)
                    all_y1 = max(r['y1'] for r in drawing_rects)
                    label_rect = {
                        'x0': all_x0, 'y0': all_y0,
                        'x1': all_x1, 'y1': all_y1,
                        'width': all_x1 - all_x0,
                        'height': all_y1 - all_y0
                    }
                else:
                    # Fallback: use center of page
                    label_rect = {
                        'x0': page_w * 0.1, 'y0': page_h * 0.1,
                        'x1': page_w * 0.1 + label_w_pt,
                        'y1': page_h * 0.1 + label_h_pt,
                        'width': label_w_pt, 'height': label_h_pt
                    }

            # Now find symbol regions INSIDE the label rect
            label_x0 = label_rect['x0']
            label_y0 = label_rect['y0']
            label_w = label_rect['width']
            label_h = label_rect['height']

            for r in drawing_rects:
                # Skip the label outline itself
                if r == label_rect:
                    continue
                # Check if rect is inside the label
                if (r['x0'] >= label_x0 - 2 and r['x1'] <= label_x0 + label_w + 2 and
                    r['y0'] >= label_y0 - 2 and r['y1'] <= label_y0 + label_h + 2):
                    # Convert to relative position (0-1 range within label)
                    rel_x = (r['x0'] - label_x0) / label_w
                    rel_y = (r['y0'] - label_y0) / label_h
                    rel_w = r['width'] / label_w
                    rel_h = r['height'] / label_h
                    result['symbols'].append({
                        'rel_x': rel_x,
                        'rel_y': 1.0 - rel_y - rel_h,  # flip Y (PDF is top-down)
                        'rel_w': rel_w,
                        'rel_h': rel_h,
                        'type': 'symbol_region'
                    })

            # Extract text with positions (within label area)
            words = page.extract_words() or []
            for w in words:
                wx0 = w.get('x0', 0)
                wy0 = w.get('top', 0)
                # Check if inside label rect
                if (wx0 >= label_x0 - 2 and wx0 <= label_x0 + label_w + 2 and
                    wy0 >= label_y0 - 2 and wy0 <= label_y0 + label_h + 2):
                    rel_x = (wx0 - label_x0) / label_w
                    rel_y = 1.0 - (wy0 - label_y0) / label_h
                    result['texts'].append({
                        'text': w.get('text', ''),
                        'rel_x': rel_x,
                        'rel_y': rel_y,
                        'font_size': w.get('size', 8)
                    })

    except Exception as e:
        log.error(f"Error parsing PDF {pdf_path}: {e}")

    return result


def find_symbol_image(sym_code):
    """Find symbol image by code (tries exact then partial match)."""
    symbols_dir = str(SYMBOLS_DIR)
    if not os.path.exists(symbols_dir):
        return None
    codes = [sym_code]
    if sym_code.startswith('LS-'):
        codes.append(sym_code[3:])
    for code in codes:
        for ext in SYMBOL_IMAGE_EXTENSIONS:
            p = os.path.join(symbols_dir, f"{code}{ext}")
            if os.path.exists(p):
                return p
        for f in os.listdir(symbols_dir):
            if f.startswith(code) and any(f.lower().endswith(e) for e in SYMBOL_IMAGE_EXTENSIONS):
                return os.path.join(symbols_dir, f)
    return None


def get_available_symbol_images():
    """Get all available PNG/JPG/SVG symbol images."""
    symbols = []
    symbols_dir = str(SYMBOLS_DIR)
    if not os.path.exists(symbols_dir):
        return symbols
    for f in sorted(os.listdir(symbols_dir)):
        if any(f.lower().endswith(ext) for ext in SYMBOL_IMAGE_EXTENSIONS):
            path = os.path.join(symbols_dir, f)
            img = Image.open(path)
            code = re.match(r'(\d+)', f)
            symbols.append({
                'code': code.group(1) if code else f.split('.')[0],
                'filename': f,
                'path': path,
                'width_px': img.size[0],
                'height_px': img.size[1],
                'aspect': img.size[0] / img.size[1]
            })
    return symbols


def match_symbols_to_regions(regions, available_symbols):
    """Match available PNG symbols to detected regions by aspect ratio."""
    if not regions or not available_symbols:
        return []

    matched = []
    used_symbols = set()

    # Sort regions by area (largest first)
    sorted_regions = sorted(regions, key=lambda r: r['rel_w'] * r['rel_h'], reverse=True)

    for region in sorted_regions:
        if region['rel_w'] <= 0 or region['rel_h'] <= 0:
            continue
        region_aspect = region['rel_w'] / region['rel_h']

        best_sym = None
        best_err = float('inf')

        for i, sym in enumerate(available_symbols):
            if i in used_symbols:
                continue
            err = abs(sym['aspect'] - region_aspect)
            if err < best_err:
                best_err = err
                best_sym = i

        if best_sym is not None and best_err < 5.0:  # Generous tolerance
            matched.append({
                'region': region,
                'symbol': available_symbols[best_sym]
            })
            used_symbols.add(best_sym)

    # If we have unmatched symbols and regions, fill remaining
    remaining_regions = [r for i, r in enumerate(sorted_regions)
                         if not any(m['region'] == r for m in matched)]
    remaining_symbols = [s for i, s in enumerate(available_symbols)
                         if i not in used_symbols]

    for region, sym in zip(remaining_regions, remaining_symbols):
        matched.append({'region': region, 'symbol': sym})

    return matched


def load_catalog():
    """Build catalog from PDF label specs."""
    global _catalog_cache, _catalog_cache_time
    now = time.time()
    if _catalog_cache and (now - _catalog_cache_time) < CACHE_TTL:
        return _catalog_cache

    pdf_labels = find_pdf_labels()
    available_symbols = get_available_symbol_images()
    products = []

    for pdf in pdf_labels:
        layout = extract_label_from_pdf(pdf['path'])
        if not layout:
            continue

        # Match symbols to regions
        matches = match_symbols_to_regions(layout['symbols'], available_symbols)

        label_id = pdf['filename'].replace('.pdf', '')
        products.append({
            'product_code': layout['drawing_number'],
            'product_desc': layout['title'],
            'label_size': f"{layout['height_mm']} X {layout['width_mm']} mm",
            'sheet_name': label_id,
            'symbol_count': len(matches),
            'width_mm': layout['width_mm'],
            'height_mm': layout['height_mm'],
            'layout': layout,
            'matched_symbols': matches,
            'texts': layout['texts']
        })

    _catalog_cache = {'products': products, 'available_symbols': available_symbols}
    _catalog_cache_time = now
    return _catalog_cache


def generate_label_image(label_id, dpi=600):
    """Generate label image with precise symbol placement from PDF."""
    catalog = load_catalog()
    product = None
    for p in catalog['products']:
        if p['sheet_name'] == label_id:
            product = p
            break

    if not product:
        raise ValueError(f"Label '{label_id}' not found")

    w_mm = product['width_mm']
    h_mm = product['height_mm']
    label_width = w_mm * MM_TO_INCH
    label_height = h_mm * MM_TO_INCH

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(label_width, label_height), dpi=dpi)
    ax.set_xlim(0, label_width)
    ax.set_ylim(0, label_height)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw label border
    border = 0.03
    rect = mpatches.FancyBboxPatch(
        (border, border), label_width - 2*border, label_height - 2*border,
        boxstyle="round,pad=0.02", linewidth=0.8,
        edgecolor='black', facecolor='none', linestyle='-', alpha=0.6, zorder=1
    )
    ax.add_patch(rect)

    # Place matched symbols at their extracted positions
    placed = 0
    for match in product['matched_symbols']:
        region = match['region']
        sym = match['symbol']

        # Convert relative positions to absolute (inches)
        x = region['rel_x'] * label_width
        y = region['rel_y'] * label_height
        w = region['rel_w'] * label_width
        h = region['rel_h'] * label_height

        try:
            img = Image.open(sym['path'])
            if img.mode == 'RGBA':
                white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(white_bg, img)
            img = img.convert('RGB')

            ax.imshow(img, extent=[x, x + w, y, y + h],
                      aspect='auto', interpolation='lanczos', zorder=3)
            placed += 1
        except Exception as e:
            log.warning(f"Could not place {sym['filename']}: {e}")

    # Draw editable text placeholders (empty boxes where text was detected)
    for txt in product.get('texts', []):
        tx = txt['rel_x'] * label_width
        ty = txt['rel_y'] * label_height
        # Draw an empty dashed box as placeholder
        text_box = mpatches.FancyBboxPatch(
            (tx, ty - 0.08), 0.8, 0.12,
            boxstyle="square,pad=0.01", linewidth=0.3,
            edgecolor='#999', facecolor='#f8f8f8', linestyle='--',
            alpha=0.5, zorder=2
        )
        ax.add_patch(text_box)
        # Label the placeholder with small gray text
        ax.text(tx + 0.02, ty - 0.02, '[editable]',
                fontsize=3, color='#aaa', va='center', zorder=4)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue(), placed, product


# === API ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = _APP_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>B300 Label Generator</h1><p>index.html not found</p>")


@app.get("/api/catalog")
async def get_catalog():
    """Return available labels parsed from PDF specs."""
    catalog = load_catalog()
    return {
        "products": [{
            'product_code': p['product_code'],
            'product_desc': p['product_desc'],
            'label_size': p['label_size'],
            'sheet_name': p['sheet_name'],
            'symbol_count': p['symbol_count']
        } for p in catalog['products']],
        "count": len(catalog['products'])
    }


@app.get("/api/generate/{label_id}")
async def generate(label_id: str, dpi: int = 600):
    """Generate a label with symbols placed per PDF spec."""
    try:
        img_bytes, placed_count, product = generate_label_image(label_id, dpi=dpi)
        img_b64 = base64.b64encode(img_bytes).decode()
        return {
            "image": f"data:image/png;base64,{img_b64}",
            "product_code": product['product_code'],
            "product_desc": product['product_desc'],
            "label_size": product['label_size'],
            "symbols_placed": placed_count,
            "convention": "pdf-extracted",
            "text_placeholders": len(product.get('texts', [])),
            "dimensions": {"width_mm": product['width_mm'], "height_mm": product['height_mm']}
        }
    except Exception as e:
        log.error(f"Error generating label {label_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{label_id}")
async def download(label_id: str, dpi: int = 600):
    """Download generated label as PNG."""
    try:
        img_bytes, _, product = generate_label_image(label_id, dpi=dpi)
        filename = f"B300_{product['product_code']}_{label_id}.png"
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    pdfs = find_pdf_labels()
    symbols = get_available_symbol_images()
    return {
        "status": "ok",
        "app": "B300 Label Generator v2",
        "pdf_labels_found": len(pdfs),
        "pdf_files": [p['filename'] for p in pdfs],
        "symbols_available": len(symbols),
        "symbol_files": [s['filename'] for s in symbols],
        "pdfplumber_available": HAS_PDFPLUMBER
    }
