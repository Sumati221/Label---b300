"""
B300 Label Generator App
Reads the CDLM Excel for symbol-to-product mapping and PDF specs for label
dimensions. Auto-layouts available symbol PNGs on the label canvas.
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("b300-label-generator")

app = FastAPI(title="B300 Label Generator", docs_url="/docs")

# === CONFIGURATION ===
_APP_DIR = Path(__file__).parent
DATA_DIR = _APP_DIR / "data"
SYMBOLS_DIR = DATA_DIR / "symbols"
DPI = 600
MM_TO_INCH = 1 / 25.4
SYMBOL_IMAGE_EXTENSIONS = ['.png', '.jpg', '.svg']

# CDLM sheets to skip
SKIP_SHEETS = {'Approval Sheet', 'Document History', 'Summary'}

# === CACHED STATE ===
_catalog_cache = None
_catalog_cache_time = 0
CACHE_TTL = 300


# === HELPER FUNCTIONS ===

def find_cdlm_file():
    """Find the CDLM Excel file in the symbols directory."""
    for f in os.listdir(str(SYMBOLS_DIR)):
        if f.endswith('.xlsx') and 'CDLM' in f.upper() and not f.startswith('~'):
            return str(SYMBOLS_DIR / f)
    # Fallback: any xlsx that isn't a temp file
    for f in os.listdir(str(SYMBOLS_DIR)):
        if f.endswith('.xlsx') and not f.startswith('~') and not f.startswith('.'):
            return str(SYMBOLS_DIR / f)
    return None


def find_pdf_specs():
    """Find PDF spec files and extract label dimensions."""
    specs = []
    for f in os.listdir(str(SYMBOLS_DIR)):
        if f.lower().endswith('.pdf') and 'MART' in f.upper():
            pdf_path = str(SYMBOLS_DIR / f)
            label_size = extract_pdf_dimensions(pdf_path)
            title = extract_pdf_title(pdf_path)
            specs.append({
                'filename': f,
                'path': pdf_path,
                'title': title,
                'label_size_mm': label_size
            })
    return specs


def extract_pdf_dimensions(pdf_path):
    """Extract label dimensions (mm) from a PDF spec drawing."""
    if not HAS_PDFPLUMBER:
        return (50, 85)  # Default fallback
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ''
                # Look for dimension patterns like "50mm x 85mm" or "(50mm x 85mm)"
                match = re.search(r'\(?(\d+)\s*mm\s*[xX\u00d7]\s*(\d+)\s*mm\)?', text)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
    except Exception as e:
        log.warning(f"Could not parse PDF {pdf_path}: {e}")
    return (50, 85)  # Default


def extract_pdf_title(pdf_path):
    """Extract title/description from PDF spec."""
    if not HAS_PDFPLUMBER:
        return os.path.basename(pdf_path).replace('.pdf', '')
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ''
                # Look for TITLE section
                match = re.search(r'TITLE\s*\n(.+?)\n', text)
                if match:
                    return match.group(1).strip()
                # Look for drawing number
                match = re.search(r'DRAWING NUMBER.*?\n(\S+)', text)
                if match:
                    return match.group(1).strip()
    except Exception:
        pass
    return os.path.basename(pdf_path).replace('.pdf', '')


def find_symbol_image(sym_code):
    """Find a symbol image by code. Tries exact then partial match."""
    symbols_dir = str(SYMBOLS_DIR)
    if not os.path.exists(symbols_dir):
        return None
    codes_to_try = [sym_code]
    if sym_code.startswith('LS-'):
        codes_to_try.append(sym_code[3:])
    for code in codes_to_try:
        for ext in SYMBOL_IMAGE_EXTENSIONS:
            path = os.path.join(symbols_dir, f"{code}{ext}")
            if os.path.exists(path):
                return path
        # Partial match (e.g., 100012_600dpi.png)
        for f in os.listdir(symbols_dir):
            if f.startswith(code) and any(f.lower().endswith(e) for e in SYMBOL_IMAGE_EXTENSIONS):
                return os.path.join(symbols_dir, f)
    return None


def get_available_symbols():
    """Scan the symbols directory for available PNG/JPG/SVG images."""
    symbols = []
    symbols_dir = str(SYMBOLS_DIR)
    if not os.path.exists(symbols_dir):
        return symbols
    for f in sorted(os.listdir(symbols_dir)):
        if any(f.lower().endswith(ext) for ext in SYMBOL_IMAGE_EXTENSIONS):
            code = re.match(r'(\d+)', f)
            code_str = code.group(1) if code else f.split('.')[0]
            img = Image.open(os.path.join(symbols_dir, f))
            symbols.append({
                'code': code_str,
                'filename': f,
                'path': os.path.join(symbols_dir, f),
                'width_px': img.size[0],
                'height_px': img.size[1]
            })
    return symbols


def parse_cdlm_products():
    """Parse the CDLM Excel to extract products and their symbol requirements."""
    cdlm_path = find_cdlm_file()
    if not cdlm_path:
        return []

    products = []
    try:
        xl = pd.ExcelFile(cdlm_path)
        product_sheets = [s for s in xl.sheet_names if s not in SKIP_SHEETS
                          and 'PL ' in s]  # Product Label sheets start with 'PL '

        for sheet_name in product_sheets:
            df = pd.read_excel(cdlm_path, sheet_name=sheet_name, header=None)
            # Extract symbols from the sheet
            symbols_needed = []
            for idx in range(2, len(df)):
                sym_id = df.iloc[idx, 4] if len(df.columns) > 4 else None
                sym_name = df.iloc[idx, 6] if len(df.columns) > 6 else None
                if pd.notna(sym_id) and str(sym_id).startswith('LS-'):
                    # Check if we have an image for this symbol
                    img_path = find_symbol_image(str(sym_id))
                    symbols_needed.append({
                        'id': str(sym_id),
                        'name': str(sym_name) if pd.notna(sym_name) else str(sym_id),
                        'has_image': img_path is not None,
                        'image_path': img_path
                    })

            products.append({
                'sheet_name': sheet_name,
                'product_name': sheet_name.replace('PL ', ''),
                'symbol_count': len(symbols_needed),
                'symbols': symbols_needed,
                'available_symbols': len([s for s in symbols_needed if s['has_image']])
            })
    except Exception as e:
        log.error(f"Error parsing CDLM: {e}")

    return products


def load_catalog():
    """Build the product catalog from CDLM + PDF specs + available symbols."""
    global _catalog_cache, _catalog_cache_time
    now = time.time()
    if _catalog_cache and (now - _catalog_cache_time) < CACHE_TTL:
        return _catalog_cache

    # Get PDF specs for label dimensions
    pdf_specs = find_pdf_specs()

    # Get CDLM products
    cdlm_products = parse_cdlm_products()

    # Get available symbol images
    available_symbols = get_available_symbols()

    catalog = {
        'pdf_specs': pdf_specs,
        'cdlm_products': cdlm_products,
        'available_symbols': available_symbols,
        'labels': []
    }

    # Build label entries from PDF specs
    for spec in pdf_specs:
        w_mm, h_mm = spec['label_size_mm']
        catalog['labels'].append({
            'id': spec['filename'].replace('.pdf', ''),
            'title': spec['title'],
            'filename': spec['filename'],
            'width_mm': w_mm,
            'height_mm': h_mm,
            'symbols': [s for s in available_symbols]  # All available symbols
        })

    # If no PDFs, create a default label from available symbols
    if not catalog['labels'] and available_symbols:
        catalog['labels'].append({
            'id': 'default-label',
            'title': 'B300 Label (Default 50x85mm)',
            'filename': None,
            'width_mm': 50,
            'height_mm': 85,
            'symbols': available_symbols
        })

    _catalog_cache = catalog
    _catalog_cache_time = now
    return catalog


def auto_layout_symbols(symbols, label_width_in, label_height_in):
    """Automatically position symbols on a label using grid layout."""
    placements = []
    margin = 0.1  # inches
    usable_w = label_width_in - 2 * margin
    usable_h = label_height_in - 2 * margin

    if not symbols:
        return placements

    # Calculate grid
    n = len(symbols)
    cols = min(n, max(1, int(np.ceil(np.sqrt(n * (usable_w / usable_h))))))
    rows = int(np.ceil(n / cols))

    cell_w = usable_w / cols
    cell_h = usable_h / rows

    for idx, sym in enumerate(symbols):
        row = idx // cols
        col = idx % cols

        # Scale symbol to fit cell (with padding)
        padding = 0.05
        max_w = cell_w - 2 * padding
        max_h = cell_h - 2 * padding

        sym_w_in = sym['width_px'] / DPI
        sym_h_in = sym['height_px'] / DPI

        scale = min(max_w / sym_w_in, max_h / sym_h_in, 1.0)
        w = sym_w_in * scale
        h = sym_h_in * scale

        # Center in cell
        x = margin + col * cell_w + (cell_w - w) / 2
        y = margin + (rows - 1 - row) * cell_h + (cell_h - h) / 2

        placements.append({
            'symbol': sym,
            'x': x,
            'y': y,
            'width': w,
            'height': h
        })

    return placements


def generate_label_image(label_id, dpi=600):
    """Generate a label image and return as PNG bytes."""
    catalog = load_catalog()
    label = None
    for l in catalog['labels']:
        if l['id'] == label_id:
            label = l
            break

    if not label:
        raise ValueError(f"Label '{label_id}' not found")

    w_mm = label['width_mm']
    h_mm = label['height_mm']
    label_width = w_mm * MM_TO_INCH
    label_height = h_mm * MM_TO_INCH

    # Auto-layout symbols
    placements = auto_layout_symbols(label['symbols'], label_width, label_height)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(label_width, label_height), dpi=dpi)
    ax.set_xlim(0, label_width)
    ax.set_ylim(0, label_height)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw border
    border_margin = 0.04
    rect = mpatches.FancyBboxPatch(
        (border_margin, border_margin),
        label_width - 2 * border_margin,
        label_height - 2 * border_margin,
        boxstyle="round,pad=0.03", linewidth=0.8,
        edgecolor='black', facecolor='none', linestyle='--', alpha=0.4, zorder=1
    )
    ax.add_patch(rect)

    # Place symbols
    placed = 0
    for p in placements:
        img_path = p['symbol']['path']
        try:
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(white_bg, img)
            img = img.convert('RGB')

            left, right = p['x'], p['x'] + p['width']
            bottom, top = p['y'], p['y'] + p['height']
            ax.imshow(img, extent=[left, right, bottom, top],
                      aspect='auto', interpolation='lanczos', zorder=3)
            placed += 1
        except Exception as e:
            log.warning(f"Could not place symbol {p['symbol']['code']}: {e}")

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue(), placed, label


# === API ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = _APP_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>B300 Label Generator</h1><p>index.html not found</p>")


@app.get("/api/catalog")
async def get_catalog():
    """Return available labels, symbols, and PDF specs."""
    catalog = load_catalog()
    return {
        "labels": [{
            'id': l['id'],
            'title': l['title'],
            'width_mm': l['width_mm'],
            'height_mm': l['height_mm'],
            'symbol_count': len(l['symbols'])
        } for l in catalog['labels']],
        "available_symbols": [{
            'code': s['code'],
            'filename': s['filename'],
            'width_px': s['width_px'],
            'height_px': s['height_px']
        } for s in catalog['available_symbols']],
        "pdf_specs": [{
            'filename': p['filename'],
            'title': p['title'],
            'label_size_mm': p['label_size_mm']
        } for p in catalog.get('pdf_specs', [])],
        "cdlm_products": [{
            'product_name': p['product_name'],
            'symbol_count': p['symbol_count'],
            'available_symbols': p['available_symbols']
        } for p in catalog.get('cdlm_products', [])]
    }


@app.get("/api/generate/{label_id}")
async def generate(label_id: str, dpi: int = 600):
    """Generate a label for the specified label spec."""
    try:
        img_bytes, placed_count, label = generate_label_image(label_id, dpi=dpi)
        img_b64 = base64.b64encode(img_bytes).decode()
        return {
            "image": f"data:image/png;base64,{img_b64}",
            "product_code": label['id'].split('-M-')[0] if '-M-' in label['id'] else label['id'],
            "product_desc": label['title'],
            "label_size": f"{label['width_mm']} X {label['height_mm']} mm",
            "symbols_placed": placed_count,
            "convention": "auto-layout",
            "dimensions": {"width_mm": label['width_mm'], "height_mm": label['height_mm']}
        }
    except Exception as e:
        log.error(f"Error generating label {label_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{label_id}")
async def download(label_id: str, dpi: int = 600):
    """Download a label as PNG."""
    try:
        img_bytes, _, label = generate_label_image(label_id, dpi=dpi)
        filename = f"B300_{label['id']}.png"
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    cdlm = find_cdlm_file()
    symbols = get_available_symbols()
    return {
        "status": "ok",
        "app": "B300 Label Generator",
        "cdlm_found": cdlm is not None,
        "cdlm_file": os.path.basename(cdlm) if cdlm else None,
        "symbols_available": len(symbols),
        "pdfplumber_available": HAS_PDFPLUMBER
    }
