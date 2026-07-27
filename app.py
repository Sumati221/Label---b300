"""
B300 Label Generator App
Generates B300 product labels from relation files with exact symbol placement
and DXF die-cut outlines. Adapted from the CAPNOSTAT label-agent.
"""
import os
import re
import io
import json
import time
import base64
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
import ezdxf
from ezdxf import bbox as ebbox

# cairosvg is optional - requires libcairo2 system library
try:
    import cairosvg
    HAS_CAIROSVG = True
except (ImportError, OSError):
    HAS_CAIROSVG = False

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("b300-label-generator")

app = FastAPI(title="B300 Label Generator", docs_url="/docs")

# === CONFIGURATION ===
_APP_DIR = Path(__file__).parent
RELATION_FILE = str(_APP_DIR / "data" / "b300_labels_relation.xlsx")
SYMBOLS_LIBRARY = str(_APP_DIR / "data" / "symbols")
DXF_FOLDER = str(_APP_DIR / "data" / "dxf")
DPI = 600
SVG_RENDER_DPI = 1200
SKIP_SHEETS = {'Summary', 'Symbols Summary'}
SPECIFIC_SHAPE_KEYWORDS = ['flag', 'pouch', 'bag', 'box']
SYMBOL_IMAGE_EXTENSIONS = ['.jpg', '.pcx', '.png', '.svg']

# B300-specific text symbols (update as needed)
TEXT_SYMBOLS = {}

# === CACHED STATE ===
_catalog_cache = None
_catalog_cache_time = 0
CACHE_TTL = 300  # 5 minutes


# === HELPER FUNCTIONS ===

def find_symbol_image(symbols_library, sym_code):
    # Try exact match first
    for ext in SYMBOL_IMAGE_EXTENSIONS:
        if ext == '.svg' and not HAS_CAIROSVG:
            continue
        path = os.path.join(symbols_library, f"{sym_code}{ext}")
        if os.path.exists(path):
            return path, ext
    # Try partial match (e.g., 100012_600dpi.png, 100183-600dpi.png)
    if os.path.exists(symbols_library):
        for f in os.listdir(symbols_library):
            if f.startswith(sym_code) and any(f.lower().endswith(ext) for ext in SYMBOL_IMAGE_EXTENSIONS):
                ext = os.path.splitext(f)[1].lower()
                if ext == '.svg' and not HAS_CAIROSVG:
                    continue
                return os.path.join(symbols_library, f), ext
    return None, None


def load_symbol_image(img_path, img_format, dpi=300):
    if img_format == '.svg':
        if not HAS_CAIROSVG:
            log.warning(f"Skipping SVG (cairosvg not available): {img_path}")
            return None
        png_data = cairosvg.svg2png(url=img_path, dpi=dpi)
        img = Image.open(io.BytesIO(png_data))
        if img.mode == 'RGBA':
            white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white_bg, img)
        img = img.convert('RGB')
    else:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white_bg, img)
        img = img.convert('RGB')
    return img


def scan_available_dxf_files():
    dxf_files = {}
    if os.path.exists(DXF_FOLDER):
        for f in os.listdir(DXF_FOLDER):
            if f.lower().endswith('.dxf'):
                key = f.lower().replace('.dxf', '').strip()
                dxf_files[key] = os.path.join(DXF_FOLDER, f)
    return dxf_files


def get_dxf_for_label(sheet_name, label_size_str):
    available = scan_available_dxf_files()
    sheet_lower = sheet_name.lower()
    for keyword in SPECIFIC_SHAPE_KEYWORDS:
        if keyword in sheet_lower:
            for key, path in available.items():
                if keyword in key:
                    return path
            return None
    for key, path in available.items():
        if 'carton' in key:
            return path
    return None


def draw_dxf_outline(ax, dxf_path, label_width, label_height):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    entities = list(msp)
    bb = ebbox.extents(entities)
    if not bb.has_data:
        return
    dxf_w = bb.extmax[0] - bb.extmin[0]
    dxf_h = bb.extmax[1] - bb.extmin[1]
    margin = 0.02
    scale_x = (label_width - 2 * margin) / dxf_w
    scale_y = (label_height - 2 * margin) / dxf_h
    offset_x = margin - bb.extmin[0] * scale_x
    offset_y = margin - bb.extmin[1] * scale_y
    for entity in entities:
        if entity.dxftype() == 'SPLINE':
            pts = list(entity.control_points)
            if len(pts) >= 2:
                xs = [p[0] * scale_x + offset_x for p in pts]
                ys = [p[1] * scale_y + offset_y for p in pts]
                ax.plot(xs, ys, 'k-', linewidth=1.0, alpha=0.7, zorder=1)
        elif entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            xs = [start[0] * scale_x + offset_x, end[0] * scale_x + offset_x]
            ys = [start[1] * scale_y + offset_y, end[1] * scale_y + offset_y]
            ax.plot(xs, ys, 'k-', linewidth=1.0, alpha=0.7, zorder=1)
        elif entity.dxftype() == 'LWPOLYLINE':
            pts = list(entity.get_points(format='xy'))
            if len(pts) >= 2:
                xs = [p[0] * scale_x + offset_x for p in pts]
                ys = [p[1] * scale_y + offset_y for p in pts]
                if entity.closed:
                    xs.append(xs[0])
                    ys.append(ys[0])
                ax.plot(xs, ys, 'k-', linewidth=1.0, alpha=0.7, zorder=1)
        elif entity.dxftype() == 'POLYLINE':
            pts = [(v.dxf.location[0], v.dxf.location[1]) for v in entity.vertices]
            if len(pts) >= 2:
                xs = [p[0] * scale_x + offset_x for p in pts]
                ys = [p[1] * scale_y + offset_y for p in pts]
                ax.plot(xs, ys, 'k-', linewidth=1.0, alpha=0.7, zorder=1)
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_ang = entity.dxf.start_angle
            end_ang = entity.dxf.end_angle
            if end_ang < start_ang:
                end_ang += 360
            angles = np.linspace(np.radians(start_ang), np.radians(end_ang), 32)
            xs = [(center[0] + radius * np.cos(a)) * scale_x + offset_x for a in angles]
            ys = [(center[1] + radius * np.sin(a)) * scale_y + offset_y for a in angles]
            ax.plot(xs, ys, 'k-', linewidth=1.0, alpha=0.7, zorder=1)


def draw_fallback_outline(ax, label_width, label_height, sheet_name):
    margin = 0.05
    if 'Flag' in sheet_name:
        notch_depth = 0.15
        notch_y = label_height / 2
        notch_half = label_height * 0.3
        path_x = [margin, label_width - margin, label_width - margin, margin,
                  margin, margin + notch_depth, margin, margin]
        path_y = [margin, margin, label_height - margin, label_height - margin,
                  notch_y + notch_half, notch_y, notch_y - notch_half, margin]
        ax.plot(path_x, path_y, 'k--', linewidth=0.6, alpha=0.5, zorder=1)
    else:
        rect = mpatches.FancyBboxPatch(
            (margin, margin), label_width - 2 * margin, label_height - 2 * margin,
            boxstyle="round,pad=0.05", linewidth=0.6,
            edgecolor='black', facecolor='none', linestyle='--', alpha=0.5, zorder=1
        )
        ax.add_patch(rect)


def parse_label_size(size_str):
    match = re.search(r'(\d+\.?\d*)\s*X\s*(\d+\.?\d*)', str(size_str), re.IGNORECASE)
    if match:
        dim1 = float(match.group(1))
        dim2 = float(match.group(2))
        width = max(dim1, dim2)
        height = min(dim1, dim2)
        return height, width
    return 4.0, 8.0


def parse_size(size_str):
    w_match = re.search(r'Width:\s*([\d.]+)', str(size_str))
    h_match = re.search(r'Height:\s*([\d.]+)', str(size_str))
    if w_match and h_match:
        return float(w_match.group(1)), float(h_match.group(1))
    return None, None


def parse_font_size(size_str):
    match = re.search(r'Size:\s*([\d.]+)\s*pt', str(size_str))
    if match:
        return float(match.group(1))
    return None


def parse_rotation(size_str):
    match = re.search(r'Rotation\s+(-?\d+)\s*degrees?\s*(clockwise|counterclockwise|ccw|cw)?',
                      str(size_str), re.IGNORECASE)
    if match:
        angle = float(match.group(1))
        direction = (match.group(2) or '').lower()
        if direction in ('clockwise', 'cw'):
            return angle
        elif direction in ('counterclockwise', 'ccw'):
            return -angle
        return angle
    return 0.0


def parse_position(pos_str):
    x_match = re.search(r'X:\s*([\d.]+)', str(pos_str))
    y_match = re.search(r'Y:\s*([\d.]+)', str(pos_str))
    if x_match and y_match:
        return float(x_match.group(1)), float(y_match.group(1))
    return None, None


def detect_coordinate_convention(symbol_data, label_width, label_height):
    overflows_bl = 0
    overflows_c = 0
    for _, row in symbol_data.iterrows():
        w, h, x, y = row['width'], row['height'], row['x'], row['y']
        if pd.isna(w) or pd.isna(h) or pd.isna(x) or pd.isna(y):
            continue
        if x + w > label_width + 0.1 or y + h > label_height + 0.1:
            overflows_bl += 1
        if x + w / 2 > label_width + 0.1 or y + h / 2 > label_height + 0.1:
            overflows_c += 1
    return 'centered' if overflows_bl > overflows_c else 'bottom-left'


def load_product_catalog():
    """Load the relation file and build the product catalog."""
    global _catalog_cache, _catalog_cache_time
    now = time.time()
    if _catalog_cache and (now - _catalog_cache_time) < CACHE_TTL:
        return _catalog_cache

    catalog = []
    if not os.path.exists(RELATION_FILE):
        log.error(f"Relation file not found: {RELATION_FILE}")
        return catalog

    try:
        xl = pd.ExcelFile(RELATION_FILE)
        product_sheets = [s for s in xl.sheet_names if s not in SKIP_SHEETS]

        for sheet_name in product_sheets:
            df_sheet = pd.read_excel(RELATION_FILE, sheet_name=sheet_name, header=None)
            product_code = str(df_sheet.iloc[1, 1]).strip() if pd.notna(df_sheet.iloc[1, 1]) else 'Unknown'
            product_desc = str(df_sheet.iloc[1, 2]).strip() if pd.notna(df_sheet.iloc[1, 2]) else ''
            label_size = str(df_sheet.iloc[2, 2]).strip() if pd.notna(df_sheet.iloc[2, 2]) else ''
            sym_count = len(df_sheet.iloc[5:].dropna(subset=[2]))

            catalog.append({
                'product_code': product_code,
                'product_desc': product_desc,
                'label_size': label_size,
                'sheet_name': sheet_name,
                'symbol_count': sym_count
            })
    except Exception as e:
        log.error(f"Error loading catalog: {e}")

    _catalog_cache = catalog
    _catalog_cache_time = now
    return catalog


def load_symbol_data(sheet_name):
    """Load and parse symbol placement data from a specific sheet."""
    df_raw = pd.read_excel(RELATION_FILE, sheet_name=sheet_name, header=None)
    product_code = str(df_raw.iloc[1, 1]).strip() if pd.notna(df_raw.iloc[1, 1]) else 'Unknown'
    product_desc = str(df_raw.iloc[1, 2]).strip() if pd.notna(df_raw.iloc[1, 2]) else ''
    label_size_str = str(df_raw.iloc[2, 2]).strip() if pd.notna(df_raw.iloc[2, 2]) else ''
    label_height, label_width = parse_label_size(label_size_str)

    symbol_data = df_raw.iloc[5:, :5].reset_index(drop=True)
    symbol_data.columns = ['col0', 'col1', 'SYMBOL_CODE', 'SIZE', 'POSITION']
    symbol_data = symbol_data[['col1', 'SYMBOL_CODE', 'SIZE', 'POSITION']].dropna(subset=['SYMBOL_CODE'])
    symbol_data['width'], symbol_data['height'] = zip(*symbol_data['SIZE'].apply(parse_size))
    symbol_data['x'], symbol_data['y'] = zip(*symbol_data['POSITION'].apply(parse_position))
    symbol_data['font_pt'] = symbol_data['SIZE'].apply(parse_font_size)
    symbol_data['rotation'] = symbol_data['SIZE'].apply(parse_rotation)
    symbol_data['text_content'] = symbol_data['col1'].apply(
        lambda v: str(v).strip() if pd.notna(v) else None
    )

    convention = detect_coordinate_convention(symbol_data, label_width, label_height)
    return {
        'product_code': product_code,
        'product_desc': product_desc,
        'label_size_str': label_size_str,
        'label_width': label_width,
        'label_height': label_height,
        'symbol_data': symbol_data,
        'sheet_name': sheet_name,
        'convention': convention
    }


def place_symbols(ax, symbol_data, symbols_library, LABEL_WIDTH, LABEL_HEIGHT, convention='bottom-left'):
    """Place all symbols using positions and sizes from the relation file."""
    placed_count = 0
    sorted_data = symbol_data.copy()
    sorted_data['_area'] = sorted_data['width'].fillna(0) * sorted_data['height'].fillna(0)
    sorted_data = sorted_data.sort_values('_area', ascending=False)

    for _, row in sorted_data.iterrows():
        sym_code = row['SYMBOL_CODE']
        x_pos = row['x']
        y_pos = row['y']
        w = row['width']
        h = row['height']
        font_pt = row.get('font_pt', None)
        text_content = row.get('text_content', None)
        rotation = row.get('rotation', 0.0)
        if pd.isna(rotation):
            rotation = 0.0

        if pd.isna(x_pos) or pd.isna(y_pos):
            continue

        if w is not None and not pd.isna(w) and h is not None and not pd.isna(h):
            if convention == 'bottom-left':
                left, right = x_pos, x_pos + w
                bottom, top = y_pos, y_pos + h
            else:
                left, right = x_pos - w / 2, x_pos + w / 2
                bottom, top = y_pos - h / 2, y_pos + h / 2
        else:
            left, right, bottom, top = None, None, None, None

        has_font = font_pt is not None and not pd.isna(font_pt)
        has_text = text_content is not None and text_content != 'nan' and str(text_content).strip() != ''

        # CASE 1: Font-based text
        if has_font and has_text:
            ax.text(x_pos, y_pos, text_content,
                    fontsize=font_pt, fontfamily='DejaVu Sans',
                    fontweight='bold', color='black',
                    ha='left', va='bottom', rotation=rotation, zorder=5)
            placed_count += 1
            continue

        # CASE 2: Text-only symbol from TEXT_SYMBOLS dict
        if sym_code in TEXT_SYMBOLS:
            ts = TEXT_SYMBOLS[sym_code]
            ax.text(x_pos, y_pos, ts['text'],
                    fontsize=font_pt or 8, fontfamily=ts.get('font', 'DejaVu Sans'),
                    fontweight=ts.get('weight', 'normal'), color=ts.get('color', 'black'),
                    ha='left', va='bottom', rotation=rotation, zorder=5)
            placed_count += 1
            continue

        # CASE 3: Image file
        # Try the code as-is, then strip 'LS-' prefix for numeric-only filenames
        img_path, img_format = find_symbol_image(symbols_library, sym_code)
        if not img_path and sym_code.startswith('LS-'):
            img_path, img_format = find_symbol_image(symbols_library, sym_code[3:])
        if img_path:
            img = load_symbol_image(img_path, img_format,
                                    dpi=SVG_RENDER_DPI if img_format == '.svg' else DPI)
            if img is None:
                continue
            if left is not None:
                ax.imshow(img, extent=[left, right, bottom, top],
                          aspect='auto', interpolation='lanczos', zorder=3)
            else:
                # Fallback: small default size
                ax.imshow(img, extent=[x_pos, x_pos + 0.5, y_pos, y_pos + 0.5],
                          aspect='auto', interpolation='lanczos', zorder=3)
            placed_count += 1
        else:
            # Symbol not found - draw placeholder box
            if left is not None:
                rect = mpatches.Rectangle((left, bottom), w, h,
                                          linewidth=0.5, edgecolor='red',
                                          facecolor='#fff0f0', alpha=0.5, zorder=2)
                ax.add_patch(rect)
                ax.text((left + right) / 2, (bottom + top) / 2, sym_code,
                        fontsize=4, ha='center', va='center', color='red', zorder=4)
            placed_count += 1

    return placed_count


def generate_label_image(sheet_name, dpi=600):
    """Generate a label image for the given sheet and return as PNG bytes."""
    data = load_symbol_data(sheet_name)
    label_width = data['label_width']
    label_height = data['label_height']
    symbol_data = data['symbol_data']
    convention = data['convention']

    fig, ax = plt.subplots(1, 1, figsize=(label_width, label_height), dpi=dpi)
    ax.set_xlim(0, label_width)
    ax.set_ylim(0, label_height)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw die-cut outline
    dxf_path = get_dxf_for_label(sheet_name, data['label_size_str'])
    if dxf_path:
        draw_dxf_outline(ax, dxf_path, label_width, label_height)
    else:
        draw_fallback_outline(ax, label_width, label_height, sheet_name)

    # Place symbols
    placed = place_symbols(ax, symbol_data, SYMBOLS_LIBRARY, label_width, label_height, convention)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), placed, data


# === API ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = _APP_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>B300 Label Generator</h1><p>index.html not found</p>")


@app.get("/api/catalog")
async def get_catalog():
    """Return the product catalog from the relation file."""
    catalog = load_product_catalog()
    return {"products": catalog, "count": len(catalog)}


@app.get("/api/generate/{sheet_name}")
async def generate(sheet_name: str, dpi: int = 600):
    """Generate a label for the specified product sheet."""
    try:
        img_bytes, placed_count, data = generate_label_image(sheet_name, dpi=dpi)
        img_b64 = base64.b64encode(img_bytes).decode()
        return {
            "image": f"data:image/png;base64,{img_b64}",
            "product_code": data['product_code'],
            "product_desc": data['product_desc'],
            "label_size": data['label_size_str'],
            "symbols_placed": placed_count,
            "convention": data['convention'],
            "dimensions": {"width": data['label_width'], "height": data['label_height']}
        }
    except Exception as e:
        log.error(f"Error generating label for {sheet_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{sheet_name}")
async def download(sheet_name: str, dpi: int = 600):
    """Download a label as a PNG file."""
    try:
        img_bytes, _, data = generate_label_image(sheet_name, dpi=dpi)
        filename = f"B300_{data['product_code']}_{sheet_name}.png"
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "app": "B300 Label Generator",
        "relation_file_exists": os.path.exists(RELATION_FILE),
        "symbols_dir_exists": os.path.exists(SYMBOLS_LIBRARY),
        "dxf_dir_exists": os.path.exists(DXF_FOLDER)
    }
