# B300 Label Generator

Generates B300 product labels with regulatory symbols from relation files.

## Structure

```
.
├── app.py                  # FastAPI app (label generation engine)
├── app.yaml                # Databricks App config
├── requirements.txt        # Python dependencies
├── index.html              # Web UI
└── data/
    ├── b300_labels_relation.xlsx   # Master layout file
    ├── symbols/                     # Symbol PNG/SVG images
    │   ├── 100012_600dpi.png       # Philips Wordmark
    │   ├── 100025_600dpi.png       # Regulatory mark
    │   └── 100183-600dpi.png       # Info bar
    └── dxf/                         # Die-cut outlines (optional)
```

## Quick Start

1. Add symbol images to `data/symbols/`
2. Update `data/b300_labels_relation.xlsx` with label layouts
3. Deploy: `databricks apps deploy b300-label-agent --source-code-path <path>`

## API Endpoints

- `GET /` — Web UI
- `GET /api/catalog` — List available products
- `GET /api/generate/{sheet_name}?dpi=600` — Generate label (base64 PNG)
- `GET /api/download/{sheet_name}?dpi=600` — Download label PNG
- `GET /api/health` — Health check
