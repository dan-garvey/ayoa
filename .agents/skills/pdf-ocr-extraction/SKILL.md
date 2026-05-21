---
name: pdf-ocr-extraction
description: Use when extracting readable text from scanned or image-heavy PDF pages, especially private/proprietary source PDFs where generic text extraction is blank or incomplete and page-aligned OCR artifacts are needed under private_extractions/.
---

# PDF OCR Extraction

Use page-aligned rendering as the source of truth. Embedded image streams can be useful for debugging, but they may not map cleanly to PDF page numbers when a page mixes selectable text and decorative images.

## Workflow

1. Confirm the source PDF exists and choose an output folder under `private_extractions/`.
2. Render the target PDF page range to JPEGs with `pypdfium2` at about 300 DPI.
3. Run Surya OCR on the rendered page folder.
4. Postprocess Surya `results.json` into per-page raw text, column-ordered text, line JSON, a summary JSON, and combined Markdown files.
5. Validate coverage by checking that the rendered page count matches the requested range and reviewing zero-line pages. Zero-line pages are acceptable only for full-page artwork or intentionally blank pages.

## Manual Map Template Import

For adventure map/floorplan pages, OCR and image labels are only the first pass.
Manual import must produce a `TacticalMapTemplate` wrapper before any map can
seed combat.

Required map-template review steps:

1. Inspect the rendered map image and curated label file.
2. Inspect nearby OCR/source text for keyed areas on the map page and adjacent
   pages in the same section.
3. Record whether the map is `reference_only` or can seed `DndBattleMapState`.
4. If it can seed combat, draft one active floor/submap at a time. Do not put
   multiple floors into one battle map state.
5. Record wrapper metadata separately from the live battle map:
   - spawn anchors
   - keyed area links
   - stairs and vertical links
   - secret doors, hidden routes, traps, concealed areas, and reveal conditions
   - review status and caveats
6. Compile only valid `DndBattleMapState` fields into the live seed:
   `present`, `map_name`, `width`, `height`, `square_size_ft`, `tokens`,
   rectangular `terrain`, `areas`, and `notes`.

Nearby-text cross-check is mandatory before marking topology reviewed. Use it
to catch:

- scale and orientation mistakes
- stairs, shafts, drops, bridges, and inter-map vertical links
- secret doors or concealed passages mentioned in keyed text but not obvious on
  the image
- walls, windows, parapets, pits, water, rubble, furniture, and terrain that
  affect movement, cover, or line of sight
- room labels that OCR misread or omitted
- overland maps that should remain reference-only
- spawn anchors that land inside walls, hazards, furniture, or secret-only areas

Be conservative. If exact walls, secrets, doors, or vertical links are not
confirmed by both image and nearby text, keep `review_status="draft"` and write
the uncertainty into review notes.

## Commands

Use `.venv/bin/python` and `.venv/bin/surya_ocr` from the repo root.

Render pages:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pypdfium2 as pdfium

pdf_path = Path("stories/curse_of_strahd.pdf")
out = Path("private_extractions/<slug>/pdf_rendered_pages")
start_page = 31
end_page = 60
out.mkdir(parents=True, exist_ok=True)
for p in out.glob("page_*.jpg"):
    p.unlink()

doc = pdfium.PdfDocument(str(pdf_path))
scale = 300 / 72
for page_number in range(start_page, end_page + 1):
    page = doc[page_number - 1]
    bitmap = page.render(scale=scale, rotation=0)
    image = bitmap.to_pil()
    image.save(out / f"page_{page_number:03d}.jpg", quality=90, optimize=True)
    page.close()
PY
```

OCR pages:

```bash
rm -rf private_extractions/<slug>/surya_pdf_pages_raw
.venv/bin/surya_ocr \
  private_extractions/<slug>/pdf_rendered_pages \
  --output_dir private_extractions/<slug>/surya_pdf_pages_raw
```

Postprocess the raw JSON with these reading-order rules:

- Strip simple Surya inline tags such as `<b>` and `<i>` and collapse whitespace.
- Preserve a raw file sorted by `(y0, x0)` for comparison.
- Infer one to three columns from text-line midpoint clusters, excluding very wide headings, low-confidence one-character noise, and footer lines.
- Treat wide top lines that cross the page midpoint as page titles before body columns.
- Assign body lines to the nearest inferred column and sort each column by `(y0, x0)`.
- Put footer lines after body text.
- Keep line JSON with bboxes and confidence values so bad ordering can be debugged later.

Write:

- `surya_pdf_pages_ocr_text/page_###_surya_raw.txt`
- `surya_pdf_pages_ocr_text/page_###_surya_columns.txt`
- `surya_pdf_pages_ocr_text/page_###_surya_lines.json`
- `surya_pdf_pages_ocr_text/surya_pdf_pages_summary.json`
- `surya_pdf_pages_raw_order.md`
- `surya_pdf_pages_column_order.md`

## Validation

Run these checks before reporting completion:

```bash
find private_extractions/<slug>/pdf_rendered_pages -maxdepth 1 -type f -name 'page_*.jpg' | sort | wc -l
.venv/bin/python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("private_extractions/<slug>/surya_pdf_pages_ocr_text/surya_pdf_pages_summary.json").read_text())
print("entries", len(summary))
print("zero-line pages", [r["page"] for r in summary if r["column_line_count"] == 0])
print("total lines", sum(r["column_line_count"] for r in summary))
PY
```

Spot-check the first, last, and any table-heavy or art-heavy page. Report coverage and caveats, not long source text.

Do not stage `private_extractions/`, local source PDFs, `.venv/`, or OCR model caches unless the user explicitly requests it.
