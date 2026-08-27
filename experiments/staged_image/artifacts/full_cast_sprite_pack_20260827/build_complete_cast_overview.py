#!/usr/bin/env python3
"""Build the experimental full-cast contact-sheet index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
SHEETS = ROOT / "contact_sheets"
OUTPUT = ROOT / "complete_cast_pose_expression_overview.png"
METADATA = ROOT / "complete_cast_pose_expression_overview.json"

ENTRIES = [
    ("Aveline Morcant", "aveline_morcant_complete_sweep.png"),
    ("Castor Valebrand", "castor_valebrand_complete_sweep.png"),
    ("Halcyon of the Gilded March", "halcyon_of_the_gilded_march_complete_sweep.png"),
    ("Iselle the Guide", "iselle_the_guide_complete_sweep.png"),
    ("Liora Fen", "liora_fen_complete_sweep.png"),
    ("Renna Holt", "renna_holt_complete_sweep.png"),
    ("Seris Nightglass", "seris_nightglass_complete_sweep.png"),
    ("Soren Ironvow", "soren_ironvow_complete_sweep.png"),
    ("Veil the Unnumbered", "veil_the_unnumbered_complete_sweep.png"),
    ("Warden of the Eighth", "warden_of_the_eighth_complete_sweep.png"),
    ("Wren", "wren_complete_sweep.png"),
    ("Veiled one-star - masculine", "veiled_masculine_complete_sweep.png"),
    ("Veiled one-star - feminine", "veiled_feminine_complete_sweep.png"),
]

COLUMNS = 4
CELL_WIDTH = 780
IMAGE_HEIGHT = 528
TITLE_HEIGHT = 40
CELL_HEIGHT = IMAGE_HEIGHT + TITLE_HEIGHT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    records = []
    rows = (len(ENTRIES) + COLUMNS - 1) // COLUMNS
    canvas = Image.new(
        "RGB",
        (COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT),
        (12, 15, 20),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, filename) in enumerate(ENTRIES):
        path = SHEETS / filename
        image = Image.open(path).convert("RGB")
        if image.size != (1920, 1300):
            raise RuntimeError(f"unexpected contact-sheet dimensions for {path}: {image.size}")
        contained = ImageOps.contain(
            image, (CELL_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS
        )
        column = index % COLUMNS
        row = index // COLUMNS
        x = column * CELL_WIDTH + (CELL_WIDTH - contained.width) // 2
        y = row * CELL_HEIGHT
        canvas.paste(contained, (x, y))
        draw.text(
            (column * CELL_WIDTH + 12, y + IMAGE_HEIGHT + 12),
            title,
            fill=(255, 255, 255),
        )
        records.append(
            {
                "title": title,
                "path": str(path),
                "sha256": sha256(path),
                "dimensions": list(image.size),
            }
        )

    canvas.save(OUTPUT, optimize=True)
    metadata = {
        "status": "experimental_unlocked_review_index",
        "production_bound": False,
        "layout": {
            "columns": COLUMNS,
            "rows": rows,
            "cell_dimensions": [CELL_WIDTH, CELL_HEIGHT],
            "output_dimensions": list(canvas.size),
            "resampling": "Pillow LANCZOS for index only; source sheets unchanged",
        },
        "contact_sheets": records,
        "output": {
            "path": str(OUTPUT),
            "sha256": sha256(OUTPUT),
            "dimensions": list(canvas.size),
            "mode": "RGB",
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["output"], indent=2))


if __name__ == "__main__":
    main()
