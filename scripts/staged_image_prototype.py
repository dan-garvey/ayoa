#!/usr/bin/env python3
"""CLI entry point for the isolated staged-image experiment."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.staged_image.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
