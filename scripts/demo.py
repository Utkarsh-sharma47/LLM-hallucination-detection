"""Run the terminal demo.

Usage:
    python scripts/demo.py
    python scripts/demo.py --once --question "What is the capital of France?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glassbox.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
