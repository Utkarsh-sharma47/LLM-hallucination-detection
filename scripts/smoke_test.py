"""Run the offline smoke test.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glassbox.smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
