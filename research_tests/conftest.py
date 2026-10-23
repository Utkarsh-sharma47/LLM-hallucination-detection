"""Load the research package in src/glassbox for this suite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != ROOT.resolve()]
sys.path.insert(0, str(SRC))

for name in list(sys.modules):
    if name != "glassbox" and not name.startswith("glassbox."):
        continue
    module_file = getattr(sys.modules[name], "__file__", "") or ""
    if str(SRC) not in module_file:
        del sys.modules[name]
