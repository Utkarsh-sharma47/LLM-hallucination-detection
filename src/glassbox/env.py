"""Small ``.env`` reader.

The smoke test does not call this module. Experiment scripts may call it later.
Values are never logged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    """Load ``KEY=VALUE`` lines into the process environment.

    Returns the keys that were parsed, in file order. Existing variables are
    left unchanged unless ``override`` is true. Quoted values lose one matching
    pair of single or double quotes. Inline comments and shell expansion are
    not supported.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return []

    loaded: list[str] = []
    for lineno, raw in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{env_path}:{lineno}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if _KEY.fullmatch(key) is None:
            raise ValueError(f"{env_path}:{lineno}: invalid environment variable name")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        loaded.append(key)
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded
