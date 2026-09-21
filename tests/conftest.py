"""Put the repository root and verification/ on sys.path for contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "verification"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
