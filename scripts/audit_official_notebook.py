"""Standalone entry point for the official-notebook audit (claim A18)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebba.claims import run  # noqa: E402


def main() -> int:
    result = run("A18")
    print(json.dumps(result.measured, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
