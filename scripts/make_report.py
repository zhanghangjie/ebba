"""Aggregate claim JSON files into verdicts.json, report.md and report.html."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebba import report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--claims", default=None)
    args = parser.parse_args()
    paths = report.build(
        Path(args.out) if args.out else None,
        Path(args.claims) if args.claims else None,
    )
    for key, path in paths.items():
        print(f"{key:9s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
