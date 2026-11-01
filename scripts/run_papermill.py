"""Execute all notebooks with papermill, recording timings and exit status.

Usage:
    python scripts/run_papermill.py                     # CPU, full proxy
    python scripts/run_papermill.py --fast              # smaller proxy
    python scripts/run_papermill.py --only 01_formula_audit
    python scripts/run_papermill.py --profile full      # notebook 07 only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROXY_NOTEBOOKS = [
    "00_overview",
    "01_formula_audit",
    "02_theory",
    "03_table_audit",
    "04_detection",
    "05_removal",
    "06_verdicts",
]
FULL_NOTEBOOKS = ["07_full_comparison"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--device", default=None, help="cpu | cuda | auto")
    parser.add_argument("--profile", choices=("proxy", "full"),
                        default=os.environ.get("EBBA_PROFILE", "proxy"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.fast and args.profile == "full":
        parser.error("--fast is a proxy-only option")
    if args.fast:
        os.environ["EBBA_FAST"] = "1"
    if args.device:
        os.environ["EBBA_DEVICE"] = args.device
    os.environ["EBBA_PROFILE"] = args.profile
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("EBBA_THREADS", "4" if args.profile == "proxy" else "8")
    os.environ.setdefault("EBBA_ROOT", str(ROOT))
    if args.profile == "full":
        os.environ.setdefault("EBBA_ARTIFACTS", str(ROOT / "artifacts-full"))
        os.environ.setdefault("EBBA_REPORTS", str(ROOT / "reports-full"))
    else:
        os.environ.setdefault("EBBA_ARTIFACTS", str(ROOT / "artifacts"))
        os.environ.setdefault("EBBA_REPORTS", str(ROOT / "reports"))

    import papermill as pm

    names = args.only or (FULL_NOTEBOOKS if args.profile == "full" else PROXY_NOTEBOOKS)
    if args.output_dir is None:
        args.output_dir = str(Path(os.environ["EBBA_REPORTS"]) / "executed")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "profile": args.profile,
        "fast": bool(args.fast),
        "device": os.environ.get("EBBA_DEVICE", "auto"),
        "notebooks": [],
        "started_at": time.time(),
    }
    status = 0

    for name in names:
        src = ROOT / "notebooks" / f"{name}.ipynb"
        dst = out_dir / f"{name}.ipynb"
        if not src.exists():
            print(f"[papermill] missing notebook: {src}", file=sys.stderr)
            status = 1
            metadata["notebooks"].append({"name": name, "status": "missing"})
            continue
        print(f"[papermill] running {name}")
        t0 = time.time()
        try:
            pm.execute_notebook(
                str(src), str(dst),
                kernel_name="python3",
                cwd=str(ROOT),
                progress_bar=False,
                request_save_on_cell_execute=False,
                execution_timeout=args.timeout,
            )
            rec = {"name": name, "status": "ok", "seconds": round(time.time() - t0, 1)}
        except Exception as exc:  # keep going so the report shows what ran
            rec = {
                "name": name, "status": "error", "seconds": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            status = 1
            print(f"[papermill] FAILED {name}: {exc}", file=sys.stderr)
        metadata["notebooks"].append(rec)
        print(f"[papermill] {name} {rec['status']} in {rec['seconds']}s")

    metadata["total_seconds"] = round(time.time() - metadata["started_at"], 1)
    (Path(os.environ["EBBA_REPORTS"]) / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
