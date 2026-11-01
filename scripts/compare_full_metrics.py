# -*- coding: utf-8 -*-
"""Compare two full-profile metric files field by field.

Used to prove that a re-run (e.g. from the self-contained full-only image)
reproduces the production numbers instead of merely producing plausible ones.

    python scripts/compare_full_metrics.py OLD.json NEW.json [--tol 5e-3]

Exit code 0 when every checked field is within tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def get(d, path, default=None):
    # "/" is the escape for keys that themselves contain a dot, e.g.
    # "detection/badnets/lambda_3.0/zscore".
    sep = "/" if "/" in path else "."
    cur = d
    for key in path.split(sep):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# (label, json path, kind)  kind: num | int | bool | str
SPEC = [
    ("models.clean.ba", "models.clean.ba", "num"),
    ("models.badnets.ba", "models.badnets.ba", "num"),
    ("models.badnets.asr", "models.badnets.asr", "num"),
    ("models.blend.ba", "models.blend.ba", "num"),
    ("models.blend.asr", "models.blend.asr", "num"),
    ("protocol.ood_size", "protocol.ood_size", "int"),
    ("badnets.tp", "clustering.badnets.birch_embed.tp", "int"),
    ("badnets.fp", "clustering.badnets.birch_embed.fp", "int"),
    ("badnets.fn", "clustering.badnets.birch_embed.fn", "int"),
    ("badnets.tn", "clustering.badnets.birch_embed.tn", "int"),
    ("badnets.pdr", "clustering.badnets.birch_embed.pdr", "num"),
    ("badnets.f1", "clustering.badnets.birch_embed.f1", "num"),
    ("badnets.recall_nite", "nte.badnets.recovery_poisoned", "num"),
    ("badnets.asr_before", "removal.badnets.asr_before_pool", "num"),
    ("badnets.asr_after", "removal.badnets.asr_after_pool", "num"),
    ("badnets.ba_after", "removal.badnets.ba_after_pool", "num"),
    ("badnets.refined.pdr", "clustering.badnets.refined.pdr", "num"),
    ("blend.tp", "clustering.blend.birch_embed.tp", "int"),
    ("blend.fp", "clustering.blend.birch_embed.fp", "int"),
    ("blend.fn", "clustering.blend.birch_embed.fn", "int"),
    ("blend.tn", "clustering.blend.birch_embed.tn", "int"),
    ("blend.pdr", "clustering.blend.birch_embed.pdr", "num"),
    ("blend.f1", "clustering.blend.birch_embed.f1", "num"),
    ("blend.recovery_nite", "nte.blend.recovery_poisoned", "num"),
    ("blend.asr_before", "removal.blend.asr_before_pool", "num"),
    ("blend.asr_after", "removal.blend.asr_after_pool", "num"),
    ("blend.ba_after", "removal.blend.ba_after_pool", "num"),
    ("blend.refined.pdr", "clustering.blend.refined.pdr", "num"),
    ("det.clean.z2.2", "detection/clean/lambda_2.2/zscore", "num"),
    ("det.badnets.z2.2", "detection/badnets/lambda_2.2/zscore", "num"),
    ("det.blend.z2.2", "detection/blend/lambda_2.2/zscore", "num"),
    ("det.badnets.flag3", "detection/badnets/lambda_3.0/is_backdoored", "bool"),
    ("det.clean.flag3", "detection/clean/lambda_3.0/is_backdoored", "bool"),
    ("det.blend.flag3", "detection/blend/lambda_3.0/is_backdoored", "bool"),
    ("fig1.badnets.top3", "fig1.badnets.top3", "num"),
    ("fig1.badnets.rank", "fig1.badnets.mean_rank", "num"),
    ("fig2.badnets.target_drop", "fig2.badnets.probability_shift.target_decrease_fraction", "num"),
    ("fig2.badnets.gt_rise", "fig2.badnets.probability_shift.gt_increase_fraction", "num"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="tolerance for float fields (GPU nondeterminism); ints must match exactly")
    args = ap.parse_args()

    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))

    failures = []
    print(f"{'field':26s} {'old':>10s} {'new':>10s} {'delta':>10s}  status")
    print("-" * 72)
    for label, path, kind in SPEC:
        a, b = get(old, path), get(new, path)
        if a is None or b is None:
            failures.append((label, a, b, "missing"))
            print(f"{label:26s} {str(a):>10s} {str(b):>10s} {'':>10s}  MISSING")
            continue
        if kind == "int":
            ok = a == b
            delta = b - a
            shown = f"{delta:+d}" if not ok else "0"
        elif kind == "bool":
            ok = bool(a) == bool(b)
            delta = ""
            shown = "=" if ok else "≠"
        else:
            delta = float(b) - float(a)
            ok = abs(delta) <= args.tol
            shown = f"{delta:+.1e}" if abs(delta) < 1e-3 else f"{delta:+.4f}"
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failures.append((label, a, b, status))
        av = f"{a:.4f}" if isinstance(a, float) else str(a)
        bv = f"{b:.4f}" if isinstance(b, float) else str(b)
        print(f"{label:26s} {av:>10s} {bv:>10s} {str(shown):>10s}  {status}")

    print("-" * 72)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} field(s) outside tolerance)")
        return 1
    print(f"RESULT: PASS ({len(SPEC)} fields)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
