"""Append the GTSRB test split to the full-profile task-agnostic image set.

The full profile starts with CIFAR-100 test (10,000 images) as OOD.  The paper
also uses the other two datasets plus internet images, so adding GTSRB gives
the detection step a genuinely different visual domain.  GTSRB images are
resized to 32x32 to match the CIFAR input geometry and their labels are offset
by +100 so they never collide with CIFAR-100's 0-99 labels.

The script reads the GTSRB directory given on the command line (it is not
redistributed with this repository), keeps a backup of the CIFAR-100-only
array, rewrites ``cifar100_ood.npz`` and updates ``full_meta.json``.

Example
-------
python -I scripts/rebuild_full_ood.py \
    --gtsrb-root artifacts-full/data/raw/gtsrb \
    --data-dir artifacts-full/data
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

LABEL_OFFSET = 100


def _load_gtsrb(root: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = root / "GT-final_test.csv"
    image_dir = root / "GTSRB" / "Final_Test" / "Images"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing {image_dir}")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            name = (row.get("Filename") or "").strip()
            if not name:
                continue
            path = image_dir / name
            if not path.exists():
                raise FileNotFoundError(f"CSV references missing image {path}")
            with Image.open(path) as image:
                image = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
                xs.append(np.asarray(image, dtype=np.uint8))
            ys.append(int(row["ClassId"]))
    if not xs:
        raise RuntimeError("no GTSRB images loaded")
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    if len(np.unique(y)) != 43:
        print(f"[rebuild-ood] warning: {len(np.unique(y))} GTSRB classes, expected 43")
    return x, y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gtsrb-root", required=True,
                   help="directory containing GT-final_test.csv and GTSRB/Final_Test/Images")
    p.add_argument("--data-dir", default="artifacts-full/data")
    p.add_argument("--backup-name", default="cifar100_ood_cifar100only.npz")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if the OOD file already contains GTSRB")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    ood_path = data_dir / "cifar100_ood.npz"
    meta_path = data_dir / "full_meta.json"

    with np.load(ood_path) as bundle:
        x_old = bundle["x"].astype(np.uint8, copy=False)
        y_old = bundle["y"].astype(np.int64, copy=False)

    if y_old.max() >= LABEL_OFFSET:
        if not args.force:
            print(f"[rebuild-ood] {ood_path} already contains offset labels "
                  f"(max={int(y_old.max())}); pass --force to rebuild")
            return 0
        keep = y_old < LABEL_OFFSET
        print(f"[rebuild-ood] --force: dropping {int((~keep).sum())} existing GTSRB rows")
        x_old, y_old = x_old[keep], y_old[keep]

    x_g, y_g = _load_gtsrb(Path(args.gtsrb_root))
    x = np.concatenate([x_old, x_g], axis=0)
    y = np.concatenate([y_old, y_g + LABEL_OFFSET], axis=0)
    if x.shape[1:] != (32, 32, 3):
        raise SystemExit(f"unexpected image shape {x.shape[1:]}")

    backup = data_dir / args.backup_name
    if not backup.exists():
        np.savez_compressed(backup, x=x_old, y=y_old, indices=np.arange(len(x_old)))
        print(f"[rebuild-ood] backed up CIFAR-100-only OOD to {backup}")

    np.savez_compressed(ood_path, x=x, y=y, indices=np.arange(len(x)))
    print(f"[rebuild-ood] wrote {ood_path}: ood={len(x)} "
          f"(cifar100={len(x_old)}, gtsrb={len(x_g)})")

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sources = list(meta.get("ood_sources") or ["cifar100_test"])
        if "gtsrb_test" not in sources:
            sources.append("gtsrb_test")
        meta.update({
            "ood": int(len(x)),
            "ood_sources": sources,
            "gtsrb_test": int(len(x_g)),
            "gtsrb_classes": int(len(np.unique(y_g))),
            "gtsrb_resize": [32, 32],
        })
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[rebuild-ood] updated {meta_path}")
    else:
        print(f"[rebuild-ood] warning: {meta_path} missing; metadata not updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
