"""Convert the Blend trigger image into the 32x32 uint8 array used by the full profile.

The paper uses the default "hello kitty" Blend trigger.  The image is not
redistributed with this repository; pass a local copy and keep the resulting
``.npy`` under ``artifacts-full/data/`` (the host-mounted directory).

Example
-------
python -I scripts/prepare_trigger.py \
    --image /path/to/hello_kitty.jpeg \
    --out artifacts-full/data/blend_hello_kitty.npy
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=32)
    args = p.parse_args()

    image = Image.open(args.image).convert("RGB")
    image = image.resize((args.size, args.size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (args.size, args.size, 3):
        raise SystemExit(f"unexpected trigger shape: {array.shape}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} shape={array.shape} dtype={array.dtype} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
