"""Print the runtime fingerprint used by the notebooks and the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebba import artifacts, config, determinism  # noqa: E402


def main() -> int:
    import numpy as np
    import platform
    import torch

    print(f"python      : {platform.python_version()} ({platform.platform()})")
    print(f"numpy       : {np.__version__}")
    print(f"torch       : {torch.__version__}")
    print(f"profile     : {config.PROFILE} (model={config.MODEL_NAME}, epochs={config.EPOCHS})")
    print(f"threads     : {config.NUM_THREADS} (torch={torch.get_num_threads()})")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device : {torch.cuda.get_device_name(0)}")
        print(f"capability  : {torch.cuda.get_device_capability(0)}")
        print(f"arch list   : {torch.cuda.get_arch_list()}")
    print(f"device      : {determinism.get_device()}")
    print(f"root        : {config.PROJECT_ROOT}")
    print(f"artifacts   : {config.ARTIFACTS_DIR}")
    print(f"reports     : {config.REPORTS_DIR}")
    try:
        manifest = artifacts.ensure_artifacts(config.ARTIFACTS_DIR)
        print("manifest    : OK")
        print(f"cifar src   : {manifest.get('extra', {}).get('cifar_source', 'n/a')}")
        print(json.dumps(manifest.get("metrics", {}), indent=2))
        print(json.dumps(manifest.get("extra", {}).get("gates", {}), indent=2))
    except Exception as exc:
        print(f"manifest    : {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
