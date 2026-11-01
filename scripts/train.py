"""Retrain one proxy checkpoint and compare it with the baked-in artifact.

    python scripts/train.py --attack badnets
    python scripts/train.py --attack blend --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebba import artifacts, attacks, config, data, determinism, models  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=("clean", *config.ATTACKS), default="badnets")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--robust", action=argparse.BooleanOptionalAction,
                        default=config.ROBUST_AUG,
                        help="match build_artifacts.py's corruption augmentation")
    parser.add_argument("--output-dir", default=str(Path(config.ARTIFACTS_DIR) / "retrained"))
    args = parser.parse_args()

    determinism.set_global_seed(config.SEED)
    determinism.configure_torch(config.NUM_THREADS)
    device = determinism.get_device(args.device)

    training = data.load_training_data(args.attack)
    model, history = models.train_model(
        training["x"], training["y"], epochs=args.epochs, device=device,
        verbose=True, robust=args.robust,
    )
    x_test, y_test = data.load_test_data()
    trigger_attack = "badnets" if args.attack == "clean" else args.attack
    poisoned = attacks.apply_trigger(x_test, trigger_attack)
    ba = float((models.predict_logits(model, x_test, device).argmax(1) == y_test).mean())
    asr = float((models.predict_logits(model, poisoned, device).argmax(1) == config.TARGET_LABEL).mean())

    out = Path(args.output_dir)
    checkpoint = out / config.CHECKPOINTS[args.attack]
    meta = {
        "attack": args.attack, "epochs": args.epochs, "seed": config.SEED,
        "retrained": True, "metrics": {"ba": ba, "asr": asr},
    }
    models.save_checkpoint(checkpoint, model, meta, history)
    print(f"[retrain] {args.attack}: BA={ba:.4f} ASR={asr:.4f} -> {checkpoint}")

    manifest_path = Path(config.ARTIFACTS_DIR) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest.get("metrics", {}).get(args.attack, {})
        if base:
            print("[compare] baked  BA={:.4f} ASR={:.4f}".format(base["ba"], base["asr"]))
            print("[compare] delta  BA={:+.4f} ASR={:+.4f}".format(ba - base["ba"], asr - base["asr"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
