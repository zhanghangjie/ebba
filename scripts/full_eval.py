"""Evaluate the full profile (ResNet-18, full CIFAR-10) against the paper.

The full profile deliberately does **not** replace the fast proxy report.  It
answers a narrower question: after switching to the paper backbone and the
paper training recipe, do the EBBA/EBBA+ mechanisms move toward the paper's
numbers?  The output is a machine-readable JSON plus a standalone HTML page
under ``reports-full/``.

What is measured
----------------
* BA / ASR for clean, BadNets and Blend models on the full 10k test split.
* EBBA detection on task-agnostic images (CIFAR-100 test, optionally GTSRB).
* Fig. 1 target-label rank statistics.
* Fig. 2 probability/rank shifts under the 80 corruptions.
* NTE recovery (Eq. 12), t-SNE + Birch PDR/F1, four clustering methods.
* Diagnostics (not paper methods): a post-hoc two-component NTE threshold inside
  the selected cluster, and a cap/temperature/class-centring NTE sweep.
* Post-removal ASR / BA estimates.

What is not measured
--------------------
WaNet, FIBA and DUBA, GTSRB/ImageNet as *main* datasets, internet OOD images,
and the paper's exact training details that are not published.  These remain
explicit limitations in the HTML report.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebba import artifacts, attacks, clustering, config, corruptions, data, metrics, models  # noqa: E402
from ebba.claims import tables  # noqa: E402
from ebba.energy import detect, label_energy  # noqa: E402
from ebba.transferred import nte, recover_labels  # noqa: E402

PAPER_DATASET = "Cifar10"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_models(device: torch.device):
    artifacts.ensure_artifacts()
    out = {}
    for name in ("clean", *config.ATTACKS):
        path = Path(config.CHECKPOINT_DIR) / config.CHECKPOINTS[name]
        model, meta = models.load_checkpoint(path, map_location="cpu")
        model.to(device)
        model.eval()
        out[name] = {"model": model, "meta": meta, "path": path}
    return out


def _predict(model, images: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    return models.predict_logits(model, images, device=device, batch_size=batch_size)


def _embeddings(model, images: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Penultimate (512-d) ResNet-18 features, batchwise."""
    captured: list[np.ndarray] = []

    def hook(_module, _inputs, output):
        captured.append(output.flatten(1).detach().float().cpu().numpy())

    handle = model.avgpool.register_forward_hook(hook)
    outs: list[np.ndarray] = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                captured.clear()
                batch = models.to_tensor(images[start:start + batch_size]).to(device)
                model(batch)
                outs.append(captured[0])
    finally:
        handle.remove()
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 512), np.float32)


def _test_metrics(model, x_test, y_test, attack, device):
    clean_logits = _predict(model, x_test, device)
    poisoned = attacks.apply_trigger(x_test, attack)
    poison_logits = _predict(model, poisoned, device)
    return {
        "ba": float((clean_logits.argmax(1) == y_test).mean()),
        "asr": float((poison_logits.argmax(1) == config.TARGET_LABEL).mean()),
        "clean_logits": clean_logits,
        "poison_logits": poison_logits,
    }


def _detection(models_by_name, x_ood, device):
    rows = {}
    for name, rec in models_by_name.items():
        logits = _predict(rec["model"], x_ood, device)
        idx, counts = data.balanced_select_by_pseudo(logits, config.OOD_MIN_PER_CLASS)
        energy = label_energy(logits[idx], config.TEMPERATURE, "paper")
        row = {
            "n_selected": int(len(idx)),
            "per_label_counts": {str(k): int(v) for k, v in counts.items()},
            "energy": energy.tolist(),
            "mean": float(energy.mean()),
            "std": float(energy.std()),
        }
        for lam in (config.PROXY_LAMBDA, config.LAMBDA):
            det = detect(energy, lam)
            row[f"lambda_{lam}"] = det
        rows[name] = row
    return rows


def _fig1(models_by_name, x_ood, device):
    rows = {}
    target = config.TARGET_LABEL
    for name, rec in models_by_name.items():
        logits = _predict(rec["model"], x_ood, device)
        n = len(logits)
        ranks = metrics.ranks(logits, np.full(n, target, dtype=np.int64))
        rows[name] = {
            "n": int(n),
            "top1": float((ranks == 0).mean()),
            "top3": float((ranks < 3).mean()),
            "mean_rank": float(ranks.mean()),
            "uniform_mean_rank": (config.NUM_CLASSES - 1) / 2.0,
        }
    return rows


def _fig2(model, x_test, y_test, attack, device, max_images):
    target = config.TARGET_LABEL
    per_class = max(2, max_images // (config.NUM_CLASSES - 1))
    chosen = []
    for cls in range(config.NUM_CLASSES):
        if cls == target:
            continue
        chosen.extend(np.flatnonzero(y_test == cls)[:per_class].tolist())
    chosen = np.asarray(chosen[:max_images], dtype=np.int64)
    x = attacks.apply_trigger(x_test[chosen], attack)
    y = y_test[chosen]
    corrupt = corruptions.corrupt_all(x)
    source = _predict(model, x, device)
    corrupt_logits = _predict(
        model, corrupt.reshape(-1, *x.shape[1:]), device
    ).reshape(len(corrupt), len(x), config.NUM_CLASSES)
    return {
        "n_images": int(len(x)),
        "n_corruptions": int(len(corrupt)),
        "probability_shift": metrics.probability_shift_summary(source, corrupt_logits, y, target),
        "rank_shift": metrics.rank_shift_summary(source, corrupt_logits, y, target),
    }


def _candidate_pipeline(model, x_test, y_test, attack, device, max_poisoned, max_clean_target):
    target = config.TARGET_LABEL
    poisoned_test = attacks.apply_trigger(x_test, attack)
    candidate = data.make_candidate_set(
        x_test, y_test, poisoned_test, target,
        n_poisoned=max_poisoned, n_clean_target=max_clean_target,
    )
    source_logits = _predict(model, candidate["images"], device)
    keep = source_logits.argmax(1) == target
    if not keep.any():
        return None
    x = candidate["images"][keep]
    gt = candidate["ground_truth"][keep]
    is_poisoned = candidate["is_poisoned"][keep]
    source_logits = source_logits[keep]
    source_pred = source_logits.argmax(1)

    corrupt = corruptions.corrupt_all(x)
    corrupt_logits = _predict(
        model, corrupt.reshape(-1, *x.shape[1:]), device
    ).reshape(len(corrupt), len(x), config.NUM_CLASSES)
    # The paper says t-SNE is applied to the *perturbed images*.  We use the
    # ResNet penultimate feature of each corrupted image, averaged over the 80
    # corruptions.  The NTE vector is kept as a second, legacy feature space.
    corrupt_embed = _embeddings(
        model, corrupt.reshape(-1, *x.shape[1:]), device
    ).reshape(len(corrupt), len(x), -1).mean(axis=0)
    scores = nte(source_logits, corrupt_logits, temperature=1.0, cap=5.0)
    pred_labels = recover_labels(scores)
    peak = scores.max(axis=1)

    # P0 sweep: the paper's Eq. 9-11 are written without a per-corruption cap
    # and with T=1, while the official notebook caps at 5.  With
    # NTE = exp(s' - s) - 1 and s in [0, 1], a single-corruption value is at
    # most e - 1 ~= 1.72, so *no* cap >= 2 can bind at T=1.  The T=2 rows
    # (where the maximum is e^2 - 1 ~= 6.39) are the only place a cap matters;
    # keeping both makes that algebra visible in the report.  Class-centring
    # subtracts each label's mean NTE across candidates before Eq. 12.
    nte_variants = {}
    for name, kwargs in (
        ("cap5_T1", dict(temperature=1.0, cap=5.0)),
        ("uncapped_T1", dict(temperature=1.0, cap=None)),
        ("cap5_T2", dict(temperature=2.0, cap=5.0)),
        ("uncapped_T2", dict(temperature=2.0, cap=None)),
    ):
        nte_variants[name] = nte(source_logits, corrupt_logits, **kwargs)
    nte_variants["cap5_T1_classcenter"] = scores - scores.mean(axis=0, keepdims=True)
    others = np.delete(source_logits, target, axis=1)
    margin = source_logits[:, target] - others.max(axis=1)
    return {
        "x": x,
        "gt": gt,
        "is_poisoned": is_poisoned,
        "source_logits": source_logits,
        "source_pred": source_pred,
        "corrupt_logits": corrupt_logits,
        "corrupt_embed": corrupt_embed,
        "nte": scores,
        "nte_variants": nte_variants,
        "pred_labels": pred_labels,
        "peak": peak,
        "margin": margin,
        "coverage": float(keep.mean()),
        "n_candidate": int(len(candidate["images"])),
        "n_kept": int(len(x)),
    }


def _refine_cluster(peak: np.ndarray, base_pred: np.ndarray) -> tuple[np.ndarray, dict]:
    """P0: optional 1-D two-component refinement inside the poisoned cluster.

    Birch can put clean target-class images in the same cluster as poisoned
    ones.  The paper says a basic binary classifier can separate them; this is
    an unsupervised version of that step.  It is applied only when a two-
    component Gaussian mixture beats one component by a clear BIC margin, so a
    homogeneous cluster is left untouched.
    """
    from sklearn.mixture import GaussianMixture

    idx = np.flatnonzero(base_pred)
    if len(idx) < 20:
        return base_pred, {"applied": False, "reason": "fewer than 20 cluster members"}
    values = np.asarray(peak[idx], dtype=np.float64).reshape(-1, 1)
    one = GaussianMixture(n_components=1, random_state=config.SEED).fit(values)
    two = GaussianMixture(
        n_components=2, random_state=config.SEED, n_init=3
    ).fit(values)
    weights = two.weights_.ravel()
    means = two.means_.ravel()
    delta_bic = float(one.bic(values) - two.bic(values))
    applied = bool(delta_bic > 10.0 and weights.min() >= 0.02)
    if not applied:
        return base_pred, {
            "applied": False,
            "delta_bic": delta_bic,
            "weights": sorted(float(w) for w in weights),
            "means": sorted(float(m) for m in means),
        }
    high = int(np.argmax(means))
    keep = two.predict(values) == high
    refined = np.zeros_like(base_pred)
    refined[idx[keep]] = True
    return refined, {
        "applied": True,
        "delta_bic": delta_bic,
        "weights": sorted(float(w) for w in weights),
        "means": sorted(float(m) for m in means),
        "kept_fraction": float(keep.mean()),
    }


def _clustering(rec):
    # Primary configuration follows the paper's wording: t-SNE on the
    # perturbed-image representation.  The paper does not say how the poisoned
    # cluster is chosen without labels, so we use the cluster with the larger
    # mean peak NTE (the method's own energy-transfer signal).
    # t-SNE is used for the paper's visualisation, but clustering on the
    # standardized features is the reproducible path: at 1.5k points the exact
    # t-SNE solver still moved the Birch split between runs (GPU batch-size
    # differences in the logits are enough to change the embedding).
    primary = clustering.run_pipeline(
        rec["corrupt_embed"], method="birch", use_tsne=False,
        identification_scores=rec["peak"],
    )
    primary["identification_rule"] = "peak_nte"
    main = metrics.binary_metrics(primary["pred_poisoned"], rec["is_poisoned"])
    labels = primary["cluster_labels"]
    embedding = primary["embedding"]
    refined_pred, refined_stats = _refine_cluster(rec["peak"], primary["pred_poisoned"])
    refined = metrics.binary_metrics(refined_pred, rec["is_poisoned"])

    # P0 sweep: NTE cap/temperature/class-centring variants.  Recovery uses
    # Eq. 12 directly; PDR keeps the same Birch partition and only swaps the
    # peak-NTE orientation score, so the comparison isolates the NTE variant.
    nte_variants = {}
    poisoned = rec["is_poisoned"]
    clean = ~poisoned
    for name, variant_scores in rec["nte_variants"].items():
        variant_pred = recover_labels(variant_scores)
        variant_peak = np.asarray(variant_scores).max(axis=1)
        oriented, oriented_cluster, _ = clustering.identify_poisoned_cluster(
            labels, variant_peak
        )
        variant_metrics = metrics.binary_metrics(oriented, poisoned)
        # Removal depends on the *recovered labels*, so Eq. 12 variants can
        # change the post-removal ASR even when the cluster partition is fixed.
        relabelled = rec["source_pred"].copy()
        relabelled[oriented] = variant_pred[oriented]
        nte_variants[name] = {
            "recovery_poisoned": float(
                (variant_pred[poisoned] == rec["gt"][poisoned]).mean()
            ),
            "recovery_clean_target": float(
                (variant_pred[clean] == rec["gt"][clean]).mean()
            ),
            "pdr": variant_metrics["pdr"],
            "f1": variant_metrics["f1"],
            "precision": variant_metrics["precision"],
            "poisoned_cluster": oriented_cluster,
            "asr_after_pool": float(
                (relabelled[poisoned] == config.TARGET_LABEL).mean()
            ),
            "ba_after_pool": float(
                (relabelled[clean] == rec["gt"][clean]).mean()
            ),
        }

    methods = {}
    for method in ("hc", "birch", "meanshift", "dbscan"):
        if method == "birch":
            m_labels, used = labels, primary["method_used"]
        else:
            m_labels, used = clustering.cluster_labels(embedding, method)
        pred, cluster, stats = clustering.identify_poisoned_cluster(m_labels, rec["peak"])
        m = metrics.binary_metrics(pred, rec["is_poisoned"])
        methods[method] = {
            "pdr": m["pdr"], "f1": m["f1"], "precision": m["precision"],
            "method_used": used, "poisoned_cluster": cluster,
            "cluster_confidence": stats,
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        }

    # Legacy/sensitivity: cluster the NTE vector and orient with target margin.
    # This is what the proxy harness (B06/B07) uses; it is reported so the
    # reader can see how much the feature-space choice matters.
    legacy = clustering.run_pipeline(
        rec["nte"], method="birch", use_tsne=False,
        identification_scores=rec["margin"],
    )
    legacy_m = metrics.binary_metrics(legacy["pred_poisoned"], rec["is_poisoned"])

    # Oracle orientation: use the true labels to pick the better cluster.  This
    # is an upper bound, not a deployable rule.
    oracle = max(
        (metrics.binary_metrics(labels == c, rec["is_poisoned"])
         for c in np.unique(labels) if c >= 0),
        key=lambda d: d["f1"],
    )
    pdrs = [v["pdr"] for v in methods.values()]
    return {
        # Kept for the removal step; not written to JSON (full arrays).
        "pred_poisoned": primary["pred_poisoned"],
        "pred_refined": refined_pred,
        "feature_space": "perturbed-image ResNet-18 embedding (512-d, mean over 80 corruptions, standardized; Birch without t-SNE)",
        "identification_rule": "larger mean peak NTE",
        "birch_embed": main,
        "refined": {
            "rule": "two-component Gaussian mixture on peak NTE inside the poisoned cluster (BIC-gated)",
            "pdr": refined["pdr"], "f1": refined["f1"],
            "precision": refined["precision"], "accuracy": refined["accuracy"],
            "tp": refined["tp"], "fp": refined["fp"],
            "fn": refined["fn"], "tn": refined["tn"],
            "stats": refined_stats,
        },
        "nte_variants": nte_variants,
        "birch_embed_detail": {
            "pdr": main["pdr"], "f1": main["f1"], "precision": main["precision"],
            "tp": main["tp"], "fp": main["fp"], "fn": main["fn"], "tn": main["tn"],
            "method_used": primary["method_used"],
            "poisoned_cluster": primary["poisoned_cluster"],
            "cluster_confidence": primary["cluster_confidence"],
        },
        "birch_margin": legacy_m,
        "birch_margin_detail": {
            "pdr": legacy_m["pdr"], "f1": legacy_m["f1"],
            "precision": legacy_m["precision"],
            "tp": legacy_m["tp"], "fp": legacy_m["fp"],
            "fn": legacy_m["fn"], "tn": legacy_m["tn"],
            "feature_space": "NTE vector (10-d, standardized; legacy proxy path)",
            "identification_rule": "larger mean target margin",
            "method_used": legacy["method_used"],
            "poisoned_cluster": legacy["poisoned_cluster"],
            "cluster_confidence": legacy["cluster_confidence"],
        },
        "methods": methods,
        "pdr_spread": float(max(pdrs) - min(pdrs)),
        "peak_rule": {
            "pdr": main["pdr"], "f1": main["f1"],
            "precision": main["precision"], "poisoned_cluster": primary["poisoned_cluster"],
        },
        "oracle": {
            "pdr": oracle["pdr"], "f1": oracle["f1"], "precision": oracle["precision"],
        },
    }


def _removal(rec, cluster, test_metrics):
    target = config.TARGET_LABEL
    poisoned = rec["is_poisoned"]
    clean = ~poisoned
    asr_before = float((rec["source_pred"][poisoned] == target).mean())
    ba_before_pool = float((rec["source_pred"][clean] == rec["gt"][clean]).mean())

    def apply(pred):
        final = rec["source_pred"].copy()
        final[pred] = rec["pred_labels"][pred]
        return {
            "asr_after_pool": float((final[poisoned] == target).mean()),
            "ba_after_pool": float((final[clean] == rec["gt"][clean]).mean()),
            "fp": int((pred & clean).sum()),
        }

    base = apply(cluster["pred_poisoned"])
    refined = apply(cluster["pred_refined"])
    return {
        "asr_before_pool": asr_before,
        "asr_after_pool": base["asr_after_pool"],
        "asr_reduction_pool": asr_before - base["asr_after_pool"],
        "ba_before_pool": ba_before_pool,
        "ba_after_pool": base["ba_after_pool"],
        "ba_drop_pool": ba_before_pool - base["ba_after_pool"],
        "fp": base["fp"],
        "asr_after_pool_refined": refined["asr_after_pool"],
        "asr_reduction_pool_refined": asr_before - refined["asr_after_pool"],
        "ba_after_pool_refined": refined["ba_after_pool"],
        "fp_refined": refined["fp"],
        "n_poisoned_pool": int(poisoned.sum()),
        "n_clean_pool": int(clean.sum()),
        "ba_before_test": test_metrics["ba"],
        "asr_before_test": test_metrics["asr"],
    }


def _proxy_values() -> dict:
    """Read the already-produced proxy verdicts for a side-by-side column."""
    path = Path(config.PROJECT_ROOT) / "reports" / "verdicts.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        claims = payload.get("claims", {})
        out = {}
        for attack in config.ATTACKS:
            b07 = claims.get("B07", {}).get("measured", {}).get("attacks", {}).get(attack, {})
            b09 = claims.get("B09", {}).get("measured", {}).get("attacks", {}).get(attack, {})
            b02 = claims.get("B02", {}).get("measured", {}).get("models", {}).get(attack, {})
            b08 = claims.get("B08", {}).get("measured", {}).get("attacks", {}).get(attack, {})
            b01 = claims.get("B01", {}).get("measured", {}).get("models", {}).get(attack, {})
            b05 = claims.get("B05", {}).get("measured", {}).get("attacks", {}).get(attack, {})
            out[attack] = {
                "ba": b09.get("ba_before_test"),
                "asr": b09.get("asr_before_test"),
                "pdr": b07.get("pdr"),
                "f1": b07.get("f1"),
                "removal_asr": b09.get("asr_after_test_estimate"),
                "ba_drop": b09.get("ba_drop_test_estimate"),
                "detected": b02.get("is_backdoored"),
                "target": b02.get("target"),
                "pdr_spread": b08.get("pdr_spread"),
                "top3": b01.get("top3"),
                "nte_recovery": b05.get("recovery_accuracy_poisoned"),
            }
        return out
    except Exception:
        return {}


def _paper_values() -> dict:
    out = {}
    for attack in config.ATTACKS:
        t3 = tables.TABLE3[(PAPER_DATASET, "BadNets" if attack == "badnets" else "Blend")]
        t4 = tables.TABLE4[(PAPER_DATASET, "BadNets" if attack == "badnets" else "Blend")]
        out[attack] = {
            "pdr": t3["EBBA+"][0],
            "f1": t3["EBBA+"][1],
            "no_defense_ba": t4["No Defense"][0] / 100.0,
            "no_defense_asr": t4["No Defense"][1] / 100.0,
            "ebba_plus_ba": t4["EBBA+"][0] / 100.0,
            "ebba_plus_asr": t4["EBBA+"][1] / 100.0,
        }
    return out


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def _write_html(payload: dict, out_path: Path) -> None:
    env = payload["environment"]
    models_ = payload["models"]
    det = payload["detection"]
    fig1 = payload["fig1"]
    fig2 = payload["fig2"]
    nte_rows = payload["nte"]
    clust = payload["clustering"]
    removal = payload["removal"]
    paper = payload["paper"]
    proxy = payload.get("proxy", {})

    rows = []
    for attack in config.ATTACKS:
        m = models_.get(attack, {})
        p = paper.get(attack, {})
        c = clust.get(attack, {})
        r = removal.get(attack, {})
        px = proxy.get(attack, {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(attack)}</td>"
            f"<td>{_fmt(p.get('no_defense_ba'))} / {_fmt(p.get('no_defense_asr'))}</td>"
            f"<td>{_fmt(m.get('ba'))} / {_fmt(m.get('asr'))}</td>"
            f"<td>{_fmt(px.get('ba'))} / {_fmt(px.get('asr'))}</td>"
            f"<td>{_fmt(p.get('pdr'))} / {_fmt(p.get('f1'))}</td>"
            f"<td>{_fmt(c.get('birch_embed', {}).get('pdr'))} / "
            f"{_fmt(c.get('birch_embed', {}).get('f1'))} "
            f"<span class='muted'>(细化 {_fmt(c.get('refined', {}).get('pdr'))} / "
            f"{_fmt(c.get('refined', {}).get('f1'))})</span></td>"
            f"<td>{_fmt(px.get('pdr'))} / {_fmt(px.get('f1'))}</td>"
            f"<td>{_fmt(p.get('ebba_plus_asr'))}</td>"
            f"<td>{_fmt(r.get('asr_after_pool'))} "
            f"<span class='muted'>(细化 {_fmt(r.get('asr_after_pool_refined'))})</span></td>"
            f"<td>{_fmt(px.get('removal_asr'))}</td>"
            "</tr>"
        )

    det_rows = []
    for name, rec in det.items():
        d22 = rec.get(f"lambda_{config.PROXY_LAMBDA}", {})
        d30 = rec.get(f"lambda_{config.LAMBDA}", {})
        det_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_fmt(d22.get('target'))}</td>"
            f"<td>{_fmt(d22.get('zscore'))}</td>"
            f"<td>{_fmt(d22.get('is_backdoored'))}</td>"
            f"<td>{_fmt(d30.get('zscore'))}</td>"
            f"<td>{_fmt(d30.get('is_backdoored'))}</td>"
            "</tr>"
        )

    fig1_rows = []
    for name, rec in fig1.items():
        fig1_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_fmt(rec.get('top1'))}</td>"
            f"<td>{_fmt(rec.get('top3'))}</td>"
            f"<td>{_fmt(rec.get('mean_rank'))}</td>"
            f"<td>{_fmt(rec.get('uniform_mean_rank'))}</td>"
            "</tr>"
        )

    fig2_rows = []
    for attack, rec in fig2.items():
        ps = rec.get("probability_shift", {})
        rs = rec.get("rank_shift", {})
        fig2_rows.append(
            "<tr>"
            f"<td>{html.escape(attack)}</td>"
            f"<td>{_fmt(ps.get('target_decrease_fraction'))}</td>"
            f"<td>{_fmt(ps.get('gt_increase_fraction'))}</td>"
            f"<td>{_fmt(ps.get('both_fraction'))}</td>"
            f"<td>{_fmt(rs.get('gt_forward_fraction'))}</td>"
            f"<td>{_fmt(rs.get('target_backward_fraction'))}</td>"
            "</tr>"
        )

    nte_table = []
    for attack, rec in nte_rows.items():
        nte_table.append(
            "<tr>"
            f"<td>{html.escape(attack)}</td>"
            f"<td>{_fmt(rec.get('coverage'))}</td>"
            f"<td>{_fmt(rec.get('recovery_poisoned'))}</td>"
            f"<td>{_fmt(rec.get('recovery_clean_target'))}</td>"
            f"<td>{_fmt(rec.get('n_kept'))}</td>"
            "</tr>"
        )

    cluster_rows = []
    for attack, rec in clust.items():
        methods = rec.get("methods", {})
        ref = rec.get("refined", {})
        stats = ref.get("stats", {}) or {}
        applied = stats.get("applied")
        cluster_rows.append(
            "<tr>"
            f"<td>{html.escape(attack)}</td>"
            f"<td>{_fmt(rec.get('birch_embed_detail', {}).get('pdr'))}</td>"
            f"<td>{_fmt(rec.get('birch_embed_detail', {}).get('f1'))}</td>"
            f"<td>{_fmt(rec.get('birch_embed_detail', {}).get('precision'))}</td>"
            f"<td>{_fmt(ref.get('pdr'))}</td>"
            f"<td>{_fmt(ref.get('f1'))}</td>"
            f"<td>{_fmt(ref.get('precision'))}</td>"
            f"<td>{_fmt(ref.get('fp'))}</td>"
            f"<td>{_fmt(applied)}"
            f"<span class='muted'> (ΔBIC {_fmt(stats.get('delta_bic'), 1)})</span></td>"
            f"<td>{_fmt(rec.get('birch_margin_detail', {}).get('pdr'))}</td>"
            f"<td>{_fmt(rec.get('oracle', {}).get('pdr'))}</td>"
            f"<td>{_fmt(rec.get('pdr_spread'))}</td>"
            f"<td>{_fmt(methods.get('hc', {}).get('pdr'))} / "
            f"{_fmt(methods.get('birch', {}).get('pdr'))} / "
            f"{_fmt(methods.get('meanshift', {}).get('pdr'))} / "
            f"{_fmt(methods.get('dbscan', {}).get('pdr'))}</td>"
            "</tr>"
        )

    variant_rows = []
    variant_names = []
    for rec in clust.values():
        for name in (rec.get("nte_variants") or {}):
            if name not in variant_names:
                variant_names.append(name)
    for attack, rec in clust.items():
        variants = rec.get("nte_variants") or {}
        for name in variant_names:
            v = variants.get(name, {})
            variant_rows.append(
                "<tr>"
                f"<td>{html.escape(attack)}</td>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{_fmt(v.get('recovery_poisoned'))}</td>"
                f"<td>{_fmt(v.get('recovery_clean_target'))}</td>"
                f"<td>{_fmt(v.get('pdr'))}</td>"
                f"<td>{_fmt(v.get('f1'))}</td>"
                f"<td>{_fmt(v.get('precision'))}</td>"
                f"<td>{_fmt(v.get('asr_after_pool'))}</td>"
                f"<td>{_fmt(v.get('ba_after_pool'))}</td>"
                f"<td>{_fmt(v.get('poisoned_cluster'))}</td>"
                "</tr>"
            )

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         max-width:1180px;margin:32px auto;padding:0 20px;color:#1f2328;line-height:1.55}
    h1{font-size:26px} h2{font-size:19px;margin-top:30px;border-bottom:1px solid #d8dee4;padding-bottom:6px}
    table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
    th,td{border:1px solid #d8dee4;padding:7px 9px;text-align:left;vertical-align:top}
    th{background:#f6f8fa} code{background:#f6f8fa;padding:1px 5px;border-radius:4px}
    .note{background:#fff8e5;border-left:4px solid #d4a72c;padding:10px 14px;margin:14px 0}
    .ok{color:#1b7f3a;font-weight:600}.warn{color:#9a6700;font-weight:600}
    .muted{color:#57606a;font-size:12px}
    """
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>EBBA full profile — ResNet-18 / 200 epochs</title>
<style>{css}</style></head><body>
<h1>EBBA / EBBA+ 全量档对比（ResNet-18，完整 CIFAR-10）</h1>
<p class="muted">生成时间：{html.escape(payload['generated_at'])} ·
设备：{html.escape(str(env.get('device')))} ·
GPU：{html.escape(str(env.get('gpu')))} ·
torch：{html.escape(str(env.get('torch')))}</p>

<div class="note">
<b>这不是论文的逐位复现。</b> 本档使用 ResNet-18、完整 CIFAR-10 50k/10k 和论文的
200-epoch 训练配方，但攻击实现、OOD 图像（CIFAR-100 test，可选 GTSRB）、聚类定向规则
和随机种子与原文不完全相同。数值应作为“同骨干、同规模下的独立复核”，不是论文表格的替代品。
</div>

<h2>1. 主结果：BA / ASR、PDR / F1、去除后 ASR</h2>
<table>
<tr><th rowspan="2">攻击</th><th colspan="3">BA / ASR（未防御）</th>
<th colspan="3">PDR / F1</th><th colspan="3">去除后 ASR</th></tr>
<tr><th>论文 CIFAR-10</th><th>ResNet-18 full</th><th>SmallCNN proxy</th>
<th>论文 EBBA+</th><th>ResNet-18 full</th><th>SmallCNN proxy</th>
<th>论文 EBBA+</th><th>ResNet-18 full（池内）</th><th>SmallCNN proxy（全测试集估计）</th></tr>
{''.join(rows)}
</table>
<p class="muted">论文列取自 Table 3（PDR/F1）与 Table 4（BA/ASR）。
full 的去除后 ASR 是候选池内指标；proxy 列沿用 B09 的全测试集估计，两者口径不同，不能直接相减。
括号内的“细化”值使用 P0 的池内二次阈值（见第 6 节），是为了说明分数有多少来自聚类本身、
多少来自事后加的可调规则；<b>论文数字对应的是括号外的值</b>，比较论文时请用括号外。</p>

<h2>2. EBBA 检测（任务无关图像）</h2>
<table>
<tr><th>模型</th><th>λ=2.2 目标</th><th>λ=2.2 z</th><th>λ=2.2 检出</th>
<th>λ=3 z</th><th>λ=3 检出</th></tr>
{''.join(det_rows)}
</table>
<p class="muted">λ=3 是论文值；K=10 时单标签 z 的上界正好是 3（A20），严格 &gt;3σ 不可达，
因此同时报告 λ_proxy=2.2。干净模型的期望是“不检出”，后门模型的期望是“目标类 0 检出”。</p>

<h2>3. Fig. 1：目标标签在任务无关图像中的排名</h2>
<table>
<tr><th>模型</th><th>top-1</th><th>top-3</th><th>平均排名</th><th>均匀基线平均排名</th></tr>
{''.join(fig1_rows)}
</table>

<h2>4. Fig. 2：损坏后的概率/排名迁移</h2>
<table>
<tr><th>攻击</th><th>目标概率下降比例</th><th>真值概率上升比例</th><th>两者同时发生</th>
<th>真值排名前移比例</th><th>目标排名后移比例</th></tr>
{''.join(fig2_rows)}
</table>

<h2>5. NTE 标签恢复（Eq. 12）</h2>
<table>
<tr><th>攻击</th><th>候选池覆盖率</th><th>中毒图恢复率</th><th>干净目标类恢复率</th><th>进入 EBBA+ 的图数</th></tr>
{''.join(nte_table)}
</table>

<h2>6. EBBA+ 聚类：PDR / F1 与特征空间敏感性</h2>
<table>
<tr><th rowspan="2">攻击</th><th colspan="3">主配置（Birch + 峰值 NTE 定向）</th>
<th colspan="5">池内二次阈值（簇内 peak NTE 的 1D 高斯混合，BIC 门控）</th>
<th colspan="4">对照</th></tr>
<tr><th>PDR</th><th>F1</th><th>精确率</th>
<th>PDR</th><th>F1</th><th>精确率</th><th>FP</th><th>是否启用</th>
<th>NTE 特征空间 PDR（旧路径）</th><th>Oracle PDR</th><th>四方法极差</th>
<th>PDR（HC / Birch / MeanShift / DBSCAN）</th></tr>
{''.join(cluster_rows)}
</table>
<p class="muted">主配置对<b>损坏后图像的 ResNet-18 倒数第二层特征</b>（80 种损坏取平均）做标准化 + Birch，
并用簇内平均峰值 NTE 认定中毒簇（论文未说明无标签时的定向规则）。论文的 t-SNE 仅用于
可视化：在本档 1.5k 候选点上，t-SNE 的精确解仍会随 GPU logits 的批次差异改变 Birch 划分，
因此不作为可复现的聚类路径。第二组是 P0 新增的池内二次阈值：在已定向的中毒簇内部对
peak NTE 拟合 1 组分与 2 组分高斯混合，只有 ΔBIC&gt;10 且最小分量权重 ≥2% 时才把低分量
改判为干净，用于压低误报；<b>它是在看到本档数据后加入的可调规则，不是论文方法</b>，
因此只有 PDR/F1 上升而 FP 不失控才值得保留。
Oracle 是上界，不是可部署规则。</p>

<h3>6b. NTE 上限/温度/类别归一化扫描（P0）</h3>
<table>
<tr><th>攻击</th><th>NTE 变体</th><th>中毒图恢复率</th><th>干净目标类恢复率</th>
<th>PDR（同一 Birch 划分，仅换定向分数）</th><th>F1</th><th>精确率</th>
<th>去除后池内 ASR</th><th>去除后池内 BA</th><th>选定簇</th></tr>
{''.join(variant_rows)}
</table>
<p class="muted">Eq. 9/10 在论文的 T=1 下是 <code>exp(s' − s) − 1</code>；由于 <code>s ∈ [0,1]</code>，
单个损坏的 NTE 最大只有 <code>e−1 ≈ 1.72</code>，所以 <b>任何 ≥2 的 cap 在 T=1 下都不可能生效</b>
（表中 cap5_T1 与 uncapped_T1 必然逐位相同，这不是 bug）。论文官方 Notebook 的
<code>min(exp(2s), cap)</code> 是另一套归一化，cap 在那里才会截断。T=2 行的上界是
<code>e²−1 ≈ 6.39</code>，cap=5 才可能起作用，因此用这两行判断“上限”是否影响结论。
最后一行按类别中心化（<code>scores − mean_over_samples</code>），消除类别偏置。
恢复率回答 Eq. 12，PDR 回答聚类定向是否受同一变体影响；“选定簇”是簇编号。
同一簇划分下，去除步骤用的是 Eq. 12 的恢复标签，所以“去除后 ASR/BA”会随归一化变化——
这是本表最有行动价值的两列。</p>

<h2>7. 结果怎么读</h2>
<ul>
<li><b>看方向与量级，不看小数点。</b> 论文的 PDR/F1 约 0.95；full 档若落在 0.80–0.95，
说明机制在论文骨干上成立，但仍受 OOD 与攻击实现差异影响。</li>
<li><b>BA/ASR 是实验台门槛。</b> 若 ASR 明显低于论文的 ~99%，PDR/F1 的比较意义会下降，
因为可检测的后门信号更弱；报告仍如实展示。</li>
<li><b>聚类极差是论文 Table 5/7 的关键断言。</b> 论文报告 0.010；full 档若仍大于 0.05，
说明“换聚类方法几乎不变”依赖簇本身的高纯度，而不是方法无关。</li>
<li><b>去除后 ASR 是池内指标。</b> 它回答“被 EBBA+ 检查的中毒图还有多少仍成功”，
不等同于论文在完整测试集上的 ASR；两者口径不同。</li>
<li><b>池内二次阈值与 NTE 扫描是诊断，不是论文方法。</b> 它们用来判断差距来自特征几何、
定向规则还是能量定义；若只有加上可调规则后数字才接近论文，应如实写成
“论文方法 + 事后规则”，不能算作论文方法的复现成绩。</li>
<li><b>不能把本页写成“复现了论文的 96% PDR”。</b> 正确表述是“在 ResNet-18/200 epochs
的独立设定下，PDR/F1 达到某值，与论文的差距为某值”。</li>
</ul>

<h2>8. 数据与协议</h2>
<pre>{html.escape(json.dumps(payload.get('protocol', {}), indent=2, ensure_ascii=False))}</pre>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    if config.PROFILE != "full":
        raise SystemExit(
            "full_eval.py requires EBBA_PROFILE=full (or the ebba-full service). "
            "The proxy report is produced by scripts/run_papermill.py."
        )
    determinism_seed = args.seed
    torch.manual_seed(determinism_seed)
    np.random.seed(determinism_seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[full-eval] CUDA unavailable; falling back to CPU", flush=True)
        device = torch.device("cpu")

    t0 = time.time()
    manifest = artifacts.ensure_artifacts()
    models_by_name = _load_models(device)
    bundle = artifacts.load_data()
    x_test, y_test = bundle["x_test"], bundle["y_test"]
    x_ood = bundle["x_ood"]
    print(f"[full-eval] models loaded; test={len(x_test)} ood={len(x_ood)}", flush=True)

    out = {
        "profile": "full",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": args.threads,
        },
        "protocol": {
            "profile": config.PROFILE,
            "model": "ResNet18CIFAR",
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "milestones": list(config.LR_MILESTONES),
            "train_size": len(bundle["x_train"]),
            "test_size": len(x_test),
            "ood_size": len(x_ood),
            "ood_source": "CIFAR-100 test" + (" + GTSRB test" if config.OOD_GTSRB else ""),
            "poison_rate": config.POISON_RATE,
            "target": config.TARGET_LABEL,
            "lambda_paper": config.LAMBDA,
            "lambda_proxy": config.PROXY_LAMBDA,
            "temperature": config.TEMPERATURE,
        },
        "manifest": manifest,
        "models": {},
        "detection": {},
        "fig1": {},
        "fig2": {},
        "nte": {},
        "clustering": {},
        "removal": {},
        "paper": _paper_values(),
        "proxy": _proxy_values(),
    }

    test_metrics = {}
    for name, rec in models_by_name.items():
        attack = "badnets" if name == "clean" else name
        tm = _test_metrics(rec["model"], x_test, y_test, attack, device)
        test_metrics[name] = tm
        out["models"][name] = {
            "ba": tm["ba"],
            "asr": tm["asr"] if name != "clean" else None,
            "checkpoint": str(rec["path"]),
            "meta": rec["meta"],
        }
        print(f"[full-eval] {name}: BA={tm['ba']:.4f} ASR={tm['asr']}", flush=True)

    out["detection"] = _detection(models_by_name, x_ood, device)
    out["fig1"] = _fig1(models_by_name, x_ood, device)
    print("[full-eval] detection + Fig.1 done", flush=True)

    for attack in config.ATTACKS:
        model = models_by_name[attack]["model"]
        out["fig2"][attack] = _fig2(
            model, x_test, y_test, attack, device, args.max_fig2
        )
        print(f"[full-eval] Fig.2 {attack} done", flush=True)

    for attack in config.ATTACKS:
        model = models_by_name[attack]["model"]
        rec = _candidate_pipeline(
            model, x_test, y_test, attack, device,
            max_poisoned=args.max_poisoned,
            max_clean_target=args.max_clean_target,
        )
        if rec is None:
            out["nte"][attack] = {"error": "no candidate predicted as target"}
            continue
        poisoned = rec["is_poisoned"]
        clean = ~poisoned
        out["nte"][attack] = {
            "coverage": rec["coverage"],
            "n_candidate": rec["n_candidate"],
            "n_kept": rec["n_kept"],
            "n_poisoned": int(poisoned.sum()),
            "n_clean_target": int(clean.sum()),
            "recovery_poisoned": float((rec["pred_labels"][poisoned] == rec["gt"][poisoned]).mean()),
            "recovery_clean_target": float((rec["pred_labels"][clean] == rec["gt"][clean]).mean()),
        }
        cluster = _clustering(rec)
        out["clustering"][attack] = dict(cluster)
        rem = _removal(rec, cluster, test_metrics[attack])
        out["clustering"][attack].pop("pred_poisoned", None)
        # Global estimates use the same convention as the proxy B09 claim.
        n_test = len(y_test)
        n_target_class = int((y_test == config.TARGET_LABEL).sum())
        rem["ba_drop_test_estimate"] = (
            rem["fp"] * (n_target_class / max(rem["n_clean_pool"], 1)) / n_test
        )
        rem["asr_after_test_estimate"] = (
            test_metrics[attack]["asr"] * rem["asr_after_pool"]
        )
        out["removal"][attack] = rem
        best_variant = max(
            cluster["nte_variants"].items(),
            key=lambda kv: kv[1]["pdr"],
            default=("n/a", {"pdr": float("nan"), "recovery_poisoned": float("nan")}),
        )
        print(
            f"[full-eval] {attack}: PDR={cluster['birch_embed']['pdr']:.4f} "
            f"F1={cluster['birch_embed']['f1']:.4f} "
            f"refined_PDR={cluster['refined']['pdr']:.4f} "
            f"FP={rem['fp']}->{rem['fp_refined']} "
            f"ASR_after_pool={rem['asr_after_pool']:.4f}"
            f"->{rem['asr_after_pool_refined']:.4f} "
            f"best_NTE={best_variant[0]} "
            f"(PDR={best_variant[1]['pdr']:.4f}, rec={best_variant[1]['recovery_poisoned']:.4f})",
            flush=True,
        )

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_jsonable(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html_path = Path(args.output_html)
    _write_html(_jsonable(out), html_path)
    print(f"[full-eval] wrote {out_path} and {html_path} in {out['elapsed_seconds']}s")
    return _jsonable(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--threads", type=int, default=config.NUM_THREADS)
    p.add_argument("--max-poisoned", type=int, default=config.NTE_NUM_POISONED)
    p.add_argument("--max-clean-target", type=int, default=config.NTE_NUM_CLEAN_TARGET)
    p.add_argument("--max-fig2", type=int, default=200)
    p.add_argument("--output-json", default=str(Path(config.REPORTS_DIR) / "full_metrics.json"))
    p.add_argument("--output-html", default=str(Path(config.REPORTS_DIR) / "full_report.html"))
    p.add_argument("--smoke", action="store_true",
                   help="small evaluation samples; use with a 1-epoch smoke artifact")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        args.max_poisoned = min(args.max_poisoned, 100)
        args.max_clean_target = min(args.max_clean_target, 50)
        args.max_fig2 = min(args.max_fig2, 30)
        args.threads = min(args.threads, 8)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
