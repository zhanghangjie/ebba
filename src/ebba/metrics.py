"""Evaluation metrics: BA, ASR, PDR, F1 and label-rank statistics."""

from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def attack_success_rate(logits: np.ndarray, target: int) -> float:
    """Fraction of poisoned inputs predicted as the target label."""
    return float((logits.argmax(axis=1) == target).mean())


def binary_metrics(pred_positive: np.ndarray, true_positive: np.ndarray) -> dict:
    pred = np.asarray(pred_positive, dtype=bool)
    true = np.asarray(true_positive, dtype=bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    pdr = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * pdr / max(precision + pdr, 1e-12)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pdr": pdr, "precision": precision, "f1": f1,
        "accuracy": (tp + tn) / max(len(pred), 1),
    }


def ranks(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """0-based rank of ``labels[i]`` in descending order for each row.

    Rank 0 means the label is the top prediction.  Ties are resolved by the
    position produced by a stable sort, which is deterministic.
    """
    logits = np.asarray(logits)
    ranks = np.empty(len(logits), dtype=np.int64)
    for i, (row, label) in enumerate(zip(logits, labels)):
        order = np.argsort(-row, kind="stable")
        ranks[i] = int(np.flatnonzero(order == label)[0])
    return ranks


def rank_summary(logits: np.ndarray, target: int, clean_label: int) -> dict:
    """Rank statistics used by Fig. 1 / Fig. 2.

    Returns the target-label rank and a randomly chosen non-target label rank
    for each sample, plus aggregate statistics.
    """
    n, k = logits.shape
    target_ranks = ranks(logits, np.full(n, target, dtype=np.int64))
    clean_ranks = ranks(logits, np.full(n, clean_label, dtype=np.int64))
    uniform = (k - 1) / 2.0
    return {
        "target_ranks": target_ranks,
        "clean_ranks": clean_ranks,
        "target_mean_rank": float(target_ranks.mean()),
        "clean_mean_rank": float(clean_ranks.mean()),
        "uniform_mean_rank": uniform,
        "target_top1": float((target_ranks == 0).mean()),
        "target_top3": float((target_ranks < 3).mean()),
        "target_topk_fraction": float((target_ranks < max(1, k // 2)).mean()),
        "clean_top3": float((clean_ranks < 3).mean()),
    }


def rank_shift_summary(
    source_logits: np.ndarray,
    corrupt_logits: np.ndarray,
    ground_truth: np.ndarray,
    target: int,
) -> dict:
    """Fraction of corruptions that move GT forward / target backward (Fig. 2)."""
    gt0 = ranks(source_logits, ground_truth)
    t0 = ranks(source_logits, np.full(len(source_logits), target, dtype=np.int64))
    gt_moves, tgt_moves = [], []
    for j in range(corrupt_logits.shape[0]):
        gt_moves.append(ranks(corrupt_logits[j], ground_truth))
        tgt_moves.append(ranks(corrupt_logits[j], np.full(len(source_logits), target, dtype=np.int64)))
    gt_moves = np.stack(gt_moves)
    tgt_moves = np.stack(tgt_moves)
    forward = gt_moves < gt0[None, :]
    backward = tgt_moves > t0[None, :]
    return {
        "gt_forward_fraction": float(forward.mean()),
        "target_backward_fraction": float(backward.mean()),
        "gt_mean_rank_before": float(gt0.mean()),
        "gt_mean_rank_after": float(gt_moves.mean()),
        "target_mean_rank_before": float(t0.mean()),
        "target_mean_rank_after": float(tgt_moves.mean()),
        # Fraction of samples whose GT ends up in the top 3 after corruption.
        "gt_top3_after": float((gt_moves < 3).mean()),
        "per_image_forward_fraction": forward.mean(axis=0).tolist(),
    }


def probability_shift_summary(
    source_logits: np.ndarray,
    corrupt_logits: np.ndarray,
    ground_truth: np.ndarray,
    target: int,
) -> dict:
    """Direct check of Lemma 2 at the softmax level."""
    from .energy import softmax

    s = softmax(source_logits)
    sc = softmax(corrupt_logits.reshape(-1, corrupt_logits.shape[-1])).reshape(corrupt_logits.shape)
    rows = np.arange(len(ground_truth))
    gt_before = s[rows, ground_truth]
    gt_after = sc[:, rows, ground_truth]
    t_before = s[rows, target]
    t_after = sc[:, rows, target]
    return {
        "target_decrease_fraction": float((t_after < t_before[None]).mean()),
        "gt_increase_fraction": float((gt_after > gt_before[None]).mean()),
        "both_fraction": float(((t_after < t_before[None]) & (gt_after > gt_before[None])).mean()),
        "mean_target_drop": float((t_before[None] - t_after).mean()),
        "mean_gt_gain": float((gt_after - gt_before[None]).mean()),
    }
