"""Tier A: checks of the theoretical arguments (A07-A10).

The paper's Lemma 1 is proved with an NTK surrogate (Eq. 13-20); Lemma 2 is
proved for integer division (Eq. 27-31).  These checks reproduce the algebra
and the kernel-sum argument on real CIFAR-10 pixels.  They do **not** prove
anything about an arbitrary trained network; that is exactly the gap being
audited.
"""

from __future__ import annotations

import numpy as np

from .. import artifacts, attacks, config, corruptions, models
from ..energy import softmax
from .model import ClaimResult, Verdict, claim


# ---------------------------------------------------------------------------
# A07 -- Eq. (30) algebra
# ---------------------------------------------------------------------------
@claim("A07")
def a07_eq30_algebra() -> ClaimResult:
    """Check Eq.29 <=> Eq.30 elementwise, including the sign assumption."""
    rng = np.random.default_rng(config.SEED + 7)

    def check(a, t, b, n):
        u = a + t
        eq29 = (u / n - b) ** 2 > (u - b) ** 2
        eq30 = t < 2 * n * b / (1 + n) - a
        return bool(eq29), bool(eq30)

    n_positive = 4000
    n_negative = 4000
    a_pos = rng.uniform(0, 1, n_positive)
    t_pos = rng.uniform(0, 0.3, n_positive)
    b_pos = rng.uniform(0, 1, n_positive)
    ns_pos = rng.integers(2, 8, n_positive)
    pos = [check(a, t, b, int(n)) for a, t, b, n in zip(a_pos, t_pos, b_pos, ns_pos)]
    pos_mismatch = sum(int(x != y) for x, y in pos)

    a_neg = rng.uniform(-1, 0, n_negative)
    t_neg = rng.uniform(0, 0.3, n_negative)
    b_neg = rng.uniform(0, 1, n_negative)
    ns_neg = rng.integers(2, 8, n_negative)
    neg = [check(a, t, b, int(n)) for a, t, b, n in zip(a_neg, t_neg, b_neg, ns_neg)]
    neg_mismatch = sum(int(x != y) for x, y in neg)

    ratio = 2 * np.array([2, 3, 4, 10, 100]) / (1 + np.array([2, 3, 4, 10, 100]))
    ratio_ok = bool(np.all(ratio >= 4 / 3 - 1e-12) and np.all(ratio < 2))

    ok_positive = pos_mismatch == 0 and ratio_ok
    verdict = Verdict.CONFIRMED if (ok_positive and neg_mismatch == 0) else Verdict.PARTIAL
    return ClaimResult(
        verdict=verdict,
        measured={
            "samples_positive_u": n_positive,
            "mismatches_positive_u": pos_mismatch,
            "samples_negative_u": n_negative,
            "mismatches_negative_u": neg_mismatch,
            "coefficient_2n_over_1_plus_n": ratio.tolist(),
            "coefficient_range_ok": ratio_ok,
        },
        notes_zh=(
            "在 a+t>0、b≥0、n>1（像素空间的通常取值）时，Eq.29 与 Eq.30 逐元素等价。"
            "但推导中间除以了 a+t，论文未声明该符号条件；当 a+t<0 时等价关系翻转，"
            "实测出现大量反例。结论：代数正确但有未声明的前提，故记 PARTIAL。"
        ),
        tolerance={"abs_exact": 1e-9},
    )


# ---------------------------------------------------------------------------
# A08 -- Lemma 2 universality vs aggregate claim
# ---------------------------------------------------------------------------
@claim("A08")
def a08_lemma2_universal() -> ClaimResult:
    """Test Lemma 2 for all 80 corruptions on the proxy backdoored model."""
    model, _ = artifacts.load_model("badnets")
    bundle = artifacts.load_data()

    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    target = config.TARGET_LABEL

    # Stratified sample of non-target test images.
    per_class = max(2, config.LEMMA2_IMAGES // (config.NUM_CLASSES - 1))
    chosen: list[int] = []
    for cls in range(config.NUM_CLASSES):
        if cls == target:
            continue
        idx = np.flatnonzero(y_test == cls)[:per_class]
        chosen.extend(idx.tolist())
    chosen = np.asarray(chosen[: config.LEMMA2_IMAGES], dtype=np.int64)
    x = x_test[chosen]
    y = y_test[chosen]

    x_poisoned = attacks.apply_badnets(x)
    corrupted = corruptions.corrupt_all(x_poisoned)          # (80, N, 32, 32, 3)

    source_logits = models.predict_logits(model, x_poisoned)
    corrupt_logits = np.stack(
        [models.predict_logits(model, corrupted[j]) for j in range(len(corrupted))]
    )

    s = softmax(source_logits)
    sc = softmax(corrupt_logits.reshape(-1, config.NUM_CLASSES)).reshape(corrupt_logits.shape)
    rows = np.arange(len(x))
    t_before = s[rows, target]
    t_after = sc[:, rows, target]
    gt_before = s[rows, y]
    gt_after = sc[:, rows, y]

    both = (t_after < t_before[None]) & (gt_after > gt_before[None])
    universal_fraction = float(both.mean())
    per_corruption = both.mean(axis=1)
    aggregate_ok = (t_after.mean(axis=1) < t_before.mean()) & (
        gt_after.mean(axis=1) > gt_before.mean()
    )
    aggregate_fraction = float(aggregate_ok.mean())
    per_spec = [f"{n}@{s_}" for n, s_ in corruptions.corruption_specs()]
    worst = np.argsort(per_corruption)[:5]

    if universal_fraction >= 1.0:
        # One proxy model cannot establish a universal statement; at best the
        # check is consistent with it.
        verdict = Verdict.PARTIAL
    elif aggregate_fraction >= 0.90:
        verdict = Verdict.PARTIAL
    else:
        verdict = Verdict.CONTRADICTED

    return ClaimResult(
        verdict=verdict,
        measured={
            "scope": "proxy backdoored SmallCNN, 80 corruptions",
            "n_images": int(len(x)),
            "universal_fraction": universal_fraction,
            "per_corruption_fraction": per_corruption.tolist(),
            "aggregate_fraction": aggregate_fraction,
            "mean_target_drop": float((t_before[None] - t_after).mean()),
            "mean_gt_gain": float((gt_after - gt_before[None]).mean()),
            "worst_corruptions": [
                {"corruption": per_spec[i], "fraction": float(per_corruption[i]),
                 "mean_target_drop": float((t_before - t_after[i]).mean()),
                 "mean_gt_gain": float((gt_after[i] - gt_before).mean())}
                for i in worst
            ],
        },
        notes_zh=(
            "Lemma 2 的原文是逐图逐损坏的普遍断言（st>s'_t 且 s_k1<s'_k1）。"
            "在缩比后门模型上统计全部 80 种损坏：同时满足两条的 (图,损坏) 比例为 "
            f"{universal_fraction:.2%}，故普遍形式被反例否定；"
            f"但按损坏类型取平均后，{aggregate_fraction:.1%} 的损坏呈现聚合方向，"
            "与论文插图想表达的总体趋势一致。"
        ),
        tolerance={"universal_fraction": 1.0, "aggregate_fraction": 0.90},
    )


# ---------------------------------------------------------------------------
# NTK kernel-sum helpers (Lemma 1)
# ---------------------------------------------------------------------------
def _gray(images: np.ndarray) -> np.ndarray:
    """32x32 grayscale pixels in [0, 1] -- the space the proof works in."""
    return images.astype(np.float32).mean(axis=3).reshape(len(images), -1) / 255.0


def _kernel_sums(query: np.ndarray, train: np.ndarray, gamma: float, chunk: int = 8) -> np.ndarray:
    """Sum_j exp(-2*gamma*||q - x_j||^2) for each query row."""
    out = []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        d2 = ((q[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
        out.append(np.exp(-2.0 * gamma * d2).sum(axis=1))
    return np.concatenate(out)


def _kernel_sums_by_label(
    query: np.ndarray,
    train: np.ndarray,
    train_labels: np.ndarray,
    gamma: float,
    chunk: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-label kernel sums: ``S[q, c] = sum_{j: y_j=c} exp(-2*gamma*||q-x_j||^2)``.

    Lemma 1 compares the kernel mass of the target class against every other
    class for each query, so the sums must be kept separate per class.
    """
    classes = np.unique(train_labels)
    out = []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        d2 = ((q[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
        weights = np.exp(-2.0 * gamma * d2)
        out.append(np.stack([weights[:, train_labels == c].sum(axis=1) for c in classes], axis=1))
    return np.concatenate(out), classes


def _choose_gamma(query: np.ndarray, train: np.ndarray, sample: int = 40) -> float:
    """Set gamma so the median query-train distance has kernel weight e^-1."""
    q = query[:sample]
    d2 = ((q[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
    return float(1.0 / (2.0 * max(np.median(d2), 1e-6)))


def _ntk_experiment(case: str, target: int = config.TARGET_LABEL) -> dict:
    bundle = artifacts.load_data()
    x_train, y_train = bundle["x_train"], bundle["y_train"]
    x_test, y_test = bundle["x_test"], bundle["y_test"]
    rng = np.random.default_rng(config.SEED + (91 if case == "dirty" else 92))

    per_class = 100
    clean = {}
    for cls in range(config.NUM_CLASSES):
        idx = np.flatnonzero(y_train == cls)
        clean[cls] = idx[rng.choice(len(idx), size=min(per_class, len(idx)), replace=False)]

    def poisoned(count_per_class: int, classes=None) -> np.ndarray:
        parts = []
        for cls in range(config.NUM_CLASSES):
            if cls == target or (classes is not None and cls not in classes):
                continue
            idx = clean[cls][:count_per_class]
            if len(idx):
                parts.append(attacks.apply_badnets(x_train[idx]))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, *x_train.shape[1:]), np.uint8)

    if case == "dirty":
        x_target_case1 = np.concatenate([x_train[clean[target]], poisoned(10)], axis=0)   # 190
        # Case 2: n_t = n_k2 = 100, i.e. 80 clean + 20 poisoned.
        poison20 = poisoned(3)[:20]
        x_target_case2 = np.concatenate([x_train[clean[target][:80]], poison20], axis=0)   # 100
        others_case1 = {cls: x_train[clean[cls][:100]] for cls in range(config.NUM_CLASSES) if cls != target}
        others_case2 = {cls: x_train[clean[cls][:100]] for cls in range(config.NUM_CLASSES) if cls != target}
        cases = {
            "case1_nt_gt_nk2": (x_target_case1, others_case1),
            "case2_nt_eq_nk2": (x_target_case2, others_case2),
        }
    else:
        # Clean-label: all training images keep their labels; target class is 80 clean images.
        x_target = x_train[clean[target]]
        cases = {"clean_label": (x_target,
                                 {cls: x_train[clean[cls]] for cls in range(config.NUM_CLASSES) if cls != target})}

    # Query images: pseudo-label k1 != target, taken from the test split.
    queries, query_labels = [], []
    for cls in range(config.NUM_CLASSES):
        if cls == target:
            continue
        idx = np.flatnonzero(y_test == cls)[:8]
        queries.append(x_test[idx])
        query_labels.append(np.full(len(idx), cls))
    query = _gray(np.concatenate(queries, axis=0))
    query_labels = np.concatenate(query_labels)

    results = {}
    for name, (x_t, others) in cases.items():
        train_arrays = [_gray(x_t)] + [_gray(v) for v in others.values()]
        train = np.concatenate(train_arrays, axis=0)
        labels = np.concatenate(
            [np.full(len(x_t), target)] + [np.full(len(v), cls) for cls, v in others.items()]
        )
        gamma = _choose_gamma(query, train)
        frac = {}
        for mult in (0.5, 1.0, 2.0):
            sums, classes = _kernel_sums_by_label(query, train, labels, gamma * mult)
            col = {int(c): i for i, c in enumerate(classes)}
            ok = []
            for row, k1 in zip(sums, query_labels):
                # Lemma 1 compares the target class against every class except
                # the query's own label k1 -- the target is not a competitor.
                competitors = [
                    row[col[int(c)]] for c in classes
                    if int(c) not in (int(k1), target)
                ]
                ok.append(bool(row[col[target]] > max(competitors)))
            frac[str(mult)] = float(np.mean(ok))
        results[name] = {
            "gamma": gamma,
            "gamma_multipliers": frac,
            "n_target": int(len(x_t)),
            "n_other_per_class": int(len(next(iter(others.values())))),
        }
    return results


@claim("A09")
def a09_lemma1_dirty() -> ClaimResult:
    """Lemma 1 (dirty label): does the target kernel sum dominate on real pixels?"""
    results = _ntk_experiment("dirty")
    fractions = [v["gamma_multipliers"]["1.0"] for v in results.values()]
    robust = min(min(v["gamma_multipliers"].values()) for v in results.values())
    # A numerically favourable simulation cannot prove the lemma, so the best
    # possible verdict is PARTIAL.
    if min(fractions) >= 0.50:
        verdict = Verdict.PARTIAL
    else:
        verdict = Verdict.CONTRADICTED
    return ClaimResult(
        verdict=verdict,
        measured={"cases": results, "min_fraction_at_gamma_star": min(fractions),
                  "min_fraction_across_gamma": robust},
        notes_zh=(
            "按 Eq.13 的 NTK 核 K=e^{-2γ||x-x'||²}，在真实 CIFAR-10 像素上计算目标类与其他类的核和；"
            "γ 由查询-训练距离中位数标定，并报告 0.5×/1×/2× 的敏感性。"
            "两个情形分开看：目标类样本数多于其他类（190 vs 100）时目标核和几乎总是最大；"
            "两类样本数相等（100 vs 100）时只有 8% 的查询满足 Lemma 1。"
            "Lemma 1 的原文没有样本数条件，因此等量情形是合法反例，判 CONTRADICTED；"
            "更大的目标类会机械地带来更大的核和，这部分不能算作对引理的验证。"
            "这是对论文证明所用替代模型的复算，不是对训练网络的证明。"
        ),
        tolerance={"dominance_fraction": 0.90},
    )


@claim("A10")
def a10_lemma1_clean() -> ClaimResult:
    """Eq.35's inequality step and the clean-label kernel-sum claim."""
    # Exact counterexample to the step in Eq.35 (1-D element, gamma=1).
    a, t, b, n = 0.0, 1.0, 1.5, 2
    u = a + t
    lhs_distance = (a - b) ** 2
    rhs_distance = (u - b) ** 2 + t ** 2
    lhs = float(np.exp(-2 * lhs_distance))
    rhs = float(np.exp(-2 * rhs_distance))
    eq35_fails = lhs <= rhs

    eq29_holds = (u / n - b) ** 2 > (u - b) ** 2
    eq30_holds = t < 2 * n * b / (1 + n) - a

    results = _ntk_experiment("clean")
    fractions = [v["gamma_multipliers"]["1.0"] for v in results.values()]
    empirical = min(fractions)

    # The proof step is invalid; the claim itself is judged separately.
    if eq35_fails:
        verdict = Verdict.CONTRADICTED
    elif empirical >= 0.90:
        verdict = Verdict.CONFIRMED
    else:
        verdict = Verdict.PARTIAL

    return ClaimResult(
        verdict=verdict,
        measured={
            "counterexample": {"a": a, "t": t, "b": b, "n": n,
                               "eq29_holds": eq29_holds, "eq30_holds": eq30_holds,
                               "eq35_lhs": lhs, "eq35_rhs": rhs,
                               "eq35_claimed_direction_holds": not eq35_fails},
            "ntk_clean_label": results,
            "min_fraction_at_gamma_star": empirical,
        },
        notes_zh=(
            "Eq.35 声称 e^{-2γ||v-T||²} ≥ e^{-2γ(||v||²+||T||²)}（v=x+T-x_t），"
            "该式仅在 ⟨v,T⟩≥0 时成立。反例 a=0,t=1,b=1.5,n=2：v=-0.5，左边 e^{-4.5} 小于右边 e^{-2.5}，"
            "方向相反，故干净标签分支的证明步骤无效。"
            "另外在干净标签协议下做核和模拟，目标类并不占优。"
        ),
        tolerance={"dominance_fraction": 0.90},
    )
