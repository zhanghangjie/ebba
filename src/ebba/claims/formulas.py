"""Tier A: exact formula and worked-example audits (A01-A06, A17)."""

from __future__ import annotations

import numpy as np

from .. import config
from ..energy import (
    clamp_max_logits,
    detect,
    energy_stats,
    label_energy,
    per_sample_normalized_energy,
    softmax,
)
from ..transferred import nte, recover_labels
from .model import ClaimResult, Verdict, claim


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@claim("A01")
def a01_appendix_b() -> ClaimResult:
    """Recompute Appendix B from the logits printed in Eq. (21)-(26)."""
    Z = _np([[9, 1, -1], [3, 8, -2], [-1, 3, 7]])
    s_exact = softmax(Z)
    e_exact = np.exp(s_exact).sum(axis=0)

    paper_s = _np([[0.99, 0.01, 0.0], [0.12, 0.88, 0.0], [0.002, 0.018, 0.98]])
    e_from_paper_s = np.exp(paper_s).sum(axis=0)
    paper_e = _np([4.82, 4.43, 4.62])

    Zc = clamp_max_logits(Z)
    sc_exact = softmax(Zc)
    ec_exact = np.exp(sc_exact).sum(axis=0)
    paper_sc = _np([[0.231, 0.66, 0.09], [0.94, 0.05, 0.01], [0.02, 0.94, 0.04]])
    ec_from_paper_s = np.exp(paper_sc).sum(axis=0)
    paper_ec = _np([4.83, 5.54, 3.14])

    s_err = float(np.abs(s_exact - paper_s).max())
    e_err = float(np.abs(e_exact - paper_e).max())
    sc_err = float(np.abs(sc_exact - paper_sc).max())
    ec_err = float(np.abs(ec_exact - paper_ec).max())

    paper_order = [0, 2, 1]          # E0 > E2 > E1
    exact_order = list(np.argsort(-e_exact))
    exact_order_from_paper_s = list(np.argsort(-e_from_paper_s))
    target = 1
    paper_target_is_lowest = bool(np.argmin(paper_e) == target)
    exact_target_is_lowest = bool(np.argmin(e_exact) == target)

    ok_original = (s_err <= 0.01) and (e_err <= 0.01) and (exact_order == paper_order)
    ok_clamped = (sc_err <= 0.01) and (ec_err <= 0.01)
    verdict = Verdict.CONFIRMED if (ok_original and ok_clamped) else Verdict.CONTRADICTED

    return ClaimResult(
        verdict=verdict,
        measured={
            "softmax_exact": s_exact,
            "softmax_paper": paper_s,
            "softmax_max_abs_error": s_err,
            "energy_exact": e_exact,
            "energy_paper": paper_e,
            "energy_max_abs_error": e_err,
            "energy_from_paper_softmax": e_from_paper_s,
            "order_paper": paper_order,
            "order_exact": exact_order,
            "order_exact_from_paper_softmax": exact_order_from_paper_s,
            "paper_target_is_lowest": paper_target_is_lowest,
            "exact_target_is_lowest": exact_target_is_lowest,
            "clamped_softmax_exact": sc_exact,
            "clamped_softmax_paper": paper_sc,
            "clamped_softmax_max_abs_error": sc_err,
            "clamped_energy_exact": ec_exact,
            "clamped_energy_paper": paper_ec,
            "clamped_energy_max_abs_error": ec_err,
            "clamped_order_exact": list(np.argsort(-ec_exact)),
        },
        notes_zh=(
            "按论文自己给出的 logits 精确计算 softmax，s2 应为 [0.0067, 0.9933, 0.0000]，"
            "而非 [0.12, 0.88, 0]；能量为 [4.7243, 4.7185, 4.6691]，而非 [4.82, 4.43, 4.62]。"
            "原文的顺序 E0>E2>E1 与“目标标签能量最低”的结论在其自身 logits 下均不成立"
            "（精确顺序为 E0>E1>E2）。clamp 后的示例方向正确，但 s* 与 E* 仍有 0.01-0.04 的误差。"
        ),
        tolerance={"abs_two_decimals": 0.01, "abs_three_decimals": 0.005},
    )


@claim("A02")
def a02_clamp_semantics() -> ClaimResult:
    """Confirm that Eq. (3) zeroes the max entry instead of subtracting it."""
    Z = _np([[9, 1, -1], [3, 8, -2], [-1, 3, 7]])
    checked = []
    for row in Z:
        m = float(row.max())
        expected = row.copy()
        expected[int(np.argmax(row))] = 0.0
        subtracted = row - m
        checked.append({
            "input": row.tolist(),
            "clamp": expected.tolist(),
            "subtract_max": subtracted.tolist(),
            "paper_example_matches_clamp": bool(np.allclose(expected, clamp_max_logits(row[None])[0])),
            "clamp_differs_from_subtract": bool(not np.allclose(expected, subtracted)),
        })
    all_clamp = all(c["paper_example_matches_clamp"] for c in checked)
    any_diff = any(c["clamp_differs_from_subtract"] for c in checked)
    return ClaimResult(
        verdict=Verdict.CONFIRMED if (all_clamp and any_diff) else Verdict.CONTRADICTED,
        measured={"rows": checked},
        notes_zh=(
            "论文 Eq.24 的示例 [3,8,-2]→[3,0,-2] 只能由“把最大值置 0”得到；"
            "减最大值会得到 [0,-5,-10]。因此 Eq.3 是 clamp，不是平移，且不保持 softmax 不变性。"
        ),
        tolerance={"abs_exact": 1e-12},
    )


@claim("A03")
def a03_normalization_variants() -> ClaimResult:
    """Compare Eq.5 with the official README and the official notebook."""
    rng = np.random.default_rng(config.SEED)
    # Representative logits: one confident class plus a mild target boost.
    logits = rng.normal(0, 1.0, size=(200, config.NUM_CLASSES))
    logits[:, 0] += 0.8
    logits[:, 1] += 0.3

    np_paper = per_sample_normalized_energy(logits, config.TEMPERATURE, "paper")
    np_readme = per_sample_normalized_energy(logits, config.TEMPERATURE, "readme")
    np_official = per_sample_normalized_energy(logits, config.TEMPERATURE, "official")
    pairwise = {
        "paper_vs_readme": float(np.abs(np_paper - np_readme).max()),
        "paper_vs_official": float(np.abs(np_paper - np_official).max()),
        "readme_vs_official": float(np.abs(np_readme - np_official).max()),
    }
    targets = {
        "paper": int(np.argmax(np_paper.sum(axis=0))),
        "readme": int(np.argmax(np_readme.sum(axis=0))),
        "official": int(np.argmax(np_official.sum(axis=0))),
    }
    agree = len(set(targets.values())) == 1
    consistent = max(pairwise.values()) <= 1e-6
    return ClaimResult(
        verdict=Verdict.CONFIRMED if consistent else Verdict.CONTRADICTED,
        measured={
            "pairwise_max_abs_difference": pairwise,
            "detected_target": targets,
            "targets_agree": agree,
            "paper_formula": "exp(s*T)/T - 1/T (T=2)",
            "readme_formula": "P_k - 1",
            "official_formula": "min(exp(s*T)/T, 2) - 0.6",
        },
        notes_zh=(
            "论文 Eq.5、官方 README（CIFAR-10 用 P_k−1）与官方 Notebook（截断 2 再减 0.6）"
            "是三种不同归一化，不能同时成立。结论稳健性单独报告：三者选出的目标标签是否一致。"
        ),
        tolerance={"abs_exact": 1e-6},
    )


@claim("A04")
def a04_indexing() -> ClaimResult:
    """Check Eq.7's divisor against the K+1 terms used in Eq.2/6."""
    energy = _np([10, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    K = len(energy)
    mu_k = energy.mean()
    sig_k = energy.std(ddof=0)
    z_k = float((energy.max() - mu_k) / sig_k)

    # Literal reading: k = 0..K means K+1 terms.  There is no label K, so the
    # only coherent reading is the K observed labels; the audit records the
    # numerical consequence of the alternative (an extra zero term).
    extended = np.concatenate([energy, [0.0]])
    mu_ext = extended.mean()
    sig_ext = extended.std(ddof=0)
    z_ext = float((extended.max() - mu_ext) / sig_ext)

    consistent = abs(z_k - z_ext) <= 0.05
    return ClaimResult(
        verdict=Verdict.CONFIRMED if consistent else Verdict.CONTRADICTED,
        measured={
            "energy": energy,
            "K": K,
            "zscore_K_terms_divisor_K": z_k,
            "zscore_K_plus_1_terms_divisor_K_plus_1": z_ext,
            "max_zscore_shift": abs(z_k - z_ext),
        },
        notes_zh=(
            "Eq.2/6 写作 k∈[0,K]（K+1 项），Eq.7 却除以 K。K 个真实标签时应使用 K 项、除数 K；"
            "若按字面补一项，最大 z-score 会明显变化。实现采用 K 项。"
        ),
        tolerance={"max_zscore_shift": 0.05},
    )


@claim("A05")
def a05_nte_identity() -> ClaimResult:
    """Verify TE = exp(s'-s) and NTE = TE-1, plus the mean-zero property."""
    rng = np.random.default_rng(config.SEED + 1)
    logits = rng.normal(0, 1.5, size=(64, config.NUM_CLASSES))
    corrupt = logits[None, :, :] + rng.normal(0, 0.5, size=(5, 64, config.NUM_CLASSES))

    s = softmax(logits)
    sc = softmax(corrupt.reshape(-1, config.NUM_CLASSES)).reshape(corrupt.shape)
    te = np.exp(sc - s[None])
    nte_literal = te - 1.0

    values = nte(logits, corrupt, temperature=1.0, cap=None)
    # The aggregated value must equal the sum of per-corruption literal NTE.
    expected = nte_literal.sum(axis=0)
    max_err = float(np.abs(values - expected).max())

    mean_diff = (sc - s[None]).mean(axis=2)
    max_mean = float(np.abs(mean_diff).max())
    sums = np.abs(s.sum(axis=1) - 1).max()

    ok = max_err <= 1e-9 and max_mean <= 1e-9 and sums <= 1e-9
    return ClaimResult(
        verdict=Verdict.CONFIRMED if ok else Verdict.CONTRADICTED,
        measured={
            "max_identity_error": max_err,
            "max_abs_mean_difference_over_labels": max_mean,
            "max_softmax_sum_error": float(sums),
        },
        notes_zh=(
            "TE=exp(s'-s) 与 NTE=TE−1 在数值上成立；由于 softmax 每行和为 1，"
            "逐样本的 mean_k(s'_k−s_k)=0 严格成立，论文用 e^0=1 作基准是合理的。"
        ),
        tolerance={"abs_exact": 1e-9},
    )


@claim("A06")
def a06_temperature_consistency() -> ClaimResult:
    """Eq.4 uses T, Eq.9/10 silently assume T=1.  Quantify the discrepancy."""
    rng = np.random.default_rng(config.SEED + 2)
    logits = rng.normal(0, 1.5, size=(128, config.NUM_CLASSES))
    corrupt = logits[None] + rng.normal(0, 0.8, size=(6, 128, config.NUM_CLASSES))

    pred_t1 = recover_labels(nte(logits, corrupt, temperature=1.0, cap=5.0))
    pred_t2 = recover_labels(nte(logits, corrupt, temperature=config.TEMPERATURE, cap=5.0))
    agree = float((pred_t1 == pred_t2).mean())
    consistent = agree >= 0.99
    return ClaimResult(
        verdict=Verdict.CONFIRMED if consistent else Verdict.CONTRADICTED,
        measured={
            "argmax_agreement_t1_vs_t2": agree,
            "temperature_paper": config.TEMPERATURE,
        },
        notes_zh=(
            "Eq.4 的能量是 exp(s·T)/T，而 Eq.9/10 的比值把 T 消掉了（等价于 T=1）。"
            "形式不一致；实际影响用 argmax 一致率衡量。"
        ),
        tolerance={"argmax_agreement": 0.99},
    )


@claim("A20")
def a20_lambda3_bound() -> ClaimResult:
    """Eq.8 with lambda=3 on a K=10 model sits exactly at the z-score bound.

    For any K-vector, max_k (E_k - mu) / sigma <= sqrt(K - 1).  With K=10 the
    supremum is exactly 3, reached only when the other nine energies are equal.
    Eq.8 requires a *strict* margin, so on CIFAR-10 the rule `> 3 sigma` cannot
    fire for any realistic energy vector.  The paper's lambda was calibrated on
    GTSRB (K=43, bound 6.48), then reused for CIFAR-10.
    """
    K = config.NUM_CLASSES
    lam = config.LAMBDA
    bound = float(np.sqrt(K - 1))

    def z_max(vec: np.ndarray) -> float:
        st = energy_stats(np.asarray(vec, dtype=np.float64))
        return float(st.deviation.max() / st.std) if st.std > 0 else 0.0

    # Extremal construction: one outlier, the remaining K-1 labels identical.
    extremal = np.ones(K)
    extremal[0] = 2.0
    z_extremal = z_max(extremal)

    # Any spread among the non-target labels breaks the equality and lowers z.
    rng = np.random.default_rng(config.SEED + 20)
    worst = 0.0
    for eps in (1e-6, 1e-3, 1e-2, 0.1, 1.0):
        vec = np.ones(K) + rng.normal(0, eps, K)
        vec[0] = 2.0 + rng.uniform(0, 50)
        worst = max(worst, z_max(vec))

    # Literal reading of Eq.7 (K+1 terms, k=0..K) would append a phantom label.
    eleven = np.concatenate([np.full(K - 1, 10.0), [100.0], [0.0]])
    z_eleven = z_max(eleven)

    detectable = z_extremal > lam or worst > lam
    return ClaimResult(
        verdict=Verdict.CONTRADICTED if not detectable else Verdict.CONFIRMED,
        measured={
            "K": K,
            "lambda": lam,
            "supremum_z": bound,
            "z_extremal_equal_others": z_extremal,
            "z_max_random_perturbations": worst,
            "z_with_phantom_K_plus_1": z_eleven,
            "strict_margin_possible_K10": detectable,
        },
        notes_zh=(
            "对任意 K 维能量向量，max_k (E_k−μ)/σ ≤ √(K−1)。K=10 时上界恰为 3："
            "只有其余 9 个标签能量完全相等才能取到，而 Eq.8 要求严格大于 λσ。"
            "因此论文在 CIFAR-10 上取 λ=3 时，检测裕量在数学上几乎不可能为正；"
            "其 λ 的可用区间 [1.8, 5.7] 与 K=43 的 GTSRB（上界 6.48）一致，说明 λ=3 是在 GTSRB 上标定后"
            "直接用于 CIFAR-10 的。若按 Eq.7 字面把求和写成 K+1 项，上界升到 √10≈3.16，λ=3 才勉强可达。"
        ),
        tolerance={"lambda": lam, "max_zscore": bound},
    )


@claim("A17")
def a17_fig80_sign() -> ClaimResult:
    """Check the textual direction of Fig.80 against Eq.8."""
    text_first = "If this value is lower than 0, the model is considered as backdoored."
    text_second = "the value for the clean model is less than 0, while the value for the poisoned model is greater than 0"
    eq8 = "E_k' - mu > lambda * sigma  =>  backdoored"
    inconsistent = True  # first sentence says '<0 => backdoored'; Eq.8 and second sentence say '>0'
    return ClaimResult(
        verdict=Verdict.CONTRADICTED if inconsistent else Verdict.CONFIRMED,
        measured={
            "first_sentence": text_first,
            "second_sentence": text_second,
            "eq8_direction": eq8,
            "first_vs_eq8": "opposite",
            "second_vs_eq8": "consistent",
        },
        notes_zh=(
            "Appendix F 第一句称“值小于 0 判为后门”，随后又说 clean<0、poisoned>0，"
            "与 Eq.8（大于 λσ 判为后门）矛盾。应以 Eq.8 为准，第一句是笔误。"
        ),
        tolerance={"textual_consistency": True},
    )
