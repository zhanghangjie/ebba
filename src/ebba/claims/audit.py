"""Tier A: arithmetic and textual audits of the paper's tables (A11-A16, A18-A19).

No model training is involved.  Every check is a deterministic calculation on
the transcribed numbers in ``tables.py`` or on the source of the official
reference notebook (parsed as JSON, never executed).
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config
from . import tables
from .model import ClaimResult, Verdict, claim


# ---------------------------------------------------------------------------
# A11 -- BA lower bound
# ---------------------------------------------------------------------------
@claim("A11")
def a11_ba_bound() -> ClaimResult:
    """Check EBBA+ BA >= (K-1)/K * ba for every Table 4 row."""
    rows = []
    for row in tables.table4_rows():
        dataset = row["dataset"]
        k = tables.NUM_CLASSES_BY_DATASET[dataset]
        ba = row["No Defense_ba"]
        bound = (k - 1) / k * ba
        observed = row["EBBA+_ba"]
        margin = observed - bound
        rows.append({
            "dataset": dataset,
            "attack": row["attack"],
            "K": k,
            "ba": ba,
            "bound": bound,
            "ebba_ba": observed,
            "margin": margin,
            "violates": bool(margin < -0.005),
        })
    violations = [r for r in rows if r["violates"]]
    return ClaimResult(
        verdict=Verdict.CONTRADICTED if violations else Verdict.CONFIRMED,
        measured={"rows": rows, "n_violations": len(violations)},
        notes_zh=(
            "按 Sec. 5.2 的下界 (K−1)/K×ba 逐行核对 Table 4。"
            f"15 行中 {len(violations)} 行低于下界（容差 0.005），"
            "说明该式是经验观察而非可证定理，或表中数字有舍入/实验偏差。"
        ),
        tolerance={"abs_margin": 0.005},
    )


# ---------------------------------------------------------------------------
# A12 -- ASR training threshold
# ---------------------------------------------------------------------------
@claim("A12")
def a12_asr_threshold() -> ClaimResult:
    """Check the claim that every attack reaches ASR >= 98.8%."""
    rows = []
    for row in tables.table4_rows():
        asr = row["No Defense_asr"]
        rows.append({
            "dataset": row["dataset"], "attack": row["attack"],
            "no_defense_asr": asr, "passes": bool(asr >= 98.8),
        })
    failures = [r for r in rows if not r["passes"]]
    return ClaimResult(
        verdict=Verdict.CONTRADICTED if failures else Verdict.CONFIRMED,
        measured={"rows": rows, "n_failures": len(failures),
                  "min_asr": min(r["no_defense_asr"] for r in rows)},
        notes_zh=(
            "论文 Sec. 5.1 称所有攻击均训练到 ASR 98.8% 以上。Table 4 的 No Defense 列中"
            f"有 {len(failures)} 行低于 98.8%（ImageNet Blend 98.62、ImageNet FIBA 98.59），"
            "属于可直接核对的数值反例。"
        ),
        tolerance={"threshold": 98.8},
    )


# ---------------------------------------------------------------------------
# A13 -- GTSRB test size
# ---------------------------------------------------------------------------
@claim("A13")
def a13_gtsrb_size() -> ClaimResult:
    """Table 6's GTSRB test size versus the standard 12,630 images."""
    stated = tables.TABLE6["Gtsrb"]["test"]
    standard = tables.STANDARD_TEST_SIZES["Gtsrb"]
    train_stated = tables.TABLE6["Gtsrb"]["train"]
    train_standard = 39209
    return ClaimResult(
        verdict=Verdict.CONFIRMED if stated == standard else Verdict.CONTRADICTED,
        measured={
            "stated_test": stated,
            "standard_test": standard,
            "difference": stated - standard,
            "stated_train": train_stated,
            "standard_train": train_standard,
            "train_matches": train_stated == train_standard,
            "looks_like_transposition": stated == int(str(standard)[::-1]),
        },
        notes_zh=(
            "GTSRB 测试集的标准大小是 12630，论文 Table 6 写成 12603（数字转置）。"
            "训练集 39209 与标准值一致，因此这不是“使用了子集”，而是笔误。"
        ),
        tolerance={"exact_size": 12630},
    )


# ---------------------------------------------------------------------------
# A14 -- clustering tables
# ---------------------------------------------------------------------------
@claim("A14")
def a14_clustering_tables() -> ClaimResult:
    """Check the PDR spread across clustering methods and Table 5 == Table 7."""
    values5 = list(tables.TABLE5.values())
    spread5 = max(values5) - min(values5)
    identical = tables.TABLE5 == tables.TABLE7
    ok = spread5 <= 0.05
    return ClaimResult(
        verdict=Verdict.CONFIRMED if ok else Verdict.CONTRADICTED,
        measured={
            "table5": tables.TABLE5,
            "table7": tables.TABLE7,
            "tables_identical": identical,
            "max_spread": spread5,
            "best_method": max(tables.TABLE5, key=tables.TABLE5.get),
        },
        notes_zh=(
            "Table 5 与 Table 7（正文与附录）数值完全相同，均为 HC 0.972、Birch 0.968、"
            "Mean Shift 0.965、DBSCAN 0.975。极差 0.010，支持“聚类方法影响很小”的说法；"
            "但同一组数字在两处重复出现，不构成独立证据。"
        ),
        tolerance={"max_pdr_spread": 0.05},
    )


# ---------------------------------------------------------------------------
# A15 -- textual claims about Table 4
# ---------------------------------------------------------------------------
@claim("A15")
def a15_table4_text() -> ClaimResult:
    """Check 'BA is best' and 'ASR is lower' against every Table 4 row."""
    rows = []
    ba_best = 0
    asr_best = 0
    asr_below_no_defense = 0
    for row in tables.table4_rows():
        # "No Defense" is the undefended reference, not a defense baseline.
        ba_methods = [m for m in tables.TABLE4_METHODS if m not in ("EBBA+", "No Defense")]
        asr_methods = [m for m in tables.TABLE4_METHODS if m != "EBBA+"]
        ba_others = [row[f"{m}_ba"] for m in ba_methods]
        asr_others = [row[f"{m}_asr"] for m in asr_methods]
        is_ba_best = row["EBBA+_ba"] > max(ba_others)
        is_asr_best = row["EBBA+_asr"] < min(asr_others)
        ba_best += int(is_ba_best)
        asr_best += int(is_asr_best)
        asr_below_no_defense += int(row["EBBA+_asr"] < row["No Defense_asr"])
        rows.append({
            "dataset": row["dataset"], "attack": row["attack"],
            "ebba_ba": row["EBBA+_ba"], "best_other_ba": max(ba_others),
            "ba_best": is_ba_best,
            "ebba_asr": row["EBBA+_asr"], "best_other_asr": min(asr_others),
            "asr_lowest": is_asr_best,
        })
    ok = ba_best >= 15 and asr_best >= 15
    return ClaimResult(
        verdict=Verdict.CONFIRMED if ok else Verdict.CONTRADICTED,
        measured={
            "rows": rows,
            "n_rows": len(rows),
            "ba_best_rows": ba_best,
            "asr_lowest_rows": asr_best,
            "asr_below_no_defense_rows": asr_below_no_defense,
        },
        notes_zh=(
            "Table 4 标题称 EBBA+ 的 BA 在所有方法中最好、ASR 与之相当。"
            f"逐行核对：BA 严格最高 {ba_best}/15 行，ASR 严格最低 {asr_best}/15 行。"
            "论文正文的措辞是“lower or competitive”，并非全面最优；"
            "标题中的“best”对 2 行（Cifar10 Blend、Cifar10 DUBA）不成立。"
        ),
        tolerance={"min_best_ba_rows": 15, "min_best_asr_rows": 15},
    )


# ---------------------------------------------------------------------------
# A16 -- Table 3 text
# ---------------------------------------------------------------------------
@claim("A16")
def a16_table3_text() -> ClaimResult:
    """Check 'outperforms all other methods in PDR and F1'."""
    rows = []
    violations = []
    for row in tables.table3_rows():
        best_pdr = max(row[f"{m}_pdr"] for m in ("FTD", "TeCo", "SCALE"))
        best_f1 = max(row[f"{m}_f1"] for m in ("FTD", "TeCo", "SCALE"))
        pdr_worse = row["EBBA+_pdr"] < best_pdr
        f1_worse = row["EBBA+_f1"] < best_f1
        rec = {
            "dataset": row["dataset"], "attack": row["attack"],
            "ebba_pdr": row["EBBA+_pdr"], "best_other_pdr": best_pdr,
            "ebba_f1": row["EBBA+_f1"], "best_other_f1": best_f1,
            "pdr_worse": pdr_worse, "f1_worse": f1_worse,
        }
        rows.append(rec)
        if pdr_worse or f1_worse:
            violations.append(rec)
    return ClaimResult(
        verdict=Verdict.CONTRADICTED if violations else Verdict.CONFIRMED,
        measured={"rows": rows, "n_violations": len(violations), "violations": violations},
        notes_zh=(
            "Table 3 标题称 EBBA+ 在 PDR 与 F1 上优于所有对比方法。"
            f"15 行中有 {len(violations)} 行在至少一个指标上严格劣于最好的基线"
            "（Cifar10 BadNets 的 PDR、Cifar10 Blend 的 F1、Cifar10 FIBA 的 PDR、"
            "GTSRB WaNet 的 PDR、ImageNet BadNets 的 F1），故“全面最优”不成立。"
            "另有若干平局，未计入反例。"
        ),
        tolerance={"max_violations": 0},
    )


# ---------------------------------------------------------------------------
# A18 -- static audit of the official notebook
# ---------------------------------------------------------------------------
def _notebook_text(path: Path) -> tuple[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") == "code":
            cells.append("".join(cell.get("source", [])))
    return "\n".join(cells), cells


def _reference_notebook_path() -> Path:
    candidates = [
        Path(config.PROJECT_ROOT) / "reference" / "EBBA.ipynb",
        Path(config.PROJECT_ROOT) / "paper" / "repo" / "EBBA.ipynb",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("official EBBA.ipynb not found (reference/EBBA.ipynb)")


@claim("A18")
def a18_official_notebook() -> ClaimResult:
    """Eight protocol checks against the paper's stated setup."""
    path = _reference_notebook_path()
    text, cells = _notebook_text(path)

    uses_gtsrb_pool = ("GTSRB" in text) and ("val_dataset" in text)
    has_balancing = any(key in text.lower() for key in ("balance", "per_class", "pseudo_label"))
    samples_704 = ("range(11)" in text) and ("range(64)" in text)
    corruptions_64 = ("range(15)" in text) and ("range(1,5)" in text)
    divide_by_one = ("data/j" in text) and ("range(1,5)" in text)
    caps_5 = "transfer > 5" in text
    caps_4 = "transfer > 4" in text
    has_clustering = any(key in text for key in ("Birch", "birch", "KMeans", "cluster"))
    # The notebook evaluates patched validation images (list_test -> label 20)
    # and clean validation images separately, so BA/ASR exist under other names.
    reports_ba_asr = ("list_label_test.append([20])" in text) and ("list_label_val.append(target)" in text)

    checks = [
        {"id": "task_agnostic_pool", "paper": "Internet images, task-agnostic",
         "official": "GTSRB Final_Validation used as the image pool", "passes": not uses_gtsrb_pool},
        {"id": "per_class_balancing", "paper": "keep N images per pseudo-label",
         "official": "no balancing code found", "passes": has_balancing},
        {"id": "sample_count", "paper": "task-agnostic subset (paper does not fix a small count)",
         "official": "11 batches x 64 = 704 images", "passes": (not samples_704)},
        {"id": "corruption_count", "paper": "80 corruption methods",
         "official": "15 types x 4 severities + 4 divisions = 64", "passes": not corruptions_64},
        {"id": "divide_by_one", "paper": "division by positive integers > 1",
         "official": "division loop starts at j=1 (identity)", "passes": not divide_by_one},
        {"id": "te_cap_consistency", "paper": "a single definition, Eq. 9-10",
         "official": "cap 5 in two helpers, cap 4 in another", "passes": not (caps_5 and caps_4)},
        {"id": "clustering", "paper": "t-SNE + Birch split poisoned/clean",
         "official": "clustering step absent", "passes": has_clustering},
        {"id": "ba_asr_evaluation", "paper": "BA and ASR reported",
         "official": "clean and patched validation sets evaluated separately", "passes": reports_ba_asr},
    ]
    failed = [c for c in checks if not c["passes"]]
    return ClaimResult(
        verdict=Verdict.CONTRADICTED if failed else Verdict.CONFIRMED,
        measured={
            "notebook": str(path),
            "n_code_cells": len(cells),
            "checks": checks,
            "n_failed": len(failed),
            "protocol_checks": len(checks),
        },
        notes_zh=(
            "对官方 PaddlePaddle Notebook 做静态解析（不执行）。8 项协议检查中"
            f"{len(failed)} 项与论文不符：使用 GTSRB 验证集（任务相关）、无每类均衡、"
            "仅 704 张图、64 种损坏（含除以 1）、TE 截断 5/4 不一致、无聚类步骤。"
            "它能作为公式与流程的交叉参考，但不是论文实验的完整实现。"
        ),
        tolerance={"protocol_checks": 8},
    )


# ---------------------------------------------------------------------------
# A19 -- adaptive attack claim
# ---------------------------------------------------------------------------
@claim("A19")
def a19_adaptive_rate() -> ClaimResult:
    """Thirteen soft-label settings; the paper itself names one failure."""
    values = [0.9, 0.6, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
    failing = [0.985]
    pass_rate = (len(values) - len(failing)) / len(values)
    return ClaimResult(
        verdict=Verdict.PARTIAL if pass_rate < 0.99 else Verdict.CONFIRMED,
        measured={
            "settings": values,
            "n_settings": len(values),
            "failing_settings": failing,
            "setting_pass_rate": pass_rate,
            "paper_statement": "only 0.985 makes EBBA confused; detection rate over 99%",
        },
        notes_zh=(
            "Appendix G.4 列出 13 个软标签参数，并明确承认 0.985 时 EBBA 失效。"
            f"按参数组计算通过率为 12/13 = {pass_rate:.1%}，低于 99%。"
            "若“检测率”指逐样本比例则无法从文中数值核实；因此该断言只能记 PARTIAL。"
        ),
        tolerance={"min_detection_rate": 0.99},
    )
