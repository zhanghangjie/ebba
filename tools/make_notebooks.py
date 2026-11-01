"""Generate the seven verification notebooks as plain nbformat-4 JSON.

Run from the repository root:

    python tools/make_notebooks.py

The notebooks contain explanation and orchestration only; every number comes
from ``ebba.claims.run``.  Keeping generation in a script makes the Chinese
commentary reviewable in a normal diff instead of hidden in ipynb JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

PREAMBLE = r'''import json
import os
import sys
from pathlib import Path

# 定位仓库根目录：容器内是 /app，本地打开时可能是 notebooks/ 或其子目录。
ROOT = Path(os.environ.get("EBBA_ROOT", Path.cwd())).resolve()
if not (ROOT / "src" / "ebba").exists():
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "src" / "ebba").exists():
            ROOT = cand
            break
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("EBBA_ROOT", str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from ebba import config, nbutil
from ebba.claims import load_registry, run, run_many

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print("project root:", ROOT)
print("artifacts   :", config.ARTIFACTS_DIR)
print("reports     :", config.REPORTS_DIR)
print("fast mode   :", config.FAST)'''


def md(text: str):
    return ("markdown", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


def _source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def build(cells) -> dict:
    out = []
    for kind, text in cells:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": _source(text),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.11.10",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS: dict[str, list] = {}

# ---------------------------------------------------------------------------
# 00 -- overview
# ---------------------------------------------------------------------------
NOTEBOOKS["00_overview"] = [
    md(r"""
# EBBA / EBBA+ 可复现验证（ICML 2024）

**论文**：Yudong Gao, Honglong Chen, Peng Sun, Zhaojie Li, Jilong Li, Shaojie Shao.
*Energy-based Backdoor Defense without Task-Specific Samples and Model Retraining*.
ICML 2024, PMLR 235. 本地文件：`paper/paper.pdf`。

这套 Notebook 的目的不是“跑一遍作者的实验”，而是把论文中**可以被独立核对**的断言逐条变成代码，
给出可复核的判定。全部数值由 `src/ebba` 计算并写入 `reports/claims/*.json`；
Notebook 只负责解释与展示，因此同样的检查可以脱离 Jupyter 用 `pytest` 运行。

## 论文主张的两件事

**EBBA（检测）**：不给任务相关样本，只拿一批任务无关图像（论文用互联网图片），
按模型伪标签每类取 N 张；把每条 logits 的最大项置 0（Eq.3），softmax 后逐标签求能量（Eq.4-6），
再对标签能量做 z-score（Eq.7）。若某个标签满足

$$E_k' - \mu > \lambda \sigma \tag{Eq.8}$$

则判定模型被植入后门，且该标签就是目标标签。

**EBBA+（样本检测 + 去除）**：对每张图施加 J 种损坏，比较损坏前后同一标签的 softmax 概率：

$$\mathrm{TE}_k^j = \frac{e^{s_k'}}{e^{s_k}} = e^{s_k' - s_k},\qquad
\mathrm{NTE}_k^j = e^{s_k' - s_k} - 1,\qquad
\mathrm{NTE}_k = \sum_{j=1}^{J}\mathrm{NTE}_k^j. \tag{Eq.9-11}$$

被模型判为目标类的样本中，`argmax_k NTE_k` 被认为是它的原始标签（Eq.12）；
先用 t-SNE 降维再用 Birch 聚类，把“目标类的干净图”和“中毒图”分开，重标后者即可去除后门，无需重训练。

## 判定语义

| 判定 | 含义 |
|---|---|
| `CONFIRMED` | 所有核心子检查在容差内通过。只有 Tier A 能获得此判定。 |
| `PARTIAL` | 部分通过；或缩比实验方向一致但规模不同。**`scope: proxy` 永不判 CONFIRMED。** |
| `CONTRADICTED` | 可直接核对的数值超差，或普适断言存在反例。 |
| `NOT_REPRODUCIBLE` | 需要容器外资源（Tier C）。 |
| `ERROR` | 运行异常。正常交付物中不允许出现。 |

## 三层验证

- **Tier A（精确层）**：附录算例、公式一致性、表格算术、文字断言、证明步骤的代数检查、官方代码静态审计。
- **Tier B（缩比实证层）**：CIFAR-10 子集 + SmallCNN + BadNets/Blend，检验方法机制是否按论文方向工作。
  所有结论明确标注为 proxy，不与论文表格逐位比较。
- **Tier C（不可复现层）**：ResNet18/ImageNet/语音文本/13 组自适应训练等，说明原因与所需资源。
"""),
    code(PREAMBLE),
    md(r"""
## 环境指纹

先记录 Python、NumPy、PyTorch、设备与路径。出现跨机器数值差异时，这张指纹是排查的起点；
`artifacts/manifest.json` 还记录了三个 checkpoint 的 SHA256 与训练指标。
"""),
    code(r"""
nbutil.env_fingerprint()"""),
    md(r"""
## 声明索引

`src/ebba/claims/registry.yaml` 是唯一事实源：每条声明登记 ID、层级、范围、论文位置、容差和所属 Notebook。
运行结果写入 `reports/claims/<ID>.json`。
"""),
    code(r"""
registry = load_registry()
from collections import Counter

print("claim count:", len(registry))
print("by tier   :", dict(sorted(Counter(v["tier"] for v in registry.values()).items())))
print("by scope  :", dict(Counter(v["scope"] for v in registry.values())))
print()
for cid, spec in registry.items():
    print(f'{cid:>4}  T{spec["tier"]}  {spec["scope"]:<6}  {spec["notebook"]:<18}  {spec["title_zh"]}')"""),
    md(r"""
## 代理实验的产物与门槛

镜像构建阶段会训练三个 SmallCNN（clean / BadNets / Blend），并把数据子集、checkpoint 和指标写入
`artifacts/manifest.json`。构建门槛检验的是**代理实验台是否可用**（模型精度、攻击成功率、干净模型不被误报），
而不是论文结论本身；NTE 恢复率等属于被检验的结果，记录但不设门槛，避免为通过构建而调参。
"""),
    code(r"""
manifest_path = config.ARTIFACTS_DIR / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest["metrics"], indent=2, ensure_ascii=False))
    print("\nprotocol:", json.dumps(manifest["protocol"], ensure_ascii=False))
    print("checkpoint sha256:")
    for rel, digest in manifest["sha256"].items():
        if rel.startswith("checkpoints/"):
            print(f"  {rel}: {digest[:16]}...")
else:
    print("未找到 artifacts/manifest.json。请先运行：python scripts/build_artifacts.py --device cpu")"""),
    md(r"""
## 如何运行

```bash
# 容器内（推荐）
docker compose run --rm ebba verify          # 约 4 分钟，产出 reports/report.html
docker compose run --rm ebba test            # pytest
docker compose run --rm ebba retrain --attack badnets

# 本地
python scripts/build_artifacts.py --device cpu
python scripts/run_papermill.py
python scripts/make_report.py
```

Notebook 顺序：`01_formula_audit` → `02_theory` → `03_table_audit` → `04_detection` → `05_removal` → `06_verdicts`。
每个 Notebook 可单独运行，但 `06_verdicts` 需要前五个已经写出结果。

## 重要限制（先读）

1. 缩比实验使用 **CIFAR-100 测试集代理“任务无关互联网图像”**，共享自然图像域，但并不是论文的数据来源。
2. 只训练 **SmallCNN、30 epochs、两种攻击**；不覆盖 ResNet18、200 epochs、WaNet/FIBA/DUBA。
   训练使用了轻度的噪声/模糊/对比度增广，这是为了补偿缩比模型远低于论文 ResNet18 的损坏鲁棒性。
3. 因此 Tier B 的 `PARTIAL` 只表示“机制方向在缩比设定下成立”，不能引用为论文 Table 3/4 的数值复现。
4. `CONTRADICTED` 针对的是**某条可核对断言**，不代表 EBBA 方法整体无效。
"""),
]

# ---------------------------------------------------------------------------
# 01 -- formula audit
# ---------------------------------------------------------------------------
NOTEBOOKS["01_formula_audit"] = [
    md(r"""
# 01 公式与算例审计（Tier A）

本 Notebook 检查论文第 5-6 页的 Eq.3-10 与附录 B 的算例。这些问题不需要训练模型，
是整份验证中最“硬”的部分：论文自己给出了输入和输出，我们只需按定义复算。

> 记号：$s$ 是原始 logits，$s^*$ 是 Eq.3 处理后的向量，$p=\mathrm{softmax}(s^*)$，
> $E_k=\sum_i \mathrm{NP}_{i,k}$ 是标签 $k$ 的能量。

$$s^*_k=\begin{cases}0,& k=\arg\max_m s_m\\ s_k,&\text{otherwise}\end{cases}\tag{Eq.3}$$

$$P_{i,k}=\frac{e^{s^*_{i,k}T}}{T},\qquad \mathrm{NP}_{i,k}=P_{i,k}-P_{i,0}\tag{Eq.4-5}$$

$$E_k=\sum_{i=1}^{N}\mathrm{NP}_{i,k},\qquad \mu=\frac{1}{K}\sum_k E_k,\qquad
\sigma^2=\frac{1}{K}\sum_k(E_k-\mu)^2\tag{Eq.6-7}$$
"""),
    code(PREAMBLE),
    md(r"""
## A02：Eq.3 究竟是“置 0”还是“减最大值”？

这两个操作在 softmax 之前并不等价：减最大值是平移不变性所需的操作，置 0 不是。
论文附录 B 的示例 `[3, 8, -2] → [3, 0, -2]` 明确支持“置 0”。代码必须按字面实现，
否则后续所有能量都会与论文不同。
"""),
    code(r"""
import numpy as np
from ebba.energy import clamp_max_logits, softmax

z = np.array([[3.0, 8.0, -2.0]])
print("输入          :", z[0])
print("Eq.3 置最大项0:", clamp_max_logits(z)[0])
print("减最大值      :", (z - z.max(axis=1, keepdims=True))[0])
print("softmax(置0)  :", softmax(clamp_max_logits(z))[0])
print("softmax(减max):", softmax(z - z.max(axis=1, keepdims=True))[0])

r = nbutil.show("A02")"""),
    md(r"""
## A01：附录 B 的 softmax 与能量能否复算？

附录 B 给出三个样本、三个类别的 logits，并印出了 softmax 与能量。我们按论文自己的公式精确计算，
再与论文印刷值比较：

| 量 | 论文 | 精确复算 |
|---|---|---|
| $s$ 第 2 行 | `[0.12, 0.88, 0.00]` | `[0.0067, 0.9933, 0.0000]` |
| $E$（未 clamp） | `[4.82, 4.43, 4.62]` | `[4.7243, 4.7185, 4.6691]` |
| 顺序 | $E_0>E_2>E_1$ | $E_0>E_1>E_2$ |
| clamp 后 $s^*$ 第 1 行 | `[0.231, 0.66, 0.09]` | `[0.2447, 0.6652, 0.0900]` |
| clamp 后 $E$ | `[4.83, 5.54, 3.14]` | `[4.8712, 5.5436, 3.1483]` |

这里有一个容易误读的点：论文的**结论方向**（clamp 后目标类能量 $E_1^*$ 最高，因而满足 Eq.8）在复算下仍然成立；
错的是未 clamp 那张表的数值和排序。判定因此是 `CONTRADICTED`，但它否定的是“算例可复算”这一断言，
不是 EBBA 的检测机制。
"""),
    code(r"""
r = nbutil.show("A01")
print("\n逐步复算（未 clamp）：")
from ebba.claims.formulas import _np
Z = _np([[9, 1, -1], [3, 8, -2], [-1, 3, 7]])
print("softmax:\n", np.round(softmax(Z), 4))
print("E = sum_i exp(s_i):", np.round(np.exp(softmax(Z)).sum(axis=0), 4))
print("论文印刷 E       :", r.measured["energy_paper"])"""),
    md(r"""
## A03：三套归一化互相矛盾

论文 Eq.5 定义 $\mathrm{NP}=e^{s^*T}/T - e^{0}/T$（$T=2$ 时为 $e^{2s^*}/2-0.5$）。
官方 README 却说 CIFAR-10 上用 `NP_k = P_k - 1`；官方 Notebook 又是 `min(exp(2s)/2, 2) - 0.6`。
三者不能同时成立。下面比较同一批 logits 上的数值差异，并检查它们是否改变检测出的目标标签。
"""),
    code(r"""
import numpy as np
from ebba.energy import per_sample_normalized_energy

rng = np.random.default_rng(2024)
logits = rng.normal(0, 1.0, size=(200, 10))
logits[:, 0] += 0.8
logits[:, 1] += 0.3

variants = {v: per_sample_normalized_energy(logits, 2.0, v) for v in ("paper", "readme", "official")}
for name, arr in variants.items():
    print(f"{name:>8}: shape={arr.shape}, sum per label={np.round(arr.sum(axis=0), 3)}")

r = nbutil.show("A03")"""),
    md(r"""
## A04：Eq.7 的下标与分母

Eq.2/6 写 $k\in[0,K]$，即 $K+1$ 项；但类别只有 $K$ 个，Eq.7 又除以 $K$。
按字面补一个 0 项会改变 $\mu,\sigma$，从而改变 z-score 与判定阈值。实现采用 $K$ 个真实类别、除数 $K$。
下面用一个人为的能量向量展示差异大小。
"""),
    code(r"""
r = nbutil.show("A04")
print("K 项：z =", round(r.measured["zscore_K_terms_divisor_K"], 4))
print("K+1 项：z =", round(r.measured["zscore_K_plus_1_terms_divisor_K_plus_1"], 4))"""),
    md(r"""
## A05：TE/NTE 的恒等式与“均值为 0”

$$\mathrm{TE}=\frac{p'}{p}=e^{s'-s},\qquad \mathrm{NTE}=\mathrm{TE}-1=e^{s'-s}-1.$$

因为 softmax 每行和为 1，$\frac{1}{K}\sum_k (s'_k-s_k)=0$ 严格成立，
所以论文用基线 $e^0=1$ 有依据。下面在随机 logits 上逐项核对，并检查实现与字面公式的误差。
"""),
    code(r"""
r = nbutil.show("A05")"""),
    md(r"""
## A06：Eq.9/10 是否丢了温度 $T$？

Eq.4 的能量是 $e^{sT}/T$，而 Eq.9/10 的比值把 $T$ 完全消掉，等价于 $T=1$。
这不是“等价变形”，而是两处用了不同的定义。对检测标签影响多大？用 argmax 一致率衡量：

- 若一致率 ≥ 0.99，公式歧义在实践中可忽略；
- 否则说明恢复出的标签依赖温度选择。
"""),
    code(r"""
r = nbutil.show("A06")"""),
    md(r"""
## A17：Fig.80 的文字符号矛盾

附录 F 第一句写“值小于 0 判为后门”，紧接着又说“clean 模型小于 0、poisoned 模型大于 0”。
后者与 Eq.8（$E'_k-\mu>\lambda\sigma$ 判为后门）一致，第一句方向相反。这是一处纯文字矛盾，
代码无法“复算”，只能记录两种说法并要求以公式为准。
"""),
    code(r"""
r = nbutil.show("A17")"""),
    md(r"""
## A20：$\lambda=3$ 在 CIFAR-10 上正好等于统计上界

对任意 $K$ 维能量向量，单侧 $z$ 分数满足

$$\max_k \frac{E_k-\mu}{\sigma}\le\sqrt{K-1}.$$

$K=10$ 时上界恰为 $\sqrt 9=3$，且**只有当其余 9 个标签的能量完全相等时**才取到。
Eq.8 要求严格大于 $\lambda\sigma$，所以 CIFAR-10 上 `margin > 0` 在一般情况下不可能成立。
论文的 $\lambda$ 可用区间 $[1.8,5.7]$ 与 GTSRB（$K=43$，上界 6.48）相符；把 3 直接用于
CIFAR-10 时，阈值刚好压在不可达的上界上。这解释了为什么缩比实验里目标标签的 $z$ 约为
2.5-2.8，而不是 $>3$。
"""),
    code(r"""
r = nbutil.show("A20")"""),
    md(r"""
## 本 Notebook 的小结

| 声明 | 结论 | 要点 |
|---|---|---|
| A01 | CONTRADICTED | 附录 B 的 softmax/能量数值不可复算，排序也印错；方向性结论仍成立 |
| A02 | CONFIRMED | Eq.3 是“最大值置 0”，不是减最大值 |
| A03 | CONTRADICTED | 论文、README、官方代码是三种归一化 |
| A04 | CONTRADICTED | $K+1$ 项与除以 $K$ 不自洽 |
| A05 | CONFIRMED | TE/NTE 恒等式与均值为 0 成立 |
| A06 | CONTRADICTED | Eq.9/10 隐含 $T=1$，与 Eq.4 不一致，且影响 argmax |
| A17 | CONTRADICTED | Fig.80 首句符号写反 |
| A20 | CONTRADICTED | $\lambda=3$ 在 $K=10$ 时等于 $z$ 的上界，严格裕量不可达 |

下一步：`02_theory.ipynb` 检查 Lemma 1/2 的证明与前提。
"""),
]

# ---------------------------------------------------------------------------
# 02 -- theory
# ---------------------------------------------------------------------------
NOTEBOOKS["02_theory"] = [
    md(r"""
# 02 理论证明审计（Tier A）

论文的两个核心引理都不是对任意网络的严格定理，而是建立在一个**替代模型**上的性质：

- **Lemma 1（脏标签）**：在 NTK 核 $K(x,x')=e^{-2\gamma\lVert x-x'\rVert^2}$ 下，
  目标类别的核和超过其他类别，因此任务无关图像会被推向目标类。
- **Lemma 2（损坏）**：对图像做损坏后，目标标签的概率下降、真值标签的概率上升。

本 Notebook 做的三件事：

1. 逐元素检查 Lemma 2 证明中的代数步骤（Eq.29 ↔ Eq.30）；
2. 在真实 CIFAR-10 像素上按论文的核函数计算核和，检查 Lemma 1 的两种情形；
3. 检查干净标签分支 Eq.35 的一次不等式放缩是否对所有取值成立。

> **注意**：这些检查复现的是“论文证明中的计算”，不是“对训练好的深度网络的证明”。
> 因此即使数值支持，也不能把 Lemma 1 升级为 CONFIRMED。
"""),
    code(PREAMBLE),
    md(r"""
## A09：Lemma 1（脏标签）的核和占优

论文证明中的核函数（Eq.13）为

$$K_\gamma(x,x')=\exp\left(-2\gamma\lVert x-x'\rVert^2\right).$$

对查询图 $x$（伪标签 $k_1\neq$ 目标类 $t$），论文要求目标类的核和大于其他任一非 $k_1$ 类：

$$\sum_{j:\,y_j=t}K(x,x_j) > \max_{k\notin\{k_1,t\}}\sum_{j:\,y_j=k}K(x,x_j).$$

代码用灰度像素（32×32，归一化到 [0,1]），$\gamma$ 由查询-训练距离中位数标定，
并在 $0.5\times,1\times,2\times$ 下检查稳健性。两个情形：

- `case1_nt_gt_nk2`：目标类 100 干净 + 90 中毒 = 190 张，其他类各 100 张；
- `case2_nt_eq_nk2`：目标类 80 干净 + 20 中毒 = 100 张，与其他类数量相同。

若最小占优比例 ≥ 0.5 记 PARTIAL（数值支持），否则 CONTRADICTED。数值模拟永远不能判 CONFIRMED。
"""),
    code(r"""
r = nbutil.show("A09")"""),
    md(r"""
## A10：干净标签分支与 Eq.35 的放缩

干净标签下目标类没有“中毒图”，证明需要处理 $v=x+T-x_t$。Eq.35 声称

$$e^{-2\gamma\lVert v-T\rVert^2}\ \ge\ e^{-2\gamma(\lVert v\rVert^2+\lVert T\rVert^2)}.$$

但这只在 $\langle v,T\rangle\ge 0$ 时成立；证明中没有声明该条件。
取 $a=0,t=1,b=1.5,n=2$，则 $v=-0.5$：

- 左边 $=e^{-2(a-b)^2}=e^{-4.5}\approx0.0111$；
- 右边 $=e^{-2((u-b)^2+t^2)}=e^{-2.5}\approx0.0821$。

方向相反，放缩无效。代码同时报告该反例与干净标签协议下的核和模拟。
"""),
    code(r"""
r = nbutil.show("A10")
print("\n反例：", json.dumps(r.measured["counterexample"], ensure_ascii=False))"""),
    md(r"""
## A07：Eq.29 ↔ Eq.30 的代数等价性

Lemma 2 的证明需要把

$$\left(\frac{a+t}{n}-b\right)^2>(a+t-b)^2 \tag{Eq.29}$$

化为

$$t<\frac{2nb}{1+n}-a. \tag{Eq.30}$$

两边开方时默认了 $a+t>0$（像素空间的合理取值），这一步论文没有写。
在 $a+t>0$ 时等价成立；在 $a+t<0$ 时方向翻转。下面各取 4000 组随机数验证。
"""),
    code(r"""
r = nbutil.show("A07")
print("u>0 反例数:", r.measured["mismatches_positive_u"])
print("u<0 反例数:", r.measured["mismatches_negative_u"])
print("系数 2n/(1+n):", [round(x, 4) for x in r.measured["coefficient_2n_over_1_plus_n"]])"""),
    md(r"""
## A08：Lemma 2 是“逐损坏普遍成立”还是“聚合趋势”？

论文 Lemma 2 的表述是逐图、逐损坏的：对任意损坏 $j$，目标标签概率下降且真值标签概率上升。
但正文引用它时通常只说“聚合超过 90%”。这两个强度不同的断言必须分开检查。

代码在缩比 BadNets 模型上取 100 张非目标测试图，施加全部 80 种损坏，统计同时满足两条的
`(图, 损坏)` 比例。判定规则：

- 普遍比例 = 1.0 也只能记 PARTIAL（单个代理模型不能证明普适命题）；
- 聚合比例 ≥ 0.90 → PARTIAL；
- 否则 → CONTRADICTED。
"""),
    code(r"""
r = nbutil.show("A08", max_measured_chars=2500)
print("普遍比例:", round(r.measured.get("universal_fraction", -1), 4))
print("聚合比例:", round(r.measured.get("aggregate_fraction", -1), 4))"""),
    md(r"""
## 本 Notebook 的小结

| 声明 | 结论 | 要点 |
|---|---|---|
| A07 | PARTIAL | Eq.29↔Eq.30 在 $a+t>0$ 时成立，但证明未声明该前提 |
| A08 | PARTIAL/CONTRADICTED | 逐图逐损坏的普遍形式通常被反例否定；聚合趋势另算 |
| A09 | PARTIAL/CONTRADICTED | NTK 核和是否占优取决于 γ 与样本构成 |
| A10 | CONTRADICTED | Eq.35 的放缩缺少 $\langle v,T\rangle\ge0$，存在明确反例 |

引理是论文动机的一部分，不是方法能否工作的充要条件；方法本身的检验在 `04_detection` 与 `05_removal`。
"""),
]

# ---------------------------------------------------------------------------
# 03 -- table audit
# ---------------------------------------------------------------------------
NOTEBOOKS["03_table_audit"] = [
    md(r"""
# 03 表格与文字断言审计（Tier A）

本 Notebook 不训练任何模型，只做两件事：

1. 把论文 Table 3-7 的数字逐格转录到 `src/ebba/claims/tables.py`，然后用程序检查
   表格内部的算术、下界、阈值与正文声称的“最优/更低”。
2. 静态解析官方 Notebook，对照论文的实验协议。

> 转录原则：数字只从 PDF 抄一次；若结论与论文不符，改的是**判定**，不是表格。
> 所有行数、取值范围由 `tests/test_claims.py` 校验。
"""),
    code(PREAMBLE),
    code(r"""
from ebba.claims import tables

print("Table 3 rows:", len(tables.table3_rows()), "| datasets:",
      sorted({r["dataset"] for r in tables.table3_rows()}))
print("Table 4 rows:", len(tables.table4_rows()), "| methods:", tables.TABLE4_METHODS)
print("Table 5/7 PDR:", tables.TABLE5, "| identical:", tables.TABLE5 == tables.TABLE7)
print("Table 6:", json.dumps(tables.TABLE6, ensure_ascii=False))"""),
    md(r"""
## A16：Table 3 是否“全面最优”？

正文称 EBBA+ 在 PDR 与 F1 上优于所有对比方法。逐行比较 15 行 × 2 个指标，
只要存在一个严格劣于最好基线的格子，就记一个反例（平局不算）。
"""),
    code(r"""
r = nbutil.show("A16", max_measured_chars=2500)"""),
    md(r"""
## A11：BA 下界是否对所有行成立？

Sec. 5.2 给出经验下界

$$\mathrm{BA}\ \ge\ \frac{K-1}{K}\,\mathrm{ba},$$

其中 $K$ 是类别数，$\mathrm{ba}$ 是防御前的良性精度。它不是定理，而是“只改被判为目标类的图”这一操作的
最坏情况估计：每个类别最多损失 $1/K$。下面用 Table 4 的每一行检查（ImageNet 取 $K=100$）。
"""),
    code(r"""
r = nbutil.show("A11", max_measured_chars=2200)"""),
    md(r"""
## A12：所有攻击的 ASR 是否都训练到 98.8% 以上？

正文用“No Defense”列的 ASR 说明攻击足够强。逐行检查 Table 4 的 No Defense 列即可。
"""),
    code(r"""
r = nbutil.show("A12")"""),
    md(r"""
## A15：Table 4 的文字结论

正文有两句话需要分开核对：

1. “EBBA+ 的 BA 最优” —— 与防御方法比较（不含 No Defense，因为无防御不是方法）；
2. “EBBA+ 把 ASR 降得更低” —— 与所有方法比较（含 No Defense）。

程序分别统计“BA 严格最好”和“ASR 严格最低”的行数。平局不计为最好/最低。
"""),
    code(r"""
r = nbutil.show("A15", max_measured_chars=2500)"""),
    md(r"""
## A13：Table 6 的 GTSRB 测试集大小

论文写 12603，GTSRB 官方测试集是 12630。两者只差数字顺序，属于转录错误。
这类错误本身不影响实验，但会让人怀疑其他表格数字的可靠性。
"""),
    code(r"""
r = nbutil.show("A13")"""),
    md(r"""
## A14：Table 5 与 Table 7 的聚类差异

论文用 Table 5（正文）和 Table 7（附录）说明四种聚类方法影响很小。
两张表**逐格相同**，所以它们不是两条独立证据。这里只检查极差是否 ≤ 0.05，
并记录“同一组数字被印了两次”这一事实。
"""),
    code(r"""
r = nbutil.show("A14")"""),
    md(r"""
## A18：官方 Notebook 是否实现了论文协议？

官方代码是 PaddlePaddle Notebook（`reference/EBBA.ipynb`）。本项目**不执行**它，
只解析 JSON 并检查 8 项协议：图像池是否任务无关、是否按伪标签均衡、样本量、损坏数量、
除法损坏是否从 1 开始、TE 截断是否一致、是否有聚类步骤、是否报告 BA/ASR。
"""),
    code(r"""
r = nbutil.show("A18", max_measured_chars=3200)
for check in r.measured["checks"]:
    mark = "PASS" if check["passes"] else "FAIL"
    print(f'{mark}  {check["id"]:<20} official: {check["official"]}')"""),
    md(r"""
## A19：自适应攻击下“检测率超过 99%”

附录 G.4 列出 13 组软标签参数，并明确写道 0.985 时 EBBA 失效。按参数组计算，
通过率至多 12/13 = 92.3%。若“检测率”指逐样本比例，则文中没有给出可复算的数值，
只能记 PARTIAL。
"""),
    code(r"""
r = nbutil.show("A19")"""),
    md(r"""
## 本 Notebook 的小结

- Table 3 的“PDR/F1 全面最优”和 Table 4 的“BA 最优、ASR 更低”都能找到反例；
- BA 下界不是定理，Table 4 内部就有两行违反；
- “ASR 都超过 98.8%”有两个反例；
- GTSRB 测试集大小印错；
- 官方 Notebook 与论文协议有 7 项不符，只能作为公式参考；
- 自适应攻击的“>99%”按参数组只有 92.3%。

这些结论的含义是：**论文的表述与表格存在可核对的不一致**，需要在引用时注明，而不是说方法无效。
"""),
]

# ---------------------------------------------------------------------------
# 04 -- detection
# ---------------------------------------------------------------------------
NOTEBOOKS["04_detection"] = [
    md(r"""
# 04 EBBA 检测机制（Tier B，缩比）

本 Notebook 检验论文的检测主张在容器可承受的规模下是否成立。所有结论都是 **proxy**：

- 数据：CIFAR-10 固定子集 20,000 训练 / 2,000 测试；任务无关图像用 CIFAR-100 测试集 5,000 张代理；
- 模型：SmallCNN（约 1.2M 参数）；
- 攻击：BadNets（3×3 白块）与 Blend（固定图案，α=0.2），目标类 0，投毒率 10%；
- 训练：SGD，30 epochs，seed 2024，带轻度损坏增广（噪声/模糊/对比度）。

**阈值说明**：缩比实验用 `λ_proxy = 2.2` 而非论文的 3.0。A20 已证明 $K=10$ 时
$\max_k(E_k-\mu)/\sigma \le \sqrt{K-1}=3$ 且只能取等号，因此 Eq.8 的严格 `>3σ` 在此不可达。
2.2 取自本代理实测的 clean z≈1.77 与后门 z≈2.66 之间；这是代理标定，不是论文设定。

按项目规则，proxy 成功只能记 `PARTIAL`，失败记 `CONTRADICTED`。

## 检测流程

1. 对任务无关图像做前向，得到伪标签；
2. 每个伪标签类别保留最多 50 张（均衡选择）；
3. 对每条样本执行 Eq.3 的 clamp，再算 Eq.4-6 的标签能量；
4. 计算 $\mu,\sigma$，检查 Eq.8 是否在某个标签上成立。
"""),
    code(PREAMBLE),
    md(r"""
## B01：目标标签是否在 top 输出中更常见？（Fig.1）

后门模型会把无关图像也推向目标类。统计目标标签在 5,000 张 OOD 图像输出中的排名：
若后门模型的 top-3 频率显著高于均匀基线 0.30，且干净模型没有同样偏移，则机制成立。
"""),
    code(r"""
r = nbutil.show("B01", max_measured_chars=2600)
for name, rec in r.measured["models"].items():
    if isinstance(rec, dict):
        print(f'{name:>8}: mean_rank={rec["mean_rank"]:.2f}, top1={rec["top1"]:.3f}, top3={rec["top3"]:.3f}')"""),
    md(r"""
## B02：Eq.8 能否在无任务样本时分开后门模型与干净模型？

对每个模型取均衡 OOD 子集，算 10 个标签能量，报告最大偏离、margin 与检测结果。
判据：后门模型 `margin > 0` 且目标标签 = 0；干净模型 `margin < 0`。
由于 $K=10$ 时 $z\le3$，此处用代理阈值 `λ_proxy=2.2`。
若只有部分攻击被正确检出，结论记 `PARTIAL`（机制存在但不普适）；全部漏检才记 `CONTRADICTED`。
"""),
    code(r"""
r = nbutil.show("B02", max_measured_chars=3000)
for name, rec in r.measured["models"].items():
    print(f'{name:>8}: target={rec["target"]}, margin={rec["margin"]:+.3f}, '
          f'z={rec["zscore"]:.2f}, backdoored={rec["is_backdoored"]}')"""),
    code(r"""
import numpy as np

models = r.measured["models"]
fig, ax = plt.subplots(figsize=(7, 3.2))
for name, rec in models.items():
    ax.bar(np.arange(10) + (0.2 if name == "badnets" else -0.2 if name == "blend" else 0),
           np.asarray(rec["energy"]) - np.mean(rec["energy"]),
           width=0.2, label=name, alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("label")
ax.set_ylabel("E_k - mean(E)")
ax.set_title("Label energy deviation on task-agnostic proxy images")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()"""),
    md(r"""
## B03：λ 的可用区间与 Fig.80

固定后门模型的目标标签，扫描 λ，要求“后门 margin > 0 且干净 margin < 0”。
论文报告可用区间约为 [1.8, 5.7]。缩比模型只需与之有重叠，且 `λ_proxy=2.2` 落在可用区间内；
区间端点由模型决定，不能直接对齐论文。`λ=3` 的可行性由 A20 单独判定。

Fig.80 是单模型图，故任一攻击复现出重叠区间即通过；Blend 在任何 λ 下都检不出，
那是 B02 的结论，不在 B03 重复计为“区间形状不符”。
"""),
    code(r"""
r = nbutil.show("B03")
for attack, rec in r.measured["attacks"].items():
    print(f'{attack}: valid={rec["valid_interval"]}, chosen_valid={rec["chosen_valid"]}')"""),
    md(r"""
## B11：均衡选择是否减少误报？（Fig.78/79）

论文强调“每类保留 N 张”而不是随便取一批图。消融方式：

- 均衡：每个伪标签最多 50 张；
- 不均衡：只保留伪标签最集中的 3 类，共 500 张。

比较目标标签相对最强非目标标签的领先量 $z_t-\max_{k\ne t}z_k$，并统计两种选择下被 Eq.8
误标的非目标标签数。论文称不均衡选择会让其他标签的能量接近目标，故均衡选择的领先量应更大且为正。
"""),
    code(r"""
r = nbutil.show("B11", max_measured_chars=2600)
for name, rec in r.measured["models"].items():
    print(f'{name:>8}: lead balanced={rec["balanced_lead"]:+.2f}, '
          f'imbalanced={rec["imbalanced_lead"]:+.2f}, '
          f'false flags {rec["balanced_false_flags"]}/{rec["imbalanced_false_flags"]}')"""),
    md(r"""
## B12：自适应软标签攻击与双边检验（机制演示）

论文的 Eq.8 是单边检验：只看“能量偏高”。如果攻击者把目标能量压到均值以下，
单边检验会漏检。附录 G.4 讨论了这一点。

这里**不训练自适应攻击**（那属于 Tier C），而是在实测能量向量上把目标能量置为
$\mu-4\sigma$，比较单边与双边 $|E-\mu|>\lambda\sigma$ 的行为。
它只说明检验统计量的性质，不能当作对论文 13 组自适应实验的复现。
"""),
    code(r"""
r = nbutil.show("B12", max_measured_chars=2600)
print("one-sided detected:", r.measured["one_sided"]["is_backdoored"])
print("two-sided detected:", r.measured["two_sided"]["is_backdoored"],
      "target:", r.measured["two_sided"]["target"])"""),
    md(r"""
## 本 Notebook 的小结

- B01-B03 检验“检测机制”是否按论文方向工作；
- B11 检验均衡选择这一实现细节；
- B12 只是检验统计量的机制演示，不代表自适应攻击已被复现。

若某个 B 声明为 `CONTRADICTED`，需要区分两种可能：方法在缩比设定下确实不成立，
或缩比模型的训练/攻击强度不足。manifest 中的模型精度与 ASR 门槛用于排除后一种解释，
但缩比模型与论文 ResNet18 的差距始终是结论的适用边界。
"""),
]

# ---------------------------------------------------------------------------
# 05 -- removal
# ---------------------------------------------------------------------------
NOTEBOOKS["05_removal"] = [
    md(r"""
# 05 EBBA+ 样本检测与去除（Tier B，缩比）

EBBA+ 不使用任务相关样本，也不需要重训练。流程：

1. 用 EBBA 找到目标标签 $t$；
2. 收集“被模型判为 $t$”的图：一部分是真正的中毒图，一部分是目标类的干净图；
3. 对每张图施加 80 种损坏，算 $\mathrm{NTE}_k=\sum_j(e^{s'_k-s_k}-1)$；
4. 用 Eq.12 的 argmax 恢复原始标签；
5. 对 NTE 向量做 t-SNE + Birch 二聚类；论文没有说明无标签时如何认定中毒簇，
   本套件用“簇内平均目标 logit 相对最强非目标 logit 的领先幅度”来定向，并在 B07 报告敏感性；
6. 只把中毒簇重标为 NTE argmax，其余不动。

> 缩比协议：候选池 = 200 张中毒测试图 + 100 张目标类干净图（模型判为目标类的才进入流程）。
> 80 种损坏 = 19 类 × 4 强度 + 除以 2..5。所有结论为 proxy。
>
> **目标标签的处理**：EBBA+ 以 EBBA 检出的目标标签为前提。本 Notebook 用真实目标标签（oracle）
> 运行样本阶段，目的是单独检验 EBBA+ 本身；EBBA 能否找对目标属于 B02。若 B02 在某攻击上失败
> （例如 Blend），完整流水线在该攻击上就不可用，这一事实不会被掩盖。
"""),
    code(PREAMBLE),
    md(r"""
## B04：损坏是否让真值标签前移、目标标签后移？

这是 Lemma 2 的聚合版本：对 100 张中毒图 × 80 种损坏，统计
真值标签概率上升、目标标签概率下降、真值标签排名前移的比例。

判定分两层：方向层要求两类概率变化各自过半数（>0.60）；论文层（Fig.2 caption 的
“真值标签排名进入前 6 的比例超过 90%”）另记。排行榜统计在缩比模型上是退化的：
中毒图本来就把真值类排在目标类之后（平均排名 1.4，几乎没有前移空间），因此只作诊断。
"""),
    code(r"""
r = nbutil.show("B04", max_measured_chars=2600)
shift = r.measured["probability_shift"]
print("gt probability increase:", round(shift["gt_increase_fraction"], 4))
print("target probability drop:", round(shift["target_decrease_fraction"], 4))
print("gt rank forward         :", round(r.measured["rank_shift"]["gt_forward_fraction"], 4))"""),
    md(r"""
## B05：NTE argmax 能否恢复中毒图的原始标签？

对候选池中每张图算 Eq.11 的 NTE 向量，取 argmax 作为恢复标签。
中毒图应能以较高比例恢复为原类。对于源预测就是目标类的干净图，损坏后第 0 个分量的概率下降，
$\mathrm{NTE}_0$ 为负，argmax 不会回到 0，因此其恢复率接近 0——这正是需要聚类、而非只靠 argmax
区分两类图的原因，不是方法失败。

论文的 ResNet18 上恢复率接近其 PDR；缩比模型在 80 种损坏下的 top-1 精度远低于此，
所以本 Notebook 同时报告两件事：是否显著高于随机（机制方向），以及是否达到 0.85（论文级幅度）。
方向成立但幅度不足记 `PARTIAL`，接近随机才记 `CONTRADICTED`。
"""),
    code(r"""
r = nbutil.show("B05")
for attack, rec in r.measured["attacks"].items():
    print(f'{attack}: poisoned recovery={rec["recovery_accuracy_poisoned"]:.3f}, '
          f'clean-target acc={rec["accuracy_clean_target"]:.3f}, coverage={rec["coverage"]:.3f}')"""),
    md(r"""
## B06：干净目标类图的能量转移是否“随机”？

论文 Sec.4.2 说：干净图的能量会随机转移到其他标签，因此 **NTE 本身无法区分干净图与中毒图**，
必须再叠加 t-SNE + 聚类。这里检验干净目标类图的 NTE argmax 分布是否接近均匀
（卡方检验 p>0.01），并报告归一化熵与最大类占比；峰值 NTE 的均值比与 AUC 只作辅助诊断。
"""),
    code(r"""
r = nbutil.show("B06", max_measured_chars=2600)
for attack, rec in r.measured["attacks"].items():
    print(f'{attack}: entropy={rec["normalized_entropy"]:.2f}, max_share={rec["max_class_share"]:.2f}, '
          f'chi2_p={rec["chi2_p_value_vs_uniform"]:.3g}, random_like={rec["random_like"]}')"""),
    md(r"""
## B07：PDR / F1 是否达到可用量级？

完整流程（t-SNE + Birch）后的中毒检测精度。论文报告约 0.95；缩比模型只要求
PDR ≥ 0.70 且 F1 ≥ 0.70。这个门槛是有意放宽的：缩比模型与数据集都远小于论文。

**簇定向是套件的假设**：论文只写了 t-SNE + Birch，没有给出无标签时认定中毒簇的规则。
这里用簇内平均目标 margin（目标 logit − 最强非目标 logit）定向；同时报告两种敏感性：
沿用“峰值 NTE 更高者为中毒簇”的旧规则，以及用真实标签选最佳簇的上界。三者并列，
读者可以判断结论在多大程度上依赖这条假设。
"""),
    code(r"""
r = nbutil.show("B07")
for attack, rec in r.measured["attacks"].items():
    print(f'{attack}: PDR={rec["pdr"]:.3f}, precision={rec["precision"]:.3f}, '
          f'F1={rec["f1"]:.3f} (tp={rec["tp"]}, fp={rec["fp"]}, fn={rec["fn"]}) '
          f'[peak-rule PDR={rec["pdr_peak_rule"]:.3f}, oracle-cluster PDR={rec["pdr_best_cluster_oracle"]:.3f}]')"""),
    md(r"""
## B08：聚类方法是否影响结果？

对 HC、Birch、Mean Shift、DBSCAN 重复同一流程，要求四种方法的 PDR 极差 ≤ 0.10。
若某方法退化，代码回退到 KMeans 并记录 `method_used="fallback_kmeans"`。

论文 Table 5/7 的极差是 0.010，那是在 PDR≈0.95、两类簇几乎完全分开的设定下得到的。
缩比模型的 PDR 只有 0.7–0.85，簇本身重叠，因此换聚类方法会实际改变判为中毒的图。
若实测极差超限，它说明的是“缩比簇不纯”，而不是论文的四种方法在某数据集上真的差很多；
两种解读都在结果里给出（各方法的 PDR 逐个列出）。
"""),
    code(r"""
r = nbutil.show("B08", max_measured_chars=2600)
for attack, rec in r.measured["attacks"].items():
    print(attack, {m: round(v["pdr"], 3) for m, v in rec["methods"].items()},
          "spread=", round(rec["pdr_spread"], 3))"""),
    md(r"""
## B09 / B10：去除后门后的 BA 与 ASR

按论文做法，只改“被判为目标类且被聚类判为中毒”的图。指标分两层：

- **池内（条件指标）**：被 EBBA+ 检查的中毒图还有多少仍成功（ASR），干净目标类池的准确率；
- **全测试集（对应 Table 4）**：BA 降幅 ≈ 池内误报数 / 2,000；ASR_after ≈ 全局 ASR_before × 池内 ASR_after。

B09 的判定看方向（ASR 至少降一半）与全局 BA（降幅 ≤0.05）；论文级幅度（ASR 降 ≥0.80、BA 降 ≤0.01）
单独记录。池内 BA 降幅会明显更大，因为全部误报都集中在 200 张目标类图上——两个数字都展示，不取好看的那个。

B10 用全测试集 BA 检查 $\mathrm{BA_{after}}\ge\frac{K-1}{K}\mathrm{BA_{before}}$。
"""),
    code(r"""
r9 = nbutil.show("B09")
r10 = nbutil.show("B10")"""),
    md(r"""
## B13 / B14：多目标与实现敏感性

- **B13**：在实测能量向量上注入第二个与目标 0 等高的异常，检查 top-2 排序能否同时标出两者。
  这是统计量的机制检查，不是双目标攻击训练。
- **B14**：扫描 NTE 的温度 $T\in\{0.5,1,2,3\}$ 与截断 $cap\in\{\text{无},4,5\}$，
  以 $(T=1,cap=5)$ 为基准统计标签恢复一致率。它回答“Eq.9/10 没写温度是否要紧”。
  温度与截断分组报告：若只改 cap 时一致率恒为 1，而改温度时降到 0.90–0.96，
  说明歧义只出在温度上（与 A06 的合成实验结论一致）。
"""),
    code(r"""
r13 = nbutil.show("B13")
r14 = nbutil.show("B14", max_measured_chars=2600)"""),
    md(r"""
## 本 Notebook 的小结

- B04-B07 检验 EBBA+ 的核心机制：损坏使目标证据下降、NTE 能恢复原标签、聚类能分开两类；
- B08 检验聚类选择是否稳健；
- B09/B10 检验“只重标、不重训”能否同时降低 ASR 并保持 BA；
- B13/B14 是机制与实现敏感性检查，不是论文实验的复现。

任何一项失败都应先查 manifest 中的模型精度与攻击成功率，再判断是方法问题还是缩比设定问题；
NTE 恢复率本身没有门槛，它是被检验的结果（B05），不是构建前提。
"""),
]

# ---------------------------------------------------------------------------
# 06 -- verdicts
# ---------------------------------------------------------------------------
NOTEBOOKS["06_verdicts"] = [
    md(r"""
# 06 汇总判定

前五个 Notebook 已把每条声明写入 `reports/claims/<ID>.json`。本 Notebook：

1. 运行 Tier C 的登记项（它们会立即返回 `NOT_REPRODUCIBLE` 并附带原因）；
2. 汇总全部 39 条声明，统计 verdict；
3. 生成 `verdicts.json`、`report.md`、`report.html`。

> 判定不是“论文好/坏”的评分。它回答的是：**这句话在给定资源下能否被独立复核，复核结果是否与原文一致。**
"""),
    code(PREAMBLE),
    md(r"""
## Tier C：容器内无法复现的条目

这些条目没有实现上的失败，只是需要论文级算力或未公开的数据/代码。显式记录它们，
是为了让报告不出现 PENDING，也避免把“没做”误读为“通过”。
"""),
    code(r"""
c_results = run_many(["C01", "C02", "C03", "C04", "C05"])
for cid, res in c_results.items():
    print(f"{cid}: {res.verdict.value} — {res.notes_zh[:80]}...")"""),
    md(r"""
## 全部声明一览
"""),
    code(r"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from ebba.claims.model import load_registry

registry = load_registry()
rows = []
for cid in registry:
    path = config.REPORTS_DIR / "claims" / f"{cid}.json"
    if not path.exists():
        rows.append({"id": cid, "tier": registry[cid]["tier"], "verdict": "PENDING",
                     "title_zh": registry[cid]["title_zh"]})
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"id": cid, "tier": payload["tier"], "verdict": payload["verdict"],
                 "title_zh": payload["title_zh"]})

nbutil.show_table([r["id"] for r in rows])
print("rows:", len(rows), "| PENDING:", sum(r["verdict"] == "PENDING" for r in rows))"""),
    md(r"""
## 判定统计与图
"""),
    code(r"""
counts = Counter(r["verdict"] for r in rows)
order = ["CONFIRMED", "PARTIAL", "CONTRADICTED", "NOT_REPRODUCIBLE", "ERROR", "PENDING"]
for key in order:
    print(f"{key:<18} {counts.get(key, 0)}")

colors = {"CONFIRMED": "#1b7f3a", "PARTIAL": "#9a6700", "CONTRADICTED": "#b42318",
          "NOT_REPRODUCIBLE": "#5f6b7a", "ERROR": "#7a1fa2", "PENDING": "#c9d1d9"}
fig, ax = plt.subplots(figsize=(6.5, 3.2))
keys = [k for k in order if counts.get(k, 0)]
ax.bar(keys, [counts[k] for k in keys], color=[colors[k] for k in keys])
ax.set_ylabel("claims")
ax.set_title("Verdict distribution")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()"""),
    md(r"""
## Tier 分布
"""),
    code(r"""
by_tier = {}
for r in rows:
    by_tier.setdefault(r["tier"], Counter())[r["verdict"]] += 1
for tier in sorted(by_tier):
    print(tier, dict(by_tier[tier]))"""),
    md(r"""
## 被否定的声明

下面逐条列出 `CONTRADICTED` 的实测要点。完整实测值在 `reports/report.html` 的折叠块中。
"""),
    code(r"""
for r in rows:
    if r["verdict"] != "CONTRADICTED":
        continue
    payload = json.loads((config.REPORTS_DIR / "claims" / f'{r["id"]}.json').read_text(encoding="utf-8"))
    print(f'{r["id"]}  {r["title_zh"]}\n   {payload["notes_zh"]}\n')"""),
    md(r"""
## 部分成立与不可复现

`PARTIAL` 有两种来源：证明前提不完整（A07、A19），以及缩比实证只支持方向（所有 B 声明）。
`NOT_REPRODUCIBLE` 是 Tier C，附有所需资源说明。
"""),
    code(r"""
for r in rows:
    if r["verdict"] in {"PARTIAL", "NOT_REPRODUCIBLE"}:
        print(f'{r["id"]:>4}  {r["verdict"]:<18} {r["title_zh"]}')"""),
    md(r"""
## 运行耗时与环境

`reports/run_metadata.json` 记录每个 Notebook 的状态与耗时；`reports/verdicts.json` 是机器可读的最终产物。
"""),
    code(r"""
meta_path = config.REPORTS_DIR / "run_metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print("device:", meta.get("device"), "| fast:", meta.get("fast"),
          "| total:", meta.get("total_seconds"), "s")
    for rec in meta["notebooks"]:
        print(f'  {rec["name"]:<18} {rec["status"]:<8} {rec.get("seconds", "-")}s')
else:
    print("run_metadata.json 不存在（直接打开 Notebook 时正常）")"""),
    md(r"""
## 结论怎么引用

1. **Tier A 的 `CONTRADICTED`** 是可直接复核的事实：附录 B 数值、表格算术、BA 下界反例、
   ASR 门槛反例、GTSRB 笔误、Table 3/4 过度声明、官方代码与论文协议不符。引用论文时应注明这些例外。
2. **Tier B 的 `PARTIAL`** 只说明方法机制在缩比设定下按预期工作，**不能**写成“复现了论文的 95% PDR”。
3. **Tier C 的 `NOT_REPRODUCIBLE`** 是范围声明，不是负面结论。
4. 任何 `ERROR` 或 `PENDING` 都应视为交付缺陷；`verify` 的退出码会反映执行故障。

最终报告：`reports/report.html`（自包含，可离线打开）与 `reports/verdicts.json`。
"""),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = OUT / f"{name}.ipynb"
        payload = build(cells)
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
