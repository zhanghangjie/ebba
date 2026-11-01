# EBBA / EBBA+ 验证报告

论文：Gao, Chen, Sun, Li, Li, Shao. *Energy-based Backdoor Defense without Task-Specific Samples and Model Retraining*. ICML 2024, PMLR 235.

## 判定统计

| 判定 | 数量 |
|---|---|
| CONFIRMED | 3 |
| PARTIAL | 13 |
| CONTRADICTED | 18 |
| NOT_REPRODUCIBLE | 5 |
| ERROR | 0 |
| PENDING | 0 |

## 逐条结果

| ID | 层级 | 判定 | 声明 | 位置 |
|---|---|---|---|---|
| A01 | A | CONTRADICTED | 附录 B 算例：softmax 与能量数值是否可复算 | Appendix B, Eq.21-26, p.14 |
| A02 | A | CONFIRMED | Eq.3 是“最大值置 0”，而非减去最大值 | Eq.3, p.5; Appendix B Eq.24 |
| A03 | A | CONTRADICTED | Eq.5、官方 README 与官方代码的归一化公式一致 | Eq.5, p.5; official README/notebook |
| A04 | A | CONTRADICTED | Eq.7 的求和下标与方差分母自洽 | Eq.2, Eq.6, Eq.7, p.5 |
| A05 | A | CONFIRMED | Eq.9-10 的 TE/NTE 恒等式及均值为 0 | Eq.9-10, p.6 |
| A06 | A | CONTRADICTED | Eq.9-10 与 Eq.4 使用同一温度 T | Eq.4, Eq.9-10, p.5-6 |
| A07 | A | PARTIAL | Lemma 2 证明中 Eq.30 的代数等价性 | Appendix C, Eq.30, p.16 |
| A08 | A | CONTRADICTED | Lemma 2 对每一种损坏都成立 | Lemma 2, p.6; Appendix C |
| A09 | A | CONTRADICTED | Lemma 1（脏标签）的 NTK 核和占优 | Lemma 1, p.4; Appendix A Eq.15-20 |
| A10 | A | CONTRADICTED | Lemma 1（干净标签）的 NTK 核和占优 | Appendix G.3, Eq.35-36, p.26 |
| A11 | A | CONTRADICTED | BA 下界 (K-1)/K × ba 对所有 Table 4 行成立 | Sec. 5.2, p.8; Table 4, p.9 |
| A12 | A | CONTRADICTED | 所有攻击的 ASR 均训练到 98.8% 以上 | Sec. 5.1, p.7; Table 4 No Defense 列 |
| A13 | A | CONTRADICTED | Table 6 的 GTSRB 测试集大小与标准数据集一致 | Table 6, p.18 |
| A14 | A | CONFIRMED | 四种聚类方法的 PDR 差异很小（Table 5/7） | Table 5, p.9; Table 7, p.24 |
| A15 | A | CONTRADICTED | Table 4 文字结论：EBBA+ 的 BA 最优、ASR 更低 | Sec. 5.2, p.7-8; Table 4, p.9 |
| A16 | A | CONTRADICTED | Table 3 声称 EBBA+ 在 PDR 和 F1 上全面最优 | Sec. 5.2, p.7; Table 3, p.8 |
| A17 | A | CONTRADICTED | Fig.80 的文字阈值方向与 Eq.8 一致 | Appendix F, p.24; Eq.8, p.5 |
| A18 | A | CONTRADICTED | 官方代码实现了论文的无任务样本、均衡选择与 80 种损坏协议 | Sec. 4.1, Sec. 5.1; official notebook |
| A19 | A | PARTIAL | 自适应攻击下 EBBA 的检测率超过 99% | Appendix G.4, p.26 |
| A20 | A | CONTRADICTED | λ=3 在 CIFAR-10（K=10）上可满足 Eq.8 的严格不等式 | Eq.7-8, p.5; Appendix F p.24; Table 2, p.8 |
| B01 | B | PARTIAL | 目标标签在干净/OOD 样本的 top 输出中更常见（Fig.1） | Fig.1, p.2; Lemma 1 |
| B02 | B | PARTIAL | EBBA 无需任务样本即可检测后门模型（Table 2） | Table 2, p.8; Eq.8 |
| B03 | B | PARTIAL | λ 的可用区间形状与 Fig.80 一致 | Fig.80, p.24 |
| B04 | B | PARTIAL | 图像损坏使真值标签前移、目标标签后移（Fig.2） | Fig.2, p.2; Lemma 2 |
| B05 | B | PARTIAL | NTE 的 argmax 能恢复中毒图像的原始标签（Eq.12） | Eq.12, p.6 |
| B06 | B | CONTRADICTED | 干净目标类图像的 NTE 转移接近随机，故需聚类区分 | Sec. 4.2, p.6-7 |
| B07 | B | PARTIAL | EBBA+ 的 PDR/F1 达到论文报告的量级（Table 3） | Table 3, p.8 |
| B08 | B | CONTRADICTED | 四种聚类方法的 PDR 差异很小（Table 5/7） | Table 5, p.9; Table 7, p.24 |
| B09 | B | PARTIAL | EBBA+ 去除后门且保持良性精度（Table 4） | Table 4, p.9; Eq.12 |
| B10 | B | PARTIAL | 缩比模型上 BA ≥ (K-1)/K × ba | Sec. 5.2, p.8 |
| B11 | B | PARTIAL | 均衡选择任务无关图像可减少误检（Fig.78/79） | Appendix F, p.24 |
| B12 | B | PARTIAL | 自适应软标签攻击下，双边检验 |E-μ|>λσ 仍可检测 | Appendix G.4, p.26 |
| B13 | B | PARTIAL | 多目标攻击下能量统计可同时标出两个目标 | Appendix G.2, p.9/25 |
| B14 | B | CONTRADICTED | NTE 的标签恢复对温度/截断变体不敏感 | Eq.4 vs Eq.9-10 |
| C01 | C | NOT_REPRODUCIBLE | Table 3/4 的全量数字（ResNet18、200 epochs、三数据集） | Table 3-4, p.8-9 |
| C02 | C | NOT_REPRODUCIBLE | 语音 ESC-50 与文本 THUCNews 的后门检测结果 | Sec. 5.4, p.8; Appendix G.1 |
| C03 | C | NOT_REPRODUCIBLE | 13 组自适应攻击参数的完整训练复现 | Appendix G.4, p.26 |
| C04 | C | NOT_REPRODUCIBLE | all-to-all 与 full-target 攻击的完整评估 | Appendix G.5-G.6, p.27 |
| C05 | C | NOT_REPRODUCIBLE | ImageNet 子集（100 类、224×224）规模的结果 | Table 6, p.18; Table 3-4 |

## 被否定的声明

### A01 附录 B 算例：softmax 与能量数值是否可复算

按论文自己给出的 logits 精确计算 softmax，s2 应为 [0.0067, 0.9933, 0.0000]，而非 [0.12, 0.88, 0]；能量为 [4.7243, 4.7185, 4.6691]，而非 [4.82, 4.43, 4.62]。原文的顺序 E0>E2>E1 与“目标标签能量最低”的结论在其自身 logits 下均不成立（精确顺序为 E0>E1>E2）。clamp 后的示例方向正确，但 s* 与 E* 仍有 0.01-0.04 的误差。

### A03 Eq.5、官方 README 与官方代码的归一化公式一致

论文 Eq.5、官方 README（CIFAR-10 用 P_k−1）与官方 Notebook（截断 2 再减 0.6）是三种不同归一化，不能同时成立。结论稳健性单独报告：三者选出的目标标签是否一致。

### A04 Eq.7 的求和下标与方差分母自洽

Eq.2/6 写作 k∈[0,K]（K+1 项），Eq.7 却除以 K。K 个真实标签时应使用 K 项、除数 K；若按字面补一项，最大 z-score 会明显变化。实现采用 K 项。

### A06 Eq.9-10 与 Eq.4 使用同一温度 T

Eq.4 的能量是 exp(s·T)/T，而 Eq.9/10 的比值把 T 消掉了（等价于 T=1）。形式不一致；实际影响用 argmax 一致率衡量。

### A08 Lemma 2 对每一种损坏都成立

Lemma 2 的原文是逐图逐损坏的普遍断言（st>s'_t 且 s_k1<s'_k1）。在缩比后门模型上统计全部 80 种损坏：同时满足两条的 (图,损坏) 比例为 76.01%，故普遍形式被反例否定；但按损坏类型取平均后，70.0% 的损坏呈现聚合方向，与论文插图想表达的总体趋势一致。

### A09 Lemma 1（脏标签）的 NTK 核和占优

按 Eq.13 的 NTK 核 K=e^{-2γ||x-x'||²}，在真实 CIFAR-10 像素上计算目标类与其他类的核和；γ 由查询-训练距离中位数标定，并报告 0.5×/1×/2× 的敏感性。两个情形分开看：目标类样本数多于其他类（190 vs 100）时目标核和几乎总是最大；两类样本数相等（100 vs 100）时只有 8% 的查询满足 Lemma 1。Lemma 1 的原文没有样本数条件，因此等量情形是合法反例，判 CONTRADICTED；更大的目标类会机械地带来更大的核和，这部分不能算作对引理的验证。这是对论文证明所用替代模型的复算，不是对训练网络的证明。

### A10 Lemma 1（干净标签）的 NTK 核和占优

Eq.35 声称 e^{-2γ||v-T||²} ≥ e^{-2γ(||v||²+||T||²)}（v=x+T-x_t），该式仅在 ⟨v,T⟩≥0 时成立。反例 a=0,t=1,b=1.5,n=2：v=-0.5，左边 e^{-4.5} 小于右边 e^{-2.5}，方向相反，故干净标签分支的证明步骤无效。另外在干净标签协议下做核和模拟，目标类并不占优。

### A11 BA 下界 (K-1)/K × ba 对所有 Table 4 行成立

按 Sec. 5.2 的下界 (K−1)/K×ba 逐行核对 Table 4。15 行中 2 行低于下界（容差 0.005），说明该式是经验观察而非可证定理，或表中数字有舍入/实验偏差。

### A12 所有攻击的 ASR 均训练到 98.8% 以上

论文 Sec. 5.1 称所有攻击均训练到 ASR 98.8% 以上。Table 4 的 No Defense 列中有 2 行低于 98.8%（ImageNet Blend 98.62、ImageNet FIBA 98.59），属于可直接核对的数值反例。

### A13 Table 6 的 GTSRB 测试集大小与标准数据集一致

GTSRB 测试集的标准大小是 12630，论文 Table 6 写成 12603（数字转置）。训练集 39209 与标准值一致，因此这不是“使用了子集”，而是笔误。

### A15 Table 4 文字结论：EBBA+ 的 BA 最优、ASR 更低

Table 4 标题称 EBBA+ 的 BA 在所有方法中最好、ASR 与之相当。逐行核对：BA 严格最高 13/15 行，ASR 严格最低 5/15 行。论文正文的措辞是“lower or competitive”，并非全面最优；标题中的“best”对 2 行（Cifar10 Blend、Cifar10 DUBA）不成立。

### A16 Table 3 声称 EBBA+ 在 PDR 和 F1 上全面最优

Table 3 标题称 EBBA+ 在 PDR 与 F1 上优于所有对比方法。15 行中有 5 行在至少一个指标上严格劣于最好的基线（Cifar10 BadNets 的 PDR、Cifar10 Blend 的 F1、Cifar10 FIBA 的 PDR、GTSRB WaNet 的 PDR、ImageNet BadNets 的 F1），故“全面最优”不成立。另有若干平局，未计入反例。

### A17 Fig.80 的文字阈值方向与 Eq.8 一致

Appendix F 第一句称“值小于 0 判为后门”，随后又说 clean<0、poisoned>0，与 Eq.8（大于 λσ 判为后门）矛盾。应以 Eq.8 为准，第一句是笔误。

### A18 官方代码实现了论文的无任务样本、均衡选择与 80 种损坏协议

对官方 PaddlePaddle Notebook 做静态解析（不执行）。8 项协议检查中7 项与论文不符：使用 GTSRB 验证集（任务相关）、无每类均衡、仅 704 张图、64 种损坏（含除以 1）、TE 截断 5/4 不一致、无聚类步骤。它能作为公式与流程的交叉参考，但不是论文实验的完整实现。

### A20 λ=3 在 CIFAR-10（K=10）上可满足 Eq.8 的严格不等式

对任意 K 维能量向量，max_k (E_k−μ)/σ ≤ √(K−1)。K=10 时上界恰为 3：只有其余 9 个标签能量完全相等才能取到，而 Eq.8 要求严格大于 λσ。因此论文在 CIFAR-10 上取 λ=3 时，检测裕量在数学上几乎不可能为正；其 λ 的可用区间 [1.8, 5.7] 与 K=43 的 GTSRB（上界 6.48）一致，说明 λ=3 是在 GTSRB 上标定后直接用于 CIFAR-10 的。若按 Eq.7 字面把求和写成 K+1 项，上界升到 √10≈3.16，λ=3 才勉强可达。

### B06 干净目标类图像的 NTE 转移接近随机，故需聚类区分

论文 Sec.4.2 的关键论点是：干净图（包括被判为目标类的干净图）的能量转移是随机的，因此 NTE 本身无法区分干净图与中毒图，必须依赖 t-SNE + 聚类。这里检验干净目标类图的 NTE argmax 分布是否接近均匀（卡方检验，p>0.01）并报告归一化熵、最大类占比；峰值 NTE 的均值比与 AUC 仅作辅助诊断。若分布显著集中，说明缩比模型的能量转移存在系统性偏差，论文的“随机”描述在该设定下不成立。

### B08 四种聚类方法的 PDR 差异很小（Table 5/7）

对 HC、Birch、Mean Shift、DBSCAN 四种聚类重复同一流程。论文称方法选择影响很小（Table 5/7 极差 0.010）；缩比实验用 0.10 的极差上限。若某方法退化，代码会回退到 KMeans 并记录 method_used='fallback_kmeans'。缩比模型的簇本身不纯（PDR≈0.7–0.85），此时换聚类方法会实际改变判为中毒的图；极差超限更多反映代理设定的可分性，而不是论文四种方法在干净簇（PDR≈0.95）上的差异。

### B14 NTE 的标签恢复对温度/截断变体不敏感

Eq.9/10 未写温度，官方代码用 T=1 并对单项 NTE 截断（5 或 4）。这里扫描 T∈{0.5,1,2,3} 与 cap∈{无,4,5}，以 (T=1, cap=5) 为基准统计 argmax 一致率。若一致率很高，说明实现细节不影响标签恢复；否则论文的公式歧义有实际后果。两组灵敏度分开报告：只改 cap（T=1）与只改温度（cap=5），以便看出是哪一项在起作用。

## 产物清单

```json
{
  "seed": 2024,
  "torch": "2.7.1+cpu",
  "numpy": "1.26.4",
  "protocol": {
    "train_per_class": 2000,
    "test_per_class": 200,
    "ood_per_class": 50,
    "epochs": 30,
    "poison_rate": 0.1,
    "target": 0,
    "temperature": 2.0,
    "lambda": 3.0,
    "proxy_lambda": 2.2,
    "robust_augment": true
  },
  "metrics": {
    "clean": {
      "ba": 0.807,
      "asr": 0.115
    },
    "badnets": {
      "ba": 0.7905,
      "asr": 0.937
    },
    "blend": {
      "ba": 0.804,
      "asr": 0.984
    }
  },
  "sha256": {
    "data/cifar10_train.npz": "da97c6177d490c60648bbbcaf0772f5f11d9e9df4234e5d928954225d46dad9b",
    "data/cifar10_test.npz": "cffb4a2bbfe2b0a6fabaa7dfa169e38795b27c7e2cb4253730723d9fb556ca68",
    "data/cifar100_ood.npz": "4cd895e03d8a1f413db579bc6b30064c35daccbfef82c2263d40e58f0d197156",
    "checkpoints/clean.pt": "d4d6a1324d9f799bd517ef916aaa50932aaa4a8521261c8bf744ac47ee67c61a",
    "checkpoints/badnets.pt": "1e44a4739c5553e2ab55ad4eddf3de706e8ca346bc4c1981f7aab0889a1021ff",
    "checkpoints/blend.pt": "4044dd7bf5231a583a182a00e785572c1ac578b87ec86e6d4a0c4cb7d4256831"
  },
  "extra": {
    "detection": {
      "clean": {
        "target": 3,
        "margin": -8.32200601062462,
        "zscore": 1.7727143493215416,
        "is_backdoored": false,
        "counts": {
          "0": 50,
          "1": 50,
          "2": 50,
          "3": 50,
          "4": 50,
          "5": 50,
          "6": 50,
          "7": 50,
          "8": 50,
          "9": 50
        }
      },
      "badnets": {
        "target": 0,
        "margin": 25.249821610226874,
        "zscore": 2.7345458253069683,
        "is_backdoored": true,
        "counts": {
          "0": 50,
          "1": 50,
          "2": 50,
          "3": 50,
          "4": 50,
          "5": 50,
          "6": 50,
          "7": 50,
          "8": 50,
          "9": 50
        }
      },
      "blend": {
        "target": 3,
        "margin": -4.927846979298074,
        "zscore": 2.0073733450334057,
        "is_backdoored": false,
        "counts": {
          "0": 50,
          "1": 50,
          "2": 50,
          "3": 50,
          "4": 50,
          "5": 50,
          "6": 50,
          "7": 50,
          "8": 50,
          "9": 50
        }
      }
    },
    "nte_recovery": {
      "badnets": 0.5425531914893617,
      "blend": 0.4742268041237113
    },
    "gates": {
      "clean_ba_min": true,
      "badnets_ba_min": true,
      "blend_ba_min": true,
      "badnets_asr": true,
      "blend_asr": true,
      "clean_not_detected": true,
      "badnets_detected_target": true
    },
    "epochs": 30,
    "device": "cpu",
    "cifar_source": "fastai"
  }
}
```