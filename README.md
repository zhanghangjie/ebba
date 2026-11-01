# EBBA / EBBA+ 论文验证套件

对 ICML 2024 论文 **Energy-based Backdoor Defense without Task-Specific Samples and Model Retraining**
（Gao, Chen, Sun, Li, Li, Shao；PMLR 235）的可复现审计。一条命令产出逐条判定、
实测数值与自包含 HTML 报告。

- **代理档**（默认，`ebba:cpu` / `ebba:gpu`）不需要 GPU 也能完整运行；约 2–4 分钟。
- **全量档**（Plan A，ResNet-18 / 完整 CIFAR-10 / 200 epochs）需要 NVIDIA GPU；
  本机 RTX 5060 Laptop 实测训练 **2 小时 50 分**（三个模型），评估约 2 分钟。
- 代理档运行期不需要联网：数据子集、三个模型与 manifest 在镜像构建阶段生成并做 SHA256 校验。
- 全量档的完整数据与 checkpoint 放在宿主机挂载目录，训练可断点续跑。
- 所有 Notebook 带中文注释；数值逻辑在 `src/ebba`，可脱离 Jupyter 用 `pytest` 运行。

## 获取

- **源码仓库**：<https://github.com/zhanghangjie/ebba>
- **Docker 镜像**：`zhanghangjie/ebba:full`（全量档，12.3 GB，digest
  `sha256:a4543d1bec29b565222f91fc51dc137bdf3d3f8bad2295f8fd22efa1d2c0e492`），
  内含 ResNet-18 checkpoint、完整 CIFAR-10 与 22,630 张任务无关 OOD 图像，
  一条命令离线重算全量指标，无需挂载、无需联网。
- 代理档镜像（`ebba:cpu` / `ebba:gpu`）未发布，按下方“快速开始”在本地构建。

仓库只提交源码、Notebook、文档、测试与运行结果（`reports/`、`reports-full/` 下的
JSON、HTML 与执行后 Notebook，约 1.5 MB）。数据子集、checkpoint、镜像 tar、`release/`
构建日志与论文 PDF 不进仓库：前三者可用 `scripts/build_artifacts.py`、
`scripts/train_full.py` 重建或直接从镜像取得；论文 PDF 受版权限制，报告只引用页码与公式。

## 快速开始

```bash
# 1. 构建（首次约 60-75 分钟：下载 CIFAR、训练 clean/BadNets/Blend 三个代理模型）
docker compose build ebba

# 2. 完整验证（约 2 分钟，CPU）
docker compose run --rm ebba verify

# 3. 打开报告
#    reports/report.html      自包含 HTML，可离线打开
#    reports/verdicts.json    机器可读的逐条判定
```

其他命令：

```bash
docker compose run --rm ebba test                    # pytest
docker compose run --rm ebba env                     # 环境与产物指纹
docker compose run --rm ebba retrain --attack badnets
docker compose run --rm ebba verify --fast           # 更小的实证规模
docker compose run --rm ebba verify --only 01_formula_audit 03_table_audit
docker compose --profile gpu run --rm ebba-gpu verify --device cuda
```

如果所在网络无法访问 Docker Hub，可换用镜像源（构建阶段仍需要访问 PyPI 与
download.pytorch.org；运行期不需要网络）：

```bash
EBBA_BASE_IMAGE=docker.m.daocloud.io/library/python:3.11.10-slim-bookworm docker compose build ebba
```

数据集默认从 fast.ai 的 S3 PNG 镜像下载（与 `cs.toronto.edu` 的原版图片逐像素相同，
但在国内网络通常快一个数量级）；如需改用 torchvision 官方源：

```bash
EBBA_CIFAR_SOURCE=torchvision docker compose build ebba
```

`verify` 的退出码只反映执行故障。`CONTRADICTED` 是预期结论，不是运行失败。

### GPU（可选）

`ebba-gpu` profile 使用 `torch 2.7.1+cu128`，构建时检查 wheel 的编译架构包含 `sm_120`
（RTX 50 系）。构建阶段不需要可见 GPU；只有 `verify --device cuda` 会真正把 logits 计算放到 GPU。
代理产物仍由构建脚本在 CPU 上生成，因此两个镜像的 checkpoint 与 manifest 相同。

```bash
docker compose --profile gpu build ebba-gpu
docker compose --profile gpu run --rm ebba-gpu verify --device cuda
```

已在 RTX 5060 Laptop（capability 12.0，arch list 含 `sm_120`）上实测：7 个 notebook 全部执行，
39 条判定与 CPU 镜像**完全一致**（3 / 13 / 18 / 5，无 ERROR）。但 GPU 的卷积归约顺序与 CPU
不同，logit 末位差异会被 t-SNE 聚类放大，因此数值不逐位相同：例如 B02 干净模型的最高能量
标签 CPU 为类 3、GPU 为类 2；B07 blend PDR 0.739 → 0.854；B08 BadNets 极差 0.137 → 0.263。
**CPU 镜像是权威结果**，GPU 用于确认可选加速路径可运行且判定稳健。

`retrain` 使用与构建脚本相同的 `robust` 增广设置，并同样从 `seed=2024` 重新初始化；
它应当逐位复现构建指标（实测 BadNets：BA 0.7905、ASR 0.937，与内置产物差值为 0）。
若指标有偏差，说明训练管线或运行环境发生了变化，而不是“正常的随机波动”。

## 全量档（Plan A）：ResNet-18 / 完整 CIFAR-10 / 200 epochs

代理档用 SmallCNN + 2 万张图回答“机制是否存在”；全量档用论文骨干和完整数据回答
“换成论文规模后，指标是否朝论文数值移动”。两者使用**不同的目录**，互不覆盖：

| | 代理档 | 全量档 |
|---|---|---|
| Profile | `EBBA_PROFILE=proxy`（默认） | `EBBA_PROFILE=full` |
| 模型 | SmallCNN | ResNet-18（CIFAR 变体，11.2M） |
| 数据 | 20k / 2k 子集 | 50k / 10k 完整 CIFAR-10 |
| 训练 | 30 epochs，cosine | 200 epochs，SGD lr=0.01，milestones 50/100/150，batch 64 |
| 产物 | `artifacts/`、`reports/` | `artifacts-full/`、`reports-full/`、`checkpoints-full/` |
| 攻击 | BadNets（白块）、Blend（程序化图案） | BadNets（**黑块**）、Blend（**hello kitty**，α=0.2） |
| OOD | CIFAR-100 子集 | CIFAR-100 test 10k + **GTSRB test 12,630** |
| 运行 | CPU，约 2–4 分钟 | GPU，训练约 2.5–3 小时（RTX 5060 Laptop 实测 2h50m）；评估约 2 分钟 |

全量档**不改变**代理档的验证结果。`verify` 会强制使用 proxy 路径；全量评估要用
`verify-full`。

### 1. 构建 GPU 镜像

```bash
docker compose --profile train build ebba-train
```

首次构建仍会生成代理档产物（约 60–75 分钟）；之后修改 `scripts/` 或 `src/` 时大多命中缓存。
镜像基于 `torch 2.7.1+cu128`，适用于 RTX 50 系（sm_120）。

### 2. 首次全量训练（可断点续跑）

```bash
# 前台运行；首次会自动下载完整 CIFAR-10/CIFAR-100 到 artifacts-full/data
docker compose --profile train run --rm ebba-train train-full --all --resume auto
```

或后台运行：

```bash
docker compose --profile train up -d ebba-train
docker compose --profile train logs -f ebba-train
```

训练顺序是 clean → badnets → blend；每个攻击的训练目录独立：

```
checkpoints-full/
  clean/    last.pt  last_prev.pt  best.pt
  badnets/  last.pt  last_prev.pt  best.pt
  blend/    last.pt  last_prev.pt  best.pt
  train.log
```

默认**每个 epoch 结束保存一次**；如需更细粒度，加 `--checkpoint-every-steps 100`
（最多丢失 100 个 batch）。最终推理 checkpoint 写入
`artifacts-full/checkpoints/{clean,badnets,blend}.pt`，指标写入 `artifacts-full/metrics/`，
最后生成 `artifacts-full/manifest.json`（带 SHA256 与 `profile=full`）。

### 3. 睡眠 / 关机 / 中断后继续

**推荐做法：睡前先停容器。**

```bash
docker compose --profile train stop ebba-train   # SIGTERM；compose 给 2 分钟保存时间
```

进程会在当前 batch 结束时写 `last.pt` 后退出。第二天直接重跑同一条命令：

```bash
docker compose --profile train run --rm ebba-train train-full --all --resume auto
```

`--resume auto` 会读取每个攻击目录的 `last.pt`（缺失时用 `last_prev.pt`），恢复模型、
优化器、scheduler、AMP scaler、当前 permutation 和 RNG 状态。配置哈希会校验数据文件、
epoch/批大小/学习率/增广等参数；换了配置会拒绝静默续跑。

**如果直接让电脑睡眠：** 进程只是被挂起，睡眠本身不会触发 checkpoint。Windows 睡眠 /
Modern Standby、Docker Desktop 和 CUDA context 都不保证在唤醒后仍可用；若容器已退出，
最多丢失最后一个 epoch 的进度。因此可靠恢复点仍是磁盘上的 `last.pt`。训练期间可运行
`scripts/keep_awake.ps1` 阻止系统睡眠：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\keep_awake.ps1
# 需要允许关屏时：
powershell -ExecutionPolicy Bypass -File scripts\keep_awake.ps1 -AllowDisplayOff
```

或临时修改电源计划（结束后记得改回）：

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
# 恢复默认（示例）
powercfg /change standby-timeout-ac 30
```

### 3b. 论文触发器与 OOD（P0/P1）

全量档默认使用论文实现细节里的触发器：

- **BadNets 黑块**：右下角 3×3、像素值 0（`EBBA_BADNETS_VALUE=0`）。代理档仍是早期训练产物
  使用的白块，避免让已有 proxy 产物失效。
- **Blend hello kitty**：α=0.2。原图不随仓库分发；把本地图片转成 32×32 数组：

  ```bash
  python -I scripts/prepare_trigger.py --image /path/to/hello_kitty.jpeg \
      --out artifacts-full/data/blend_hello_kitty.npy
  ```

  训练/评估时 `config.py` 会自动发现该文件；也可以用 `EBBA_BLEND_PATTERN` 指定路径。
  文件缺失时退回程序化图案，并把“surrogate”写进训练配置哈希（续跑会因此失效）。

- **OOD 增加 GTSRB**：`EBBA_OOD_GTSRB=1`（compose 的 full/train 服务已默认开启）。如果
  `cifar100_ood.npz` 已经存在，用：

  ```bash
  python -I scripts/rebuild_full_ood.py \
      --gtsrb-root artifacts-full/data/raw/gtsrb --data-dir artifacts-full/data
  ```

  GTSRB 会 resize 到 32×32，标签偏移 +100；原 CIFAR-100-only 数组备份为
  `cifar100_ood_cifar100only.npz`。改完数据后必须重写 manifest：

  ```bash
  docker compose --profile train run --rm ebba-train train-full --finalize-only
  ```

- **候选池**：全量档为 5,000 张中毒 + 1,000 张干净目标类（共 6,000），代理档仍是小样本。
  可用 `verify-full` 的 `--max-poisoned/--max-clean-target` 临时缩小。

> 触发器或 OOD 一变，旧 checkpoint 就不能续跑。重训时显式加 `--resume none --force`，
> 不要用 `--resume auto`。

### 4. 全量评估与报告

```bash
docker compose --profile full run --rm ebba-full verify-full --device cuda
```

产物：

```
reports-full/
  full_metrics.json                 机器可读的全量指标
  full_report.html                  自包含 HTML
  executed/07_full_comparison.ipynb 执行后的对比 Notebook
```

`07_full_comparison.ipynb` 并列展示 **论文 CIFAR-10 / ResNet-18 full / SmallCNN proxy**，
包括 BA/ASR、PDR/F1、去除后 ASR、λ=2.2/3 检测、Fig.1/Fig.2、NTE 恢复、四种聚类与
定向规则敏感性。报告里另有两组 **诊断列**（不是论文方法）：

- **池内二次阈值**：在已定向的中毒簇内对峰值 NTE 拟合 1/2 组分高斯混合（BIC 门控），
  低分量改判为干净。只有当 ΔBIC>10 且最小分量权重 ≥2% 才生效。
- **NTE 扫描**：`cap5_T1`（官方实现）、`uncapped_T1`（Eq.9-11 字面）、`cap5_T2`、
  `uncapped_T2`、`cap5_T1_classcenter`。注意 T=1 时单损坏 NTE 上界是 `e−1≈1.72`，
  所以任何 ≥2 的 cap 都不生效——`cap5_T1` 与 `uncapped_T1` 相同是公式推论。

如果想知道 PDR 对聚类特征空间有多敏感，可以单独跑诊断（只读产物，不写报告）：

```bash
docker compose --profile full run --rm --entrypoint python ebba-full \
  -I scripts/diagnose_clustering.py --attack blend --n-poisoned 1000 --n-clean 500 --device cuda
```

它会并列打印 NTE 向量、损坏图像像素、ResNet 嵌入三种特征空间，以及 t-SNE / 直接标准化
两种聚类路径的 oracle 与无标签规则 PDR。

### 5. 实测结果与“复现百分比”

以下是本机（RTX 5060 Laptop、torch 2.7.1+cu128）跑完 200 epochs 后的**实测值**，
不是规划估计。完整数据在 `reports-full/full_metrics.json`，可读版在
`reports-full/full_report.html`。

| 指标 | full 实测（BadNets / Blend） | 论文 CIFAR-10 | 差值 |
|---|---|---|---|
| BA | 92.02 / 94.15 | 91.22 / 91.35 | +0.80 / +2.80 |
| ASR | 95.93 / 99.97 | 99.36 / 99.97 | −3.43 / 0.00 |
| PDR | 0.896 / 0.763 | 0.96 / 0.95 | −0.064 / −0.187 |
| F1 | 0.905 / 0.815 | 0.96 / 0.94 | −0.055 / −0.125 |
| 去除后 ASR（池内） | 10.4 / 23.7 | 2.92 / 5.79（全测试集） | 口径不同，明显更高 |
| 四种聚类 PDR 极差 | 0.004 / 0.076 | 0.010 | Blend 超出 |
| 检测（λ=2.2 / λ=3） | 3/3 / 0/3 检出 | 干净不检出、后门检出 | **不成立** |

必读的三条实测发现：

1. **BA 与 Blend ASR 基本对齐，BadNets ASR 偏低。** 干净模型 BA 94.61；BadNets
   把 BA 拉低 2.4 个点、ASR 95.93；Blend 的 ASR 99.97 与论文逐位相同。BadNets 的
   ASR 低 3.4 个点，可能与 10% 投毒率、dirty-label 和 200 epoch 的强拟合有关。
2. **检测在本设定下不成立。** λ=3 时三个模型全部“未检出”（K=10 的单标签 z 上界正好
   是 3，严格 >3σ 不可达，见 A20）；λ=2.2 时 BadNets z=2.78、Blend z=2.29，但**干净
   模型 z=2.43 也被判为后门**。也就是说，ResNet-18 的 OOD 能量分布比 SmallCNN 更尖，
   固定 λ 无法把干净模型与后门模型分开。这是对论文 λ=3 设定的实质性反例，不是实现噪声。
3. **PDR/F1 明显低于论文，且对特征空间高度敏感。** 主配置（损坏后图像嵌入 + 峰值 NTE
   定向）给出 0.896/0.763；诊断中换用损坏后图像嵌入 + t-SNE 得到 0.998/0.386，换用
   NTE 向量 + 目标 margin 得到 0.915/0.555。论文只写了“用 t-SNE 把扰动图像投影到特征
   空间”，没有规定特征表示和“哪个簇是中毒簇”的无标签规则；不同合理解释能让 PDR 在
   0.39–1.00 之间移动。本套件选择**标准化嵌入 + Birch + 峰值 NTE**，因为它在两次重复
   运行中逐位一致；t-SNE 只保留作可视化。

Fig.1 / Fig.2 的方向成立：BadNets 的 target label 在 OOD 上的平均排名 2.17（干净模型
4.49），top-3 68.8% vs 30.3%；Blend 较弱（3.92、32.2%）。损坏后 target 概率下降的比例
为 83.7%/86.8%，真值概率上升为 70.9%/76.2%，支持能量转移机制。

**“复现百分比”不要用一个数概括。** 8 个主数值（BA/ASR/PDR/F1 各两种攻击）中，
3 个落在论文值 ±3 个百分点内（两个 BA、Blend ASR），8 个方向一致；PDR/F1 四项均未达到
0.95。若只看“方向 + 量级”，结论是支持；若看“PDR 数值”，结论是部分支持，且依赖论文
未写明的聚类细节。没有 WaNet/FIBA/DUBA，不能把这组比例外推到论文全部攻击。

差异来源必须写清楚：

1. OOD 用 CIFAR-100 test 10k（可选 GTSRB test，resize 32×32），不是论文的 Internet 图像；
2. Blend 使用固定粉色图案，不是 “hello kitty”；
3. 论文没有给出无标签时如何认定“中毒簇”；本套件主规则是簇内平均峰值 NTE，另报
   NTE 向量特征空间与 oracle 上界；
4. 论文未公开的初始化、数据顺序等细节无法完全对齐；
5. 全量档仍不覆盖 WaNet/FIBA/DUBA、GTSRB 主实验、ImageNet、语音/文本。

因此正确的结论是“在 ResNet-18/200 epochs 的独立设定下，PDR/F1 达到 X，与论文差 Y”，
而不是“逐位复现了论文的 96%”。

## 判定语义

| 判定 | 含义 |
|---|---|
| `CONFIRMED` | 所有核心子检查在容差内通过。仅 Tier A 可获得。 |
| `PARTIAL` | 部分通过；或缩比实验方向一致但规模不同。`scope: proxy` 永不判 CONFIRMED。 |
| `CONTRADICTED` | 可直接核对的数值超差，或普适断言存在反例。 |
| `NOT_REPRODUCIBLE` | 需要容器外资源。附原因与所需资源。 |
| `ERROR` | 运行异常。正常交付中不允许出现。 |

三层验证：

- **Tier A（20 条，精确层）**：附录算例、公式一致性、表格算术、文字断言、证明步骤、官方代码静态审计。
- **Tier B（14 条，缩比实证层）**：CIFAR-10 子集 + SmallCNN + BadNets/Blend，检验方法机制方向。
- **Tier C（5 条，不可复现层）**：ResNet18/ImageNet、语音文本、13 组自适应训练、all-to-all/full-target。
  显式记录，避免报告出现 PENDING，也避免把“没做”读成“通过”。

## 总体判定

39 条声明在最终镜像中的分布：

| 判定 | 数量 | 说明 |
|---|---|---|
| `CONFIRMED` | 3 | 全部在 Tier A |
| `PARTIAL` | 13 | 方向成立但幅度/规模不足，或依赖未说明的假设 |
| `CONTRADICTED` | 18 | 可核对断言不成立（15 条 Tier A + 3 条 proxy：B06/B08/B14） |
| `NOT_REPRODUCIBLE` | 5 | Tier C，容器外资源 |
| `ERROR` / `PENDING` | 0 / 0 | — |

同一镜像内连续运行两次，39 条判定与全部离散指标逐位一致。

## 精确层（Tier A）结果

运行 `docker compose run --rm ebba verify` 后由代码写出；以下为当前实现的判定。

| ID | 判定 | 声明 | 实测要点 |
|---|---|---|---|
| A01 | CONTRADICTED | 附录 B 算例可复算 | 精确能量 `[4.7243, 4.7185, 4.6691]` vs 论文 `[4.82, 4.43, 4.62]`；论文的 `E0>E2>E1` 实为 `E0>E1>E2` |
| A02 | CONFIRMED | Eq.3 是“最大值置 0” | `[3,8,-2] → [3,0,-2]`，与减最大值不同 |
| A03 | CONTRADICTED | 三处归一化一致 | 论文 Eq.5、官方 README、官方 Notebook 为三种公式 |
| A04 | CONTRADICTED | Eq.7 下标/分母自洽 | `K+1` 项与除以 `K` 不自洽，z-score 偏移 0.655 |
| A05 | CONFIRMED | TE/NTE 恒等式 | 数值误差 < 1e-9；均值恒为 0 |
| A06 | CONTRADICTED | Eq.9/10 使用同一温度 | T=1 与 T=2 的 argmax 一致率 0.953 |
| A07 | PARTIAL | Eq.30 代数等价 | 在 `a+t>0` 时成立，论文未声明该前提 |
| A08 | CONTRADICTED | Lemma 2 逐损坏普适 | 同时满足两条不等式的 (图, 损坏) 比例 0.761；按损坏取平均仅 0.763 < 0.90 |
| A09 | CONTRADICTED | Lemma 1 脏标签核和 | 目标类 190 vs 100 时 100% 占优；等量 100 vs 100 时仅 8.3% 占优（引理无样本数条件） |
| A10 | CONTRADICTED | 干净标签 Eq.35 放缩 | 反例 `a=0,t=1,b=1.5,n=2`，方向相反 |
| A11 | CONTRADICTED | BA ≥ (K−1)/K·ba | 2 行反例：ImageNet FIBA、DUBA |
| A12 | CONTRADICTED | 所有攻击 ASR ≥ 98.8% | ImageNet Blend 98.62、FIBA 98.59 |
| A13 | CONTRADICTED | GTSRB 测试集大小 | 论文 12603，标准 12630 |
| A14 | CONFIRMED | 聚类方法 PDR 差异小 | 极差 0.010；但 Table 5 与 Table 7 完全相同 |
| A15 | CONTRADICTED | Table 4 文字结论 | BA 最好 13/15 行；ASR 最低仅 5/15 行 |
| A16 | CONTRADICTED | Table 3 全面最优 | 5 行存在严格更优基线（平局不计） |
| A17 | CONTRADICTED | Fig.80 符号一致 | 首句“<0 判后门”与 Eq.8 相反 |
| A18 | CONTRADICTED | 官方代码实现论文协议 | 8 项协议检查 7 项不符 |
| A19 | PARTIAL | 自适应检测率 > 99% | 13 组参数中 0.985 失效，按组 92.3% |
| A20 | CONTRADICTED | λ=3 在 K=10 上可满足 Eq.8 | `max_k (E−μ)/σ ≤ √(K−1) = 3`，严格 `>3σ` 仅在其余 9 个标签能量完全相等时逼近，实际上不可达 |

> `CONTRADICTED` 针对的是**某条可核对断言**，不是“EBBA 方法无效”。例如 A01 的 clamp 后方向仍然成立，
> 错的是未 clamp 表的数值与排序；A16/A15 否定的是“全面最优/更低”的措辞，不是方法本身。

## 缩比实证设计（Tier B）

| 项目 | 设定 |
|---|---|
| 数据 | CIFAR-10 固定子集：20,000 训练 / 2,000 测试（每类均衡） |
| 任务无关图像 | CIFAR-100 测试集 5,000 张（代理“互联网图像”，非论文原数据） |
| 模型 | SmallCNN（32/64/128/128，约 1.2M 参数） |
| 训练 | SGD lr=0.05、momentum=0.9、wd=5e-4、cosine、batch=128、30 epochs；轻度噪声/模糊/对比度增广 |
| 攻击 | BadNets（右下 3×3 白块）、Blend（固定粉色图案，α=0.2），目标类 0，投毒率 10% |
| 损坏 | 19 类 × 4 强度 + 除以 2..5 = 80 种，逐 `(image, corruption, severity)` 派生 RNG |
| 检测阈值 | 缩比实验用 λ_proxy=2.2；论文的 λ=3 在 K=10 时等于 z 的上界（A20），严格裕量不可达 |
| 门槛 | clean BA ≥ 0.70；攻击 BA ≥ 0.68；两种攻击 ASR ≥ 0.90；干净模型不被误报；BadNets 在 λ_proxy 下检出目标 0 |

门槛只认证**实验台可用**，不认证论文结论：NTE 恢复率等是被检验的结果，记录但不设门槛，
避免为了通过构建而挑参数。构建指标与 checkpoint SHA256 写入 `artifacts/manifest.json`，报告中原样展示。

确定性：全局 seed=2024、`PYTHONHASHSEED=0`、`torch.use_deterministic_algorithms(True, warn_only=True)`、
固定 4 线程、`num_workers=0`、t-SNE `random_state=0` 且用精确解（`method="exact"`）。
离散统计（B05 恢复率、B06 标签分布）在本地 torch 2.9 / sklearn 1.9 与容器
torch 2.7.1 / sklearn 1.5.2 之间完全一致，B02 的能量只差约 1e−4（判定不变）；
但 **t-SNE 是输入低比特扰动的非连续函数**，聚类相关指标会随环境移动。
本项目中最大的一次移动是 B07 BadNets：本地 checkpoint 上 PDR 0.777，最终镜像里 0.374，
原因是两个候选簇的 margin 均值（7.55 与 9.44）排序随 checkpoint 改变，定向规则选中了
更纯但更小的簇；oracle 上界仍是 0.626。其余聚类指标移动约 0.02–0.15。
因此跨环境只承诺“判定方向一致”，不承诺聚类指标 ±2%；**同一镜像内可重复**，
并以镜像自带的 checkpoint 为权威结果。B08 在两种环境下给出了不同极差，
这也是它在缩比层被判 CONTRADICTED 的原因之一：
簇本身不纯时，“换聚类方法结果几乎不变”这一命题没有可检验的稳定性。

### Tier B 结果

`scope: proxy` 的判定上限是 `PARTIAL`；下表是当前 checkpoint 的实测值（BadNets / Blend）。

| ID | 判定 | 声明 | 实测要点 |
|---|---|---|---|
| B01 | PARTIAL | 目标标签在 top 输出中更常见（Fig.1） | BadNets top-3 0.733、平均排名 1.70；Blend 0.327 / 4.41，与干净模型（0.312 / 4.47）无异 |
| B02 | PARTIAL | 无需任务样本即可检测后门（Table 2） | BadNets margin +25.25（z=2.73，目标 0）；Blend margin −4.93（最高能量在类 3），未检出 |
| B03 | PARTIAL | λ 可用区间形状（Fig.80） | BadNets 可用区间 [1.8, 2.7]，与论文 [1.8, 5.7] 重叠；Blend 可用区间 [1.8, 2.0]，不含 λ_proxy=2.2 |
| B04 | PARTIAL | 损坏使真值上升、目标下降（Fig.2） | 真值概率上升 0.768、目标下降 0.828；排名前移仅 0.145（真值原本已排第 1.35 位，空间退化） |
| B05 | PARTIAL | NTE argmax 恢复原始标签（Eq.12） | 中毒图恢复率 0.479 / 0.397（随机 0.10，论文级 0.85 未达） |
| B06 | CONTRADICTED | 干净目标类图的 NTE 转移接近随机 | 恢复标签高度集中（最大类占 0.451 / 0.444，卡方 p≈1e−55 / 1e−48），并非论文描述的“随机” |
| B07 | PARTIAL | EBBA+ 的 PDR/F1（Table 3） | PDR 0.374 / 0.739，F1 0.530 / 0.799；峰值 NTE 规则 0.626 / 0.261；真实标签定向上界 0.626 / 0.739 |
| B08 | CONTRADICTED | 四种聚类方法差异小（Table 5/7） | PDR 极差 0.137 / 0.151（上限 0.10）；BadNets 上 0.374–0.511。缩比簇不纯（PDR≈0.6–0.8），方法选择会实际改变结果；论文的 0.010 极差对应 PDR≈0.95 的干净簇 |
| B09 | PARTIAL | 去除后门且保持良性精度（Table 4） | ASR 降幅 0.374 / 0.734；全测试集 BA 降幅估计 0.0025 / 0.0076（≤0.05） |
| B10 | PARTIAL | BA ≥ (K−1)/K × ba | 裕量 +0.077 / +0.073 |
| B11 | PARTIAL | 均衡选择减少误检（Fig.78/79） | BadNets 目标领先 +2.24 → 不均衡时 −2.12；误标 0 → 1 |
| B12 | PARTIAL | 双边检验可检出越界抑制 | 单边漏检；双边 \|E−μ\|>λσ 检出目标 0 |
| B13 | PARTIAL | 多目标能量统计 | 注入第二异常后 top-2 = [0, 1] |
| B14 | CONTRADICTED | NTE 对温度/截断不敏感 | 只改 cap 一致率恒为 1.0；改温度降到 0.922 / 0.934（论文未写 Eq.9 的温度） |

**B07 的簇定向是套件假设**：论文只说 t-SNE + Birch，没有给出无标签时认定中毒簇的规则。
本套件用簇内平均“目标 logit − 最强非目标 logit”定向；这个规则偏向**纯度**而非**召回**：
最终镜像里 BadNets 的两个簇是 `203 张 / 58.6% 中毒 / margin 7.55` 与
`78 张 / 91% 中毒 / margin 9.44`，它选中后者，于是精确率 0.910 但 PDR 只有 71/190=0.374；
若按簇大小（中毒图占多数）或峰值 NTE 选簇，则得到 0.626——也就是 B07 记录的 oracle 上界。
Blend 上 margin 规则恰好选中正确的 169 张大簇（0.739）。
因此 B07 的 `PARTIAL` 来自“机制方向成立、但无标签定向规则会显著改变召回”，
而不是某个规则被事后挑选；三种规则的数值都在结果 JSON 里并列。

## 目录

```
src/ebba/
  config.py            路径、种子、类别数、阈值
  data.py              CIFAR 子集、均衡/不均衡选择、候选池
  models.py            SmallCNN、训练、checkpoint
  attacks.py           BadNets、Blend
  corruptions.py       80 种确定性损坏
  energy.py            Eq.3-8 与三种归一化
  transferred.py       Eq.9-12、温度与截断变体
  clustering.py        t-SNE + HC/Birch/MeanShift/DBSCAN
  metrics.py           BA/ASR/PDR/F1/排名/AUC
  artifacts.py         manifest 与 SHA256 校验
  claims/
    registry.yaml      39 条声明（唯一事实源）
    formulas.py        A01-A06, A17
    theory.py          A07-A10
    audit.py           A11-A19
    empirical.py       B01-B14
    limits.py          C01-C05
  report.py            verdicts.json / Markdown / 自包含 HTML

notebooks/             00_overview … 06_verdicts + 07_full_comparison（中文注释）
scripts/               build_artifacts / train / train_full / full_eval /
                       run_papermill / make_report / check_env / keep_awake.ps1
reference/             官方 EBBA.ipynb（只读，静态解析）与审计记录
tests/                 44 个单元测试
docker/                Dockerfile、compose、entrypoint、锁定依赖
reports/               代理档运行产物（挂载）
artifacts-full/        全量档数据、checkpoint、metrics、manifest（挂载）
checkpoints-full/      全量档可续跑工作目录（挂载）
reports-full/          全量档 JSON / HTML / 执行后的 notebook（挂载）
```

## 本地运行（不用 Docker）

需要 Python 3.11 与 PyTorch。

```bash
python -m pip install -r docker/requirements-common.txt -r docker/requirements-cpu.txt
python scripts/build_artifacts.py --device cpu
python scripts/run_papermill.py
python scripts/make_report.py
python -m pytest -q tests
```

## 本套件不做什么

- 代理档不复现 Table 3/4 的精确数字，不训练 ResNet18、不跑 200 epochs。
- 全量档（Plan A）训练 ResNet18/200 epochs，但仍不实现 WaNet/FIBA/DUBA，也不跑
  GTSRB 主实验或 ImageNet；这些仍是 Tier C 的不可复现项。
- 不执行官方 PaddlePaddle Notebook；只解析其 JSON 做协议审计。
- 不包含论文 PDF（版权）；报告只引用页码、公式与转录后的数值。
- 不把 Tier B 的 `PARTIAL` 写成“复现了论文的 95% PDR”。
- 不把全量档的独立复核写成论文表格的逐位复现。

## 参考

- 论文：<https://proceedings.mlr.press/v235/gao24a.html>
- 官方代码：<https://github.com/ifen1/EBBA>（`reference/EBBA.ipynb`）
- 官方代码与论文差异：`reference/AUDIT.md`
