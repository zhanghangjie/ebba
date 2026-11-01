# EBBA 可复现验证项目：代码模块讲解

本文按**职责**把仓库拆成 9 个模块，逐个说明"它解决什么问题、关键函数怎么写的、为什么这样写"，并在关键代码处附中文注释。

> **关于注释位置**：注释写在本文档的代码片段里，没有写回源文件。原因是镜像 `ebba:full` 已经构建完成并通过 38 项指标对比，源码文件的内容参与构建与校验；此时改源文件会让"你读到的代码"和"镜像里实际跑的代码"不是同一份。要看原始实现，直接打开对应文件即可，片段都标了行号。

---

## 0. 先把仓库看成一个流水线

```
paper/paper.pdf                论文原文（只读，用于逐条摘录声明）
        │
        ▼
src/ebba/claims/registry.yaml  ← 39 条声明的唯一事实源（ID/位置/容差/所属 notebook）
        │
        ├── claims/formulas.py   A01–A06, A17, A20   纯公式复算（无模型）
        ├── claims/theory.py     A07–A10             引理代数与核和模拟
        ├── claims/audit.py      A11–A16, A18, A19   表格算术 / 文字断言 / 官方 notebook 静态审计
        ├── claims/empirical.py  B01–B14             缩比模型上的机制验证
        └── claims/limits.py     C01–C05             容器内不可复现，写明原因
                    │
                    ▼
        reports/claims/<ID>.json  ← 每条一个 JSON（verdict + 实测值 + 容差 + 中文说明）
                    │
                    ▼
        src/ebba/report.py  →  reports/verdicts.json / report.md / report.html
```

支撑这条主线的，是下面三层"实验台"：

```
数据层   data.py（固定子集） + attacks.py（投毒） + corruptions.py（80 种损坏）
模型层   models.py（SmallCNN / ResNet18CIFAR）+ 训练脚本
方法层   energy.py（EBBA, Eq.3–8） + transferred.py（EBBA+, Eq.9–12） + clustering.py（聚类）
```

一句话概括设计原则：**能被加法的都放在 `claims/` 里做成可判定对象，实验台只提供数值，不自己下结论。**

---

## 模块 1：配置与确定性

### `src/ebba/config.py`（132 行）

整个项目只有这一个"参数开关面板"。两套 profile 不是两份代码，而是同一份代码的两组常量。

```python
# src/ebba/config.py:20-23
PROFILE = os.environ.get("EBBA_PROFILE", "proxy").strip().lower()
if PROFILE not in ("proxy", "full"):
    raise ValueError(f"EBBA_PROFILE must be 'proxy' or 'full', got {PROFILE!r}")
IS_FULL = PROFILE == "full"
# 中文注释：profile 只允许两个值，写错立即报错，而不是悄悄退回默认——
# 否则一个拼写错误会让 full 实验跑成 proxy，报告却仍然"通过"。
```

路径也随之分叉，两个 profile 的产物**永不互相覆盖**：

```python
# src/ebba/config.py:28-37
PROJECT_ROOT = Path(os.environ.get("EBBA_ROOT", Path(__file__).resolve().parents[2]))
_DEFAULT_ARTIFACTS = PROJECT_ROOT / ("artifacts-full" if IS_FULL else "artifacts")
ARTIFACTS_DIR = Path(os.environ.get("EBBA_ARTIFACTS", _DEFAULT_ARTIFACTS))
# 中文注释：训练中途的 workspace 故意放在 ARTIFACTS_DIR 之外。
# 因为 artifacts 里的文件会被 manifest.json 记录 SHA256，
# 半写成的 checkpoint 一旦落进去，会被当成"最终产物"去校验。
WORK_DIR = Path(os.environ.get("EBBA_WORK", PROJECT_ROOT / ("checkpoints-full" if IS_FULL else "checkpoints")))
```

最值得注意的是 λ。论文取 λ=3，但本项目在 K=10 上把它降到了 2.2，代码里写清了原因：

```python
# src/ebba/config.py:117-125
TEMPERATURE = 2.0
LAMBDA = 3.0            # 论文值（Eq.8 / Table 2），精确层声明仍用它
# 中文注释：对任意 K 维能量向量，max_k (E_k - μ)/σ ≤ √(K-1)。
# K=10 时上界恰好是 3，而 Eq.8 要求"严格大于 λσ"，
# 所以 λ=3 在 CIFAR-10 上数学上取不到（这就是声明 A20）。
# 论文的可用区间 [1.8, 5.7] 与 K=43 的 GTSRB（上界 6.48）吻合，
# 说明 λ=3 是在 GTSRB 上标定后直接搬到 CIFAR-10 的。
PROXY_LAMBDA = 2.2      # 落在实测 clean z(1.77) 与 backdoored z(2.66) 之间
```

这里有一个容易忽略的工程决定：**公式层用论文原值 3.0，缩比实验层用 2.2**。两条线各自诚实——A20 用 3.0 才能暴露论文的问题，B02 用 2.2 才能让机制在缩比模型上真的跑起来。

### `src/ebba/determinism.py`（60 行）

```python
# src/ebba/determinism.py:56-61
def rng_for(*parts: object, seed: int = config.SEED) -> np.random.Generator:
    """返回一个由 parts 的稳定哈希派生的随机数生成器。"""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8, person=b"ebba").digest()
    sub_seed = int.from_bytes(digest, "little")
    return np.random.default_rng(seed ^ sub_seed)
# 中文注释：每个 (图片 id, 损坏名, 严重度) 独享一个 RNG，
# 因此第 37 张图的"雾"不会因为前面多算/少算了几张图而变样。
# 这是 corruptions.py 可复现的地基：结果与 batch 顺序、worker 数无关。
```

`configure_torch()` 用 `use_deterministic_algorithms(True, warn_only=True)`——`warn_only` 是刻意的：少数 GPU 算子没有确定性实现，宁可在报告里记一笔，也不要让整个实验直接崩掉。

---

## 模块 2：数据与攻击（把"论文设定"落成数组）

### `src/ebba/data.py`（388 行）

三件事：**固定子集**、**按伪标签均衡取样**、**构造 EBBA+ 候选池**。

```python
# src/ebba/data.py:35-42
def _per_class_indices(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)              # 中文注释：先打乱再截断，
        chosen.append(idx[:per_class]) # 保证每类取到的不是"文件顺序前 N 张"
    return np.sort(np.concatenate(chosen))
```

下载部分有两个安全/健壮性细节，都是被真实故障逼出来的：

```python
# src/ebba/data.py:45-63
def _download(url, dest, retries=3):
    tmp = dest.with_suffix(dest.suffix + ".part")   # 中文注释：先写 .part 再原子改名，
    ...                                             # 中断时不会留下"看起来完整的半个文件"
    if dest.name.endswith((".tgz", ".tar.gz", ".gz")):
        import gzip
        with gzip.open(tmp, "rb") as check:         # 中文注释：CDN 可能提前关闭 200 响应，
            while check.read(1 << 20): pass         # urllib 不报错；这里把 gzip 读到 EOF，
                                                    # 截断会在这里抛异常并触发重试。
```

```python
# src/ebba/data.py:72-87
def _safe_extract(archive, dest):
    """只解压留在 dest 内的普通文件。"""
    dest = Path(dest).resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue                       # 中文注释：跳过链接和设备节点
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise RuntimeError(f"unsafe path in {archive.name}: {member.name}")
            # 中文注释：拒绝绝对路径与 ".."，防止被篡改的镜像写到数据目录之外。
```

论文 EBBA 的第一步是"用任务无关图像、按伪标签每类取 N 张"。缩比实现是：

```python
# src/ebba/data.py:386-406
def balanced_select_by_pseudo(logits, n_per_class, seed):
    pred = logits.argmax(axis=1)              # 中文注释：伪标签 = 模型自己的预测，
    rng = np.random.default_rng(seed)         # 不用真实标签，才符合"任务无关"
    ...
    for cls in range(config.NUM_CLASSES):
        idx = np.flatnonzero(pred == cls)
        take = min(n_per_class, len(idx))     # 中文注释：某类不够就不补，
        chosen.extend(idx[:take])             # counts 里如实记录实际取到几张
```

候选池则是 EBBA+ 的输入，方法本身看不到 `is_poisoned`：

```python
# src/ebba/data.py:421-450
def make_candidate_set(clean_images, clean_labels, poisoned_images, target, ...):
    """构造 EBBA+ 候选池：中毒图 + 目标类的干净图。"""
    non_target = np.flatnonzero(clean_labels != target)
    target_idx = np.flatnonzero(clean_labels == target)
    p_idx = rng.choice(non_target, size=min(n_poisoned, len(non_target)), replace=False)
    c_idx = rng.choice(target_idx, size=min(n_clean_target, len(target_idx)), replace=False)
    is_poisoned = np.concatenate([np.ones(len(p_idx), bool), np.zeros(len(c_idx), bool)])
    # 中文注释：is_poisoned / ground_truth 只用于事后算 PDR/F1，
    # 从不进入聚类或 NTE 计算——否则就是标签泄漏。
```

### `src/ebba/attacks.py`（122 行）

投毒协议是 **dirty-label**：图上加触发器，标签改成目标类。

```python
# src/ebba/attacks.py:96-107
def poison_indices(labels, target, rate, seed):
    """按类分层抽取非目标类的训练图。"""
    for cls in range(config.NUM_CLASSES):
        if cls == target:
            continue                          # 中文注释：目标类自己不动，
        idx = np.flatnonzero(labels == cls)   # 否则会教模型"目标类=目标类"
        n = int(round(rate * len(idx)))
        chosen.extend(rng.choice(idx, size=n, replace=False).tolist())
```

```python
# src/ebba/attacks.py:110-131
def make_poisoned_training_set(images, labels, attack, target, rate, seed):
    idx = poison_indices(labels, target, rate, seed)
    x = images.copy()                         # 中文注释：不改原数组，
    y = labels.copy()                         # 干净模型和中毒模型共享同一份数据
    x[idx] = apply_trigger(x[idx], attack)
    y[idx] = target                           # 中文注释：dirty-label，只改被投毒样本的标签
    mask[idx] = True
    return x, y, mask
```

触发器本身分两者：BadNets 是右下角方块，Blend 是整图按 α 混合。注意方块颜色随 profile 变——代理模型用白块训练，full 用论文的黑块，配置里记了原因（`config.py:85-95`）。

---

## 模块 3：模型与训练

### `src/ebba/models.py`（270 行）

两套骨干：代理用 `SmallCNN`（约 1.2M 参数），full 用 `ResNet18CIFAR`（CIFAR 版 stem：3×3 stride 1，不接 maxpool）。

```python
# src/ebba/models.py:126-129
def to_tensor(images):
    """HWC uint8 -> 归一化后的 NCHW float32。"""
    x = torch.from_numpy(np.ascontiguousarray(images)).float().permute(0, 3, 1, 2) / 255.0
    return (x - _MEAN) / _STD
# 中文注释：归一化只在这里做一次。攻击与损坏都在 uint8 像素空间完成，
# 所以触发器强度（如 255 白块）不会被归一化放大或缩小。
```

训练循环的关键是**私有随机数生成器**，而不是全局 RNG：

```python
# src/ebba/models.py:240-270
def train_model(images, labels, epochs, seed, device, ...):
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)   # 中文注释：专用 generator
    ...
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)             # 每 epoch 的固定排列
        for start in range(0, n, config.BATCH_SIZE):
            idx = perm[start:start + config.BATCH_SIZE]
            xb = _augment_batch(x[idx], g) if augment else x[idx]
            # 中文注释：crop/flip 也来自同一个 g。
            # 只要 seed 和代码版本一致，断点续训与整段训练得到逐位相同的模型。
```

`_corruption_augment_batch`（`models.py:202-227`）是代理层专有的：给训练集加轻度噪声/模糊/对比度，让 SmallCNN 见过"被损坏的输入"，否则 NTE 在 80 种损坏下根本恢复不出标签（实测只有 12% top-1）。full profile 关掉它，回到论文的 crop+flip。

### `scripts/train_full.py`（716 行）

full 训练器单独成文件，因为它要解决缩比脚本不需要的问题：**200 epoch 的中断与恢复**。

- `_handle_signal`：收到 Ctrl-C 时先把当前 epoch 原子落盘再退出。
- `_atomic_save`：写 `.tmp` → `os.replace`，避免断电留下坏 checkpoint。
- `_resolve_resume`：按 `--resume auto` 找最新的 workspace checkpoint，并从记录的 epoch 继续。
- `_config_hash`：把超参、触发器指纹、数据规模一起哈希，**配置变了就拒绝 resume**。

---

## 模块 4：EBBA 检测（Eq.3–8）

### `src/ebba/energy.py`（131 行）

这是整个项目最核心、也最容易写错的 131 行。

**第一处：Eq.3 不是减最大值，是把最大值置 0。**

```python
# src/ebba/energy.py:35-42
def clamp_max_logits(logits, inplace=False):
    """Eq.(3)：把每个样本的最大 logit 置 0，其余保持不变。"""
    z = np.asarray(logits, dtype=np.float64)
    if not inplace:
        z = z.copy()
    rows = np.arange(z.shape[0])
    z[rows, z.argmax(axis=1)] = 0.0     # 中文注释：只改 argmax 那一格
    return z
# 论文 Appendix B 的例子 [3,8,-2] → [3,0,-2] 只能这样解释；
# 若写成 z - z.max()，会得到 [0,-5,-10]，softmax 不变，但论文的数字对不上。
```

**第二处：三套归一化并存，不是笔误，而是审计对象。**

```python
# src/ebba/energy.py:45-59
def per_sample_normalized_energy(logits, temperature=2.0, variant="paper"):
    s = softmax(clamp_max_logits(logits))
    p = np.exp(s * temperature) / temperature
    if variant == "paper":
        return p - np.exp(0.0 * temperature) / temperature   # Eq.5: 减 1/T = 0.5
    if variant == "readme":
        return p - 1.0                                       # 官方 README: P_k − 1
    if variant == "official":
        return np.minimum(p, 2.0) - 0.6                       # 官方 notebook: 截断 2 再减 0.6
    raise ValueError(f"unknown variant: {variant}")
# 中文注释：三种写法不能同时成立（声明 A03）。主实验走 "paper"，
# 另外两种只用于"结论对归一化是否敏感"的稳健性检查。
```

**第三处：Eq.7 的分母。** 论文 Eq.2/6 的求和写在 `k ∈ [0,K]`（K+1 项），Eq.7 却除以 K。实现取 K 项、除以 K：

```python
# src/ebba/energy.py:83-87
def energy_stats(energy):
    e = np.asarray(energy, dtype=np.float64)
    mu = float(e.mean())
    sigma = float(e.std(ddof=0))          # 中文注释：ddof=0 即总体标准差，
    return EnergyStats(e, mu, sigma, e - mu)  # 等价于 Eq.7 "除以 K"；若写 ddof=1 就变成除以 K-1
```

**第四处：判据是严格不等式。**

```python
# src/ebba/energy.py:90-121
def detect(energy, lam=3.0, two_sided=False):
    st = energy_stats(energy)
    scores = np.abs(st.deviation) if two_sided else st.deviation
    target = int(np.argmax(scores))
    margin = float(scores[target] - lam * st.std)
    return {"is_backdoored": bool(margin > 0), ...}
    # 中文注释：margin 可以理解为 "z − λ"（单边）。
    # 严格大于 0 才是后门——正是这个"严格"让 λ=3、K=10 的 A20 判定为不可达：
    # 上界 √9 = 3 只能在其余 9 类能量完全相等时取到，取到也是 0 而非 >0。
```

`detect()` 同时支持单边（论文 Eq.8）与双边（附录 G.4 的自适应攻击补丁）——两种都保留，报告里分别标注。

---

## 模块 5：EBBA+ 的迁移能量与 80 种损坏

### `src/ebba/transferred.py`（77 行）

```python
# src/ebba/transferred.py:20-44
def nte(source_logits, corrupt_logits, temperature=1.0, cap=None):
    """把 J 种损坏的 NTE 聚合起来。"""
    s = softmax(source_logits)                                  # 原图概率
    s_c = softmax(corrupt_logits.reshape(-1, K)).reshape(...)    # 损坏图概率
    delta = temperature * (s_c - s[None, :, :])                 # log 域做差 = s' − s
    values = np.exp(np.clip(delta, -50, 50)) - 1.0              # NTE = e^{s'−s} − 1
    if cap is not None:
        values = np.minimum(values, cap)                        # 中文注释：官方 notebook 截断 5（另一处 4）
    return values.sum(axis=0)                                   # Eq.11：对损坏求和
```

为什么在 **log 概率域**相减再取 exp，而不是直接算 `P'/P`？因为 softmax 概率常常小到 1e-8，直接相除会丢精度甚至溢出；`exp(s'−s)` 在数值上稳定得多。`clip(±50)` 是防 `exp` 溢出，实测不影响结果。

```python
# src/ebba/transferred.py:47-49
def recover_labels(nte_scores):
    """Eq.(12)：NTE 向量里最大的分量就是原始标签。"""
    return np.asarray(nte_scores).argmax(axis=1)
```

温度参数是审计需要：Eq.4 带着 T，Eq.9/10 里的 T 被约掉了（等价 T=1）。`temperature_sensitivity()` 用来量化这种不一致对 argmax 的影响（声明 A06）。

### `src/ebba/corruptions.py`（281 行）

论文说 80 种损坏，官方 notebook 实际是 15×4 + 4 = 64。这里重新实现 19 族 × 4 严重度 + 4 种整除 = **80**。

```python
# src/ebba/corruptions.py:53-62
def corruption_specs():
    specs = []
    for name in _CORRUPTION_NAMES:       # 19 族
        for sev in (1, 2, 3, 4):         # 4 档
            specs.append((name, sev))
    for n in (2, 3, 4, 5):               # 中文注释：整除从 2 开始。
        specs.append(("divide", n))      # 官方 notebook 从 1 开始，除以 1 是恒等变换，
    assert len(specs) == 80              # 等于白算一种（这是 A18 的一项检查）。
```

```python
# src/ebba/corruptions.py:304-309
def corrupt_one(img, name, severity, image_id):
    if name == "divide":
        return _to_uint8(_to_float(img) / float(severity))
    rng = rng_for(image_id, name, severity)   # 中文注释：随机性按 (图, 损坏, 档位) 定点派生
    return _to_uint8(_DISPATCH[name](img, severity, rng))
```

`corrupt_all()` 返回 `(J, N, H, W, C)`：J=80 种损坏、N 张图。这个形状决定了后面所有 NTE 计算的张量布局。

---

## 模块 6：聚类与指标

### `src/ebba/clustering.py`（128 行）

论文只说"t-SNE 降维 + Birch 聚类"，**没说在没有标签时怎么判断哪个簇是中毒簇**。这个空白是本项目唯一必须自己补的假设，代码里写得很直白：

```python
# src/ebba/clustering.py:99-116
def run_pipeline(nte_scores, method="birch", use_tsne=True, identification_scores=None):
    """EBBA+ 聚类全流程。
    论文（Sec.4.2）说用 t-SNE + Birch 分开两簇，
    但没说明无标签时如何认定哪一簇是中毒簇——这一步是本工具的假设。
    """
    embedding = tsne_embed(features) if use_tsne else StandardScaler().fit_transform(features)
    labels, used = cluster_labels(embedding, method)
    if identification_scores is None:
        confidence = np.clip(features, 0, None).max(axis=1)   # 旧路径：峰值 NTE
        rule = "peak_nte"
    else:
        confidence = np.asarray(identification_scores)         # 现路径：目标类 vs 次大类的 logit 差
        rule = "target_margin"
```

为什么从"峰值 NTE"换成"target margin"？因为 B06 的诊断发现：在缩比模型上，**峰值 NTE 与是否中毒是反相关的**，用它选簇会把干净图选成中毒图。target margin 才是后门真正制造出来的信号。这个改动记录在 `identification_rule` 字段里，报告可以重新裁决。

另外有两处稳定性处理：

```python
# src/ebba/clustering.py:29-41
tsne = TSNE(..., method="exact")
# 中文注释：候选池只有几百个点，精确解很便宜。
# Barnes-Hut 的树近似在不同 sklearn/BLAS 构建间不一致，
# 本地与容器之间曾让 PDR 相差 >0.10；exact 不会。

# src/ebba/clustering.py:62-68
valid = labels[labels >= 0]
if len(np.unique(valid)) != 2:
    return KMeans(n_clusters=2, ...).fit_predict(embedding), "fallback_kmeans"
    # 中文注释：DBSCAN 可能只给出一个簇或全是噪声。
    # 退化时退到 2-means，并把 "fallback_kmeans" 写进结果，不假装原方法成功。
```

### `src/ebba/metrics.py`（104 行）

```python
# src/ebba/metrics.py:17-31
def binary_metrics(pred_positive, true_positive):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum()); tn = int((~pred & ~true).sum())
    pdr = tp / max(tp + fn, 1)                 # 中文注释：PDR 的分母是"真正中毒的样本数"，
    precision = tp / max(tp + fp, 1)           # 不是候选池大小——TP 漏检会直接压低 PDR。
    f1 = 2 * precision * pdr / max(precision + pdr, 1e-12)
```

`ranks()` 用 `kind="stable"` 的 argsort，保证并列时排名确定；`rank_shift_summary()` 与 `probability_shift_summary()` 分别对应论文 Fig.1 / Fig.2 的两种视角（排名视角与概率视角）。

---

## 模块 7：声明体系（项目的"裁判层"）

这是整个仓库和普通实验代码最不一样的地方：**论文的每一条可检验断言，都是一个有 ID、有位置、有容差、有实现的函数。**

### `src/ebba/claims/model.py`（140 行）

```python
# src/ebba/claims/model.py:51-65
class Verdict(str, Enum):
    CONFIRMED = "CONFIRMED"            # 所有核心子检查在容差内通过
    PARTIAL = "PARTIAL"                # 部分通过 / 方向一致但幅度不同
    CONTRADICTED = "CONTRADICTED"      # 可核对数值超差，或普适断言有反例
    NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"  # 容器内无法复现（Tier C）
    ERROR = "ERROR"                    # 运行异常，整体退出码 1
```

```python
# src/ebba/claims/model.py:105-125
def run(claim_id, write=True):
    spec = load_registry().get(claim_id)      # 中文注释：元数据永远从 registry.yaml 读，
    _ensure_loaded(claim_id)                  # 不在函数里另写一份
    try:
        result = fn()
    except Exception as exc:
        result = ClaimResult(verdict=Verdict.ERROR, ...)
        # 中文注释：异常一律变成 ERROR，绝不静默当成"通过"。
        # 报告中不允许残留 PENDING/ERROR，但 ERROR 会先把退出码打成 1。
    if write:
        record(claim_id, spec, result)        # 写 reports/claims/<ID>.json
```

`_module_for()` 用**惰性 import** 按 ID 前缀路由到五个实现模块。这样只跑公式审计时不需要 torch/sklearn，`python -m pytest tests/test_energy.py` 也能在极简环境里跑。

### `src/ebba/claims/registry.yaml`

39 条声明：A01–A20（精确层）、B01–B14（缩比实证层）、C01–C05（不可复现层）。每条只声明"检验什么、在哪、容差多少"，**不预写结论**：

```yaml
# src/ebba/claims/registry.yaml:4-10
A01:
  tier: A
  scope: exact
  title_zh: "附录 B 算例：softmax 与能量数值是否可复算"
  paper_ref: "Appendix B, Eq.21-26, p.14"
  notebook: "01_formula_audit"
  tolerance: {abs_two_decimals: 0.01, abs_three_decimals: 0.005}
```

### `claims/formulas.py`（298 行）：不训练任何模型

以 A01（附录 B 算例）为例，它只做一件事——**用论文自己给的 logits 重算一遍**：

```python
# src/ebba/claims/formulas.py:27-55
Z = _np([[9, 1, -1], [3, 8, -2], [-1, 3, 7]])     # 论文 Eq.21 的 logits
s_exact = softmax(Z)                              # 精确 softmax
e_exact = np.exp(s_exact).sum(axis=0)             # 精确能量
paper_s = _np([[0.99, 0.01, 0.0], [0.12, 0.88, 0.0], [0.002, 0.018, 0.98]])
paper_e = _np([4.82, 4.43, 4.62])
s_err = float(np.abs(s_exact - paper_s).max())
e_err = float(np.abs(e_exact - paper_e).max())
verdict = Verdict.CONFIRMED if (ok_original and ok_clamped) else Verdict.CONTRADICTED
# 中文注释：判据用的是"论文印的两位小数 → 容差 0.01"，
# 不是"看起来差不多"。顺序 E0>E2>E1 也一并核对。
```

A20 则是把 λ 的数学上界直接算出来：

```python
# src/ebba/claims/formulas.py:270-295
K = config.NUM_CLASSES          # 10
lam = config.LAMBDA             # 3.0
bound = float(np.sqrt(K - 1))   # 中文注释：任意 K 维向量的 z 上界 = √(K−1) = 3.0

extremal = np.ones(K); extremal[0] = 2.0   # 构造：一个离群，其余 9 个完全相等
z_extremal = z_max(extremal)               # 恰好等于 3.0，但 Eq.8 要求 "> 3.0"

worst = 0.0
for eps in (1e-6, 1e-3, 1e-2, 0.1, 1.0):   # 中文注释：只要其余 9 个有一点点差异，
    ...                                    # 上界就严格低于 3，且差得越多降得越快
detectable = z_extremal > lam or worst > lam
verdict = Verdict.CONTRADICTED if not detectable else Verdict.CONFIRMED
```

### `claims/audit.py`（300 行）：表格算术与文字断言

以 A11（BA 下界）为例，它把 `(K−1)/K × ba` 逐行代进 Table 4 的 15 行：

```python
# src/ebba/claims/audit.py:25-41
for row in tables.table4_rows():
    k = tables.NUM_CLASSES_BY_DATASET[row["dataset"]]   # Cifar10=10, GTSRB=43, ImageNet=100
    ba = row["No Defense_ba"]
    bound = (k - 1) / k * ba
    observed = row["EBBA+_ba"]
    margin = observed - bound
    "violates": bool(margin < -0.005)                   # 中文注释：容差取 0.005（表中两位小数的半格）
violations = [r for r in rows if r["violates"]]
verdict = Verdict.CONTRADICTED if violations else CONFIRMED
# 找到反例 → 说明这条下界是经验观察，不是可证定理。
```

A18 更有意思：它**不执行**官方 PaddlePaddle notebook，而是把 ipynb 当 JSON 解析，做 8 项协议检查（是否用 GTSRB 验证集、是否有每类均衡、是否 704 张图、是否 64 种损坏、是否除以 1、TE 截断是否一致 5/4、是否有聚类、是否评 BA/ASR）。理由写在文件头：官方 demo 依赖 Paddle，且下载即执行第三方代码；静态审计足够回答"它是不是论文实验的完整实现"。

### `claims/empirical.py`（749 行）：缩比实证，永不 CONFIRMED

```python
# src/ebba/claims/empirical.py:25-27
def _proxy(ok: bool) -> Verdict:
    """Proxy claims cannot be CONFIRMED, by project rule."""
    return Verdict.PARTIAL if ok else Verdict.CONTRADICTED
# 中文注释：这是全项目的硬规则——用 SmallCNN/子集跑出来的"支持"，
# 最高只能记 PARTIAL。否则读者会把它误读成"论文的全量结果被复现了"。
```

贵的那一步只算一次，用 `lru_cache` 缓存给 B05–B10、B14 共用：

```python
# src/ebba/claims/empirical.py:202-262
@lru_cache(maxsize=len(config.ATTACKS))
def _removal(attack):
    """用 *真实* 目标标签跑 EBBA+ 的样本阶段。
    EBBA+ 只在 EBBA 已经找到目标类之后才成立；
    这里用 oracle 目标，是为了把"样本阶段"单独测量（B05–B10）。
    EBBA 能不能找到目标，是 B02 的问题。
    """
    ...
    keep = source_pred == target      # 中文注释：EBBA+ 只处理"模型判为目标类"的图
    corrupted = corruptions.corrupt_all(x)              # 80 种损坏，一次算完
    corrupt_logits = models.predict_logits(model, ...)  # 80×N 次前向
    scores = nte(source_logits, corrupt_logits, temperature=1.0, cap=5.0)
```

### `claims/limits.py`（56 行）：诚实地说"做不到"

```python
# src/ebba/claims/limits.py:14-20
def _limit(reason_zh, needed, proxy=None):
    return ClaimResult(
        verdict=Verdict.NOT_REPRODUCIBLE,
        measured={"reason_zh": reason_zh, "needed": needed, "proxy_note_zh": proxy},
    )
# 中文注释：C01–C05 不是"失败"，而是"容器内不可能"。
# 明确写出来，保证 39 条声明每条恰好一个 verdict，报告里不出现 PENDING。
```

---

## 模块 8：报告与驱动脚本

### `src/ebba/report.py`（214 行）

只读 JSON，不重算任何东西：

```python
# src/ebba/report.py:36-51
def collect(claims_dir=None):
    for cid, spec in registry.items():
        path = claims_dir / f"{cid}.json"
        if not path.exists():
            rows.append({..., "verdict": "PENDING", "notes_zh": "尚未运行"})
        # 中文注释：缺文件显式记为 PENDING，而不是悄悄跳过。
        # 这样"少跑了一条声明"在报告里是可见的。
```

HTML 是完全自包含的（内联 CSS，无 CDN、无外部字体），因为容器运行时 `--network none`。

### `scripts/build_artifacts.py`（163 行）：构建期的"实验台验收"

它不只训练，还**守门**——缩比模型如果表现不出能量异常，构建直接失败：

```python
# scripts/build_artifacts.py:144-164
gates["clean_ba_min"] = metrics["clean"]["ba"] >= 0.70
gates["badnets_asr"] = metrics["badnets"]["asr"] >= 0.90
gates["blend_asr"]   = metrics["blend"]["asr"] >= 0.90
gates["clean_not_detected"] = margin["clean"]["margin"] < 0 and not margin["clean"]["is_backdoored"]
gates["badnets_detected_target"] = (margin["badnets"]["margin"] > 0
    and margin["badnets"]["target"] == config.TARGET_LABEL)
# 中文注释：闸门检验的是"实验台是否合格"，不是"论文是否对"。
# NTE 恢复率刻意不设闸门——它是被测量的结果（B05），不是前提。
if failed:
    return 1      # 宁可构建失败，也不产出一份误导性的报告
```

### `scripts/run_papermill.py`（103 行）

逐个执行 notebook，**单本失败不中断其余**，最后把耗时与错误写进 `run_metadata.json`，并返回非零：

```python
try:
    pm.execute_notebook(src, dst, kernel_name="python3", cwd=str(ROOT), execution_timeout=args.timeout)
    rec = {"name": name, "status": "ok", "seconds": ...}
except Exception as exc:
    rec = {"name": name, "status": "error", "traceback": traceback.format_exc()[-4000:]}
    status = 1          # 中文注释：继续跑下一本，报告里能看出"哪本挂了、其余是否正常"
```

### `scripts/full_eval.py`（867 行）

full profile 的主评估器，回答一个窄问题：**换成论文骨干和论文训练配方后，机制是否更靠近论文数字？** 它的 `run()` 是一串固定阶段：模型指标 → 检测 → Fig.1 → Fig.2 → 候选池/NTE → 聚类 → 去除后估计 → 写 JSON + HTML。每个阶段打印一行，便于判断长跑卡在哪。

### `scripts/compare_full_metrics.py`（107 行）

把容器产出的 `full_metrics.json` 与生产环境基线逐字段比对，38 个字段全过才退出 0。键名里含点的（如 `lambda_3.0`）用 `/` 分隔：

```python
def get(d, path, default=None):
    sep = "/" if "/" in path else "."     # 中文注释："detection/badnets/lambda_3.0/zscore"
    ...                                   # 若按 "." 切，会把 "lambda_3.0" 拆坏
```

---

## 模块 9：容器与分发

### `docker/Dockerfile.full`

```dockerfile
# docker/Dockerfile.full:25-43
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates ...
COPY docker/requirements-common.txt /tmp/requirements-common.txt
RUN pip install -r /tmp/requirements-common.txt
# 中文注释：上面几行的字节必须与 docker/Dockerfile 完全一致，
# BuildKit 才会复用已缓存的 apt/pip 层，而不是重新下载 ~3GB CUDA wheel。

RUN if [ "$GPU" = "1" ]; then \
        python -c "import torch; flags=torch._C._cuda_getArchFlags(); ... \
        assert 'sm_120' in flags, 'torch build lacks sm_120 (need cu128)';"; fi
# 中文注释：RTX 5060 是 sm_120，只有 cu128 轮子支持。
# 把检查放在构建期，是在"构建失败"和"运行时才发现 GPU 不可用"之间选前者。
```

```dockerfile
# docker/Dockerfile.full:58-65
ENV EBBA_PROFILE=full EBBA_ARTIFACTS=/app/artifacts-full EBBA_REPORTS=/app/reports-full \
    EBBA_OOD_GTSRB=1 EBBA_BADNETS_VALUE=0
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["verify-full", "--device", "cuda"]
# 中文注释：运行环境固定在镜像里，docker run 不需要任何 -e 参数。
```

### `docker/entrypoint.sh`

```sh
# docker/entrypoint.sh:82-91
FULL_DEVICE="$DEVICE"
prev=""
for a in "$@"; do
    if [ "$prev" = "--device" ]; then FULL_DEVICE="$a"; fi
    case "$a" in --device=*) FULL_DEVICE="${a#*=}" ;; esac
    prev="$a"
done
echo "== full evaluation (device=$FULL_DEVICE) =="
python scripts/full_eval.py "$@"
# 中文注释：原来这里直接 echo $DEVICE（环境变量默认值），
# 命令行传 --device cpu 时日志仍打印 cuda，误导排查。现在回显真正传下去的值。
```

### 分发工具（本次推送时新增）

- `scripts/push_manifest.py`：`docker push` 在最后一层 manifest 上挂死，于是直接读 `docker save` 的 OCI 归档、用 registry API 上传 blob、再 PUT manifest。要点：**token 必须用 `grant_type=password` 的 form POST 换取**（GET+Basic 只会给 pull 权限）；大层用 8 MB 可续传 PATCH；token 约 5 分钟过期，需要 `TokenSource` 自动续期。
- `scripts/delete_tag.py`：按 digest 删除 manifest（同一 digest 的多个 tag 会一起消失）。注意删除需要 `delete` scope，push 权限不够。

---

## 6 个最容易误读的点

1. **`detect()` 的 margin 不是能量差，是 "z − λ"**。正负号取决于 λ，而不是能量本身。
2. **`PROXY_LAMBDA=2.2` 不是"调参调到能检出"**。它是被 A20 逼出来的：λ=3 在 K=10 上不可达；2.2 落在实测 clean z=1.77 与 backdoored z=2.66 之间，且 A20 本身仍用论文原值 3.0 判定。
3. **`CONTRADICTED` 不是程序失败**。它是一条论文断言的判定结果；`verify` 的退出码只因执行异常非零。
4. **B 类断言永远不会 CONFIRMED**。这是规则，不是能力不足的托词。
5. **`cap=5.0` 是官方实现的口径，不是论文的**。论文 Eq.9–10 没有截断；不截断的版本在 A05 里单独验证。
6. **聚类"哪一簇是中毒簇"是本工具的假设**。论文没写，代码用 `identification_rule` 字段把假设显式暴露出来，读者可以据此重新裁决。

---

## 阅读顺序建议

| 目的 | 顺序 |
|---|---|
| 想理解 EBBA 检测 | `config.py` → `energy.py` → `claims/empirical.py` 的 B02/B03 |
| 想理解 EBBA+ 去除 | `transferred.py` → `corruptions.py` → `clustering.py` → `metrics.py` |
| 想看审计方法 | `claims/registry.yaml` → `claims/formulas.py`（A01）→ `claims/audit.py`（A11） |
| 想复核结论 | `reports/claims/*.json` → `reports/report.html` |
| 想跑起来 | `docker/entrypoint.sh` → `scripts/run_papermill.py` / `scripts/full_eval.py` |
