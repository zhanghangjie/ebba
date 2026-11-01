# 复现简述：EBBA / EBBA+（ICML 2024）

> 用途：README 摘要、镜像说明或提交材料。正文约 350 字。

本次复现针对 ICML 2024 论文 *Energy-based Backdoor Defense without Task-Specific Samples and Model Retraining*（Gao 等），交付物是一套带中文注释的 Jupyter Notebook 与可离线运行的 Docker 镜像。实现为独立重写，不依赖作者代码，仅对官方示例 Notebook 做静态审计；任何人无需联网、无需私有数据，即可用一条命令在本地重算全部结论。

验证分三层：**精确层**从论文自身的公式与数值出发，逐项复核能量定义与钳制（Eq.3–8）、迁移能量与 NTE（Eq.9–12）、附录 B 算例、Table 3–7 的算术与文字断言；**缩比实证层**在 CIFAR-10 上训练 SmallCNN（BadNets、Blend），检验检测、标签恢复、聚类与去除的机制方向，一律标注 proxy，不冒充论文原表数字；**不可复现层**记录 5 项容器外无法完成的实验及其原因。39 条声明全部给出判定：CONFIRMED 3、PARTIAL 13、CONTRADICTED 18、NOT_REPRODUCIBLE 5、ERROR 0。proxy 结论不判 CONFIRMED，CONTRADICTED 是预期结论而非运行故障。

主要发现：K=10 时 z 统计量的严格上界为 √(K−1)=3.0，论文 λ=3 的严格不等式在 CIFAR-10 上不可达（与论文 λ∈[1.8,5.7] 对应 GTSRB 的 K=43 相符）；附录 B 算例的 softmax 与能量数值有误；BA 下界与「ASR 均高于 98.8%」存在反例；GTSRB 测试集 12603 应为 12630；Table 3/4 的「全面最优」表述被反例推翻。方法主线在缩比实验中的方向成立。

全量复现以镜像 `zhanghangjie/ebba:full`（12.3 GB，digest `sha256:a4543d1b…e492`）发布：内置 ResNet-18 检查点、CIFAR-10 全量 50,000/10,000 与 22,630 张任务无关 OOD 图像，一条命令离线重算 BA/ASR、PDR/F1、NTE 恢复率、去除后 ASR、检测 z 分数与 Fig.1/Fig.2 统计，在 RTX 5060 笔记本 GPU 上约 452 秒。同一份代码在本机与容器内产出 38/38 个指标字段一致。局限：镜像只含全量档，缩比用的 7 个 Notebook 未随镜像发布；模型训练默认不重跑（镜像保留 `train-full` 入口，可重训 200 epochs）。

---

## 镜像简介（Docker Hub，≤100 字符）

> EBBA/EBBA+（ICML 2024）独立复现：39 条声明逐条判定，一条命令离线重算全量指标。

## 关键数字（便于引用）

| 项目 | 数值 |
|---|---|
| 声明判定 | 39 条：CONFIRMED 3 / PARTIAL 13 / CONTRADICTED 18 / NOT_REPRODUCIBLE 5 / ERROR 0 |
| z 统计量上界 | K=10 → √(K−1)=3.0；论文 λ=3 为严格不等式，不可达 |
| 全量配置 | ResNet-18，CIFAR-10 50,000/10,000，OOD 22,630 张，200 epochs |
| 运行时间 | RTX 5060 Laptop，452 s（本机对照 441 s） |
| 复现一致性 | 38/38 指标字段在 5e-3 容差内一致 |
| 干净模型 BA | 0.9461 |
| BadNets | BA 0.9240，ASR 0.9857，PDR 0.8908，F1 0.9353 |
| Blend | BA 0.9421，ASR 0.9990，PDR 0.6480，F1 0.7710 |
| 检测 z（clean/BadNets/Blend） | 2.4220 / 2.7254 / 2.2246，λ=3 下均不报警 |
| 镜像 | `zhanghangjie/ebba:full`，12.3 GB，digest `sha256:a4543d1bec29b565222f91fc51dc137bdf3d3f8bad2295f8fd22efa1d2c0e492` |
