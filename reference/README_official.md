# 官方代码参考（只读）

本目录中的 `EBBA.ipynb` 是论文作者公开的演示 Notebook，**本项目的任何命令都不会执行它**，
只在声明 A18 中对它做静态解析（读取 JSON、抽取代码单元、字符串检查）。

- 来源：<https://github.com/ifen1/EBBA>（论文第 2 页脚注给出的代码地址）
- 文件名：`EBBA.ipynb`
- 用途：交叉核对论文公式与流程；不是论文实验的完整实现
- 许可：归原作者所有，此处仅为验证目的保存副本，未做修改

## 官方 README 原文（`paper/repo/README.md`）

> # EBBA
> A demo for EBBA and EBBA+
>
> Datasets: GTSRB
>
> DNNs: ResNet or VGG
>
> Note: For Cifar10, we set the NP_k as P_k-1.
>
> Note that the original formula **also works**, but in our paper's experiments, we indeed changed it to this form.
>
> If you have questions, please contact YudongGao0504@163.com

这条注记本身就是 A03 的证据之一：论文 Eq. 5 与官方实现在 CIFAR-10 上使用了两套不同的归一化。

## 与论文的差异

见 [AUDIT.md](AUDIT.md)。摘要：8 项协议检查中 7 项不符——使用 GTSRB 验证集（任务相关图像）、
无每类均衡、仅 704 张图、64 种损坏（含除以 1 的恒等变换）、TE 截断阈值 5/4 不一致、
无聚类步骤。它适合核对公式写法，不能用于复核 Table 3/4 的数字。
