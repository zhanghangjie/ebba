"""Tier C: claims that cannot be reproduced inside this container.

These are not failures of the paper and not failures of the code.  They are
recorded explicitly so that every registry entry has exactly one verdict and
the report never shows a PENDING row.  Each entry states what would be needed
and why the scaled proxy in Tier B is not a substitute.
"""

from __future__ import annotations

from .model import ClaimResult, Verdict, claim


def _limit(reason_zh: str, needed: list[str], proxy: str | None = None) -> ClaimResult:
    return ClaimResult(
        verdict=Verdict.NOT_REPRODUCIBLE,
        measured={"reason_zh": reason_zh, "needed": needed, "proxy_note_zh": proxy},
        notes_zh=reason_zh,
        tolerance={},
    )


@claim("C01")
def c01_full_tables() -> ClaimResult:
    return _limit(
        "Table 3/4 的全量数字需要 ResNet18、200 epochs、CIFAR-10/GTSRB/ImageNet 三个数据集"
        "与五种攻击（BadNets、Blend、WaNet、FIBA、DUBA）的完整训练，单次复现约需数十 GPU 小时；"
        "本容器只提供 20,000 张 CIFAR-10 子集、SmallCNN、30 epochs、两种攻击的缩比代理。",
        ["ResNet18/VGG 训练", "三个数据集的完整训练集", "WaNet/FIBA/DUBA 攻击实现", "GPU 集群"],
        "Tier B 的 PDR/F1、BA/ASR 只验证方法方向，不能与表中数字逐项比较。",
    )


@claim("C02")
def c02_speech_text() -> ClaimResult:
    return _limit(
        "ESC-50 语音与 THUCNews 文本实验需要对应数据集、语音/文本骨干网络与任务相关的"
        "预处理流程；这些资源不在容器内，且与图像管线的接口不同。",
        ["ESC-50 数据集", "THUCNews 数据集", "语音/文本骨干网络", "领域特定的损坏算子"],
        "本容器不提供该实验的代理。",
    )


@claim("C03")
def c03_adaptive_training() -> ClaimResult:
    return _limit(
        "13 组自适应软标签攻击需要为每组参数重新训练被攻击模型，且攻击实现未公开；"
        "容器只能对论文已报告的 13 个数值做算术审计（A19）。",
        ["自适应攻击训练代码", "13 组重训练", "与论文一致的攻击预算"],
        "B12 只在合成分数上演示双边检验机制，不构成对自适应攻击的复现。",
    )


@claim("C04")
def c04_all_to_all() -> ClaimResult:
    return _limit(
        "all-to-all 与 full-target 攻击需要修改投毒标签生成器并对每种攻击重新训练；"
        "论文只给出结论性描述，缺少足够细节逐项复算。",
        ["all-to-all 投毒生成器", "full-target 训练", "论文级算力"],
        "本容器的多目标检查（B13）是合成机制演示，不是该实验的复现。",
    )


@claim("C05")
def c05_imagenet_scale() -> ClaimResult:
    return _limit(
        "ImageNet 子集（100 类、224×224）需要 ImageNet 训练数据与 ResNet 级模型，"
        "下载量与算力都超出轻量容器的设计目标。",
        ["ImageNet 子集", "ResNet18/VGG", "GPU 训练"],
        "Tier B 的 CIFAR-10 代理不覆盖高分辨率与 100 类设定。",
    )
