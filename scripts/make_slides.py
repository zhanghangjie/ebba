# -*- coding: utf-8 -*-
"""生成《EBBA / EBBA+ 复现验证》汇报 PPT。

所有数字来自 reports-full/full_metrics.json 与 claims 判定结果，脚本只负责排版。
用法：py -3 scripts/make_slides.py
输出：slides/EBBA_复现汇报.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ---------------------------------------------------------------- palette
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
CARD = RGBColor(0xF7, 0xF6, 0xF3)
HEAD = RGBColor(0xF0, 0xEF, 0xEC)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
HAIR = RGBColor(0xE1, 0xE0, 0xD9)
RULE = RGBColor(0xC3, 0xC2, 0xB7)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
BLUE_L = RGBColor(0xCD, 0xE2, 0xFB)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
GOOD = RGBColor(0x0C, 0xA3, 0x0C)
WARN = RGBColor(0xFA, 0xB2, 0x19)
CRIT = RGBColor(0xD0, 0x3B, 0x3B)

FONT = "Microsoft YaHei"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

PAGE = {"n": 0}


# ---------------------------------------------------------------- helpers
def set_run(run, text, size=14, bold=False, color=INK, font=FONT):
    run.text = text
    f = run.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    if rPr.find(qn("a:ea")) is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    rPr.find(qn("a:ea")).set("typeface", font)


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tf


def para(tf, text, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT,
         before=0, spacing=1.15, font=FONT, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(before)
    p.line_spacing = spacing
    run = p.add_run()
    set_run(run, text, size, bold, color, font)
    return p


def rect(slide, x, y, w, h, fill=CARD, line=None, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            s.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    if s.has_text_frame:
        s.text_frame.text = ""
    return s


def hline(slide, x, y, w, color=HAIR, thick=0.012):
    return rect(slide, x, y, w, thick, fill=color, shape=MSO_SHAPE.RECTANGLE)


def new_slide(kicker, title):
    slide = prs.slides.add_slide(BLANK)
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = SURFACE
    PAGE["n"] += 1
    if kicker:
        tf = textbox(slide, 0.75, 0.40, 11.8, 0.26)
        para(tf, kicker, 10, False, MUTED, first=True)
    tf = textbox(slide, 0.75, 0.66, 11.9, 0.55)
    para(tf, title, 25, True, INK, first=True)
    hline(slide, 0.75, 1.30, 11.83)
    tf = textbox(slide, 0.75, 7.06, 9.0, 0.24)
    para(tf, "EBBA / ICML 2024 · 独立复现验证", 8.5, False, MUTED, first=True)
    tf = textbox(slide, 11.6, 7.06, 1.0, 0.24)
    para(tf, f"{PAGE['n']:02d}", 9, False, MUTED, PP_ALIGN.RIGHT, first=True)
    return slide


def table(slide, x, y, widths, rows, row_h=0.36, font=10.5,
          aligns=None, header=True, zebra=True):
    nrows, ncols = len(rows), len(widths)
    shape = slide.shapes.add_table(
        nrows, ncols, Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * nrows)
    )
    tbl = shape.table
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    for r in range(nrows):
        tbl.rows[r].height = Inches(row_h)
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow", "firstCol", "lastCol", "bandCol"):
        if tblPr.get(attr) is not None:
            tblPr.set(attr, "0")
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            color = INK
            bold = False
            if isinstance(val, tuple):
                text, color = val[0], val[1]
                bold = len(val) > 2 and val[2]
            else:
                text = val
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if header and r == 0:
                cell.fill.fore_color.rgb = HEAD
            elif zebra and r % 2 == 0:
                cell.fill.fore_color.rgb = CARD
            else:
                cell.fill.fore_color.rgb = SURFACE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (aligns[c] if aligns else PP_ALIGN.LEFT)
            p.line_spacing = 1.0
            run = p.add_run()
            set_run(run, str(text), font, bold or (header and r == 0),
                    INK2 if (header and r == 0) else color)
    return shape


def card(slide, x, y, w, h, title, lines, accent=BLUE, title_size=13, body_size=11.5):
    rect(slide, x, y, w, h)
    rect(slide, x, y + 0.16, 0.045, h - 0.32, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    tf = textbox(slide, x + 0.24, y + 0.16, w - 0.42, 0.3)
    para(tf, title, title_size, True, INK, first=True)
    tf = textbox(slide, x + 0.24, y + 0.56, w - 0.42, h - 0.7)
    for i, line in enumerate(lines):
        para(tf, line, body_size, False, INK2, before=0 if i == 0 else 6, first=(i == 0))


def chain(slide, x, y, items, accent=BLUE, gap=0.30, h=0.62):
    """横向步骤链：圆角块 + 序号 + 箭头。"""
    w = (11.83 - gap * (len(items) - 1)) / len(items)
    for i, item in enumerate(items):
        bx = x + i * (w + gap)
        rect(slide, bx, y, w, h)
        num = rect(slide, bx + 0.16, y + 0.16, 0.30, 0.30, fill=accent, shape=MSO_SHAPE.OVAL)
        tf = num.text_frame
        tf.margin_left = 0
        tf.margin_right = 0
        tf.margin_top = 0
        tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_run(p.add_run(), str(i + 1), 10, True, RGBColor(0xFF, 0xFF, 0xFF))
        tf = textbox(slide, bx + 0.54, y + 0.10, w - 0.64, h - 0.16, MSO_ANCHOR.MIDDLE)
        para(tf, item, 10.5, False, INK2, first=True, spacing=1.05)


# ================================================================ 01 cover
s = prs.slides.add_slide(BLANK)
bg = s.background.fill
bg.solid()
bg.fore_color.rgb = SURFACE
rect(s, 0, 0, 0.22, 7.5, fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
tf = textbox(s, 0.95, 1.35, 11.0, 0.3)
para(tf, "ICML 2024 · 独立复现验证报告", 13, True, BLUE, first=True)
tf = textbox(s, 0.95, 1.85, 11.5, 1.9)
para(tf, "能量后门防御的独立复现", 40, True, INK, first=True, spacing=1.05)
para(tf, "EBBA（检测）与 EBBA+（去除）", 26, False, INK2, before=6)
tf = textbox(s, 0.95, 3.55, 11.2, 0.9)
para(tf, "Energy-based Backdoor Defense without Task-Specific Samples and Model Retraining",
     12.5, False, MUTED, first=True, font=MONO)
para(tf, "Gao et al., ICML 2024", 11, False, MUTED, before=4)
hline(s, 0.95, 4.55, 5.2, RULE, 0.02)
tf = textbox(s, 0.95, 4.80, 11.4, 0.9)
para(tf, "ResNet-18 · CIFAR-10 全量 50k/10k · 200 epochs · BadNets / Blend · GTSRB OOD 22,630",
     13, False, INK2, first=True)
para(tf, "39 条论文断言 · 5 个判定等级 · 端到端可一键重跑", 11.5, False, MUTED, before=6)
tf = textbox(s, 0.95, 6.55, 11.0, 0.4)
para(tf, "2026-10", 11, False, MUTED, first=True)

# ================================================================ 02 summary
s = new_slide("结论先行", "一句话结论：方法主体复现了，实验矩阵没有")
cards = [
    ("39", "条断言全部给出判定", "3 CONFIRMED / 13 PARTIAL / 18 CONTRADICTED / 5 NOT_REPRODUCIBLE / 0 ERROR", BLUE),
    ("18", "条与论文不符", "其中 3 条是可直接复算的数值或逻辑错误，不是转述问题", CRIT),
    ("5/8", "个主指标在 ±3pp 内", "8 个方向一致；最大缺口是 Blend 的 PDR：0.648 vs 0.95", ORANGE),
]
for i, (num, label, sub, accent) in enumerate(cards):
    x = 0.75 + i * 3.99
    rect(s, x, 1.60, 3.75, 1.95)
    rect(s, x, 1.60, 3.75, 0.05, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    tf = textbox(s, x + 0.28, 1.82, 3.2, 0.7)
    para(tf, num, 38, True, accent, first=True, spacing=0.95)
    tf = textbox(s, x + 0.28, 2.52, 3.2, 0.3)
    para(tf, label, 12, True, INK, first=True)
    tf = textbox(s, x + 0.28, 2.86, 3.2, 0.6)
    para(tf, sub, 9.5, False, MUTED, first=True, spacing=1.1)

tf = textbox(s, 0.75, 3.85, 11.8, 2.9)
para(tf, "三条支撑判断", 12, True, INK, first=True)
for text in [
    "方法链完整：Eq.3–8 检测、Eq.9–12 迁移能量、80 种损坏、聚类重标注，全部在 ResNet-18 上跑通，没有用玩具模型替代。",
    "数值层分裂：攻击训练指标 4/4 对齐；PDR/F1 两项达标、两项不达标，根因定位到 Eq.12 的标签恢复率（Blend 仅 0.36）。",
    "覆盖层很窄：2/5 攻击 × 1/3 数据集，baseline 0 个。论文“优于 SOTA”的结论本身没有被复现，只复现了它的绝对数值。",
]:
    para(tf, "·  " + text, 12.5, False, INK2, before=9, spacing=1.3)

hline(s, 0.75, 6.05, 11.83)
tf = textbox(s, 0.75, 6.18, 11.8, 0.3)
para(tf, "报告结构：方法 → 装置 → 攻击 → 检测 → 机制 → 瓶颈 → 去除 → 消融 → 论文问题 → 覆盖度 → 自评",
     9.5, False, MUTED, first=True)

# ================================================================ 03 method
s = new_slide("论文方法", "两段流水线：先判模型有没有后门，再把中毒样本挑出来")
tf = textbox(s, 0.75, 1.48, 11.8, 0.3)
para(tf, "EBBA —— 模型级检测（只用任务无关图像，不需要中毒样本）", 13, True, INK, first=True)
chain(s, 0.75, 1.88, [
    "任务无关图像\n按伪标签每类取 N 张",
    "max logit 置 0\nsoftmax（Eq.3）",
    "逐标签能量\nEq.4–6，T=2",
    "标准分 z 超过 λ\n判为后门（Eq.8）",
], accent=BLUE, h=0.9)

tf = textbox(s, 0.75, 3.12, 11.8, 0.3)
para(tf, "EBBA+ —— 样本级检测与去除（Eq.9–12，不重训练、不加数据）", 13, True, INK, first=True)
chain(s, 0.75, 3.52, [
    "J=80 种图像损坏\nD_j(x)",
    "迁移能量 TE / NTE\nEq.9–11",
    "argmax NTE\n复原原始标签",
    "特征空间聚类\n定向中毒簇",
    "只改预测\n不碰权重",
], accent=ORANGE, h=0.9)

rect(s, 0.75, 4.85, 11.83, 1.75)
tf = textbox(s, 1.00, 5.05, 11.3, 1.4)
para(tf, "复现时的两个关键判断", 12, True, INK, first=True)
para(tf, "·  Eq.3 是“把最大 logit 置 0”，不是“减最大值”——论文示例 [3,8,−2] → [3,0,−2] 给出唯一解释，两者不等价。",
     11.5, False, INK2, before=6)
para(tf, "·  论文没写聚类用什么特征、怎么判断哪个簇是中毒簇。这是复现结果分歧的主要来源，我们的做法与理由都写在报告里。",
     11.5, False, INK2, before=4)

# ================================================================ 04 tiers
s = new_slide("验证方法", "先分类再动手：哪些能证伪，哪些只能实验")
rows = [
    ["层级", "判据", "条数", "处理方式"],
    ["Tier A 精确层", "用论文自身的公式或表格就能证伪", "19", "全自动复核，容差在运行前固定"],
    ["Tier B 实证层", "需要跑实验，允许缩比代理", "15", "proxy（SmallCNN）与 full（ResNet-18）两档"],
    ["Tier C 不可复现", "依赖容器外资源（ImageNet 全量、语音/文本）", "5", "标注原因，不假装做过"],
]
table(s, 0.75, 1.62, [2.3, 4.1, 1.0, 4.43], rows, row_h=0.62, font=11.5,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.LEFT])

tf = textbox(s, 0.75, 4.30, 11.8, 0.3)
para(tf, "四条判定纪律", 12, True, INK, first=True)
rules = [
    ("proxy 永不 CONFIRMED", "缩比实验最多判 PARTIAL，避免用小模型给论文盖章。"),
    ("CONTRADICTED 不是失败", "它是结论。verify 的退出码只反映执行故障。"),
    ("容差先定后跑", "代数 1e-6、两位小数 0.01、三位小数 0.005，运行前写死。"),
    ("ERROR 必须为 0", "报告里不得残留 PENDING / ERROR，否则整体退出码 1。"),
]
for i, (head, body) in enumerate(rules):
    x = 0.75 + (i % 2) * 5.97
    y = 4.70 + (i // 2) * 1.05
    rect(s, x, y, 5.80, 0.88)
    tf = textbox(s, x + 0.20, y + 0.13, 5.4, 0.26)
    para(tf, head, 11.5, True, INK, first=True)
    tf = textbox(s, x + 0.20, y + 0.43, 5.4, 0.36)
    para(tf, body, 10, False, INK2, first=True)

# ================================================================ 05 setup
s = new_slide("实验装置", "把能对齐的都对齐：模型、训练、触发器、OOD")
rows = [
    ["项目", "论文设定", "本项目实际"],
    ["数据集", "CIFAR-10 / GTSRB / ImageNet 子集", "CIFAR-10 全量 50,000 训练 / 10,000 测试"],
    ["模型与训练", "ResNet-18，200 epochs", "ResNet-18，200 epochs，SGD 0.01，milestones 50/100/150"],
    ["攻击", "BadNets / Blend / WaNet / FIBA / DUBA", "BadNets / Blend（其余 3 种未做）"],
    ["触发器", "黑色 3×3 块；hello kitty 图 α=0.2", "同论文，图案 sha256 入库，随 checkpoint 一起校验"],
    ["任务无关图像", "Internet images", "CIFAR-100 测试 10k + GTSRB 测试 12,630 = 22,630 张"],
    ["评估池", "完整测试集", "5,000 中毒 + 1,000 干净目标类（池内口径）"],
]
table(s, 0.75, 1.62, [2.0, 4.5, 5.33], rows, row_h=0.60, font=11,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
tf = textbox(s, 0.75, 6.05, 11.8, 0.7)
para(tf, "为什么 OOD 要加 GTSRB：CIFAR-10 只有 10 类，λ=3 在数学上不可达（见第 7 页）；"
         "GTSRB 有 43 类，才让 λ=3 具备可测性。", 10.5, False, MUTED, first=True, spacing=1.2)

# ================================================================ 06 attack
s = new_slide("第一步：攻击是否训成", "四个数全在 ±3pp 内，模型足以支撑后续评估")
rows = [
    ["攻击", "BA 复现", "BA 论文", "Δ", "ASR 复现", "ASR 论文", "Δ"],
    ["BadNets", "92.40", "91.22", ("+1.18", GOOD), "98.57", "99.36", ("−0.79", GOOD)],
    ["Blend", "94.21", "91.35", ("+2.86", GOOD), "99.90", "99.97", ("−0.07", GOOD)],
]
table(s, 0.75, 1.75, [2.0, 1.6, 1.6, 1.3, 1.7, 1.7, 1.3], rows, row_h=0.62, font=13,
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 6)

rect(s, 0.75, 3.78, 11.83, 1.35)
tf = textbox(s, 1.00, 3.94, 11.3, 1.1)
para(tf, "为什么先校这一层", 12, True, INK, first=True)
para(tf, "防御指标建立在模型本身的攻击强度上。若 BA/ASR 与论文差得远，后面 PDR 的低值就无法归因——"
         "分不清是“防御方法不行”还是“模型根本没学到后门”。四个数对齐后，后续差异才能归给防御链路。",
     11.5, False, INK2, before=6, spacing=1.25)

tf = textbox(s, 0.75, 5.35, 11.8, 0.3)
para(tf, "训练配置（与论文一致）", 12, True, INK, first=True)
rows2 = [
    ["优化器", "学习率", "batch", "weight decay", "AMP", "投毒率 / 目标类"],
    ["SGD momentum 0.9", "0.01，50/100/150 各 ×0.1", "64", "5e-4", "是", "10% / 类 0，只投非目标类"],
]
table(s, 0.75, 5.70, [2.3, 3.3, 1.0, 1.7, 0.9, 3.3], rows2, row_h=0.44, font=10,
      aligns=[PP_ALIGN.LEFT] * 6)

# ================================================================ 07 lambda
s = new_slide("检测链（Eq.3–8）", "一个必须说清楚的数学问题：K=10 时 λ=3 不可达")
rect(s, 0.75, 1.55, 6.0, 3.15)
tf = textbox(s, 1.00, 1.75, 5.5, 2.8)
para(tf, "推导", 12, True, INK, first=True)
para(tf, "z_k = (E′_k − μ) / σ", 15, False, INK, before=10, font=MONO)
para(tf, "对任意 K 维向量，标准分的最大值满足", 11.5, False, INK2, before=8)
para(tf, "max_k z_k ≤ √(K − 1)", 15, False, INK, before=6, font=MONO)
para(tf, "K=10  →  √9 = 3.0：Eq.8 是严格 “>”，上界恰好等于 3，不可达", 11.5, False, CRIT, before=10)
para(tf, "K=43  →  √42 ≈ 6.48：λ=3 才有意义", 11.5, False, GOOD, before=4)

rows = [
    ["对象", "z（λ=2.2）", "判定"],
    ["干净模型", "2.422", ("标记", CRIT)],
    ["BadNets", "2.725", ("标记 · target 0", CRIT)],
    ["Blend", "2.225", ("标记 · target 3", CRIT)],
]
table(s, 6.95, 1.75, [2.1, 1.6, 2.0], rows, row_h=0.52, font=11.5,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.LEFT])
tf = textbox(s, 6.95, 4.05, 5.6, 0.9)
para(tf, "λ=3 时三者都不被标记。", 11.5, False, CRIT, first=True)
para(tf, "论文 Table 2 中 CIFAR-10 一列的检测结论，在严格不等号下无法成立。", 10.5, False, INK2, before=4)

rect(s, 0.75, 5.00, 11.83, 1.55)
tf = textbox(s, 1.00, 5.18, 11.3, 1.2)
para(tf, "这不是调参问题", 12, True, INK, first=True)
para(tf, "λ 是一个判据，不是超参。上界由类别数决定，与数据、模型、训练都无关。"
         "论文把 λ 的可选区间写成 [1.8, 5.7]，恰好覆盖 GTSRB 的 K=43 上界 6.48——"
         "这更像是从 GTSRB 实验里得到的经验值，但论文把它当作通用默认值。", 11.5, False, INK2, before=6, spacing=1.25)

# ================================================================ 08 mechanism
s = new_slide("机制验证（Fig.1 / Fig.2）", "两个定性机制都成立，方向完全一致")
card(s, 0.75, 1.55, 5.85, 2.20, "Fig.1：任务无关图像被排进目标类", [
    "后门模型把 OOD 图像预测成目标类：",
    "BadNets  top-3 命中 0.678，平均排名 2.27",
    "干净模型  top-3 命中 0.337，平均排名 4.18",
    "Blend  top-3 命中 0.367，平均排名 3.84",
], accent=BLUE, body_size=10.5)

card(s, 6.73, 1.55, 5.85, 2.20, "Fig.2：损坏让目标类概率掉、真类概率升", [
    "逐损坏统计概率变化（80 种取平均）：",
    "BadNets  目标类 −0.8075，真类 +0.6770",
    "Blend  目标类 −0.8847，真类 +0.8287",
    "这正是 NTE 能用于复原标签的前提。",
], accent=ORANGE, body_size=10.5)

rect(s, 0.75, 4.00, 11.83, 1.80)
tf = textbox(s, 1.00, 4.18, 11.3, 1.5)
para(tf, "读法", 12, True, INK, first=True)
para(tf, "Fig.1 说明“为什么只用任务无关图像就能检测”：后门会给目标类一个人为的能量偏置，"
         "这个偏置与输入是不是真实任务样本无关。", 11.5, False, INK2, before=8)
para(tf, "Fig.2 说明“为什么损坏能暴露原始标签”：损坏破坏的是后门特征的稳定性，干净特征恢复得更快。"
         "两个面板都只在后门模型上成立，干净模型的对应数值接近随机。", 11.5, False, INK2, before=6)

hline(s, 0.75, 6.05, 11.83)
tf = textbox(s, 0.75, 6.18, 11.8, 0.35)
para(tf, "两项机制都是分布偏移而非二值判定——这也是后面 PDR 对特征空间如此敏感的原因。",
     10.5, False, MUTED, first=True)

# ================================================================ 09 NTE
s = new_slide("瓶颈定位（Eq.9–12）", "Blend 的 PDR 为什么低：恢复率只有 0.36")
rows = [
    ["指标", "BadNets", "Blend", "论文"],
    ["Eq.12 标签恢复率", "0.7802", ("0.3622", CRIT), "聚合后 >90%"],
    ["+ 类中心化变体", ("0.8329", GOOD), ("0.5411", WARN), "—"],
    ["干净目标类恢复率", ("≈0 → 0.585", GOOD), ("≈0 → 0.422", GOOD), "未报告"],
]
table(s, 0.75, 1.70, [3.2, 2.6, 2.6, 3.43], rows, row_h=0.58, font=12,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT])

rect(s, 0.75, 4.15, 11.83, 1.35)
tf = textbox(s, 1.00, 4.31, 11.3, 1.1)
para(tf, "恢复率是整条链的总闸", 12, True, INK, first=True)
para(tf, "聚类拿到的标签本身就是 Eq.12 的输出。恢复错了，聚类分得再干净也只会把错误标签固化下来。"
         "Blend 的恢复率 0.36，意味着六成以上样本的“原始标签”是错的——这是它 PDR 只有 0.648 的直接原因。",
     11.5, False, INK2, before=6, spacing=1.25)

card(s, 0.75, 5.62, 5.85, 1.20, "类中心化做了什么", [
    "对 NTE 分数减去逐类均值，去掉类别偏置。",
    "恢复率明显上升，但去除后 ASR 几乎不变（见第 12 页）。",
], accent=GOOD, title_size=11.5, body_size=9.5)
card(s, 6.73, 5.62, 5.85, 1.20, "为什么它不是“解决方案”", [
    "恢复率与去除效果在这套数据上不单调对应。",
    "真正的瓶颈在聚类划分，不在打分环节。",
], accent=CRIT, title_size=11.5, body_size=9.5)

# ================================================================ 10 clustering
s = new_slide("聚类与重标注（Table 3）", "BadNets 接近论文，Blend 差 30 个点")
rows = [
    ["攻击", "PDR 复现", "PDR 论文", "F1 复现", "F1 论文", "精确率", "TP / FP / FN", "聚类极差"],
    ["BadNets", "0.8908", "0.96", "0.9353", "0.96", "0.9845", "4382 / 69 / 537", "0.0307"],
    ["Blend", ("0.6480", CRIT), "0.95", ("0.7710", CRIT), "0.94", "0.9515", "3237 / 165 / 1758", "0.0176"],
]
table(s, 0.75, 1.70, [1.6, 1.45, 1.35, 1.35, 1.35, 1.25, 2.2, 1.28], rows, row_h=0.58, font=11.5,
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 7)

rect(s, 0.75, 3.58, 11.83, 1.35)
tf = textbox(s, 1.00, 3.74, 11.3, 1.1)
para(tf, "精确率高、召回低", 12, True, INK, first=True)
para(tf, "BadNets 只错了 69 个干净样本（FP），但漏了 537 个中毒样本；Blend 漏了 1,758 个。"
         "说明判据偏保守——被判为中毒的确实多半是中毒的，问题在于另一簇里混进了大量中毒样本。",
     11.5, False, INK2, before=6, spacing=1.25)

card(s, 0.75, 5.10, 3.80, 1.70, "四种聚类并列", [
    "HC / Birch / MeanShift / DBSCAN",
    "PDR 极差 0.031 / 0.018",
    "论文 Table 5 报的是 0.010",
], accent=BLUE, title_size=11.5, body_size=9.5)
card(s, 4.77, 5.10, 3.80, 1.70, "Oracle 上界", [
    "假设已知哪个簇是中毒簇",
    "PDR 与主结果完全相同",
    "→ 瓶颈是两簇划分本身",
], accent=ORANGE, title_size=11.5, body_size=9.5)
card(s, 8.79, 5.10, 3.80, 1.70, "特征空间偏离", [
    "论文用 t-SNE 投影后聚类",
    "我们用 512 维特征标准化 + Birch",
    "原因：t-SNE 解随 GPU 批次漂移",
], accent=MUTED, title_size=11.5, body_size=9.5)

# ================================================================ 11 removal
s = new_slide("去除效果（Table 4）", "只改预测不重训：ASR 降到 10.9% / 35.2%，论文是 2.9% / 5.8%")
# chart geometry
cx, cy, cw, ch = 1.35, 1.95, 9.4, 3.3
vmax = 40.0
for v in (0, 10, 20, 30, 40):
    y = cy + ch * (1 - v / vmax)
    hline(s, cx, y, cw, HAIR, 0.01)
    tf = textbox(s, cx - 0.62, y - 0.13, 0.55, 0.24)
    para(tf, str(v), 9, False, MUTED, PP_ALIGN.RIGHT, first=True)
hline(s, cx, cy + ch, cw, RULE, 0.02)

groups = [("BadNets", 10.92, 2.92), ("Blend", 35.20, 5.79)]
bw, gap = 0.78, 0.09
gx = [cx + 1.55, cx + 5.35]
for (label, ours, paper), gxx in zip(groups, gx):
    for k, (val, color, name) in enumerate([(ours, BLUE, "复现"), (paper, ORANGE, "论文")]):
        bx = gxx + k * (bw + gap)
        bh = ch * val / vmax
        bar = rect(s, bx, cy + ch - bh, bw, bh, fill=color, shape=MSO_SHAPE.RECTANGLE)
        tf = textbox(s, bx - 0.15, cy + ch - bh - 0.30, bw + 0.3, 0.26)
        para(tf, f"{val:.2f}%", 10.5, True, INK, PP_ALIGN.CENTER, first=True)
    tf = textbox(s, gxx - 0.35, cy + ch + 0.12, 2.2, 0.3)
    para(tf, label, 11.5, True, INK, PP_ALIGN.CENTER, first=True)

# legend
for i, (name, color) in enumerate([("复现", BLUE), ("论文 EBBA+", ORANGE)]):
    lx = 10.10
    ly = 1.55 + i * 0.30
    rect(s, lx, ly + 0.06, 0.16, 0.16, fill=color, shape=MSO_SHAPE.RECTANGLE)
    tf = textbox(s, lx + 0.24, ly, 1.6, 0.24)
    para(tf, name, 9.5, False, INK2, first=True)

tf = textbox(s, 10.05, 2.55, 2.45, 2.0)
para(tf, "论文 EBBA+ 的 BA：", 9.5, True, INK2, first=True, spacing=1.2)
para(tf, "89.92 / 89.72", 10, False, INK2, before=2)
para(tf, "复现后的 BA：", 9.5, True, INK2, before=8)
para(tf, "92.86 / 82.72", 10, False, INK2, before=2)

rect(s, 0.75, 5.88, 11.83, 0.95)
tf = textbox(s, 1.00, 6.00, 11.3, 0.75)
para(tf, "BadNets 的去除已接近论文量级；Blend 的 35.2% 说明“不重训”这条路径在强触发器上会失效。",
     11.5, False, INK2, first=True)
para(tf, "注意口径：论文是全测试集，我们是池内（5,000 中毒 + 1,000 干净），两列不能逐格直接相减。",
     9.5, False, MUTED, before=4)

# ================================================================ 12 ablations
s = new_slide("消融与负结果", "四个“看起来该有用”的改动，三个被证伪")
items = [
    ("NTE 上限（cap）是空操作", "T=1 时分数在 [0,1]，单次损坏的 NTE ≤ e−1 ≈ 1.72，任何 ≥2 的 cap 永远不生效；"
     "T=2 上界升到 e²−1 ≈ 6.39，cap=5 才可能绑定。论文与官方实现之间 cap 的差异因此解释不了任何结果差异。", CRIT),
    ("池内二次阈值：失败", "在选中簇里用 BIC 门控切两成分高斯，意图捞回误判的干净样本。"
     "PDR 从 0.891 掉到 0.256。作为负结果保留在报告里。", CRIT),
    ("Oracle 上界：等于主结果", "假设直接知道哪个簇是中毒簇，PDR 与主结果完全相同。"
     "说明问题不在选簇规则，而在 Birch 把两类样本分成两簇这件事本身。", WARN),
    ("类中心化：一半有效", "恢复率 0.78→0.83、0.36→0.54，干净目标类恢复从 ~0 提到 0.59/0.42；"
     "但去除后 ASR 几乎不动（10.9→15.3、35.2→36.4）。", GOOD),
]
for i, (title, body, accent) in enumerate(items):
    x = 0.75 + (i % 2) * 5.97
    y = 1.55 + (i // 2) * 2.36
    rect(s, x, y, 5.80, 2.16)
    rect(s, x, y + 0.14, 0.045, 1.88, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    tf = textbox(s, x + 0.26, y + 0.18, 5.3, 0.45)
    para(tf, title, 13, True, INK, first=True)
    tf = textbox(s, x + 0.26, y + 0.66, 5.3, 1.4)
    para(tf, body, 11, False, INK2, first=True, spacing=1.3)

hline(s, 0.75, 6.32, 11.83)
tf = textbox(s, 0.75, 6.45, 11.8, 0.35)
para(tf, "三个被证伪的改动指向同一件事：瓶颈不在打分或阈值，而在两簇划分本身。",
     11.5, False, INK, first=True)

# ================================================================ 13 paper issues
s = new_slide("论文问题（18 条 CONTRADICTED 摘选）", "不是转述争议，是可以当场复算的反例")
rows = [
    ["#", "论文断言", "复核结果", "判定"],
    ["A20", "λ=3 可检出 CIFAR-10 后门模型", "K=10 时 max z ≤ 3，严格 “>” 不可达", ("逻辑错误", CRIT)],
    ["A01", "附录 B 算例：s₂=[0.12,0.88,0]，E=[4.82,4.43,4.62]", "按论文 logits 复算：s₂=[0.007,0.993,0]，E=[4.724,4.719,4.669]", ("数值错误", CRIT)],
    ["A08", "Lemma 2 逐损坏普适成立", "同时满足两条不等式的比例 0.761；按损坏平均 0.763 < 0.90", ("有反例", CRIT)],
    ["A13", "BA 下界 (K−1)/K × BA", "ImageNet FIBA 88.09 < 88.179；DUBA 87.91 < 88.031", ("有反例", CRIT)],
    ["A12", "所有攻击 ASR ≥ 98.8%", "ImageNet Blend 98.62、FIBA 98.59", ("有反例", CRIT)],
    ["A15", "Table 4：EBBA+ 的 BA 最优、ASR 最低", "BA 最优 13/15 行；ASR 最低仅 5/15 行", ("过度声明", WARN)],
    ["A16", "Table 5 与 Table 7 相互独立", "两表数值逐格完全相同", ("疑似重复", WARN)],
    ["A19", "GTSRB 测试集 12,603 张", "标准为 12,630（数字转置）", ("笔误", WARN)],
]
table(s, 0.75, 1.60, [0.70, 4.30, 5.60, 1.23], rows, row_h=0.50, font=9.5,
      aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
tf = textbox(s, 0.75, 6.25, 11.8, 0.6)
para(tf, "另有官方 notebook 与论文设定不符：用 GTSRB 验证集、每类不均衡、仅 704 张、64 种损坏（含“除以 1”）、无聚类与 BA/ASR 评估。",
     9.5, False, MUTED, first=True, spacing=1.2)

# ================================================================ 14 coverage
s = new_slide("覆盖度", "复现的是“一条完整主线”，不是“整张实验矩阵”")
rows = [
    ["论文实验", "规模", "已覆盖", "比例"],
    ["Table 4 去除（3 数据集 × 5 攻击）", "15 行 × 6 baseline", "2 行", ("13%", CRIT)],
    ["Table 3 中毒样本检测", "15 行 × 3 baseline", "2 行", ("13%", CRIT)],
    ["Table 2 模型检测", "15 格 × 4 baseline", "2 格（1/3 场景）", ("13%", CRIT)],
    ["攻击种类", "BadNets/Blend/WaNet/FIBA/DUBA", "2 种", ("40%", WARN)],
    ["数据集", "CIFAR-10 / GTSRB / ImageNet", "1 个（GTSRB 仅作 OOD）", ("33%", WARN)],
    ["语音 / 文本（附录 G.1）", "ESC-50 + 文本替换", "未做", ("0%", CRIT)],
]
table(s, 0.75, 1.62, [4.50, 3.90, 2.30, 1.13], rows, row_h=0.50, font=10.5,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])

tf = textbox(s, 0.75, 5.32, 11.8, 1.5)
para(tf, "两个数字要分开说", 12, True, INK, first=True)
para(tf, "方法流水线覆盖 ≈90%：公式、损坏、聚类、重标注、训练配方都在，缺的是 t-SNE 的一致性和选簇规则的原作者写法。",
     11.5, False, INK2, before=8)
para(tf, "实验矩阵覆盖 ≈13%：baseline 一个没跑，“优于 SOTA”这句话本报告不背书，只背书 EBBA+ 的绝对数值。",
     11.5, False, INK2, before=5)

# ================================================================ 15 self score
s = new_slide("对照评议维度", "自评 3.68 / 5：发现问题强，实验完整度与工程交付弱")
dims = [
    ("(a) 完整验证论文内容", 0.30, 0.50, "实验矩阵只到 13%，baseline 为 0"),
    ("(b) 理解验证思路", 0.68, 0.75, "三层分层与容差纪律清晰；λ=2.2 是判断性取舍"),
    ("(c) 代码与注释清晰", 0.40, 0.50, "notebook 全中文；README 第 5 节仍是旧数值"),
    ("(d) 打包后可正常运行", 0.55, 0.75, "镜像比源码旧 19h；全量档需 8GB GPU 约 4–5h"),
    ("(e) 发现论文不足", 0.95, 1.00, "18 条 CONTRADICTED，含 1 条数学不可达"),
    ("(f) 找到新解决方案", 0.45, 1.00, "类中心化提升恢复率，但没降 ASR"),
    ("(g) 对方案进行探索", 0.35, 0.50, "4 变体 × 4 聚类 × oracle；仅 1 数据集 2 攻击"),
]
y0 = 1.52
for i, (name, score, full, note) in enumerate(dims):
    y = y0 + i * 0.66
    tf = textbox(s, 0.75, y, 3.7, 0.28)
    para(tf, name, 10.5, True, INK, first=True)
    rect(s, 4.55, y + 0.05, 3.0, 0.20, fill=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    frac = score / full
    rect(s, 4.55, y + 0.05, max(0.10, 3.0 * frac), 0.20, fill=BLUE,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    tf = textbox(s, 7.70, y - 0.02, 0.95, 0.26)
    para(tf, f"{score:.2f}/{full:.2f}", 10, True, INK2, first=True)
    tf = textbox(s, 8.80, y, 3.8, 0.3)
    para(tf, note, 9.5, False, MUTED, first=True)

hline(s, 0.75, 6.22, 11.83, RULE, 0.015)
tf = textbox(s, 0.75, 6.35, 5.0, 0.4)
para(tf, "合计", 13, True, INK, first=True)
tf = textbox(s, 4.55, 6.35, 3.0, 0.4)
para(tf, "3.68 / 5.00", 15, True, BLUE, first=True)
tf = textbox(s, 6.60, 6.38, 5.9, 0.4)
para(tf, "风险点：(d) 若评审当场重建镜像，分数可能进一步下降", 10, False, CRIT, first=True)

# ================================================================ 16 close
s = new_slide("结论与后续", "把“能跑”和“已覆盖”分开说，是这份报告的核心")
rect(s, 0.75, 1.55, 5.85, 2.35)
tf = textbox(s, 1.00, 1.75, 5.4, 2.0)
para(tf, "结论", 13, True, INK, first=True)
para(tf, "方法主体已复现：一条完整链路、四个对齐的攻击指标、两个成立的定性机制。", 11, False, INK2, before=10, spacing=1.3)
para(tf, "实验主体未复现：13% 的表格覆盖、0 个 baseline。", 11, False, INK2, before=6, spacing=1.3)
para(tf, "论文存在一个数学层面的问题（λ=3 / K=10）和若干可复算的数值错误。", 11, False, INK2, before=6, spacing=1.3)

rect(s, 6.73, 1.55, 5.85, 2.35)
tf = textbox(s, 6.98, 1.75, 5.4, 2.0)
para(tf, "下一步（按性价比）", 13, True, INK, first=True)
para(tf, "1. 修 Blend 的 Eq.12 恢复率——它是所有下游指标的总闸。", 11, False, INK2, before=10)
para(tf, "2. 补 WaNet / FIBA / DUBA，Table 4 从 2 行变 5 行。", 11, False, INK2, before=6)
para(tf, "3. 补 GTSRB 主实验与 2–3 个 baseline（NC / SCALE / FTD）。", 11, False, INK2, before=6)

rect(s, 0.75, 4.15, 11.83, 2.2)
tf = textbox(s, 1.00, 4.32, 11.3, 0.3)
para(tf, "一键复现", 12, True, INK, first=True)
cmds = [
    "docker compose run --rm ebba test",
    "docker compose run --rm ebba verify",
    "docker compose --profile full run --rm ebba-full verify-full --device cuda",
]
for i, c in enumerate(cmds):
    tf = textbox(s, 1.00, 4.70 + i * 0.40, 11.2, 0.3)
    para(tf, c, 11, False, INK2, first=True, font=MONO)
tf = textbox(s, 1.00, 5.95, 11.2, 0.3)
para(tf, "产物：reports/verdicts.json · reports/report.html · reports-full/full_metrics.json",
     9.5, False, MUTED, first=True)

out = Path(__file__).resolve().parents[1] / "slides" / "EBBA_复现汇报.pptx"
out.parent.mkdir(parents=True, exist_ok=True)
prs.save(out)
print(f"saved {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
