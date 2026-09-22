# -*- coding: utf-8 -*-
"""生成采样对比图：5 组 temperature/top_k 的生成原文并排（图形坐标绝对定位）"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

cjk = None
for name in ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi"]:
    try:
        font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
        cjk = name; break
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = [cjk or "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("中文字体:", cjk)

d = json.load(open("report_data.json", encoding="utf-8"))
SP = d["sampling"]
ORDER = [
    ("T1.0_k20",  "temperature = 1.0,  top_k = 20   （基线）"),
    ("T0.5_k20",  "temperature = 0.5,  top_k = 20"),
    ("T1.5_k20",  "temperature = 1.5,  top_k = 20"),
    ("T1.0_k5",   "temperature = 1.0,  top_k = 5"),
    ("T1.0_kall", "temperature = 1.0,  top_k = 699  （不限制）"),
]

def first(tag, prompt="春"):
    for s in SP.get(tag, []):
        if s["prompt"] == prompt: return s["text"]
    return ""

N = len(ORDER)
TOP_MARGIN, BOT_MARGIN = 0.055, 0.015
H = 0.205 * N + TOP_MARGIN + BOT_MARGIN          # 每格占的图形高度比例
fig = plt.figure(figsize=(9.6, 11.0))
fig.patch.set_facecolor("white")

# 标题
fig.text(0.5, 1 - TOP_MARGIN / 2, "不同 temperature / top_k 的生成原文对比（提示词「春」第 1 段）",
         ha="center", va="center", fontsize=12.5)

grid_h = 1 - TOP_MARGIN - BOT_MARGIN
cell_h = grid_h / N
for i, (tag, label) in enumerate(ORDER):
    top = 1 - TOP_MARGIN - i * cell_h
    bot = top - cell_h * 0.93
    # 边框
    fig.add_artist(plt.Rectangle((0.03, bot), 0.94, top - bot,
                                 transform=fig.transFigure, fill=False, lw=0.8, ec="#8a8a8a"))
    # 标签
    fig.text(0.048, top - cell_h * 0.14, label, fontsize=11, fontweight="bold", va="center")
    # 正文（最多 3 行，每行 34 字）
    body = first(tag).replace("\n", "").replace("\r", "")
    lines, cur = [], ""
    for ch in body:
        cur += ch
        if len(cur) >= 34:
            lines.append(cur); cur = ""
        if len(lines) == 3: break
    if cur and len(lines) < 3: lines.append(cur)
    for k, ln in enumerate(lines):
        fig.text(0.048, top - cell_h * 0.36 - k * cell_h * 0.19, ln, fontsize=10.5, va="center")

plt.savefig("sampling_compare.png", dpi=150)
print("-> sampling_compare.png")
