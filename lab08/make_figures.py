# -*- coding: utf-8 -*-
"""生成报告用的两张图：
  fig1_curves.png  —— 教师 / 从零学生 / 蒸馏学生的训练损失曲线 + 留出集 loss 对比
  fig2_heatmap.png —— T x alpha 4x4 热力图（由 make_heatmap.py 生成，这里只复制文件名）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_distill")

for font in ("Microsoft YaHei", "SimHei", "SimSun"):
    plt.rcParams["font.sans-serif"] = [font]
    break
plt.rcParams["axes.unicode_minus"] = False

teacher = json.load(open(os.path.join(OUT, "results_teacher.json"), encoding="utf-8"))
main = json.load(open(os.path.join(OUT, "results_all.json"), encoding="utf-8"))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

curves = [
    (teacher["teacher"]["curve"], "教师 Teacher", "#1f77b4"),
    (main["scratch"]["curve"], "学生·从零 Student-scratch", "#ff7f0e"),
    (main["kd"]["curve"], "学生·蒸馏 Student-KD", "#2ca02c"),
]
for c, lab, col in curves:
    if c:
        ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.3, color=col)
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("training loss")
ax[0].set_title("训练损失曲线")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

names = ["教师", "从零学生", "蒸馏学生"]
vals = [main["teacher"]["val_loss"], main["scratch"]["val_loss"], main["kd"]["val_loss"]]
bars = ax[1].bar(names, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
for b, v in zip(bars, vals):
    ax[1].text(b.get_x() + b.get_width() / 2, v + 0.02, "%.4f" % v,
               ha="center", va="bottom", fontsize=8)
ax[1].set_ylabel("留出集 loss")
ax[1].set_title("留出集损失对比")
ax[1].set_ylim(0, max(vals) * 1.25)
ax[1].grid(alpha=0.3, axis="y")

fig.tight_layout()
path = os.path.join(OUT, "fig1_curves.png")
fig.savefig(path, dpi=160)
print("[fig1] saved", path)
