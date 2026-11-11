# -*- coding: utf-8 -*-
"""拓展任务 B：把 T x alpha 的 16 组消融结果画成 4x4 热力图。

数据来源：runs_distill/results_ablate.json（由 distill_lm.py --mode ablate 生成）
输出：runs_distill/heatmap.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_distill")

with open(os.path.join(OUT, "results_ablate.json"), encoding="utf-8") as f:
    res = json.load(f)

rows = res["ablation"]
Ts = sorted({r["T"] for r in rows})
As = sorted({r["alpha"] for r in rows})

grid = np.zeros((len(Ts), len(As)))
for r in rows:
    grid[Ts.index(r["T"]), As.index(r["alpha"])] = r["val_loss"]

best_idx = np.unravel_index(np.argmin(grid), grid.shape)
best_T, best_a = Ts[best_idx[0]], As[best_idx[1]]

# 中文字体
for font in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        plt.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(5.6, 4.4))
im = ax.imshow(grid, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(As)))
ax.set_xticklabels([f"α={a}" for a in As])
ax.set_yticks(range(len(Ts)))
ax.set_yticklabels([f"T={t}" for t in Ts])
ax.set_xlabel("权重 α")
ax.set_ylabel("温度 T")
ax.set_title("T × α 消融：留出集 loss（越暗越好）", fontsize=10)

for i in range(len(Ts)):
    for j in range(len(As)):
        v = grid[i, j]
        ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8,
                color="white" if v < grid.mean() else "black")
# 圈出最优点
ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                           fill=False, edgecolor="red", linewidth=2.2))
ax.text(best_idx[1], best_idx[0] + 0.32, "最优", ha="center", va="center",
        fontsize=8, color="red", fontweight="bold")

fig.colorbar(im, ax=ax, label="val loss")
fig.tight_layout()
path = os.path.join(OUT, "heatmap.png")
fig.savefig(path, dpi=160)
print("[heatmap] 最优 T=%g alpha=%g val=%.4f" % (best_T, best_a, grid[best_idx]))
print("[heatmap] saved", path)
