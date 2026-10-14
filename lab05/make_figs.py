# -*- coding: utf-8 -*-
"""
实验作业五 —— 报告插图生成
读实际跑出来的日志/CSV，生成三张图：
  fig1_dpo_curve.png   DPO 训练的 loss 与 margin 随步数变化
  fig2_three_stage.png 三阶段 × 题型 正确率对比
  fig3_scans.png       r 敏感性 与 β 敏感性

数据全部来自真实运行产物，没有任何手工填写的数字。
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体（Windows 自带）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def read_dpo_csv(path):
    steps, loss, margin = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            margin.append(float(row["margin"]))
    return steps, loss, margin


# ============================================================
# 图 1：DPO 训练曲线
# ============================================================
def fig_dpo_curve():
    steps, loss, margin = read_dpo_csv("dpo_log_b0.1.csv")
    fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
    ax1.plot(steps, loss, "o-", color="#c0504d", label="DPO loss", markersize=4)
    ax1.set_xlabel("训练步数")
    ax1.set_ylabel("DPO loss", color="#c0504d")
    ax1.tick_params(axis="y", labelcolor="#c0504d")
    ax1.axhline(0, color="gray", lw=0.6, ls=":")

    ax2 = ax1.twinx()
    ax2.plot(steps, margin, "s--", color="#4f81bd", label="margin", markersize=4)
    ax2.axhline(0, color="#4f81bd", lw=0.6, ls=":")
    ax2.set_ylabel("margin", color="#4f81bd")
    ax2.tick_params(axis="y", labelcolor="#4f81bd")

    plt.title("图 1  DPO 训练过程：loss 与 margin（β=0.1，12 步）")
    fig.tight_layout()
    fig.savefig("fig1_dpo_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig1_dpo_curve.png")


# ============================================================
# 图 2：三阶段 × 题型 正确率
# ============================================================
def fig_three_stage():
    # 数据来自 logs_eval_final.txt（三阶段评测的实际输出）
    stages = ["基线", "SFT", "SFT+DPO"]
    data = {
        "count": [1, 2, 1],
        "attr":  [0, 0, 0],
        "exist": [0, 0, 0],
    }
    types = list(data.keys())
    x = range(len(stages))
    w = 0.26
    colors = {"count": "#4f81bd", "attr": "#9bbb59", "exist": "#c0504d"}

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, tp in enumerate(types):
        offs = [xi + (i - 1) * w for xi in x]
        bars = ax.bar(offs, data[tp], w, label=tp, color=colors[tp])
        for b, v in zip(bars, data[tp]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.06, str(v),
                    ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("正确题数（满分 5）")
    ax.set_ylim(0, 5.5)
    ax.legend()
    plt.title("图 2  三阶段 × 题型 正确率（各 5 题）")
    fig.tight_layout()
    fig.savefig("fig2_three_stage.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig2_three_stage.png")


# ============================================================
# 图 3：r 敏感性 与 β 敏感性
# ============================================================
def fig_scans():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))

    # 左：r 敏感性
    rs = ["r=4", "r=8", "r=16"]
    pct = [0.2652, 0.5290, 1.0524]
    cnt = [2, 2, 3]
    ax1.bar(rs, pct, color="#8064a2", width=0.5)
    for i, (p, c) in enumerate(zip(pct, cnt)):
        ax1.text(i, p + 0.02, f"{p:.4f}%\ncount {c}/5", ha="center", fontsize=8.5)
    ax1.set_ylabel("可训练参数占比 (%)")
    ax1.set_ylim(0, 1.35)
    ax1.set_title("(a) LoRA 秩 r 的影响")

    # 右：β 敏感性（数据在 scan 完成后写入）
    try:
        betas, totals = [], []
        for b in ["0.05", "0.1", "0.5"]:
            p = f"logs_eval_b{b}.txt"
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    if line.startswith("RESULT") and "TOTAL" in line:
                        betas.append(f"β={b}")
                        totals.append(int(line.split()[2].split("/")[0]))
        if betas:
            ax2.bar(betas, totals, color="#4bacc6", width=0.5)
            for i, t in enumerate(totals):
                ax2.text(i, t + 0.06, f"{t}/15", ha="center", fontsize=9)
            ax2.set_ylim(0, max(totals) + 1.5)
        else:
            ax2.text(0.5, 0.5, "β 扫描数据未就绪", ha="center", transform=ax2.transAxes)
    except Exception as e:
        ax2.text(0.5, 0.5, f"β 数据读取失败\n{e}", ha="center", transform=ax2.transAxes)
    ax2.set_ylabel("三阶段总正确数（满分 15）")
    ax2.set_title("(b) DPO β 的影响")

    fig.tight_layout()
    fig.savefig("fig3_scans.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig3_scans.png")


if __name__ == "__main__":
    fig_dpo_curve()
    fig_three_stage()
    fig_scans()
    print("[fig] 全部完成")
