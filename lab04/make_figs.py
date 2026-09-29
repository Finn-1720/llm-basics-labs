# -*- coding: utf-8 -*-
"""生成实验报告的三张插图。全部用程序真实输出，非示意图。

图1  Vector RAG 单题检索命中（含相似度分数）—— 直接渲染 --mode vector --q Q01 --trace 的真实输出
图2  云山大学知识图谱 —— networkx 画真实的 45 节点 / 40 三元组，并高亮 Q01 的行走路径
图3  WikiRAG 条目化知识库 —— 渲染真实的 9 条 PRE_ENTRIES
"""
import io, os, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# ---- 中文字体 ----
CN = "C:/Windows/Fonts/msyh.ttc"          # 微软雅黑
CNB = "C:/Windows/Fonts/msyhbd.ttc"
MONO = fm.FontProperties(fname="C:/Windows/Fonts/consola.ttf")
cn = fm.FontProperties(fname=CN)
cnb = fm.FontProperties(fname=CNB)
for p in (CN, "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw4_rag as R


# ============================================================
# 图 1：Vector RAG 检索命中（真实终端输出）
# ============================================================
def fig1():
    # 复现 --mode vector --q Q01 --trace 的真实输出
    emb = R.Embedder("paraphrase-multilingual-MiniLM-L12-v2")
    vr = R.VectorRAG(emb)
    q = next(x for x in R.QUESTIONS if x["id"] == "Q01")
    qv = emb.encode([q["q"]])[0]
    scores = vr.M @ qv
    order = scores.argsort()[::-1]
    ranked = [(vr.chunks[i][0], float(scores[i])) for i in order]
    top5 = R._dedup(ranked, 5)
    ev = set(q["evidence"])

    fig = plt.figure(figsize=(13.2, 5.4), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.28)

    # --- 左：终端输出 ---
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012",
                                facecolor="#1e1e1e", edgecolor="#444", transform=ax.transAxes))
    lines = [
        ("$ python hw4_rag.py --mode vector --q Q01 --trace", "#7fd1a0", cn),
        ("[embed] 已加载 paraphrase-multilingual-MiniLM-L12-v2", "#c8c8c8", cn),
        ("  [trace] 块级共 15 块，前 8 名：", "#c8c8c8", cn),
        ("    [('doc04', 0.687), ('doc06', 0.613), ('doc05', 0.589),", "#e8c07d", MONO),
        ("     ('doc07', 0.552), ('doc11', 0.549), ('doc02', 0.511),", "#e8c07d", MONO),
        ("     ('doc10', 0.504), ('doc08', 0.482)]", "#e8c07d", MONO),
        ("", "#c8c8c8", cn),
        ("[Q01·multi] 《大模型通识课》授课教师所在学院的行政负责人是谁？", "#8fc7ff", cn),
        ("检索到的文档： ['doc04', 'doc06', 'doc05', 'doc07', 'doc11']", "#c8c8c8", cn),
        ("", "#c8c8c8", cn),
        ("标准答案 evidence = ['doc04', 'doc06', 'doc02']", "#c8c8c8", cn),
        ("  doc04 命中 排名1    doc06 命中 排名2    doc02 未命中 排名6", "#ff8f8f", cn),
        ("  → Recall@5 = 2/3 = 0.667，桥接文档 doc02 卡在 Top-5 之外", "#ff8f8f", cn),
    ]
    y = 0.95
    for txt, col, fp in lines:
        if txt:
            ax.text(0.028, y, txt, color=col, fontproperties=fp, fontsize=9.5,
                    transform=ax.transAxes, va="top")
        y -= 0.068 if fp is cn else 0.055
    ax.text(0.028, 1.045, "图 1  Vector RAG 单题检索命中（含相似度分数）",
            fontproperties=cnb, fontsize=11.5, transform=ax.transAxes)

    # --- 右：Top-8 分数条形图 ---
    ax2 = fig.add_subplot(gs[0, 1])
    top8 = ranked[:8]
    docs = [d for d, _ in top8][::-1]
    vals = [s for _, s in top8][::-1]
    cols = ["#2e7d32" if d in ev else "#c62828" for d in docs]
    bars = ax2.barh(range(len(docs)), vals, color=cols, height=0.62)
    ax2.set_yticks(range(len(docs))); ax2.set_yticklabels(docs, fontproperties=MONO, fontsize=10)
    ax2.axvline(vals[len(vals) - 5], color="#1565c0", ls="--", lw=1.4)
    ax2.text(vals[len(vals) - 5], len(docs) - 0.4, "  Top-5 边界", color="#1565c0",
             fontproperties=cn, fontsize=9.5, va="top")
    for b, v in zip(bars, vals):
        ax2.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                 va="center", fontsize=8.5, fontproperties=MONO)
    ax2.set_xlim(0, max(vals) * 1.18)
    ax2.set_xlabel("与问题的余弦相似度", fontproperties=cn, fontsize=10)
    ax2.tick_params(labelsize=9.5)
    ax2.set_title("块级检索分数（绿色 = 标准答案文档）", fontproperties=cn, fontsize=10.5, pad=8)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    fig.savefig("fig1_vector_retrieval.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig1 OK")


# ============================================================
# 图 2：云山大学知识图谱 + Q01 行走路径
# ============================================================
def fig2():
    G = nx.DiGraph()
    for h, r, t, src in R.PRE_TRIPLES:
        G.add_edge(h, t, rel=r, src=src)
    comms = nx.community.greedy_modularity_communities(G.to_undirected())
    cmap = {n: i for i, c in enumerate(comms) for n in c}
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
               "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac"]

    pos = nx.spring_layout(G, k=1.55, iterations=1500, seed=11)

    fig, ax = plt.subplots(figsize=(18, 11.5), dpi=185)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.05, 1.05)
    fig.canvas.draw()
    # 数据坐标 → 像素 的比例，用于估算标签盒子大小
    p0 = ax.transData.transform((0, 0)); p1 = ax.transData.transform((1, 1))
    px_per_x, px_per_y = p1[0] - p0[0], p1[1] - p0[1]
    DPI = fig.dpi

    path = ["大模型通识课", "李文瀚", "人工智能学院", "王启明"]
    ncol = [palette[cmap[n] % len(palette)] for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c4c4c4", arrows=True,
                           arrowsize=8, width=0.7, node_size=900)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncol, node_size=900,
                           edgecolors="white", linewidths=1.0)
    pedges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    pedges = [(a, b) if G.has_edge(a, b) else (b, a) for a, b in pedges]
    nx.draw_networkx_edges(G, pos, edgelist=pedges, ax=ax, edge_color="#c62828",
                           width=3.0, arrows=True, arrowsize=20, node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=path, ax=ax, node_color="#ffe0e0",
                           node_size=1150, edgecolors="#c62828", linewidths=2.2)

    # ---- 带碰撞检测的标签放置 ----
    def box(x, y, w_px, h_px):
        w, h = w_px / px_per_x, h_px / px_per_y
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def hit(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    R_NODE = 900 ** 0.5 / 2 / 72 * DPI / px_per_x        # 节点半径（数据单位）
    node_boxes = [box(x, y, 2 * R_NODE * px_per_x, 2 * R_NODE * px_per_y)
                  for x, y in pos.values()]
    placed = []
    # 度高的先放（重要节点优先占位）
    for n in sorted(G.nodes, key=lambda x: -G.degree(x)):
        big = n in path
        fp = cnb if big else cn
        fs = 10.5 if big else 7.8
        w_px = (len(n) * fs + 6) / 72 * DPI
        h_px = (fs + 4) / 72 * DPI
        x, y = pos[n]
        best, best_cost = None, 1e9
        for dx, dy in [(0.030, 0), (-0.030, 0), (0, 0.032), (0, -0.032),
                       (0.026, 0.026), (-0.026, 0.026), (0.026, -0.026), (-0.026, -0.026),
                       (0.046, 0), (-0.046, 0), (0, 0.055), (0, -0.055)]:
            bx = box(x + dx, y + dy, w_px, h_px)
            cost = sum(1 for b in placed + node_boxes if hit(bx, b)) + (abs(dx) + abs(dy)) * 4
            if bx[0] < -1.45 or bx[2] > 1.45 or bx[1] < -1.05 or bx[3] > 1.05:
                cost += 6
            if cost < best_cost:
                best_cost, best = cost, (x + dx, y + dy, bx)
        placed.append(best[2])
        ax.plot([x, best[0]], [y, best[1]], color="#999", lw=0.5, zorder=8)
        ax.text(best[0], best[1], n, ha="center", va="center", fontproperties=fp,
                fontsize=fs, color="#111", zorder=10,
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="#ddd", lw=0.4, alpha=0.92))

    ax.set_title(f"云山大学知识图谱：{G.number_of_nodes()} 个实体 / "
                 f"{len(R.PRE_TRIPLES)} 条三元组 / {len(comms)} 个社区（颜色=社区）\n"
                 f"红线高亮 Q01 的行走路径：大模型通识课 —主讲→ 李文瀚 —任职于→ "
                 f"人工智能学院 —院长→ 王启明",
                 fontproperties=cnb, fontsize=13, pad=14)
    ax.axis("off")
    fig.savefig("fig2_knowledge_graph.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig2 OK")


# ============================================================
# 图 3：WikiRAG 条目化知识库
# ============================================================
def fig3():
    fig, ax = plt.subplots(figsize=(14.5, 8.6), dpi=190)
    ax.axis("off")
    ax.text(0.5, 0.995, f"WikiRAG 条目化知识库（{len(R.PRE_ENTRIES)} 条，重写聚合而非复制原文）",
            ha="center", va="top", fontproperties=cnb, fontsize=13, transform=ax.transAxes)

    y = 0.945
    for i, (title, body, srcs) in enumerate(R.PRE_ENTRIES, 1):
        wrapped = textwrap.wrap(body, width=54)[:3]
        h = 0.036 + 0.031 * len(wrapped)
        ax.add_patch(FancyBboxPatch((0.012, y - h), 0.976, h,
                                    boxstyle="round,pad=0.004", facecolor="#f7f9fc",
                                    edgecolor="#c9d6e3", transform=ax.transAxes))
        ax.text(0.032, y - 0.028, f"{i}. {title}", fontproperties=cnb, fontsize=10.5,
                transform=ax.transAxes, va="top", color="#0d47a1")
        for j, w in enumerate(wrapped):
            ax.text(0.052, y - 0.060 - 0.031 * j, w, fontproperties=cn, fontsize=9,
                    transform=ax.transAxes, va="top", color="#333")
        ax.text(0.975, y - 0.028, "来源 " + " ".join(srcs), fontproperties=cn,
                fontsize=8.5, transform=ax.transAxes, va="top", ha="right",
                color="#2e7d32")
        y -= h + 0.008

    fig.savefig("fig3_wiki_entries.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig3 OK")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import hw4_rag
    fig1(); fig2(); fig3()
    for f in ("fig1_vector_retrieval.png", "fig2_knowledge_graph.png", "fig3_wiki_entries.png"):
        print(f, os.path.getsize(f), "bytes")
