# -*- coding: utf-8 -*-
"""逐题导出：各范式检索到的文档、首个命中排名、Recall。"""
import io, sys
sys.argv = ["x"]
import hw4_rag as R
from hw4_rag import *

emb = R.Embedder("paraphrase-multilingual-MiniLM-L12-v2")
sysd = {"Vector": R.VectorRAG(emb), "Graph": R.GraphRAG(emb), "Wiki": R.WikiRAG(emb)}
out = io.StringIO()
w = lambda s="": out.write(s + "\n")
w(f"{'题号':<5}{'类型':<7}{'范式':<8}{'首个命中排名':>7}{'Recall':>8}  检索到的文档")
rows = []
for q in R.QUESTIONS:
    for nm, s in sysd.items():
        kw = {"scope": "global"} if (nm == "Graph" and q["type"] == "global") else {}
        docs = s.retrieve(q["q"], 5, **kw)
        ev = set(q["evidence"])
        rank = next((i + 1 for i, d in enumerate(docs) if d in ev), 0)
        rec = len(ev & set(docs)) / len(ev)
        miss = [d for d in ev if d not in docs]
        rows.append((q["id"], q["type"], nm, rank, rec, docs, miss))
        w(f"{q['id']:<5}{q['type']:<7}{nm:<8}{rank:>7}{rec:>8.3f}  {docs}")
    w()
io.open("results/qdetail.txt", "w", encoding="utf-8").write(out.getvalue())

w2 = io.StringIO()
w2.write("=== 有失败的题目（Recall<1）===\n")
for qid, tp, nm, rank, rec, docs, miss in rows:
    if rec < 1.0:
        w2.write(f"{qid} [{tp}] {nm}: Recall={rec:.3f} 首次命中排名={rank}  漏掉={miss}\n")
io.open("results/failures.txt", "w", encoding="utf-8").write(w2.getvalue())
print(out.getvalue()[:1200])
