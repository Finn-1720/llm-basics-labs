# -*- coding: utf-8 -*-
"""生成表 8 需要的三段样例：教师 / 从零学生 / 蒸馏学生（提示词：月）

从零学生没有单独保存权重，这里用与 --mode all 完全相同的种子和训练流程复现一次，
并用 val_loss 与 results_all.json 对照，确认复现无误。
"""
import json
import os

import torch

from distill_lm import (GPT, TEACHER_CFG, STUDENT_CFG, make_data, evaluate,
                        train, build_corpus)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_distill")
SEED = 0
BLOCK = 96
BS = 24
torch.set_num_threads(8)

tok, train_ids, val_ids = make_data(BLOCK)

# 教师
teacher = GPT(tok.vocab_size, TEACHER_CFG)
teacher.load_state_dict(torch.load(os.path.join(OUT, "teacher.pt"), weights_only=True))

# 从零学生：与 run() 中 new_student() + train(seed=0) 完全一致
torch.manual_seed(SEED + 7)
scratch = GPT(tok.vocab_size, STUDENT_CFG)
train(scratch, train_ids, 120, BS, BLOCK, lr=3e-3, seed=SEED,
      log_every=10**9, tag="scratch-repro")
scratch_val = evaluate(scratch, val_ids, BS, BLOCK)

# 蒸馏学生
kd = GPT(tok.vocab_size, STUDENT_CFG)
kd.load_state_dict(torch.load(os.path.join(OUT, "student_all.pt"), weights_only=True))
kd_val = evaluate(kd, val_ids, BS, BLOCK)

ref = json.load(open(os.path.join(OUT, "results_all.json"), encoding="utf-8"))
print("[check] scratch val %.4f  参考 %.4f" % (scratch_val, ref["scratch"]["val_loss"]))
print("[check] kd      val %.4f  参考 %.4f" % (kd_val, ref["kd"]["val_loss"]))

prompt = "月"
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
out = {}
for name, m in (("teacher", teacher), ("scratch", scratch), ("kd", kd)):
    torch.manual_seed(12345)
    with torch.no_grad():
        g = m.generate(ids, 60, temperature=0.9, top_k=20)
    s = tok.decode(g[0].tolist())
    out[name] = s
    print("[%s] %s" % (name, s.replace("\n", " ")))

json.dump(out, open(os.path.join(OUT, "samples.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("[done] runs_distill/samples.json")
