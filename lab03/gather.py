# -*- coding: utf-8 -*-
"""把各次运行的日志与生成结果汇总成 report_data.json"""
import os, re, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

def read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

def parse_log(p):
    t = read(p)
    if not t: return None
    one = lambda re_: (re.search(re_, t).group(1) if re.search(re_, t) else None)
    steps = re.findall(r"step\s+\d+/(\d+) \| loss ([\d.]+)", t)
    return {
        "device": one(r"设备: (\w+)"),
        "params": one(r"模型参数量: ([\d,]+)"),
        "init":   one(r"初始 loss ([\d.]+)"),
        "final":  one(r"最终 loss ([\d.]+)"),
        "time":   one(r"耗时 ([\d.]+) 秒"),
        "iters":  steps[-1][0] if steps else None,
    }

def read_samples(p):
    """返回 [(prompt, idx, text), ...]"""
    t = read(p)
    out = []
    for blk in t.split("# 提示词"):
        if not blk.strip(): continue
        head, _, body = blk.partition("\n")
        m = re.match(r"「(.+?)」\s*#(\d+)", head.strip())
        if m: out.append((m.group(1), int(m.group(2)), body.strip()))
    return out

RUNS = {
    "baseline_2000_cpu": "runs/01_baseline_cpu",
    "baseline_2000_gpu": "runs/11_baseline_gpu",
    "baseline_1000":     "runs/00_baseline_1000",
    "exp1_lr1e-2":       "runs/02_exp1_lr1e-2",
    "exp1_lr1e-4":       "runs/03_exp1_lr1e-4",
    "exp2_L1":           "runs/04_exp2_L1",
    "exp2_L4":           "runs/05_exp2_L4",
    "exp3_E64H2":        "runs/06_exp3_E64_H2",
    "exp3_E256H8":       "runs/07_exp3_E256_H8",
    "exp4_T32":          "runs/08_exp4_T32",
    "exp5_nopos":        "runs/09_exp5_nopos",
    "gpu_exp1_lr1e-2":   "runs_gpu/02_exp1_lr1e-2",
    "gpu_exp1_lr1e-4":   "runs_gpu/03_exp1_lr1e-4",
    "gpu_exp2_L1":       "runs_gpu/04_exp2_L1",
    "gpu_exp2_L4":       "runs_gpu/05_exp2_L4",
    "gpu_exp3_E64H2":    "runs_gpu/06_exp3_E64_H2",
    "gpu_exp3_E256H8":   "runs_gpu/07_exp3_E256_H8",
    "gpu_exp4_T32":      "runs_gpu/08_exp4_T32",
    "gpu_exp5_nopos":    "runs_gpu/09_exp5_nopos",
}

data = {"runs": {}, "samples": {}, "sampling": {}, "curves": {}}
for name, d in RUNS.items():
    data["runs"][name] = parse_log(os.path.join(d, "run.log"))
    g = sorted(glob.glob(os.path.join(d, "generated_*.txt")))
    if g: data["samples"][name] = [{"prompt": p, "idx": i, "text": tx} for p, i, tx in read_samples(g[0])]
    c = sorted(glob.glob(os.path.join(d, "loss_curve_*.png")))
    if c: data["curves"][name] = c[0].replace("\\", "/")

for tag in ["T0.5_k20", "T1.0_k20", "T1.5_k20", "T1.0_k5", "T1.0_kall"]:
    d = os.path.join("runs/10_exp6_sampling", tag)
    g = sorted(glob.glob(os.path.join(d, "generated_*.txt")))
    data["sampling"][tag] = [{"prompt": p, "idx": i, "text": tx} for p, i, tx in read_samples(g[0])] if g else []
    c = sorted(glob.glob(os.path.join(d, "loss_curve_*.png")))
    if c: data["curves"]["sampling_" + tag] = c[0].replace("\\", "/")

json.dump(data, open("report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("=== runs ===")
for k, v in data["runs"].items():
    if v: print(f"  {k:<20} params={v['params']:<9} init={v['init']:<8} final={v['final']:<8} {v['time']}s  iters={v['iters']}")
print("\n=== sampling ===")
for k, v in data["sampling"].items():
    print(f"  {k:<12} {len(v)} 段")
print("\n=== curves ===")
for k, v in data["curves"].items(): print(f"  {k:<24} {v}")
print("\n-> report_data.json")
