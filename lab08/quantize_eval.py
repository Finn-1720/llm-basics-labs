# -*- coding: utf-8 -*-
"""拓展任务 D：动态量化导出（int8）+ 体积 / 延迟 / 留出集损失测量。

对象：distill_lm.py --mode all 训练出的蒸馏学生 runs_distill/student_all.pt
输出：runs_distill/quant_result.json、runs_distill/student_int8.pt
"""
import json
import os
import time

import torch
import torch.nn as nn

from distill_lm import GPT, STUDENT_CFG, make_data, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_distill")
BLOCK = STUDENT_CFG["block_size"]

torch.set_num_threads(8)


def model_size_mb(state_dict):
    total = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return total / 1024 / 1024


def measure_latency(model, batch_size=1, iters=30):
    x = torch.randint(0, 100, (batch_size, BLOCK))
    model.eval()
    with torch.no_grad():
        for _ in range(5):          # 预热
            model(x)
        t0 = time.time()
        for _ in range(iters):
            model(x)
        dt = (time.time() - t0) / iters
    return dt * 1000  # ms


def main():
    tok, train_ids, val_ids = make_data(BLOCK)

    model = GPT(tok.vocab_size, STUDENT_CFG)
    sd = torch.load(os.path.join(OUT, "student_all.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    fp32_loss = evaluate(model, val_ids, 24, BLOCK)
    fp32_size = model_size_mb(model.state_dict())
    fp32_lat1 = measure_latency(model, 1)
    fp32_lat24 = measure_latency(model, 24)

    q = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    int8_loss = evaluate(q, val_ids, 24, BLOCK)
    int8_size = model_size_mb(q.state_dict())
    int8_lat1 = measure_latency(q, 1)
    int8_lat24 = measure_latency(q, 24)

    torch.save(q.state_dict(), os.path.join(OUT, "student_int8.pt"))

    res = {
        "fp32": {"size_mb": fp32_size, "val_loss": fp32_loss,
                 "ppl": float(torch.exp(torch.tensor(fp32_loss))),
                 "latency_ms_b1": fp32_lat1, "latency_ms_b24": fp32_lat24},
        "int8": {"size_mb": int8_size, "val_loss": int8_loss,
                 "ppl": float(torch.exp(torch.tensor(int8_loss))),
                 "latency_ms_b1": int8_lat1, "latency_ms_b24": int8_lat24},
        "size_ratio": fp32_size / int8_size,
        "speedup_b1": fp32_lat1 / int8_lat1,
        "speedup_b24": fp32_lat24 / int8_lat24,
    }
    with open(os.path.join(OUT, "quant_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("[quant] fp32  size=%.3f MB  val_loss=%.4f  b1=%.2f ms  b24=%.2f ms"
          % (fp32_size, fp32_loss, fp32_lat1, fp32_lat24))
    print("[quant] int8  size=%.3f MB  val_loss=%.4f  b1=%.2f ms  b24=%.2f ms"
          % (int8_size, int8_loss, int8_lat1, int8_lat24))
    print("[quant] 体积压缩 %.2fx  加速 b1=%.2fx  b24=%.2fx  loss 变化 %+.4f"
          % (fp32_size / int8_size, fp32_lat1 / int8_lat1,
             fp32_lat24 / int8_lat24, int8_loss - fp32_loss))


if __name__ == "__main__":
    main()
