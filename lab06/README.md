# lab06 — SmolVLM-256M 精准定位的 RSI 递归自进化

## 运行

```bash
pip install torch transformers peft accelerate pillow
export HF_HUB_OFFLINE=1                 # 模型已本地化

python hw6_rsi_localize.py --mode make_data          # 40 训练图 + 15 评测图 + GT
python hw6_rsi_localize.py --mode make_sft           # GT → 网格坐标种子数据 15 条
python hw6_rsi_localize.py --mode infer              # 基线 15 题
python hw6_rsi_localize.py --mode sft --steps 10     # QLoRA SFT
python hw6_rsi_localize.py --mode eval               # 三阶段评测
python hw6_rsi_localize.py --mode rsi --rounds 2 --steps 10   # RSI 两轮
```

拓展任务 A（GRID=8）在 `ext_grid8/` 下，代码把 `GRID, IMG = 4, 256` 改为 `8, 256`，
并把 `parse_cell` 的正则从 `[1-4]` 扩到 `[1-8]`。

## 对参考代码的修改（3 处）

| # | 内容 | 原因 |
|---|---|---|
| 1 | `AutoModelForVision2Seq` → `AutoModelForImageTextToText` | transformers v5 已移除该类 |
| 2 | `torch_dtype=` → `dtype=` | v5 已弃用 |
| 3 | `resolve_model_path()` 本地模型优先 | 本机无法访问 huggingface.co |

## 关键实测结果（GRID=4，指南默认 10 步）

| 阶段 | loc(8) | count(4) | exist(3) | 总计(15) | loc IoU |
|---|---|---|---|---|---|
| 基线 | 0/8 | 4/4 | 0/3 | 4/15 | 0.000 |
| SFT（10 步） | 0/8 | 4/4 | 0/3 | 4/15 | 0.000 |
| 纠错后 | 0/8 | 4/4 | 0/3 | 4/15 | 0.000 |
| RSI 第 1 轮 | 0/8 | 4/4 | 0/3 | 4/15 | 0.000 |
| RSI 第 2 轮 | 2/8 | 4/4 | 0/3 | 6/15 | 0.000 |

RSI 两轮：自查改写 **0** 次、验证器纠错 7 次、**新增可用样本 0 条**、数据池 15 → 15。

## 三个值得记录的问题

1. **40 张训练图从未被使用**：`make_dataset` 生成了它们，但 `build_questions`
   只遍历 `range(EVAL_N)` 读评测图，因此 SFT 与评测共用同一批图。
2. **10 步不足一个 epoch**（15 条样本），种子 SFT 连输出格式都没对齐；
   加大到 30 步后格式可学，但模型坍缩到众数答案「第1行第2列第3列第4列。」。
3. **count 题退化**：4 道 count 题标准答案全部为 1，「永远答 1」即可满分，该列不具区分度。

## 拓展 A 结论

GRID=8、训练 30 步时，模型输出退化为「第1行第2行第3行第4行第5行第6行」（连「列」都没有），
loc 与 IoU 均为 0，比 GRID=4 更差。输出空间从 16 类扩到 64 类而样本仍为 15 条，
每类不到 0.25 个样本——**离散粒度必须与模型容量和数据量匹配**。
