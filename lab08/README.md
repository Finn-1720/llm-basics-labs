# lab08 — 自制语言模型的知识蒸馏（纯 CPU）

教师 0.74 M 参数 → 学生 0.15 M 参数（5.0×），全流程纯 CPU，无外部数据、无预训练权重。

## 运行

```bash
pip install torch matplotlib          # CPU 版 torch 即可
python distill_lm.py --mode quick              # 自检，约 4 s
python distill_lm.py --mode teacher --retrain  # 训练教师，约 6 s
python distill_lm.py --mode all                # 从零 vs 蒸馏，约 8 s
python distill_lm.py --mode ablate             # T×α 16 组，约 22 s
```

拓展实验：

```bash
python distill_lm.py --mode all --feat 0.1 --tag feat01   # A 特征蒸馏
python make_heatmap.py                                     # B 热力图
python quantize_eval.py                                    # D 动态量化
```

## 实测结果（seed=0，T=4，α=0.5）

| 模型 | 参数量 | 步数 | 耗时 | 留出集 loss | PPL | 与教师一致率 |
|---|---|---|---|---|---|---|
| 教师 Teacher | 737,600 | 200 | 5.7 s | 0.9078 | 2.48 | — |
| 学生·从零 | 147,584 | 120 | 1.5 s | 1.4144 | 4.11 | 0.644 |
| 学生·蒸馏 | 147,584 | 120 | 2.7 s | 1.3216 | 3.75 | 0.707 |

相对下降 **6.6%**。T×α 最优：**T=8、α=0.3（1.5573）**；最差：T=8、α=0.9（1.7166）。

## 对参考代码的修改（1 处）

| # | 内容 | 原因 |
|---|---|---|
| 1 | 新增 `--tag` 参数 | 拓展 A 要用 `--feat 0` 与 `--feat 0.1` 各跑一次，原代码两次都写 `results_all.json` 会互相覆盖 |

其余代码与实验指南附录逐行一致。

## 拓展实验

- **A 特征蒸馏（FitNet）**：`--feat 0.1` → 留出集 1.3216 → 1.3154（−0.47%），一致率 0.707 → 0.706。收益很小，说明中间层特征与输出层软标签信息高度重叠。
- **B 热力图**：`results/heatmap.png`，4×4 网格，最优点红框标出。
- **D 动态量化**：`quantize_eval.py`。体积 0.633 MB → 0.175 MB（3.62×），loss 1.3216 → 1.3218（几乎无损）；但 batch=1 时 0.67 ms → 1.22 ms（变慢），batch=24 时 3.03 ms → 2.86 ms（快 6%）。

## 三个值得记录的问题

1. **指南 PDF 里的代码不能直接复制**：长行被折行，续行顶格且丢缩进，直接粘贴会 `SyntaxError`。用 PyMuPDF 逐字符取 x 坐标重建空格与缩进，再按括号/引号配对合并续行，最后 `ast.parse` 校验通过。
2. **torch 2.14 的量化 API 已迁移**：`torch.ao.quantization.quantize_dynamic` 会打印「migrate to torchao」警告，但功能仍可用。
3. **语料规模与指南略有出入**：本机实测 `vocab=323, train_chars=52457, val_chars=5657`，指南写的是 322 / 49457 / 5357。差值与 `n_lines` 相同（每条样本差 1 字符），但按指南附录代码逐行还原后就是本机这个数，故以实测为准。

## 未做的拓展

C（自蒸馏）、E（教师规模扫描）、F（换真实语料）未做，报告拓展表中标注为「未完成」。
