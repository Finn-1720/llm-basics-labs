# lab03 — 手搓最小 LLM（字符级 GPT）

实验作业三的代码与全部实验记录。对应教材第 5 章《Transformer 模型》。

## 环境

```
Python            3.12.10
PyTorch           2.14.0+cpu     ← 基线（指南要求）
                  2.14.0+cu130   ← GPU 对比用，需 Blackwell (sm_120)
matplotlib        3.11.2
```

指南的基线命令是 CPU 版：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> 注意：RTX 5060 Ti 的 compute capability 是 **12.0（Blackwell, sm_120）**，
> 必须用 cu128 及以上的轮子，否则 `torch.cuda.is_available()` 会返回 `False`
> （老版本轮子不含 sm_120 的 kernel）。cu130 的 wheel 为 1.85 GB，
> 国内镜像（阿里云 324 KB/s、交大 575 KB/s）都比官方源（1.88 MB/s）慢。

## 文件

| 文件 | 说明 |
|---|---|
| `min_llm.py` | 主程序。字符级 GPT：词嵌入 + 可学习位置编码 + Pre-Norm Transformer 块 + 权重共享 |
| `run_experiments.sh` | 表 5 的 6 组对照实验，CPU 版 |
| `run_experiments_gpu.sh` | 同上，GPU 版 |
| `gather.py` | 把各次运行的日志与生成结果汇总成 `report_data.json` |
| `make_compare_fig.py` | 生成采样对比图 `sampling_compare.png` |
| `report_data.json` | 全部实验数据（供报告引用） |
| `runs/` | CPU 全部实验：日志、loss 曲线、生成原文 |
| `runs_gpu/` | GPU 全部实验 |

## 用法

```bash
# 基线：训练 2000 步并生成
python min_llm.py

# 快速跑通（800 步）
python min_llm.py --iters 800

# 对照实验
python min_llm.py --n_layer 1 --n_embd 64 --iters 1000
python min_llm.py --no_pos --iters 1000
python min_llm.py --temperature 1.5

# GPU（需 CUDA 版 PyTorch）
python min_llm.py --device cuda

# 全部对照实验
bash run_experiments.sh
```

### 参数

`--iters` `--batch_size` `--block_size` `--n_embd` `--n_head` `--n_layer`
`--lr` `--seed` `--temperature` `--top_k` `--max_new_tokens`
`--no_pos`（去掉位置编码）`--extra_corpus`（追加自备语料）
`--device {cpu,cuda,auto}`　`--save_model` / `--load_model`（采样实验复用权重）

## 基线配置

语料 72 首唐诗、2163 字，字符级分词，词表 **V = 699**。

| 项 | 值 |
|---|---|
| 嵌入维度 C | 128 |
| 注意力头数 H | 4（每头 32 维） |
| 层数 L | 2 |
| 上下文长度 T | 128 |
| MLP 隐层 | 4C = 512，GELU |
| 学习率 / 优化器 | 1e-3 / AdamW |
| 批大小 × 步数 | 32 × 2000 |

**参数量 502,656** —— 含偏置与 LayerNorm 时与公式 `V×C + T×C + L×12C² = 499,072`
相差 0.71%（要求 < 1%）。

## 主要结果

**基线（2000 步）**

| | CPU | GPU (RTX 5060 Ti) |
|---|---|---|
| 初始 loss | 6.5627 | 6.5626 |
| 最终 loss | 0.0263 | 0.0260 |
| 耗时 | 100.9 s | 10.7 s |
| 速度 | 19.8 it/s | 187.4 it/s |

GPU 快 **9.4×**。前 200 步 loss 逐步相同（1.4693 / 0.1524），说明是同一套数学。

**对照实验（1000 步）**

| 配置 | 参数量 | 最终 loss | CPU | GPU |
|---|---|---|---|---|
| 基线 | 502,656 | 0.0245 | 47.1 s | — |
| lr=1e-2 | 502,656 | 0.0729 | 63.6 s | 5.0 s |
| lr=1e-4 | 502,656 | 0.6444 | 47.4 s | 5.6 s |
| n_layer=1 | 304,384 | 0.0334 | 24.4 s | 3.9 s |
| n_layer=4 | 899,200 | 0.0318 | 97.3 s | 8.6 s |
| n_embd=64 (H2) | 153,024 | 0.0550 | 21.3 s | 5.4 s |
| n_embd=256 (H8) | 1,791,744 | 0.0259 | 124.4 s | 7.8 s |
| block=32 | 490,368 | 0.1108 | 11.4 s | 5.2 s |
| --no_pos | 486,272 | 0.0470 | 48.1 s | 5.3 s |

**采样（固定基线模型）**

| temperature | top_k | 整句照抄率 | 4-gram 重复率 |
|---|---|---|---|
| 1.0 | 20 | 0.89 | 0.001 |
| 0.5 | 20 | 0.92 | 0.001 |
| 1.5 | 20 | **0.25** | **0.039** |
| 1.0 | 5 | 0.92 | 0.001 |
| 1.0 | 699 | 0.85 | 0.007 |

温度 0.5 与 1.0 的输出**逐字相同**；只有 1.5 打破了逐字背诵。

## 两个值得记的观察

1. **小语料下模型是"背"而不是"学"**。语料仅 2163 字、模型 50 万参数，
   loss 可降到 0.026，生成近乎原句。这也是思考题 (5) 讨论的规模问题。

2. **去掉位置编码后生成仍基本成句**（loss 0.0470）。原因：因果掩码提供了方向性，
   而在记忆型任务里字符共现本身就携带了顺序信息。若语料大到记不住，
   位置编码的作用才会显现。
