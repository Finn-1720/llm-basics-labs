#!/usr/bin/env bash
# 任务三：表 5 的 6 组对照实验（单一变量原则）
# exp1~exp5 统一用 --iters 1000；exp6 只改采样参数，不重新训练
set -u
export PYTHONIOENCODING=utf-8
# 可移植写法：PY 可用环境变量覆盖（如 PY=/path/to/venv/Scripts/python.exe）
PY=${PY:-python}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT" || exit 1
mkdir -p runs

run () {           # run <dir> <args...>
  local d="runs/$1"; shift
  mkdir -p "$d"
  echo "############ $d : $* ############"
  ( cd "$d" && "$PY" ../../min_llm.py "$@" --iters 1000 ) > "$d/run.log" 2>&1
  echo "  exit=$?  -> $(grep -E '模型参数量|初始 loss' "$d/run.log" | head -2 | tr '\n' ' ')"
}

# ---- exp1 学习率：1e-2 与 1e-4（基线 1e-3 见 01_baseline_cpu）----
run 02_exp1_lr1e-2  --lr 1e-2
run 03_exp1_lr1e-4  --lr 1e-4

# ---- exp2 层数：1 与 4（基线 2）----
run 04_exp2_L1      --n_layer 1
run 05_exp2_L4      --n_layer 4

# ---- exp3 嵌入维度：64(头2) 与 256(头8)（基线 128/4）----
run 06_exp3_E64_H2  --n_embd 64  --n_head 2
run 07_exp3_E256_H8 --n_embd 256 --n_head 8

# ---- exp4 上下文：32（基线 128）----
run 08_exp4_T32     --block_size 32

# ---- exp5 位置编码：去掉（基线 有）----
run 09_exp5_nopos   --no_pos

echo
echo "############ 全部完成 ############"
