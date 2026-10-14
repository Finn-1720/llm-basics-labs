# lab05 — SmolVLM-256M 的 QLoRA 高效微调与 DPO 偏好对齐

## 运行

```bash
pip install torch transformers peft accelerate bitsandbytes num2words torchvision pillow
export HF_HUB_OFFLINE=1                 # 模型已本地化，运行时不再联网

python hw5_dpo.py --mode make_data      # 生成 15 张合成图像
python hw5_dpo.py --mode infer          # 基线推理 15 题
python hw5_dpo.py --mode sft            # QLoRA SFT（30 步）
python hw5_dpo.py --mode dpo            # DPO（12 步）
python hw5_dpo.py --mode eval           # 三阶段对比
python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1   # 手算复核
```

敏感性扫描与扩展：

```bash
python hw5_dpo.py --mode sft --r 4      # 秩敏感性（任务五 b）
python hw5_dpo.py --mode sft --r 16
python hw5_dpo.py --mode dpo --beta 0.5 # β 敏感性（任务五 a）
python hw5_dpo.py --mode dpo --use-self-pairs   # 加入自拟 5 组偏好对
```

## 环境注意（本机为 Windows + RTX 5060 Ti / sm_120）

- **transformers 5.17 的破坏性变更**：`AutoModelForVision2Seq` 已删除，须用
  `AutoModelForImageTextToText`；`torch_dtype=` 已弃用，改用 `dtype=`
- **两个隐藏硬依赖**：`num2words`（SmolVLMProcessor 源码中直接 raise）与
  `torchvision`（idefics3 的图像处理模块在顶层导入它，连 PIL 后端也需要）
- **模型下载**：huggingface.co 不可达时改用 ModelScope
  （`HuggingFaceTB/SmolVLM-256M-Instruct`，16 个文件 / 494 MB），
  代码中的 `resolve_model_path()` 会优先加载本地目录

## 对参考代码的修改（5 处）与增补（3 处）

| # | 内容 | 原因 |
|---|---|---|
| 修改 1 | `AutoModelForVision2Seq` → `AutoModelForImageTextToText` | transformers v5 已移除该类 |
| 修改 2 | `torch_dtype=` → `dtype=` | v5 已弃用 |
| 修改 3 | `resolve_model_path()` 本地模型优先 | 本机无法访问 huggingface.co |
| 修改 4 | 修正 `dpo_check()` 的符号错误 | 指南原式会算出**负损失** |
| 修改 5 | 去掉量化模型上的 `.float()` | v5 对量化模型禁用 `.float()` |
| 增补 1 | `--r` 参数 | 任务五(b) 秩敏感性扫描 |
| 增补 2 | `dpo_log.csv` 逐步落盘 | 报告画训练曲线 |
| 增补 3 | `--use-self-pairs` + 自拟 5 组偏好对 | 任务三 |

## 目录

| 路径 | 内容 |
|---|---|
| `hw5_dpo.py` | 主程序（含上述 5 处修改 + 3 处增补） |
| `make_figs.py` | 生成报告插图（读真实日志，无手工数字） |
| `fill_report.py` | 用实测数据填报告样板 |
| `logs/` | 各阶段完整日志、逐题记录、DPO 逐步 loss/margin |
| `figs/` | 报告插图 3 张 |
| `scan_*.sh` / `time_all.sh` | 敏感性扫描与耗时实测脚本 |

## 关键实测结果

| 阶段 | count | attr | exist | 总计 |
|---|---|---|---|---|
| 基线 | 1/5 | 0/5 | 0/5 | 1/15 |
| SFT（30 步） | 2/5 | 0/5 | 0/5 | 2/15 |
| SFT+DPO（12 步） | 1/5 | 0/5 | 0/5 | 1/15 |

- 可训练参数占比：**0.5290%**（1,363,968 / 257,848,896），与 r 严格成正比
  （r=4 → 0.2652%，r=16 → 1.0524%）
- exist 类在三阶段**全部输出「No.」**——是整体偏置，不是「诚实」
- DPO 未提升任务正确率；β=0.5 时 margin 大幅为负（−2.106）正确率却最高（3/15）
