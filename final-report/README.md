# 期末报告 — CPU-Budget Compression Stack

《人工智能——深度学习大模型与智能体》期末报告（arXiv 风格）。

**题目**：Compressing and Deploying Generative Models under a CPU-Only Budget:
A Three-Stage Study of Knowledge Distillation, Quantization and Parameter-Efficient Adaptation

**作者**：赖宇轩（2025311489） · 哈尔滨工业大学（深圳） · 指导教师：曾国坤

## 文件

| 路径 | 说明 |
|---|---|
| `2025311489_赖宇轩_期末报告.docx` | 期末报告正文（英文，8 章 + 参考文献 + 附录） |
| `demo/index.html` | 交互式演示系统（单文件、离线、无依赖） |
| `figures/` | 报告插图源文件（300 dpi PNG） |

## 报告主线

把课程中分散讲授的三种压缩手段放进同一条纯 CPU 流水线，测量它们的**交互**而非各自的效果：

| 阶段 | 手段 | 对应实验 | 关键结果 |
|---|---|---|---|
| A 容量 | 知识蒸馏 | lab08 | 0.74 M → 0.15 M（5.0×），留出集 CE 1.4144 → 1.3216（−6.6%） |
| B 精度 | 动态 INT8 量化 | lab08 | 0.633 → 0.175 MB（3.62×），CE 仅 +0.0002 |
| C 部署 | 4-bit 扩散 + 深度 ControlNet | lab07 | 512²/4 步 106.2 s；每步耗时随像素数线性增长 |
| 交叉验证 | QLoRA / RSI 闭环 | lab05, lab06 | 0.53% 可训练参数；闭环未产出新样本 |

## 演示系统

直接用浏览器打开 `demo/index.html`（无需 Python 环境）：

- 四模型对比卡片
- **T × α 双滑块**：拖动即可查看 16 格消融中的留出集 loss 与排名
- CPU 扩散延迟条形图

## 复现

```bash
git clone https://github.com/Finn-1720/llm-basics-labs.git
cd llm-basics-labs/lab08 && python distill_lm.py --mode all
cd ../lab07 && python scripts/run_pipeline.py --name test512 --res 512 --steps 4
```

报告中的每个数字都可在 `lab03`、`lab05`–`lab08` 的日志与结果文件中追溯。

## 说明

- 本工作未向 arXiv、会议或期刊投稿；附录 A 已如实说明。
- 报告正文为英文，遵循课程提供的 `Final-Report-Template-arXiv-style` 模板。
