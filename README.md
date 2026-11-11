# 大模型基础实验汇总

《大模型基础与应用探索》课程实验作业汇总仓库，每堂课的实验代码放在对应文件夹中。

## 目录结构

| 文件夹 | 内容 |
|--------|------|
| [lab01](lab01/) | 实验作业一：使用 TRAE IDE 设计更好的神经网络（PyTorch 手写数字分类，控制变量对照实验） |
| [lab02](lab02/) | 实验作业二：ConvLSTM 的算法应用与改进（弹跳小球时空序列预测，控制变量对照实验） |
| [lab03](lab03/) | 实验作业三：手搓最小 LLM（字符级 GPT，唐诗语料，CPU 基线 + GPU 对比 + 5 组控制变量对照 + 采样分析） |
| [lab04](lab04/) | 实验作业四：Vector RAG / GraphRAG / WikiRAG 对比实验（K 敏感性扫描 + MRR 手算复核 + 本地 Qwen3-14B 生成） |
| [lab05](lab05/) | 实验作业五：SmolVLM-256M QLoRA 微调与 DPO 偏好对齐（GPU 实测 + β/r 敏感性扫描 + DPO 手算复核） |
| [lab06](lab06/) | 实验作业六：SmolVLM-256M 精准定位的 RSI 递归自进化（网格坐标定位 + 验证器纠错闭环 + GRID=8 拓展） |
| [lab07](lab07/) | 实验作业七：白膜转写实纯 CPU 深度控制流水线（Blender 深度图 + SD-Turbo ControlNet，GGUF 模型见 Releases） |
| [lab08](lab08/) | 实验作业八：自制语言模型知识蒸馏（教师 0.74M → 学生 0.15M，T×α 消融 + 特征蒸馏/热力图/动态量化） |

## 实验一概要

在 sklearn 手写数字数据集（load_digits）上从零构建 MLP，通过基线 + 8 组对照实验 + 失败实验找到最优模型：

- 基线（1×32 Sigmoid + SGD 0.1）：22.22%
- 最优组合（ReLU + 2×128 + BatchNorm + Adam 0.001）：**95.65%**（3 次复测平均）

详见 [lab01/README.md](lab01/README.md)。

## 实验二概要

在自合成的弹跳小球序列上从零实现 ConvLSTM 细胞，完成"看过去几帧、预测下一帧"任务：

- 基线（1 层·C_h=32·K=3·看 4 帧·MSE）：测试 MSE 0.00458
- 最优组合（2 层 + C_h=64 + K=5）：测试 MSE **0.00245**（3 次复测平均，比基线降 46%）
- 参数量手算：细胞 38144 + 输出卷积 289 = 38433（与代码核对一致）

详见 [lab02/README.md](lab02/README.md)。

## 实验三概要

在 72 首唐诗（2163 字，词表 699）上从零实现字符级 GPT（嵌入 + 可学习位置编码 + Pre-Norm Transformer + 权重共享），完成"预测下一个字符"的预训练与自回归生成：

- 基线（C=128·H=4·L=2，50.3 万参数，2000 步）：最终 loss 0.0263，CPU 100.9 s
- 参数量手算 499,072 与代码核对 502,656，误差 0.71%（< 1%）
- 5 组控制变量实验（学习率 / 层数 / 嵌入维度 / 上下文 / 位置编码）+ 温度与 top-k 采样分析
- GPU（RTX 5060 Ti, sm_120）对比：快 9.4×，loss 曲线前 200 步与 CPU 逐步一致

详见 [lab03/README.md](lab03/README.md)。

## 实验四概要

在「云山大学」语料库（15 篇文档 + 20 道标注问题）上实现并对比三种 RAG 范式：

- Vector RAG（分块 + 向量检索）/ GraphRAG（三元组 + 实体图谱检索）/ WikiRAG（条目库检索）
- 指标：Recall@5、MRR、忠实度；K=1/2/3/5 敏感性扫描
- 自拟 5 道题（Q16–Q20）+ `--trace` 检索中间量调试 + Qwen3-14B 本地生成

详见 [lab04/README.md](lab04/README.md)。

## 实验五概要

在 SmolVLM-256M 视觉语言模型上完成 QLoRA 高效微调与 DPO 偏好对齐：

- 三阶段对比（基线 / SFT / SFT+DPO）× 15 题评测，可训练参数仅 0.53%
- DPO loss 手算复核（修正指南符号错误）+ β 与秩 r 敏感性扫描
- transformers v5 适配（AutoModelForImageTextToText、dtype=）+ ModelScope 本地模型

详见 [lab05/README.md](lab05/README.md)。

## 实验六概要

在 SmolVLM-256M 上实现「预测 → 渲染高亮格回灌 → 自查/验证器纠错 → 数据池回流」的 RSI 递归自进化定位实验：

- 4×4 网格坐标定位（16 类输出空间），解析式颜色掩码验证器（零训练开销）
- 三阶段评测 + 2 轮 RSI；拓展任务 GRID=8 对比（loc/IoU 退化分析）
- 记录三个设计问题：训练图未使用、count 题无区分度、10 步不足一个 epoch

详见 [lab06/README.md](lab06/README.md)。

## 实验七概要

Blender 白模场景 → Mist 深度图/法线图 → stable-diffusion.cpp（自编译纯 CPU）SD-Turbo Q4_0 + SD2.1 depth ControlNet 生成写实图：

- 512²/4 步 106.2s，768²/4 步 164.5s（M4 纯 CPU），每步耗时 ≈ k×像素数规律验证
- 几何一致性：clay 与 photoreal 边缘重合率 40%~53%
- Blender 5.2 API 适配（合成器大改）+ Metal 崩溃转纯 CPU + 首次 768² 异常耗时定位
- 模型文件（2.45GB）发布在 [Releases](https://github.com/Finn-1720/llm-basics-labs/releases)

详见 [lab07/README.md](lab07/README.md)。

## 实验八概要

纯 CPU 完成一次真实的字符级语言模型知识蒸馏（教师 0.74M 参数 → 学生 0.15M，压缩 5×）：

- 留出集 loss：从零学生 1.4144 → 蒸馏学生 1.3216（−6.6%），PPL 4.11 → 3.75，与教师一致率 0.644 → 0.707
- 16 组 T×α 消融：最优 T=8/α=0.3（1.5573），揭示「高温必须配小权重」的耦合规律
- 拓展 A 特征蒸馏（FitNet）、B 热力图、D 动态量化（体积 3.62×压缩，精度几乎无损）

详见 [lab08/README.md](lab08/README.md)。
