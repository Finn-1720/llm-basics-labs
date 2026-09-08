# 实验作业一：使用 TRAE IDE 设计更好的神经网络

《人工智能——深度学习大模型智能体》实验作业一，对应教材第 2 章《神经元与神经网络》。

在 sklearn 手写数字数据集（load_digits）上，用 PyTorch 从零构建 MLP 分类器，通过控制变量对照实验找到"更好的神经网络"。

## 文件说明

| 文件 | 说明 |
|------|------|
| `mlp_digits.py` | 实验主程序：数据划分、MLP 模型、训练循环、曲线绘制（隐藏层结构 / 激活函数 / Dropout / BatchNorm / 优化器 / 学习率 / weight_decay 均可通过命令行参数配置） |
| `verify_env.py` | 环境验证脚本 |
| `result_*.png` | 各组实验的损失与准确率曲线（基线、实验 1–7、最优组合 3 次复测、失败实验） |

## 环境

- Python 3.10+，PyTorch 2.0+，scikit-learn，NumPy，Matplotlib
- CPU 即可完成全部实验（单组 < 0.1 s）

## 运行方式

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch scikit-learn matplotlib numpy

# 验证环境
python verify_env.py

# 基线实验（其余各组同理，只改动一个参数）
python mlp_digits.py --name 基线 --act sigmoid --hidden 32 --opt sgd --lr 0.1 --png result_baseline.png

# 各实验组示例
python mlp_digits.py --act relu --png result_exp1.png                                  # 实验1 换激活函数
python mlp_digits.py --hidden 128,128 --png result_exp2.png                            # 实验2 加深网络
python mlp_digits.py --opt adam --lr 0.01 --png result_exp3.png                        # 实验3 换优化器
python mlp_digits.py --lr 0.01 --png result_exp4.png                                   # 实验4 调学习率
python mlp_digits.py --wd 1e-4 --png result_exp5.png                                   # 实验5 L2 正则
python mlp_digits.py --dropout 0.2 --png result_exp6.png                               # 实验6 Dropout
python mlp_digits.py --bn --png result_exp7.png                                        # 实验7 BatchNorm
python mlp_digits.py --act relu --hidden 128,128 --opt adam --lr 0.001 --bn \
  --seed 42 --png result_exp8_run1.png                                                 # 实验8 最优组合
python mlp_digits.py --lr 10 --png result_lr_too_big.png                               # 失败实验
```

## 实验结果（固定随机种子，真实运行）

| 组 | 改动 | 测试准确率 |
|----|------|-----------|
| 基线 | 1×32 Sigmoid + SGD(0.1) | 22.22% |
| 1 | Sigmoid → ReLU | 47.50% |
| 2 | 2 层 × 128 | 7.22% |
| 3 | SGD → Adam(0.01) | 82.22% |
| 4 | 学习率 0.1 → 0.01 | 10.00% |
| 5 | L2（wd=1e-4） | 22.22% |
| 6 | Dropout（p=0.2） | 21.39% |
| 7 | BatchNorm | 79.17% |
| 8 | ReLU + 2×128 + BN + Adam(0.001) | **95.65%（3 次平均）** |
| 失败 | 学习率 10.0 | 36.67%（震荡不收敛） |

详细分析与思考题作答见实验报告。
