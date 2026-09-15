# 实验作业二：ConvLSTM 的算法应用与改进

《人工智能——深度学习大模型智能体》实验作业二，对应教材第 4 章《循环神经网络》· 时空序列预测。

在自合成的弹跳小球序列上，从零实现 ConvLSTM 细胞（PyTorch 无内置实现），完成"看过去几帧、预测下一帧"任务，并通过控制变量对照实验改进模型。

## 文件说明

| 文件 | 说明 |
|------|------|
| `convlstm_bounce.py` | 实验主程序：数据合成、ConvLSTMCell/ConvLSTM/FlattenLSTM、训练循环、曲线与预测对比图（层数/通道/卷积核/输入帧数/损失函数/模型结构均可命令行配置） |
| `result_*.png` | 各组实验的损失与 MSE 曲线（基线、实验 1-6、最优组合 3 次复测）及预测帧对比图 |

## 环境与运行

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch numpy matplotlib

# 基线（1 层，C_h=32，K=3，看 4 帧，MSE）
python convlstm_bounce.py --name 基线 --png result_baseline.png

# 对照实验（每次只改一个变量）
python convlstm_bounce.py --layers 2 --png result_exp1.png        # 加深层数
python convlstm_bounce.py --hid 64 --png result_exp2.png          # 隐藏通道
python convlstm_bounce.py --k 5 --png result_exp3.png             # 卷积核
python convlstm_bounce.py --frames 8 --png result_exp4.png        # 输入帧数
python convlstm_bounce.py --model flatten --png result_exp5.png   # 全连接 LSTM 对照
python convlstm_bounce.py --loss l1 --png result_exp6.png         # L1 损失

# 最优组合 + 预测对比图
python convlstm_bounce.py --layers 2 --hid 64 --k 5 --seed 42 \
  --png result_best_run1.png --pred-png result_pred.png
```

数据由脚本自动生成（2200 条：训练 2000 + 测试 200，每条 10 帧 32×32 弹跳小球，种子 42），首次运行后缓存为 `bounce_data.npy`。

## 实验结果（固定随机种子，真实运行）

| 组 | 改动 | 测试 MSE | 测试 MAE | 耗时 (s) |
|----|------|---------|---------|---------|
| 基线 | 1 层，C_h=32，K=3，看 4 帧，MSE | 0.00458 | 0.01268 | 11.6 |
| 1 | 2 层 ConvLSTM | 0.00392 | 0.01188 | 24.3 |
| 2 | C_h 32→64 | 0.00414 | 0.01515 | 31.2 |
| 3 | K=3→K=5 | 0.00372 | 0.01352 | 25.9 |
| 4 | 看 4 帧→看 8 帧 | 0.00459 | 0.01266 | 23.5 |
| 5 | 全连接 LSTM（展平） | 0.00724 | 0.03347 | 2.0 |
| 6 | MSE→L1 | 0.01184 | 0.01264 | 12.4 |
| 最优 | 2 层 + C_h=64 + K=5 | **0.00245**（3 次平均） | **0.00943** | 252.0 |

参数量手算（基线）：细胞 4×(3×3×1×32 + 3×3×32×32 + 32) = 38144，输出卷积 289，总计 38433（与 `sum(p.numel())` 一致）。

详细分析与思考题作答见实验报告。
