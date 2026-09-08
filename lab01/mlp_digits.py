# -*- coding: utf-8 -*-
"""实验作业一：使用 PyTorch 从零构建 MLP，在 load_digits 上做控制变量对照实验。

用法示例：
  python mlp_digits.py --name baseline --act sigmoid --hidden 32 --opt sgd --lr 0.1 --png result_baseline.png
每组实验只改动一个变量（相对基线），其余配置保持不变。
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")                      # 无窗口环境也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]   # macOS 中文字体
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 数据 ----------
X, y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 16.0            # 归一化到 [0, 1]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)   # 8:2 分层划分，固定种子
X_tr = torch.tensor(X_tr)
y_tr = torch.tensor(y_tr, dtype=torch.long)   # 必须 Long，否则 CrossEntropyLoss 报错
X_te = torch.tensor(X_te)
y_te = torch.tensor(y_te, dtype=torch.long)

# ---------- 2. 模型 ----------
class MLP(nn.Module):
    """多层感知器：隐藏层结构、激活函数、Dropout、BatchNorm 均可参数配置。"""
    def __init__(self, hidden=(32,), act="sigmoid", dropout=0.0, use_bn=False):
        super().__init__()
        act_layer = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
                     "relu": nn.ReLU, "gelu": nn.GELU}[act]
        layers, prev = [], 64              # 输入层 64 维（8x8 图像展平）
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:                     # 改进点①：BatchNorm
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout > 0:                # 改进点②：Dropout
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 10)) # 输出层 10 类
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- 3. 训练与评估 ----------
def run(model, opt_name="sgd", lr=0.1, epochs=30, wd=0.0):
    if opt_name == "sgd":                  # 改进点③：优化器可切换
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"loss": [], "acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)  # 前向 + 损失
        loss.backward()                    # 反向传播
        opt.step()                         # 参数更新
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        hist["loss"].append(loss.item())
        hist["acc"].append(acc)
    return hist, time.time() - t0

# ---------- 4. 绘图 ----------
def save_fig(hist, title, png):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(hist["loss"], color="tab:red")
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(np.array(hist["acc"]) * 100, color="tab:blue")
    axes[1].set_title("测试准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率 (%)")
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp")
    ap.add_argument("--hidden", default="32")      # 逗号分隔，如 128,128
    ap.add_argument("--act", default="sigmoid")
    ap.add_argument("--opt", default="sgd")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bn", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--png", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)              # 固定随机种子，保证可复现
    hidden = tuple(int(x) for x in a.hidden.split(","))
    model = MLP(hidden=hidden, act=a.act, dropout=a.dropout, use_bn=a.bn)
    hist, dt = run(model, opt_name=a.opt, lr=a.lr, epochs=a.epochs, wd=a.wd)
    acc = hist["acc"][-1] * 100
    print(f"[{a.name}] 最终测试准确率: {acc:.2f}%  训练耗时: {dt:.2f}s  "
          f"初始loss={hist['loss'][0]:.2f} 末loss={hist['loss'][-1]:.4f}")
    if a.png:
        save_fig(hist, a.name, a.png)
