# -*- coding: utf-8 -*-
"""实验作业二：ConvLSTM 弹跳小球时空序列预测（控制变量对照实验）。

用法示例：
  python convlstm_bounce.py --name 基线 --png result_baseline.png
  python convlstm_bounce.py --name 实验2 --hid 64 --png result_exp2.png
每组实验只改动一个变量（相对基线），其余配置保持一致。
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]   # macOS 中文字体
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("运行设备:", DEVICE)

# ---------- 1. 数据合成：弹跳小球 ----------
def make_sequences(n_seq=2200, T=10, size=32, r=2, seed=42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    seqs = np.zeros((n_seq, T, size, size), np.float32)
    for s in range(n_seq):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        for t in range(T):
            x, y = x + vx, y + vy
            if x < r or x > size - r:
                vx = -vx
            if y < r or y > size - r:
                vy = -vy
            seqs[s, t] = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)
    return seqs[..., None]   # (N, T, 32, 32, 1)


def load_data():
    try:
        data = np.load("bounce_data.npy")
    except FileNotFoundError:
        data = make_sequences()
        np.save("bounce_data.npy", data)
    return data


data = load_data()
N_TRAIN = 2000
train_x = torch.tensor(data[:N_TRAIN, :8]).permute(0, 1, 4, 2, 3)  # 前8帧作输入池
train_y8 = torch.tensor(data[:N_TRAIN, 1:9]).squeeze(-1)           # 对应第2~9帧
test_x = torch.tensor(data[N_TRAIN:, :8]).permute(0, 1, 4, 2, 3)
test_y8 = torch.tensor(data[N_TRAIN:, 1:9]).squeeze(-1)
test_x4, test_y4 = test_x[:, :4], test_y8[:, 3]                    # 看4帧→预测第5帧
print(f"数据: 训练 {N_TRAIN} 条, 测试 {len(test_x)} 条, 每条 10 帧 32x32")

# ---------- 2. ConvLSTM ----------
class ConvLSTMCell(nn.Module):
    """把 LSTM 的矩阵乘法替换为卷积：一个 Conv2d 合并 4 个门。"""
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)
        self.hid = hid_ch

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = z.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_new = f * c + i * torch.tanh(g)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid=32, k=3, layers=1):
        super().__init__()
        chs = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(chs[i], chs[i + 1], k) for i in range(layers)])
        self.out = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, x):   # x: (B, T, C, H, W)
        b, _, _, hsize, wsize = x.shape
        states = [(torch.zeros(b, c.hid, hsize, wsize, device=x.device),
                   torch.zeros(b, c.hid, hsize, wsize, device=x.device))
                  for c in self.cells]
        for t in range(x.shape[1]):
            xt = x[:, t]
            for j, cell in enumerate(self.cells):
                states[j] = cell(xt, states[j])
                xt = states[j][0]
        return self.out(states[-1][0]).squeeze(1)


class FlattenLSTM(nn.Module):
    """结构对照：把 4 帧展平成 4x1024 向量喂全连接 LSTM。"""
    def __init__(self, frames=4, hid=256):
        super().__init__()
        self.frames = frames
        self.lstm = nn.LSTM(1024, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1024)

    def forward(self, x):
        b, t, _, hsize, wsize = x.shape
        x = x.reshape(b, t, -1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).reshape(b, hsize, wsize)


# ---------- 3. 训练与评估 ----------
def run(model, loss_name="mse", lr=0.001, epochs=5, bs=64, frames=4):
    loss_fn = nn.MSELoss() if loss_name == "mse" else nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_tr = train_x[:, :frames].to(DEVICE)
    y_tr = (train_y8[:, frames - 1]).to(DEVICE)
    x_te = test_x[:, :frames].to(DEVICE)
    y_te = (test_y8[:, frames - 1]).to(DEVICE)
    model = model.to(DEVICE)
    n = x_tr.shape[0]
    hist = {"loss": [], "mse": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        last_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            last_loss = loss.item()
        model.eval()
        with torch.no_grad():
            pred = model(x_te)
            mse = (pred - y_te).pow(2).mean().item()
        hist["loss"].append(last_loss)
        hist["mse"].append(mse)
        print(f"  epoch {ep + 1} train_loss {last_loss:.5f} test_MSE {mse:.5f}")
    with torch.no_grad():
        pred = model(x_te)
        mse = (pred - y_te).pow(2).mean().item()
        mae = (pred - y_te).abs().mean().item()
    return hist, time.time() - t0, mse, mae, model


def save_curve(hist, title, png):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(hist["loss"], color="tab:red")
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(hist["mse"], color="tab:blue")
    axes[1].set_title("测试 MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp")
    ap.add_argument("--model", default="convlstm")     # convlstm / flatten
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--loss", default="mse")           # mse / l1
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--png", default="")
    ap.add_argument("--params", action="store_true")   # 打印参数量明细
    ap.add_argument("--pred-png", default="")          # 生成"输入|真实|预测"对比图
    a = ap.parse_args()

    torch.manual_seed(a.seed)

    if a.model == "flatten":
        model = FlattenLSTM(frames=a.frames)
    else:
        model = ConvLSTM(hid=a.hid, k=a.k, layers=a.layers)

    if a.params:
        total = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total}")
        for name, p in model.named_parameters():
            print(f"  {name}: {tuple(p.shape)} = {p.numel()}")

    hist, dt, mse, mae, model = run(model, loss_name=a.loss,
                                    epochs=a.epochs, bs=a.bs, frames=a.frames)
    print(f"[{a.name}] 测试MSE {mse:.5f} 测试MAE {mae:.5f} 耗时 {dt:.1f}s")
    if a.png:
        save_curve(hist, a.name, a.png)

    if a.pred_png:
        # 输入帧 | 真实帧 | 预测帧 对比图（3 个测试样本）
        model.eval()
        x_te = test_x[:, :a.frames].to(DEVICE)
        y_te = (test_y8[:, a.frames - 1]).to(DEVICE)
        idxs = [0, 5, 10]      # 取 3 个样本
        n_in = a.frames
        fig, axes = plt.subplots(3, n_in + 2, figsize=(2.2 * (n_in + 2), 7))
        if n_in + 2 == 6:
            axes = axes.reshape(3, -1)
        with torch.no_grad():
            pred = model(x_te)
        for row, si in enumerate(idxs):
            frames = [test_x[si, j, 0].cpu() for j in range(n_in)]
            frames += [y_te[si].cpu(), pred[si].cpu()]
            titles = [f"输入帧{j+1}" for j in range(n_in)] + ["真实帧", "预测帧"]
            smse = (pred[si] - y_te[si]).pow(2).mean().item()
            for col, (fr, ti) in enumerate(zip(frames, titles)):
                ax = axes[row, col]
                ax.imshow(fr, cmap="viridis", vmin=0, vmax=1)
                ax.set_title(ti, fontsize=9)
                ax.axis("off")
            axes[row, -1].set_title(f"预测帧\n样本MSE={smse:.4f}", fontsize=9)
        fig.suptitle(f"最优组合：输入 {n_in} 帧 | 真实帧 | 预测帧 对比（3 个样本）")
        fig.tight_layout()
        fig.savefig(a.pred_png, dpi=150)
        plt.close(fig)
        print(f"预测对比图已保存: {a.pred_png}")
