# -*- coding: utf-8 -*-
"""
自制语言模型的知识蒸馏—— 实验作业八参考框架（纯CPU 可运行）
对应教材第15 章《模型量化与压缩技术》+ 第5 章《Transformer 模型》

单文件、零外部数据：
  * 内置「仿古诗句」合成语料生成器（可复现，不依赖网络与版权）
  * 字符级GPT（教师 / 学生同架构、不同宽度）
  * 三种训练：教师监督训练、学生从零训练（hard label）、学生蒸馏训练（soft label）
  * 温度 T / 权重alpha 消融、中间层特征蒸馏、动态量化导出

常用命令：
  python distill_lm.py --mode quick     # 30 秒自检，确认环境没问题
  python distill_lm.py --mode teacher   # 只训练教师并保存teacher.pt（约5 分钟）
  python distill_lm.py --mode all       # 教师 -> 从零学生-> 蒸馏学生-> 画图（约2 分钟）
  python distill_lm.py --mode ablate    # 温度/权重4x4 消融（约6 分钟）
  python distill_lm.py --mode all --feat 0.1                    # 加特征蒸馏
"""

# ============================================================
# 【本人修改1】新增 --tag 参数
#   原因：拓展任务 A（FitNet 特征蒸馏）需要用 --feat 0 与 --feat 0.1 各跑一次，
#   而原代码 save() 用 tag=args.mode 命名，两次都会写 results_all.json，
#   后一次会覆盖前一次。加 --tag 后缀后两组结果可以并存、直接对比。
#   除这一处外，其余代码与实验指南附录逐行一致。
# ============================================================

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_distill")


# ============================================================
# 1. 语料：可复现的合成「仿古诗句」
#    四个槽位+ 两个收尾模板，使下一个字的分布具有真实的多样性与结构，
#    这正是教师soft label 能传给学生的「暗知识」。
# ============================================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]

POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",]


def build_corpus(n_lines=3000, seed=0):
    """用固定随机种子生成训练语料；留出集换一个种子，检验泛化而非记忆。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s 在%s%s 了一夜%s，%s。" % (s, p, o, rng.choice(["看", "听", "守", "忆"]),m))
        else:
            out.append("%s%s 道：“%s%s。”%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
    out.append("\n")
    out.extend(POEMS)
    text = "".join(out)
    return text


# ============================================================
# 2. 字符级分词
# ============================================================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ============================================================
# 3. 最小 GPT（Pre-Norm，与作业三同构，只改宽度/深度）
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ============================================================
# 4. 批采样：scratch 与kd 使用同一generator 种子，保证数据顺序一致
# ============================================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters
@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    """学生与教师在留出集上的top-1 一致率，衡量「学到了多少」"""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    student.train()
    teacher.train()
    return hit / tot


# ============================================================
# 5. 训练：hard label / soft label（KD）
# ============================================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0,
          teacher=None, T=4.0, alpha=0.5, feat_weight=0.0,
          log_every=50, tag=""):
    """
    alpha = 0   -> 纯交叉熵（从零训练的对照组）
    alpha > 0   -> L = (1-alpha)*CE + alpha*T^2*KL(教师软标签|| 学生)
    feat_weight -> 额外加最后一层隐状态MSE（FitNet 式特征蒸馏）
    """
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = []
    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss = ce
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
            # 关键：先reshape 成 (B*T, V) 再算 KL。
            # 若直接对(B,T,V) 调用 kl_div(reduction="batchmean")，它只除以B，
            # 序列维会被重复累加，KL 被放大T 倍（T=96 时数值大得离谱）。
            V = logits.size(-1)
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
            loss = (1 - alpha) * ce + alpha * kd
        if proj is not None:
            _, _, hs = model(x, need_hidden=True)
            with torch.no_grad():
                _, _, ht = teacher(x, need_hidden=True)
            loss = loss + feat_weight * F.mse_loss(proj(hs), ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if teacher is not None and alpha > 0:
                extra = "  ce %.4f  kd(T^2) %.4f" % (ce.item(), kd.item())
            print("  [%s] iter %4d/%d  loss %.4f%s  (%.1fs)"
                  % (tag or "train", it + 1, iters, loss.item(), extra, time.time() - t0))
    return hist, time.time() - t0


# ============================================================
# 6. 主流程
# ============================================================
TEACHER_CFG = dict(n_layer=2, n_head=4, n_embd=160, block_size=96, dropout=0.0)
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embd=64, block_size=96, dropout=0.0)


def make_data(block):
    train_text = build_corpus(3000, seed=0)
    val_text = build_corpus(300, seed=999)          # 同语法、不同种子-> 留出集
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    print("[data] vocab=%d  train_chars=%d  val_chars=%d  block=%d  batch=%d"
          % (tok.vocab_size, len(train_ids), len(val_ids), block, bs))

    teacher_path = os.path.join(OUT_DIR, "teacher.pt")
    res = {"vocab": tok.vocab_size, "block": block, "batch": bs,
           "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids))}

    # ---------- (1) 教师：已训练过就直接复用，避免重复等待----------
    torch.manual_seed(args.seed)
    teacher = GPT(tok.vocab_size, TEACHER_CFG)
    print("[teacher] params = %.3f M" % (teacher.n_param / 1e6))
    if os.path.exists(teacher_path) and not args.retrain and args.mode != "teacher":
        teacher.load_state_dict(torch.load(teacher_path))
        tv = evaluate(teacher, val_ids, bs, block)
        res["teacher"] = {"params": teacher.n_param, "val_loss": tv,
                          "ppl": math.exp(tv), "time_s": 0.0, "curve": [],
                          "cached": True}
        print("[teacher] 复用 %s  val_loss=%.4f  ppl=%.2f"
              % (os.path.basename(teacher_path), tv, math.exp(tv)))
    else:
        th, ttime = train(teacher, train_ids, args.teacher_iters, bs, block,
                          lr=args.lr, seed=args.seed, log_every=args.log_every, tag="teacher")
        tv = evaluate(teacher, val_ids, bs, block)
        torch.save(teacher.state_dict(), teacher_path)
        res["teacher"] = {"params": teacher.n_param, "val_loss": tv,
                          "ppl": math.exp(tv), "time_s": ttime, "curve": th}
        print("[teacher] val_loss=%.4f  ppl=%.2f  %.1fs" % (tv, math.exp(tv), ttime))

    # 【本人修改1】结果文件加后缀，避免拓展实验（--feat 0.1）覆盖主实验结果
    tag_suffix = ("_" + args.tag) if args.tag else ""

    if args.mode == "teacher":
        save(res, teacher, tok, tag="teacher" + tag_suffix)
        return
    if args.mode == "ablate":
        ablate(args, tok, train_ids, teacher, val_ids, bs, block, res)
        save(res, teacher, tok, tag="ablate" + tag_suffix)
        print("\n[done] 结果已写入%s" % OUT_DIR)
        return

    # ---------- (2) 学生配置----------
    torch.manual_seed(args.seed + 7)
    probe = GPT(tok.vocab_size, STUDENT_CFG)
    sp = probe.n_param
    print("[student] params = %.3f M  (压缩率%.1fx)" % (sp / 1e6, teacher.n_param / sp))

    def new_student():
        torch.manual_seed(args.seed + 7)      # 同初始化，保证公平对比
        return GPT(tok.vocab_size, STUDENT_CFG)

    def eval_student(m, hist, tcost, name, **extra):
        v = evaluate(m, val_ids, bs, block)
        row = {"params": sp, "val_loss": v, "ppl": math.exp(v), "time_s": tcost,
               "curve": hist, "agree": agreement(m, teacher, val_ids, bs, block)}
        row.update(extra)
        print("[%s] val_loss=%.4f  ppl=%.2f  agree=%.3f  %.1fs"
              % (name, v, math.exp(v), row["agree"], tcost))
        res[name] = row
        return row

    # ---------- (3) 从零训练（hard label 对照组）----------
    s0 = new_student()
    h0, t0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, log_every=args.log_every, tag="scratch")
    row0 = eval_student(s0, h0, t0, "scratch", T=None, alpha=0.0)

    # ---------- (4) 蒸馏训练（soft label）----------
    s1 = new_student()
    h1, t1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                   feat_weight=args.feat, log_every=args.log_every, tag="kd")
    row1 = eval_student(s1, h1, t1, "kd", T=args.T, alpha=args.alpha, feat=args.feat)

    print("\n[对比] 从零%.4f  ->  蒸馏 %.4f   相对下降%.1f%%"
          % (row0["val_loss"], row1["val_loss"],
             100 * (row0["val_loss"] - row1["val_loss"]) / row0["val_loss"]))

    # ---------- (5) 生成样例----------
    prompt = "月"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        g = s1.generate(ids, 60, temperature=0.9, top_k=20)
    sample = tok.decode(g[0].tolist())
    res["sample"] = sample
    print("[generate] " + sample.replace("\n", " "))

    # ---------- (6) 画图 ----------
    plot_curves(res, os.path.join(OUT_DIR, "curves.png"))

    save(res, s1, tok, tag=args.mode + tag_suffix)
    print("\n[done] 结果与曲线已写入%s" % OUT_DIR)
def ablate(args, tok, train_ids, teacher, val_ids, bs, block, res):
    """T x alpha 网格扫描。
    教师前向只算一次并缓存32 个批次，之后每组只训练学生，
    相比每组重新跑教师前向可省掉约一半时间。"""
    grid_T = [1, 2, 4, 8]
    grid_a = [0.3, 0.5, 0.7, 0.9]
    V = tok.vocab_size
    gen = torch.Generator().manual_seed(2024)
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))

    rows = []
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
            m.train()
            for it in range(args.abl_iters):
                x, y, tl = cache[it % len(cache)]
                logits = m(x)[0]
                ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
                s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
                t_p = F.softmax(tl.reshape(-1, V) / T, dim=-1)
                kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
                loss = (1 - a) * ce + a * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = evaluate(m, val_ids, bs, block)
            print("  T=%d alpha=%.1f -> val %.4f" % (T, a, v))
            rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v)})
    res["ablation"] = rows
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    print("[ablate] 最优 T=%d alpha=%.1f val=%.4f ｜最差 T=%d alpha=%.1f val=%.4f"
          % (best["T"], best["alpha"], best["val_loss"],
             worst["T"], worst["alpha"], worst["val_loss"]))
    plot_curves(res, os.path.join(OUT_DIR, "curves.png"))


def plot_curves(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] 跳过绘图（%s）" % e)
        return
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, lab in (("teacher", "Teacher"), ("scratch", "Student-scratch"), ("kd", "Student-KD")):
        if k in res:
            c = res[k]["curve"]
            ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.2)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("training loss")
    ax[0].set_title("training loss")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if "ablation" in res:
        for T in sorted({r["T"] for r in res["ablation"]}):
            xs, ys = [], []
            for r in res["ablation"]:
                if r["T"] == T:
                    xs.append(r["alpha"])
                    ys.append(r["val_loss"])
            ax[1].plot(xs, ys, marker="o", label="T=%d" % T, linewidth=1.2)
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("val loss")
        ax[1].set_title("temperature / weight ablation")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    else:
        ax[1].bar(["scratch", "kd"], [res["scratch"]["val_loss"], res["kd"]["val_loss"]])
        ax[1].set_title("val loss")
        ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("[plot] saved %s" % path)


def save(res, student, tok, tag="run"):
    with open(os.path.join(OUT_DIR, "results_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    torch.save(student.state_dict(), os.path.join(OUT_DIR, "student_%s.pt" % tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["quick", "teacher", "all", "ablate"])
    ap.add_argument("--teacher-iters", type=int, default=200)
    ap.add_argument("--student-iters", type=int, default=120)
    ap.add_argument("--abl-iters", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--feat", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true", help="忽略已缓存的teacher.pt，重新训练教师")
    ap.add_argument("--log-every", type=int, default=50)
    # 【本人修改1】拓展实验 A 需要分别保留 --feat 0 与 --feat 0.1 的结果，加后缀区分
    ap.add_argument("--tag", default="", help="结果文件名后缀（拓展实验用，避免覆盖主结果）")
    args = ap.parse_args()
    if args.mode == "quick":
        global OUT_DIR
        OUT_DIR = os.path.join(OUT_DIR, "quick")
        args.teacher_iters, args.student_iters, args.abl_iters = 30, 20, 10
        args.batch_size, args.log_every = 8, 10
        args.mode = "all"
    run(args)


if __name__ == "__main__":
    main()
