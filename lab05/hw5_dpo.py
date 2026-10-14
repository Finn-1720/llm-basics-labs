# -*- coding: utf-8 -*-
"""
SmolVLM-256M QLoRA 微调 + DPO 偏好对齐 —— 实验作业五参考框架
对应教材「参数高效微调」与「偏好对齐/ DPO」相关章节

单文件、开箱即跑：内置 PIL 合成「几何图形」图像 15 张、问答 15 题、
SFT 样本 15 条、偏好对 12 组。不配 GPU 也能完成基线推理与小规模演示
（CPU 上训练极慢，仅建议各跑 5~10 步体验流程；正式训练用 GPU）。

常用命令：
    python hw5_dpo.py --mode make_data   # 生成 15 张合成图像（首次运行自动执行）
    python hw5_dpo.py --mode infer       # 基线推理 15 题，记录基线正确率
    python hw5_dpo.py --mode sft         # QLoRA SFT（产出 ./sft_adapter）
    python hw5_dpo.py --mode dpo         # 在 SFT 之上做 DPO（产出 ./dpo_adapter）
    python hw5_dpo.py --mode eval        # 三阶段（基线/SFT/SFT+DPO）× 15 题对比
    python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1   # DPO loss 手算复核

================================================================================
【本人修改】5 处 + 【本人增补】3 处（其余逻辑与指南参考代码逐字一致）

修改 5 —— 去掉量化模型上不合法的 `.float()`
    指南 `--mode dpo` 分支写的是 `model.get_base_model().float().to("cpu")`。
    transformers v5 起，对已量化模型调用 `.float()` 会直接抛
    ValueError（"`.float()` is not supported for quantized model"）。
    4-bit 基座本身就是训练所需形态，去掉 `.float()` 即可，量化状态不变。

修改 4 —— 修正指南参考代码中 DPO 手算复核函数的符号错误
    指南第 9 页 `dpo_check()` 写的是
        loss = -math.log(1 + math.exp(-margin)) if margin >= 0
               else margin - math.log(1 + math.exp(margin))
    由 σ(m)=1/(1+e^-m) 得 -log σ(m) = log(1+e^-m)，指南多了一个负号，
    会算出**负的损失**（margin=0.01 时得 -0.6882，正确值 +0.6882）。
    已改为 `log(1+e^-margin)` / `-margin + log(1+e^margin)`，并与独立验算核对。
    **注意：训练路径 `dpo_step()` 里的 `-F.logsigmoid(margin)` 是正确的，
    指南只有这个手算复核辅助函数写反了。**

修改 1 —— 适配 transformers v5 的模型类改名
    本机环境是 transformers 5.17.0，v5 已**彻底移除** `AutoModelForVision2Seq`
    （不是弃用，是删除；全包 grep 零命中）。SmolVLM 的 Auto 映射现在指向
    `AutoModelForImageTextToText` → `SmolVLMForConditionalGeneration`。
    若沿用旧类名会直接 AttributeError。

修改 2 —— 适配 transformers v5 的 dtype 参数改名
    `torch_dtype=` 在 v5 中已弃用（`modeling_utils.py` 里有 warning_once），
    改为 `dtype=`。另外 v5 的默认 dtype 从 float32 改为 "auto"，所以显式
    指定 dtype 比 v4 时代更重要。

修改 3 —— 支持从本地目录加载模型
    本机无法访问 huggingface.co（直连和代理都不通），模型改从 ModelScope
    下载到本地。`resolve_model_path()` 会在本地目录存在时优先使用它，
    避免加载阶段联网。设 LOCAL_MODEL_DIR 环境变量可覆盖路径。

增补 1（`--r` 参数）—— 让 LoRA 秩可从命令行调节，用于任务五(b) 的秩敏感性
    扫描（r=4/8/16）。默认仍为指南的 r=8，不改变默认行为。

增补 2（`dpo_log.csv`）—— `dpo_step()` 逐步记录 loss 与 margin 并落盘，
    供报告画训练曲线。指南只打印到 stdout，没有留数据文件；
    训练逻辑本身一行未改，只是多存了一份中间量。

增补 3（`resolve_model_path()`）—— 见修改 3。
================================================================================
"""
import argparse, math, os, random
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# 【本人修改 3】本地模型目录：存在则优先使用，避免加载时联网
#   模型从 ModelScope 下载（本机无法访问 huggingface.co）。
#   按顺序在下列候选路径中查找，找到含 config.json 的目录就用它；
#   也可用环境变量 LOCAL_MODEL_DIR 显式指定。
_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_CANDIDATES = [
    os.environ.get("LOCAL_MODEL_DIR", ""),                       # 显式指定优先
    os.path.join(_HERE, "smolvlm_model"),                        # 脚本同目录
    os.path.join(_HERE, "..", "models", "SmolVLM-256M-Instruct"),  # llm-lab/models/
    r"D:\claude\llm-lab\models\SmolVLM-256M-Instruct",           # 本机实际位置
]


def resolve_model_path():
    """本地存在完整模型就用本地，否则回退到 HF 仓库名（需联网）"""
    for cand in LOCAL_MODEL_CANDIDATES:
        if not cand:
            continue
        if os.path.exists(os.path.join(cand, "config.json")) and \
           os.path.exists(os.path.join(cand, "model.safetensors")):
            return os.path.abspath(cand)
    print(f"[warn] 未找到本地模型，将尝试联网加载 {MODEL_ID}")
    return MODEL_ID


IMG_DIR = "hw5_imgs"
SFT_DIR, DPO_DIR = "sft_adapter", "dpo_adapter"

# 【本人增补 1】LoRA 秩，由 --r 设置（默认 8，与指南一致）
LORA_R = 8

# 【本人增补 3】是否把自拟 5 组偏好对并入训练集，由 --use-self-pairs 设置
USE_SELF_PAIRS = False

# ============================================================
# 1. 合成图像库「几何图形」：15 张图，三类问题的答案均已埋入
#    每张图 = 白底 256x256，画若干「颜色-形状」对（位置固定可复现）
# ============================================================
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
# img_id -> [(颜色, 形状, 个数), ...]（形状：圆=ellipse，方=rectangle）
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],                                        # 判断题素材：只有一个黄方
    "img12": [("红", "方", 1), ("蓝", "方", 1)],                       # 没有圆形→「有没有红圆」答「没有」
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}


def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    rnd = random.Random(42)
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                s = rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                if shape == "圆":
                    d.ellipse(box, outline=COLORS[color], width=4)
                else:
                    d.rectangle(box, outline=COLORS[color], width=4)
        im.save(f"{IMG_DIR}/{img_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像到 {IMG_DIR}/")


def load_image(img_id):
    p = f"{IMG_DIR}/{img_id}.png"
    if not os.path.exists(p):
        make_images()
    return Image.open(p).convert("RGB")


# ============================================================
# 2. 问题集：3 类 × 5 题（gold = 标准答案；与参考答案逐字匹配计分）
#    count=数数（局部事实） attr=属性识别 exist=存在性/组合判断
# ============================================================
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]

# SFT 数据：图像 + 问题 → 简短标准答案（教模型「答题格式 + 事实」）
SFT_DATA = [
    {"img": q["img"], "prompt": q["q"],
     "answer": f"{q['gold']}。" if q["type"] != "count" else f"{q['gold']}个。"}
    for q in QUESTIONS
]

# 偏好对（12 组）：chosen = 正确回答；rejected = 典型幻觉/错误
#   rejected 的三类错误：数错（off-by-one / 混淆）、形状幻觉、对不存在的物体硬答「有」
PRE_PAIRS = [
    {"img": "img01", "prompt": "图中有几个红色圆形？",
     "chosen": "2 个。", "rejected": "3 个。图中有三个红色圆形整齐排列。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3 个。", "rejected": "2 个。"},
    {"img": "img03", "prompt": "图中有几个黄色方形？",
     "chosen": "2 个。", "rejected": "没有看到黄色图形。"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4 个。", "rejected": "3 个。"},
    {"img": "img10", "prompt": "图中有几个蓝色圆形？",
     "chosen": "3 个。", "rejected": "图中有三个蓝色方形。"},
    {"img": "img01", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。", "rejected": "圆形。"},
    {"img": "img05", "prompt": "图中红色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形，而且有两个红色方形。"},
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？",
     "chosen": "方形。", "rejected": "图中没有绿色图形。"},
    {"img": "img11", "prompt": "图里有没有黄色方形？",
     "chosen": "有。", "rejected": "没有。"},
    {"img": "img12", "prompt": "图里有没有红色圆形？",
     "chosen": "没有。", "rejected": "有，图左上角有一个红色圆形。"},
    {"img": "img08", "prompt": "图里有没有绿色方形？",
     "chosen": "没有。", "rejected": "有，一共两个绿色方形。"},
    {"img": "img15", "prompt": "图里有没有蓝色图形？",
     "chosen": "没有。图中只有黄色和绿色的圆形。", "rejected": "有，两个蓝色圆形。"},
]

# 【本人增补 3】自拟 5 组偏好对（任务三）
#   指南原有的 12 组 rejected 覆盖三类：数错 / 形状幻觉 / 硬答「有」。
#   这 5 组补的是**本机实测中真实出现、但原设计没覆盖**的失败模式，
#   每条 rejected 都取自我们跑出来的真实输出（见报告 4.1/4.3/6.1）。
#   用 --use-self-pairs 开启，可与原 12 组做对照。
SELF_PAIRS = [
    # 1) 枚举而非计数：SFT 后 P04 实际输出「1. 2. 3. 4. …」
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4个。", "rejected": "1. 2. 3. 4. 5. 6. 7. 8."},
    # 2) 颜色幻觉：基线 P07 实际输出「形状是白色的。」
    {"img": "img05", "prompt": "图中红色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "形状是白色的。"},
    # 3) 语言漂移（中问英答）：SFT 后 P09 实际输出「Green.」
    {"img": "img13", "prompt": "图中黄色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "Green."},
    # 4) 无关词/截断：SFT 后 P08 实际输出「半身。」
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？",
     "chosen": "方形。", "rejected": "半身。"},
    # 5) 冗长废话型（指南建议）：描述了一大段却始终不给数字
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3个。",
     "rejected": "这是一张包含几何图形的图片。图中分布着若干圆形，"
                 "它们位于画面不同的位置，大小也不完全相同。综合考虑颜色"
                 "和形状，需要仔细辨认才能确定数量。"},
]


# ============================================================
# 3. 模型加载：QLoRA（教材：NF4 量化 + LoRA 低秩适配器）
#    256M 小模型按理可全参微调，但本实验统一走 QLoRA 流程以体会方法
# ============================================================
def load_model(mode="infer"):
    """mode: infer=原精度推理 / train=4bit 量化+LoRA"""
    # 【本人修改 1】AutoModelForVision2Seq → AutoModelForImageTextToText
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    model_path = resolve_model_path()          # 【本人修改 3】
    processor = AutoProcessor.from_pretrained(model_path)
    # 【本人修改 2】torch_dtype= → dtype=
    kwargs = {"dtype": torch.float32}
    if mode == "train":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32)
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)   # 【本人修改 1】
    if mode == "train":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        # 【本人增补 1】r 由全局 LORA_R 控制（默认 8，与指南一致），
        #   lora_alpha 按社区惯例取 2r，保证不同 r 下缩放一致，便于任务五(b) 对比。
        lora = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()     # ← 报告 4.2 节要抄的可训练参数占比
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def build_prompt(processor, q):
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


@torch.no_grad()
def answer(processor, model, img, q, max_new=24):
    inputs = processor(text=[build_prompt(processor, q)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text.strip()


def judge(q, pred):
    """宽松判分：标准答案出现在预测里即算对（「没有」须整词出现）"""
    p, g = pred.replace(" ", ""), q["gold"]
    if g == "没有":
        return "没有" in p and "有" in p
    return g in p


# ============================================================
# 4. SFT（教材：监督微调）：只对答案 token 计损失，全流程手写便于理解
# ============================================================
def sft_step(model, processor, opt, epochs=2, steps=60, log_every=10):
    dev = next(model.parameters()).device
    model.train()
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(SFT_DATA)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[sft] 已达 {steps} 步，提前结束")
                return
            s = SFT_DATA[idx]
            full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                             images=[load_image(s["img"])],
                             return_tensors="pt").to(dev)
            with torch.no_grad():
                prompt_len = processor(text=[build_prompt(processor, s["prompt"])],
                                       images=[load_image(s["img"])],
                                       return_tensors="pt")["input_ids"].shape[1]
            labels = full["input_ids"].clone()
            labels[:, :prompt_len] = -100          # 只学答案
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (i + 1) % log_every == 0:
                print(f"[sft] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f}")
    model.save_pretrained(SFT_DIR)
    print(f"[sft] 适配器已保存到 ./{SFT_DIR}")


# ============================================================
# 5. DPO（教材：直接偏好优化）
#    loss = -log σ( β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] )
#    参考模型 = 关闭 LoRA 适配器后的基座（disable_adapter 上下文）
# ============================================================
def seq_logprob(model, processor, img, prompt, resp):
    """返回「答案 token 的平均对数概率 × 长度」＝序列总对数概率"""
    dev = next(model.parameters()).device
    full = processor(text=[build_prompt(processor, prompt) + resp],
                     images=[img], return_tensors="pt").to(dev)
    plen = processor(text=[build_prompt(processor, prompt)], images=[img],
                     return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    logits = model(**full).logits[:, :-1, :]
    tgt = labels[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    mask = tgt.ne(-100)
    return lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].sum()


def dpo_step(model, processor, opt, beta=0.1, epochs=1, steps=40, log_every=5):
    dev = next(model.parameters()).device
    model.train()
    log_rows = []
    # 【本人增补 3】--use-self-pairs 时把自拟 5 组并入偏好集
    pairs = PRE_PAIRS + SELF_PAIRS if USE_SELF_PAIRS else PRE_PAIRS
    print(f"[dpo] 偏好对数量 = {len(pairs)}（原 12 组{' + 自拟 5 组' if USE_SELF_PAIRS else ''}）")
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(pairs)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if ep * len(order) + i >= steps:
                print(f"[dpo] 已达 {steps} 步，提前结束")
                return log_rows
            p = pairs[idx]
            img = load_image(p["img"])
            pol_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
            pol_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            with model.disable_adapter():          # 冻结的参考模型 π_ref
                ref_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
                ref_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(margin)
            loss.backward()
            opt.step()
            opt.zero_grad()
            # 【本人增补】记录逐步 loss/margin，供报告画训练曲线（不改变训练逻辑）
            log_rows.append((ep * len(order) + i + 1, loss.item(), margin.item()))
            if (i + 1) % log_every == 0:
                print(f"[dpo] ep{ep+1} step{i+1}/{len(order)} loss={loss.item():.4f} "
                      f"margin={margin.item():.3f}")
    model.save_pretrained(DPO_DIR)
    print(f"[dpo] 适配器已保存到 ./{DPO_DIR}")
    return log_rows


def dpo_check(arg):
    """手算复核：依次给 pol_w, ref_w, pol_l, ref_l, beta"""
    pw, rw, pl, rl, beta = [float(x) for x in arg.split(",")]
    margin = beta * ((pw - rw) - (pl - rl))
    # 【本人修改 4】修正指南参考代码中的符号错误
    #   指南原文：loss = -math.log(1 + math.exp(-margin)) if margin >= 0
    #              else margin - math.log(1 + math.exp(margin))
    #   由 σ(m) = 1/(1+e^-m) 可得 -log σ(m) = log(1 + e^-m)，
    #   指南多写了一个负号，会算出**负的损失**（如 margin=0.01 时得 -0.6882，
    #   而正确值是 +0.6884）。注意：训练路径 dpo_step() 里的
    #   `loss = -F.logsigmoid(margin)` 是**正确**的，只有这个手算复核函数写反了。
    loss = (math.log(1 + math.exp(-margin)) if margin >= 0
            else -margin + math.log(1 + math.exp(margin)))
    print(f"margin = β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] = {margin:.4f}")
    print(f"DPO loss = -log σ(margin) = {loss:.4f}"
          f"（margin 越大 loss 越小；margin<0 说明模型更偏好 rejected，需要继续训练）")


# ============================================================
# 6. 三阶段对比评测：基线 / SFT / SFT+DPO × 15 题
# ============================================================
def evaluate(processor, model, tag):
    print(f"\n===== {tag} × 15 题 =====")
    ok_by_type = {}
    for tp in ("count", "attr", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        ok = sum(judge(q, answer(processor, model, load_image(q["img"]), q["q"])) for q in qs)
        ok_by_type[tp] = (ok, len(qs))
        print(f"{tag:<10}{tp:<8}正确 {ok}/{len(qs)}")
    total = sum(o for o, _ in ok_by_type.values())
    print(f"{tag:<10}总计    正确 {total}/15\n")
    return total


# ============================================================
# 7. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业五：SmolVLM-256M QLoRA + DPO")
    ap.add_argument("--mode", choices=["make_data", "infer", "sft", "dpo", "eval"],
                    default="infer")
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认训练步数（CPU 演示用 5~10）")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 温度系数 β")
    ap.add_argument("--dpo_check", default=None, help="如 1.2,0.8,0.9,0.6,0.1")
    # 【本人增补】r 可调，用于任务五(b) 的秩敏感性扫描
    ap.add_argument("--r", type=int, default=8, help="LoRA 秩 r（默认 8）")
    # 【本人增补 3】把自拟 5 组偏好对并入 DPO 训练集
    ap.add_argument("--use-self-pairs", action="store_true",
                    help="DPO 训练时加入自拟的 5 组偏好对（任务三/步骤6）")
    args = ap.parse_args()

    # 【本人增补 1】把 --r 接到 LoRA 配置上（默认 8，不传则行为与指南一致）
    global LORA_R, USE_SELF_PAIRS
    LORA_R = args.r
    USE_SELF_PAIRS = args.use_self_pairs

    if args.dpo_check:
        dpo_check(args.dpo_check)
        return
    if args.mode == "make_data":
        make_images()
        return
    if args.mode == "infer":
        processor, model = load_model("infer")
        for q in QUESTIONS:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            mark = "√" if judge(q, pred) else "×"
            print(f"[{q['id']}·{q['type']}] {mark} Q: {q['q']}  A: {pred}（gold: {q['gold']}）")
        return
    if args.mode == "sft":
        processor, model = load_model("train")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sft_step(model, processor, opt, steps=args.steps or 60)
        return
    if args.mode == "dpo":
        if not os.path.exists(SFT_DIR):
            raise SystemExit("先运行 --mode sft 生成 SFT 适配器")
        processor, model = load_model("train")
        from peft import PeftModel
        # 【本人修改 5】去掉 .float()
        #   指南原文：model.get_base_model().float().to("cpu")
        #   transformers v5 起，对已量化的模型调用 .float() 会直接抛
        #   ValueError（"`.float()` is not supported for quantized model"）。
        #   4-bit 基座本身就是训练所需的形态，无需也不允许再转 float32。
        model = PeftModel.from_pretrained(
            model.get_base_model().to("cpu"), SFT_DIR, is_trainable=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        rows = dpo_step(model, processor, opt, beta=args.beta, steps=args.steps or 40)
        # 【本人增补】把逐步 loss/margin 落盘，供报告画曲线
        if rows:
            with open("dpo_log.csv", "w", encoding="utf-8") as f:
                f.write("step,loss,margin\n")
                for st, ls, mg in rows:
                    f.write(f"{st},{ls:.6f},{mg:.6f}\n")
            print(f"[dpo] 逐步 loss/margin 已写入 dpo_log.csv（{len(rows)} 步）")
        return
    if args.mode == "eval":
        processor, model = load_model("infer")
        evaluate(processor, model, "基线")
        if os.path.exists(SFT_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, SFT_DIR).to(model.device)
            evaluate(processor, model, "SFT")
            if os.path.exists(DPO_DIR):
                model = PeftModel.from_pretrained(model, DPO_DIR)
                evaluate(processor, model, "SFT+DPO")
        print("提示：把三阶段结果抄进报告表 4；另抽 2 题做人工忠实度评分（答案是否仅陈述图中事实）。")


if __name__ == "__main__":
    main()
