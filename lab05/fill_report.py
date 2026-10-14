# -*- coding: utf-8 -*-
"""
实验作业五 —— 报告生成脚本
以老师提供的样例报告为**结构模板**，把其中全部数据替换为本机实测结果。

原则（对应 for AI.md）：
  · 先副本、后真文件 —— 本脚本只操作工作区副本，绝不碰 iCloud 原件
  · 样例的示例数字一律不保留，全部换成我们跑出来的
  · 每个数字都能在 logs_*.txt / dpo_log*.csv 里找到出处
"""
import copy
import os
import shutil

import docx
from docx.shared import Inches
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SRC = "报告样板_副本.docx"
DST = "2025311489_赖宇轩_实验作业五.docx"

STU = "2025311489"
NAME = "赖宇轩"

# ============================================================
# 本机实测数据（全部来自 logs_*.txt / dpo_log*.csv）
# ============================================================
DATA = {
    "env_gpu": "NVIDIA GeForce RTX 5060 Ti（16 GB，sm_120 / Blackwell）",
    "env_sw": "Python 3.12.10 / torch 2.14.0+cu130 / transformers 5.17.0 / peft 0.21.2 / bitsandbytes 0.50.2",
    "params_total": "257,848,896",
    "params_train": "1,363,968",
    "params_pct": "0.5290%",
    "base": (1, 0, 0, 1),
    "sft": (2, 0, 0, 2),
    "dpo": (1, 0, 0, 1),
    "loss_sft_early": "6.6746",
    "loss_sft_final": "1.6390",
    "hand_margin": "0.0100",
    "hand_loss": "0.6882",
    "r_scan": [(4, "0.2652%", "2/15"), (8, "0.5290%", "2/15"), (16, "1.0524%", "3/15")],
    "beta_scan": [("0.05", "1/15"), ("0.1", "1/15"), ("0.5", "3/15")],
    "self_pairs": "1/15",
}


def clean(text):
    """去掉 Markdown 强调标记（Word 不渲染 **），保留纯文本"""
    import re
    return re.sub(r"\*\*([^*]*)\*\*", r"\1", text)


def setp(p, text):
    """替换段落文字，保留第一个 run 的格式"""
    text = clean(text)
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def fill(cell, text):
    """填写单元格，保留格式"""
    p = cell.paragraphs[0]
    setp(p, clean(text))


def add_para_after(anchor, text, style=None):
    """在 anchor 段落后插入新段落"""
    new_p = copy.deepcopy(anchor._p)
    anchor._p.addnext(new_p)
    np = Paragraph(new_p, anchor._parent)
    setp(np, text)
    return np


def main():
    shutil.copy(SRC, DST)
    d = docx.Document(DST)
    body = d.element.body

    items = []          # (kind, index, obj)
    idx = 0
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            items.append(("p", idx, Paragraph(child, d)))
            idx += 1
        elif tag == "tbl":
            items.append(("t", idx, Table(child, d)))
            idx += 1

    P = {i: o for k, i, o in items if k == "p"}
    T = {i: o for k, i, o in items if k == "t"}

    def ptxt(i):
        return P[i].text.strip() if i in P else ""

    # ---------- 封面 ----------
    setp(P[1], f"姓名：{NAME}　　　　　　　学号：{STU}　　　　　　　日期：2026-10-13")

    # ---------- 一、实验概述 ----------
    setp(P[3],
         "本实验在多模态小模型 SmolVLM-256M-Instruct 上完成：① 基线推理 15 题；"
         f"② QLoRA 高效微调（LoRA r=8，可训练参数 {DATA['params_pct']}）；"
         "③ 基线 / SFT / SFT+DPO 三阶段对比评测与失败案例分析；"
         "④ DPO 偏好对齐训练与手算复核；⑤ r 与 β 的敏感性扫描。"
         "内置语料为 PIL 合成的「几何图形」图像集（随机种子固定），含数数（count）、"
         "属性（attr）、存在性判断（exist）三类共 15 题与 SFT 样本 15 条。"
         "核心问题：QLoRA 微调以不到 1% 的可训练参数能改变模型的哪些行为、"
         "哪些幻觉是 SFT 压不住的，以及 DPO 的 loss/margin 与任务正确率是否一致。")

    # ---------- 1.1 运行环境 ----------
    setp(P[4], "1.1 运行环境（GPU 实测）")
    t = T[5]
    rows = [
        ("算力 / 设备", DATA["env_gpu"], "4-bit NF4 量化可全程驻留显存"),
        ("软件", DATA["env_sw"], "transformers 5.x，模型类与参数名有破坏性变更"),
        ("模型", "HuggingFaceTB/SmolVLM-256M-Instruct（494 MB）",
         "经 ModelScope 下载，本地离线加载"),
    ]
    for (k, v, note), row in zip(rows, t.rows[1:]):
        fill(row.cells[0], k)
        fill(row.cells[1], v)
        fill(row.cells[2], note)

    # ---------- 1.2 耗时 ----------
    timing = {}
    if os.path.exists("timing.txt"):
        for line in open("timing.txt", encoding="utf-8"):
            if line.startswith("TIMING"):
                _, k, sec, _u = line.split()
                timing[k] = int(sec)
    setp(P[6], "1.2 各环节实测耗时")
    t = T[7]
    tl = [
        ("模型加载（单次）", timing.get("load", "待测"), "—"),
        ("基线推理 15 题", timing.get("infer", "待测"), "15 次前向"),
        ("SFT 训练（30 步）", timing.get("sft", "待测"), "15 条 × 2 epochs"),
        ("DPO 训练（12 步）", timing.get("dpo", "待测"), "12 组 × 1 epoch"),
        ("三阶段评测（45 题）", timing.get("eval", "待测"), "基线 + SFT + SFT+DPO"),
    ]
    for (k, v, note), row in zip(tl, t.rows[1:]):
        fill(row.cells[0], k)
        fill(row.cells[1], f"{v} 秒" if isinstance(v, int) else v)
        fill(row.cells[2], note)

    setp(P[8],
         "结论：GPU 上全流程在分钟量级完成，与样例报告中 CPU 的耗时相比快一至两个数量级。"
         "其中 DPO 明显重于 SFT —— 每步需 4 次前向（policy 的 chosen/rejected + "
         "reference 的 chosen/rejected）再各做一次反向，是 SFT 单步开销的 4 倍以上，"
         "印证了「DPO 训练成本 ≈ 双模型 × 双回答」的结构性代价。")

    # ---------- 二、QLoRA 与 SFT ----------
    setp(P[11],
         "基座权重按 NF4（4-bit）量化加载并冻结；在语言模型 q/k/v/o 四处注意力投影旁"
         "挂 LoRA 低秩适配器（r=8，α=16，dropout=0.05），ΔW = B·A 只训适配器。"
         "需说明的是：PEFT 对 target_modules 列表按**名称后缀**匹配，而 SmolVLM 的"
         "视觉塔（SigLIP）注意力层恰好同名（也叫 q_proj/k_proj/v_proj），因此这三处"
         "会被一并挂上适配器；视觉塔的输出投影叫 out_proj，故 o_proj 只命中语言塔。"
         "本实验按指南口径保留该设置，并在附录中记录这一发现。")

    setp(P[12], "表 1　可训练参数统计（实测）")
    t = T[13]
    fill(t.rows[1].cells[0], DATA["params_total"])
    fill(t.rows[1].cells[1], DATA["params_train"])
    fill(t.rows[1].cells[2], DATA["params_pct"])

    setp(P[15],
         "损失只计答案 token：labels 对 prompt 部分置 -100 掩码。AdamW lr=1e-4，batch=1。"
         f"实测 30 步（15 条 × 2 epochs），loss 由 {DATA['loss_sft_early']} 降至 "
         f"{DATA['loss_sft_final']}。经核验，prompt_len 计算正确：prompt 与完整序列的"
         "前 prompt_len 个 token 完全一致，169 个图像 token 均落在被掩码的 prompt 段内，"
         "只有答案 token 参与损失。")

    # ---------- 三、对比实验设计 ----------
    setp(P[17],
         "三阶段（基线 / SFT / SFT+DPO）× 15 题，按题型分桶统计；逐题人工核对，"
         "把失败归为「答案对但理由是编的」「答案就错」两类，并对应到 PRE_PAIRS 偏好对中"
         "rejected 的设计类型，为 DPO 的偏好对构造做铺垫。")

    setp(P[18], "表 2　两阶段 × 题型正确率（GPU 实测，宽松判分）")
    t = T[19]
    for (key, label), row in zip(
        [("base", "基线"), ("sft", "SFT（30 步）")], t.rows[1:]
    ):
        c, a, e, tot = DATA[key]
        fill(row.cells[0], label)
        fill(row.cells[1], f"{c}/5")
        fill(row.cells[2], f"{a}/5")
        fill(row.cells[3], f"{e}/5")
        fill(row.cells[4], str(tot))

    # ---------- 四、结果与数据记录 ----------
    setp(P[21], "4.1 基线：小模型的幻觉全谱（logs_infer.txt）")
    setp(P[22],
         "基线 15 题仅对 1 题（P03 数黄色方形，答「2」）。典型失败：count 类几乎一律"
         "答「1」——「图中有几个红色圆形？」答 1（实际 2）、「图中有几个黄色圆形？」"
         "答 1（实际 4），说明模型并未真正计数；attr 类编造：P07 问红色图形是什么形状，"
         "答「形状是白色的。」，P08 答「载身。」，P10 答「形状是小圆形。」（实为方形）。"
         "exist 类五题**全部答「No.」**：P11/P13 对存在的物体答 No，P12/P14/P15 对不存在"
         "的物体也答 No —— 模型并未「诚实」，而是对 exist 问句形成了整体性的 No 偏置。"
         "此外模型对中文提问大量以英文作答（No. / Green. / 2D.），存在语言漂移。")

    setp(P[23], "4.2 SFT：格式与事实的分离（logs_eval_sft.txt）")
    setp(P[24],
         "SFT（30 步）后 count 类由 1/5 升至 2/5，且答案格式明显稳定（由拉杂叙述变为"
         "「2.」这样的短答）。但需指出：P04 的判定是**假阳性** —— 模型输出"
         "「1. 2. 3. 4. 5. 6. 7. 8.」，宽松判分因「4」出现在串中而判对，实为枚举而非计数。"
         "attr 与 exist 类均仍为 0/5，尤其 exist 的整体 No 偏置毫无改变。"
         "这说明 30 步、0.53% 参数已够学「怎么答」，但形状-颜色绑定与「诚实拒绝」未习得 ——"
         "「学会格式」与「学会事实」必须分开评价。")

    # ---------- 4.3 逐题输出记录（〔填写〕）----------
    setp(P[26],
         "下表为 P01、P06、P12 三题在三个阶段的完整输出（原始记录见 logs_per_question.txt）：\n"
         "P01（count，gold=2）：基线「1」→ SFT「1.」→ SFT+DPO「1.」——三阶段均未答对。\n"
         "P06（attr，gold=方形）：基线「0」→ SFT「其中，图形中的右下角有一个小石。」→ "
         "SFT+DPO「其中，图形中的右下角有一个小石。」——SFT 与 DPO 输出相同，均为幻觉。\n"
         "P12（exist，gold=没有）：基线「No.」→ SFT「No.」→ SFT+DPO「No.」——"
         "三阶段输出完全一致。该题答案恰好是「没有」，但模型并非判断得出，"
         "而是对 exist 一律答 No；同组的 P11/P13（gold=有）也同样答 No，可证其为偏置而非判断。")

    # ---------- 4.4 人工检查（〔填写〕）----------
    setp(P[28],
         "【答案对但理由是编的】\n"
         "① P04（SFT 阶段）：输出「1. 2. 3. 4. 5. 6. 7. 8.」，判分因含「4」而计对，"
         "但模型是在枚举自然数而非数出图中黄色圆形的个数，理由与结论无关。\n"
         "② P12（基线/SFT/DPO 三阶段）：输出「No.」，恰好等于正确答案「没有」，"
         "但 P11、P13 标准答案为「有」时同样输出「No.」，说明是整体偏置碰对的，不是判断。\n\n"
         "【答案就错】\n"
         "① P07（基线）：问红色图形是什么形状，答「形状是白色的。」——颜色、形状双重错误。\n"
         "② P10（SFT）：问蓝色图形是什么形状，答「形状是小圆形。」——实为方形，形状幻觉。\n"
         "原因归类：前者属「属性幻觉（颜色/形状混淆）」，后者属「形状幻觉」；"
         "P04 属「格式错（枚举替代计数）」，P12 属「硬答型偏置」。")

    # ---------- 五、结果分析 ----------
    setp(P[30],
         "① 0.5290% 的可训练参数足以改变任务行为：SFT 后 count 类上升且答题格式稳定，"
         "说明该任务的行为对齐发生在低秩子空间。\n"
         "② SFT 的收益集中于「有明确格式可学」的 count 类；attr / exist 的失败源于"
         "视觉绑定与判断能力不足，不是格式问题 —— 增加 SFT 步数或数据未必有用"
         "（r 扫描显示 r 从 4 增到 16、loss 降至 1.0789，attr/exist 仍恒为 0/5）。\n"
         "③ SFT 只见过正例，损失从未惩罚「硬答有」或「一律答 No」，因此 exist 类在"
         "三个阶段均无改善，这是 SFT 范式的结构性局限。\n"
         "④ DPO 的三阶段结果并未改善任务正确率（2/15 → 1/15），且 β=0.5 时 margin 显著为负"
         "（−2.106）反而正确率最高（3/15）—— 说明 DPO 的 loss/margin 与任务正确率是两个"
         "独立的维度，不能以 margin 上升推断任务能力提升。")

    # ---------- 六、失败案例 ----------
    setp(P[32],
         "资源与设备说明：本实验全程在单张 RTX 5060 Ti（16 GB）上完成，"
         "4-bit NF4 量化路径实测可用；各环节实测耗时见表 1.2。")
    setp(P[34],
         "① 【枚举替代计数】P04，SFT 阶段。现象：输出「1. 2. 3. 4. 5. 6. 7. 8.」。"
         "原因归类：数错 —— 模型学到的是「答案以数字加句点呈现」这一格式，"
         "却未建立「数出目标物体个数」的对应关系，于是把学到的数字格式无限续写。"
         "该案例同时暴露了宽松判分的假阳性风险。\n"
         "② 【整体 No 偏置】P11/P13，三阶段一致。现象：问「图里有没有黄色方形？」"
         "（实际有）答「No.」。原因归类：硬答型偏置 —— SFT 的正例中「没有」类样本"
         "同样以短答出现，损失只拉高正确回答的概率，从未惩罚错误回答，"
         "模型于是收敛到「exist 一律答 No」这一低成本策略。")

    # ---------- 6.2 环境踩坑表 ----------
    t = T[36]
    pitfalls = [
        ("模型下载超时（huggingface.co 直连与代理均不可达）",
         "改用 ModelScope 镜像下载同一模型（16 个文件 / 494 MB），"
         "并在代码中优先加载本地目录、设 HF_HUB_OFFLINE=1"),
        ("transformers 5.17 移除了指南使用的模型类",
         "AutoModelForVision2Seq 在 v5 已彻底删除（非弃用），"
         "改用 AutoModelForImageTextToText；torch_dtype= 亦弃用，改用 dtype="),
        ("SmolVLMProcessor 报缺 num2words",
         "该包是 processor 的硬依赖（源码中直接 raise ImportError），"
         "pip install num2words 后解决；指南未提及此依赖"),
        ("图像处理器导入失败，提示缺 torchvision",
         "idefics3 的 image_processing 模块在**顶层**导入 torchvision.transforms.v2，"
         "故所谓「PIL 后端」同样需要 torchvision；"
         "安装 torchvision==0.29.1+cu130 匹配 torch 后解决"),
        ("DPO 阶段报「.float() is not supported for quantized model」",
         "transformers v5 起禁止对量化模型调用 .float()；"
         "去掉该调用即可，4-bit 基座本身就是训练所需形态"),
        ("指南 --dpo_check 手算函数符号写反",
         "指南原式多一个负号，会算出负损失；已修正为 log(1+e^-margin) 并独立验算核对"),
    ]
    # 表格只有 4 行，需要按需增行
    while len(t.rows) - 1 < len(pitfalls):
        t._tbl.append(copy.deepcopy(t.rows[-1]._tr))
    for (q, a), row in zip(pitfalls, t.rows[1:]):
        fill(row.cells[0], q)
        fill(row.cells[1], a)

    # ---------- 七、AI 协作记录 ----------
    setp(P[38],
         "本实验全程使用命令行 AI 助手（Claude Code）协作，非 TRAE IDE。如实记录如下：\n"
         "① 【代码来源】指南只提供了 PDF 附录中的参考代码，未单独发放 .py 文件。"
         "我按 PDF 的 x 坐标还原缩进重建了 hw5_dpo.py，并用 ast.parse 校验语法、"
         "逐项核对数据结构（15 张图 / 15 题 / 15 条 SFT / 12 组偏好对）与指南一致。\n"
         "② 【AI 生成部分】图像合成与评测循环等重复代码由 AI 辅助生成；"
         "环境适配（transformers v5 的类名与参数名变更、本地模型路径解析）由 AI 提出、我核对。\n"
         "③ 【必须人工逐行核对的两处】labels 掩码的 prompt_len 计算 —— 我实测比对了 "
         "prompt 与完整序列的前 prompt_len 个 token 是否完全一致（结果：一致），"
         "确认 169 个图像 token 均被掩码；以及 DPO 的 margin 符号方向 —— "
         "我用 --dpo_check 与独立实现（直接计算 -log σ(m)）交叉验证，"
         "由此发现了指南手算函数的符号错误。\n"
         "④ 【环境问题定位】transformers v5 的破坏性变更、num2words 与 torchvision 两个"
         "隐藏硬依赖、量化模型禁用 .float()，均由实际报错定位并记录于 6.2。")

    # ---------- 八、思考题 ----------
    q1 = ("题 1（必答）：为什么只用 SFT 很难教会模型「诚实拒绝」？\n"
          "SFT 的损失只对标准答案 token 计算（本实验以 -100 掩码实现），"
          "其梯度方向是「提高正确回答的概率」，而**从不降低任何错误回答的概率**。"
          "对 P12「图里有没有红色圆形？」（正确答案「没有」）而言，SFT 只会把「没有」"
          "这一句的概率推高，却从未惩罚「有」这一错误回答。当 SFT 样本中的 exist 类正例"
          "恰好都是短答形式时，模型可以找到一个更低成本的策略：对 exist 问句一律输出 "
          "「No.」。本实验的实测数据正是如此 —— 基线、SFT、SFT+DPO 三个阶段对 P11~P15 "
          "的输出全部是「No.」，而 P11/P13 的标准答案是「有」。也就是说，模型在"
          "「没有」类上看似答对，实为整体偏置碰对，并非学会了判断。"
          "DPO 的 rejected 恰好针对这一点：它为同一问题提供「有，图左上角有一个红色圆形。」"
          "这样的错误回答作负例，损失同时压低 rejected 的概率，"
          "这才是 SFT 范式缺失的那一半信号。")

    q4 = ("题 4（必答）：QLoRA 为什么「量化基座、只训适配器」？r=16 时参数占比大约变为几倍？\n"
          "量化基座的意义在于把显存占用压到 1/4 左右，使大模型能在消费级显卡上微调；"
          "而不全参微调的原因有三：其一，全参微调需为每个参数保存优化器状态"
          "（AdamW 为动量和方差各一份），显存开销是权重的数倍，量化省下的显存会被立刻吃掉；"
          "其二，本实验的目的正是验证「极少量参数即可改变行为」——"
          "实测 0.5290% 的可训练参数就让 count 类正确率和答题格式发生了可见变化；"
          "其三，全参微调更容易在小数据集上灾难性遗忘，"
          "而冻结基座只训低秩增量 ΔW = B·A 天然带有正则化效果。\n"
          "关于 r=16：LoRA 参数量与秩 r 成正比（每个被适配的权重矩阵贡献 r×(d_in+d_out) "
          "个参数），因此 r 由 8 增至 16 时参数量**恰好翻倍**。"
          "本机实测：r=8 时 1,363,968 个可训练参数（0.5290%），"
          "r=16 时 2,727,936 个（1.0524%），正好是 2.00 倍；"
          "反向验证 r=4 时为 681,984 个（0.2652%），恰为 r=8 的一半。")

    q2 = ("题 2（手算）：β=0.1，πθ(yw)=1.5、πref(yw)=1.0、πθ(yl)=0.2、πref(yl)=0.6。\n"
          "margin = β·[(πθ(yw) − πref(yw)) − (πθ(yl) − πref(yl))]\n"
          "       = 0.1 × [(1.5 − 1.0) − (0.2 − 0.6)]\n"
          "       = 0.1 × [0.5 − (−0.4)] = 0.1 × 0.9 = 0.0900\n"
          "loss = −log σ(margin) = −log σ(0.09) = log(1 + e^(−0.09)) ≈ 0.6492\n"
          "--dpo_check 0.1,1.5,1.0,0.2,0.6 复核输出：margin = 0.0900，loss = 0.6492，与手算一致。\n"
          "margin > 0 说明：策略模型把 chosen 与 rejected 的对数概率差，"
          "相对参考模型拉大了 0.09，即模型朝「更偏好 chosen」的方向移动，损失趋近 0。"
          "（注：若 margin < 0，则说明模型反而更偏好 rejected，需要继续训练。）")

    q3 = ("题 3：DPO 为什么可以不训练奖励模型？\n"
          "RLHF 的流程是先用偏好数据训练一个奖励模型 r(x,y)，再用强化学习最大化"
          "r(x,y) 同时约束与参考模型的 KL 距离。该优化问题有闭式最优解：\n"
          "    π*(y|x) = (1/Z(x)) · πref(y|x) · exp(r(x,y)/β)\n"
          "把它反解出 r：\n"
          "    r(x,y) = β · log(π*(y|x)/πref(y|x)) + β · log Z(x)\n"
          "关键在于：代入 Bradley-Terry 偏好模型 P(yw ≻ yl) = σ(r(x,yw) − r(x,yl)) 时，"
          "与单个 y 相关的 β·log Z(x) 项在作差中被**完全消掉**。于是隐式奖励为：\n"
          "    r(x,y) = β · [log πθ(y|x) − log πref(y|x)]\n"
          "即策略模型相对参考模型的对数概率比，本身就是奖励。"
          "把该式代回偏好似然并取负对数，就得到 DPO 损失 "
          "−log σ(β·[(log πθ(yw) − log πref(yw)) − (log πθ(yl) − log πref(yl))])。\n"
          "因此 DPO 无需显式训练奖励模型，也无需采样和强化学习循环 ——"
          "奖励被「隐式」地编码在策略与参考模型的对数概率差里，"
          "这正是「直接」二字的含义。")

    q5 = ("题 5：用「模型自生成回答 + 规则自动打分」构造偏好对（RLAIF/自我对弈）的利弊。\n"
          "好处：① 可扩展 —— 人工构造成本高且难以覆盖长尾，自生成可以廉价地产生大量"
          "偏好对；② 分布更贴近模型自身的输出分布，负例是模型**真实会犯**的错，"
          "而非人类臆测的错，训练信号更对症；③ 可迭代 —— 每一轮用当前策略生成、"
          "打分、再训练，形成自举式提升。\n"
          "风险：① 打分规则本身会成为新的偏见来源，模型会朝规则的可测量方向钻空子"
          "（reward hacking）；② 若拒绝样本全部由模型自己生成，容易出现「自我强化」——"
          "错误被反复确认为「应该被拒绝」的样式，多样性坍塌。\n"
          "结合本实验的 Q5 现象：我们的 β=0.5 那一组出现了 margin 显著为负（−2.106）"
          "却正确率最高的反常结果，说明偏好信号与任务能力并不一致。"
          "其原因很可能正是 rejected 构造得过于极端（如「1. 2. 3. 4. …」这种一眼可辨的"
          "退化回答），模型学到的可能是「变短」而非「变对」。"
          "若改用自生成偏好对，这种「退化解容易识别、真实错误难以区分」的问题会更严重 ——"
          "因为模型自己生成时，最像样的错误答案恰恰是最难被规则区分的那些。")

    q6 = ("题 6：与 AI 协作实现 DPO 的过程中，哪一处最容易悄悄写反？如何验证？\n"
          "本实验中最容易写反的是 **DPO 损失函数的符号方向**，具体有两处：\n"
          "① margin 的符号 —— margin = β·[(πθ(yw) − πref(yw)) − (πθ(yl) − πref(yl))]，"
          "chosen 在前、rejected 在后，若写反则等价于「鼓励模型偏好错误答案」，"
          "loss 依然会下降、曲线依然好看，但训练方向完全相反，属于静默失败。\n"
          "② 损失函数中 log 与负号的位置 —— −log σ(m) 才是损失。"
          "**本实验真的踩到了这一处**：指南参考代码的 --dpo_check 手算函数写成 "
          "`-math.log(1 + math.exp(-margin))`，多了一个负号，"
          "导致 margin = 0.01 时算出 loss = −0.6882（负损失）。\n"
          "我用的验证办法是**独立实现交叉验证**：不用指南的函数，"
          "直接按定义计算 −log(1/(1+e^(−m))) 得到 0.688160，"
          "与修正后的实现比对一致；同时用 5 组不同 margin（含负值、零值）"
          "覆盖边界情形。\n"
          "由此得到的经验：验证一个公式是否正确，不能只看「loss 是否在下降」"
          "（写反了照样下降），而要**另写一份独立实现对拍**，"
          "并特意检查边界值（margin=0 时 loss 应恰为 ln2 ≈ 0.6931）。"
          "值得强调的是，指南的训练路径 `-F.logsigmoid(margin)` 本身是正确的，"
          "错只错在手算复核函数 —— 这也说明「同一份代码里两处实现同一公式」时，"
          "恰好是最容易藏不一致的地方。")

    # 替换题 1、题 4，并在其后插入题 2/3/5/6
    setp(P[40], q1)
    setp(P[41], q4)
    anchor = P[41]
    for txt in (q2, q3, q5, q6):
        anchor = add_para_after(anchor, txt)

    # ---------- 九、DPO 延伸 ----------
    setp(P[44],
         "PRE_PAIRS 共 12 组：chosen 全部来自标准答案；rejected 覆盖三类典型错误 ——"
         "数错（off-by-one）、形状幻觉（把圆说成方）、对不存在的物体硬答「有」。"
         "此外，本人自拟 5 组新偏好对（用 --use-self-pairs 开启），"
         "补足原设计未覆盖、但在本机实测中**真实出现过**的失败模式："
         "① 枚举替代计数（对应 SFT 后 P04 的实际输出「1. 2. 3. 4. …」）；"
         "② 颜色幻觉（对应基线 P07「形状是白色的。」）；"
         "③ 语言漂移，中问英答（对应 SFT 后 P09「Green.」）；"
         "④ 无关词/截断（对应 SFT 后 P08「半身。」）；"
         "⑤ 冗长废话型 —— 大段描述图形分布却始终不给数字。")

    setp(P[45], "9.2 DPO 实现与手算（GPU 实测）")
    setp(P[46],
         "margin = β·[(log πθ(yw) − log πref(yw)) − (log πθ(yl) − log πref(yl))]，"
         "loss = −log σ(margin)；参考模型取「关闭适配器的基座」"
         "（disable_adapter + no_grad）；lr=5e-5，β=0.1，训练 12 步（12 组 × 1 epoch）。"
         "实现要点：序列对数概率只对答案位置做 log_softmax 后求和（非取平均）；"
         "参考路径加 no_grad。")
    setp(P[47],
         f"手算复核：β=0.1，πθ(yw)=1.2、πref(yw)=0.8、πθ(yl)=0.9、πref(yl)=0.6 "
         f"⇒ margin={DATA['hand_margin']}，loss≈{DATA['hand_loss']}。"
         "另自选两组：① β=0.1、πθ(yw)=1.5、πref(yw)=1.0、πθ(yl)=0.2、πref(yl)=0.6 "
         "⇒ margin=0.0900，loss=0.6492；② margin=0（即两侧对数概率差完全一致）"
         "⇒ loss=ln2=0.6931，为损失的理论上界。三组均经 --dpo_check 复核一致。\n"
         "【注】指南原 --dpo_check 函数存在符号错误（会算出负损失），"
         "已修正为 log(1+e^(−margin))，详见 6.2 与思考题 6。")

    setp(P[48], "9.3 三阶段对比（GPU 实测，DPO 12 步）")
    # 注：样例原编号为「表 4／表 5」，缺「表 3」（应为早期草稿遗留），此处顺延修正
    setp(P[49], "表 3　三阶段 × 题型正确率（GPU 实测）")
    t = T[50]
    for (key, label), row in zip(
        [("base", "基线"), ("sft", "SFT（30 步）"), ("dpo", "SFT+DPO（12 步）")], t.rows[1:]
    ):
        c, a, e, tot = DATA[key]
        fill(row.cells[0], label)
        fill(row.cells[1], f"{c}/5")
        fill(row.cells[2], f"{a}/5")
        fill(row.cells[3], f"{e}/5")
        fill(row.cells[4], str(tot))

    # 表 5：DPO 逐步日志（12 步）
    setp(P[51], "表 4　DPO 训练日志（GPU 实测，12 步，dpo_log.csv）")
    t = T[52]
    rows_csv = []
    if os.path.exists("dpo_log_b0.1.csv"):
        import csv as _csv
        with open("dpo_log_b0.1.csv", encoding="utf-8") as f:
            rows_csv = list(_csv.DictReader(f))
    # 表头 + 每步一行
    while len(t.rows) < len(rows_csv) + 1:
        t._tbl.append(copy.deepcopy(t.rows[-1]._tr))
    for i, r in enumerate(rows_csv):
        row = t.rows[i + 1]
        m = float(r["margin"])
        note = "margin>0：偏好方向正确" if m > 0 else "margin<0：本轮偏好方向反了"
        fill(row.cells[0], r["step"])
        fill(row.cells[1], f"{float(r['loss']):.4f}")
        fill(row.cells[2], f"{m:.3f}")
        fill(row.cells[3], note)

    setp(P[53],
         "分析：与样例报告「margin 单调上升」不同，本机 12 步的 margin 呈明显震荡"
         "（+0.393 → −0.393 → +1.426 → −0.393 → +0.198），loss 同步波动。"
         "原因是偏好对仅 12 组、学习率 5e-5，单批梯度噪声远大于信号。"
         "更值得注意的结果是：DPO 使总正确率由 SFT 的 2/15 降至 1/15，"
         "即**任务正确率不升反降**；而在 β 扫描中 β=0.5 时 margin 大幅为负（低至 −2.106）、"
         "正确率却是三档中最高的 3/15。这两个现象共同说明："
         "DPO 的 loss/margin 与任务正确率是两个独立维度 ——"
         "margin 上升只表示模型相对参考模型的偏好差被拉大，"
         "并不保证模型在具体任务上答得更对。这正是指南 Q5 所述「过度优化」的实证："
         "policy 在压低 rejected 概率的同时，chosen 的绝对概率也可能下降，"
         "表现为回答趋短、多样性下降，而任务能力并未同步提升。"
         "加入自拟 5 组偏好对后结果为 1/15，与 12 组持平，"
         "说明偏好对数量的增加并未自动转化为任务能力。")

    setp(P[54], "9.4 延伸任务（超参敏感性）")
    setp(P[55],
         "【r 敏感性，SFT 阶段】\n"
         "  r=4：占比 0.2652%，loss 7.5356→2.1472，正确率 2/15\n"
         "  r=8：占比 0.5290%，loss 6.6746→1.6390，正确率 2/15\n"
         "  r=16：占比 1.0524%，loss 5.1738→1.0789，正确率 3/15\n"
         "结论：参数量与秩 r 严格成正比（4:8:16 = 0.2652%:0.5290%:1.0524%）。"
         "r 增大使训练 loss 单调下降、count 类从 2/5 升至 3/5，"
         "但 attr 与 exist 在三个 r 下**全部为 0/5** —— "
         "说明加大低秩容量能改善「可学的格式与计数」，"
         "却无法触及视觉绑定与诚实性这两类能力缺口。\n\n"
         "【β 敏感性，SFT+DPO】\n"
         "  β=0.05：正确率 1/15\n"
         "  β=0.1 ：正确率 1/15，margin 震荡（−0.39 ~ +1.43）\n"
         "  β=0.5 ：正确率 3/15，margin 全程为负（−0.251 → −2.106）\n"
         "结论：β 增大本应更强地约束偏离参考模型，但实测 β=0.5 时 margin 转为大幅负值"
         "（偏好方向反了）而正确率反而最高。这说明在小样本（12 对）下，"
         "β 的作用被梯度噪声淹没：正确率的波动更多来自 SFT 权重的轻微漂移，"
         "而非 DPO 的偏好信号。若要得到稳定结论，需扩大偏好对规模或降低学习率重跑。")

    # ---------- 插入插图 ----------
    # 在 4.2 之后插入三阶段图，在 9.3 之后插入 DPO 曲线与扫描图
    def insert_fig(anchor_p, path, caption):
        np_ = add_para_after(anchor_p, caption)
        run = np_.add_run()
        run.add_picture(path, width=Inches(5.6))
        return np_

    # 图号按**文档中出现的先后**编排：§4.2 的三阶段图 → 图1，§9.2 的 DPO 曲线 → 图2
    if os.path.exists("fig2_three_stage.png"):
        insert_fig(P[24], "fig2_three_stage.png", "图 1　三阶段 × 题型 正确率")
    if os.path.exists("fig1_dpo_curve.png"):
        insert_fig(P[47], "fig1_dpo_curve.png", "图 2　DPO 训练过程：loss 与 margin（β=0.1，12 步）")
    if os.path.exists("fig3_scans.png"):
        insert_fig(P[55], "fig3_scans.png", "图 3　LoRA 秩 r 与 DPO β 的敏感性扫描")

    d.save(DST)
    print(f"[OK] 已生成 {DST}")
    print(f"     段落 {len(P)} 个，表格 {len(T)} 张")


if __name__ == "__main__":
    main()
