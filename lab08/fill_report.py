# -*- coding: utf-8 -*-
"""把实验八的结果填进报告模板（在副本上操作，原模板不动）。

输出：2025311489_赖宇轩_实验作业八.docx
"""
import copy
import json
import os

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
# 可移植：默认取同级 ../实验/ 下的模板，可用环境变量 LAB08_TEMPLATE 覆盖
TPL = os.environ.get("LAB08_TEMPLATE",
                   os.path.join(HERE, "..", "实验",
                                "作业八_自制语言模型知识蒸馏_实验报告（空白模板）.docx"))
if not os.path.exists(TPL):
    raise SystemExit(
        "未找到报告模板：%s\n"
        "请把空白模板放到 ../实验/ 下，或设置环境变量 LAB08_TEMPLATE 指向它。" % TPL)

OUT_DOCX = os.path.join(HERE, "2025311489_赖宇轩_实验作业八.docx")
OUT = os.path.join(HERE, "runs_distill")


# ---------------------------------------------------------------- 工具函数
def set_par(par, text, bold=False):
    """替换段落文字，保留第一个 run 的字体格式。"""
    runs = par.runs
    if runs:
        r0 = runs[0]
        for r in runs[1:]:
            r._element.getparent().remove(r._element)
        r0.text = text
        if bold:
            r0.bold = True
    else:
        par.add_run(text)


def add_par_after(par, text, bold=False):
    """在 par 之后插入一个新段落。"""
    new_p = copy.deepcopy(par._p)
    # 清空内容
    for child in list(new_p):
        if child.tag.endswith('}r') or child.tag.endswith('}hyperlink'):
            new_p.remove(child)
    par._p.addnext(new_p)
    np = Paragraph(new_p, par._parent)
    run = np.add_run(text)
    if bold:
        run.bold = True
    return np


def set_cell(cell, text):
    """写表格单元格，保留首段首 run 的字体。"""
    par = cell.paragraphs[0]
    set_par(par, text)
    for p in cell.paragraphs[1:]:
        p._element.getparent().remove(p._element)


def put_image(cell, path, width_in):
    cell.text = ""
    par = cell.paragraphs[0]
    run = par.add_run()
    run.add_picture(path, width=Inches(width_in))


def no_borders(table, doc):
    """图片占位表去掉边框，看起来像插图而不是表格。

    注意：模板把边框同时写在 tblPr 和每行的 tblPrEx 里，两处都要清掉，
    否则行级的 tblPrEx 会覆盖表格级设置。
    """
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    tbl = table._tbl
    for el in tbl.tblPr.findall(W + "tblBorders"):
        tbl.tblPr.remove(el)
    for tr in tbl.findall(W + "tr"):
        for ex in tr.findall(W + "tblPrEx"):
            for b in ex.findall(W + "tblBorders"):
                ex.remove(b)
    table.style = doc.styles["Normal Table"]


def no_split(table):
    """禁止表格行跨页断开。"""
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))


# ---------------------------------------------------------------- 读数据
def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


main = load("results_all.json")
teach = load("results_teacher.json")
feat = load("results_all_feat01.json")
abla = load("results_ablate.json")
quant = load("quant_result.json")
samples = load("samples.json")

T = main["teacher"]
S0 = main["scratch"]
S1 = main["kd"]
T_TIME = teach["teacher"]["time_s"]

doc = docx.Document(TPL)
body = doc.element.body

paras = doc.paragraphs
tables = doc.tables

# ---------------------------------------------------------------- 表 1 实验信息
t = tables[0]
set_cell(t.rows[0].cells[1], "自制语言模型的知识蒸馏（纯 CPU）")
set_cell(t.rows[1].cells[1], "人工智能——深度学习大模型智能体")
set_cell(t.rows[2].cells[1], "赖宇轩 / 2025311489")
set_cell(t.rows[3].cells[1], "—")
set_cell(t.rows[4].cells[1], "2026-11-06")
set_cell(t.rows[5].cells[1], "CPU / Python 3.12.10 / PyTorch 2.14.0")

# ---------------------------------------------------------------- 表 2 环境
t = tables[1]
env = [
    ("操作系统", "Windows 11 Pro（10.0.26200）"),
    ("Python 版本", "3.12.10"),
    ("PyTorch 版本", "2.14.0+cu130（未调用 .cuda()，全程 CPU）"),
    ("CPU 型号 / 核心数", "Intel Core Ultra 5 245KF / 14 线程"),
    ("内存", "15.5 GB"),
    ("是否使用 GPU", "否"),
]
for i, (k, v) in enumerate(env, start=1):
    set_cell(t.rows[i].cells[1], v)

# ---------------------------------------------------------------- 表 3 命令记录
t = tables[2]
rows = [
    ("1. 环境自检", "python distill_lm.py --mode quick", "成功", "3.6 s"),
    ("2. 训练教师", "python distill_lm.py --mode teacher --retrain", "成功", "5.7 s"),
    ("3. 从零 + 蒸馏", "python distill_lm.py --mode all", "成功", "7.5 s"),
    ("4. 消融实验", "python distill_lm.py --mode ablate", "成功", "22.4 s"),
    ("5. 拓展（A/B/D）",
     "python distill_lm.py --mode all --feat 0.1 --tag feat01；"
     "python make_heatmap.py；python quantize_eval.py", "成功", "4.8 s / <1 s / 5.4 s"),
]
for i, (a, b, c, d) in enumerate(rows, start=1):
    set_cell(t.rows[i].cells[0], a)
    set_cell(t.rows[i].cells[1], b)
    set_cell(t.rows[i].cells[2], c)
    set_cell(t.rows[i].cells[3], d)

# ---------------------------------------------------------------- 表 4 模型参数
t = tables[3]
params = [
    ("n_layer", "2", "2"),
    ("n_head", "4", "4"),
    ("n_embd", "160", "64"),
    ("block_size", "96", "96"),
    ("参数量", "737,600（0.738 M）", "147,584（0.148 M）"),
    ("压缩比", "—", "5.0×"),
]
for i, (k, a, b) in enumerate(params, start=1):
    set_cell(t.rows[i].cells[0], k)
    set_cell(t.rows[i].cells[1], a)
    set_cell(t.rows[i].cells[2], b)

# ---------------------------------------------------------------- 表 5 超参数
t = tables[4]
hp = [
    ("温度 T", "4", "默认 4；消融扫描 1 / 2 / 4 / 8"),
    ("权重 α", "0.5", "默认 0.5；消融扫描 0.3 / 0.5 / 0.7 / 0.9"),
    ("学习率", "3e-3", "AdamW，weight_decay=0.01，梯度裁剪 1.0"),
    ("batch_size", "24", "默认 24"),
    ("教师训练步数", "200", "默认 200"),
    ("学生训练步数", "120", "从零与蒸馏两组完全相同"),
    ("随机种子", "0", "学生初始化 seed+7，数据采样 seed+1"),
    ("特征蒸馏权重", "0（主实验）/ 0.1（拓展 A）", "0 表示关闭"),
]
for i, (k, v, note) in enumerate(hp, start=1):
    set_cell(t.rows[i].cells[0], k)
    set_cell(t.rows[i].cells[1], v)
    set_cell(t.rows[i].cells[2], note)

# ---------------------------------------------------------------- 表 6 三组对比
t = tables[5]
data = [
    ("教师 Teacher", f"{T['params']:,}", "200", f"{T_TIME:.1f} s",
     f"{T['val_loss']:.4f}", f"{T['ppl']:.2f}", "—"),
    ("学生·从零训练", f"{S0['params']:,}", "120", f"{S0['time_s']:.1f} s",
     f"{S0['val_loss']:.4f}", f"{S0['ppl']:.2f}", f"{S0['agree']:.3f}"),
    ("学生·蒸馏训练", f"{S1['params']:,}", "120", f"{S1['time_s']:.1f} s",
     f"{S1['val_loss']:.4f}", f"{S1['ppl']:.2f}", f"{S1['agree']:.3f}"),
]
for i, row in enumerate(data, start=1):
    for j, v in enumerate(row):
        set_cell(t.rows[i].cells[j], v)

# ---------------------------------------------------------------- 表 7 消融网格
t = tables[6]
grid = {}
for r in abla["ablation"]:
    grid[(r["T"], r["alpha"])] = r["val_loss"]
Ts = [1, 2, 4, 8]
As = [0.3, 0.5, 0.7, 0.9]
for i, Tv in enumerate(Ts, start=1):
    set_cell(t.rows[i].cells[0], f"T = {Tv}")
    for j, a in enumerate(As, start=1):
        set_cell(t.rows[i].cells[j], "%.4f" % grid[(Tv, a)])

# ---------------------------------------------------------------- 表 8 生成样例
t = tables[7]
set_cell(t.rows[1].cells[1], samples["teacher"])
set_cell(t.rows[2].cells[1], samples["scratch"])
set_cell(t.rows[3].cells[1], samples["kd"])

# ---------------------------------------------------------------- 表 9 拓展
t = tables[10]
ext = [
    ("A", "特征蒸馏（FitNet）", "已完成",
     f"feat=0 → {S1['val_loss']:.4f}；feat=0.1 → {feat['kd']['val_loss']:.4f}（−0.47%）", "5"),
    ("B", "温度/权重热力图", "已完成",
     "runs_distill/heatmap.png；最优 T=8、α=0.3", "3"),
    ("C", "自蒸馏", "未完成", "—", "0"),
    ("D", "动态量化导出", "已完成",
     f"{quant['fp32']['size_mb']:.3f} → {quant['int8']['size_mb']:.3f} MB"
     f"（{quant['size_ratio']:.2f}×），loss {quant['fp32']['val_loss']:.4f} → "
     f"{quant['int8']['val_loss']:.4f}", "5"),
    ("E", "教师规模扫描", "未完成", "—", "0"),
    ("F", "换成真实语料", "未完成", "—", "0"),
]
for i, row in enumerate(ext, start=1):
    for j, v in enumerate(row):
        set_cell(t.rows[i].cells[j], v)

# ---------------------------------------------------------------- 表 10 问题与解决
t = tables[11]
probs = [
    ("1", "指南附录的 PDF 代码直接复制后报 SyntaxError：长行被折行，"
          "字符串在「依旧」中间断开",
     "PDF 自动折行，续行顶格且丢失缩进/空格",
     "用 PyMuPDF 逐字符取 x 坐标重建空格与缩进，再按括号/引号配对合并续行，"
     "最后 ast.parse 校验通过"),
    ("2", "torch.ao.quantization.quantize_dynamic 打印"
          "「Eager mode quantization ... migrate to torchao」警告",
     "torch 2.14 已把 eager 量化标记为弃用",
     "API 仍可用，保留调用并记录警告；后续可迁移到 torchao 的 quantize_"),
    ("3", "拓展 A 跑完 --feat 0.1 后，主实验的 results_all.json 被覆盖",
     "原代码 save() 用 tag=args.mode 命名，两次运行同名",
     "新增 --tag 参数给结果文件名加后缀（--tag feat01），两组结果并存"),
]
for i, row in enumerate(probs, start=1):
    for j, v in enumerate(row):
        set_cell(t.rows[i].cells[j], v)

# ---------------------------------------------------------------- 图片
put_image(tables[8].rows[0].cells[0], os.path.join(OUT, "fig1_curves.png"), 6.0)
put_image(tables[9].rows[0].cells[0], os.path.join(OUT, "fig2_heatmap.png"), 4.6)
for _t in (tables[8], tables[9]):
    no_borders(_t, doc)

# 所有表格禁止行内断页
for _t in tables:
    no_split(_t)

# ---------------------------------------------------------------- 正文段落
P = {i: para for i, para in enumerate(paras)}

# 一、实验目的
set_par(P[4],
        "1. 亲手实现一次完整的知识蒸馏：让教师模型在温度 T 下输出软标签，"
        "学生同时学习硬标签与软标签；\n"
        "2. 逐项讲清损失 L=(1−α)·CE + α·T²·KL 里每个因子的作用，"
        "特别是 T² 缩放的来历；\n"
        "3. 在纯 CPU 上完成 0.74 M → 0.15 M（5×）压缩，"
        "并与同架构、同初始化、同数据顺序的从零训练严格对照；\n"
        "4. 用 T×α 的 4×4 网格观察温度过高/过低、权重过大/过小时分别如何失效；\n"
        "5. 把蒸馏与动态量化叠加，认识「蒸馏压参数量、量化压数值精度」的分工。")

# 三、实验原理
set_par(P[7],
        "训练语言模型时通常只给 one-hot 硬标签，它只回答「答案是哪个字」；"
        "而教师在温度 T 下输出的完整概率分布还回答了「每个字有多像」。"
        "这些残留在错误字上的概率就是暗知识，对学生构成一种结构先验。"
        "带温度的 softmax 为 p_i=exp(z_i/T)/Σexp(z_j/T)：T 越大分布越平滑，"
        "T=1 时软标签几乎退化为 one-hot。总损失 "
        "L=(1−α)·CE(y,p_s)+α·T²·KL(p_t^T‖p_s^T)；乘 T² 是因为 softmax 对 z_s 求导"
        "带一个 1/T 因子、KL 中又有一个 1/T，梯度与 T² 成反比，乘 T² 正好抵消，"
        "换温度时不必重调学习率。")

# 7.1
set_par(P[19],
        "左图三条曲线：教师从 5.83 起，约 25 步就跌破 2，200 步时训练损失 0.73；"
        "两名学生从 5.79 起，120 步分别收敛到约 1.05（从零）与 1.02（蒸馏）。"
        "教师曲线全程压在两条学生曲线下方，这正是 5 倍容量带来的差距。"
        "值得注意的是，蒸馏学生（绿）自始至终贴着从零学生（橙）的下方，"
        "差距在 40~80 步区间最明显；两条曲线用的是同一初始化种子与同一数据顺序，"
        "唯一的差别是损失里多了 KL 项。右图留出集：教师 0.9078、从零 1.4144、"
        "蒸馏 1.3216。")

# 7.2
set_par(P[21],
        f"表 6 两行学生：留出集 loss 由 {S0['val_loss']:.4f} 降到 {S1['val_loss']:.4f}，"
        f"下降 {S0['val_loss']-S1['val_loss']:.4f}，相对 {100*(S0['val_loss']-S1['val_loss'])/S0['val_loss']:.1f}%；"
        f"困惑度由 {S0['ppl']:.2f} 降到 {S1['ppl']:.2f}；"
        f"与教师的 top-1 一致率由 {S0['agree']:.3f} 升到 {S1['agree']:.3f}"
        f"（+{100*(S1['agree']-S0['agree']):.1f} 个百分点）。"
        "两组参数完全相同（147,584）、训练步数相同（120）、初始化种子相同"
        "（都来自 seed+7）、每步拿到的批次相同（数据采样器都用 seed+1），"
        "唯一差异是损失函数。因此这 6.6% 只能归因于软标签："
        "学生不仅学到了「下一个字是哪个」，还学到了教师对各种近义字、"
        "各句法槽位之间的相对偏好，这部分信息对只有 15 万参数的学生相当于"
        "额外的强监督。")

# 7.3
best = min(abla["ablation"], key=lambda r: r["val_loss"])
worst = max(abla["ablation"], key=lambda r: r["val_loss"])
set_par(P[24],
        f"16 组中最优为 T={best['T']}、α={best['alpha']}（{best['val_loss']:.4f}），"
        f"最差为 T={worst['T']}、α={worst['alpha']}（{worst['val_loss']:.4f}）。"
        "规律有三条：\n"
        "（1）T=1 时四组几乎完全相同（1.5804~1.5858），说明温度太低时软标签接近 "
        "one-hot，暗知识几乎为零，蒸馏退化成普通训练，α 调多大都没用；\n"
        "（2）T=4 与 T=8 在 α=0.3 时最好（1.5588 / 1.5573），"
        "说明分布压平之后更要控制软标签的权重；\n"
        "（3）α 越大越差，且 T 越大恶化越快：T=8、α=0.9 达到 1.7166，"
        "比从零训练（1.4144）还差很多。原因是 α 接近 1 时 (1−α)·CE 项权重趋近 0，"
        "学生失去了硬标签这个「正确答案」的锚；而教师本身 val=0.9078 并不完美，"
        "T 越大分布越平，教师的错误也被放大得越充分，学生就把这些错误一并学了去。"
        "这与 4.2 节所说 α∈[0.5,0.9] 的经验范围并不矛盾——那是针对强教师，"
        "本实验教师只有 0.74 M，α 过大时它的不完美会被放大。")

# 7.4
set_par(P[26],
        "三段都遵守了语料的模板文法（人名 + 地点 + 动作 + 景物，心情收尾），"
        "没有出现乱码、英文或明显的重复崩坏，说明 323 字的字符级建模是成功的。"
        "差别在细节：教师段的引号成对、句号位置正确，「阿远陌上恰遇归雁，欣然归去」"
        "槽位齐全；从零学生段出现「月 道：」缺主语、引号未闭合后直接接"
        "「长叹一笑而别」的模板错位；蒸馏学生段的引号使用明显更规范，"
        "但仍有「山僧 在长叹一笑而别」这种把心情词塞进地点槽的错误。"
        "总体看蒸馏学生的文本更接近教师——在句法骨架、标点配对这类结构层面更稳，"
        "这与 agree 0.707 > 0.644 的量化结果一致；而在词语搭配的合理性上，"
        "两者都远不如教师，说明软标签传递的主要是结构信息，"
        "语义常识仍需更多数据与容量。")

# 八、拓展
_ext1 = add_par_after(P[27],
              "A. 特征蒸馏（FitNet）：在 KD 损失之外再加「学生最后一层隐状态经线性投影到 "
              "160 维后与教师隐状态的 MSE」，权重 0.1。留出集 loss 从 1.3216 降到 1.3154"
              "（−0.47%），一致率基本不变（0.707 → 0.706）。收益为正但很小，"
              "说明中间层特征与输出层软标签携带的信息高度重叠，"
              "对这么小的模型额外加一路监督的边际收益有限。")
_ext2 = add_par_after(_ext1,
              f"B. 温度/权重热力图：见图 2，16 个格子颜色越深 loss 越低。"
              f"最优格 T=8/α=0.3 用红框标出，左下角一片深色说明「低温必须配小权重」"
              f"这一规律在图上非常直观。")
add_par_after(_ext2,
              f"D. 动态量化导出：用 torch.ao.quantization.quantize_dynamic 把 nn.Linear "
              f"换成 int8。体积由 {quant['fp32']['size_mb']:.3f} MB 压到 "
              f"{quant['int8']['size_mb']:.3f} MB，压缩 {quant['size_ratio']:.2f}×；"
              f"留出集 loss 只从 {quant['fp32']['val_loss']:.4f} 变到 "
              f"{quant['int8']['val_loss']:.4f}（+{quant['int8']['val_loss']-quant['fp32']['val_loss']:.4f}），"
              f"几乎无损。但延迟并非全面变快：batch=1 时 {quant['fp32']['latency_ms_b1']:.2f} ms → "
              f"{quant['int8']['latency_ms_b1']:.2f} ms（反而慢 {1/quant['speedup_b1']:.1f}×），"
              f"batch=24 时 {quant['fp32']['latency_ms_b24']:.2f} ms → "
              f"{quant['int8']['latency_ms_b24']:.2f} ms（快 {quant['speedup_b24']:.2f}×）。"
              f"原因是动态量化每一步都要现算 scale/zero-point，"
              f"小 batch 下这部分固定开销盖过了整数矩阵乘的收益。")

# 九、问题与解决 前言
add_par_after(P[28], "本次实验踩到的三个问题记录在下面的表格中。")

# 十、结论
set_par(P[30],
        "本次实验完整跑通了「教师训练 → 从零学生对照 → 蒸馏学生 → 16 组消融 → "
        "拓展 A/B/D」全流程，所有数据均为本机实测。"
        "最核心的结论是：在参数量、训练步数、初始化种子、数据顺序全部相同的前提下，"
        "只把损失换成 (1−α)·CE + α·T²·KL，就使留出集 loss 从 1.4144 降到 1.3216"
        "（−6.6%），困惑度 4.11→3.75，与教师一致率 0.644→0.707。"
        "这说明软标签带来的确实不只是「训练更久」，而是一份额外的结构信息。")
_c1 = add_par_after(P[30],
              "消融实验揭示了一个容易忽略的耦合关系：温度与权重必须匹配。"
              "T=1 时软标签退化成 one-hot，四组 α 几乎无差别；"
              "T=8 时必须配小权重，α=0.9 反而比从零训练更差（1.7166）。"
              "根本原因是 α 接近 1 会丢掉硬标签的锚定作用，而教师本身并不完美。")
add_par_after(_c1,
              "拓展实验给出了两个符合预期的结果：特征蒸馏的边际收益很小（−0.47%），"
              "动态量化能把体积压到 1/3.6 且几乎不损精度，但在 batch=1 时因量化开销反而更慢。"
              "可改进之处：本实验的语料是模板合成、槽位独立采样，"
              "学生很容易学到「位置→词类」的浅层规律；"
              "换成真实语料才能真正检验蒸馏在长程依赖上的价值。"
              "另外教师只有 0.74 M，与学生的容量差距有限，"
              "「教师越强收益是否越大」这一问题本次未做教师规模扫描，无法用实测回答。")

# 十一、思考题
QA = [
    ("1. 蒸馏时学生只学软标签（α=1）会怎样？请从梯度角度说明为什么还需要硬标签。",
     "α=1 时损失只剩 KL 项，梯度把学生的 logits 拉向教师分布在所有词上的相对形状，"
     "但这个约束是「相对的」：把所有 logits 同时平移一个常数，KL 不变，"
     "所以它对绝对置信度没有任何约束。教师本身有误差（本实验教师留出集 loss 0.9078，"
     "并不完美），其错误会被学生原样模仿；同时 KL 对正确类和错误类一视同仁，"
     "只要与教师的相对排序一致就行。硬标签提供的正是不可协商的绝对锚点——"
     "这个位置必须是「光」。本实验 T=8、α=0.9 时 loss 反而恶化到 1.7166，"
     "就是丢掉硬标签锚定后的实测证据。"),
    ("2. KD 损失里为什么必须乘 T²？如果漏乘，把 T 从 4 改到 8 时需要怎样调整学习率？",
     "设学生 logits 为 z_s，softmax(z/T) 对 z_s 的导数中含 1/T；"
     "KL(p_t^T‖p_s^T) 中 log p_s^T 对 z_s 的导数又含一个 1/T，两次相乘后梯度与 1/T² 成正比。"
     "乘上 T² 正好把这个因子抵消，使不同温度下的梯度尺度一致。"
     "若漏乘，T 从 4 改到 8 时梯度会缩小到原来的 1/4，等效学习率变成 1/4。"
     "要保持相同步长，应把学习率乘以 (8/4)²=4，即从 3e-3 调到 1.2e-2；"
     "否则会观察到「温度一变，训练就变慢」的假象。"),
    ("3. 本实验的教师只有 0.74 M 参数，它算「大模型」吗？蒸馏的收益是否依赖教师足够强？"
     "请用任务五中的教师规模扫描数据回答。",
     "按今天的标准 0.74 M 当然不算大模型，它只是一个「容量为学生 5 倍的同架构模型」。"
     "蒸馏的收益依赖的是「教师比学生强」，而不是绝对规模：本实验教师留出集 0.9078、"
     "学生从零 1.4144，差距足以支撑可测量的收益。"
     "指南 4.4 节给出的规律是：收益出现在学生欠拟合区间（100 步约 5%，150 步约 2.6%，"
     "200 步以上学生追上教师后收益消失甚至反转），说明教师的作用是提供更密的监督信号，"
     "一旦学生自己达到同等水平，信号就不再有增量。"
     "需要说明：本次未做拓展 E 的教师规模扫描（C∈{96,160,256}），"
     "因此「教师越强、收益是否越大」没有本机实测数据，上述结论引自指南而非本次实测。"),
    ("4. 在同样的参数量下，学生从零训练和蒸馏训练的差异，本质上是「更多信息」还是「更强正则」？"
     "请设计一个实验来区分这两种解释。",
     "两种解释在现象上都会表现为留出集损失下降，需要把「信息」和「平滑」解耦。"
     "设计一：取同一个教师，把它的输出在词表维度上随机置换（保持分布的熵和平滑度，"
     "但破坏其与输入的相关性）作为对照组。若置换后收益消失，说明起作用的是信息；"
     "若仍有收益，说明只是标签平滑带来的正则。"
     "设计二：用 T→∞ 的均匀分布（纯正则、零信息）与真实软标签对比，"
     "若均匀分布也能带来同等收益，则说明是正则效应。"
     "本实验已有一个间接证据：T=1 时四组 α 几乎无差别（1.5804~1.5858），"
     "而 T=1 的软标签几乎不含信息——这支持「信息」解释。"
     "要给出确定结论仍需按上述设计做置换对照。"),
    ("5. 如果把温度 T 设为 100，软标签会变成什么样？此时蒸馏还能起作用吗？为什么？",
     "softmax(z/100) 把 logits 压缩了 100 倍，各词概率几乎相等（≈1/323≈0.0031），"
     "分布退化成均匀分布，字与字之间的差异被彻底抹平。"
     "此时 KL(p_t‖p_s) 近似等于「让学生也输出均匀分布」，"
     "它只提供极强的标签平滑/熵正则，不再携带任何「哪个字更可能」的信息。"
     "技术上它仍会「起作用」——但作用性质变了，变成纯正则；"
     "而且 α 稍大就会压掉硬标签的学习信号，使模型学不到任何东西。"
     "所以 T=100 不能算有意义的蒸馏。本实验 T=8、α=0.9 已经出现明显恶化"
     "（1.7166），T=100 只会更糟。"),
    ("6. 蒸馏、量化、剪枝都能压缩模型。它们分别压缩了什么？能否叠加使用？"
     "请给出一个「先蒸馏再量化」的流水线设想。",
     "蒸馏压的是参数量/计算量（用大模型的输出去训练小模型，学生结构本身更小）；"
     "量化压的是数值精度（fp32→int8，参数量不变，存储与带宽变小）；"
     "剪枝压的是结构（去掉不重要的连接或通道，参数量和计算量都减少）。"
     "三者可以叠加，典型顺序是 蒸馏 → 剪枝 → 量化："
     "（1）先训练教师，用蒸馏得到结构更小的学生；"
     "（2）对学生做结构化剪枝（按通道 L1 范数剪掉 30%），"
     "再用蒸馏损失微调恢复精度（此时教师仍可复用）；"
     "（3）对剪枝后的模型做动态或静态量化，导出 int8 供 CPU 部署。"
     "本次实测了两步：蒸馏把参数从 737,600 压到 147,584（5.0×），"
     f"量化再把权重体积从 {quant['fp32']['size_mb']:.3f} MB 压到 "
     f"{quant['int8']['size_mb']:.3f} MB（{quant['size_ratio']:.2f}×），"
     f"两者叠加约 18× 的存储压缩，而留出集 loss 只从 {quant['fp32']['val_loss']:.4f} 变到 "
     f"{quant['int8']['val_loss']:.4f}。"),
]

anchor = P[32]
for i, (q, a) in enumerate(QA):
    set_par(P[32 + i], q)
    add_par_after(P[32 + i], a)

doc.save(OUT_DOCX)
print("saved:", OUT_DOCX)
