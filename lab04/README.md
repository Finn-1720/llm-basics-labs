# lab04 — Vector RAG / GraphRAG / WikiRAG 对比实验

## 运行

```bash
pip install sentence-transformers networkx numpy openai
export HF_ENDPOINT=https://hf-mirror.com        # 模型下载走镜像
python hw4_rag.py --mode eval                   # 三范式 × 20 题检索评测（无需 API）
python hw4_rag.py --mode eval --k 1             # K 敏感性扫描：1 / 2 / 3 / 5
python hw4_rag.py --mode vector --q Q01 --trace # 单题，带检索中间量
python hw4_rag.py --mrr_check 3,1,2             # MRR 手算复核
```

生成环节需要任一 OpenAI 兼容接口，配置三个环境变量：

```bash
export LLM_API_BASE=...   LLM_API_KEY=...   LLM_MODEL=...
export LLM_NO_THINK=1     # 若模型默认开思维链（如 Qwen3），关掉它
```

本次实验的生成环节用的是**本地 llama.cpp + Qwen3-14B-Q4_K_M**
（`llama-server` 提供 OpenAI 兼容接口，未使用云端 API）。

## 对参考代码的修改（3 处）

1. **新增 5 道自拟题 Q16~Q20**（事实/多跳/全局，均标注 evidence）——任务四①
2. **新增 `--trace`**：打印块级相似度分数、命中实体、沿边收集的三元组、条目级 Top-3。
   没有它就无法定位 GraphRAG 事实题为何卡在 0.714（实体匹配失败→退化为社区查询）
3. **新增 `LLM_NO_THINK` 开关**：Qwen3 等混合推理模型默认输出思维链，
   会让 `content` 为空、内容全在 `reasoning_content` 里；该开关通过
   `chat_template_kwargs` 关闭思维链

## 目录

| 路径 | 内容 |
|---|---|
| `hw4_rag.py` | 主程序（含上述 3 处修改） |
| `make_figs.py` | 生成报告三张插图（图 2 含带碰撞检测的中文标签放置） |
| `qdetail.py` | 逐题导出各范式的检索结果、首次命中排名、Recall |
| `results/` | `eval_k1/2/3/5.txt` 四档 K 的全量评测；`qdetail.txt` 逐题明细；`failures.txt` 失败清单 |
| `traces/` | 单题 `--trace` 输出（检索中间量） |
| `answers/` | 3 范式 × 3 题的 LLM 生成答案 |
| `fig1~3*.png` | 报告插图 |
| `fill_report.py` / `verify_report.py` | 报告生成与验证脚本 |
