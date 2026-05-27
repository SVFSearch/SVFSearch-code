# SVFSearch

<div align="center">

[🌐 项目主页](https://svfsearch.github.io/SVFSearch-page/) &nbsp;·&nbsp; [🤗 数据集](https://huggingface.co/datasets/svfsearch/SVFSearchData) &nbsp;·&nbsp; [English](README.md)

</div>

---

> **说明：** 本仓库是 **SVFSearch** 中的 **Plan-Act-Replan Agent** 代码实现。  
> **mmsearch-r1-game** 代码可参考：https://github.com/SVFSearch/SVFSearch-mmsearch-r1-game

---

本仓库包含 **SVFSearch** 的推理与评测代码——一个具备动态工具选择能力的多模态检索增强问答系统。当前提供三条主运行路径：

- **`run_agent.py`** — Plan-Act-Replan 智能体（动态工具选择）
- **`run_workflow.py`** — 固定流程基线（`img_ann → query vote → text_ann → answer`）
- **`run_direct_qa.py`** — 不依赖外部检索服务的直接问答基线

---

## 目录

1. [核心入口](#1-核心入口)
2. [项目结构](#2-项目结构)
3. [环境要求](#3-环境要求)
4. [服务与默认端口](#4-服务与默认端口)
5. [快速开始](#5-快速开始)
6. [输入数据格式](#6-输入数据格式)
7. [运行评测](#7-运行评测)
8. [批量多模型评测](#8-批量多模型评测)
9. [输出文件](#9-输出文件)
10. [关键环境变量](#10-关键环境变量)
11. [机制说明](#11-机制说明)
12. [常见问题](#12-常见问题)

---

## 1. 核心入口

| 入口文件 | 作用 | 适用场景 |
| :--- | :--- | :--- |
| `run_agent.py` | 动态规划 + 多工具调用 | 主评测流程 |
| `run_workflow.py` | 固定工作流 | 稳定对照基线 |
| `run_direct_qa.py` | 仅模型直接作答 | 无检索快速基线 |
| `run_benchmark.sh` | 多模型批量评测脚本 | 批处理实验 |

---

## 2. 项目结构

```text
.
├── run_agent.py
├── run_workflow.py
├── run_benchmark.sh
├── run_direct_qa.py
├── qa_agent/
│   ├── config.py
│   ├── graph.py
│   ├── llm_client.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── retrieval.py
│   ├── schema.py
│   └── tool_skills.py
├── tools/
│   ├── img_emb_server.py
│   ├── text_emb_server.py
│   ├── multimodal_emb_server.py
│   ├── bm25_server.py
│   ├── kn_lookup_server.py
│   └── *_client.sh / *_test.py
├── skills/
└── data/
```

---

## 3. 环境要求

- Python `3.10`（推荐）
- Linux + NVIDIA GPU（vLLM 与 embedding 服务默认 GPU 推理）
- 已正确安装 CUDA / PyTorch / vLLM

```bash
pip install -r requirements.txt
```

> **注意：** `requirements.txt` 中可能包含机器相关 wheel 路径（例如本地构建路径）。如与当前环境不一致，请替换为你本机可用版本。

---

## 4. 服务与默认端口

默认配置位于 `qa_agent/config.py`。

| 服务 | 默认地址 | 说明 |
| :--- | :--- | :--- |
| LLM API | `http://127.0.0.1:8000/v1` | OpenAI 兼容接口 |
| `img_ann` | `http://127.0.0.1:8001/img_ann` | 图像召回 |
| `kn_lookup` | `http://127.0.0.1:8002/kn_lookup` | 知识补全 |
| `multimodal_ann` | `http://127.0.0.1:8003/multimodal_ann` | 图文联合召回 |
| `text_ann` | `http://127.0.0.1:8004/text_ann` | 文本召回 |
| `bm25_ann` | `http://127.0.0.1:8005/bm25_ann` | 关键词召回 |

---

## 5. 快速开始

### 5.1 启动 vLLM

```bash
vllm serve <YOUR_VLM_MODEL> --host 0.0.0.0 --port 8000
```

或参考辅助脚本：

```bash
bash tools/vllm_serve.sh
```

### 5.2 启动 ANN / 知识服务

**`text_ann`**

```bash
export TEXT_EMB_MODEL_PATH=/path/to/text-embedding-model
export TEXT_ANN_PATH=/path/to/text_ann_corpus.jsonl
python tools/text_emb_server.py
```

**`img_ann`**

```bash
export IMG_EMB_MODEL_PATH=/path/to/image-backbone
export IMG_EMB_CKPT_PATH=/path/to/image-ckpt.pt
export IMG_ANN_PATH=/path/to/image_ann_pool.jsonl
python tools/img_emb_server.py
```

**`multimodal_ann`**

```bash
export MULTIMODAL_EMB_MODEL_PATH=/path/to/multimodal-embedding-model
export MULTIMODAL_ANN_PATH=/path/to/query_multimodal.final.jsonl
python tools/multimodal_emb_server.py
```

**`kn_lookup`**

```bash
python tools/kn_lookup_server.py
```

自定义知识文件：

```bash
KN_FILES=/path/a.jsonl:/path/b.jsonl python tools/kn_lookup_server.py
```

**`bm25_ann`**（可选）

```bash
export BM25_DATA_PATH=/path/to/corpus.jsonl
python tools/bm25_server.py
```

---

## 6. 输入数据格式

三条入口均接受 JSONL 格式输入，每行结构如下：

```json
{
  "query": "可选字段，direct_qa --use_kn 时会用到",
  "img": "/path/to/image.jpg",
  "qa": {
    "question": "题目",
    "options": ["选项A", "选项B", "选项C", "选项D"],
    "answer": "选项A"
  }
}
```

> **建议：**
> - `qa.answer` 使用 `A/B/C/D` 或完整选项文本均可。
> - 通常建议显式传 `--input`，避免依赖默认路径。

基准数据集已发布于 [🤗 SVFSearchData](https://huggingface.co/datasets/svfsearch/SVFSearchData)。

---

## 7. 运行评测

### 7.1 Plan-Act-Replan Agent（主流程）

```bash
python run_agent.py \
  --input /path/to/benchmark.jsonl \
  --output outputs/predictions.jsonl \
  --answer-sheet outputs/answer_sheet.jsonl \
  --stats outputs/stats.json \
  --log-file outputs/run.log \
  --llm-model <YOUR_SERVED_MODEL_NAME> \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-api-key EMPTY \
  --text-ann-url http://127.0.0.1:8004/text_ann \
  --img-ann-url http://127.0.0.1:8001/img_ann \
  --multimodal-ann-url http://127.0.0.1:8003/multimodal_ann
```

### 7.2 固定 Workflow 基线

```bash
python run_workflow.py \
  --input /path/to/benchmark.jsonl \
  --output outputs/workflow_predictions.jsonl \
  --answer-sheet outputs/workflow_answer_sheet.jsonl \
  --stats outputs/workflow_stats.json \
  --log-file outputs/workflow_run.log \
  --llm-model <YOUR_SERVED_MODEL_NAME> \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-api-key EMPTY \
  --text-ann-url http://127.0.0.1:8004/text_ann \
  --img-ann-url http://127.0.0.1:8001/img_ann
```

### 7.3 Direct QA 基线

```bash
python run_direct_qa.py \
  --input /path/to/benchmark.jsonl \
  --output outputs/direct_qa.jsonl \
  --model <YOUR_LOCAL_MODEL_OR_HF_PATH>
```

可选 `--use_kn`：
- 会读取当前工作目录下 `query_rag_kn_part_1.jsonl` 与 `qwen_rag_kn_part_2.jsonl`。
- 若这两个文件不在当前目录，请调整脚本或建立软链接。

---

## 8. 批量多模型评测

`run_benchmark.sh` 会按 `MODELS` 列表逐个执行：

1. 启动 vLLM
2. 运行 `run_agent.py` 或 `run_workflow.py`
3. 关闭 vLLM 并清理 GPU 进程

```bash
# 使用 agent 流程
RUNNER=agent INPUT_PATH=/path/to/benchmark.jsonl bash run_benchmark.sh

# 使用 workflow 流程
RUNNER=workflow INPUT_PATH=/path/to/benchmark.jsonl bash run_benchmark.sh

# 演习模式（不实际执行）
DRY_RUN=1 bash run_benchmark.sh
```

执行前请先修改 `run_benchmark.sh` 中与机器相关的配置：

- `MODELS` — 模型名、TP 大小、显存占比
- `INPUT_PATH` — 默认路径可能与你的数据集不匹配
- `CUDA_VISIBLE_DEVICES` — 按实际 GPU 拓扑调整

---

## 9. 输出文件

| 文件 | 内容 |
| :--- | :--- |
| `predictions.jsonl` | 精简预测结果 |
| `answer_sheet.jsonl` | 调试详情（`route` / `evidence` / `trace` / `raw_output`） |
| `stats.json` | 汇总指标（accuracy、工具调用次数、时延等） |
| `run.log` | 运行日志（已过滤 base64 图像大字段） |

---

## 10. 关键环境变量

常见变量（详见 `qa_agent/config.py`）：

| 变量 | 说明 |
| :--- | :--- |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | LLM 服务配置 |
| `TEXT_ANN_URL`, `IMG_ANN_URL`, `MULTIMODAL_ANN_URL`, `BM25_ANN_URL` | ANN 服务地址 |
| `ANN_TOPK`, `ANN_TIMEOUT` | 检索参数 |
| `MAX_PLAN_ROUNDS`, `PLAN_MAX_ATTEMPTS`, `ANSWER_MAX_ATTEMPTS` | 智能体循环上限 |
| `KN_LOOKUP_URL`, `KN_LOOKUP_TIMEOUT` | 知识查询服务 |
| `IMG_ANN_KN_TOP_QUERIES`, `IMG_ANN_KN_SELECT_MODE` | `majority` 或 `llm` |
| `DEBUG` | 开启详细调试输出 |

---

## 11. 机制说明

- `skills/*/SKILL.md` 会在 `run_agent.py` 的规划阶段注入 prompt。
- `kn_lookup` 会在 `img_ann` 后自动触发，无需 planner 手工调用。
- `bm25_ann` 由 planner 生成 `bm25_query` 后调用，适用于关键词精确匹配场景。

---

## 12. 常见问题

| 问题 | 解决方案 |
| :--- | :--- |
| `FileNotFoundError: data/benckmark.jsonl` | 显式传 `--input` 并确认文件路径。 |
| vLLM 健康检查失败 | 确认 `--llm-base-url` 与服务端口是否一致。 |
| ANN 返回空结果 | 确认对应服务已启动，索引文件路径是否正确。 |
| 依赖安装失败 | 替换 `requirements.txt` 中与当前机器不兼容的本地 wheel 项。 |

---

## 引用

如果本工作对你有所帮助，请引用我们的论文：

```bibtex
@misc{mao2026svfsearchmultimodalknowledgeintensivebenchmark,
      title={SVFSearch: A Multimodal Knowledge-Intensive Benchmark for Short-Video Frame Search in the Gaming Vertical Domain}, 
      author={Lingtao Mao and Huangyu Dai and Xinyu Sun and Zihan Liang and Ben Chen and Chenyi Lei and Wenwu Ou},
      year={2026},
      eprint={2605.17946},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.17946}, 
}
```
