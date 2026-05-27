# SVFSearch

<div align="center">

[🌐 Project Page](https://svfsearch.github.io/SVFSearch-page/) &nbsp;·&nbsp; [🤗 Dataset](https://huggingface.co/datasets/svfsearch/SVFSearchData) &nbsp;·&nbsp; [中文文档](README_CN.md)

</div>

---

> **Note:** This repository is the **Plan-Act-Replan Agent** codebase in **SVFSearch**.  
> For **mmsearch-r1-game** code, see: https://github.com/SVFSearch/SVFSearch-mmsearch-r1-game

---

This repository contains the inference and evaluation code for **SVFSearch** — a multimodal retrieval-augmented QA system with dynamic tool routing. Three runnable entry points are provided:

- **`run_agent.py`** — Plan-Act-Replan agent (dynamic tool routing)
- **`run_workflow.py`** — fixed baseline workflow (`img_ann → query vote → text_ann → answer`)
- **`run_direct_qa.py`** — direct QA baseline without external retrieval services

---

## Contents

1. [Main Entrypoints](#1-main-entrypoints)
2. [Repository Layout](#2-repository-layout)
3. [Requirements](#3-requirements)
4. [Services and Default Ports](#4-services-and-default-ports)
5. [Quick Start](#5-quick-start)
6. [Input Data Format](#6-input-data-format)
7. [Run Evaluation](#7-run-evaluation)
8. [Multi-Model Batch Benchmark](#8-multi-model-batch-benchmark)
9. [Output Files](#9-output-files)
10. [Key Environment Variables](#10-key-environment-variables)
11. [Pipeline Notes](#11-pipeline-notes)
12. [FAQ](#12-faq)

---

## 1. Main Entrypoints

| Entry | Purpose | Recommended Use |
| :--- | :--- | :--- |
| `run_agent.py` | Dynamic planning + multi-tool retrieval | Primary benchmark pipeline |
| `run_workflow.py` | Fixed 4-step workflow | Stable baseline comparison |
| `run_direct_qa.py` | Direct model answering only | Fast no-retrieval baseline |
| `run_benchmark.sh` | Multi-model batch runner | Large-scale experiments |

---

## 2. Repository Layout

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

## 3. Requirements

- Python `3.10` (recommended)
- Linux + NVIDIA GPU (vLLM and embedding services are GPU-first)
- CUDA / PyTorch / vLLM properly installed

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` may include machine-specific wheel paths. Replace any incompatible entries with versions that match your own environment.

---

## 4. Services and Default Ports

Default values are defined in `qa_agent/config.py`.

| Service | Default Endpoint | Description |
| :--- | :--- | :--- |
| LLM API | `http://127.0.0.1:8000/v1` | OpenAI-compatible endpoint |
| `img_ann` | `http://127.0.0.1:8001/img_ann` | Image retrieval |
| `kn_lookup` | `http://127.0.0.1:8002/kn_lookup` | Knowledge enrichment |
| `multimodal_ann` | `http://127.0.0.1:8003/multimodal_ann` | Multimodal retrieval |
| `text_ann` | `http://127.0.0.1:8004/text_ann` | Text retrieval |
| `bm25_ann` | `http://127.0.0.1:8005/bm25_ann` | Keyword retrieval |

---

## 5. Quick Start

### 5.1 Start vLLM

```bash
vllm serve <YOUR_VLM_MODEL> --host 0.0.0.0 --port 8000
```

Or use the helper script:

```bash
bash tools/vllm_serve.sh
```

### 5.2 Start ANN / Knowledge Services

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

Custom knowledge files:

```bash
KN_FILES=/path/a.jsonl:/path/b.jsonl python tools/kn_lookup_server.py
```

**`bm25_ann`** (optional)

```bash
export BM25_DATA_PATH=/path/to/corpus.jsonl
python tools/bm25_server.py
```

---

## 6. Input Data Format

All three entry points accept JSONL input where each line has the following structure:

```json
{
  "query": "optional; used by direct_qa --use_kn",
  "img": "/path/to/image.jpg",
  "qa": {
    "question": "question text",
    "options": ["option A", "option B", "option C", "option D"],
    "answer": "option A"
  }
}
```

> **Tips:**
> - `qa.answer` can be either `A/B/C/D` or the full option text.
> - Prefer passing `--input` explicitly instead of relying on defaults.

The benchmark dataset is available at [🤗 SVFSearchData](https://huggingface.co/datasets/svfsearch/SVFSearchData).

---

## 7. Run Evaluation

### 7.1 Plan-Act-Replan Agent (Primary)

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

### 7.2 Fixed Workflow Baseline

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

### 7.3 Direct QA Baseline

```bash
python run_direct_qa.py \
  --input /path/to/benchmark.jsonl \
  --output outputs/direct_qa.jsonl \
  --model <YOUR_LOCAL_MODEL_OR_HF_PATH>
```

Optional `--use_kn`:
- Reads `query_rag_kn_part_1.jsonl` and `qwen_rag_kn_part_2.jsonl` from the current working directory.
- If these files are elsewhere, update the script or create symlinks.

---

## 8. Multi-Model Batch Benchmark

`run_benchmark.sh` executes each model in `MODELS` sequentially:

1. Start vLLM
2. Run `run_agent.py` or `run_workflow.py`
3. Stop vLLM and clean up GPU processes

```bash
# Run agent pipeline
RUNNER=agent INPUT_PATH=/path/to/benchmark.jsonl bash run_benchmark.sh

# Run workflow pipeline
RUNNER=workflow INPUT_PATH=/path/to/benchmark.jsonl bash run_benchmark.sh

# Dry run (no actual execution)
DRY_RUN=1 bash run_benchmark.sh
```

Before running, adjust the machine-specific config inside `run_benchmark.sh`:

- `MODELS` — model name, TP size, GPU memory ratio
- `INPUT_PATH` — default may not match your dataset
- `CUDA_VISIBLE_DEVICES` — match your actual GPU topology

---

## 9. Output Files

| File | Description |
| :--- | :--- |
| `predictions.jsonl` | Compact prediction results |
| `answer_sheet.jsonl` | Debug details (`route` / `evidence` / `trace` / `raw_output`) |
| `stats.json` | Aggregate metrics (accuracy, tool usage, latency, etc.) |
| `run.log` | Runtime logs (base64 image blobs are filtered out) |

---

## 10. Key Environment Variables

Common variables (see `qa_agent/config.py` for the full list):

| Variable | Description |
| :--- | :--- |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | LLM service configuration |
| `TEXT_ANN_URL`, `IMG_ANN_URL`, `MULTIMODAL_ANN_URL`, `BM25_ANN_URL` | ANN service endpoints |
| `ANN_TOPK`, `ANN_TIMEOUT` | Retrieval parameters |
| `MAX_PLAN_ROUNDS`, `PLAN_MAX_ATTEMPTS`, `ANSWER_MAX_ATTEMPTS` | Agent loop limits |
| `KN_LOOKUP_URL`, `KN_LOOKUP_TIMEOUT` | Knowledge lookup service |
| `IMG_ANN_KN_TOP_QUERIES`, `IMG_ANN_KN_SELECT_MODE` | `majority` or `llm` |
| `DEBUG` | Enable verbose debug output |

---

## 11. Pipeline Notes

- `skills/*/SKILL.md` files are injected into the planning prompt in `run_agent.py`.
- `kn_lookup` is auto-triggered after `img_ann`; no manual planner call is needed.
- `bm25_ann` is called with planner-generated `bm25_query`, useful for exact keyword matching.

---

## 12. FAQ

| Problem | Solution |
| :--- | :--- |
| `FileNotFoundError: data/benckmark.jsonl` | Pass `--input` explicitly and verify the path. |
| vLLM health check failed | Verify `--llm-base-url` matches the service port. |
| Empty ANN results | Check that the service is running and index paths are correct. |
| Dependency installation failure | Replace machine-specific wheel entries in `requirements.txt`. |

---

## Citation

If you find this work useful, please cite our paper:

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
