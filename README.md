# SVFSearch-code

SVFSearch 的推理与评测代码仓库，当前项目包含三类可运行入口：

- `run_agent.py`：Plan-Act-Replan 智能体（动态选择 `img_ann / text_ann / multimodal_ann / bm25_ann`）
- `run_workflow.py`：固定流程基线（`img_ann -> query投票 -> text_ann -> answer`）
- `run_direct_qa.py`：不依赖外部检索服务的直接问答基线

## 1. 项目结构

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

## 2. 环境要求

- 推荐 Python `3.10`
- Linux + NVIDIA GPU（vLLM 与 embedding 服务默认 GPU 推理）
- 已正确安装 CUDA / PyTorch / vLLM

安装依赖：

```bash
pip install -r requirements.txt
```

注意：

- `requirements.txt` 包含机器相关路径 wheel（例如本地 `flash_attn` 路径）
- 如果你本机路径不同，请替换为你自己环境中可用的版本

## 3. 输入数据格式

`run_agent.py` / `run_workflow.py` / `run_direct_qa.py` 读取的 JSONL 每行应为：

```json
{
  "query": "可选字段，direct_qa --use_kn 时会用",
  "img": "/path/to/image.jpg",
  "qa": {
    "question": "题目",
    "options": ["选项A", "选项B", "选项C", "选项D"],
    "answer": "选项A"
  }
}
```

说明：

- `qa.answer` 建议使用完整选项文本

## 4. 服务与端口

默认配置（见 `qa_agent/config.py`）：

- LLM API：`http://127.0.0.1:8000/v1`
- `img_ann`：`http://127.0.0.1:8001/img_ann`
- `kn_lookup`：`http://127.0.0.1:8002/kn_lookup`
- `multimodal_ann`：`http://127.0.0.1:8003/multimodal_ann`
- `text_ann`：`http://127.0.0.1:8004/text_ann`
- `bm25_ann`：`http://127.0.0.1:8005/bm25_ann`

## 5. 启动服务

### 5.1 启动 vLLM

```bash
vllm serve <YOUR_VLM_MODEL> --host 0.0.0.0 --port 8000
```

或参考：

```bash
bash tools/vllm_serve.sh
```

### 5.2 启动 text_ann

```bash
export TEXT_EMB_MODEL_PATH=/path/to/text-embedding-model
export TEXT_ANN_PATH=/path/to/text_ann_corpus.jsonl
python tools/text_emb_server.py
```

### 5.3 启动 img_ann

```bash
export IMG_EMB_MODEL_PATH=/path/to/image-backbone
export IMG_EMB_CKPT_PATH=/path/to/image-ckpt.pt
export IMG_ANN_PATH=/path/to/image_ann_pool.jsonl
python tools/img_emb_server.py
```

### 5.4 启动 multimodal_ann

```bash
export MULTIMODAL_EMB_MODEL_PATH=/path/to/multimodal-embedding-model
export MULTIMODAL_ANN_PATH=/path/to/query_multimodal.final.jsonl
python tools/multimodal_emb_server.py
```

### 5.5 启动 kn_lookup

```bash
python tools/kn_lookup_server.py
```

如需自定义知识文件：

```bash
KN_FILES=/path/a.jsonl:/path/b.jsonl python tools/kn_lookup_server.py
```

### 5.6 启动 bm25_ann（可选）

```bash
export BM25_DATA_PATH=/path/to/corpus.jsonl
python tools/bm25_server.py
```

## 6. 快速验证服务

仓库已提供客户端脚本：

```bash
bash tools/text_emb_client.sh "原神 雷电将军" 5
bash tools/img_emb_client.sh /path/to/image.jpg 5
bash tools/multimodal_emb_client.sh "天涯明月刀 毒圈" /path/to/image.jpg 5
bash tools/kn_lookup_client.sh "原神 八重神子" "原神 纳西妲"
bash tools/bm25_client.sh "原神 雷电将军 终结技 名称" 5
```

## 7. 运行评测

### 7.1 Plan-Act-Replan Agent（推荐主流程）

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

- 会读取当前工作目录下 `query_rag_kn_part_1.jsonl` 与 `qwen_rag_kn_part_2.jsonl`
- 如果这两个文件不在当前目录，需自行调整脚本或软链接

## 8. 批量多模型评测

`run_benchmark.sh` 会对 `MODELS` 列表逐个执行：

1. 启动 vLLM
2. 跑 `run_agent.py` 或 `run_workflow.py`
3. 关闭 vLLM，清理 GPU 进程

示例：

```bash
RUNNER=agent INPUT_PATH=/path/to/benchmark.jsonl bash run_benchmark.sh
RUNNER=workflow INPUT_PATH=/path/to/benchmark.jsonl bash run_benchmark.sh
DRY_RUN=1 bash run_benchmark.sh
```

使用前请先修改 `run_benchmark.sh`：

- `MODELS`（模型名、TP、显存占比）
- `INPUT_PATH`（默认是 `query2QA.jsonl`，仓库中未提供）
- `CUDA_VISIBLE_DEVICES` 对应 GPU 实际拓扑

## 9. 输出文件说明

- `predictions.jsonl`：精简结果（题目、预测、正确性）
- `answer_sheet.jsonl`：调试详情（route / evidence / trace / raw_output）
- `stats.json`：汇总指标（accuracy、工具调用统计、时延等）
- `run.log`：日志（已自动清洗 base64 图像内容，避免日志膨胀）

## 10. 关键环境变量

见 `qa_agent/config.py`，常用变量：

- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- `TEXT_ANN_URL`, `IMG_ANN_URL`, `MULTIMODAL_ANN_URL`, `BM25_ANN_URL`
- `ANN_TOPK`, `ANN_TIMEOUT`
- `MAX_PLAN_ROUNDS`, `PLAN_MAX_ATTEMPTS`, `ANSWER_MAX_ATTEMPTS`
- `KN_LOOKUP_URL`, `KN_LOOKUP_TIMEOUT`
- `IMG_ANN_KN_TOP_QUERIES`, `IMG_ANN_KN_SELECT_MODE` (`majority` / `llm`)
- `DEBUG`

## 11. 机制说明

- `skills/*/SKILL.md` 会在 `run_agent.py` 的规划阶段注入 prompt
- `kn_lookup` 会在 `img_ann` 后自动触发，不需要 planner 手工调用
- `bm25_ann` 由 planner 生成 `bm25_query` 后调用，适合关键词精确命中场景

## 12. 常见问题

- `FileNotFoundError: data/benckmark.jsonl`：请显式传 `--input`，并确认路径
- vLLM 健康检查失败：确认 `--llm-base-url` 与实际服务端口一致
- ANN 返回空：优先检查对应服务是否启动、索引文件路径是否正确
- 安装依赖失败：优先处理 `requirements.txt` 中与你机器不兼容的本地 wheel 项
