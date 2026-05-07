---
name: kn-lookup
description: Knowledge base lookup service. Given a list of query strings (obtained from img_ann results), returns the corresponding long-form knowledge content from the internal knowledge base. This service is automatically triggered after img_ann and does not need to be called manually by the planner.
---

# kn-lookup

## Overview

`kn_lookup` 是一个本地知识库查询服务。它根据图像召回（`img_ann`）返回的 `query` 字段，在预构建的知识字典中检索对应的详细文本内容（如游戏规则、地图介绍、武器说明等）。

该服务在 `img_ann` 调用后**自动触发**，不需要规划器单独决策调用。

---

## 工作流程

```
img_ann 召回 → 提取 top-N query → 策略选取 → kn_lookup → 知识内容追加到证据
```

1. `img_ann` 返回 `records`，每条记录含 `query` 字段（与图库图片关联的文本查询）。
2. 从 top-N 记录中提取 `query`，通过策略（众数投票 / LLM 相关性判断）选出最有价值的查询。
3. 调用 `kn_lookup` 服务，查出对应的知识内容。
4. 将知识内容拼入证据（`evidence`）中，供答题器参考。

---

## Request

- Method: `POST`
- Endpoint: `/kn_lookup`
- JSON payload:

```json
{
  "queries": ["CS:GO Dust II", "CS:GO Mirage"]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `queries` | `list[str]` | 要查询的 query 字符串列表，来自 `img_ann` records 的 `query` 字段 |

---

## Response

```json
{
  "results": [
    {
      "query": "CS:GO Dust II",
      "found": true,
      "contents": [
        "### 游戏内容介绍：CS:GO Dust II\n\n...",
        "在CS:GO地图Dust II中，..."
      ]
    },
    {
      "query": "CS:GO Unknown Map",
      "found": false,
      "contents": []
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `found` | 是否在知识库中找到该 query |
| `contents` | 匹配的知识内容列表（同一 query 在两个文件中均有时，返回两条） |

---

## Query 选取策略

### 众数投票（默认，`majority`）
统计 `img_ann` top-N 结果中各 query 出现次数，选出频率最高的 K 个 query 进行知识查询。

- 优点：零额外 LLM 开销，快速稳定。
- 适用：图像库中同一主题出现频繁时（如同一游戏地图有多张截图）。

### LLM 相关性选取（`llm`）
将所有候选 query 连同当前题目发给 LLM，由 LLM 判断哪些 query 对解题最有帮助，输出选中的 query 列表。

- 优点：能结合题意做语义对齐，精度更高。
- 适用：图像库 query 多样，需要语义判断时。

---

## 服务启动

```bash
python tools/kn_lookup_server.py          # 默认端口 8002
KN_PORT=9002 python tools/kn_lookup_server.py
KN_FILES=tools/query_rag_kn_part_1.jsonl:tools/qwen_rag_kn_part_2.jsonl python tools/kn_lookup_server.py
```

## 健康检查

```
GET /health
→ {"status": "ok", "num_unique_queries": 22800}
```

## Decision Hints（供规划器参考）

- `kn_lookup` 由 pipeline 自动触发，无需规划器主动选择。
- 当 `img_ann` 证据中 `query` 字段有明确语义（如游戏名称、地图名、武器名）时，知识库极有可能命中，补充知识效果显著。
- 若知识库未命中（`found=false`），不影响其他证据正常使用。
