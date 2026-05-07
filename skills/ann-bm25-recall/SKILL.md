---
name: ann-bm25-recall
description: BM25 keyword retrieval tool skill. Use when the question contains specific names, terms or IDs that should be matched exactly in the knowledge base, and semantic embedding may miss them.
---

# ann-bm25-recall

## Tool Name

`bm25_ann`

## When to use

- 题目包含**专有名词、角色名、技能名、地图名、道具名**等需要精确匹配的关键词。
- 语义向量召回（`text_ann` / `multimodal_ann`）结果不相关，或分数偏低时。
- 你希望用**关键词组合**直接锁定知识片段，而非依赖语义相似度。
- 图像已能确认某个具体名称，但仍需其文字描述或规则说明。

## 特别说明：query 由大模型生成

与其他 ANN 工具直接使用原始问题作为查询不同，`bm25_ann` 的查询由 **planner 在规划时
生成**，写入 `bm25_query` 字段。  
Query 应是**简短的关键词组合**，而非完整问句，例如：

| 原始问题 | 好的 bm25_query |
|---|---|
| 原神中雷电将军的终结技叫什么名字？ | `原神 雷电将军 终结技 名称` |
| 火影忍者手游里鸣人有哪些觉醒？ | `火影忍者手游 漩涡鸣人 觉醒` |
| CS:GO Dust II 的A点炸弹区在哪里？ | `CS:GO Dust2 A点 炸弹区 位置` |

## Request

- Method: `POST`
- Endpoint: `/bm25_ann`
- JSON payload:

```json
{
  "query": "原神 雷电将军 终结技 名称",
  "top_k": 5
}
```

> `query` 填入 planner 生成的 `bm25_query`，而非原始问题。

## Response

- 读取 `scores` 字段作为召回结果列表。
- 常见字段：`title`、`content`、`chunk_id`、`score`（BM25 分数，越高越相关）。
- 若所有文档均无词语重叠，`scores` 为空列表。

## Decision Hints

- 若问题核心是**精确关键词命中**，优先本工具。
- 若问题需要**语义相似度**或**跨语言理解**，优先 `text_ann` 或 `multimodal_ann`。
- 本工具与 `text_ann` 互补：两者同时用可提升召回覆盖度。
