---
name: ann-text-recall
description: Text ANN retrieval tool skill. Use when the question needs external textual knowledge, concept definitions, or rule details not directly visible in the image.
---

# ann-text-recall

## Tool Name

`text_ann`

## When to use

- 题目主要是概念、机制、背景知识，图像本身无法直接给出完整答案。
- 需要通过文本语义找相近知识片段（如术语解释、规则描述）。
- 你已经看过图像，但仍存在明显知识缺口。

## Request

- Method: `POST`
- Endpoint: `/text_ann`
- JSON payload:

```json
{
  "query": "天涯明月刀 一代宗师毒圈 基本规则",
  "top_k": 5
}
```

## Response

- 读取 `scores` 字段作为召回结果列表。
- 常见字段：`title`、`content`、`score`、`chunk_id`。

## Decision Hints

- 若问题核心是“文本知识匹配”，优先尝试本工具。
- 若问题核心是“画面视觉细节”，优先考虑 `img_ann` 或 `multimodal_ann`。
