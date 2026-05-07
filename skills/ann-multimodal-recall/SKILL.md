---
name: ann-multimodal-recall
description: Multimodal ANN retrieval tool skill. Use when both text intent and image evidence are needed together to recall the most relevant knowledge.
---

# ann-multimodal-recall

## Tool Name

`multimodal_ann`

## When to use

- 问题需要同时结合“图像内容 + 查询文本”才能准确检索。
- 单独文本 ANN 或图像 ANN 的结果都不稳定时。
- 你希望一次召回就兼顾语义和视觉相关性。

## Request

- Method: `POST`
- Endpoint: `/multimodal_ann`
- JSON payload:

```json
{
  "query": "天涯明月刀 一代宗师毒圈 基本规则",
  "image_path": "/path/to/image.jpg",
  "top_k": 5
}
```

## Response

- 读取 `scores` 字段作为召回结果列表。
- 常见字段：`title`、`content`、`source_query`、`source_best_img`、`score`。

## Decision Hints

- 当你不确定该选文本还是图像召回时，优先尝试本工具。
- 若仍无法回答，可再补一个 `text_ann` 或 `img_ann`。
