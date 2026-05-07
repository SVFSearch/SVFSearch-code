---
name: ann-image-recall
description: Image ANN retrieval tool skill. Use when the answer depends on visual similarity, UI style, iconography, or scene details from the image.
---

# ann-image-recall

## Tool Name

`img_ann`

## When to use

- 题目与图像中的视觉元素高度相关（界面样式、图标、角色外观、场景布局）。
- 需要在图库中找“视觉上最像”的样本作为参考。
- 文本召回不足以解释图像细节。

## Request

- Method: `POST`
- Endpoint: `/img_ann`
- JSON payload:

```json
{
  "img": "/path/to/image.jpg",
  "top_k": 5
}
```

## Response

- 读取 `scores` 字段作为召回结果列表。
- 常见字段：`img`、`query`、`pid`、`score`。

## Knowledge Enrichment（自动触发，无需额外规划）

调用 `img_ann` 后，pipeline 会自动执行以下步骤：

1. 从召回 top-N 结果中提取每条记录的 `query` 字段（该字段为与图库图片关联的知识查询文本）。
2. 通过策略（众数投票或 LLM 相关性判断）选出最有价值的 query。
3. 调用内部 `kn_lookup` 服务，检索对应详细知识内容（游戏地图介绍、武器说明、规则解释等）。
4. 将知识内容追加至证据，供答题器使用。

因此，`img_ann` 不仅返回视觉相似图像，**还会自动补充相关知识文本**。

## Decision Hints

- 当”看图找相似”是主需求时优先本工具。
- 若题目同时依赖文本语义和图像语义，可优先尝试 `multimodal_ann`。
- 当题目涉及游戏机制、地图规则、武器参数等知识型内容，且图像有明确游戏/场景标识时，`img_ann` + 自动知识补充的组合效果更优。
