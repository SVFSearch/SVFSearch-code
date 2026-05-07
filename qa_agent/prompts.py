from __future__ import annotations

import json
from typing import Sequence


def format_options(options: list[str]) -> str:
    rows: list[str] = []
    for idx, option in enumerate(options):
        rows.append(f"{chr(ord('A') + idx)}. {option}")
    return "\n".join(rows)


def build_plan_prompt(
    question: str,
    options: list[str],
    evidence_text: str,
    tool_history: list[dict[str, object]],
    used_tools: list[str],
    round_idx: int,
    max_rounds: int,
    ann_skill_docs: str,
) -> str:
    return "\n".join(
        [
            "你是该 benchmark 的专用工具规划器。",
            "你只可决定：直接回答，或调用一个 ANN 工具。",
            "ANN工具说明来自 skills 文档：",
            ann_skill_docs,
            "",
            f"当前轮次: {round_idx}/{max_rounds}",
            "题目：",
            question,
            "",
            "选项：",
            format_options(options),
            "",
            f"已使用工具: {json.dumps(used_tools, ensure_ascii=False)}",
            f"工具历史: {json.dumps(tool_history, ensure_ascii=False)}",
            "",
            "当前有效证据：",
            evidence_text or "[empty]",
            "",
            "只输出 JSON：",
            "{",
            '  "can_answer_now": <true/false>,',
            '  "selected_tool": "text_ann|img_ann|multimodal_ann|bm25_ann|none",',
            '  "bm25_query": "<仅当 selected_tool=bm25_ann 时填写：用于 BM25 检索的关键词组合，省略助词，如\'原神 雷电将军 终结技\'>",',
            '  "reason": "<简短原因>",',
            '  "confidence": <0.0-1.0>',
            "},",
            "",
            "规则：",
            "- can_answer_now=true 时，selected_tool 必须为 none。",
            "- can_answer_now=false 时，selected_tool 必须是 ANN 工具之一。",
            "- 若证据已足够，优先直接回答。",
            "- 选择 bm25_ann 时，必须同时填写 bm25_query（关键词组合，不超过10词）。",
        ]
    )


def build_answer_prompt(
    question: str,
    options: list[str],
    evidence_text: str,
) -> str:
    return "\n".join(
        [
            "你是该 benchmark 的专用答题器。",
            "请基于图像和有效证据做单选题判断。",
            "仅输出 JSON，不要 markdown。",
            "",
            "题目：",
            question,
            "",
            "选项：",
            format_options(options),
            "",
            "有效证据：",
            evidence_text or "[empty]",
            "",
            "输出 JSON：",
            "{",
            '  "choice_index": <从0开始整数>,',
            '  "choice": "<选项原文>",',
            '  "rationale": "<简短理由>",',
            '  "confidence": <0.0-1.0>',
            "}",
        ]
    )


def build_kn_query_select_prompt(
    question: str,
    options: Sequence[str],
    candidate_queries: Sequence[str],
    top_n: int,
) -> str:
    """Prompt for LLM-based knowledge query selection.

    Used when ``img_ann_kn_select_mode == "llm"``.  The model should return a
    JSON object with a ``selected`` list of query strings from the candidates.
    """
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(candidate_queries))
    return "\n".join(
        [
            "你是一个知识查询选择器。",
            "以下是通过图像相似性检索得到的候选知识查询列表。",
            f"请从中选出最多 {top_n} 个对解答题目最有帮助的查询，仅输出 JSON。",
            "",
            "题目：",
            question,
            "",
            "选项：",
            format_options(list(options)),
            "",
            "候选查询列表：",
            numbered,
            "",
            "输出 JSON：",
            "{",
            '  "selected": ["<query_1>", "<query_2>"]',
            "}",
            "",
            "规则：",
            "- selected 中的字符串必须原样来自候选列表，不可修改。",
            f"- 最多选 {top_n} 个，至少选 1 个。",
            "- 若所有查询均不相关，选最接近题目主题的 1 个。",
        ]
    )
