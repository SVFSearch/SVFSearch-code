from __future__ import annotations

import json
import logging
import os
import re
import time

_THINK_TAG_RE = re.compile(r"(?:<think>.*?</think>|</think>)", re.DOTALL | re.IGNORECASE)
from typing import Any
from typing import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

from qa_agent.config import AgentConfig
from qa_agent.llm_client import MultimodalLLMClient
from qa_agent.prompts import build_answer_prompt
from qa_agent.prompts import build_kn_query_select_prompt
from qa_agent.prompts import build_plan_prompt
from qa_agent.retrieval import ANNRetriever
from qa_agent.retrieval import KnowledgeLookupClient
from qa_agent.schema import QAPrediction
from qa_agent.schema import QAItem
from qa_agent.tool_skills import ANNToolSkills

T = TypeVar("T")

PLANNER_SYSTEM_PROMPT = (
    "你是一个多模态选择题工具规划器。"
    "你只负责判断是否需要调用ANN工具，以及选哪一个。"
    "必须只输出严格JSON。"
)

ANSWER_SYSTEM_PROMPT = (
    "你是一个多模态单选题解题助手。"
    "请优先依据图像证据，并结合召回证据作答。"
    "必须只输出严格JSON。"
)

ALLOWED_TOOLS = ["multimodal_ann", "text_ann", "img_ann", "bm25_ann"]


def _run_with_retry(call: Callable[[], T], max_attempts: int, retry_delay: float = 1.0) -> tuple[T | None, Exception | None]:
    attempts = max(1, max_attempts)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return call(), None
        except Exception as exc:
            last_exc = exc
            logger.warning("attempt %d/%d failed: %s: %s", attempt + 1, attempts, type(exc).__name__, exc)
            if attempt < attempts - 1:
                time.sleep(retry_delay)
    return None, last_exc


def _safe_json_dict(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    # Strip Qwen3-style <think>...</think> blocks before any JSON extraction.
    # When enable_thinking is ignored by the server the model may embed a large
    # chain-of-thought block before the actual JSON output; leaving those tags
    # in place causes find("{") to land inside the thinking section and the
    # subsequent rfind("}") to capture a non-JSON span.
    text = _THINK_TAG_RE.sub("", text).strip()
    stripped = text.replace("```json", "").replace("```", "").strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return default


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_answer(options: list[str], default_index: int, reason: str, raw_output: str) -> dict[str, Any]:
    if not options:
        return {"choice_index": 0, "choice": "", "rationale": reason, "raw_output": raw_output, "confidence": 0.0}
    idx = max(0, min(default_index, len(options) - 1))
    return {
        "choice_index": idx,
        "choice": options[idx],
        "rationale": reason,
        "raw_output": raw_output,
        "confidence": 0.0,
    }


def _calc_correct(gt_answer: str | None, pred_index: int, pred_answer: str, options: list[str]) -> bool | None:
    if gt_answer is None:
        return None
    gt = str(gt_answer).strip()
    if not gt:
        return None
    if gt == pred_answer:
        return True
    upper = gt.upper()
    if len(upper) == 1 and upper in {"A", "B", "C", "D", "E"}:
        return (ord(upper) - ord("A")) == pred_index
    if gt.isdigit():
        v = int(gt)
        if 0 <= v < len(options):
            return v == pred_index
        if 1 <= v <= len(options):
            return (v - 1) == pred_index
    return False


class BenchmarkAgent:
    def __init__(self, config: AgentConfig):
        if not config.llm_model:
            raise ValueError("Missing LLM model. Pass --llm-model or set LLM_MODEL.")
        self.config = config
        self.llm = MultimodalLLMClient(
            model=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            timeout=config.llm_timeout,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            max_tokens=config.llm_max_tokens,
        )
        self.retriever = ANNRetriever(
            text_ann_url=config.text_ann_url,
            img_ann_url=config.img_ann_url,
            multimodal_ann_url=config.multimodal_ann_url,
            bm25_ann_url=config.bm25_ann_url,
            timeout=config.ann_timeout,
        )
        self.ann_skills = ANNToolSkills()
        self.ann_skill_docs = self.ann_skills.render_for_prompt()
        self.kn_client = KnowledgeLookupClient(
            url=config.kn_lookup_url,
            timeout=config.kn_lookup_timeout,
        )

    def solve(self, item: QAItem) -> QAPrediction:
        question = item.question
        options = item.options
        image_path = item.image_path

        trace: list[dict[str, Any]] = []
        plan_history: list[dict[str, Any]] = []
        tool_history: list[dict[str, Any]] = []
        used_tools: list[str] = []
        evidence_blocks: list[str] = []

        trace.append(
            {
                "node": "load_skills",
                "input": {},
                "output": {
                    "ann_skills": self.ann_skills.to_trace(),
                },
                "latency_ms": 0.0,
            }
        )

        for round_idx in range(1, max(1, self.config.max_plan_rounds) + 1):
            evidence_text = "\n".join(evidence_blocks)
            plan_prompt = build_plan_prompt(
                question=question,
                options=options,
                evidence_text=evidence_text,
                tool_history=tool_history,
                used_tools=used_tools,
                round_idx=round_idx,
                max_rounds=self.config.max_plan_rounds,
                ann_skill_docs=self.ann_skill_docs,
            )

            started = time.perf_counter()
            plan_raw, plan_err = _run_with_retry(
                lambda: self._chat(
                    image_path=image_path,
                    user_prompt=plan_prompt,
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                ),
                self.config.plan_max_attempts,
            )
            plan_ms = (time.perf_counter() - started) * 1000
            if plan_raw and len(plan_raw) > 8000:
                logger.warning(
                    "plan response unusually large (%d chars, %.1fs) for item %s round %d"
                    " — possible thinking-token leak; <think> blocks will be stripped.",
                    len(plan_raw), plan_ms / 1000, item.item_id, round_idx,
                )
            decision = self._normalize_plan_decision(
                plan_raw=plan_raw,
                plan_err=plan_err,
                used_tools=used_tools,
                round_idx=round_idx,
            )
            plan_history.append(decision)
            trace.append(
                {
                    "node": "plan",
                    "input": {
                        "round_idx": round_idx,
                        "evidence": evidence_text,
                        "used_tools": used_tools,
                    },
                    "output": {
                        "decision": decision,
                        "plan_raw": plan_raw if self.config.debug else "<hidden>",
                    },
                    "latency_ms": round(plan_ms, 3),
                }
            )

            selected_tool = str(decision.get("selected_tool", "none"))
            if bool(decision.get("can_answer_now", False)) or selected_tool == "none":
                break

            tool_started = time.perf_counter()
            recall = self.retriever.call(
                tool=selected_tool,
                question=question,
                image_path=image_path,
                top_k=self.config.ann_topk,
                bm25_query=str(decision.get("bm25_query", "")) or None,
            )
            tool_ms = (time.perf_counter() - tool_started) * 1000
            evidence = self.retriever.extract_effective_info(recall, max_items=self.config.max_evidence_items)
            evidence_blocks.append(evidence)
            if selected_tool not in used_tools:
                used_tools.append(selected_tool)

            obs = {
                "round": round_idx,
                "tool": selected_tool,
                "ok": recall.error is None,
                "result_count": len(recall.records),
                "error": recall.error,
            }
            tool_history.append(obs)
            act_input: dict[str, Any] = {"round_idx": round_idx, "tool": selected_tool}
            if selected_tool == "bm25_ann":
                act_input["bm25_query"] = str(decision.get("bm25_query", ""))
            trace.append(
                {
                    "node": "act",
                    "input": act_input,
                    "output": {
                        "observation": obs,
                        "raw_response": recall.raw_response if self.config.debug else "<hidden>",
                    },
                    "latency_ms": round(tool_ms, 3),
                }
            )

            # ------------------------------------------------------------------
            # Knowledge enrichment: auto-triggered after img_ann
            # ------------------------------------------------------------------
            if selected_tool == "img_ann" and recall.records:
                kn_started = time.perf_counter()
                kn_evidence, kn_trace = self._enrich_with_knowledge(
                    records=recall.records,
                    question=question,
                    options=options,
                    image_path=image_path,
                )
                kn_ms = (time.perf_counter() - kn_started) * 1000
                if kn_evidence:
                    evidence_blocks.append(kn_evidence)
                trace.append(
                    {
                        "node": "kn_enrich",
                        "input": {"round_idx": round_idx, "mode": self.config.img_ann_kn_select_mode},
                        "output": kn_trace if self.config.debug else {"evidence_added": bool(kn_evidence)},
                        "latency_ms": round(kn_ms, 3),
                    }
                )

        final_evidence = "\n".join(evidence_blocks)
        answer_prompt = build_answer_prompt(
            question=question,
            options=options,
            evidence_text=final_evidence,
        )
        answer_started = time.perf_counter()
        answer_raw, answer_err = _run_with_retry(
            lambda: self._chat(
                image_path=image_path,
                user_prompt=answer_prompt,
                system_prompt=ANSWER_SYSTEM_PROMPT,
            ),
            self.config.answer_max_attempts,
        )
        answer_ms = (time.perf_counter() - answer_started) * 1000

        if answer_raw is None:
            err_text = f"{type(answer_err).__name__}: {answer_err}" if answer_err else "unknown"
            final = _fallback_answer(options, self.config.default_choice_index, f"LLM answer failed: {err_text}", "")
            raw_output = ""
        else:
            if len(answer_raw) > 8000:
                logger.warning(
                    "answer response unusually large (%d chars, %.1fs) for item %s"
                    " — possible thinking-token leak; <think> blocks will be stripped.",
                    len(answer_raw), answer_ms / 1000, item.item_id,
                )
            raw_output = answer_raw
            final = self._parse_answer(text=answer_raw, options=options)

        trace.append(
            {
                "node": "answer",
                "input": {
                    "plan_history": plan_history,
                    "tool_history": tool_history,
                    "evidence": final_evidence,
                },
                "output": {
                    "answer": final,
                    "raw_output": raw_output if self.config.debug else "<hidden>",
                },
                "latency_ms": round(answer_ms, 3),
            }
        )

        pred_index = int(final["choice_index"])
        pred_answer = str(final["choice"])
        is_correct = _calc_correct(item.gt_answer, pred_index, pred_answer, options)

        return QAPrediction(
            item_id=item.item_id,
            image_path=item.image_path,
            question=item.question,
            options=item.options,
            pred_index=pred_index,
            pred_answer=pred_answer,
            rationale=str(final.get("rationale", "")),
            raw_output=str(final.get("raw_output", raw_output)),
            gt_answer=item.gt_answer,
            is_llm_correct=is_correct,
            route={
                "mode": "plan-act-replan",
                "rounds": len(plan_history),
                "used_tools": used_tools,
            },
            plan=[
                f"round_{i+1}: can_answer_now={x.get('can_answer_now')} selected_tool={x.get('selected_tool')}"
                for i, x in enumerate(plan_history)
            ],
            evidence=final_evidence,
            trace=trace,
        )

    def _chat(self, image_path: str, user_prompt: str, system_prompt: str) -> str:
        if image_path and os.path.exists(image_path):
            return self.llm.chat_with_image(
                image_path=image_path,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
        return self.llm.chat_text(
            user_prompt=user_prompt + "\n\n图像文件不可用，请基于文本信息作答。",
            system_prompt=system_prompt,
        )

    def _normalize_plan_decision(
        self,
        plan_raw: str | None,
        plan_err: Exception | None,
        used_tools: list[str],
        round_idx: int,
    ) -> dict[str, Any]:
        parsed = _safe_json_dict(plan_raw or "") if plan_raw else None
        if parsed is None:
            fallback_tool = self._next_tool(used_tools)
            if fallback_tool == "none" or round_idx >= self.config.max_plan_rounds:
                return {
                    "can_answer_now": True,
                    "selected_tool": "none",
                    "reason": f"plan parse failed: {type(plan_err).__name__ if plan_err else 'invalid_json'}",
                    "confidence": 0.0,
                }
            return {
                "can_answer_now": False,
                "selected_tool": fallback_tool,
                "reason": f"plan parse failed, fallback tool {fallback_tool}",
                "confidence": 0.0,
            }

        can_answer_now = _to_bool(parsed.get("can_answer_now"), default=False)
        selected_tool = str(parsed.get("selected_tool", "none")).strip().lower()
        reason = str(parsed.get("reason", "")).strip() or "planner decision"
        confidence = _to_float(parsed.get("confidence"))

        if can_answer_now:
            return {
                "can_answer_now": True,
                "selected_tool": "none",
                "reason": reason,
                "confidence": confidence,
            }

        if selected_tool not in ALLOWED_TOOLS:
            selected_tool = self._next_tool(used_tools)
        if not self.config.allow_tool_repeat and selected_tool in used_tools:
            selected_tool = self._next_tool(used_tools)
        if selected_tool == "none" or round_idx >= self.config.max_plan_rounds:
            return {
                "can_answer_now": True,
                "selected_tool": "none",
                "reason": reason + " | no more useful tool / reach max rounds",
                "confidence": confidence,
            }
        return {
            "can_answer_now": False,
            "selected_tool": selected_tool,
            "reason": reason,
            "confidence": confidence,
        }

    def _next_tool(self, used_tools: list[str]) -> str:
        if self.config.allow_tool_repeat:
            # Round-robin through tools based on how many times they've been used
            counts = {t: used_tools.count(t) for t in ALLOWED_TOOLS}
            return min(ALLOWED_TOOLS, key=lambda t: counts[t])
        for t in ALLOWED_TOOLS:
            if t not in used_tools:
                return t
        return "none"

    def _parse_answer(self, text: str, options: list[str]) -> dict[str, Any]:
        parsed = _safe_json_dict(text) or {}

        idx: int | None = None
        raw_idx = parsed.get("choice_index")
        if isinstance(raw_idx, int):
            idx = raw_idx
        elif isinstance(raw_idx, float) and raw_idx.is_integer():
            idx = int(raw_idx)
        elif isinstance(raw_idx, str) and raw_idx.strip().isdigit():
            idx = int(raw_idx.strip())

        if idx is None:
            raw_choice = parsed.get("choice")
            if isinstance(raw_choice, str):
                raw_choice_stripped = raw_choice.strip()
                for i, option in enumerate(options):
                    if option == raw_choice_stripped:
                        idx = i
                        break

        if idx is None:
            for i, option in enumerate(options):
                if option and option in text:
                    idx = i
                    break

        if idx is None:
            upper = text.upper()
            for letter, i in [("A", 0), ("B", 1), ("C", 2), ("D", 3), ("E", 4)]:
                if re.search(rf"\b{letter}\b", upper):
                    idx = i
                    break

        if idx is None:
            idx = self.config.default_choice_index

        if options:
            idx = max(0, min(idx, len(options) - 1))
            choice = options[idx]
        else:
            idx = 0
            choice = ""
        rationale = parsed.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = "答案由模型输出解析得到。"
        confidence = _to_float(parsed.get("confidence"))
        return {
            "choice_index": idx,
            "choice": choice,
            "rationale": rationale.strip(),
            "raw_output": text,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Knowledge enrichment helpers
    # ------------------------------------------------------------------

    def _enrich_with_knowledge(
        self,
        records: list[dict[str, Any]],
        question: str,
        options: list[str],
        image_path: str,
    ) -> tuple[str, dict[str, Any]]:
        """Select queries from img_ann records, call kn_lookup, return evidence.

        Returns ``(evidence_text, trace_dict)``.  ``evidence_text`` is empty
        string when nothing useful was found (caller should check before
        appending to evidence_blocks).
        """
        top_n = max(1, self.config.img_ann_kn_top_queries)
        mode = self.config.img_ann_kn_select_mode

        # Step 1 – collect all unique query candidates from records
        candidate_queries: list[str] = list(
            dict.fromkeys(  # preserve order, deduplicate
                str(r.get("query", "")).strip()
                for r in records
                if str(r.get("query", "")).strip()
            )
        )
        if not candidate_queries:
            return "", {"selected": [], "knowledge_found": False, "reason": "no query in records"}

        # Step 2 – select which queries to look up
        if mode == "llm" and len(candidate_queries) > top_n:
            selected = self._select_queries_by_llm(
                candidate_queries=candidate_queries,
                question=question,
                options=options,
                image_path=image_path,
                top_n=top_n,
            )
        else:
            selected = KnowledgeLookupClient.select_by_majority(records, top_n)

        if not selected:
            selected = candidate_queries[:top_n]

        # Step 3 – lookup knowledge
        knowledge = self.kn_client.lookup(selected)
        evidence = self.kn_client.format_knowledge(
            knowledge, max_content_chars=self.config.kn_max_content_chars
        )

        trace: dict[str, Any] = {
            "mode": mode,
            "candidates": candidate_queries,
            "selected": selected,
            "found_queries": list(knowledge.keys()),
            "knowledge_found": bool(knowledge),
        }
        return evidence, trace

    def _select_queries_by_llm(
        self,
        candidate_queries: list[str],
        question: str,
        options: list[str],
        image_path: str,
        top_n: int,
    ) -> list[str]:
        """Ask the LLM to pick the most relevant queries from candidates."""
        KN_SELECT_SYSTEM = (
            "你是一个查询相关性判断器。"
            "从候选列表中选出对解题最有帮助的知识查询，输出严格JSON。"
        )
        prompt = build_kn_query_select_prompt(
            question=question,
            options=options,
            candidate_queries=candidate_queries,
            top_n=top_n,
        )
        raw, err = _run_with_retry(
            lambda: self._chat(
                image_path=image_path,
                user_prompt=prompt,
                system_prompt=KN_SELECT_SYSTEM,
            ),
            max_attempts=2,
        )
        if not raw:
            logger.warning("LLM query selection failed: %s", err)
            return KnowledgeLookupClient.select_by_majority(
                [{"query": q} for q in candidate_queries], top_n
            )

        parsed = _safe_json_dict(raw) or {}
        selected_raw = parsed.get("selected", [])
        if not isinstance(selected_raw, list):
            return KnowledgeLookupClient.select_by_majority(
                [{"query": q} for q in candidate_queries], top_n
            )

        # Keep only valid candidates (prevent hallucinated queries)
        valid = [s for s in selected_raw if isinstance(s, str) and s.strip() in candidate_queries]
        return valid[:top_n] if valid else KnowledgeLookupClient.select_by_majority(
            [{"query": q} for q in candidate_queries], top_n
        )
