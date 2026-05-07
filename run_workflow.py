"""Fixed 4-step retrieval pipeline (no LLM planning).

Pipeline:
  1. img_ann recall (top-5)
  2. Majority-vote on `query` fields → voted_query (fallback: highest-score record)
  3. text_ann recall with "<voted_query> <question>"
  4. LLM answer with image + combined evidence
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from qa_agent.config import AgentConfig
from qa_agent.llm_client import MultimodalLLMClient
from qa_agent.pipeline import _fmt_eta
from qa_agent.pipeline import _running_accuracy
from qa_agent.pipeline import compute_accuracy
from qa_agent.pipeline import load_benchmark
from qa_agent.prompts import build_answer_prompt
from qa_agent.retrieval import ANNRetriever
from qa_agent.schema import QAItem
from qa_agent.schema import QAPrediction

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"(?:<think>.*?</think>|</think>)", re.DOTALL | re.IGNORECASE)

ANSWER_SYSTEM_PROMPT = (
    "你是一个多模态单选题解题助手。"
    "请优先依据图像证据，并结合召回证据作答。"
    "必须只输出严格JSON。"
)


# ---------------------------------------------------------------------------
# Helpers (re-implemented standalone to avoid coupling with BenchmarkAgent)
# ---------------------------------------------------------------------------

def _safe_json_dict(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = _THINK_TAG_RE.sub("", text).strip()
    stripped = text.replace("```json", "").replace("```", "").strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start: end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _parse_answer(text: str, options: list[str], default_index: int = 0) -> dict[str, Any]:
    """Extract answer fields from LLM output text."""
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
        idx = default_index

    if options:
        idx = max(0, min(idx, len(options) - 1))
        choice = options[idx]
    else:
        idx = 0
        choice = ""

    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = "答案由模型输出解析得到。"

    return {
        "choice_index": idx,
        "choice": choice,
        "rationale": rationale.strip(),
        "raw_output": text,
        "confidence": _to_float(parsed.get("confidence")),
    }


def _vote_query(records: list[dict[str, Any]]) -> tuple[str, str]:
    """Pick the best query from img_ann records via majority voting.

    Returns ``(voted_query, vote_method)`` where vote_method is one of:
    - "majority"       : the most-common query string across all records
    - "fallback_score" : no query fields found, use the highest-score record's query
    - "none"           : no records / nothing usable
    """
    if not records:
        return "", "none"

    # Collect query strings (may be empty / missing)
    queries = [
        str(r.get("query", "")).strip()
        for r in records
        if str(r.get("query", "")).strip()
    ]

    if queries:
        counts = Counter(queries)
        winner, _ = counts.most_common(1)[0]
        return winner, "majority"

    # Fallback: use the query from the record with the highest score
    best = max(records, key=lambda r: _to_float(r.get("score")) or 0.0)
    fallback = str(best.get("query", "")).strip()
    if fallback:
        return fallback, "fallback_score"

    return "", "none"


# ---------------------------------------------------------------------------
# WorkflowPipeline
# ---------------------------------------------------------------------------

class WorkflowPipeline:
    """Fixed 4-step pipeline: img_ann → vote → text_ann → answer."""

    def __init__(self, config: AgentConfig) -> None:
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

    def solve(self, item: QAItem) -> QAPrediction:
        question = item.question
        options = item.options
        image_path = item.image_path
        trace: list[dict[str, Any]] = []
        top_k = max(1, self.config.ann_topk)

        # ------------------------------------------------------------------
        # Step 1: img_ann recall
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        img_recall = self.retriever.call(
            tool="img_ann",
            question=question,
            image_path=image_path,
            top_k=top_k,
        )
        img_ms = (time.perf_counter() - t0) * 1000

        img_evidence = self.retriever.extract_effective_info(
            img_recall, max_items=self.config.max_evidence_items
        )
        trace.append({
            "node": "img_ann",
            "input": {"image_path": image_path, "top_k": top_k},
            "output": {
                "result_count": len(img_recall.records),
                "error": img_recall.error,
                "evidence": img_evidence,
            },
            "latency_ms": round(img_ms, 3),
        })
        logger.debug("img_ann: %d records, %.1fms", len(img_recall.records), img_ms)

        # ------------------------------------------------------------------
        # Step 2: majority-vote on query field
        # ------------------------------------------------------------------
        t1 = time.perf_counter()
        voted_query, vote_method = _vote_query(img_recall.records)
        vote_ms = (time.perf_counter() - t1) * 1000

        trace.append({
            "node": "query_vote",
            "input": {"records_count": len(img_recall.records)},
            "output": {
                "voted_query": voted_query,
                "vote_method": vote_method,
            },
            "latency_ms": round(vote_ms, 3),
        })
        logger.debug("query_vote: voted_query=%r  method=%s", voted_query, vote_method)

        # ------------------------------------------------------------------
        # Step 3: text_ann recall with voted_query + question
        # ------------------------------------------------------------------
        text_query = f"{voted_query} {question}".strip() if voted_query else question
        t2 = time.perf_counter()
        text_recall = self.retriever.call(
            tool="text_ann",
            question=text_query,
            image_path=image_path,
            top_k=top_k,
        )
        text_ms = (time.perf_counter() - t2) * 1000

        text_evidence = self.retriever.extract_effective_info(
            text_recall, max_items=self.config.max_evidence_items
        )
        trace.append({
            "node": "text_ann",
            "input": {"text_query": text_query, "top_k": top_k},
            "output": {
                "result_count": len(text_recall.records),
                "error": text_recall.error,
                "evidence": text_evidence,
            },
            "latency_ms": round(text_ms, 3),
        })
        logger.debug("text_ann: %d records, %.1fms", len(text_recall.records), text_ms)

        # ------------------------------------------------------------------
        # Step 4: LLM answer with image + combined evidence
        # ------------------------------------------------------------------
        evidence_parts = [p for p in [img_evidence, text_evidence] if p]
        final_evidence = "\n".join(evidence_parts)

        answer_prompt = build_answer_prompt(
            question=question,
            options=options,
            evidence_text=final_evidence,
        )

        t3 = time.perf_counter()
        answer_raw: str | None = None
        answer_err: Exception | None = None
        for attempt in range(max(1, self.config.answer_max_attempts)):
            try:
                if image_path and os.path.exists(image_path):
                    answer_raw = self.llm.chat_with_image(
                        image_path=image_path,
                        user_prompt=answer_prompt,
                        system_prompt=ANSWER_SYSTEM_PROMPT,
                    )
                else:
                    answer_raw = self.llm.chat_text(
                        user_prompt=answer_prompt + "\n\n图像文件不可用，请基于文本信息作答。",
                        system_prompt=ANSWER_SYSTEM_PROMPT,
                    )
                break
            except Exception as exc:
                answer_err = exc
                logger.warning(
                    "answer attempt %d/%d failed: %s: %s",
                    attempt + 1, self.config.answer_max_attempts,
                    type(exc).__name__, exc,
                )
                if attempt < self.config.answer_max_attempts - 1:
                    time.sleep(1.0)

        answer_ms = (time.perf_counter() - t3) * 1000

        if answer_raw is None:
            err_text = f"{type(answer_err).__name__}: {answer_err}" if answer_err else "unknown"
            logger.error("LLM answer failed for item %s: %s", item.item_id, err_text)
            idx = max(0, min(self.config.default_choice_index, len(options) - 1))
            final = {
                "choice_index": idx,
                "choice": options[idx] if options else "",
                "rationale": f"LLM answer failed: {err_text}",
                "raw_output": "",
                "confidence": 0.0,
            }
        else:
            if len(answer_raw) > 8000:
                logger.warning(
                    "answer response unusually large (%d chars, %.1fs) for item %s"
                    " — possible thinking-token leak; <think> blocks will be stripped.",
                    len(answer_raw), answer_ms / 1000, item.item_id,
                )
            final = _parse_answer(answer_raw, options, self.config.default_choice_index)

        trace.append({
            "node": "answer",
            "input": {"evidence": final_evidence},
            "output": {
                "answer": final,
                "raw_output": answer_raw if self.config.debug else "<hidden>",
            },
            "latency_ms": round(answer_ms, 3),
        })

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
            raw_output=str(final.get("raw_output", "")),
            gt_answer=item.gt_answer,
            is_llm_correct=is_correct,
            route={
                "mode": "workflow",
                "voted_query": voted_query,
                "vote_method": vote_method,
                "img_ann_count": len(img_recall.records),
                "text_ann_count": len(text_recall.records),
            },
            evidence=final_evidence,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def _summary_record(prediction: QAPrediction) -> dict:
    return {
        "item_id": prediction.item_id,
        "img": prediction.image_path,
        "question": prediction.question,
        "options": prediction.options,
        "gt_answer": prediction.gt_answer,
        "llm_answer": prediction.pred_answer,
        "llm_choice_index": prediction.pred_index,
        "is_llm_correct": prediction.is_llm_correct,
    }


def _debug_record(prediction: QAPrediction) -> dict:
    return {
        "item_id": prediction.item_id,
        "img": prediction.image_path,
        "question": prediction.question,
        "options": prediction.options,
        "gt_answer": prediction.gt_answer,
        "llm_answer": prediction.pred_answer,
        "llm_choice_index": prediction.pred_index,
        "is_llm_correct": prediction.is_llm_correct,
        "rationale": prediction.rationale,
        "raw_output": prediction.raw_output,
        "route": prediction.route,
        "evidence": prediction.evidence,
        "trace": prediction.trace,
    }


def run_workflow(
    pipeline: WorkflowPipeline,
    items: list[QAItem],
    prediction_output_path: str | None = None,
    answer_sheet_output_path: str | None = None,
) -> list[QAPrediction]:
    predictions: list[QAPrediction] = []
    pred_handle = None
    answer_handle = None
    run_start = time.perf_counter()

    try:
        if prediction_output_path:
            out = Path(prediction_output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            pred_handle = open(out, "w", encoding="utf-8")
        if answer_sheet_output_path:
            out = Path(answer_sheet_output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            answer_handle = open(out, "w", encoding="utf-8")

        total = len(items)
        for i, item in enumerate(items, start=1):
            t0 = time.perf_counter()
            pred = pipeline.solve(item)
            item_ms = (time.perf_counter() - t0) * 1000

            predictions.append(pred)
            if pred_handle is not None:
                pred_handle.write(json.dumps(_summary_record(pred), ensure_ascii=False) + "\n")
                pred_handle.flush()
            if answer_handle is not None:
                answer_handle.write(json.dumps(_debug_record(pred), ensure_ascii=False) + "\n")
                answer_handle.flush()

            elapsed = time.perf_counter() - run_start
            eta = _fmt_eta(elapsed, i, total)
            running_acc = _running_accuracy(predictions)
            acc_str = f"{running_acc:.2%}" if running_acc is not None else "N/A"
            route = pred.route or {}
            print(
                f"[{i}/{total}] {eta} | {item_ms/1000:.1f}s | "
                f"acc={acc_str} | "
                f"gt={pred.gt_answer} pred={pred.pred_answer} ok={pred.is_llm_correct} | "
                f"vote={route.get('vote_method')} query={repr(route.get('voted_query', ''))[:40]}"
            )
    finally:
        if pred_handle is not None:
            pred_handle.close()
        if answer_handle is not None:
            answer_handle.close()

    return predictions


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_workflow_stats(predictions: list[QAPrediction]) -> dict[str, Any]:
    total = len(predictions)
    with_gt = [p for p in predictions if p.is_llm_correct is not None]
    correct = sum(1 for p in with_gt if p.is_llm_correct)

    vote_method_counts: Counter[str] = Counter()
    img_ann_counts: list[int] = []
    text_ann_counts: list[int] = []

    # Per-step latency accumulators
    step_latency: dict[str, list[float]] = {
        "img_ann": [],
        "query_vote": [],
        "text_ann": [],
        "answer": [],
    }
    per_q_latency: list[float] = []

    for p in predictions:
        route = p.route or {}
        vm = route.get("vote_method")
        if vm:
            vote_method_counts[str(vm)] += 1
        n_img = route.get("img_ann_count")
        if isinstance(n_img, int):
            img_ann_counts.append(n_img)
        n_txt = route.get("text_ann_count")
        if isinstance(n_txt, int):
            text_ann_counts.append(n_txt)

        if p.trace:
            q_total_ms = 0.0
            for node in p.trace:
                name = node.get("node", "")
                ms = node.get("latency_ms")
                if isinstance(ms, (int, float)):
                    q_total_ms += ms
                    if name in step_latency:
                        step_latency[name].append(ms)
            per_q_latency.append(q_total_ms)

    def _avg(lst: list) -> float | None:
        return round(sum(lst) / len(lst), 1) if lst else None

    return {
        "total_items": total,
        "items_with_gt": len(with_gt),
        "correct": correct,
        "accuracy": round(correct / len(with_gt), 4) if with_gt else None,
        "vote_method_distribution": dict(vote_method_counts),
        "avg_img_ann_results": _avg(img_ann_counts),
        "avg_text_ann_results": _avg(text_ann_counts),
        "avg_latency_ms_per_item": _avg(per_q_latency),
        "total_latency_ms": round(sum(per_q_latency), 1) if per_q_latency else 0.0,
        "avg_step_latency_ms": {k: _avg(v) for k, v in step_latency.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed workflow pipeline (img_ann → vote → text_ann → answer).")
    parser.add_argument("--input", default="data/benckmark.jsonl", help="Benchmark jsonl path.")
    parser.add_argument("--output", default="outputs/workflow_predictions.jsonl", help="Compact result output path.")
    parser.add_argument("--answer-sheet", default="outputs/workflow_answer_sheet.jsonl", help="Debug result output path.")
    parser.add_argument("--stats", default="outputs/workflow_stats.json", help="Stats output path.")
    parser.add_argument("--log-file", default="outputs/workflow_run.log", help="Log file path.")

    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    parser.add_argument("--text-ann-url", default=None)
    parser.add_argument("--img-ann-url", default=None)
    parser.add_argument("--ann-timeout", type=int, default=None)
    parser.add_argument("--ann-topk", type=int, default=None)
    parser.add_argument("--max-evidence-items", type=int, default=None)
    parser.add_argument("--answer-max-attempts", type=int, default=None)
    parser.add_argument("--default-choice-index", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> AgentConfig:
    config = AgentConfig.from_env()

    if args.llm_model:
        config.llm_model = args.llm_model
    if args.llm_base_url:
        config.llm_base_url = args.llm_base_url
    if args.llm_api_key:
        config.llm_api_key = args.llm_api_key
    if args.llm_timeout is not None:
        config.llm_timeout = max(1, args.llm_timeout)
    if args.temperature is not None:
        config.llm_temperature = args.temperature
    if args.top_p is not None:
        config.llm_top_p = args.top_p
    if args.max_tokens is not None:
        config.llm_max_tokens = max(1, args.max_tokens)

    if args.text_ann_url:
        config.text_ann_url = args.text_ann_url
    if args.img_ann_url:
        config.img_ann_url = args.img_ann_url
    if args.ann_timeout is not None:
        config.ann_timeout = max(1, args.ann_timeout)
    if args.ann_topk is not None:
        config.ann_topk = max(1, args.ann_topk)
    if args.max_evidence_items is not None:
        config.max_evidence_items = max(1, args.max_evidence_items)
    if args.answer_max_attempts is not None:
        config.answer_max_attempts = max(1, args.answer_max_attempts)
    if args.default_choice_index is not None:
        config.default_choice_index = args.default_choice_index
    if args.debug:
        config.debug = True

    return config


class _StripBase64Filter(logging.Filter):
    _BASE64_RE = re.compile(r"data:image/[^;'\"]+;base64,[A-Za-z0-9+/]+=*", re.ASCII)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if "base64," in msg:
                record.msg = self._BASE64_RE.sub("<img>", msg)
                record.args = None
        except Exception:
            pass
        return True


def _check_llm_server(base_url: str, api_key: str, model: str, timeout: int = 10) -> None:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    except Exception as exc:
        raise SystemExit(
            f"[ERROR] LLM server health check failed ({base_url}): {type(exc).__name__}: {exc}\n"
            "Please make sure the model server is running before launching the workflow."
        ) from exc


def _setup_logging(log_file: str, debug: bool) -> None:
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger().addFilter(_StripBase64Filter())


def main() -> None:
    args = parse_args()
    config = merge_config(args)
    if not config.llm_model:
        raise SystemExit("Missing LLM model. Pass --llm-model or set env LLM_MODEL.")

    _setup_logging(args.log_file, config.debug)
    log = logging.getLogger(__name__)

    log.info("checking LLM server at %s …", config.llm_base_url)
    _check_llm_server(config.llm_base_url, config.llm_api_key, config.llm_model)
    log.info("LLM server OK")

    for p in [args.output, args.answer_sheet, args.stats]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    log.info("=== workflow run start ===")
    log.info("input=%s", args.input)
    log.info(
        "llm_model=%s  llm_base_url=%s  temperature=%.2f  max_tokens=%d",
        config.llm_model, config.llm_base_url, config.llm_temperature, config.llm_max_tokens,
    )
    log.info(
        "ann_topk=%d  img_ann_url=%s  text_ann_url=%s",
        config.ann_topk, config.img_ann_url, config.text_ann_url,
    )

    items = load_benchmark(args.input)
    log.info("loaded %d items from %s", len(items), args.input)

    pipeline = WorkflowPipeline(config=config)
    t_start = time.perf_counter()

    predictions = run_workflow(
        pipeline=pipeline,
        items=items,
        prediction_output_path=args.output,
        answer_sheet_output_path=args.answer_sheet,
    )

    elapsed = time.perf_counter() - t_start

    acc = compute_accuracy(predictions)
    acc_str = f"{acc:.2%}" if acc is not None else "N/A (no gt)"

    stats = compute_workflow_stats(predictions)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["input"] = args.input
    stats["output"] = args.output
    stats["answer_sheet"] = args.answer_sheet
    stats["llm_model"] = config.llm_model

    stats_path = Path(args.stats)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    elapsed_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    step_lat = stats.get("avg_step_latency_ms", {})
    summary_lines = [
        "",
        "=" * 60,
        f"  Accuracy         : {acc_str}",
        f"  Total items      : {stats['total_items']}",
        f"  With ground truth: {stats['items_with_gt']}",
        f"  Correct          : {stats['correct']}",
        f"  Vote methods     : {stats['vote_method_distribution']}",
        f"  Avg img results  : {stats['avg_img_ann_results']}",
        f"  Avg txt results  : {stats['avg_text_ann_results']}",
        f"  Step latency(ms) : img={step_lat.get('img_ann')}  "
        f"vote={step_lat.get('query_vote')}  "
        f"txt={step_lat.get('text_ann')}  "
        f"ans={step_lat.get('answer')}",
        f"  Avg latency/item : {(stats['avg_latency_ms_per_item'] or 0)/1000:.1f}s",
        f"  Total elapsed    : {elapsed_str}",
        "=" * 60,
        f"  predictions   → {args.output}",
        f"  answer_sheet  → {args.answer_sheet}",
        f"  stats         → {args.stats}",
        f"  log           → {args.log_file}",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    log.info(summary)
    log.info("=== workflow run end ===")


if __name__ == "__main__":
    main()
