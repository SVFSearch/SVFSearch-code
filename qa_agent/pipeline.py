from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from qa_agent.graph import BenchmarkAgent
from qa_agent.schema import QAPrediction
from qa_agent.schema import QAItem


def load_benchmark(path: str) -> list[QAItem]:
    items: list[QAItem] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            row = json.loads(line)
            qa = row.get("qa", {})
            question = qa.get("question")
            options = qa.get("options")
            image_path = row.get("img")
            if not isinstance(question, str) or not question.strip():
                print(f"[warn] line {idx}: missing question, skipped")
                skipped += 1
                continue
            if not isinstance(options, list) or not options:
                print(f"[warn] line {idx}: missing options, skipped")
                skipped += 1
                continue
            if not isinstance(image_path, str) or not image_path.strip():
                print(f"[warn] line {idx}: missing image_path, skipped")
                skipped += 1
                continue

            items.append(
                QAItem(
                    item_id=str(idx),
                    image_path=image_path.strip(),
                    question=question.strip(),
                    options=[str(x) for x in options],
                    gt_answer=qa.get("answer"),
                )
            )
    if skipped:
        print(f"[warn] {skipped} item(s) skipped due to missing fields.")
    return items


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
        "plan": prediction.plan,
        "evidence": prediction.evidence,
        "trace": prediction.trace,
    }


def _fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "ETA: --"
    avg = elapsed / done
    remaining = avg * (total - done)
    m, s = divmod(int(remaining), 60)
    h, m = divmod(m, 60)
    if h:
        return f"ETA: {h}h{m:02d}m"
    return f"ETA: {m}m{s:02d}s"


def run_benchmark(
    agent: BenchmarkAgent,
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
            pred = agent.solve(item)
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
            print(
                f"[{i}/{total}] {eta} | {item_ms/1000:.1f}s | "
                f"acc={acc_str} | "
                f"gt={pred.gt_answer} pred={pred.pred_answer} ok={pred.is_llm_correct} | "
                f"tools={pred.route.get('used_tools', []) if pred.route else []}"
            )
    finally:
        if pred_handle is not None:
            pred_handle.close()
        if answer_handle is not None:
            answer_handle.close()
    return predictions


def _running_accuracy(predictions: list[QAPrediction]) -> float | None:
    scores = [p.is_llm_correct for p in predictions if p.is_llm_correct is not None]
    if not scores:
        return None
    return sum(1 for x in scores if x) / len(scores)


def compute_accuracy(predictions: list[QAPrediction]) -> float | None:
    return _running_accuracy(predictions)


def compute_stats(predictions: list[QAPrediction]) -> dict[str, Any]:
    """Compute a summary statistics dict from all predictions."""
    total = len(predictions)
    with_gt = [p for p in predictions if p.is_llm_correct is not None]
    correct = sum(1 for p in with_gt if p.is_llm_correct)

    tool_counter: Counter[str] = Counter()
    rounds_list: list[int] = []
    kn_hit = 0
    kn_total = 0
    latencies_ms: list[float] = []

    for p in predictions:
        if p.route:
            for t in p.route.get("used_tools", []):
                tool_counter[t] += 1
            r = p.route.get("rounds")
            if isinstance(r, int):
                rounds_list.append(r)

        if p.trace:
            for node in p.trace:
                if node.get("node") == "kn_enrich":
                    kn_total += 1
                    out = node.get("output", {})
                    if out.get("knowledge_found") or out.get("evidence_added"):
                        kn_hit += 1
                # sum up latency across all nodes
                ms = node.get("latency_ms")
                if isinstance(ms, (int, float)):
                    latencies_ms.append(ms)

    # Per-question total latency (sum of all node latencies per prediction)
    per_q_latency: list[float] = []
    for p in predictions:
        if p.trace:
            q_ms = sum(
                n.get("latency_ms", 0) for n in p.trace if isinstance(n.get("latency_ms"), (int, float))
            )
            per_q_latency.append(q_ms)

    avg_q_ms = sum(per_q_latency) / len(per_q_latency) if per_q_latency else 0.0

    return {
        "total_items": total,
        "items_with_gt": len(with_gt),
        "correct": correct,
        "accuracy": round(correct / len(with_gt), 4) if with_gt else None,
        "tool_usage_count": dict(tool_counter),
        "kn_lookup_triggered": kn_total,
        "kn_lookup_hit": kn_hit,
        "kn_lookup_hit_rate": round(kn_hit / kn_total, 4) if kn_total else None,
        "avg_plan_rounds": round(sum(rounds_list) / len(rounds_list), 2) if rounds_list else None,
        "avg_latency_ms_per_item": round(avg_q_ms, 1),
        "total_latency_ms": round(sum(per_q_latency), 1),
    }
