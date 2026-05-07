from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from openai import OpenAI

from qa_agent.config import AgentConfig
from qa_agent.graph import BenchmarkAgent
from qa_agent.pipeline import compute_accuracy
from qa_agent.pipeline import compute_stats
from qa_agent.pipeline import load_benchmark
from qa_agent.pipeline import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal plan-act-replan ANN QA agent.")
    parser.add_argument("--input", default="data/benckmark.jsonl", help="Benchmark jsonl path.")
    parser.add_argument("--output", default="outputs/predictions.jsonl", help="Compact result output path.")
    parser.add_argument("--answer-sheet", default="outputs/answer_sheet.jsonl", help="Debug result output path.")
    parser.add_argument("--stats", default="outputs/stats.json", help="Stats output path.")
    parser.add_argument("--log-file", default="outputs/run.log", help="Log file path (appended).")

    parser.add_argument("--llm-model", default=None, help="OpenAI-compatible model name/path.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-api-key", default=None, help="LLM api key.")
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    parser.add_argument("--text-ann-url", default=None)
    parser.add_argument("--img-ann-url", default=None)
    parser.add_argument("--multimodal-ann-url", default=None)
    parser.add_argument("--ann-timeout", type=int, default=None)
    parser.add_argument("--ann-topk", type=int, default=None)

    parser.add_argument("--max-plan-rounds", type=int, default=None)
    parser.add_argument("--plan-max-attempts", type=int, default=None)
    parser.add_argument("--answer-max-attempts", type=int, default=None)
    parser.add_argument("--allow-tool-repeat", action="store_true")
    parser.add_argument("--max-evidence-items", type=int, default=None)
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
    if args.multimodal_ann_url:
        config.multimodal_ann_url = args.multimodal_ann_url
    if args.ann_timeout is not None:
        config.ann_timeout = max(1, args.ann_timeout)
    if args.ann_topk is not None:
        config.ann_topk = max(1, args.ann_topk)

    if args.max_plan_rounds is not None:
        config.max_plan_rounds = max(1, args.max_plan_rounds)
    if args.plan_max_attempts is not None:
        config.plan_max_attempts = max(1, args.plan_max_attempts)
    if args.answer_max_attempts is not None:
        config.answer_max_attempts = max(1, args.answer_max_attempts)
    if args.allow_tool_repeat:
        config.allow_tool_repeat = True
    if args.max_evidence_items is not None:
        config.max_evidence_items = max(1, args.max_evidence_items)
    if args.default_choice_index is not None:
        config.default_choice_index = args.default_choice_index
    if args.debug:
        config.debug = True

    return config


class _StripBase64Filter(logging.Filter):
    """Replace inline base64 image data-URLs with a short placeholder.

    The OpenAI / httpcore DEBUG lines log the full request body, which embeds
    the image as a data:image/...;base64,<very long string>.  Those blobs make
    log files enormous (each request line can be ~1 MB) while carrying no
    useful diagnostic information.  This filter rewrites every log record
    before it reaches any handler.
    """

    _BASE64_RE = re.compile(
        r"data:image/[^;'\"]+;base64,[A-Za-z0-9+/]+=*",
        re.ASCII,
    )

    def filter(self, record: logging.LogRecord) -> bool:  # always allow
        try:
            # getMessage() merges record.msg % record.args into a final string.
            msg = record.getMessage()
            if "base64," in msg:
                record.msg = self._BASE64_RE.sub("<img>", msg)
                record.args = None   # args already merged above
        except Exception:
            pass
        return True


def _check_llm_server(base_url: str, api_key: str, model: str, timeout: int = 10) -> None:
    """Probe the LLM server with a minimal request and raise SystemExit on failure.

    Called once before the benchmark loop so that a down / wrong server is
    caught immediately instead of silently producing all-fallback answers.
    """
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
            "Please make sure the model server is running before launching the benchmark."
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
    # Attach the base64-stripping filter to the root logger so every handler
    # (both stream and file) sees clean records.
    logging.getLogger().addFilter(_StripBase64Filter())


def main() -> None:
    args = parse_args()
    config = merge_config(args)
    if not config.llm_model:
        raise SystemExit("Missing LLM model. Pass --llm-model or set env LLM_MODEL.")

    _setup_logging(args.log_file, config.debug)
    logger = logging.getLogger(__name__)

    # Fail fast if the LLM server is unreachable before processing any items.
    logger.info("checking LLM server at %s …", config.llm_base_url)
    _check_llm_server(config.llm_base_url, config.llm_api_key, config.llm_model)
    logger.info("LLM server OK")

    # Ensure output dirs exist
    for p in [args.output, args.answer_sheet, args.stats]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== benchmark run start ===")
    logger.info("input=%s", args.input)
    logger.info(
        "llm_model=%s  llm_base_url=%s  temperature=%.2f  max_tokens=%d",
        config.llm_model, config.llm_base_url, config.llm_temperature, config.llm_max_tokens,
    )
    logger.info(
        "ann_topk=%d  max_plan_rounds=%d  kn_select_mode=%s  kn_top_queries=%d",
        config.ann_topk, config.max_plan_rounds,
        config.img_ann_kn_select_mode, config.img_ann_kn_top_queries,
    )

    items = load_benchmark(args.input)
    logger.info("loaded %d items from %s", len(items), args.input)

    agent = BenchmarkAgent(config=config)
    t_start = time.perf_counter()

    predictions = run_benchmark(
        agent=agent,
        items=items,
        prediction_output_path=args.output,
        answer_sheet_output_path=args.answer_sheet,
    )

    elapsed = time.perf_counter() - t_start

    # ── Accuracy ──────────────────────────────────────────────────────────────
    acc = compute_accuracy(predictions)
    acc_str = f"{acc:.2%}" if acc is not None else "N/A (no gt)"

    # ── Stats file ────────────────────────────────────────────────────────────
    stats = compute_stats(predictions)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["input"] = args.input
    stats["output"] = args.output
    stats["answer_sheet"] = args.answer_sheet
    stats["llm_model"] = config.llm_model
    stats["kn_lookup_url"] = config.kn_lookup_url
    stats["img_ann_kn_select_mode"] = config.img_ann_kn_select_mode

    stats_path = Path(args.stats)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Final summary ─────────────────────────────────────────────────────────
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    elapsed_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    summary_lines = [
        "",
        "=" * 60,
        f"  Accuracy         : {acc_str}",
        f"  Total items      : {stats['total_items']}",
        f"  With ground truth: {stats['items_with_gt']}",
        f"  Correct          : {stats['correct']}",
        f"  Tool usage       : {stats['tool_usage_count']}",
        f"  KN hit rate      : {stats['kn_lookup_hit']}/{stats['kn_lookup_triggered']}"
        + (f" ({stats['kn_lookup_hit_rate']:.2%})" if stats['kn_lookup_hit_rate'] is not None else ""),
        f"  Avg plan rounds  : {stats['avg_plan_rounds']}",
        f"  Avg latency/item : {stats['avg_latency_ms_per_item']/1000:.1f}s",
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
    logger.info(summary)
    logger.info("=== benchmark run end ===")


if __name__ == "__main__":
    main()
