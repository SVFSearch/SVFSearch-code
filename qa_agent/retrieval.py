from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class ToolRecall:
    tool: str
    request_payload: dict[str, Any]
    records: list[dict[str, Any]]
    raw_response: str
    error: str | None = None


class ANNRetriever:
    """Minimal ANN retriever for text/img/multimodal/bm25 tools."""

    def __init__(
        self,
        text_ann_url: str,
        img_ann_url: str,
        multimodal_ann_url: str,
        bm25_ann_url: str,
        timeout: int = 30,
    ) -> None:
        self.text_ann_url = text_ann_url
        self.img_ann_url = img_ann_url
        self.multimodal_ann_url = multimodal_ann_url
        self.bm25_ann_url = bm25_ann_url
        self.timeout = timeout

    def call(
        self,
        tool: str,
        question: str,
        image_path: str,
        top_k: int,
        bm25_query: str | None = None,
    ) -> ToolRecall:
        top_k = max(1, int(top_k))
        if tool == "text_ann":
            url = self.text_ann_url
            payload = {"query": question, "top_k": top_k}
        elif tool == "img_ann":
            url = self.img_ann_url
            payload = {"img": image_path, "top_k": top_k}
        elif tool == "multimodal_ann":
            url = self.multimodal_ann_url
            payload = {"query": question, "image_path": image_path, "top_k": top_k}
        elif tool == "bm25_ann":
            url = self.bm25_ann_url
            # Use the LLM-generated keyword query when provided; fall back to
            # the original question so the tool degrades gracefully.
            kw_query = bm25_query.strip() if bm25_query and bm25_query.strip() else question
            payload = {"query": kw_query, "top_k": top_k}
        else:
            return ToolRecall(
                tool=tool,
                request_payload={},
                records=[],
                raw_response="",
                error=f"unsupported tool: {tool}",
            )

        raw = ""
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            raw = response.text
            response.raise_for_status()
            data = response.json()
            records = self._normalize_records(data)
            return ToolRecall(
                tool=tool,
                request_payload=payload,
                records=records,
                raw_response=raw,
                error=None,
            )
        except Exception as exc:
            return ToolRecall(
                tool=tool,
                request_payload=payload,
                records=[],
                raw_response=raw,
                error=f"{type(exc).__name__}: {exc}",
            )

    def extract_effective_info(self, recall: ToolRecall, max_items: int = 3) -> str:
        if recall.error:
            return f"[{recall.tool}] error: {recall.error}"
        if not recall.records:
            return f"[{recall.tool}] empty"

        lines: list[str] = []
        for rank, record in enumerate(recall.records[: max(1, max_items)], start=1):
            score = self._to_float(record.get("score"))
            score_text = f"{score:.4f}" if score is not None else "N/A"
            content = self._pick_content(record)
            lines.append(f"rank={rank}, score={score_text}, info={content}")
        return f"[{recall.tool}] " + " | ".join(lines)

    def _normalize_records(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, dict) and isinstance(data.get("scores"), list):
            return [x for x in data["scores"] if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def _pick_content(self, record: dict[str, Any]) -> str:
        keys = ["title", "content", "query", "source_query", "img", "source_best_img", "chunk_id", "pid"]
        parts: list[str] = []
        for key in keys:
            val = record.get(key)
            if val is None:
                continue
            text = str(val).replace("\n", " ").strip()
            if not text:
                continue
            if len(text) > 400:
                text = text[:400] + "..."
            parts.append(f"{key}={text}")
        return "; ".join(parts) or str(record)

    def _to_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Knowledge lookup client
# ---------------------------------------------------------------------------

class KnowledgeLookupClient:
    """HTTP client for the kn_lookup_server.

    Wraps POST /kn_lookup and provides helpers for query selection and
    evidence formatting.
    """

    def __init__(self, url: str, timeout: int = 10) -> None:
        self.url = url
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def lookup(self, queries: list[str]) -> dict[str, list[str]]:
        """Call the service and return {query: [content, ...]} for found queries.

        Returns an empty dict on any error (fail-safe: knowledge enrichment is
        best-effort and must not break the main pipeline).
        """
        if not queries:
            return {}
        try:
            response = requests.post(
                self.url,
                json={"queries": queries},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            result: dict[str, list[str]] = {}
            for item in data.get("results", []):
                if item.get("found") and item.get("contents"):
                    result[item["query"]] = item["contents"]
            return result
        except Exception as exc:
            logger.warning("kn_lookup failed: %s: %s", type(exc).__name__, exc)
            return {}

    # ------------------------------------------------------------------
    # Query selection strategies
    # ------------------------------------------------------------------

    @staticmethod
    def select_by_majority(records: list[dict[str, Any]], top_n: int) -> list[str]:
        """Pick the top_n most frequent query values from img_ann records."""
        queries = [
            str(r.get("query", "")).strip()
            for r in records
            if str(r.get("query", "")).strip()
        ]
        if not queries:
            return []
        counts = Counter(queries)
        return [q for q, _ in counts.most_common(top_n)]

    # ------------------------------------------------------------------
    # Evidence formatting
    # ------------------------------------------------------------------

    def format_knowledge(
        self,
        knowledge: dict[str, list[str]],
        max_content_chars: int = 800,
    ) -> str:
        """Convert {query: [content, ...]} to a human-readable evidence block."""
        if not knowledge:
            return ""
        blocks: list[str] = []
        for query, contents in knowledge.items():
            for content in contents:
                if len(content) > max_content_chars:
                    content = content[:max_content_chars] + "..."
                blocks.append(f"[knowledge: {query}]\n{content}")
        return "\n\n".join(blocks)
