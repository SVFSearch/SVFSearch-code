"""BM25 keyword retrieval server.

Loads a JSONL knowledge corpus, builds an in-memory BM25Okapi index and
exposes a POST /bm25_ann endpoint with the same response shape as the other
ANN services so the agent can treat it uniformly.

Chinese text is tokenised with jieba when available; otherwise the server
falls back to a simple regex word-split (adequate for ASCII / space-delimited
languages, but jieba is strongly recommended for Chinese corpora).

Supported corpus formats (auto-detected per line):
  1. Chunk format  – {"chunk_result": [{"chunk_id":…, "title":…, "content":…, …}]}
     (same file used by text_emb_server.py)
  2. KN format     – {"query":…, "content":…}
     (same file used by kn_lookup_server.py)

Start:
    python tools/bm25_server.py
    BM25_PORT=9005 python tools/bm25_server.py
    BM25_DATA_PATH=/path/to/corpus.jsonl python tools/bm25_server.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Optional jieba import (Chinese word segmentation)
# ---------------------------------------------------------------------------
try:
    import jieba
    jieba.setLogLevel(jieba.logging.WARNING)  # type: ignore[attr-defined]
    _HAS_JIEBA = True
    print("[info] jieba loaded — Chinese tokenisation enabled")
except ImportError:
    _HAS_JIEBA = False
    print("[warn] jieba not found — falling back to regex tokenisation")

from rank_bm25 import BM25Okapi  # noqa: E402  (after optional imports)

# ---------------------------------------------------------------------------
# Config (all overridable via environment variables)
# ---------------------------------------------------------------------------
BM25_DATA_PATH = os.getenv(
    "BM25_DATA_PATH",
)
BM25_PORT = int(os.getenv("BM25_PORT", "8005"))
TOP_K_DEFAULT = int(os.getenv("BM25_TOP_K_DEFAULT", "10"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="BM25 Retrieval Service")

# Global state populated on startup
_BM25: BM25Okapi | None = None
_DOCS: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Return a list of tokens for BM25 indexing / querying.

    Uses jieba for Chinese character streams when available, otherwise splits
    on non-word boundaries (handles ASCII, punctuation, etc.).
    """
    if _HAS_JIEBA:
        return [t for t in jieba.cut(text) if t.strip() and not t.isspace()]
    # Fallback: keep sequences of word characters (works for English / digits)
    return re.findall(r"\w+", text, re.UNICODE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_data(path: str) -> list[dict[str, Any]]:
    """Load documents from a JSONL file into a flat list of dicts.

    Each doc has keys: chunk_id, title, content.
    Supports both the chunk-embed format and the kn-lookup format.
    """
    docs: list[dict[str, Any]] = []
    fpath = Path(path).expanduser().resolve()
    if not fpath.exists():
        print(f"[warn] BM25 data file not found: {fpath}")
        print("[warn] BM25 index will be empty — set BM25_DATA_PATH correctly")
        return docs

    count = 0
    with open(fpath, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[warn] line {lineno}: JSON decode error: {exc}")
                continue

            # ── Format 1: chunk_result list ──────────────────────────────
            if "chunk_result" in row:
                for chunk in row.get("chunk_result", []):
                    content = str(chunk.get("content", "")).strip()
                    if not content:
                        continue
                    docs.append(
                        {
                            "chunk_id": str(chunk.get("chunk_id", f"line{lineno}")),
                            "title": str(chunk.get("title", "")),
                            "content": content,
                        }
                    )
                    count += 1

            # ── Format 2: flat query / content ───────────────────────────
            elif "content" in row:
                content = str(row.get("content", "")).strip()
                if not content:
                    continue
                docs.append(
                    {
                        "chunk_id": str(row.get("pid", row.get("query", f"line{lineno}"))),
                        "title": str(row.get("query", row.get("title", ""))),
                        "content": content,
                    }
                )
                count += 1

    print(f"[info] loaded {count} documents from {fpath}")
    return docs


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    global _BM25, _DOCS
    _DOCS = _load_data(BM25_DATA_PATH)
    if _DOCS:
        print("[info] building BM25 index …")
        corpus_tokens = [
            _tokenize(doc["title"] + " " + doc["content"]) for doc in _DOCS
        ]
        _BM25 = BM25Okapi(corpus_tokens)
        print(f"[info] BM25 index ready: {len(_DOCS)} documents")
    else:
        print("[warn] BM25 index is empty — check BM25_DATA_PATH")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class Req(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/bm25_ann")
def bm25_search(req: Req) -> dict[str, Any]:
    """BM25 keyword search over the loaded corpus.

    Returns a JSON object compatible with the other ANN services:
        {"query": "…", "scores": [{"chunk_id":…, "title":…, "content":…, "score":…}, …]}

    Only documents with score > 0 (at least one matching term) are returned.
    """
    if _BM25 is None or not _DOCS:
        return {"query": req.query, "scores": []}

    tokens = _tokenize(req.query)
    if not tokens:
        return {"query": req.query, "scores": []}

    top_k = max(1, min(req.top_k, len(_DOCS)))
    raw_scores: np.ndarray = _BM25.get_scores(tokens)

    # Sort descending; keep only docs with score > 0 (actual term overlap)
    sorted_indices = np.argsort(raw_scores)[::-1][:top_k]
    results: list[dict[str, Any]] = []
    for idx in sorted_indices:
        score = float(raw_scores[idx])
        if score <= 0.0:
            break  # All remaining docs have no term overlap — stop early
        results.append(
            {
                "chunk_id": _DOCS[idx]["chunk_id"],
                "title": _DOCS[idx]["title"],
                "content": _DOCS[idx]["content"],
                "score": round(score, 6),
            }
        )

    return {"query": req.query, "scores": results}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "num_docs": len(_DOCS), "index_ready": _BM25 is not None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=BM25_PORT)