"""Knowledge lookup server.

Loads two JSONL knowledge files (query_rag_kn_part_1.jsonl and
qwen_rag_kn_part_2.jsonl) into memory and exposes a POST /kn_lookup
endpoint that accepts a list of queries and returns all matching content
from both files.

Start:
    python tools/kn_lookup_server.py                        # default port 8002
    KN_PORT=9002 python tools/kn_lookup_server.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_KN_FILES = [
    TOOLS_DIR / "query_rag_kn_part_1.jsonl",
    TOOLS_DIR / "qwen_rag_kn_part_2.jsonl",
]


def _resolve_kn_files() -> list[Path]:
    """Resolve knowledge file list from env or fall back to local defaults.

    Priority:
    1) KN_FILES (':' separated absolute/relative paths)
    2) KN_FILE_1 / KN_FILE_2
    3) tools/query_rag_kn_part_1.jsonl + tools/qwen_rag_kn_part_2.jsonl
    """
    raw = os.getenv("KN_FILES", "").strip()
    if raw:
        return [Path(x).expanduser() for x in raw.split(":") if x.strip()]

    f1 = os.getenv("KN_FILE_1", "").strip()
    f2 = os.getenv("KN_FILE_2", "").strip()
    files = [Path(x).expanduser() for x in [f1, f2] if x]
    return files or DEFAULT_KN_FILES


KN_FILES = _resolve_kn_files()
KN_PORT = int(os.getenv("KN_PORT", "8002"))
# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="Knowledge Lookup Service")

# {query_text: [content_from_file1, content_from_file2, ...]}
_KN: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Startup: load both files
# ---------------------------------------------------------------------------

def _load_knowledge() -> dict[str, list[str]]:
    kn: dict[str, list[str]] = {}
    for path in KN_FILES:
        path = path.expanduser().resolve()
        if not path.exists():
            print(f"[warn] knowledge file missing: {path}")
            continue
        count = 0
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"[warn] {path.name}:{lineno} json decode error: {exc}")
                    continue
                query = str(row.get("query", "")).strip()
                content = str(row.get("content", "")).strip()
                if query and content:
                    kn.setdefault(query, []).append(content)
                    count += 1
        print(f"[info] loaded {count} records from {path} ({len(kn)} unique queries so far)")
    if not kn:
        print("[warn] knowledge base is empty; check KN_FILES/KN_FILE_1/KN_FILE_2")
    return kn


@app.on_event("startup")
def _startup() -> None:
    global _KN
    _KN = _load_knowledge()
    print(f"[info] knowledge base ready: {len(_KN)} unique queries total")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class LookupRequest(BaseModel):
    queries: list[str]


class QueryResult(BaseModel):
    query: str
    found: bool
    contents: list[str]  # one entry per file that has this query


class LookupResponse(BaseModel):
    results: list[QueryResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/kn_lookup", response_model=LookupResponse)
def kn_lookup(req: LookupRequest) -> LookupResponse:
    """Return knowledge content for each requested query.

    If a query exists in both files, both contents are returned in order.
    If a query is not found, ``found=False`` and ``contents=[]``.
    """
    results: list[QueryResult] = []
    for q in req.queries:
        q_stripped = q.strip()
        contents = _KN.get(q_stripped, [])
        results.append(
            QueryResult(query=q_stripped, found=bool(contents), contents=contents)
        )
    return LookupResponse(results=results)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "num_unique_queries": len(_KN)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=KN_PORT)
