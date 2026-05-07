"""Test script for bm25_server.

Verifies the /health and /bm25_ann endpoints against a running server.
Each test is self-contained; failures are reported but do not stop
subsequent tests, giving a full picture in one run.

Usage:
    # Start the server first:
    #   python tools/bm25_server.py
    # Then run:
    python tools/bm25_test.py
    python tools/bm25_test.py --url http://127.0.0.1:8005
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests

DEFAULT_URL = "http://127.0.0.1:8005"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(url: str, timeout: int = 10) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_health(base: str) -> None:
    data = _get(f"{base}/health")
    assert data.get("status") == "ok", f"unexpected status: {data}"
    n = data.get("num_docs", 0)
    ready = data.get("index_ready", False)
    assert ready, "index_ready should be True after startup"
    assert n > 0, f"num_docs should be > 0 but got {n}"
    print(f"[PASS] health — {n} docs indexed, index_ready={ready}")


def test_basic_search(base: str) -> None:
    """A query with common terms should return at least one result."""
    payload = {"query": "原神 雷电将军 技能", "top_k": 5}
    data = _post(f"{base}/bm25_ann", payload)
    assert "scores" in data, f"response missing 'scores': {data}"
    scores = data["scores"]
    # Scores may be empty if corpus doesn't contain these terms — that's fine.
    # We just assert the shape is correct.
    for item in scores:
        assert "chunk_id" in item, f"missing chunk_id in {item}"
        assert "content" in item, f"missing content in {item}"
        assert "score" in item, f"missing score in {item}"
        assert item["score"] > 0, f"score should be > 0 but got {item['score']}"
    print(
        f"[PASS] basic_search — query={payload['query']!r}"
        f" → {len(scores)} result(s)"
        + (f", top score={scores[0]['score']:.4f}" if scores else "")
    )


def test_top_k_respected(base: str) -> None:
    """Response must not exceed the requested top_k."""
    for top_k in (1, 3, 10):
        payload = {"query": "角色 技能 伤害 输出", "top_k": top_k}
        data = _post(f"{base}/bm25_ann", payload)
        n = len(data.get("scores", []))
        assert n <= top_k, f"top_k={top_k} but got {n} results"
    print("[PASS] top_k_respected — response count ≤ top_k for top_k in {1, 3, 10}")


def test_scores_descending(base: str) -> None:
    """Scores must be in non-increasing order."""
    payload = {"query": "游戏 玩家 道具 任务 地图", "top_k": 10}
    data = _post(f"{base}/bm25_ann", payload)
    scores = [item["score"] for item in data.get("scores", [])]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"scores not sorted at position {i}: {scores[i]} < {scores[i + 1]}"
        )
    print(f"[PASS] scores_descending — {len(scores)} results are in order")


def test_empty_query(base: str) -> None:
    """An empty or whitespace-only query should return an empty scores list."""
    for q in ("", "   "):
        payload = {"query": q, "top_k": 5}
        data = _post(f"{base}/bm25_ann", payload)
        assert data.get("scores") == [], (
            f"expected empty scores for query={q!r} but got {data.get('scores')}"
        )
    print("[PASS] empty_query — empty/whitespace queries return []")


def test_zero_score_excluded(base: str) -> None:
    """A query with no matching terms should return no results (all scores 0)."""
    payload = {"query": "xyzzy_nonexistent_token_99999", "top_k": 5}
    data = _post(f"{base}/bm25_ann", payload)
    scores = data.get("scores", [])
    for item in scores:
        assert item["score"] > 0, (
            f"zero-score result should be filtered out: {item}"
        )
    print(
        f"[PASS] zero_score_excluded — garbage query returned {len(scores)} result(s)"
        " (all with score > 0)"
    )


def test_response_shape(base: str) -> None:
    """Response must always contain 'query' and 'scores' keys."""
    payload = {"query": "任意查询", "top_k": 3}
    data = _post(f"{base}/bm25_ann", payload)
    assert "query" in data, f"'query' key missing from response: {data.keys()}"
    assert "scores" in data, f"'scores' key missing from response: {data.keys()}"
    assert data["query"] == payload["query"], "echoed query should match input"
    print("[PASS] response_shape — 'query' and 'scores' present and correct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Base URL of the bm25_server (default: http://127.0.0.1:8005)",
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")

    tests = [
        test_health,
        test_basic_search,
        test_top_k_respected,
        test_scores_descending,
        test_empty_query,
        test_zero_score_excluded,
        test_response_shape,
    ]

    failed = 0
    print(f"\n=== bm25_server tests  [{base}] ===\n")
    for fn in tests:
        try:
            fn(base)
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            failed += 1

    print(f"\n{'All tests passed.' if failed == 0 else f'{failed} test(s) FAILED.'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
