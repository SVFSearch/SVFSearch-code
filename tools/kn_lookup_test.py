"""Test script for kn_lookup_server.

Usage:
    # Start the server first:
    #   python tools/kn_lookup_server.py
    # Then run:
    python tools/kn_lookup_test.py
    python tools/kn_lookup_test.py --url http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import sys

import requests

DEFAULT_URL = "http://127.0.0.1:8002"


def _post(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def _get(url: str) -> dict:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_health(base: str) -> None:
    data = _get(f"{base}/health")
    assert data["status"] == "ok", f"unexpected status: {data}"
    n = data["num_unique_queries"]
    assert n > 0, "knowledge base is empty"
    print(f"[PASS] health — {n} unique queries loaded")


def test_lookup_known_single(base: str) -> None:
    payload = {"queries": ["CS:GO Dust II"]}
    data = _post(f"{base}/kn_lookup", payload)
    result = data["results"][0]
    assert result["found"], "expected CS:GO Dust II to be found"
    assert len(result["contents"]) >= 1, "expected at least one content"
    preview = result["contents"][0][:120].replace("\n", " ")
    print(f"[PASS] lookup_known_single — found {len(result['contents'])} content(s), preview: {preview!r}")


def test_lookup_known_multiple(base: str) -> None:
    queries = ["CS:GO Dust II", "CS:GO Mirage", "CS:GO Inferno"]
    payload = {"queries": queries}
    data = _post(f"{base}/kn_lookup", payload)
    results = data["results"]
    assert len(results) == len(queries)
    for r in results:
        status = "found" if r["found"] else "missing"
        count = len(r["contents"])
        print(f"  [{status}] {r['query']!r} -> {count} content(s)")
    found_count = sum(1 for r in results if r["found"])
    assert found_count >= 1, "expected at least one known query to match"
    print(f"[PASS] lookup_known_multiple — {found_count}/{len(queries)} queries found")


def test_lookup_missing(base: str) -> None:
    payload = {"queries": ["__this_query_does_not_exist_xyz_999__"]}
    data = _post(f"{base}/kn_lookup", payload)
    result = data["results"][0]
    assert not result["found"], "expected missing query to return found=False"
    assert result["contents"] == [], "expected empty contents for missing query"
    print("[PASS] lookup_missing — correctly returned found=False, contents=[]")


def test_lookup_empty_list(base: str) -> None:
    payload = {"queries": []}
    data = _post(f"{base}/kn_lookup", payload)
    assert data["results"] == [], "expected empty results for empty query list"
    print("[PASS] lookup_empty_list — empty query list returns empty results")


def test_lookup_both_files(base: str) -> None:
    """If a query exists in both part_1 and part_2, we should get 2 contents."""
    payload = {"queries": ["CS:GO Mirage"]}
    data = _post(f"{base}/kn_lookup", payload)
    result = data["results"][0]
    if result["found"]:
        n = len(result["contents"])
        print(f"[PASS] lookup_both_files — CS:GO Mirage has {n} content(s) "
              f"({'both files' if n >= 2 else 'one file only'})")
    else:
        print("[SKIP] lookup_both_files — CS:GO Mirage not found in test data")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the kn_lookup server")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    tests = [
        test_health,
        test_lookup_known_single,
        test_lookup_known_multiple,
        test_lookup_missing,
        test_lookup_empty_list,
        test_lookup_both_files,
    ]

    failed = 0
    print(f"\n=== kn_lookup_server tests  [{base}] ===\n")
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
