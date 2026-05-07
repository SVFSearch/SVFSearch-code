#!/usr/bin/env bash
set -euo pipefail

URL="${TEXT_ANN_URL:-http://127.0.0.1:8004/text_ann}"
QUERY="${1:-原神 雷电将军}"
TOP_K="${2:-5}"

curl -X POST "${URL}" \
  -H 'Content-Type: application/json' \
  -d "{
        \"query\": \"${QUERY}\",
        \"top_k\": ${TOP_K}
      }"
