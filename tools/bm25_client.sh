set -euo pipefail

URL="${BM25_ANN_URL:-http://127.0.0.1:8005/bm25_ann}"
QUERY="${1:-原神 雷电将军 技能}"
TOP_K="${2:-5}"

curl -s -X POST "${URL}" \
  -H 'Content-Type: application/json' \
  -d "{
        \"query\": \"${QUERY}\",
        \"top_k\": ${TOP_K}
      }"
