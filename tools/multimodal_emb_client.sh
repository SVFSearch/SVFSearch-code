#!/usr/bin/env bash
set -euo pipefail

URL="${MULTIMODAL_ANN_URL:-http://127.0.0.1:8003/multimodal_ann}"
QUERY="${1:-天涯明月刀 一代宗师毒圈 毒圈的基本规则}"
IMAGE_PATH="${2:-/path/to/image.jpg}"
TOP_K="${3:-5}"

curl -X POST "${URL}" \
  -H 'Content-Type: application/json' \
  -d "{
        \"query\": \"${QUERY}\",
        \"image_path\": \"${IMAGE_PATH}\",
        \"top_k\": ${TOP_K}
      }"
