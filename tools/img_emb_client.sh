#!/usr/bin/env bash
set -euo pipefail

URL="${IMG_ANN_URL:-http://127.0.0.1:8001/img_ann}"
IMAGE_PATH="${1:-/path/to/image.jpg}"
TOP_K="${2:-5}"

curl -X POST "${URL}" \
  -H "Content-Type: application/json" \
  -d "{
        \"img\": \"${IMAGE_PATH}\",
        \"top_k\": ${TOP_K}
      }"
