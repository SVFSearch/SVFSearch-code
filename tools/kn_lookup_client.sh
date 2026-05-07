#!/usr/bin/env bash
set -euo pipefail

URL="${KN_LOOKUP_URL:-http://127.0.0.1:8002/kn_lookup}"
Q1="${1:-原神 八重神子}"
Q2="${2:-原神 纳西妲}"

curl -X POST "${URL}" \
  -H 'Content-Type: application/json' \
  -d "{
        \"queries\": [\"${Q1}\", \"${Q2}\"]
      }"
