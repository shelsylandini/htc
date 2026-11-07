set -euo pipefail
CONFIG="${1:-recipes/batch/main.toml}"
OUT="${2:-runs/closedloop_oa.pt}"
closedloop-oa distil --config "$CONFIG" --out "$OUT"
