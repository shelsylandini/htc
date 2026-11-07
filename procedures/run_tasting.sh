set -euo pipefail
CONFIG="${1:-recipes/batch/main.toml}"
CKPT="${2:-runs/closedloop_oa.pt}"
closedloop-oa taste --config "$CONFIG" --ckpt "$CKPT"
closedloop-oa blend --config "$CONFIG"
