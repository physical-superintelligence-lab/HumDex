#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

python wuji_data_collect.py \
  --input_root "${SCRIPT_DIR}/deploy_real/twist2_demonstration" \
  --output_name "wuji_right" \
  --hand_side "right" \
  --max_files -1 \
  "$@"