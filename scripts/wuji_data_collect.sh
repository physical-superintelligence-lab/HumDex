#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../deploy_real"

# Runtime configuration
input_root="${SCRIPT_DIR}/../deploy_real/humdex_demonstration"     # root directory containing `**/episode_*/data.json`
output_name="wuji_right"                                           # output NPZ file name
hand_side="right"                                                  # which hand side to export: left, right, or both
max_files=-1                                                       # max number of `data.json` files to process (`-1` means all)

python wuji_data_collect.py \
  --input_root "${input_root}" \
  --output_name "${output_name}" \
  --hand_side "${hand_side}" \
  --max_files "${max_files}" \
  "$@"


# will generate `wuji_policy/data/wuji_right.npz`