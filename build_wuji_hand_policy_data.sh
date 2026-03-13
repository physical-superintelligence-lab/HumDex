#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash build_wuji_hand_policy_data.sh <session_dir_name> [hand_side] [output_name]
# Example:
#   bash build_wuji_hand_policy_data.sh right_hand_000 right wuji_right_000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

DATA_ROOT="${REPO_ROOT}/deploy_real/humdex_demonstration"
SESSION_DIR_NAME="${1:-}"
HAND_SIDE="${2:-left}"
OUTPUT_NAME="${3:-wuji_${HAND_SIDE}}"
OUTPUT_DIR="${REPO_ROOT}/wuji_policy/data"
INTERMEDIATE_ROOT="${REPO_ROOT}/deploy_real/wuji_hand_policy_dataset_tmp/${OUTPUT_NAME}_${HAND_SIDE}"

if [[ -z "${SESSION_DIR_NAME}" ]]; then
  echo "[ERROR] Missing session_dir_name."
  echo "[USAGE] bash build_wuji_hand_policy_data.sh <session_dir_name> [hand_side] [output_name]"
  exit 1
fi

if [[ "${HAND_SIDE}" != "left" && "${HAND_SIDE}" != "right" && "${HAND_SIDE}" != "both" ]]; then
  echo "[ERROR] hand_side must be left/right/both, got: ${HAND_SIDE}"
  exit 1
fi

INPUT_ROOT="${DATA_ROOT}/${SESSION_DIR_NAME}"
if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "[ERROR] input root not found: ${INPUT_ROOT}"
  echo "[HINT] Put your data under: ${DATA_ROOT}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${INTERMEDIATE_ROOT}"

echo "[INFO] input_root   : ${INPUT_ROOT}"
echo "[INFO] hand_side    : ${HAND_SIDE}"
echo "[INFO] intermediate : ${INTERMEDIATE_ROOT}"
echo "[INFO] output       : ${OUTPUT_DIR}/${OUTPUT_NAME}.npz"

python3 "${REPO_ROOT}/deploy_real/build_wuji_hand_policy_data.py" \
  --session_dir_name "${SESSION_DIR_NAME}" \
  --hand_side "${HAND_SIDE}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_name "${OUTPUT_NAME}" \
  --intermediate_root "${INTERMEDIATE_ROOT}"

echo "[DONE] Dataset written to: ${OUTPUT_DIR}/${OUTPUT_NAME}.npz"
