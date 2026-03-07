#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash deploy_real/collect.sh
#   bash deploy_real/collect.sh 20260306_2310_twist2_left
#   bash deploy_real/collect.sh 20260306_2310_twist2_left left wuji_left
#
# Args:
#   $1 session_dir_name (optional): subdirectory under "humdex demonstration"
#   $2 hand_side (optional): left|right|both, default left
#   $3 output_name (optional): output npz base name (without .npz)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${REPO_ROOT}/deploy_real/humdex_demonstration"
DEFAULT_SESSION="20260306_2310_twist2_left"
HAND_SIDE="${2:-left}"
OUTPUT_NAME="${3:-wuji_${HAND_SIDE}}"
OUTPUT_DIR="${REPO_ROOT}/wuji_policy/data"
INTERMEDIATE_ROOT="${REPO_ROOT}/deploy_real/wuji_hand_policy_dataset_tmp/${OUTPUT_NAME}_${HAND_SIDE}"

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  INPUT_ROOT="${DATA_ROOT}/$1"
else
  INPUT_ROOT="${DATA_ROOT}/${DEFAULT_SESSION}"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] data root not found: ${DATA_ROOT}"
  exit 1
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "[ERROR] input root not found: ${INPUT_ROOT}"
  echo "[HINT] Put your data under: ${DATA_ROOT}"
  exit 1
fi

if [[ "${HAND_SIDE}" != "left" && "${HAND_SIDE}" != "right" && "${HAND_SIDE}" != "both" ]]; then
  echo "[ERROR] hand_side must be left/right/both, got: ${HAND_SIDE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${INTERMEDIATE_ROOT}"

echo "[INFO] input_root : ${INPUT_ROOT}"
echo "[INFO] hand_side  : ${HAND_SIDE}"
echo "[INFO] intermediate: ${INTERMEDIATE_ROOT}"
echo "[INFO] output_dir : ${OUTPUT_DIR}"
echo "[INFO] output_name: ${OUTPUT_NAME}.npz"

python3 "${SCRIPT_DIR}/collect.py" \
  --input_root "${INPUT_ROOT}" \
  --hand_side "${HAND_SIDE}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_name "${OUTPUT_NAME}" \
  --intermediate_root "${INTERMEDIATE_ROOT}"

echo "[DONE] Dataset written to: ${OUTPUT_DIR}/${OUTPUT_NAME}.npz"
