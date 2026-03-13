#!/usr/bin/env bash

SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}/deploy_real"

# Runtime configuration
session_dir_name="right_hand_000"   # folder name under deploy_real/humdex_demonstration
hand_side="right"                   # "left" / "right" / "both"
output_name="wuji_right_000"
output_dir="${SCRIPT_DIR}/wuji_policy/data"
intermediate_root="${SCRIPT_DIR}/deploy_real/wuji_hand_policy_dataset_tmp/${output_name}_${hand_side}"

python ../deploy_real/build_wuji_hand_policy_data.py \
    --session_dir_name ${session_dir_name} \
    --hand_side ${hand_side} \
    --output_dir ${output_dir} \
    --output_name ${output_name} \
    --intermediate_root ${intermediate_root} \
    --cleanup_intermediate
