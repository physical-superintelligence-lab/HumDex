#!/bin/bash

# Wuji Hand Controller via Redis
# Read hand control data from Redis and drive Wuji hand in real time.

source ~/miniconda3/bin/activate twist2
SCRIPT_DIR=$(dirname $(realpath $0))
cd deploy_real

# Runtime configuration
redis_ip="localhost"
hand_side="left"  # "left" or "right"
target_fps=50
retarget_config="${SCRIPT_DIR}/wuji-retargeting/example/config/retarget_manus_${hand_side}.yaml"

# Start controller
python server_wuji_hand_redis.py \
    --hand_side ${hand_side} \
    --config ${retarget_config} \
    --serial_number 3555374E3533 \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --no_smooth \
    --debug_pinch \
    --debug_pinch_every 50 \
    --disable_dexpilot_projection \
    # --use_model \
    # --policy_tag wuji_left_new_1 \
    # --policy_epoch 200
    # --pinch_project_dist_max 0.12 \
    # --disable_dexpilot_projection \
    # --pinch_project_ratio 0.03 \
    # --pinch_escape_ratio 0.04 \


