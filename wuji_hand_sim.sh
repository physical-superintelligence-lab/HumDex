SCRIPT_DIR=$(dirname $(realpath $0))
cd deploy_real

# Runtime configuration
redis_ip="localhost"
hand_side="left"  # "left" or "right"
target_fps=50
hand="manus"      # "manus" or "vdhand"
retarget_config="${SCRIPT_DIR}/wuji-retargeting/example/config/retarget_${hand}_${hand_side}.yaml"

# Start controller
python server_wuji_hand_sim_redis.py \
    --hand_side ${hand_side} \
    --config ${retarget_config} \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --freshness_ms 500 \
    # --disable_dexpilot_projection \
    # --print_distances_and_exit \
    # --mjcf_path /abs/path/to/left.xml \
    # --verbose \
    # --no_write_action_target \
    # --use_model \
    # --policy_tag geort_filter_wuji \
    # --policy_epoch -1 \
    # --clamp_min -1.5 \
    # --clamp_max 1.5 \
    # --max_delta_per_step 0.08 \


