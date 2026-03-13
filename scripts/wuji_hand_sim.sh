SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}/../deploy_real"

# Runtime configuration
redis_ip="localhost"
hand_side="left"  # "left" or "right"
target_fps=50
glove="manus"      # "manus" or "vdhand"
retarget_config="${SCRIPT_DIR}/../wuji-retargeting/example/config/retarget_${glove}_${hand_side}.yaml"
policy_tag="wuji_${hand_side}"
policy_epoch=-1

# Start controller
python server_wuji_hand_sim_redis.py \
    --hand_side ${hand_side} \
    --config ${retarget_config} \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --no_smooth \
    # --use_model \
    # --policy_tag ${policy_tag} \
    # --policy_epoch ${policy_epoch}
