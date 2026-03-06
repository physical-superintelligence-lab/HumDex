SCRIPT_DIR=$(dirname $(realpath $0))
cd deploy_real

# Runtime configuration
redis_ip="localhost"
hand_side="left"  # "left" or "right"
target_fps=50
glove="manus"      # "manus" or "vdhand"
retarget_config="${SCRIPT_DIR}/wuji-retargeting/example/config/retarget_${glove}_${hand_side}.yaml"

# Start controller
python server_wuji_hand_sim_redis.py \
    --hand_side ${hand_side} \
    --config ${retarget_config} \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --no_smooth \
    # --use_model \
    # --policy_tag geort_filter_wuji \
    # --policy_epoch -1 \


