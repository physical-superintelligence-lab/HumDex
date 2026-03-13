SCRIPT_DIR=$(dirname $(realpath $0))
cd deploy_real

# Runtime configuration
redis_ip="localhost"
hand_side="right"  # "left" or "right"
target_fps=50
glove="vdhand"      # "manus" or "vdhand"
retarget_config="${SCRIPT_DIR}/wuji-retargeting/example/config/retarget_${glove}_${hand_side}.yaml"
# Keep GeoRT args for optional use_model mode.
policy_tag="wuji_right_000"
policy_epoch=200

# Start controller
python ../deploy_real/server_wuji_hand_sim_redis.py \
    --hand_side ${hand_side} \
    --config ${retarget_config} \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --no_smooth \
    --use_model \
    --policy_tag ${policy_tag} \
    --policy_epoch ${policy_epoch}
