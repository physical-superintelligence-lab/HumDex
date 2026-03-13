
SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/../assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
net=enp14s0  # NIC connected to the robot

cd "${SCRIPT_DIR}/../deploy_real"

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cpu \
    # --safety_rate_limit \
    # --safety_rate_limit_scope arms \
    # --max_dof_delta_per_step 1.0 \
    # --max_dof_delta_print_every 200 \
    # --use_hand \
    # --smooth_body 0.5
    # --record_proprio \
