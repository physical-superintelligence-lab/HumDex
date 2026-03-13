SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}/../wuji_policy"

# Runtime configuration
human_data_name="wuji_right_example"   # data/<name>.npz
hand_config="wuji_right"           # training/config/<hand_config>.json
ckpt_tag="wuji_right_example"
qpos_key="qpos"
n_samples=20000
batch_size=2048
lr=1e-4
epoch=1
save_every=10
ckpt_root="${SCRIPT_DIR}/../wuji_policy/checkpoint"

# Start training
python training/trainer.py \
    -hand ${hand_config} \
    -human_data ${human_data_name} \
    -ckpt_tag ${ckpt_tag} \
    --qpos_key ${qpos_key} \
    --n_samples ${n_samples} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --epoch ${epoch} \
    --save_every ${save_every} \
    --ckpt_root ${ckpt_root}
