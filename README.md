# HumDex: Humanoid Dexterous Manipulation Made Easy
By Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang

![Banner for HumDex](./assets/HumDex.png)

## Content Table

- [Installation](#installation)
- [Wuji Policy](#wuji-policy-geort)
- [Teleop](#teleop)
- [G1 Controller](#g1-controller)
- [Wuji Hand Controller](#wuji-hand-controller)
- [Data Collection](#data-collection)

---

## Installation

We will have two conda environments for Humdex. One is called `humdex`, which can be used for controller training, controller deployment, and teleop data collection. The other is called `gmr`, which can be used for online motion retargeting.

### 1) Create `gmr` Environment

```bash
conda create -n gmr python=3.10 -y
conda activate gmr

git clone https://github.com/YanjieZe/GMR.git
cd GMR

# install GMR
pip install -e .
cd ..

pip install python-osc zmq pyyaml

conda install -c conda-forge libstdcxx-ng -y
```

### 2) Create `humdex` Environment
```bash
conda create -n humdex python=3.8 -y
conda activate humdex

# install wuji-retargeting

cd wuji-retargeting
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
```

For the rest of `humdex` environment setup, follow TWIST2 README:

- [Step 2: Install Isaac Gym](https://github.com/LiangHeng121/TWIST2?tab=readme-ov-file#step-2-install-isaac-gym)
- [Step 3: Install Packages](https://github.com/LiangHeng121/TWIST2?tab=readme-ov-file#step-3-install-packages)
- [Step 4: Install Unitree SDK2 for Laptop Sim2Real](https://github.com/LiangHeng121/TWIST2?tab=readme-ov-file#step-4-install-unitree-sdk2-for-laptop-sim2real)

### 3) Clone `GR00T-WholeBodyControl` (for sonic)

```bash
cd ..
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull
```

Then follow the official [doc](https://nvlabs.github.io/GR00T-WholeBodyControl/) to install its environment.

---

<<<<<<< Updated upstream
=======
## Wuji Policy

### 1) Install

```bash
conda activate humdex
cd wuji_policy
pip install -r requirements.txt
pip install -e .
```

### 2) Train (supervised)

Data format (`.npz`) used by `wuji_policy/geort/trainer.py`:
- required key: `fingertips_rel_wrist` with shape `[T, 5, 3]` (or `[T, 5, >=3]`)
- supervision key: `qpos` (recommended), or `robot_qpos`, `joint`, `joint_angle`, `joint_angles`

If your raw collection files are not in this format yet, first run:

```bash
conda activate humdex
bash wuji_data_collect.sh
```

By default, `wuji_data_collect.sh` reads from `deploy_real/humdex_demonstration`, which is the default output root used by `data_record.sh`.

This will generate:
- `HumDex/wuji_policy/data/collect.npz`

You can also pass a single input file:

```bash
python wuji_data_collect.py --input_path /path/to/raw_xxx.npz
```

which outputs:
- `HumDex/wuji_policy/data/raw_xxx.npz`

If needed, set a custom output name:

```bash
bash wuji_data_collect.sh --input_path /path/to/raw_dir --output_name wuji_left
```

Example:

```bash
cd wuji_policy
python geort/trainer.py \
  -hand wuji_right \
  -human_data wuji_right \
  -ckpt_tag geort_wuji \
  --qpos_key qpos \
  --n_samples 20000 \
  --batch_size 2048 \
  --lr 1e-4 \
  --save_every 10 \
  --ckpt_root ./checkpoint
```

`-human_data` should match the output npz stem in `wuji_policy/data` (e.g., `collect` for `collect.npz`, `wuji_left` for `wuji_left.npz`).

For left hand, replace `wuji_right` with `wuji_left`.

### 3) Inference (model-based vs optimal-based)

Both `wuji_policy/server_wuji_hand_redis.py` and `wuji_policy/deploy2.py` support:
- `--use_model`: model-based (GeoRT)
- default (without it): optimal-based (`WujiHandRetargeter`)

You can override alias by setting `--policy_tag`.

Example (model-based):

```bash
cd wuji_policy
python server_wuji_hand_redis.py \
  --hand_side right \
  --use_model \
  --checkpoint right_last \
  --policy_epoch -1
```

Example (optimal-based):

```bash
cd wuji_policy
python server_wuji_hand_redis.py \
  --hand_side right
```

Checkpoint location:
- `wuji_policy/checkpoint/<your_tag_or_run_name>/`
- each checkpoint folder should contain at least `config.json` and `last.pth` (or `epoch_*.pth`)

---

>>>>>>> Stashed changes
## Teleop

For a concise end-to-end setup flow, see [`doc/teleop.md`](doc/teleop.md).

### 1) Unified Entry

```bash
conda activate gmr

bash teleop.sh [options] [-- extra_args]
```

Supported selectors:

- `--policy {twist2|sonic}` (default `twist2`)
- `--body {vdmocap|slimevr}` (default `vdmocap`)
- `--hand {vdhand|manus}` (default `vdhand`)

### 2) Common Examples

Run default combo:

```bash
bash teleop.sh
```

Run sonic + vdmocap + manus:

```bash
bash teleop.sh --policy sonic --body vdmocap --hand manus
```

Run twist2 + slimevr + vdhand:

```bash
bash teleop.sh --policy twist2 --body slimevr --hand vdhand
```

### 3) Config Files

Main YAML:

- `deploy_real/config/teleop.yaml`

Config Structure:

- `runtime`: loop settings like `target_fps`, `print_every`, `max_steps`
- `network`: redis + mocap transport (`network.redis`, `network.mocap.default/body/hand`)
- `retarget`: retarget core settings (`actual_human_height`, `hands`, `format`, `offset_to_ground`)
- `control`: runtime control settings (`safe_idle_pose_id`, `ramp_*_seconds`, `ramp_ease`)
- `adapters`: source-specific settings (`vdmocap`, `vdhand`, `manus`, `slimevr`)
- `policy`: policy-specific settings (`policy.sonic`, `policy.twist2`)


### 4) Keyboard Behavior

- `k`: toggle send/default mode
- `p`: toggle hold mode

---

## G1 Controller

### 1) Sim Controller

```bash
## for --policy twist2
conda activate humdex
# Warm arp the redis server at first time
bash run_motion_server.sh
bash sim2sim.sh

## for --policy sonic
# Terminal 1 — MuJoCo simulator
cd ../GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — C++ deployment
cd ../GR00T-WholeBodyControl/gear_sonic_deploys
source scripts/setup_env.sh
bash deploy.sh sim --input-type zmq
```

### 2) Real Controller

```bash
## for --policy twist2
conda activate humdex
bash run_motion_server.sh
# edit `net` in `sim2real.sh` to your real NIC name before running
bash sim2real.sh

## for --policy sonic
cd ../GR00T-WholeBodyControl/gear_sonic_deploys
source scripts/setup_env.sh
bash deploy.sh real --input-type zmq
```

---

## Wuji Hand Controller

For Wuji device setup details, see [`doc/wuji.md`](doc/wuji.md).

### 1) Sim Hand Controller

```bash
conda activate humdex
bash wuji_hand_sim.sh
```

### 2) Real Hand Controller

```bash
conda activate humdex
bash wuji_hand_real.sh
```

---

## Wuji Policy

### 1) Install

```bash
conda activate humdex
cd wuji_policy
pip install -r requirements.txt
pip install -e .
```

### 2) Train (supervised)

Data format (`.npz`) used by `wuji_policy/geort/trainer.py`:
- required key: `fingertips_rel_wrist` with shape `[T, 5, 3]` (or `[T, 5, >=3]`)
- supervision key: `qpos` (recommended), or `robot_qpos`, `joint`, `joint_angle`, `joint_angles`

Example:

```bash
cd wuji_policy
python geort/trainer.py \
  -hand wuji_right \
  -human_data wuji_right \
  -ckpt_tag geort_wuji \
  --qpos_key qpos \
  --n_samples 20000 \
  --batch_size 2048 \
  --lr 1e-4 \
  --save_every 10 \
  --ckpt_root ./checkpoint
```

For left hand, replace `wuji_right` with `wuji_left`.

### 3) Inference (model-based vs optimal-based)

Both `wuji_policy/server_wuji_hand_redis.py` and `wuji_policy/deploy2.py` support:
- `--use_model`: model-based (GeoRT)
- default (without it): optimal-based (`WujiHandRetargeter`)

You can override alias by setting `--policy_tag`.

Example (model-based):

```bash
cd wuji_policy
python server_wuji_hand_redis.py \
  --hand_side right \
  --use_model \
  --checkpoint right_last \
  --policy_epoch -1
```

Example (optimal-based):

```bash
cd wuji_policy
python server_wuji_hand_redis.py \
  --hand_side right
```

Checkpoint location:
- `wuji_policy/checkpoint/<your_tag_or_run_name>/`
- each checkpoint folder should contain at least `config.json` and `last.pth` (or `epoch_*.pth`)

---


## Data Collection

For robot/human recording workflow and saved data layout, see [`doc/data_collection.md`](doc/data_collection.md).

### 1) Start Camera Stream on g1

```bash
bash realsense_zmq_pub_g1.sh
```

### 2) Keyboard Data Recording

```bash
bash data_record.sh

# sonic channel
bash data_record.sh --channel sonic
```

### 3) Human Data Recording

```bash
bash data_record_human.sh

# sonic channel
bash data_record_human.sh --channel sonic
```

---
