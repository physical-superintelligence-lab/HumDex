# HumDex: Humanoid Dexterous Manipulation Made Easy
By Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang

![Banner for HumDex](./assets/HumDex.png)

<p align="center">
  <a href="https://psi-lab.ai/humdex"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2603.12260"><img src="https://img.shields.io/badge/paper-arxiv-red" alt="Paper"></a>
  <a href="https://github.com/physical-superintelligence-lab/humdex/issues"><img src="https://img.shields.io/github/issues/physical-superintelligence-lab/humdex?color=yellow" alt="Issues"></a>
</p>

## Content Table

- [Installation](#installation)
- [Teleop](#teleop)
- [G1 Controller](#g1-controller)
- [Wuji Hand Controller](#wuji-hand-controller)
- [Wuji Policy](#wuji-policy)
- [Data Collection](#data-collection)
- [Policy Learning](#policy-learning)

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

# install wuji_policy
cd wuji_policy
pip install -r requirements.txt
pip install -e .

# install for act
cd act
pip install -r requirements.txt

pip install pyrealsense2
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
- `p`: toggle send/hold mode

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
cd ../GR00T-WholeBodyControl/gear_sonic_deploy
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


### 1) Build training `.npz` from collected data

```bash
conda activate humdex
bash wuji_data_collect.sh
```

This generates:
- `wuji_policy/data/wuji_right.npz`

### 2) Training

Data format (`.npz`) used by `wuji_policy/geort/trainer.py`:
- required key: `fingertips_rel_wrist` with shape `[T, 5, 3]`
- supervision key: `qpos`

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

`-human_data` should match the output NPZ stem in `wuji_policy/data` (e.g., `wuji_right` for `wuji_right.npz`).

For left hand, replace `wuji_right` with `wuji_left`.

### 3) Inference

The scripts in [Wuji Hand Controller](#wuji-hand-controller) use `wuji-retargeting` by default.
To use a trained policy instead:

1. In `wuji_hand_real.sh` and `wuji_hand_sim.sh`, set:
   - `policy_tag` to your checkpoint tag
   - `policy_epoch` to your checkpoint epoch (use `-1` for latest)
2. In both scripts, uncomment:
   - `--use_model`
   - `--policy_tag ${policy_tag}`
   - `--policy_epoch ${policy_epoch}`

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

## Policy Learning

### 1) Data Processing

**a) Human data preprocessing:**

For human tracking data, approximate proprioceptive state with previous-frame action.  
Skip this step for robot data.

```bash
cd act
python scripts/convert_human_data.py \
  --dataset_dir /path/to/human_dataset
```

Processes `episode_*` folders in-place (creates `data_original.json` + `rgb_original/` backups).  
`--no_backup` to skip creating backups.

**b) Convert dataset to HDF5:**

```bash
cd act
python convert_to_hdf5.py \
  --dataset_dir /path/to/dataset \
  --output /path/to/output.hdf5 \
  --verify
```

To merge multiple data folders into one HDF5:

```bash
python convert_to_hdf5.py \
  --dataset_dirs /path/to/dataset1 /path/to/dataset2 \
  --output merged.hdf5
```

HDF5 per-episode structure:
- `state_body` (T, 31), `action_body` (T, 35)
- `state_wuji_hand_{left,right}` (T, 20), `action_wuji_qpos_target_{left,right}` (T, 20)
- `head` (T,) JPEG bytes

### 2) Policy Training

```bash
cd act
python imitate_episodes.py \
  --ckpt_root ./checkpoints \
  --policy_class ACT \
  --task_name my_task \
  --batch_size 8 \
  --seed 42 \
  --num_epochs 3000 \
  --lr 1e-5 \
  --dataset_path /path/to/dataset.hdf5 \
  --hand_side right \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --temporal_agg \
  --wandb
```

`--hand_side`: `left`, `right`, or `both`.  
`--sequential_training --epochs_per_dataset 2000 1000`: train on multiple datasets sequentially.  
`--resume --ckpt_dir ./checkpoints/my_task/<run_dir>`: resume from checkpoint.

Checkpoint location: `<ckpt_root>/<task_name>/<timestamp>/`

### 3) Policy Inference

**a) Offline evaluation (policy + dataset observations):**

```bash
cd act
python policy_inference.py eval_offline \
  --ckpt_dir ./checkpoints/my_task/<run_dir> \
  --dataset /path/to/dataset.hdf5 \
  --episode 0 \
  --hand_side right \
  --temporal_agg \
  --save_actions
```

Outputs predicted vs GT action `.npy` files to `ckpt_dir`.

**b) Online evaluation (real-time robot inference):**

```bash
cd act
python policy_inference.py eval_online \
  --ckpt_dir ./checkpoints/my_task/<run_dir> \
  --hand_side right \
  --redis_ip localhost \
  --temporal_agg
```

Toggle inference on/off with keyboard (Space = send, H = hold position).

**c) (Optional) Robot initialization with data initial pose:**

```bash
cd act
python policy_inference.py init_pose \
  --dataset /path/to/dataset.hdf5 \
  --episode 0 \
  --hand_side right \
  --redis_ip localhost \
  --ramp_seconds 3.0
```

Publishes a fixed body+hand action from the dataset and holds until Ctrl-C.
Use before `eval_online` to initialize the robot.
