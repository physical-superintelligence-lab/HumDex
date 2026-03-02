# HumDex: Humanoid Dexterous Manipulation Made Easy
By Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang

![Banner for HumDex](./assets/HumDex.png)

## Content Table

- [Installation](#installation)
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

pip install python-osc

conda install -c conda-forge libstdcxx-ng -y
```

### 2) Create `humdex` Environment
```bash
conda create -n humdex python=3.8 -y
conda activate humdex

# install wuji-retargeting

cd wuji-retargeting
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

## Teleop

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

- `runtime`: loop settings like `target_fps`, `measure_fps`, `print_every`, `max_steps`, `dry_run`
- `network`: redis + mocap transport (`network.redis`, `network.mocap.default/body/hand`)
- `retarget`: retarget/ramp/smoothing settings
- `control`: keyboard and hand-control toggles
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

## Data Collection

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
