# Teleoperation Quick Guide

This document describes a practical teleoperation startup flow using the scripts in this repository.

## 1. Prepare Your Tracking Devices

Set up your body/hand tracking stack based on your hardware combination:

- VDMocap/VDHand: see `doc/vdmocap_vdhand.md`
- SlimeVR: see `doc/slimevr.md`
- MANUS: see `doc/manus.md`

After device setup is complete, choose the teleop combination in `teleop.sh` (`policy`, `body`, and `hand` selectors).


## 2. Start the G1 Controller

Follow `doc/g1.md` first for robot-side preparation.

### 2.1 Sim Controller

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

### 2.2 Real Controller

```bash
## for --policy twist2
conda activate humdex
bash run_motion_server.sh
bash sim2real.sh

## for --policy sonic
cd ../GR00T-WholeBodyControl/gear_sonic_deploys
source scripts/setup_env.sh
bash deploy.sh real --input-type zmq
```


## 3. Start the Wuji Hand Controller

Follow `doc/wuji.md` first for wuji hand setup.

### 3.1 Sim Hand Controller

```bash
conda activate humdex
bash wuji_hand_sim.sh
```

### 3.2 Real Hand Controller

```bash
conda activate humdex
bash wuji_hand_real.sh
```


## 4. Launch Teleop

Use the unified teleop entry:

```bash
conda activate gmr
bash teleop.sh [options] [-- extra_args]
```

Supported selectors:

- `--policy {twist2|sonic}` (default `twist2`)
- `--body {vdmocap|slimevr}` (default `vdmocap`)
- `--hand {vdhand|manus}` (default `vdhand`)

Common examples:

```bash
# default combo
bash teleop.sh

# sonic + vdmocap + manus
bash teleop.sh --policy sonic --body vdmocap --hand manus

# twist2 + slimevr + vdhand
bash teleop.sh --policy twist2 --body slimevr --hand vdhand
```

Make sure your tracker setup and teleop selectors match the same data sources.