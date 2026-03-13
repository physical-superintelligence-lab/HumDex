# Teleoperation Quick Guide

This document describes a practical teleoperation startup flow using the scripts in this repository.

## 1. Prepare Your Tracking Devices

Set up your body/hand tracking stack based on your hardware combination:

- VDMocap/VDHand: see [`vdmocap_vdhand.md`](vdmocap_vdhand.md)
- SlimeVR: see [`slimevr.md`](slimevr.md)
- MANUS: see [`manus.md`](manus.md)

After device setup is complete, choose the teleop combination in `teleop.sh` (`policy`, `body`, and `hand` selectors).


## 2. Start the G1 Controller

Follow [`g1.md`](g1.md) first for robot-side preparation.

### 2.1 Sim Controller

```bash
## for --policy twist2
conda activate humdex
# Warm arp the redis server at first time
bash scripts/run_motion_server.sh
bash scripts/sim2sim.sh



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
bash scripts/run_motion_server.sh
bash scripts/sim2real.sh



## for --policy sonic
cd ../GR00T-WholeBodyControl/gear_sonic_deploys
source scripts/setup_env.sh
bash deploy.sh real --input-type zmq
```


## 3. Start the Wuji Hand Controller

Follow [`wuji.md`](wuji.md) first for wuji hand setup.

### 3.1 Sim Hand Controller

```bash
conda activate humdex
bash scripts/wuji_hand_sim.sh
```

### 3.2 Real Hand Controller

```bash
conda activate humdex
bash scripts/wuji_hand_real.sh
```


## 4. Launch Teleop

Use the unified teleop entry:

```bash
conda activate gmr
bash scripts/teleop.sh [options] [-- extra_args]
```

Supported selectors:

- `--policy {twist2|sonic}` (default `twist2`)
- `--body {vdmocap|slimevr}` (default `vdmocap`)
- `--hand {vdhand|manus}` (default `vdhand`)

Common examples:

```bash
# default combo
bash scripts/teleop.sh

# sonic + vdmocap + manus
bash scripts/teleop.sh --policy sonic --body vdmocap --hand manus

# twist2 + slimevr + vdhand
bash scripts/teleop.sh --policy twist2 --body slimevr --hand vdhand
```

Keyboard controls during teleop:

- `k`: toggle between **send mode** and **default mode**.
  - send mode: live retargeted commands are sent to the robot.
  - default mode: the robot returns to (or stays at) the configured safe default pose.
- `p`: toggle **hold mode**.
  - hold mode on: freeze at the current command/pose and stop following new incoming teleop motion.
  - hold mode off: resume normal live following.

Make sure your tracker setup and teleop selectors match the same data sources.