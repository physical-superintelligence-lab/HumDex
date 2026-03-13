# Data Collection Guide

This document describes robot-side and human-side data collection workflows in this repository.

## 1. Robot Data Collection

### 1.1 Start RealSense ZMQ Publisher on G1

Plug the G1 camera USB into the G1 host, then SSH to `unitree@192.168.123.164` (password: `123`).

Copy `deploy_real/server_realsense_zmq_pub.py` to `~` on G1, then create a dedicated `realsense` conda environment and install dependencies manually.

```bash
# on local workstation (repo root)
scp deploy_real/server_realsense_zmq_pub.py unitree@192.168.123.164:~/
```

```bash
# on g1 after ssh login
conda create -y -n realsense python=3.10
conda activate realsense
python -m pip install --upgrade pip
python -m pip install pyrealsense2 pyzmq numpy opencv-python rich zmq
```

After the environment is ready, start the camera publisher with:

```bash
# from local workstation
bash scripts/realsense_zmq_pub_g1.sh
```

### 1.2 Start Teleop

Start teleoperation first, following [`teleop.md`](teleop.md).

### 1.3 Start Robot Data Recording

Use the teleop recorder:

```bash
bash scripts/data_record.sh

# sonic channel
bash scripts/data_record.sh --channel sonic
```


## 2. Human Data Collection

### 2.1 Wear RealSense on Human Subject

Plug the RealSense USB into the workstation, then wear the RealSense using:

- a 3D-printed adapter from [link](https://drive.google.com/file/d/1PKtpvaxZI7zmqRgvxg64AasXwEP1iWsf/view?usp=sharing)
- a GoPro neck mount from [link](https://www.amazon.com/TELESIN-Magnetic-Release-Necklace-Insta360/dp/B0CZ41S1ZQ/ref=sr_1_1_sspa?crid=2G13WWF52ZKYD&dib=eyJ2IjoiMSJ9.0eDdz0m5oeRrIjXp4IIcJF_SgH-CECEHq6yza_EyAefQ9SxyBDzOkLRAI9O8PH8WH5TlGKEsGbYKtxwlwTUpwzHRxuY59GhFi_XovB6RovVXXvgDO4_AHKdq6cPyAtTxEquAhsHdMem50uXqA-LL8oU8rcHwVFneuSipbTCSxXb4Nd-6ZqDhFlMsF8q-iyrlmQSBI5TrmAGmD4Jh8GqQwzVkWDOkfZMSPRPoOblS9tg.bI1ysBZpCOwbrNvHFJ5RY34iDENihSk5FHbd0SyCth8&dib_tag=se&keywords=GoPro+Neck+Mount&qid=1772824475&sprefix=gopro+neck+mount%2Caps%2C171&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1)

### 2.2 Start Teleop

Start teleoperation first, following [`teleop.md`](teleop.md).

### 2.3 Start Human Data Recording

Use the human recorder:

```bash
bash scripts/data_record_human.sh

# sonic channel
bash scripts/data_record_human.sh --channel sonic
```


## 3. Recording Controls

Both recorders use the same basic controls:

- `r`: start/stop one episode
- `q`: quit recorder

## 4. Data Layout and Saved Structure

By default, both recorders save under:

- `deploy_real/humdex_demonstration/<task_name>/`

where `<task_name>` is generated as:

- `YYYYMMDD_HHMM_<channel>`

Each episode is saved as:

- `episode_0001/`
  - `rgb/` (JPEG frames, e.g. `000000.jpg`)
  - `data.json` (per-frame metadata and states/actions)


Typical per-frame fields in `data.json` include:

- `idx`, `rgb`, `t_img`, `t_record_ms`
- `state_body`, `action_body`
- `hand_tracking_left/right`
- `action_wuji_qpos_target_left/right`, `state_wuji_hand_left/right`
- `t_action`, `t_state`, `t_action_wuji_hand_left/right`, `t_state_wuji_hand_left/right`
- `body_zmq`, `body_zmq_decoded` (when channel is `sonic`)
