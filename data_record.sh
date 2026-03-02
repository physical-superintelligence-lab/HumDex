#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

# Redis（teleop/sim2real 写入的那个 Redis）
redis_ip="localhost"
channel="twist2"   # twist2 | sonic
sonic_body_backend="zmq"  # redis | zmq
body_zmq_ip="127.0.0.1"
body_zmq_port=5556
body_zmq_topic="pose"

# 图像服务器（在 g1 上跑的相机发布端 IP/端口）
vision_ip="192.168.123.164"
vision_port=5555

data_frequency=30
task_name_base=$(date +"%Y%m%d_%H%M")
task_name="${task_name_base}_${channel}"

python server_data_record.py \
  --redis_ip "${redis_ip}" \
  --channel "${channel}" \
  --sonic_body_backend "${sonic_body_backend}" \
  --body_zmq_ip "${body_zmq_ip}" \
  --body_zmq_port "${body_zmq_port}" \
  --body_zmq_topic "${body_zmq_topic}" \
  --frequency "${data_frequency}" \
  --task_name "${task_name}" \
  --vision_backend zmq \
  --vision_ip "${vision_ip}" \
  --vision_port "${vision_port}" \
  --save_episode_video \
  # --keyboard_backend evdev \
  # --evdev_device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
  # "$@"


