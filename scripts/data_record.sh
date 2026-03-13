#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../deploy_real"

channel="twist2"   # twist2 | sonic
redis_ip="localhost"
body_zmq_ip="127.0.0.1"
body_zmq_port=5556
body_zmq_topic="pose"

# Vision server endpoint running on g1
vision_ip="192.168.123.164"
vision_port=5555

data_frequency=30
task_name_base=$(date +"%Y%m%d_%H%M")
task_name="${task_name_base}_${channel}"

python server_data_record.py \
  --redis_ip "${redis_ip}" \
  --channel "${channel}" \
  --body_zmq_ip "${body_zmq_ip}" \
  --body_zmq_port "${body_zmq_port}" \
  --body_zmq_topic "${body_zmq_topic}" \
  --frequency "${data_frequency}" \
  --task_name "${task_name}" \
  --vision_backend zmq \
  --vision_ip "${vision_ip}" \
  --vision_port "${vision_port}" \
  --save_episode_video \
  "$@"


