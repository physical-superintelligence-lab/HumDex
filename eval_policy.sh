#!/usr/bin/env bash
set -euo pipefail

# TWIST2 real-robot evaluation helper.
# Edit the variables below, then run:
#   bash eval_policy.sh
#
# Notes:
# - Assumes your low-level controller (e.g. deploy_real/server_low_level_g1_real*.py) is already running and reading Redis.
# - Use k/p safety toggles in replay/infer:
#   - k: send safe idle (stop executing dataset/policy actions)
#   - p: hold last commanded pose (keep publishing cached action)

# -----------------------------------------------------------------------------
# Make this script self-contained (avoid relying on an activated conda env).
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

TWIST2_ENV_PREFIX="${TWIST2_ENV_PREFIX:-/home/heng/miniconda3/envs/twist2}"
PYTHON_BIN="${TWIST2_PYTHON:-${TWIST2_ENV_PREFIX}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Could not find Python at: ${PYTHON_BIN}"
  echo "   - Set TWIST2_PYTHON=/path/to/python or TWIST2_ENV_PREFIX=/path/to/env"
  exit 1
fi

# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------
DATASET="/home/heng/heng/G1/TWIST2/act/data/20250125_opendoor_robot_31D.hdf5"
CKPT_DIR="/home/heng/heng/G1/TWIST2/act/ckpt/open_door/pickball/chunk400_20260128_020303"
# CKPT_DIR="/home/heng/heng/G1/TWIST2/act/ckpt/pickbasket/robot"
EPISODE=23
### 25
# 32 35 29
# 45 62 59 57 56
#    66 -- 73
#    47 49
#    11
FREQ=30

CKPT_FILE="policy_last.ckpt"  # policy_best.ckpt | policy_last.ckpt

# Vision (robot RGB stream)
VISION_IP="192.168.123.164"
VISION_PORT=5555

# Redis key namespace (must match low-level server)
ROBOT_KEY="unitree_g1_with_hands"

# Policy architecture (must match training checkpoint)
CHUNK_SIZE=400
HIDDEN_DIM=512
DIM_FEEDFORWARD=3200

# Fake-obs inference (dataset-driven) start timestep
START_TIMESTEP=0

# ---------------------------------------------------------------------
# 1) Move robot to init pose by holding dataset action (Ctrl-C when ready)
# ---------------------------------------------------------------------
"${PYTHON_BIN}" deploy_real/policy_inference.py init_pose \
  --dataset "${DATASET}" \
  --episode "${EPISODE}" \
  --timestep 0 \
  --frequency "${FREQ}" \
  --redis_ip localhost \
  --robot_key "${ROBOT_KEY}" \
  --hand_side both \
  --ramp_seconds 3.0 \
  --ramp_ease cosine \
  --toggle_ramp_seconds 3.0 \
  --toggle_ramp_ease cosine

# ---------------------------------------------------------------------
# 2) Replay on real robot + RGB stream + sim preview
# ---------------------------------------------------------------------
# python deploy_real/policy_inference.py replay \
#   --dataset "${DATASET}" \
#   --episode "${EPISODE}" \
#   --frequency "${FREQ}" \
#   --redis_ip localhost \
#   --vision_ip "${VISION_IP}" --vision_port "${VISION_PORT}" \
#   --hand_side both \
#   # --sim_only --sim_save_vid /home/heng/heng/G1/TWIST2/act/test/replay_hang_bothhands.mp4

# ---------------------------------------------------------------------
# 3A) Policy inference on real robot (REAL obs) + RGB stream + sim preview
# # # ---------------------------------------------------------------------
"${PYTHON_BIN}" deploy_real/policy_inference.py infer \
  --ckpt_dir "${CKPT_DIR}" \
  --ckpt_file "${CKPT_FILE}" \
  --frequency "${FREQ}" \
  --redis_ip localhost \
  --vision_ip "${VISION_IP}" --vision_port "${VISION_PORT}" \
  --chunk_size "${CHUNK_SIZE}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --dim_feedforward "${DIM_FEEDFORWARD}" \
  --obs_source real \
  --max_timesteps 2000 --temporal_agg --hand_side both  --state_body_31d --save_rgb_video \
  --toggle_ramp_seconds 3.0 --toggle_ramp_ease cosine 

# ---------------------------------------------------------------------
# 3B) Policy inference on real robot (FAKE obs from dataset) + RGB stream + sim preview
#     NOTE: policy input uses dataset (qpos,image), but actions are still published to Redis (real robot executes).
# # ---------------------------------------------------------------------
# "${PYTHON_BIN}" deploy_real/policy_inference.py infer \
#   --ckpt_dir "${CKPT_DIR}" \
#   --frequency "${FREQ}" \
#   --redis_ip localhost \
#   --vision_ip "${VISION_IP}" --vision_port "${VISION_PORT}" \
#   --chunk_size "${CHUNK_SIZE}" \
#   --hidden_dim "${HIDDEN_DIM}" \
#   --dim_feedforward "${DIM_FEEDFORWARD}" \
#   --obs_source dataset \
#   --dataset "${DATASET}" \
#   --episode "${EPISODE}" \
#   --start_timestep "${START_TIMESTEP}" \
#   --max_timesteps 2000 --sim_save_vid --save_rgb_video \
#   --temporal_agg --hand_side both --state_body_31d \
#   --toggle_ramp_seconds 3.0 --toggle_ramp_ease cosine


# ---------------------------------------------------------------------
# 3C) Policy inference on real robot (VIDEO obs from dataset) + RGB stream + sim preview
# # ---------------------------------------------------------------------
# "${PYTHON_BIN}" deploy_real/policy_inference.py infer \
#   --ckpt_dir "${CKPT_DIR}" \
#   --frequency "${FREQ}" \
#   --redis_ip localhost \
#   --vision_ip "${VISION_IP}" --vision_port "${VISION_PORT}" \
#   --chunk_size "${CHUNK_SIZE}" \
#   --hidden_dim "${HIDDEN_DIM}" \
#   --dim_feedforward "${DIM_FEEDFORWARD}" \
#   --obs_source video \
#   --dataset "${DATASET}" \
#   --episode "${EPISODE}" \
#   --start_timestep "${START_TIMESTEP}" \
#   --video_path "/abs/path/to/offline_video.mp4" \
#   --video_loop \
#   --max_timesteps 2000 --sim_save_vid --save_rgb_video \
#   --temporal_agg --hand_side both --state_body_31d \
#   --toggle_ramp_seconds 3.0 --toggle_ramp_ease cosine


# ---------------------------------------------------------------------
# 4) OFFLINE EVAL: Dataset -> policy inference -> SIM ONLY (no Redis publish)
#     Output: 2x2 grid video (Body GT | Body Pred, Hand GT | Hand Pred) saved to ckpt_dir/eval_ep{episode}.mp4
#     NOTE: This is pure simulation visualization, does NOT publish actions to Redis.
# ---------------------------------------------------------------------
# python deploy_real/policy_inference.py eval \
#   --ckpt_dir "${CKPT_DIR}" \
#   --dataset "${DATASET}" \
#   --episode "${EPISODE}" \
#   --chunk_size "${CHUNK_SIZE}" \
#   --hidden_dim "${HIDDEN_DIM}" \
#   --dim_feedforward "${DIM_FEEDFORWARD}" \
#   --temporal_agg --hand_side both --state_body_31d

# ---------------------------------------------------------------------
# 5) FAKE-obs inference (dataset) -> publish to real + record sim video WITH hand visualization
# ---------------------------------------------------------------------
# python deploy_real/policy_inference.py infer \
#   --ckpt_dir "${CKPT_DIR}" \
#   --frequency "${FREQ}" \
#   --redis_ip localhost \
#   --vision_ip "${VISION_IP}" --vision_port "${VISION_PORT}" \
#   --chunk_size "${CHUNK_SIZE}" \
#   --hidden_dim "${HIDDEN_DIM}" \
#   --dim_feedforward "${DIM_FEEDFORWARD}" \
#   --obs_source dataset \
#   --dataset "${DATASET}" \
#   --episode "${EPISODE}" \
#   --start_timestep "${START_TIMESTEP}" \
#   --max_timesteps 800 \
#   --sim_save_vid \
#   --sim_hand \
#   --temporal_agg --record_run --hand_side both



