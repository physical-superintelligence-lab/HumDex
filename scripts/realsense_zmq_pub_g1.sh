#!/bin/bash
set -euo pipefail


HOST="${HOST:-unitree@192.168.123.164}"
REMOTE_DIR="${REMOTE_DIR:-~}"
CONDA_ENV="${CONDA_ENV:-realsense}"

PY_ARGS=(--bind 0.0.0.0 --port 5555 --width 640 --height 480 --fps 30 --jpeg_quality 80)
REST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --remote_dir) REMOTE_DIR="$2"; shift 2;;
    --conda_env) CONDA_ENV="$2"; shift 2;;
    -h|--help)
      echo "usage: $0 [--host unitree@192.168.123.164] [--remote_dir ~] [--conda_env realsense] [server_realsense_zmq_pub.py args...]"
      exit 0
      ;;
    *)
      REST_ARGS+=("$1")
      shift
      ;;
  esac
done

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail
source ~/miniconda3/bin/activate "__CONDA_ENV__"
REMOTE_DIR="__REMOTE_DIR__"
if [[ "${REMOTE_DIR}" == "~" ]]; then
  REMOTE_DIR="${HOME}"
elif [[ "${REMOTE_DIR}" == ~/* ]]; then
  REMOTE_DIR="${HOME}/${REMOTE_DIR#~/}"
fi
cd "${REMOTE_DIR}"
sudo killall -9 videohub_pc4 >/dev/null 2>&1 || true
sleep 0.1
exec /usr/bin/python3 ./server_realsense_zmq_pub.py __PY_ARGS__
EOF
)

ALL_ARGS=("${PY_ARGS[@]}" "${REST_ARGS[@]}")
REMOTE_CMD="${REMOTE_CMD/__CONDA_ENV__/${CONDA_ENV}}"
REMOTE_CMD="${REMOTE_CMD/__REMOTE_DIR__/${REMOTE_DIR}}"
REMOTE_CMD="${REMOTE_CMD/__PY_ARGS__/${ALL_ARGS[*]}}"

echo "[local] ssh ${HOST} Start RealSense ZMQ PUB..."
ssh -t "${HOST}" "bash -lc $(printf '%q' "${REMOTE_CMD}")"


