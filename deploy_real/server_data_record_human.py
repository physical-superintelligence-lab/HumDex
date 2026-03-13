#!/usr/bin/env python3
"""
Record-only data pipeline (no robot or hand hardware control):

- Optional: launch vdmocap->Redis in background via `vdmocap_teleop.sh`.
- Main process responsibilities:
  - Capture RealSense frames locally.
  - Read state/action/hand-tracking keys from Redis.
  - Write episodes with EpisodeWriter.

Keyboard:
- r: start/stop recording
- q: quit

Safety:
- This script intentionally does not start sim2real or Wuji control.
  Deprecated flags such as `--start_sim2real/--start_wuji` are ignored.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import redis
from rich import print

# Ensure repo root is on sys.path so `import deploy_real.*` works even when
# launching from inside `deploy_real/` (e.g. via `bash data_record_xdmocap_oneclick.sh`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy_real.data_utils.episode_writer import EpisodeWriter


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_json_loads(raw: Optional[bytes]) -> Any:
    if raw is None:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


def _to_builtin(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    return x


class SonicBodyZmqSource:
    def __init__(self, ip: str, port: int, topic: str):
        import zmq  # type: ignore

        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, str(topic))
        self._sock.connect(f"tcp://{ip}:{int(port)}")
        self._topic = str(topic)
        self._latest: Optional[Dict[str, Any]] = None
        self._unpack_fn = None
        for mod in [
            "gear_sonic.utils.teleop.zmq.zmq_planner_sender",
            "gear_sonic.utils.zmq_utils",
        ]:
            try:
                m = __import__(mod, fromlist=["unpack_pose_message"])
                fn = getattr(m, "unpack_pose_message", None)
                if callable(fn):
                    self._unpack_fn = fn
                    break
            except Exception:
                continue

    def _decode(self, raw_msg: bytes) -> Dict[str, Any]:
        payload = raw_msg
        if raw_msg.startswith((self._topic + " ").encode("utf-8")):
            payload = raw_msg.split(b" ", 1)[1]
        decoded: Any = None
        if callable(self._unpack_fn):
            try:
                decoded = self._unpack_fn(raw_msg)  # type: ignore[misc]
            except Exception:
                decoded = None
        if decoded is None:
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except Exception:
                decoded = None
        return {
            "timestamp_ms": now_ms(),
            "topic": self._topic,
            "decoded": _to_builtin(decoded) if decoded is not None else None,
            "raw_b64": base64.b64encode(payload).decode("ascii"),
        }

    def get_latest(self) -> Optional[Dict[str, Any]]:
        zmq = self._zmq
        updated = False
        while True:
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
                self._latest = self._decode(raw)
                updated = True
            except zmq.Again:
                break
            except Exception:
                break
        if updated or (self._latest is not None):
            return dict(self._latest) if isinstance(self._latest, dict) else self._latest
        return None

    def close(self):
        try:
            self._sock.close(0)
        except Exception:
            pass
        try:
            self._ctx.term()
        except Exception:
            pass


# -----------------------------
# Optional: local Wuji retarget (no wuji_server required)
# -----------------------------

# 26D hand joint names aligned with teleop hand_tracking_* keys.
HAND_JOINT_NAMES_26 = [
    "Wrist",
    "Palm",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]

# Mapping indices from 26D hand format to MediaPipe 21D format.
# MediaPipe format: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26D format: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,  # 0: Wrist -> Wrist (use Palm as Wrist to stay aligned with server_wuji_hand_redis.py)
    2,  # 1: ThumbMetacarpal -> Thumb CMC
    3,  # 2: ThumbProximal -> Thumb MCP
    4,  # 3: ThumbDistal -> Thumb IP
    5,  # 4: ThumbTip -> Thumb Tip
    6,  # 5: IndexMetacarpal -> Index MCP
    7,  # 6: IndexProximal -> Index PIP
    8,  # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (skip IndexDistal)
    11,  # 9: MiddleMetacarpal -> Middle MCP
    12,  # 10: MiddleProximal -> Middle PIP
    13,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (skip MiddleDistal)
    16,  # 13: RingMetacarpal -> Ring MCP
    17,  # 14: RingProximal -> Ring PIP
    18,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (skip RingDistal)
    21,  # 17: LittleMetacarpal -> Pinky MCP
    22,  # 18: LittleProximal -> Pinky PIP
    23,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (skip LittleDistal)
]


def hand_26d_to_mediapipe_21d(hand_data_dict: Dict[str, Any], hand_side: str = "left") -> np.ndarray:
    """
    Convert a 26D hand_tracking dict into 21D MediaPipe coordinates (21, 3).
    This implementation is aligned with
    `deploy_real/server_wuji_hand_redis.py::hand_26d_to_mediapipe_21d`.
    """
    side = str(hand_side).lower()
    prefix = "LeftHand" if side == "left" else "RightHand"
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = prefix + joint_name
        v = hand_data_dict.get(key, None)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            pos = np.asarray(v[0], dtype=np.float32).reshape(3)
            joint_positions_26[i] = pos
        else:
            joint_positions_26[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    mp21 = joint_positions_26[np.asarray(MEDIAPIPE_MAPPING_26_TO_21, dtype=np.int32)]
    wrist_pos = mp21[0].copy()
    mp21 = mp21 - wrist_pos
    # Keep scale_factor=1.0 for consistency (no additional scaling here).
    return mp21


def _try_import_wuji_retargeting():
    """
    Lazy import to avoid hard dependency when user doesn't need local retarget.
    Returns (WujiHandRetargeter, apply_mediapipe_transformations) or (None, None)
    """
    try:
        project_root = Path(__file__).resolve().parents[1]
        wuji_path = project_root / "wuji_retargeting"
        if str(wuji_path) not in sys.path:
            sys.path.insert(0, str(wuji_path))
        from wuji_retargeting import WujiHandRetargeter  # type: ignore
        from wuji_retargeting.mediapipe import apply_mediapipe_transformations  # type: ignore

        return WujiHandRetargeter, apply_mediapipe_transformations
    except Exception:
        return None, None


def _try_import_training():
    """
    Lazy import training model package from repo's `wuji_retarget/`.
    Returns training module or None.
    """
    try:
        project_root = Path(__file__).resolve().parents[1]
        training_root = project_root / "wuji_retarget"
        if str(training_root) not in sys.path:
            sys.path.insert(0, str(training_root))
        import training  # type: ignore

        return training
    except Exception:
        return None


class RealSenseVisionSource:
    """Direct RealSense capture (color; optional depth) via pyrealsense2."""

    def __init__(self, width: int, height: int, fps: int, enable_depth: bool = False):
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as e:
            raise ImportError("pyrealsense2 is required for --vision_backend realsense.") from e

        self._rs = rs
        self._enable_depth = bool(enable_depth)
        self._running = True
        self._lock = None  # lazy: avoid threading unless needed

        self._latest_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self._latest_depth: Optional[np.ndarray] = None

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if self._enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)

    def get_rgb(self) -> np.ndarray:
        rs = self._rs
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if color:
                self._latest_rgb = np.asanyarray(color.get_data())
            if self._enable_depth:
                depth = frames.get_depth_frame()
                self._latest_depth = None if not depth else np.asanyarray(depth.get_data())
        except Exception:
            pass
        return self._latest_rgb.copy()

    def get_depth(self) -> Optional[np.ndarray]:
        return None if self._latest_depth is None else self._latest_depth.copy()

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_folder = os.path.join(here, "humdex_demonstration")

    p = argparse.ArgumentParser(description="VDMocap/Redis + RealSense recorder (record-only, no robot/hand control)")

    # recording
    p.add_argument("--data_folder", default=default_data_folder)
    p.add_argument("--task_name", default=cur_time)
    p.add_argument("--frequency", type=int, default=30)
    p.add_argument("--robot_key", default="unitree_g1_with_hands")
    p.add_argument("--channel", choices=["twist2", "sonic"], default="twist2")
    p.add_argument("--body_zmq_ip", default="127.0.0.1")
    p.add_argument("--body_zmq_port", type=int, default=5556)
    p.add_argument("--body_zmq_topic", default="pose")
    p.add_argument("--no_window", action="store_true", help="Disable preview window (keyboard control unavailable in no-window mode)")
    p.add_argument("--record_on_start", action="store_true", help="Start recording immediately on launch (useful with --no_window)")

    # redis
    p.add_argument("--redis_ip", default="localhost")
    p.add_argument("--redis_port", type=int, default=6379)

    # local wuji retarget (record-only; does NOT control hardware)
    p.add_argument("--local_wuji_retarget", type=int, default=1, help="Enable local wuji qpos target computation on recorder side (0/1, default=1)")
    p.add_argument(
        "--local_wuji_retarget_overwrite",
        type=int,
        default=1,
        help="If Redis already has action_wuji_qpos_target_*, overwrite with local retarget output (0/1)",
    )
    p.add_argument("--local_wuji_write_redis", type=int, default=1, help="Write local retarget output back to Redis (0/1)")

    # local wuji mode: DexPilot retarget (default) vs training model inference
    p.add_argument("--local_wuji_use_model", type=int, default=0, help="Use training model inference for local wuji target generation (0/1, default=0)")
    # Keep argument names aligned with deploy2.py / wuji_hand_model_deploy.sh.
    p.add_argument("--local_wuji_policy_tag", type=str, default="training_filter_wuji", help="Local training model tag (--local_wuji_use_model=1)")
    p.add_argument("--local_wuji_policy_epoch", type=int, default=-1, help="Local training model epoch (--local_wuji_use_model=1)")
    p.add_argument("--local_wuji_policy_tag_left", type=str, default="", help="Left-hand tag (optional; empty uses local_wuji_policy_tag)")
    p.add_argument("--local_wuji_policy_epoch_left", type=int, default=-999999, help="Left-hand epoch (optional; -999999 uses local_wuji_policy_epoch)")
    p.add_argument("--local_wuji_policy_tag_right", type=str, default="", help="Right-hand tag (optional; empty uses local_wuji_policy_tag)")
    p.add_argument("--local_wuji_policy_epoch_right", type=int, default=-999999, help="Right-hand epoch (optional; -999999 uses local_wuji_policy_epoch)")
    p.add_argument("--local_wuji_use_fingertips5", type=int, default=1, help="Use 5 fingertips (5,3) as model input (0/1, default=1)")
    # Safety limits for model mode to avoid spikes/out-of-range outputs.
    p.add_argument("--local_wuji_clamp_min", type=float, default=-1.5, help="Minimum clamp value for model output")
    p.add_argument("--local_wuji_clamp_max", type=float, default=1.5, help="Maximum clamp value for model output")
    p.add_argument("--local_wuji_max_delta_per_step", type=float, default=0.08, help="Maximum per-step output delta for model output")

    # vision
    p.add_argument("--rs_w", type=int, default=640)
    p.add_argument("--rs_h", type=int, default=480)
    p.add_argument("--rs_fps", type=int, default=30)
    p.add_argument("--rs_depth", action="store_true")

    # vdmocap teleop (as subprocess). Keep old xdmocap flags as aliases.
    p.add_argument("--start_vdmocap", "--start_xdmocap", dest="start_vdmocap", action="store_true", help="Start vdmocap_teleop.sh in background")
    p.add_argument(
        "--vdmocap_teleop_sh",
        "--xdmocap_teleop_sh",
        dest="vdmocap_teleop_sh",
        type=str,
        default=os.path.join(os.path.dirname(here), "vdmocap_teleop.sh"),
    )
    p.add_argument(
        "--vdmocap_extra_args",
        "--xdmocap_extra_args",
        dest="vdmocap_extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed to vdmocap_teleop.sh (after --)",
    )

    # Deprecated compatibility args: kept but intentionally ignored.
    p.add_argument("--start_sim2real", action="store_true", help="(deprecated/ignored) kept for compatibility; this script never starts robot control")
    p.add_argument("--policy", type=str, default=os.path.join(os.path.dirname(here), "assets/ckpts/twist2_1017_20k.onnx"))
    p.add_argument("--config", type=str, default="robot_control/configs/g1.yaml")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--net", type=str, default="eno1")
    p.add_argument("--use_hand", action="store_true", help="Enable Unitree gripper hand (Dex3_1_Controller)")
    p.add_argument("--record_proprio", action="store_true")
    p.add_argument("--smooth_body", type=float, default=0.0)
    p.add_argument("--safety_rate_limit", action="store_true")
    p.add_argument("--safety_rate_limit_scope", choices=["all", "arms"], default="arms")
    p.add_argument("--max_dof_delta_per_step", type=float, default=1.0)
    p.add_argument("--max_dof_delta_print_every", type=int, default=200)

    # Deprecated compatibility args: kept but intentionally ignored.
    p.add_argument("--start_wuji", action="store_true", help="(deprecated/ignored) kept for compatibility; this script never starts Wuji hand control")
    p.add_argument("--wuji_hands", choices=["none", "left", "right", "both"], default="right")
    p.add_argument("--wuji_target_fps", type=int, default=50)
    p.add_argument("--wuji_no_smooth", action="store_true")
    p.add_argument("--wuji_smooth_steps", type=int, default=5)
    p.add_argument("--wuji_left_serial", type=str, default="")
    p.add_argument("--wuji_right_serial", type=str, default="")

    return p.parse_args()


def build_redis_key_candidates(channel: str, suffix: str, *, body_from_redis: bool = True) -> List[tuple[str, List[str]]]:
    base: List[tuple[str, List[str]]] = [
        ("state_body", [f"state_body_{suffix}"]) if body_from_redis else ("state_body", []),
        ("t_state", ["t_state"]) if body_from_redis else ("t_state", []),
        ("action_body", [f"action_body_{suffix}"]) if body_from_redis else ("action_body", []),
        ("t_action", ["t_action"]),
        ("hand_tracking_left", [f"hand_tracking_left_{suffix}"]),
        ("hand_tracking_right", [f"hand_tracking_right_{suffix}"]),
        ("action_wuji_qpos_target_left", [f"action_wuji_qpos_target_left_{suffix}"]),
        ("action_wuji_qpos_target_right", [f"action_wuji_qpos_target_right_{suffix}"]),
        ("t_action_wuji_hand_left", [f"t_action_wuji_hand_left_{suffix}"]),
        ("t_action_wuji_hand_right", [f"t_action_wuji_hand_right_{suffix}"]),
        ("state_wuji_hand_left", [f"state_wuji_hand_left_{suffix}"]),
        ("state_wuji_hand_right", [f"state_wuji_hand_right_{suffix}"]),
        ("t_state_wuji_hand_left", [f"t_state_wuji_hand_left_{suffix}"]),
        ("t_state_wuji_hand_right", [f"t_state_wuji_hand_right_{suffix}"]),
    ]
    if str(channel).lower() != "sonic":
        return base
    with_fallback: List[tuple[str, List[str]]] = []
    for dk, cands in base:
        c = list(cands)
        if c and c[0].endswith(f"_{suffix}") and dk not in ["hand_tracking_left", "hand_tracking_right"]:
            c.append(c[0].replace(f"_{suffix}", f"_sonic_{suffix}"))
        with_fallback.append((dk, c))
    return with_fallback


def main() -> int:
    args = parse_args()

    # Redis connection (main recorder)
    try:
        pool = redis.ConnectionPool(
            host=args.redis_ip,
            port=args.redis_port,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.2,
            socket_connect_timeout=0.2,
        )
        client = redis.Redis(connection_pool=pool)
        pipe = client.pipeline()
        client.ping()
    except Exception as e:
        print(f"[ERROR] Failed to connect to Redis: {e}")
        return 2

    # Start vdmocap teleop as background subprocess
    xdm_proc: Optional[subprocess.Popen] = None
    if bool(args.start_vdmocap):
        teleop_sh = os.path.abspath(str(args.vdmocap_teleop_sh))
        if not os.path.exists(teleop_sh):
            print(f"[ERROR] vdmocap_teleop.sh not found: {teleop_sh}")
            return 2
        cmd = ["bash", teleop_sh] + list(args.vdmocap_extra_args or [])
        print(f"[vdmocap] starting: {' '.join(cmd)}")
        xdm_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

    # Safety: explicitly ignore controller-start flags (compatibility only)
    if bool(args.start_sim2real):
        print("[WARN] --start_sim2real is ignored: this is a record-only script and never starts robot control.")
    if bool(args.start_wuji):
        print("[WARN] --start_wuji is ignored: this is a record-only script and never starts Wuji hand control.")

    # Vision (RealSense required)
    vision = None
    try:
        vision = RealSenseVisionSource(
            width=int(args.rs_w),
            height=int(args.rs_h),
            fps=int(args.rs_fps),
            enable_depth=bool(args.rs_depth),
        )
    except Exception as e:
        print(f"[ERROR] RealSense initialization failed: {e}")
        if xdm_proc is not None:
            try:
                os.killpg(os.getpgid(xdm_proc.pid), signal.SIGTERM)
            except Exception:
                pass
        return 3

    # Recorder (schema: compatible superset of data_record.sh + keyboard recorder)
    task_dir = os.path.join(str(args.data_folder), str(args.task_name))
    data_keys = ["rgb"]
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=int(args.frequency),
        image_shape=(int(args.rs_h), int(args.rs_w), 3),
        data_keys=data_keys,
    )

    suffix = str(args.robot_key)
    use_body_zmq = (str(args.channel).lower() == "sonic")
    key_specs = build_redis_key_candidates(channel=args.channel, suffix=suffix, body_from_redis=(not use_body_zmq))
    flat_keys: List[str] = []
    for _dk, cands in key_specs:
        for k in cands:
            if k not in flat_keys:
                flat_keys.append(k)
    body_zmq: Optional[SonicBodyZmqSource] = None
    if use_body_zmq:
        body_zmq = SonicBodyZmqSource(
            ip=str(args.body_zmq_ip),
            port=int(args.body_zmq_port),
            topic=str(args.body_zmq_topic),
        )

    window_name = "TWIST2 OneClick Recorder (r=start/stop, q=quit)"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    recording = False
    step_count = 0
    control_dt = 1.0 / max(1.0, float(args.frequency))

    stop_requested = False

    def _handle_sig(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print("=" * 70)
    print("VDMocap/Redis Recorder (record-only)")
    print("=" * 70)
    print(f"- save_to: {task_dir}")
    print(f"- channel: {args.channel}")
    print(f"- redis: {args.redis_ip}:{args.redis_port}  suffix={suffix}")
    print("- realsense: True")
    print(f"- rs: {args.rs_w}x{args.rs_h}@{args.rs_fps} depth={bool(args.rs_depth)}")
    print(f"- start_vdmocap: {bool(args.start_vdmocap)}")
    print(f"- record_on_start: {bool(args.record_on_start)}")
    print("Keys: r=start/stop, q=quit (note: keyboard control is unavailable with --no_window)")
    print("=" * 70)

    if bool(args.record_on_start):
        if recorder.create_episode():
            recording = True
            step_count = 0
            print("[OK] episode recording started (record_on_start)")
        else:
            recording = False

    # Optional: local Wuji retargeter init (lazy)
    WujiHandRetargeter = None
    apply_mediapipe_transformations = None
    retargeter_left = None
    retargeter_right = None
    training_mod = None
    model_left = None
    model_right = None
    last_wuji_left = None
    last_wuji_right = None
    _warned_local_wuji = False

    try:
        while not stop_requested:
            t0 = time.time()
            if vision is not None:
                rgb = vision.get_rgb()
            else:
                rgb = np.zeros((int(args.rs_h), int(args.rs_w), 3), dtype=np.uint8)

            if not args.no_window:
                overlay = rgb.copy()
                status = "REC: ON" if recording else "REC: OFF"
                cv2.putText(
                    overlay,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if recording else (0, 0, 255),
                    2,
                )
                cv2.putText(overlay, "press r=start/stop, q=quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
                    if recording:
                        if recorder.create_episode():
                            step_count = 0
                            print("[OK] episode recording started")
                        else:
                            recording = False
                    else:
                        recorder.save_episode()
                        print("[OK] episode saving triggered")

            if recording:
                data: Dict[str, Any] = {"idx": step_count}
                if vision is not None:
                    data["rgb"] = rgb
                    data["t_img"] = now_ms()
                data["t_record_ms"] = now_ms()
                if vision is not None and bool(args.rs_depth):
                    depth = vision.get_depth()
                    data["depth"] = depth.tolist() if depth is not None else None

                try:
                    for k in flat_keys:
                        pipe.get(k)
                    results = pipe.execute()
                    kv = {k: v for k, v in zip(flat_keys, results)}
                    for dk, cands in key_specs:
                        raw = None
                        for k in cands:
                            v = kv.get(k, None)
                            if v is not None:
                                raw = v
                                break
                        data[dk] = safe_json_loads(raw)
                except Exception as e:
                    print(f"[WARN] Redis read error: {e}")
                    continue
                if body_zmq is not None:
                    zmq_packet = body_zmq.get_latest()
                    data["body_zmq"] = zmq_packet
                    if isinstance(zmq_packet, dict):
                        decoded = zmq_packet.get("decoded", None)
                        if isinstance(decoded, dict):
                            data["body_zmq_decoded"] = decoded


                # Local retarget: hand_tracking_* -> action_wuji_qpos_target_* (even without wuji_server)
                if bool(int(args.local_wuji_retarget)):
                    if WujiHandRetargeter is None or apply_mediapipe_transformations is None:
                        WujiHandRetargeter, apply_mediapipe_transformations = _try_import_wuji_retargeting()
                        if (WujiHandRetargeter is None or apply_mediapipe_transformations is None) and (not _warned_local_wuji):
                            _warned_local_wuji = True
                            print("[WARN] Local wuji retarget initialization failed; action_wuji_qpos_target_* generation will be skipped.")
                    if bool(int(args.local_wuji_use_model)) and training_mod is None:
                        training_mod = _try_import_training()
                        if training_mod is None and (not _warned_local_wuji):
                            _warned_local_wuji = True
                            print("[WARN] Local wuji model initialization failed (cannot import training); action_wuji_qpos_target_* generation will be skipped.")

                    def _maybe_retarget_one(side: str) -> None:
                        nonlocal retargeter_left, retargeter_right, training_mod, model_left, model_right, last_wuji_left, last_wuji_right
                        if WujiHandRetargeter is None or apply_mediapipe_transformations is None:
                            return
                        s = str(side).lower()
                        assert s in ["left", "right"]

                        key_ht = f"hand_tracking_{s}"
                        key_q = f"action_wuji_qpos_target_{s}"
                        key_t = f"t_action_wuji_hand_{s}"


                        # If already present and not overwriting, keep Redis value.
                        if  (not bool(int(args.local_wuji_retarget_overwrite))):
                            return


                        ht = data.get(key_ht, None)
                        if not isinstance(ht, dict):
                            return
                        if not bool(ht.get("is_active", False)):
                            return

                        hand_dict = {k: v for k, v in ht.items() if k not in ["is_active", "timestamp"]}
                        mp21 = hand_26d_to_mediapipe_21d(hand_dict, hand_side=s)
                        mp_trans = apply_mediapipe_transformations(mp21, hand_type=s)

                        # Choose DexPilot retarget (default) vs training model inference
                        if bool(int(args.local_wuji_use_model)):
                            if training_mod is None:
                                return
                            # per-side tag/epoch override
                            tag = str(args.local_wuji_policy_tag_left if s == "left" and str(args.local_wuji_policy_tag_left) else
                                      args.local_wuji_policy_tag_right if s == "right" and str(args.local_wuji_policy_tag_right) else
                                      args.local_wuji_policy_tag)
                            epoch_default = int(args.local_wuji_policy_epoch)
                            epoch_override = int(args.local_wuji_policy_epoch_left) if s == "left" else int(args.local_wuji_policy_epoch_right)
                            epoch = epoch_default if epoch_override == -999999 else epoch_override

                            if s == "left":
                                if model_left is None:
                                    print(f"[local_wuji:model] loading left: tag={tag}, epoch={epoch}")
                                    model_left = training_mod.load_model(tag, epoch=epoch)
                                    try:
                                        model_left.eval()
                                    except Exception:
                                        pass
                                model = model_left
                            else:
                                if model_right is None:
                                    print(f"[local_wuji:model] loading right: tag={tag}, epoch={epoch}")
                                    model_right = training_mod.load_model(tag, epoch=epoch)
                                    try:
                                        model_right.eval()
                                    except Exception:
                                        pass
                                model = model_right

                            pts21 = np.asarray(mp_trans, dtype=np.float32).reshape(21, 3)
                            if bool(int(args.local_wuji_use_fingertips5)):
                                human_points = pts21[[4, 8, 12, 16, 20], :3]  # (5,3)
                            else:
                                human_points = pts21
                            q = model.forward(human_points)
                            q = np.asarray(q, dtype=np.float32).reshape(-1)
                            if q.shape[0] != 20:
                                return
                            wuji_20d = q.reshape(5, 4)

                            # safety: clamp + rate limit
                            wuji_20d = np.clip(wuji_20d, float(args.local_wuji_clamp_min), float(args.local_wuji_clamp_max))
                            max_delta = float(args.local_wuji_max_delta_per_step)
                            if s == "left" and last_wuji_left is not None:
                                delta = wuji_20d - last_wuji_left
                                delta = np.clip(delta, -max_delta, max_delta)
                                wuji_20d = last_wuji_left + delta
                            if s == "right" and last_wuji_right is not None:
                                delta = wuji_20d - last_wuji_right
                                delta = np.clip(delta, -max_delta, max_delta)
                                wuji_20d = last_wuji_right + delta
                        else:
                            # DexPilot retargeter
                            if s == "left":
                                if retargeter_left is None:
                                    retargeter_left = WujiHandRetargeter(hand_side="left")
                                rr = retargeter_left.retarget(mp_trans)
                            else:
                                if retargeter_right is None:
                                    retargeter_right = WujiHandRetargeter(hand_side="right")
                                rr = retargeter_right.retarget(mp_trans)
                            wuji_20d = np.asarray(rr.robot_qpos, dtype=np.float32).reshape(5, 4)

                        if s == "left":
                            last_wuji_left = wuji_20d.copy()
                        else:
                            last_wuji_right = wuji_20d.copy()
                        now2 = now_ms()
                        data[key_q] = wuji_20d.reshape(-1).tolist()
                        data[key_t] = int(now2)


                        if bool(int(args.local_wuji_write_redis)):
                            try:
                                client.set(f"action_wuji_qpos_target_{s}_{suffix}", json.dumps(data[key_q]))
                                client.set(f"t_action_wuji_hand_{s}_{suffix}", int(now2))
                            except Exception:
                                pass

                    try:
                        _maybe_retarget_one("left")
                        _maybe_retarget_one("right")
                    except Exception:
                        # local retarget failure should not break recording
                        pass

                recorder.add_item(data)
                step_count += 1

                elapsed = time.time() - t0
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                time.sleep(0.01)

    finally:
        try:
            if recording:
                recorder.save_episode()
        except Exception:
            pass
        try:
            recorder.close()
        except Exception:
            pass
        try:
            if vision is not None:
                vision.close()
        except Exception:
            pass
        try:
            if body_zmq is not None:
                body_zmq.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if xdm_proc is not None:
            try:
                os.killpg(os.getpgid(xdm_proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                xdm_proc.wait(timeout=1.0)
            except Exception:
                pass

    print(f"\nDone! Episodes saved under: {task_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


