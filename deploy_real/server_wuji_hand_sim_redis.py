#!/usr/bin/env python3
"""
Wuji Hand SIM Visualization via Redis

Goals:
- Read hand_tracking_* (26D dict) from Redis.
- Follow the same post-processing path as deploy_real/server_wuji_hand_redis.py:
  26D -> 21D MediaPipe -> apply_mediapipe_transformations -> Retargeter(from YAML)
- Map retarget output to MuJoCo hand model
  (wuji_retargeting/example/utils/mujoco-sim/model/{left,right}.xml)
  and visualize in real time.

Notes:
- Simulation visualization only; no real wujihandpy hardware control.
- Supports follow/hold/default modes via wuji_hand_mode_*.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np  # type: ignore

try:
    import redis  # type: ignore
except Exception as e:
    raise SystemExit(
        "[ERROR] Missing dependency `redis` (python-redis).\n"
        "   Install it in your runtime environment, for example:\n"
        "     - pip install redis\n"
        "     - or conda install -c conda-forge redis-py\n"
        f"   Original error: {e}"
    )


def now_ms() -> int:
    return int(time.time() * 1000)


# 26D hand joint names (aligned with deploy_real/server_wuji_hand_redis.py).
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


# # 26D -> 21D MediaPipe mapping (aligned with deploy_real/server_wuji_hand_redis.py)
# MEDIAPIPE_MAPPING_26_TO_21 = [
#     1,  # 0: Wrist -> Wrist
#     2,  # 1: ThumbMetacarpal -> Thumb CMC
#     3,  # 2: ThumbProximal -> Thumb MCP
#     4,  # 3: ThumbDistal -> Thumb IP
#     5,  # 4: ThumbTip -> Thumb Tip
#     6,  # 5: IndexMetacarpal -> Index MCP
#     7,  # 6: IndexProximal -> Index PIP
#     8,  # 7: IndexIntermediate -> Index DIP
#     10,  # 8: IndexTip -> Index Tip (skip IndexDistal)
#     11,  # 9: MiddleMetacarpal -> Middle MCP
#     12,  # 10: MiddleProximal -> Middle PIP
#     13,  # 11: MiddleIntermediate -> Middle DIP
#     15,  # 12: MiddleTip -> Middle Tip (skip MiddleDistal)
#     16,  # 13: RingMetacarpal -> Ring MCP
#     17,  # 14: RingProximal -> Ring PIP
#     18,  # 15: RingIntermediate -> Ring DIP
#     20,  # 16: RingTip -> Ring Tip (skip RingDistal)
#     21,  # 17: LittleMetacarpal -> Pinky MCP
#     22,  # 18: LittleProximal -> Pinky PIP
#     23,  # 19: LittleIntermediate -> Pinky DIP
#     25,  # 20: LittleTip -> Pinky Tip (skip LittleDistal)
# ]

MEDIAPIPE_MAPPING_26_TO_21 = [
    1,   # 0: Wrist -> Wrist
    2,   # 1: ThumbMetacarpal -> Thumb CMC
    3,   # 2: ThumbProximal -> Thumb MCP
    4,   # 3: ThumbDistal -> Thumb IP
    5,   # 4: ThumbTip -> Thumb Tip
    7,   # 5: IndexMetacarpal -> Index MCP
    8,   # 6: IndexProximal -> Index PIP
    9,   # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (skip IndexDistal)
    12,  # 9: MiddleMetacarpal -> Middle MCP
    13,  # 10: MiddleProximal -> Middle PIP
    14,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (skip MiddleDistal)
    17,  # 13: RingMetacarpal -> Ring MCP
    18,  # 14: RingProximal -> Ring PIP
    19,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (skip RingDistal)
    22,  # 17: LittleMetacarpal -> Pinky MCP
    23,  # 18: LittleProximal -> Pinky PIP
    24,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (skip LittleDistal)
]



def hand_26d_to_mediapipe_21d(hand_data_dict: Dict[str, Any], hand_side: str, print_distances: bool = False) -> np.ndarray:
    """
    Convert a 26D dict (keys: LeftHandWrist/RightHandWrist...) to
    (21,3) MediaPipe keypoints.
    Logic matches deploy_real/server_wuji_hand_redis.py:
    - Use position values
    - Use wrist as origin (subtract wrist from all points)
    """
    side = hand_side.strip().lower()
    assert side in ["left", "right"]
    hand_side_prefix = "LeftHand" if side == "left" else "RightHand"

    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]
            joint_positions_26[i] = np.asarray(pos, dtype=np.float32).reshape(3)
        else:
            joint_positions_26[i] = np.zeros(3, dtype=np.float32)

    mp21 = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21].astype(np.float32)
    wrist_pos = mp21[0].copy()
    mp21 = mp21 - wrist_pos

    # Compute and print wrist-to-fingertip distances on demand.
    if bool(print_distances):
        print("Thumb position:", mp21[4])
        print("Index position:", mp21[8])
        print("Middle position:", mp21[12])
        print("Ring position:", mp21[16])
        print("Pinky position:", mp21[20])

        fingertip_indices = {
            "Thumb": 4,    # ThumbTip
            "Index": 8,    # IndexTip
            "Middle": 12,  # MiddleTip
            "Ring": 16,    # RingTip
            "Pinky": 20,   # PinkyTip
        }
        print("\nWrist-to-fingertip distance (meters):")
        print("-" * 50)
        wrist0 = mp21[0]  # should be [0,0,0]
        for finger_name, tip_idx in fingertip_indices.items():
            tip_pos = mp21[int(tip_idx)]
            distance = float(np.linalg.norm(tip_pos - wrist0))
            print(f"  {finger_name:6s} (index {int(tip_idx):2d}): {distance*100:6.2f} cm ({distance:.4f} m)")
        print("-" * 50)
    return mp21


def _has_nonfinite_tracking_values(hand_data_dict: Dict[str, Any]) -> bool:
    """
    Validate tracking payload values before retargeting.
    Each joint is expected as [pos3, quat4]. We only need pos3 here.
    """
    for _k, v in hand_data_dict.items():
        if not (isinstance(v, (list, tuple)) and len(v) >= 1):
            continue
        pos = v[0]
        try:
            p = np.asarray(pos, dtype=np.float32).reshape(3)
        except Exception:
            continue
        if not np.isfinite(p).all():
            return True
    return False


def _build_wuji_reorder_idx(retargeter: Any) -> Optional[np.ndarray]:
    """
    Retarget qpos order is not always finger{i}_joint{j} natural order.
    Reorder by joint name to:
      finger1_joint1..4, finger2_joint1..4, ... finger5_joint1..4
    """
    try:
        opt = getattr(retargeter, "optimizer", None)
        names = []
        if opt is not None:
            names = list(getattr(opt, "target_joint_names", []))
            if not names:
                robot = getattr(opt, "robot", None)
                names = list(getattr(robot, "dof_joint_names", [])) if robot is not None else []
    except Exception:
        names = []

    if not names:
        return None

    desired_joint_names = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]
    name2idx = {n: i for i, n in enumerate(names)}
    if not all(n in name2idx for n in desired_joint_names):
        return None
    return np.array([name2idx[n] for n in desired_joint_names], dtype=np.int32)


class WujiHandSimRedisViz:
    def __init__(
        self,
        *,
        redis_ip: str,
        hand_side: str,
        target_fps: float,
        freshness_ms: int,
        smooth_enabled: bool,
        smooth_steps: int,
        # mode switch: DexPilot retarget (default) vs GeoRT model inference
        use_model: bool,
        policy_tag: str,
        policy_epoch: int,
        use_fingertips5: bool,
        clamp_min: float,
        clamp_max: float,
        max_delta_per_step: float,
        config_path: Optional[str],
    ) -> None:
        self.hand_side = hand_side.strip().lower()
        assert self.hand_side in ["left", "right"]
        self.target_fps = float(target_fps)
        self.freshness_ms = int(freshness_ms)
        self.smooth_enabled = bool(smooth_enabled)
        self.smooth_steps = max(1, int(smooth_steps))
        self.use_model = bool(use_model)
        self.policy_tag = str(policy_tag)
        self.policy_epoch = int(policy_epoch)
        self.use_fingertips5 = bool(use_fingertips5)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.max_delta_per_step = float(max_delta_per_step)
        self.config_path = config_path

        self.running = True
        self._stop_requested_by_signal: Optional[int] = None
        self._nonfinite_drop_warn_count = 0

        # Redis keys (keep consistent with deploy_real/server_wuji_hand_redis.py)
        self.robot_key = "unitree_g1_with_hands"
        self.redis_key_hand_tracking = f"hand_tracking_{self.hand_side}_{self.robot_key}"
        self.redis_key_action_wuji_qpos_target = f"action_wuji_qpos_target_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_action_wuji_hand = f"t_action_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_wuji_mode = f"wuji_hand_mode_{self.hand_side}_{self.robot_key}"

        print(f"[WujiHandSim] connecting redis {redis_ip}:6379 ...")
        self.redis_client = redis.Redis(host=str(redis_ip), port=6379, decode_responses=False)
        self.redis_client.ping()
        print("[WujiHandSim] redis ok")

        # Import wuji_retargeting (no wujihandpy dependency)
        project_root = Path(__file__).resolve().parents[1]
        wuji_retargeting_v2_path = project_root / "wuji-retargeting"
        wuji_retargeting_legacy_path = project_root / "wuji_retargeting"
        for _p in [wuji_retargeting_v2_path, wuji_retargeting_legacy_path]:
            if _p.exists() and str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
        try:
            from wuji_retargeting import Retargeter  # type: ignore
            from wuji_retargeting.mediapipe import apply_mediapipe_transformations  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Cannot import modern wuji-retargeting (Retargeter).\n"
                "Install it in your runtime environment, for example:\n"
                "  pip install -e ./wuji-retargeting\n"
                f"Original error: {repr(e)}"
            ) from e

        self._apply_mediapipe_transformations = apply_mediapipe_transformations
        self.retargeter = None
        self._wuji_reorder_idx = None
        self.model_infer = None

        if self.use_model:
            # GeoRT model inference (same API as deploy2.py)
            try:
                wuji_retarget_path = project_root / "wuji_policy"
                if str(wuji_retarget_path) not in sys.path:
                    sys.path.insert(0, str(wuji_retarget_path))
                import training  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Cannot import training (wuji_policy/ must be on PYTHONPATH). Error: {e}") from e

            print(f"[WujiHandSim] [INFO] load GeoRT model: tag={self.policy_tag}, epoch={self.policy_epoch}")
            self.model_infer = training.load_model(self.policy_tag, epoch=self.policy_epoch)
            try:
                self.model_infer.eval()
            except Exception:
                pass
            print("[WujiHandSim] [OK] GeoRT model loaded")
        else:
            if not self.config_path:
                raise ValueError("Missing --config, cannot initialize modern Retargeter")
            cfg = Path(self.config_path).expanduser().resolve()
            if not cfg.exists():
                raise FileNotFoundError(f"YAML config file not found: {cfg}")
            print(f"[WujiHandSim] [INFO] init Retargeter({self.hand_side}) with config: {cfg}")
            self.retargeter = Retargeter.from_yaml(str(cfg), hand_side=self.hand_side)
            self._wuji_reorder_idx = _build_wuji_reorder_idx(self.retargeter)
            if self._wuji_reorder_idx is None:
                print("[WujiHandSim] [WARN] Failed to infer reorder indices from joint names; fallback to reshape(5,4)")
        # Load MuJoCo model
        try:
            import mujoco  # type: ignore
            import mujoco.viewer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"mujoco is not installed or GUI dependencies are missing: {e}") from e

        self._mujoco = mujoco
        self._mjviewer = mujoco.viewer

        mjcf_path = str(
            (project_root / "wuji-retargeting" / "example" / "utils" / "mujoco-sim" / "wuji_hand_description" / "mjcf" / f"{self.hand_side}.xml").resolve()
        )
        self.mjcf_path = str(mjcf_path)
        if not Path(self.mjcf_path).exists():
            raise FileNotFoundError(f"MuJoCo model file not found: {self.mjcf_path}")

        self.model = self._mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = self._mujoco.MjData(self.model)

        # init ctrl to mid-range for stability (same as wuji_retargeting sim)
        for i in range(int(self.model.nu)):
            if bool(self.model.actuator_ctrllimited[i]):
                lo, hi = self.model.actuator_ctrlrange[i]
                self.data.ctrl[i] = 0.5 * (float(lo) + float(hi))
            else:
                self.data.ctrl[i] = 0.0

        for _ in range(100):
            self._mujoco.mj_step(self.model, self.data)

        self.zero_pose = np.zeros((5, 4), dtype=np.float32)
        self.last_qpos = self.zero_pose.copy()

    def _get_mode(self) -> str:
        try:
            mode_raw = self.redis_client.get(self.redis_key_wuji_mode)
            if isinstance(mode_raw, bytes):
                mode_raw = mode_raw.decode("utf-8", errors="ignore")
            mode = str(mode_raw) if mode_raw is not None else "follow"
        except Exception:
            mode = "follow"
        return mode.strip().lower()

    def _read_tracking26(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Return (ok, dict):
        - ok=True and dict != None: data is fresh and is_active=True
        - ok=False: missing/stale/inactive data
        """
        try:
            raw = self.redis_client.get(self.redis_key_hand_tracking)
            if raw is None:
                return False, None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return False, None
            ts = int(payload.get("timestamp", 0))
            if (now_ms() - ts) > int(self.freshness_ms):
                return False, None
            if not bool(payload.get("is_active", False)):
                return False, None
            hand_dict = {k: v for k, v in payload.items() if k not in ["is_active", "timestamp"]}
            if _has_nonfinite_tracking_values(hand_dict):
                self._nonfinite_drop_warn_count += 1
                if self._nonfinite_drop_warn_count <= 5 or (self._nonfinite_drop_warn_count % 50 == 0):
                    print(
                        "[WujiHandSim] [WARN] drop non-finite hand_tracking frame "
                        f"(count={self._nonfinite_drop_warn_count})"
                    )
                return False, None
            return True, hand_dict
        except Exception:
            return False, None

    def _retarget_to_wuji20(self, hand_dict: Dict[str, Any]) -> np.ndarray:
        mp21 = hand_26d_to_mediapipe_21d(hand_dict, self.hand_side, print_distances=False)
        if not np.isfinite(mp21).all():
            raise ValueError("Non-finite value in mp21")
        mp21_t = self._apply_mediapipe_transformations(mp21, hand_type=self.hand_side)
        if not np.isfinite(mp21_t).all():
            raise ValueError("Non-finite value in mp21_t")
        if self.use_model:
            if self.model_infer is None:
                raise RuntimeError("model_infer is None but use_model=True")
            pts21 = np.asarray(mp21_t, dtype=np.float32).reshape(21, 3)
            if self.use_fingertips5:
                human_points = pts21[[4, 8, 12, 16, 20], :3]
            else:
                human_points = pts21
            if not np.isfinite(human_points).all():
                raise ValueError("Non-finite value in model_input")
            q = self.model_infer.forward(human_points)
            q = np.asarray(q, dtype=np.float32).reshape(-1)
            if not np.isfinite(q).all():
                raise ValueError("Non-finite value in model_output")
            if q.shape[0] != 20:
                raise ValueError(f"Model output dim mismatch: expect 20, got {q.shape[0]}")
            return q.reshape(5, 4)
        else:
            if self.retargeter is None:
                raise RuntimeError("retargeter is None but use_model=False")
            qpos = np.asarray(self.retargeter.retarget(mp21_t), dtype=np.float32).reshape(-1)
            if self._wuji_reorder_idx is not None and qpos.shape[0] >= int(self._wuji_reorder_idx.max() + 1):
                qpos = qpos[self._wuji_reorder_idx]
            return qpos.reshape(5, 4)

    def _apply_safety(self, qpos_5x4: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos_5x4, dtype=np.float32).reshape(5, 4)
        q = np.clip(q, self.clamp_min, self.clamp_max)
        if self.last_qpos is not None and np.asarray(self.last_qpos).shape == q.shape:
            d = q - self.last_qpos
            d = np.clip(d, -self.max_delta_per_step, self.max_delta_per_step)
            q = self.last_qpos + d
        return q

    def _publish_action_target(self, qpos_5x4: np.ndarray) -> None:
        try:
            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(np.asarray(qpos_5x4, dtype=float).reshape(-1).tolist()))
            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
        except Exception:
            pass

    def _set_ctrl_from_qpos(self, qpos_5x4: np.ndarray) -> None:
        flat = np.asarray(qpos_5x4, dtype=np.float32).reshape(-1)
        n = min(int(self.model.nu), int(flat.shape[0]))
        self.data.ctrl[:n] = flat[:n]

    def _step_sim_towards(self, start_qpos: np.ndarray, target_qpos: np.ndarray, steps_per_ctrl: int) -> None:
        start = np.asarray(start_qpos, dtype=np.float32).reshape(5, 4)
        target = np.asarray(target_qpos, dtype=np.float32).reshape(5, 4)
        total_steps = max(1, int(steps_per_ctrl))

        if (not self.smooth_enabled) or self.smooth_steps <= 1:
            self._set_ctrl_from_qpos(target)
            for _ in range(total_steps):
                self._mujoco.mj_step(self.model, self.data)
            return

        interp_n = max(2, int(self.smooth_steps))
        base = total_steps // interp_n
        rem = total_steps % interp_n
        ts = np.linspace(0.0, 1.0, interp_n, dtype=np.float32)
        for i, t in enumerate(ts):
            q = start * (1.0 - float(t)) + target * float(t)
            self._set_ctrl_from_qpos(q)
            chunk_steps = base + (1 if i < rem else 0)
            for _ in range(chunk_steps):
                self._mujoco.mj_step(self.model, self.data)

    def run(self) -> int:
        def _handle_signal(signum: int, _frame: Any) -> None:
            self._stop_requested_by_signal = int(signum)
            self.running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print("=" * 70)
        print(f"[WujiHandSim] MuJoCo viewer for {self.hand_side} hand")
        print(f"[WujiHandSim] mjcf: {self.mjcf_path}")
        print(f"[WujiHandSim] redis: {self.redis_client.connection_pool.connection_kwargs.get('host', '')}:6379")
        print(f"[WujiHandSim] keys : {self.redis_key_hand_tracking} / {self.redis_key_wuji_mode}")
        print(f"[WujiHandSim] smoothing: {'off' if not self.smooth_enabled else f'on (steps={self.smooth_steps})'}")
        print("=" * 70)

        ctrl_dt = 1.0 / max(1.0, float(self.target_fps))
        sim_dt = float(self.model.opt.timestep)
        steps_per_ctrl = max(1, int(round(ctrl_dt / max(1e-6, sim_dt))))

        # launch viewer (passive)
        try:
            with self._mjviewer.launch_passive(self.model, self.data) as viewer:
                # camera pose like wuji_retargeting sim
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -20
                viewer.cam.distance = 0.5
                viewer.cam.lookat[:] = [0, 0, 0.05]

                last_print = 0.0
                while self.running and viewer.is_running():
                    t0 = time.time()
                    prev_qpos = np.asarray(self.last_qpos, dtype=np.float32).reshape(5, 4).copy()
                    cmd_qpos = prev_qpos.copy()

                    mode = self._get_mode()
                    if mode in ["default", "hold"]:
                        target = self.zero_pose if mode == "default" else self.last_qpos
                        cmd_qpos = np.asarray(target, dtype=np.float32).reshape(5, 4).copy()
                        self._publish_action_target(cmd_qpos)
                    else:
                        # 1) Preferred: retarget from hand_tracking_* (requires pinocchio)
                        ok, hand_dict = self._read_tracking26()
                        if ok and hand_dict is not None:
                            try:
                                q = self._retarget_to_wuji20(hand_dict).astype(np.float32)
                                if self.use_model:
                                    q = self._apply_safety(q)
                                cmd_qpos = np.asarray(q, dtype=np.float32).reshape(5, 4)
                                self._publish_action_target(cmd_qpos)
                            except Exception as e:
                                print(f"[WujiHandSim] [WARN] retarget failed: {e}")

                    self._step_sim_towards(prev_qpos, cmd_qpos, steps_per_ctrl)
                    self.last_qpos = cmd_qpos.copy()

                    viewer.sync()

                    # light status print
                    if (time.time() - last_print) > 2.0:
                        last_print = time.time()
                        print(f"[WujiHandSim] mode={mode} fps={1.0/max(1e-6, (time.time()-t0)):.1f} steps={steps_per_ctrl}")

                    # rate limit
                    elapsed = time.time() - t0
                    sleep_s = max(0.0, ctrl_dt - elapsed)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

        except Exception as e:
            print(f"[WujiHandSim] [ERROR] Failed to start viewer (possibly no GUI or DISPLAY unset): {e}")
            return 2

        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wuji hand MuJoCo sim visualization (Redis -> Retarget -> MuJoCo)")
    p.add_argument("--hand_side", type=str, default="left", choices=["left", "right"], help="left/right")
    p.add_argument("--redis_ip", type=str, default="localhost", help="Redis host")
    p.add_argument("--target_fps", type=float, default=60.0, help="Control/render update rate (Hz)")
    p.add_argument("--freshness_ms", type=int, default=500, help="Freshness threshold for hand_tracking data (ms)")
    p.add_argument("--no_smooth", action="store_true", help="Disable command smoothing")
    p.add_argument("--smooth_steps", type=int, default=5, help="Smoothing interpolation steps")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Retarget YAML config path. If empty, use default by hand side: wuji-retargeting/example/config/retarget_manus_<hand>.yaml",
    )
    # mode switch (align with deploy2.py / wuji_hand_model_deploy.sh)
    p.add_argument("--use_model", action="store_true", help="Use GeoRT model inference (default off; DexPilot retarget otherwise)")
    p.add_argument("--policy_tag", type=str, default="geort_filter_wuji", help="GeoRT model tag (--use_model)")
    p.add_argument("--policy_epoch", type=int, default=-1, help="GeoRT model epoch (--use_model)")
    p.add_argument("--use_fingertips5", action="store_true", help="Use 5 fingertips as model input (default enabled)")
    p.set_defaults(use_fingertips5=True)
    p.add_argument("--clamp_min", type=float, default=-1.5, help="Minimum clamp value for model output")
    p.add_argument("--clamp_max", type=float, default=1.5, help="Maximum clamp value for model output")
    p.add_argument("--max_delta_per_step", type=float, default=0.08, help="Maximum per-step delta for model output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    selected_config = str(args.config).strip()
    if not selected_config:
        selected_config = str(
            (Path(__file__).resolve().parents[1] / "wuji-retargeting" / "example" / "config" / f"retarget_manus_{args.hand_side}.yaml").resolve()
        )
    else:
        config_path = Path(selected_config).expanduser()
        if not config_path.is_absolute():
            candidates = [
                (Path(__file__).resolve().parents[1] / config_path).resolve(),
                (Path(__file__).resolve().parent / config_path).resolve(),
                config_path.resolve(),
            ]
            for cand in candidates:
                if cand.exists():
                    config_path = cand
                    break
            else:
                config_path = candidates[0]
        selected_config = str(config_path)
    viz = WujiHandSimRedisViz(
        redis_ip=str(args.redis_ip),
        hand_side=str(args.hand_side),
        target_fps=float(args.target_fps),
        freshness_ms=int(args.freshness_ms),
        smooth_enabled=not bool(args.no_smooth),
        smooth_steps=int(args.smooth_steps),
        use_model=bool(args.use_model),
        policy_tag=str(args.policy_tag),
        policy_epoch=int(args.policy_epoch),
        use_fingertips5=bool(args.use_fingertips5),
        clamp_min=float(args.clamp_min),
        clamp_max=float(args.clamp_max),
        max_delta_per_step=float(args.max_delta_per_step),
        config_path=selected_config,
    )
    return int(viz.run())


if __name__ == "__main__":
    raise SystemExit(main())


