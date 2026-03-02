#!/usr/bin/env python3
"""
Wuji Hand Controller via Redis (Policy Inference)

 Redis  teleop.sh (26),21 MediaPipe ,
, retarget policy model  inference,
 Wuji 5x4=20 ,.

- :Redis ,follow/hold/default ,,(OpenCV/Open3D)
-  WujiHandRetargeter.retarget(...)  policy.forward(...)
-  5 (Thumb/Index/Middle/Ring/Pinky tip) policy ((5,3))
- : + (rate limit),
"""

import argparse
import json
import time
import numpy as np
import redis
import signal
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Any

# ======== YOUR POLICY IMPORTS ========
import geort
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

# Optional OpenCV for visualization
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False

# Optional Open3D for true 3D visualization
try:
    import open3d as o3d  # type: ignore
    _O3D_AVAILABLE = True
    _O3D_IMPORT_ERROR = None
except Exception as e:
    o3d = None
    _O3D_AVAILABLE = False
    _O3D_IMPORT_ERROR = repr(e)

# ======== HARDWARE IMPORT ========
try:
    import wujihandpy
except ImportError:
    print("[ERROR] :  wujihandpy,:")
    print("   pip install wujihandpy")
    sys.exit(1)

#  wuji_retargeting ( mediapipe )
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WUJI_RETARGETING_PATH = PROJECT_ROOT / "wuji_retargeting"
if str(WUJI_RETARGETING_PATH) not in sys.path:
    sys.path.insert(0, str(WUJI_RETARGETING_PATH))

try:
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
except ImportError as e:
    print(f"[ERROR] :  wuji_retargeting.mediapipe: {e}")
    print("    wuji_retargeting ( mediapipe  transformations)")
    sys.exit(1)

# ------------------------------
# MediaPipe 21 keypoints skeleton connections
# 0: wrist
# Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
# ------------------------------
_MP_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ------------------------------
# 26D joint names (upstream format)
# ------------------------------
HAND_JOINT_NAMES_26 = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

# ------------------------------
# 26D -> 21D mapping
# :(0 index=1, Palm)
#  wrist  Wrist(index=0), 0
# .
# ------------------------------
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,   # 0: Wrist -> Wrist  (NOTE: your code uses Palm as wrist)
    2,   # 1: ThumbMetacarpal -> Thumb CMC
    3,   # 2: ThumbProximal -> Thumb MCP
    4,   # 3: ThumbDistal -> Thumb IP
    5,   # 4: ThumbTip -> Thumb Tip
    6,   # 5: IndexMetacarpal -> Index MCP
    7,   # 6: IndexProximal -> Index PIP
    8,   # 7: IndexIntermediate -> Index DIP
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


# ------------------------------
# Simple 21D visualization (OpenCV)
# ------------------------------
@dataclass
class _HandVizView:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    scale_px_per_m: float = 1200.0


def _rot_x(a: float) -> np.ndarray:
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float32)


def _rot_y(a: float) -> np.ndarray:
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)


def _rot_z(a: float) -> np.ndarray:
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)


def _viz_draw_mediapipe_hand_21d(
    pts21: np.ndarray,
    win_name: str = "hand_21d",
    canvas_size: int = 640,
    scale_px_per_m: float = 1200.0,
    flip_y: bool = True,
    show_index: bool = False,
    hand_side: str = "",
    view: Optional[_HandVizView] = None,
) -> Tuple[bool, _HandVizView]:
    if view is None:
        view = _HandVizView(scale_px_per_m=float(scale_px_per_m))
    if not _CV2_AVAILABLE or cv2 is None:
        return False, view
    if pts21 is None:
        return False, view

    pts21 = np.asarray(pts21, dtype=np.float32).reshape(21, 3)
    H = int(canvas_size)
    W = int(canvas_size)
    img = np.zeros((H, W, 3), dtype=np.uint8)

    cx = W // 2
    cy = H // 2

    yaw = float(np.deg2rad(view.yaw_deg))
    pitch = float(np.deg2rad(view.pitch_deg))
    roll = float(np.deg2rad(view.roll_deg))
    R = _rot_z(yaw) @ _rot_x(pitch) @ _rot_y(roll)

    pts3 = (R @ pts21.T).T
    xy = pts3[:, :2].copy()
    if flip_y:
        xy[:, 1] *= -1.0

    pts2 = np.zeros((21, 2), dtype=np.int32)
    s = float(view.scale_px_per_m)
    pts2[:, 0] = (cx + xy[:, 0] * s).astype(np.int32)
    pts2[:, 1] = (cy + xy[:, 1] * s).astype(np.int32)

    for a, b in _MP_HAND_CONNECTIONS:
        pa = tuple(int(x) for x in pts2[a])
        pb = tuple(int(x) for x in pts2[b])
        cv2.line(img, pa, pb, (180, 180, 180), 2, lineType=cv2.LINE_AA)

    tip_idxs = {4, 8, 12, 16, 20}
    for i in range(21):
        p = tuple(int(x) for x in pts2[i])
        is_red = (i == 0) or (i in tip_idxs)
        color = (0, 0, 255) if is_red else (0, 255, 0)
        radius = 7 if is_red else 5
        cv2.circle(img, p, radius, color, -1, lineType=cv2.LINE_AA)
        if show_index:
            cv2.putText(img, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    title = "21D MediaPipe Hand"
    if hand_side:
        title += f" ({hand_side})"

    try:
        cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            img,
            "rotate: J/L yaw  I/K pitch  U/O roll   zoom: +/-   reset: R   quit: Q/ESC",
            (10, H - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        cv2.imshow(win_name, img)
        k = int(cv2.waitKey(1) & 0xFF)
        if k in (ord('q'), 27):
            return False, view

        step_deg = 5.0
        if k == ord('j'):
            view.yaw_deg -= step_deg
        elif k == ord('l'):
            view.yaw_deg += step_deg
        elif k == ord('i'):
            view.pitch_deg -= step_deg
        elif k == ord('k'):
            view.pitch_deg += step_deg
        elif k == ord('u'):
            view.roll_deg -= step_deg
        elif k == ord('o'):
            view.roll_deg += step_deg
        elif k in (ord('+'), ord('=')):
            view.scale_px_per_m *= 1.1
        elif k in (ord('-'), ord('_')):
            view.scale_px_per_m /= 1.1
        elif k == ord('r'):
            view.yaw_deg = 0.0
            view.pitch_deg = 0.0
            view.roll_deg = 0.0
            view.scale_px_per_m = float(scale_px_per_m)
    except Exception:
        return False, view

    return True, view


class _Hand21DViz3D:
    def __init__(self, win_name: str = "hand_21d_3d", axis_len_m: float = 0.10):
        self.win_name = str(win_name)
        self.axis_len_m = float(axis_len_m)
        self._vis: Optional[Any] = None
        self._pcd: Optional[Any] = None
        self._lines: Optional[Any] = None
        self._inited = False

    def init(self) -> bool:
        if not _O3D_AVAILABLE or o3d is None:
            if _O3D_IMPORT_ERROR:
                print(f"[WARN]  Open3D : {_O3D_IMPORT_ERROR}")
            return False
        try:
            has_display = bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
            if not has_display:
                print("[WARN]  Open3D 3D : DISPLAY/WAYLAND_DISPLAY, SSH  X .")
                return False

            vis = o3d.visualization.Visualizer()
            ok = bool(vis.create_window(window_name=self.win_name, width=900, height=700, visible=True))
            if not ok:
                print("[WARN]  Open3D 3D :create_window() (/GL ).")
                return False

            self._vis = vis
            self._pcd = o3d.geometry.PointCloud()
            self._lines = o3d.geometry.LineSet()

            self._vis.add_geometry(self._pcd)
            self._vis.add_geometry(self._lines)

            if self.axis_len_m > 0:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_len_m, origin=[0, 0, 0])
                self._vis.add_geometry(axis)

            self._inited = True
            return True
        except Exception:
            print("[WARN]  Open3D 3D :( GL/GUI ).")
            return False

    def close(self):
        try:
            if self._vis is not None:
                self._vis.destroy_window()
        except Exception:
            pass
        self._vis = None
        self._pcd = None
        self._lines = None
        self._inited = False

    def update(self, pts21: np.ndarray) -> bool:
        if not self._inited:
            if not self.init():
                return False
        if self._vis is None or self._pcd is None or self._lines is None or o3d is None:
            return False

        pts21 = np.asarray(pts21, dtype=np.float32).reshape(21, 3)
        try:
            self._pcd.points = o3d.utility.Vector3dVector(pts21.astype(np.float64))
            tip_idxs = {0, 4, 8, 12, 16, 20}
            colors = np.zeros((21, 3), dtype=np.float64)
            for i in range(21):
                colors[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64) if i in tip_idxs else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            self._pcd.colors = o3d.utility.Vector3dVector(colors)

            lines = np.asarray(_MP_HAND_CONNECTIONS, dtype=np.int32)
            self._lines.points = o3d.utility.Vector3dVector(pts21.astype(np.float64))
            self._lines.lines = o3d.utility.Vector2iVector(lines)
            self._lines.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float64), (lines.shape[0], 1)))

            self._vis.update_geometry(self._pcd)
            self._vis.update_geometry(self._lines)

            ok = bool(self._vis.poll_events())
            self._vis.update_renderer()
            if not ok:
                self.close()
                return False
            return True
        except Exception:
            self.close()
            return False


def now_ms() -> int:
    return int(time.time() * 1000)


def hand_26d_to_mediapipe_21d(hand_data_dict, hand_side="left", print_distances=False) -> np.ndarray:
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)

    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]  # [x,y,z]
            joint_positions_26[i] = pos
        else:
            joint_positions_26[i] = [0.0, 0.0, 0.0]

    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]

    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos

    scale_factor = 1.0
    mediapipe_21d[1:] = mediapipe_21d[1:] * scale_factor

    if print_distances:
        fingertip_indices = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
        print("\n[INFO]  (: ):")
        for finger_name, tip_idx in fingertip_indices.items():
            tip_pos = mediapipe_21d[tip_idx]
            distance = np.linalg.norm(tip_pos - mediapipe_21d[0])
            print(f"  {finger_name:6s}: {distance*100:6.2f} cm ({distance:.4f} m)")

    return mediapipe_21d


def smooth_move(hand, controller, target_qpos, duration=0.1, steps=10):
    target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(5, 4)
    try:
        cur = controller.read_joint_actual_position()
    except Exception:
        cur = np.zeros((5, 4), dtype=np.float32)

    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / steps)


class WujiHandRedisController:
    def __init__(
        self,
        redis_ip="localhost",
        hand_side="left",
        target_fps=50,
        smooth_enabled=True,
        smooth_steps=5,
        serial_number: str = "",
        viz_hand21d: bool = False,
        viz_hand21d_size: int = 640,
        viz_hand21d_scale: float = 1200.0,
        viz_hand21d_show_index: bool = False,
        viz_hand21d_3d: bool = False,
        viz_hand21d_3d_axis_len_m: float = 0.10,
        # policy
        policy_tag: str = "geort_wuji_2",
        policy_epoch: int = -1,
        # safety
        clamp_min: float = -1.5,
        clamp_max: float = 1.5,
        max_delta_per_step: float = 0.08,
        use_fingertips5: bool = True,
    ):
        self.hand_side = hand_side.lower()
        assert self.hand_side in ["left", "right"], "hand_side must be 'left' or 'right'"

        self.target_fps = int(target_fps)
        self.control_dt = 1.0 / float(self.target_fps)
        self.smooth_enabled = bool(smooth_enabled)
        self.smooth_steps = int(smooth_steps)
        self.serial_number = (serial_number or "").strip()

        self.viz_hand21d = bool(viz_hand21d)
        self.viz_hand21d_size = int(viz_hand21d_size)
        self.viz_hand21d_scale = float(viz_hand21d_scale)
        self.viz_hand21d_show_index = bool(viz_hand21d_show_index)
        self._viz_ok = None
        self._viz_view = _HandVizView(scale_px_per_m=self.viz_hand21d_scale)

        self.viz_hand21d_3d = bool(viz_hand21d_3d)
        self.viz_hand21d_3d_axis_len_m = float(viz_hand21d_3d_axis_len_m)
        self._viz3d_ok = None
        self._viz3d = _Hand21DViz3D(win_name=f"hand_21d_3d_{self.hand_side}", axis_len_m=self.viz_hand21d_3d_axis_len_m)

        # policy config
        self.policy_tag = str(policy_tag)
        self.policy_epoch = int(policy_epoch)
        self.use_fingertips5 = bool(use_fingertips5)

        # safety config
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.max_delta_per_step = float(max_delta_per_step)

        # connect redis
        print(f"[INFO]  Redis: {redis_ip}")
        self.redis_client = redis.Redis(host=redis_ip, port=6379, decode_responses=False)
        self.redis_client.ping()
        print("[OK] Redis ")

        self.robot_key = "unitree_g1_with_hands"
        self.redis_key_hand_tracking = f"hand_tracking_{self.hand_side}_{self.robot_key}"
        self.redis_key_action_wuji_qpos_target = f"action_wuji_qpos_target_{self.hand_side}_{self.robot_key}"
        self.redis_key_state_wuji_hand = f"state_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_action_wuji_hand = f"t_action_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_state_wuji_hand = f"t_state_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_wuji_mode = f"wuji_hand_mode_{self.hand_side}_{self.robot_key}"

        # init hardware
        print(f"[INFO]  Wuji {self.hand_side} ...")
        if self.serial_number:
            print(f"[INFO]  serial_number : {self.serial_number}")
            self.hand = wujihandpy.Hand(serial_number=self.serial_number)
        else:
            self.hand = wujihandpy.Hand()

        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)
        )
        time.sleep(0.4)

        actual_pose = self.hand.read_joint_actual_position()
        self.zero_pose = np.zeros_like(actual_pose, dtype=np.float32)
        print(f"[OK] Wuji {self.hand_side} ")

        # init policy
        print("[INFO]  Retarget Policy Model...")
        self.policy = geort.load_model(self.policy_tag, epoch=self.policy_epoch)
        try:
            self.policy.eval()
        except Exception:
            pass
        print(f"[OK] Policy loaded: tag={self.policy_tag}, epoch={self.policy_epoch}")

        # state
        self.last_qpos = self.zero_pose.copy()
        self.running = True
        self._cleaned_up = False
        self._stop_requested_by_signal = None
        self._frame_count = 0
        self._has_received_data = False

        # fps monitor
        self._fps_start_time = None
        self._fps_data_frame_count = 0
        self._fps_print_interval = 100

    def _policy_infer_wuji_qpos(self, pts21: np.ndarray) -> np.ndarray:
        """
        pts21: (21,3) after transformations, already relative to wrist
        return: (5,4)
        """
        pts21 = np.asarray(pts21, dtype=np.float32).reshape(21, 3)

        if self.use_fingertips5:
            human_points = pts21[[4, 8, 12, 16, 20], :3]  # (5,3)
        else:
            # fallback: use all 21 points; if your policy supports it, modify accordingly
            human_points = pts21

        if _TORCH_AVAILABLE and torch is not None:
            with torch.no_grad():
                qpos_20 = self.policy.forward(human_points)
        else:
            qpos_20 = self.policy.forward(human_points)

        qpos_20 = np.asarray(qpos_20, dtype=np.float32).reshape(-1)
        if qpos_20.shape[0] != 20:
            raise ValueError(f"Policy output dim mismatch: expect 20, got {qpos_20.shape[0]}")
        return qpos_20.reshape(5, 4)

    def _apply_safety(self, qpos_5x4: np.ndarray) -> np.ndarray:
        """Clamp + rate limit"""
        q = np.asarray(qpos_5x4, dtype=np.float32).reshape(5, 4)

        # clamp
        q = np.clip(q, self.clamp_min, self.clamp_max)

        # rate limit
        if self.last_qpos is not None and self.last_qpos.shape == q.shape:
            delta = q - self.last_qpos
            delta = np.clip(delta, -self.max_delta_per_step, self.max_delta_per_step)
            q = self.last_qpos + delta

        return q

    def get_hand_tracking_data_from_redis(self):
        try:
            data = self.redis_client.get(self.redis_key_hand_tracking)
            if data is None:
                if not hasattr(self, "_debug_key_printed"):
                    print(f"[WARN]  Redis key '{self.redis_key_hand_tracking}' ")
                    self._debug_key_printed = True
                return None, None

            if isinstance(data, bytes):
                data = data.decode("utf-8")

            hand_data = json.loads(data)
            if not isinstance(hand_data, dict):
                print(f"[WARN]  :  dict, {type(hand_data)}")
                return None, None

            data_timestamp = hand_data.get("timestamp", 0)
            current_time_ms = int(time.time() * 1000)
            time_diff_ms = current_time_ms - data_timestamp
            if time_diff_ms > 500:
                if not hasattr(self, "_debug_stale_printed"):
                    print(f"[WARN]   (: {time_diff_ms}ms > 500ms)")
                    self._debug_stale_printed = True
                return None, None

            is_active = hand_data.get("is_active", False)
            if not is_active:
                if not hasattr(self, "_debug_inactive_printed"):
                    print("[WARN]   is_active=False")
                    self._debug_inactive_printed = True
                return None, None

            hand_dict = {k: v for k, v in hand_data.items() if k not in ["is_active", "timestamp"]}
            if not hasattr(self, "_debug_success_printed"):
                print(f"[OK]  Redis  (key: {self.redis_key_hand_tracking}, : {len(hand_dict)})")
                self._debug_success_printed = True

            return is_active, hand_dict

        except json.JSONDecodeError as e:
            print(f"[WARN]  JSON : {e}")
            return None, None
        except Exception as e:
            print(f"[WARN]   Redis : {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def run(self):
        print(f"\n[INFO]  (: {self.target_fps} Hz)")
        print(" Ctrl+C \n")

        self._fps_start_time = None
        self._fps_data_frame_count = 0

        def _handle_signal(signum, _frame):
            self._stop_requested_by_signal = signum
            self.running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while self.running:
                loop_start = time.time()

                # 0) read mode
                try:
                    mode_raw = self.redis_client.get(self.redis_key_wuji_mode)
                    if isinstance(mode_raw, bytes):
                        mode_raw = mode_raw.decode("utf-8")
                    mode = str(mode_raw) if mode_raw is not None else "follow"
                except Exception:
                    mode = "follow"
                mode = mode.strip().lower()

                # default / hold: no tracking needed
                if mode in ["default", "hold"]:
                    try:
                        target = self.zero_pose if mode == "default" else self.last_qpos
                        if target is None:
                            target = self.zero_pose

                        # write action target
                        try:
                            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(target.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
                        except Exception:
                            pass

                        # control
                        if self.smooth_enabled:
                            smooth_move(self.hand, self.controller, target, duration=self.control_dt, steps=self.smooth_steps)
                        else:
                            self.controller.set_joint_target_position(target)

                        # write state
                        try:
                            actual_qpos = self.hand.read_joint_actual_position()
                            self.redis_client.set(self.redis_key_state_wuji_hand, json.dumps(actual_qpos.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_state_wuji_hand, now_ms())
                        except Exception:
                            pass

                    except Exception as e:
                        print(f"[WARN]   {mode} : {e}")

                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.control_dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # 1) read tracking
                is_active, hand_data_dict = self.get_hand_tracking_data_from_redis()

                if is_active and hand_data_dict is not None:
                    try:
                        if self._fps_start_time is None:
                            self._fps_start_time = time.time()
                            self._fps_data_frame_count = 0

                        # 2) 26D -> 21D
                        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_data_dict, self.hand_side, print_distances=False)

                        # 2.1) viz
                        if self.viz_hand21d_3d:
                            ok3d = self._viz3d.update(mediapipe_21d)
                            if self._viz3d_ok is None:
                                self._viz3d_ok = bool(ok3d)
                                if not self._viz3d_ok:
                                    print("[WARN]  21D 3D ,.")
                                    self.viz_hand21d_3d = False
                            elif not ok3d:
                                print("[STOP]  21D 3D ()..")
                                self.viz_hand21d_3d = False

                        if self.viz_hand21d:
                            ok, self._viz_view = _viz_draw_mediapipe_hand_21d(
                                mediapipe_21d,
                                win_name=f"hand_21d_{self.hand_side}",
                                canvas_size=self.viz_hand21d_size,
                                scale_px_per_m=self.viz_hand21d_scale,
                                flip_y=True,
                                show_index=self.viz_hand21d_show_index,
                                hand_side=self.hand_side,
                                view=self._viz_view,
                            )
                            if self._viz_ok is None:
                                self._viz_ok = bool(ok)
                                if not self._viz_ok:
                                    print("[WARN]  21D ,.")
                                    self.viz_hand21d = False
                            elif not ok:
                                print("[STOP]  21D (q/ESC )..")
                                self.viz_hand21d = False

                        # 3) apply mediapipe transformations (keep same as your pipeline)
                        mediapipe_transformed = apply_mediapipe_transformations(
                            mediapipe_21d,
                            hand_type=self.hand_side
                        )

                        # 4) POLICY inference -> wuji_20d (5x4)
                        wuji_20d = self._policy_infer_wuji_qpos(mediapipe_transformed)
                        wuji_20d = self._apply_safety(wuji_20d)

                        # 4.1) write action target to redis
                        try:
                            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(wuji_20d.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
                        except Exception:
                            pass

                        # 5) control hardware
                        if self.smooth_enabled:
                            smooth_move(self.hand, self.controller, wuji_20d, duration=self.control_dt, steps=self.smooth_steps)
                        else:
                            self.controller.set_joint_target_position(wuji_20d)

                        # 5.1) write state feedback
                        try:
                            actual_qpos = self.hand.read_joint_actual_position()
                            self.redis_client.set(self.redis_key_state_wuji_hand, json.dumps(actual_qpos.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_state_wuji_hand, now_ms())
                        except Exception:
                            pass

                        self.last_qpos = wuji_20d.copy()
                        self._has_received_data = True
                        self._frame_count += 1

                        # fps stats
                        self._fps_data_frame_count += 1
                        if self._fps_data_frame_count >= self._fps_print_interval:
                            elapsed_time = time.time() - self._fps_start_time
                            actual_fps = self._fps_data_frame_count / max(elapsed_time, 1e-6)
                            print(f"[INFO] : {actual_fps:.2f} Hz (: {self.target_fps} Hz,  {self._fps_data_frame_count} )")
                            self._fps_start_time = time.time()
                            self._fps_data_frame_count = 0

                    except Exception as e:
                        print(f"[WARN]  : {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    if not self._has_received_data and self._frame_count == 0:
                        print("[WAIT] ...")
                    self._frame_count += 1

                elapsed = time.time() - loop_start
                sleep_time = max(0, self.control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self.cleanup()

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True

        print("\n[STOP] ...")
        try:
            if self._stop_requested_by_signal == signal.SIGTERM:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=0.2, steps=10)
            else:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=1.0, steps=50)
            print("[OK] ")
        except Exception:
            pass

        try:
            self.controller.close()
            self.hand.write_joint_enabled(False)
            print("[OK] ")
        except Exception:
            pass

        try:
            if self._viz3d is not None:
                self._viz3d.close()
        except Exception:
            pass
        try:
            if _CV2_AVAILABLE and cv2 is not None:
                cv2.destroyAllWindows()
        except Exception:
            pass

        print("[OK] ")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Redis teleop -> 26D -> 21D -> policy inference -> Wuji hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--hand_side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--target_fps", type=int, default=50)

    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--smooth_steps", type=int, default=5)

    parser.add_argument("--serial_number", type=str, default="")

    # viz
    parser.add_argument("--viz_hand21d", action="store_true")
    parser.add_argument("--viz_hand21d_size", type=int, default=640)
    parser.add_argument("--viz_hand21d_scale", type=float, default=1200.0)
    parser.add_argument("--viz_hand21d_show_index", action="store_true")
    parser.add_argument("--viz_hand21d_3d", action="store_true")
    parser.add_argument("--viz_hand21d_3d_axis_len_m", type=float, default=0.10)

    # policy
    parser.add_argument("--policy_tag", type=str, default="geort_filter_wuji")
    parser.add_argument("--policy_epoch", type=int, default=-1)
    parser.add_argument("--use_fingertips5", action="store_true", help="Use 5 fingertips as input (recommended). Default True.")
    parser.set_defaults(use_fingertips5=True)

    # safety
    parser.add_argument("--clamp_min", type=float, default=-1.5)
    parser.add_argument("--clamp_max", type=float, default=1.5)
    parser.add_argument("--max_delta_per_step", type=float, default=0.08)

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("Wuji Hand Controller via Redis (Policy Inference)")
    print("=" * 60)
    print(f": {args.hand_side}")
    print(f"Redis IP: {args.redis_ip}")
    print(f": {args.target_fps} Hz")
    print(f": {'' if args.no_smooth else ''}")
    if not args.no_smooth:
        print(f": {args.smooth_steps}")
    print(f"Policy: tag={args.policy_tag}, epoch={args.policy_epoch}")
    print(f"Safety: clamp=[{args.clamp_min},{args.clamp_max}], max_delta={args.max_delta_per_step}")
    print("=" * 60)

    controller = WujiHandRedisController(
        redis_ip=args.redis_ip,
        hand_side=args.hand_side,
        target_fps=args.target_fps,
        smooth_enabled=not args.no_smooth,
        smooth_steps=args.smooth_steps,
        serial_number=args.serial_number,
        viz_hand21d=args.viz_hand21d,
        viz_hand21d_size=args.viz_hand21d_size,
        viz_hand21d_scale=args.viz_hand21d_scale,
        viz_hand21d_show_index=args.viz_hand21d_show_index,
        viz_hand21d_3d=args.viz_hand21d_3d,
        viz_hand21d_3d_axis_len_m=args.viz_hand21d_3d_axis_len_m,
        policy_tag=args.policy_tag,
        policy_epoch=args.policy_epoch,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        max_delta_per_step=args.max_delta_per_step,
        use_fingertips5=args.use_fingertips5,
    )
    controller.run()


if __name__ == "__main__":
    main()
