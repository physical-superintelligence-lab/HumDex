#!/usr/bin/env python3
"""
Wuji hand controller driven by Redis hand-tracking data.

This script reads 26D hand keypoints from Redis (published by teleop),
converts them to a 21D MediaPipe layout, runs retarget/model inference,
and sends joint targets to the Wuji hand controller.
"""

import argparse
import json
from re import T
import time
import numpy as np
import redis
import signal
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from typing import Any

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


# MediaPipe 21 keypoints hand skeleton connections (indices)
# 0: wrist
# Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
_MP_HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


@dataclass
class _HandVizView:
    """View parameters for projecting 21D 3D points into a 2D canvas."""
    yaw_deg: float = 0.0    # rotation around Z axis
    pitch_deg: float = 0.0  # rotation around X axis
    roll_deg: float = 0.0   # rotation around Y axis
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
    """
    Draw MediaPipe 21D hand points in a simple interactive OpenCV window.

    - Wrist (0) and fingertips (4/8/12/16/20) are highlighted.
    - Keyboard controls support rotate/zoom/reset/quit.

    Returns:
        (ok, view)
        - ok=True: frame drawn successfully
        - ok=False: visualization unavailable or user requested quit
    """
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

    # 3D rotate -> 2D project
    yaw = float(np.deg2rad(view.yaw_deg))
    pitch = float(np.deg2rad(view.pitch_deg))
    roll = float(np.deg2rad(view.roll_deg))
    R = _rot_z(yaw) @ _rot_x(pitch) @ _rot_y(roll)
    pts3 = (R @ pts21.T).T  # (21,3)
    xy = pts3[:, :2].copy()
    if flip_y:
        xy[:, 1] *= -1.0

    pts2 = np.zeros((21, 2), dtype=np.int32)
    s = float(view.scale_px_per_m)
    pts2[:, 0] = (cx + xy[:, 0] * s).astype(np.int32)
    pts2[:, 1] = (cy + xy[:, 1] * s).astype(np.int32)

    # Draw connections
    for a, b in _MP_HAND_CONNECTIONS:
        pa = tuple(int(x) for x in pts2[a])
        pb = tuple(int(x) for x in pts2[b])
        cv2.line(img, pa, pb, (180, 180, 180), 2, lineType=cv2.LINE_AA)

    # Draw points
    tip_idxs = {4, 8, 12, 16, 20}
    for i in range(21):
        p = tuple(int(x) for x in pts2[i])
        is_red = (i == 0) or (i in tip_idxs)
        color = (0, 0, 255) if is_red else (0, 255, 0)  # BGR
        radius = 7 if is_red else 5
        cv2.circle(img, p, radius, color, -1, lineType=cv2.LINE_AA)
        if show_index:
            cv2.putText(img, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    title = "21D MediaPipe Hand"
    if hand_side:
        title += f" ({hand_side})"
    if win_name:
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
            # waitKey(1) for refresh; return False if user presses 'q'/'ESC'
            k = int(cv2.waitKey(1) & 0xFF)
            if k in (ord('q'), 27):
                return False, view
            # Key controls
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
            # likely no DISPLAY / headless
            return False, view
    return True, view


class _Hand21DViz3D:
    """
    Open3D 3D viewer for MediaPipe 21D hand keypoints.
    - Wrist/fingertips are highlighted.
    - Window can be closed by user action.
    """

    def __init__(self, win_name: str = "hand_21d_3d", axis_len_m: float = 0.10):
        self.win_name = str(win_name)
        self.axis_len_m = float(axis_len_m)
        self._vis = None  # type: Optional[Any]
        self._pcd = None  # type: Optional[Any]
        self._lines = None  # type: Optional[Any]
        self._inited = False

    def init(self) -> bool:
        if not _O3D_AVAILABLE or o3d is None:
            if _O3D_IMPORT_ERROR:
                print(f"[WARN] Open3D import failed: {_O3D_IMPORT_ERROR}")
            return False
        try:
            # Detect display availability early (common failure on headless / SSH without X forwarding)
            has_display = bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
            if not has_display:
                print("[WARN] Open3D 3D viewer disabled: DISPLAY/WAYLAND_DISPLAY is not set.")
                print("       If running over SSH, enable X forwarding (`-X`/`-Y`) or use xvfb.")
                return False

            vis = o3d.visualization.Visualizer()
            ok = bool(vis.create_window(window_name=self.win_name, width=900, height=700, visible=True))
            if not ok:
                print("[WARN] Open3D create_window() failed (likely GL/GUI environment issue).")
                print("       Check `echo $DISPLAY` and `glxinfo | head` (mesa-utils).")
                print("       Ensure OpenGL/X11 runtime libs are installed.")
                return False
            self._vis = vis

            self._pcd = o3d.geometry.PointCloud()
            self._lines = o3d.geometry.LineSet()

            # Add geometry once
            self._vis.add_geometry(self._pcd)
            self._vis.add_geometry(self._lines)

            # Optional coordinate frame
            if self.axis_len_m > 0:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_len_m, origin=[0, 0, 0])
                self._vis.add_geometry(axis)

            self._inited = True
            return True
        except Exception:
            print("[WARN] Open3D 3D viewer initialization failed (GL/GUI issue).")
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
            # Points
            self._pcd.points = o3d.utility.Vector3dVector(pts21.astype(np.float64))
            tip_idxs = {0, 4, 8, 12, 16, 20}
            colors = np.zeros((21, 3), dtype=np.float64)
            for i in range(21):
                if i in tip_idxs:
                    colors[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # red
                else:
                    colors[i] = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # green
            self._pcd.colors = o3d.utility.Vector3dVector(colors)

            # Lines
            lines = np.asarray(_MP_HAND_CONNECTIONS, dtype=np.int32)
            self._lines.points = o3d.utility.Vector3dVector(pts21.astype(np.float64))
            self._lines.lines = o3d.utility.Vector2iVector(lines)
            self._lines.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float64), (lines.shape[0], 1))
            )

            self._vis.update_geometry(self._pcd)
            self._vis.update_geometry(self._lines)

            ok = bool(self._vis.poll_events())
            self._vis.update_renderer()
            if not ok:
                # window closed
                self.close()
                return False
            return True
        except Exception:
            # any rendering issue -> disable
            self.close()
            return False

def now_ms() -> int:
    """Return current wall-clock time in milliseconds."""
    return int(time.time() * 1000)


try:
    import wujihandpy
except ImportError:
    print("[ERROR] Missing dependency: wujihandpy.")
    print("        Install with: pip install wujihandpy")
    sys.exit(1)

# Support both `wuji-retargeting` and legacy `wuji_retargeting` folder names.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WUJI_RETARGETING_V2_PATH = PROJECT_ROOT / "wuji-retargeting"
WUJI_RETARGETING_LEGACY_PATH = PROJECT_ROOT / "wuji_retargeting"
for _p in [WUJI_RETARGETING_V2_PATH, WUJI_RETARGETING_LEGACY_PATH]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Keep `wuji_retarget` importable for GeoRT model mode.
WUJI_RETARGET_PATH = PROJECT_ROOT / "wuji_retarget"
if str(WUJI_RETARGET_PATH) not in sys.path:
    sys.path.insert(0, str(WUJI_RETARGET_PATH))

try:
    from wuji_retargeting import Retargeter
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
except ImportError as e:
    print(f"[ERROR] Failed to import wuji_retargeting: {e}")
    print("        Install it first (for example: pip install -e ./wuji-retargeting)")
    sys.exit(1)


# 26D hand joint names (aligned with xrobot utilities).
HAND_JOINT_NAMES_26 = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip", 
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

# 26D -> 21D mapping to MediaPipe layout:
# MediaPipe: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26D input: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
# MEDIAPIPE_MAPPING_26_TO_21 = [
#     1,   # 0: Wrist -> Wrist
#     2,   # 1: ThumbMetacarpal -> Thumb CMC
#     3,   # 2: ThumbProximal -> Thumb MCP
#     4,   # 3: ThumbDistal -> Thumb IP
#     5,   # 4: ThumbTip -> Thumb Tip
#     6,   # 5: IndexMetacarpal -> Index MCP
#     7,   # 6: IndexProximal -> Index PIP
#     8,   # 7: IndexIntermediate -> Index DIP
#     10,  # 8: IndexTip -> Index Tip ( IndexDistal)
#     11,  # 9: MiddleMetacarpal -> Middle MCP
#     12,  # 10: MiddleProximal -> Middle PIP
#     13,  # 11: MiddleIntermediate -> Middle DIP
#     15,  # 12: MiddleTip -> Middle Tip ( MiddleDistal)
#     16,  # 13: RingMetacarpal -> Ring MCP
#     17,  # 14: RingProximal -> Ring PIP
#     18,  # 15: RingIntermediate -> Ring DIP
#     20,  # 16: RingTip -> Ring Tip ( RingDistal)
#     21,  # 17: LittleMetacarpal -> Pinky MCP
#     22,  # 18: LittleProximal -> Pinky PIP
#     23,  # 19: LittleIntermediate -> Pinky DIP
#     25,  # 20: LittleTip -> Pinky Tip ( LittleDistal)
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
    10,  # 8: IndexTip -> Index Tip ( IndexDistal)
    12,  # 9: MiddleMetacarpal -> Middle MCP
    13,  # 10: MiddleProximal -> Middle PIP
    14,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip ( MiddleDistal)
    17,  # 13: RingMetacarpal -> Ring MCP
    18,  # 14: RingProximal -> Ring PIP
    19,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip ( RingDistal)
    22,  # 17: LittleMetacarpal -> Pinky MCP
    23,  # 18: LittleProximal -> Pinky PIP
    24,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip ( LittleDistal)
]

# # 2621 MediaPipe 
# # MediaPipe : [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# # 26: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
# MEDIAPIPE_MAPPING_26_TO_21 = [
#     1,   # 0: Wrist -> Wrist
#     2,   # 1: ThumbMetacarpal -> Thumb CMC
#     3,   # 2: ThumbProximal -> Thumb MCP
#     4,   # 3: ThumbDistal -> Thumb IP
#     5,   # 4: ThumbTip -> Thumb Tip
#     6,   # 5: IndexMetacarpal -> Index MCP
#     7,   # 6: IndexProximal -> Index PIP
#     8,   # 7: IndexIntermediate -> Index DIP
#     10,  # 8: IndexTip -> Index Tip ( IndexDistal)
#     11,  # 9: MiddleMetacarpal -> Middle MCP
#     12,  # 10: MiddleProximal -> Middle PIP
#     13,  # 11: MiddleIntermediate -> Middle DIP
#     15,  # 12: MiddleTip -> Middle Tip ( MiddleDistal)
#     16,  # 13: RingMetacarpal -> Ring MCP
#     17,  # 14: RingProximal -> Ring PIP
#     18,  # 15: RingIntermediate -> Ring DIP
#     20,  # 16: RingTip -> Ring Tip ( RingDistal)
#     21,  # 17: LittleMetacarpal -> Pinky MCP
#     22,  # 18: LittleProximal -> Pinky PIP
#     23,  # 19: LittleIntermediate -> Pinky DIP
#     25,  # 20: LittleTip -> Pinky Tip ( LittleDistal)
# ]


def hand_26d_to_mediapipe_21d(hand_data_dict, hand_side="left", print_distances=False):
    """
    Convert 26D hand dict data into a (21, 3) MediaPipe-style array.

    Args:
        hand_data_dict: 26D dict, e.g. {"LeftHandWrist": [[x,y,z], [qw,qx,qy,qz]], ...}
        hand_side: "left" or "right"
        print_distances: print wrist-to-fingertip distances for debugging

    Returns:
        numpy.ndarray with shape (21, 3)
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    
    # Build 26D position array.
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]  # [x, y, z]
            joint_positions_26[i] = pos
        else:
            # Fallback to zeros when a joint key is missing.
            joint_positions_26[i] = [0.0, 0.0, 0.0]
    
    # Remap to 21D MediaPipe order.
    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]
    
    # Use wrist as origin.
    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos
    
    # Keep a dedicated scale hook for quick tuning if needed.
    scale_factor = 1.0
    mediapipe_21d[1:] = mediapipe_21d[1:] * scale_factor

    # Optional distance debug report.
    if print_distances:
        print("Thumb tip position:", mediapipe_21d[4])
        print("Index tip position:", mediapipe_21d[8])
        print("Middle tip position:", mediapipe_21d[12])
        print("Ring tip position:", mediapipe_21d[16])
        print("Pinky tip position:", mediapipe_21d[20])

        # MediaPipe fingertip indices.
        fingertip_indices = {
            "Thumb": 4,    # ThumbTip
            "Index": 8,    # IndexTip
            "Middle": 12,  # MiddleTip
            "Ring": 16,    # RingTip
            "Pinky": 20,   # PinkyTip
        }
        
        print("\n[INFO] Wrist-to-fingertip distances:")
        print("-" * 50)
        wrist_pos_scaled = mediapipe_21d[0]  # expected [0, 0, 0]
        for finger_name, tip_idx in fingertip_indices.items():
            tip_pos = mediapipe_21d[tip_idx]
            distance = np.linalg.norm(tip_pos - wrist_pos_scaled)
            print(f"  {finger_name:6s} (idx {tip_idx:2d}): {distance*100:6.2f} cm ({distance:.4f} m)")
        print("-" * 50)

        # print(mediapipe_21d)
        # print("-" * 50)
        # print("-" * 50)
        # print(joint_positions_26)
        # print("-" * 50)
    
    return mediapipe_21d


def smooth_move(hand, controller, target_qpos, duration=0.1, steps=10):
    """
    Smoothly interpolate from current qpos to target qpos (5x4).

    Args:
        hand: wujihandpy.Hand instance (kept for API compatibility)
        controller: realtime controller object
        target_qpos: numpy array with shape (5, 4)
        duration: interpolation duration in seconds
        steps: number of interpolation steps
    """
    target_qpos = target_qpos.reshape(5, 4)
    try:
        # cur = controller.get_joint_actual_position()
        cur = controller.read_joint_actual_position()
    except:
        cur = np.zeros((5, 4), dtype=np.float32)
    
    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / steps)


class WujiHandRedisController:
    """Redis-driven Wuji hand controller."""
    
    def __init__(
        self,
        redis_ip="localhost",
        hand_side="left",
        target_fps=50,
        smooth_enabled=True,
        smooth_steps=5,
        serial_number: str = "",
        # mode switch
        use_model: bool = False,
        model_tag: str = "geort_filter_wuji",
        model_epoch: int = -1,
        use_fingertips5: bool = True,
        # safety (model mode)
        clamp_min: float = -1.5,
        clamp_max: float = 1.5,
        max_delta_per_step: float = 0.08,
        viz_hand21d: bool = False,
        viz_hand21d_size: int = 640,
        viz_hand21d_scale: float = 1200.0,
        viz_hand21d_show_index: bool = False,
        viz_hand21d_3d: bool = False,
        viz_hand21d_3d_axis_len_m: float = 0.10,
        debug_pinch: bool = False,
        debug_pinch_every: int = 50,
        pinch_project_ratio: float = 0.2,
        pinch_escape_ratio: float = 0.3,
        pinch_project_dist_max: float = 0.12,
        disable_dexpilot_projection: bool = False,
        config_path: Optional[str] = None,
    ):
        """
        Args:
            redis_ip: Redis host
            hand_side: "left" or "right"
            target_fps: control loop rate in Hz
            smooth_enabled: enable interpolation before sending targets
            smooth_steps: interpolation step count
        """
        self.hand_side = hand_side.lower()
        assert self.hand_side in ["left", "right"], "hand_side must be 'left' or 'right'"
        
        self.target_fps = target_fps
        self.control_dt = 1.0 / target_fps
        self.smooth_enabled = smooth_enabled
        self.smooth_steps = smooth_steps
        self.serial_number = (serial_number or "").strip()

        # Mode: GeoRT model vs DexPilot retarget
        self.use_model = bool(use_model)
        self.model_tag = str(model_tag)
        self.model_epoch = int(model_epoch)
        self.use_fingertips5 = bool(use_fingertips5)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.max_delta_per_step = float(max_delta_per_step)

        # 2D 21D visualization (OpenCV)
        self.viz_hand21d = bool(viz_hand21d)
        self.viz_hand21d_size = int(viz_hand21d_size)
        self.viz_hand21d_scale = float(viz_hand21d_scale)
        self.viz_hand21d_show_index = bool(viz_hand21d_show_index)
        self._viz_ok = None  # None=unknown, True/False known after first draw
        self._viz_view = _HandVizView(scale_px_per_m=self.viz_hand21d_scale)

        # 3D 21D visualization (Open3D)
        self.viz_hand21d_3d = bool(viz_hand21d_3d)
        self.viz_hand21d_3d_axis_len_m = float(viz_hand21d_3d_axis_len_m)
        self._viz3d_ok = None
        self._viz3d = _Hand21DViz3D(win_name=f"hand_21d_3d_{self.hand_side}", axis_len_m=self.viz_hand21d_3d_axis_len_m)

        # Debug pinch (thumb-index distance + current projection threshold)
        self.debug_pinch = bool(debug_pinch)
        self.debug_pinch_every = max(1, int(debug_pinch_every))
        self.pinch_project_ratio = float(pinch_project_ratio)
        self.pinch_escape_ratio = float(pinch_escape_ratio)
        self.pinch_project_dist_max = float(pinch_project_dist_max)
        self.disable_dexpilot_projection = bool(disable_dexpilot_projection)
        self.config_path = config_path
        
        # Redis connection
        print(f"[INFO] Connecting to Redis: {redis_ip}")
        try:
            self.redis_client = redis.Redis(host=redis_ip, port=6379, decode_responses=False)
            self.redis_client.ping()
            print("[OK] Redis connection established.")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Redis: {e}")
            raise
        
        # Redis keys:
        # - hand_tracking_*: 26D source dict from teleop
        # - action_wuji_qpos_target_*: command target sent to Wuji
        # - state_wuji_hand_*: measured state from Wuji
        self.robot_key = "unitree_g1_with_hands"
        self.redis_key_hand_tracking = f"hand_tracking_{self.hand_side}_{self.robot_key}"
        self.redis_key_action_wuji_qpos_target = f"action_wuji_qpos_target_{self.hand_side}_{self.robot_key}"
        self.redis_key_state_wuji_hand = f"state_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_action_wuji_hand = f"t_action_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_state_wuji_hand = f"t_state_wuji_hand_{self.hand_side}_{self.robot_key}"
        # Teleop mode for Wuji hand: follow / hold / default
        self.redis_key_wuji_mode = f"wuji_hand_mode_{self.hand_side}_{self.robot_key}"
        
        # Initialize Wuji hand hardware
        print(f"[INFO] Initializing Wuji hand ({self.hand_side})...")
        if self.serial_number:
            print(f"[INFO] Using serial_number={self.serial_number}")
            self.hand = wujihandpy.Hand(serial_number=self.serial_number)
        else:
            self.hand = wujihandpy.Hand()
        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=10.0)
        )
        time.sleep(0.4)
        
        # Build a zero pose with the same shape as hardware qpos.
        # It is used for default mode and as a fallback baseline.
        # actual_pose = self.hand.get_joint_actual_position()
        actual_pose = self.hand.read_joint_actual_position()
        self.zero_pose = np.zeros_like(actual_pose)
        print(f"[OK] Wuji hand ({self.hand_side}) is ready.")
        
        #  retarget / model
        self.retargeter = None
        self._wuji_reorder_idx = None
        self._retarget_joint_names = []

        if self.use_model:
            # GeoRT model inference
            try:
                import geort  # type: ignore
            except Exception as e:
                raise ImportError(
                    f"Failed to import geort (check wuji_retarget and PYTHONPATH): {e}"
                )
            self._geort = geort
            print(f"[INFO] Loading GeoRT model: tag={self.model_tag}, epoch={self.model_epoch}")
            self.model = geort.load_model(self.model_tag, epoch=self.model_epoch)
            try:
                self.model.eval()
            except Exception:
                pass
            print("[OK] GeoRT model loaded.")
        else:
            # YAML-configured retargeter
            if not self.config_path:
                raise ValueError("Retarget mode requires --config.")
            cfg = Path(self.config_path).expanduser().resolve()
            if not cfg.exists():
                raise FileNotFoundError(f"Retarget YAML not found: {cfg}")
            print(f"[INFO] Loading retargeter ({self.hand_side}) with config: {cfg}")
            self.retargeter = Retargeter.from_yaml(str(cfg), hand_side=self.hand_side)
            # Optional: tune / disable DexPilot-like projection thresholds (if optimizer supports these attrs)
            try:
                opt = getattr(self.retargeter, "optimizer", None)
                if opt is not None:
                    if self.disable_dexpilot_projection:
                        # Make projection never trigger: dist < 0 is impossible; also turn off adaptive thresholding.
                        if hasattr(opt, "project_dist"):
                            opt.project_dist = 0.0
                        if hasattr(opt, "escape_dist"):
                            opt.escape_dist = 0.0
                    else:
                        # NOTE: wuji_retargeting uses fixed project_dist/escape_dist (no hand_scale adaptive logic).
                        # If you need "easier pinch", increase these distances directly.
                        if hasattr(opt, "project_dist"):
                            opt.project_dist = float(self.pinch_project_ratio)  # kept for backward CLI; treat as meters
                        if hasattr(opt, "escape_dist"):
                            opt.escape_dist = float(self.pinch_escape_ratio)    # kept for backward CLI; treat as meters
            except Exception:
                pass
            print("[OK] Retargeter initialized.")

            # Precompute joint reordering for hardware command (wujihandpy expects 5x4 in finger order).
            # IMPORTANT: `qpos` is in optimizer internal order. Prefer name-based reorder if available.
            # Do NOT assume a simple reshape is correct unless the joint order matches exactly.
            try:
                opt = getattr(self.retargeter, "optimizer", None)
                if opt is not None and hasattr(opt, "target_joint_names"):
                    self._retarget_joint_names = list(opt.target_joint_names)
            except Exception:
                self._retarget_joint_names = []

            desired_joint_names = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]
            if self._retarget_joint_names:
                name2idx = {n: i for i, n in enumerate(self._retarget_joint_names)}
                if all(n in name2idx for n in desired_joint_names):
                    self._wuji_reorder_idx = np.array([name2idx[n] for n in desired_joint_names], dtype=int)
                else:
                    missing = [n for n in desired_joint_names if n not in name2idx]
                    print(f"[WARN] Missing {len(missing)} expected joint names; fallback to raw reshape(5,4). Missing sample: {missing[:3]}")
            else:
                print("[WARN] Retarget joint names are unavailable; fallback to raw reshape(5,4).")
        
        # Runtime state
        self.last_qpos = self.zero_pose.copy()
        self.running = True
        self._cleaned_up = False
        self._stop_requested_by_signal = None
        self._frame_count = 0
        self._distance_printed = False
        self._has_received_data = False
        
        # FPS stats for valid data frames
        self._fps_start_time = None
        self._fps_data_frame_count = 0
        self._fps_print_interval = 100

    def _model_infer_wuji_qpos(self, pts21: np.ndarray) -> np.ndarray:
        """
        Run GeoRT inference.
          pts21: (21,3) after apply_mediapipe_transformations (wrist-relative)
          return: (5,4) joint targets
        """
        pts21 = np.asarray(pts21, dtype=np.float32).reshape(21, 3)
        if self.use_fingertips5:
            human_points = pts21[[4, 8, 12, 16, 20], :3]  # (5,3)
        else:
            human_points = pts21
        qpos_20 = self.model.forward(human_points)
        qpos_20 = np.asarray(qpos_20, dtype=np.float32).reshape(-1)
        if qpos_20.shape[0] != 20:
            raise ValueError(f"Model output dim mismatch: expect 20, got {qpos_20.shape[0]}")
        return qpos_20.reshape(5, 4)

    def _apply_safety(self, qpos_5x4: np.ndarray) -> np.ndarray:
        """Apply clamp and per-step delta limit to model output."""
        q = np.asarray(qpos_5x4, dtype=np.float32).reshape(5, 4)
        q = np.clip(q, self.clamp_min, self.clamp_max)
        if self.last_qpos is not None and np.asarray(self.last_qpos).shape == q.shape:
            delta = q - self.last_qpos
            delta = np.clip(delta, -self.max_delta_per_step, self.max_delta_per_step)
            q = self.last_qpos + delta
        return q
        
    def get_hand_tracking_data_from_redis(self):
        """
        Read hand-tracking data (26D dict) from Redis.

        Returns:
            tuple: (is_active, hand_data_dict), or (None, None) on invalid/missing data
        """
        try:
            # Read Redis key.
            data = self.redis_client.get(self.redis_key_hand_tracking)
            
            if data is None:
                # Key not published yet.
                if not hasattr(self, '_debug_key_printed'):
                    print(f"[WARN] Redis key '{self.redis_key_hand_tracking}' not found yet.")
                    self._debug_key_printed = True
                return None, None
            
            # Parse JSON. decode_responses=False means bytes are expected.
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            hand_data = json.loads(data)
            
            # Validate payload.
            if isinstance(hand_data, dict):
                # Freshness guard: teleop publishes at high rate, accept <= 500ms.
                data_timestamp = hand_data.get("timestamp", 0)
                current_time_ms = int(time.time() * 1000)
                time_diff_ms = current_time_ms - data_timestamp
                
                # Data is stale.
                if time_diff_ms > 500:
                    if not hasattr(self, '_debug_stale_printed'):
                        print(f"[WARN] Stale hand_tracking data ({time_diff_ms}ms > 500ms).")
                        self._debug_stale_printed = True
                    return None, None
                
                # Process only active hand source.
                is_active = hand_data.get("is_active", False)
                if not is_active:
                    if not hasattr(self, '_debug_inactive_printed'):
                        print("[WARN] Hand source is inactive (is_active=False).")
                        self._debug_inactive_printed = True
                    return None, None
                
                # Filter out metadata fields.
                hand_dict = {k: v for k, v in hand_data.items() 
                           if k not in ["is_active", "timestamp"]}
                
                # Print one-time success marker.
                if not hasattr(self, '_debug_success_printed'):
                    print(f"[OK] Received hand_tracking data from Redis (key={self.redis_key_hand_tracking}, fields={len(hand_dict)}).")
                    self._debug_success_printed = True
                
                return is_active, hand_dict
            else:
                print(f"[WARN] Invalid payload type: expected dict, got {type(hand_data)}.")
                return None, None
                
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to decode hand_tracking JSON: {e}")
            return None, None
        except Exception as e:
            print(f"[WARN] Redis read error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run(self):
        """Main control loop."""
        print(f"\n[INFO] Starting control loop (target: {self.target_fps} Hz)")
        print("       Press Ctrl+C to stop.\n")
        
        # Reset FPS counters.
        self._fps_start_time = None
        self._fps_data_frame_count = 0
        
        def _handle_signal(signum, _frame):
            # Graceful stop; cleanup in finally to release USB device safely.
            self._stop_requested_by_signal = signum
            self.running = False

        # Handle both SIGINT and SIGTERM for clean shutdown.
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # 1230 debug -- Yihe
        from rich.live import Live
        from rich.text import Text
        live = Live(refresh_per_second=10, auto_refresh=True)
        live.start()

        try:
            while self.running:
                loop_start = time.time()

                # 0) Read mode from Redis (default: follow)
                try:
                    mode_raw = self.redis_client.get(self.redis_key_wuji_mode)
                    if isinstance(mode_raw, bytes):
                        mode_raw = mode_raw.decode("utf-8")
                    mode = str(mode_raw) if mode_raw is not None else "follow"
                except Exception:
                    mode = "follow"
                mode = mode.strip().lower()

                # # 1230 debug -- Yihe
                # # actual_qpos = self.hand.get_joint_actual_position()
                # # print(f"actual_qpos: {actual_qpos}")
                # actual_qpos = self.hand.read_joint_actual_position()
                # # print(f"actual_qpos: {actual_qpos}")
                # with np.printoptions(precision=4, suppress=True):
                #     live.update(Text(f"actual_qpos: {actual_qpos}"))

                # 0.1) default => zero_pose, hold => last_qpos (freeze tracking).
                if mode in ["default", "hold"]:
                    try:
                        target = self.zero_pose if mode == "default" else self.last_qpos
                        if target is None:
                            target = self.zero_pose

                        # Publish action target to Redis (for logging/monitoring).
                        try:
                            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(target.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
                        except Exception:
                            pass

                        # Send command to hand hardware.
                        if self.hand is not None and self.controller is not None:
                            if self.smooth_enabled:
                                smooth_move(self.hand, self.controller, target, duration=self.control_dt, steps=self.smooth_steps)
                            else:
                                self.controller.set_joint_target_position(target)

                            # Publish measured state to Redis.
                            try:
                                # actual_qpos = self.hand.get_joint_actual_position()
                                actual_qpos = self.hand.read_joint_actual_position()
                                self.redis_client.set(self.redis_key_state_wuji_hand, json.dumps(actual_qpos.reshape(-1).tolist()))
                                self.redis_client.set(self.redis_key_t_state_wuji_hand, now_ms())
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"[WARN] Failed in mode '{mode}': {e}")

                    # Rate limit loop.
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.control_dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                
                # 1) Read hand tracking from Redis.
                is_active, hand_data_dict = self.get_hand_tracking_data_from_redis()
                # print(f"is_active: {is_active}, hand_data_dict: {hand_data_dict}")
                
                if is_active and hand_data_dict is not None:
                    try:
                        # Initialize FPS tracking on first valid frame.
                        if self._fps_start_time is None:
                            self._fps_start_time = time.time()
                            self._fps_data_frame_count = 0
                        
                        # 1. Convert 26D dict to 21D MediaPipe keypoints.
                        print_distances = False
                        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_data_dict, self.hand_side, 
                                                                  print_distances=print_distances)
                        # 1.4 Optional 3D viewer (Open3D).
                        if self.viz_hand21d_3d:
                            ok3d = self._viz3d.update(mediapipe_21d)
                            if self._viz3d_ok is None:
                                self._viz3d_ok = bool(ok3d)
                                if not self._viz3d_ok:
                                    print("[WARN] Failed to open 21D 3D viewer (Open3D GUI unavailable).")
                                    self.viz_hand21d_3d = False
                            elif not ok3d:
                                print("[STOP] 21D 3D viewer closed by user.")
                                self.viz_hand21d_3d = False
                        # 1.5 Optional 2D viewer (OpenCV).
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
                            # Detect the first visualization result once, then keep state.
                            if self._viz_ok is None:
                                self._viz_ok = bool(ok)
                                if not self._viz_ok:
                                    print("[WARN] Failed to open 21D 2D viewer (OpenCV GUI unavailable).")
                                    self.viz_hand21d = False
                            elif not ok:
                                # User pressed q/ESC.
                                print("[STOP] 21D 2D viewer closed by user (q/ESC).")
                                self.viz_hand21d = False
                        if print_distances:
                            self._distance_printed = True
                            print("\n[OK] Distance report printed; exiting as requested.")
                            break
                        
                        # 2. Apply MediaPipe coordinate transforms.
                        mediapipe_transformed = apply_mediapipe_transformations(
                            mediapipe_21d, 
                            hand_type=self.hand_side
                        )
                        
                        # 3. Retarget / Model inference
                        if self.use_model:
                            wuji_20d = self._model_infer_wuji_qpos(mediapipe_transformed)
                            wuji_20d = self._apply_safety(wuji_20d)
                        else:
                            # Retarget (YAML-configured optimizer)
                            qpos20 = np.asarray(self.retargeter.retarget(mediapipe_transformed), dtype=np.float32).reshape(-1)
                            if self._wuji_reorder_idx is not None and qpos20.shape[0] >= int(self._wuji_reorder_idx.max() + 1):
                                qpos20_wuji = qpos20[self._wuji_reorder_idx]
                                wuji_20d = qpos20_wuji.reshape(5, 4)
                            else:
                                wuji_20d = qpos20.reshape(5, 4)

                        # Debug pinch( DexPilot )
                        if (not self.use_model) and self.debug_pinch and (self._frame_count % self.debug_pinch_every == 0):
                            # 0) raw 26D from Redis dict (if present)
                            hand_side_prefix = "LeftHand" if self.hand_side == "left" else "RightHand"
                            k_thumb = hand_side_prefix + "ThumbTip"
                            k_index = hand_side_prefix + "IndexTip"
                            raw_thumb = hand_data_dict.get(k_thumb, None)
                            raw_index = hand_data_dict.get(k_index, None)
                            raw_has = (raw_thumb is not None) and (raw_index is not None)
                            if raw_has:
                                try:
                                    raw26_thumb = np.array(raw_thumb[0], dtype=np.float32)
                                    raw26_index = np.array(raw_index[0], dtype=np.float32)
                                    raw26_dist = float(np.linalg.norm(raw26_thumb - raw26_index))
                                except Exception:
                                    raw26_thumb = None
                                    raw26_index = None
                                    raw26_dist = float("nan")
                            else:
                                raw26_thumb = None
                                raw26_index = None
                                raw26_dist = float("nan")

                            # 1) after 26D->21D mapping (before apply_mediapipe_transformations)
                            try:
                                mp21_dist = float(np.linalg.norm(mediapipe_21d[4] - mediapipe_21d[8]))
                            except Exception:
                                mp21_dist = float("nan")

                            # 2) after apply_mediapipe_transformations (actual retarget input)
                            try:
                                thumb_index_dist = float(np.linalg.norm(mediapipe_transformed[4] - mediapipe_transformed[8]))
                            except Exception:
                                thumb_index_dist = float("nan")
                            msg = f"[pinch] frame={self._frame_count}"
                            if raw_has:
                                msg += f"  raw26={raw26_dist:.4f}m({raw26_dist*100:.1f}cm)"
                            else:
                                msg += f"  raw26=NA(missing {k_thumb} or {k_index})"
                            msg += f"  mp21={mp21_dist:.4f}m({mp21_dist*100:.1f}cm)"
                            msg += f"  xform={thumb_index_dist:.4f}m({thumb_index_dist*100:.1f}cm)"

                            # 3) robot FK fingertip distance (did the solver actually pinch on the robot model?)
                            try:
                                opt2 = getattr(self.retargeter, "optimizer", None)
                                if opt2 is not None and hasattr(opt2, "robot"):
                                    robot = opt2.robot
                                    qpos20_fk = qpos20.astype(np.float64).reshape(-1)
                                    robot.compute_forward_kinematics(qpos20_fk)
                                    tip1 = robot.get_link_pose(robot.get_link_index("finger1_tip_link"))[:3, 3]
                                    tip2 = robot.get_link_pose(robot.get_link_index("finger2_tip_link"))[:3, 3]
                                    robot_tip_dist = float(np.linalg.norm(tip1 - tip2))
                                    msg += f"  robot_tip12={robot_tip_dist:.4f}m({robot_tip_dist*100:.1f}cm)"
                            except Exception:
                                pass
                            try:
                                opt = getattr(self.retargeter, "optimizer", None)
                                if opt is not None:
                                    proj = getattr(opt, "last_project_dist", None)
                                    esc = getattr(opt, "last_escape_dist", None)
                                    if proj is not None and esc is not None:
                                        msg += f"  proj={float(proj):.4f}m esc={float(esc):.4f}m"
                                    if (not self.disable_dexpilot_projection) and hasattr(opt, "projected") and isinstance(opt.projected, np.ndarray) and opt.projected.size > 0:
                                        # projected[0] corresponds to thumb-index in this DexPilot setup
                                        msg += f"  projected_thumb_index={bool(opt.projected[0])}"
                            except Exception:
                                pass
                            print(msg, flush=True)

                        # 3.5 Publish Wuji action target to Redis.
                        try:
                            self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(wuji_20d.reshape(-1).tolist()))
                            self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
                        except Exception:
                            # Keep controller running even if Redis write fails.
                            pass
                        
                        # 4. Send command to Wuji hardware.
                        if self.hand is not None and self.controller is not None:
                            if self.smooth_enabled:
                                smooth_move(self.hand, self.controller, wuji_20d, 
                                          duration=self.control_dt, steps=self.smooth_steps)
                            else:
                                self.controller.set_joint_target_position(wuji_20d)

                            # 4.5 Publish measured state to Redis.
                            try:
                                # actual_qpos = self.hand.get_joint_actual_position()
                                actual_qpos = self.hand.read_joint_actual_position()
                                self.redis_client.set(self.redis_key_state_wuji_hand, json.dumps(actual_qpos.reshape(-1).tolist()))
                                self.redis_client.set(self.redis_key_t_state_wuji_hand, now_ms())
                            except Exception:
                                pass
                        
                        self.last_qpos = wuji_20d.copy()
                        self._has_received_data = True
                        self._frame_count += 1
                        
                        # FPS report on valid data frames.
                        self._fps_data_frame_count += 1
                        if self._fps_data_frame_count >= self._fps_print_interval:
                            elapsed_time = time.time() - self._fps_start_time
                            actual_fps = self._fps_data_frame_count / elapsed_time
                            print(f"[INFO] Actual control FPS: {actual_fps:.2f} Hz (target: {self.target_fps} Hz, samples: {self._fps_data_frame_count})")
                            # Reset FPS window.
                            self._fps_start_time = time.time()
                            self._fps_data_frame_count = 0
                        
                    except Exception as e:
                        print(f"[WARN] Loop processing error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Waiting for first valid frame. Hold last_qpos meanwhile.
                    if not self._has_received_data and self._frame_count == 0:
                        print("[WAIT] Waiting for active hand_tracking data...")
                    self._frame_count += 1
                
                # Rate limit loop.
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Safely stop controller and release resources."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        print("\n[STOP] Cleaning up...")
        try:
            # Use a shorter ramp on SIGTERM to reduce shutdown latency.
            if self._stop_requested_by_signal == signal.SIGTERM:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=0.2, steps=10)
            else:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=1.0, steps=50)
            print("[OK] Returned hand to zero pose.")
        except:
            pass
        
        try:
            self.controller.close()
            self.hand.write_joint_enabled(False)
            print("[OK] Controller closed and motors disabled.")
        except:
            pass

        # close viz windows (optional)
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
        
        print("[OK] Cleanup complete.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wuji hand controller (Redis -> 21D -> Retarget/Model -> hardware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Left hand, default settings
  python server_wuji_hand_redis.py --hand_side left --redis_ip localhost

  # Right hand at 50Hz
  python server_wuji_hand_redis.py --hand_side right --target_fps 50

  # Disable smoothing
  python server_wuji_hand_redis.py --hand_side left --no_smooth
        """
    )
    
    parser.add_argument(
        "--hand_side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Hand side to control (default: left)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Retarget YAML path (default: wuji-retargeting/example/config/retarget_manus_<hand>.yaml)",
    )
    
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis host (default: localhost)"
    )
    
    parser.add_argument(
        "--target_fps",
        type=int,
        default=50,
        help="Control loop target FPS (default: 50)"
    )
    
    parser.add_argument(
        "--no_smooth",
        action="store_true",
        help="Disable command smoothing"
    )
    
    parser.add_argument(
        "--smooth_steps",
        type=int,
        default=5,
        help="Smoothing interpolation steps (default: 5)"
    )

    parser.add_argument(
        "--serial_number",
        type=str,
        default="",
        help="Optional Wuji hand serial number (e.g. 337238793233)",
    )

    # =========================
    # Retarget mode switch: DexPilot retarget (default) vs GeoRT model inference
    # Aligned with `wuji_retarget/deploy2.py` and model deploy script.
    # =========================
    parser.add_argument(
        "--use_model",
        action="store_true",
        help="Use GeoRT model inference (20D output) instead of DexPilot retarget",
    )
    parser.add_argument(
        "--policy_tag",
        type=str,
        default="geort_filter_wuji",
        help="GeoRT model tag for geort.load_model(tag, epoch) (--use_model)",
    )
    parser.add_argument(
        "--policy_epoch",
        type=int,
        default=-1,
        help="GeoRT checkpoint epoch (--use_model). Use -1 for latest",
    )
    parser.add_argument(
        "--use_fingertips5",
        action="store_true",
        help="Use 5 fingertips (shape 5x3) as model input (--use_model)",
    )
    parser.set_defaults(use_fingertips5=True)

    # Safety limits for model mode.
    parser.add_argument("--clamp_min", type=float, default=-1.5, help="Minimum clamp value for model output (--use_model)")
    parser.add_argument("--clamp_max", type=float, default=1.5, help="Maximum clamp value for model output (--use_model)")
    parser.add_argument("--max_delta_per_step", type=float, default=0.08, help="Maximum per-step delta for model output (--use_model)")

    # 2D 21D visualization (OpenCV)
    parser.add_argument(
        "--viz_hand21d",
        action="store_true",
        help="Show 2D MediaPipe 21D visualization (OpenCV). Requires opencv-python",
    )
    parser.add_argument(
        "--viz_hand21d_size",
        type=int,
        default=640,
        help="2D visualization canvas size in pixels (default: 640)",
    )
    parser.add_argument(
        "--viz_hand21d_scale",
        type=float,
        default=1200.0,
        help="2D visualization scale in px/m (default: 1200)",
    )
    parser.add_argument(
        "--viz_hand21d_show_index",
        action="store_true",
        help="Show keypoint indices (0-20) in 2D visualization",
    )

    # 3D 21D visualization (Open3D)
    parser.add_argument(
        "--viz_hand21d_3d",
        action="store_true",
        help="Show 3D MediaPipe 21D visualization (Open3D). Requires open3d",
    )
    parser.add_argument(
        "--viz_hand21d_3d_axis_len_m",
        type=float,
        default=0.10,
        help="Coordinate frame axis length for 3D viewer in meters (default: 0.10; set 0 to disable)",
    )

    # Debug pinch: thumb-index distance and projection threshold
    parser.add_argument(
        "--debug_pinch",
        action="store_true",
        help="Print thumb-index pinch diagnostics",
    )
    parser.add_argument(
        "--debug_pinch_every",
        type=int,
        default=50,
        help="Print pinch diagnostics every N frames (default: 50; ~1s at 50Hz)",
    )

    # Pinch tuning (DexPilot projection thresholds)
    parser.add_argument(
        "--pinch_project_ratio",
        type=float,
        default=0.02,
        help="DexPilot pinch project_dist threshold (meters, default: 0.02)",
    )
    parser.add_argument(
        "--pinch_escape_ratio",
        type=float,
        default=0.03,
        help="DexPilot pinch escape_dist threshold (meters, default: 0.03)",
    )
    parser.add_argument(
        "--pinch_project_dist_max",
        type=float,
        default=0.12,
        help="Deprecated legacy arg kept for compatibility (unused by current retarget path)",
    )
    parser.add_argument(
        "--disable_dexpilot_projection",
        action="store_true",
        help="Disable DexPilot pinch projection logic (projected_* flags)",
    )
    
    return parser.parse_args()


def main():
    """Program entry point."""
    args = parse_arguments()
    
    print("=" * 60)
    print("Wuji Hand Controller via Redis (26D -> 21D MediaPipe -> Retarget)")
    print("=" * 60)
    print(f"Hand side: {args.hand_side}")
    print(f"Redis IP: {args.redis_ip}")
    print(f"Target FPS: {args.target_fps} Hz")
    print(f"Smoothing: {'off' if args.no_smooth else 'on'}")
    if not args.no_smooth:
        print(f"Smooth steps: {args.smooth_steps}")
    print(f"Mode: {'model_inference' if args.use_model else 'retarget_yaml'}")
    selected_config = args.config.strip()
    if not selected_config:
        selected_config = str(
            (PROJECT_ROOT / "wuji-retargeting" / "example" / "config" / f"retarget_manus_{args.hand_side}.yaml").resolve()
        )
    else:
        config_path = Path(selected_config).expanduser()
        if not config_path.is_absolute():
            candidates = [
                (PROJECT_ROOT / config_path).resolve(),
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
    if not args.use_model:
        print(f"Retarget config: {selected_config}")
    if args.use_model:
        print(f"Model: tag={args.policy_tag}, epoch={args.policy_epoch}, fingertips5={args.use_fingertips5}")
        print(f"Safety: clamp=[{args.clamp_min},{args.clamp_max}], max_delta={args.max_delta_per_step}")
    print("=" * 60)
    
    try:
        controller = WujiHandRedisController(
            redis_ip=args.redis_ip,
            hand_side=args.hand_side,
            target_fps=args.target_fps,
            smooth_enabled=not args.no_smooth,
            smooth_steps=args.smooth_steps,
            serial_number=args.serial_number,
            use_model=args.use_model,
            model_tag=args.policy_tag,
            model_epoch=args.policy_epoch,
            use_fingertips5=args.use_fingertips5,
            clamp_min=args.clamp_min,
            clamp_max=args.clamp_max,
            max_delta_per_step=args.max_delta_per_step,
            viz_hand21d=args.viz_hand21d,
            viz_hand21d_size=args.viz_hand21d_size,
            viz_hand21d_scale=args.viz_hand21d_scale,
            viz_hand21d_show_index=args.viz_hand21d_show_index,
            viz_hand21d_3d=args.viz_hand21d_3d,
            viz_hand21d_3d_axis_len_m=args.viz_hand21d_3d_axis_len_m,
            debug_pinch=args.debug_pinch,
            debug_pinch_every=args.debug_pinch_every,
            pinch_project_ratio=args.pinch_project_ratio,
            pinch_escape_ratio=args.pinch_escape_ratio,
            pinch_project_dist_max=args.pinch_project_dist_max,
            disable_dexpilot_projection=args.disable_dexpilot_projection,
            config_path=selected_config,
        )
        controller.run()
    except Exception as e:
        print(f"\n[ERROR] Controller failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()