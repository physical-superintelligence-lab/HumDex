#!/usr/bin/env python3
"""
Wuji hand controller driven by Redis hand-tracking data.

This script reads 26D hand keypoints from Redis (published by teleop),
converts them to a 21D MediaPipe layout, runs retarget/model inference,
and sends joint targets to the Wuji hand controller.
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
from typing import Optional, Dict, Any

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
WUJI_RETARGET_PATH = PROJECT_ROOT / "wuji_policy"
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

# 26D -> 21D mapping to MediaPipe layout.
# MediaPipe: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26D input: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
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


def hand_26d_to_mediapipe_21d(hand_data_dict, hand_side="left"):
    """
    Convert 26D hand dict data into a (21, 3) MediaPipe-style array.

    Args:
        hand_data_dict: 26D dict, e.g. {"LeftHandWrist": [[x,y,z], [qw,qx,qy,qz]], ...}
        hand_side: "left" or "right"

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

    return mediapipe_21d


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
                import training  # type: ignore
            except Exception as e:
                raise ImportError(
                    f"Failed to import training (check wuji_policy and PYTHONPATH): {e}"
                )
            self._training = training
            print(f"[INFO] Loading GeoRT model: tag={self.model_tag}, epoch={self.model_epoch}")
            self.model = training.load_model(self.model_tag, epoch=self.model_epoch)
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
        self._has_received_data = False
        self._nonfinite_drop_warn_count = 0
        
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
        if not np.isfinite(pts21).all():
            raise ValueError("Non-finite value in pts21 before model inference")
        if self.use_fingertips5:
            human_points = pts21[[4, 8, 12, 16, 20], :3]  # (5,3)
        else:
            human_points = pts21
        if not np.isfinite(human_points).all():
            raise ValueError("Non-finite value in model_input")
        qpos_20 = self.model.forward(human_points)
        qpos_20 = np.asarray(qpos_20, dtype=np.float32).reshape(-1)
        if not np.isfinite(qpos_20).all():
            raise ValueError("Non-finite value in model_output")
        if qpos_20.shape[0] != 20:
            raise ValueError(f"Model output dim mismatch: expect 20, got {qpos_20.shape[0]}")
        return qpos_20.reshape(5, 4)

    def _apply_safety(self, qpos_5x4: np.ndarray) -> np.ndarray:
        """Apply clamp and per-step delta limit to model output."""
        q = np.asarray(qpos_5x4, dtype=np.float32).reshape(5, 4)
        if not np.isfinite(q).all():
            q = np.asarray(self.last_qpos if self.last_qpos is not None else self.zero_pose, dtype=np.float32).reshape(5, 4)
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
                if _has_nonfinite_tracking_values(hand_dict):
                    self._nonfinite_drop_warn_count += 1
                    if self._nonfinite_drop_warn_count <= 5 or (self._nonfinite_drop_warn_count % 50 == 0):
                        print(
                            "[WARN] Dropping non-finite hand_tracking frame "
                            f"(count={self._nonfinite_drop_warn_count})."
                        )
                    return None, None
                
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
                        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_data_dict, self.hand_side)
                        if not np.isfinite(mediapipe_21d).all():
                            raise ValueError("Non-finite value in mediapipe_21d")

                        # 2. Apply MediaPipe coordinate transforms.
                        mediapipe_transformed = apply_mediapipe_transformations(
                            mediapipe_21d, 
                            hand_type=self.hand_side
                        )
                        if not np.isfinite(mediapipe_transformed).all():
                            raise ValueError("Non-finite value in mediapipe_transformed")
                        
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
        help="Training model tag for training.load_model(tag, epoch) (--use_model)",
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