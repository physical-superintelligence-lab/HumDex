#!/usr/bin/env python3
"""
Policy Inference for HumDex.

Modes:
    1. replay  - Replay GT actions from HDF5 dataset (for testing pipeline)
    2. eval    - Run policy with offline observations from dataset, visualize in sim
    3. infer   - Run policy with real-time observations from Redis/ZMQ

Usage:
    # Replay GT actions from dataset
    python policy_inference.py replay --dataset ../act/data/dataset.hdf5

    # Evaluate policy with dataset observations (sim visualization)
    python policy_inference.py eval --ckpt_dir ../act/ckpts/my_run --dataset ../act/data/dataset.hdf5

    # Run policy with real-time observations
    python policy_inference.py infer --ckpt_dir ../act/ckpts/my_run --temporal_agg
"""

import argparse
import json
import os
import pickle
import sys
import time
import threading
import select
import termios
import tty
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import h5py
import numpy as np
# import redis
import matplotlib.pyplot as plt

# Add act/ to path for policy imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act'))

# Import relative action conversion utilities
from utils import convert_actions_to_absolute


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Inference configuration."""
    # Redis
    redis_ip: str = "localhost"
    redis_port: int = 6379
    robot_key: str = "unitree_g1_with_hands"

    # Control
    frequency: float = 30.0  # Hz

    # Policy
    chunk_size: int = 50
    temporal_agg: bool = True
    use_gpu: bool = True
    
    # Hand configuration
    hand_side: str = "left"  # "left", "right", or "both"

    # State body dimension configuration
    # If True: use 31D state_body (roll/pitch + joints) instead of 34D (ang_vel + roll/pitch + joints)
    # For real robot: extract state_body[3:34] from Redis
    # For dataset: auto-detect from data, or extract [3:34] if dataset has 34D
    state_body_31d: bool = False
    
    # Dimensions (computed based on state_body_31d and hand_side)
    @property
    def state_body_dim(self) -> int:
        return 31 if self.state_body_31d else 34
    
    @property
    def hand_dim(self) -> int:
        """Hand dimension: 20 for single hand, 40 for both hands"""
        return 20 if self.hand_side in ["left", "right"] else 40
    
    @property
    def state_dim(self) -> int:
        """Total state dimension: state_body + hand(s)"""
        return self.state_body_dim + self.hand_dim
    
    @property
    def action_dim(self) -> int:
        """Total action dimension: action_body(35) + hand action(s)"""
        return 35 + self.hand_dim  # 55 (single hand) or 75 (both hands)

    # Redis keys
    @property
    def key_state_body(self) -> str:
        return f"state_body_{self.robot_key}"

    @property
    def key_state_hand_left(self) -> str:
        return f"state_wuji_hand_left_{self.robot_key}"
    
    @property
    def key_state_hand_right(self) -> str:
        return f"state_wuji_hand_right_{self.robot_key}"

    @property
    def key_action_body(self) -> str:
        return f"action_body_{self.robot_key}"

    @property
    def key_action_neck(self) -> str:
        return f"action_neck_{self.robot_key}"

    @property
    def key_action_hand_left(self) -> str:
        return f"action_hand_left_{self.robot_key}"

    @property
    def key_action_hand_right(self) -> str:
        return f"action_hand_right_{self.robot_key}"

    @property
    def key_t_action(self) -> str:
        return "t_action"


# =============================================================================
# Redis I/O
# =============================================================================

class RedisIO:
    """Read state from and publish actions to Redis."""

    def __init__(self, config: Config):
        self.config = config
        self.client = redis.Redis(
            host=config.redis_ip,
            port=config.redis_port,
            db=0,
            decode_responses=False
        )
        self.pipeline = self.client.pipeline()
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.client.ping()
            print(f"Connected to Redis at {self.config.redis_ip}:{self.config.redis_port}")
        except redis.ConnectionError as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")

    def _safe_json_load(self, raw: Optional[bytes]) -> Optional[Any]:
        if raw is None:
            return None
        try:
            return json.loads(raw.decode('utf-8'))
        except:
            return None

    def read_state(self) -> Optional[np.ndarray]:
        """
        Read current state from Redis.

        Returns:
            qpos: (state_dim,) array [state_body, state_hand(s)], or None if unavailable
                  - Single hand (left/right): state_body + state_hand (51D or 54D)
                  - Both hands: state_body + state_hand_left + state_hand_right (71D or 74D)
                  state_body is 31D if config.state_body_31d else 34D
        """
        try:
            if self.config.hand_side == "both":
                # Read both hands
                self.pipeline.get(self.config.key_state_body)
                self.pipeline.get(self.config.key_state_hand_left)
                self.pipeline.get(self.config.key_state_hand_right)
                results = self.pipeline.execute()

                state_body = self._safe_json_load(results[0])
                state_hand_left = self._safe_json_load(results[1])
                state_hand_right = self._safe_json_load(results[2])

                if state_body is None:
                    return None

                state_body = np.array(state_body, dtype=np.float32)
                
                # If state_body_31d is enabled and we have 34D data, extract [3:34]
                if self.config.state_body_31d and len(state_body) == 34:
                    state_body = state_body[3:34]  # Extract roll/pitch + joints (31D)
                elif self.config.state_body_31d and len(state_body) == 31:
                    pass  # Already 31D, use as-is
                elif not self.config.state_body_31d and len(state_body) == 31:
                    # Dataset is 31D but we expected 34D - this is a mismatch
                    print(f"Warning: state_body is 31D but state_body_31d=False. Consider using --state_body_31d")

                # Handle missing hand states
                if state_hand_left is None:
                    state_hand_left = np.zeros(20, dtype=np.float32)
                else:
                    state_hand_left = np.array(state_hand_left, dtype=np.float32)
                
                if state_hand_right is None:
                    state_hand_right = np.zeros(20, dtype=np.float32)
                else:
                    state_hand_right = np.array(state_hand_right, dtype=np.float32)

                qpos = np.concatenate([state_body, state_hand_left, state_hand_right])
                return qpos
            else:
                # Read single hand (left or right)
                hand_key = self.config.key_state_hand_left if self.config.hand_side == "left" else self.config.key_state_hand_right
                
                self.pipeline.get(self.config.key_state_body)
                self.pipeline.get(hand_key)
                results = self.pipeline.execute()

                state_body = self._safe_json_load(results[0])
                state_hand = self._safe_json_load(results[1])

                if state_body is None:
                    return None

                state_body = np.array(state_body, dtype=np.float32)
                
                # If state_body_31d is enabled and we have 34D data, extract [3:34]
                if self.config.state_body_31d and len(state_body) == 34:
                    state_body = state_body[3:34]  # Extract roll/pitch + joints (31D)
                elif self.config.state_body_31d and len(state_body) == 31:
                    pass  # Already 31D, use as-is
                elif not self.config.state_body_31d and len(state_body) == 31:
                    # Dataset is 31D but we expected 34D - this is a mismatch
                    print(f"Warning: state_body is 31D but state_body_31d=False. Consider using --state_body_31d")

                # Handle missing hand state
                if state_hand is None:
                    state_hand = np.zeros(20, dtype=np.float32)
                else:
                    state_hand = np.array(state_hand, dtype=np.float32)

                qpos = np.concatenate([state_body, state_hand])
                return qpos

        except Exception as e:
            print(f"Error reading state: {e}")
            return None

    def publish_action(
        self,
        action_body: np.ndarray,
        action_neck: Optional[np.ndarray] = None,
        action_hand_left: Optional[np.ndarray] = None,
        action_hand_right: Optional[np.ndarray] = None,
    ):
        """
        Publish action to Redis.

        Args:
            action_body: (35,) body action
            action_neck: (2,) neck action [yaw, pitch], optional
            action_hand_left: (7,) optional (will default to zeros if None)
            action_hand_right: (7,) optional (will default to zeros if None)
        """
        self.pipeline.set(self.config.key_action_body, json.dumps(action_body.tolist()))

        if action_neck is not None:
            self.pipeline.set(self.config.key_action_neck, json.dumps(action_neck.tolist()))
        else:
            # Keep key present for low-level servers that expect it
            self.pipeline.set(self.config.key_action_neck, json.dumps([0.0, 0.0]))

        # Keep hand keys present for low-level servers that expect them
        if action_hand_left is None:
            action_hand_left = np.zeros(7, dtype=np.float32)
        if action_hand_right is None:
            action_hand_right = np.zeros(7, dtype=np.float32)
        self.pipeline.set(self.config.key_action_hand_left, json.dumps(np.asarray(action_hand_left, dtype=float).tolist()))
        self.pipeline.set(self.config.key_action_hand_right, json.dumps(np.asarray(action_hand_right, dtype=float).tolist()))

        self.pipeline.set(self.config.key_t_action, int(time.time() * 1000))
        self.pipeline.execute()

    def set_wuji_hand_mode(self, mode: str):
        """Set Wuji hand mode keys if present in the system (best-effort).

        Modes expected by server_wuji_hand_redis: follow | hold | default
        """
        try:
            key_l = f"wuji_hand_mode_left_{self.config.robot_key}"
            key_r = f"wuji_hand_mode_right_{self.config.robot_key}"
            self.pipeline.set(key_l, mode)
            self.pipeline.set(key_r, mode)
            self.pipeline.execute()
        except Exception:
            pass

    def publish_wuji_qpos_target_left(self, qpos_target_20: np.ndarray):
        """Publish 20D Wuji left-hand joint target to Redis (best-effort).

        Keys follow the existing naming used by server_wuji_hand_redis.py:
          - action_wuji_qpos_target_left_{robot_key}
          - t_action_wuji_hand_left_{robot_key}
        """
        try:
            key_action = f"action_wuji_qpos_target_left_{self.config.robot_key}"
            key_t = f"t_action_wuji_hand_left_{self.config.robot_key}"
            flat = np.asarray(qpos_target_20, dtype=np.float32).reshape(-1)
            if flat.shape[0] != 20:
                return
            self.pipeline.set(key_action, json.dumps(flat.tolist()))
            self.pipeline.set(key_t, int(time.time() * 1000))
            self.pipeline.execute()
        except Exception:
            pass

    def publish_wuji_qpos_target_right(self, qpos_target_20: np.ndarray):
        """Publish 20D Wuji right-hand joint target to Redis (best-effort).

        Keys follow the existing naming used by server_wuji_hand_redis.py:
          - action_wuji_qpos_target_right_{robot_key}
          - t_action_wuji_hand_right_{robot_key}
        """
        try:
            key_action = f"action_wuji_qpos_target_right_{self.config.robot_key}"
            key_t = f"t_action_wuji_hand_right_{self.config.robot_key}"
            flat = np.asarray(qpos_target_20, dtype=np.float32).reshape(-1)
            if flat.shape[0] != 20:
                return
            self.pipeline.set(key_action, json.dumps(flat.tolist()))
            self.pipeline.set(key_t, int(time.time() * 1000))
            self.pipeline.execute()
        except Exception:
            pass

    def publish_wuji_qpos_target(self, qpos_target_20: np.ndarray, hand_side: str = "left"):
        """Publish 20D Wuji hand joint target to Redis (best-effort).

        Args:
            qpos_target_20: (20,) array of joint targets
            hand_side: "left" or "right"
        """
        if hand_side == "left":
            self.publish_wuji_qpos_target_left(qpos_target_20)
        elif hand_side == "right":
            self.publish_wuji_qpos_target_right(qpos_target_20)
        else:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side}")


# =============================================================================
# Vision Client (ZMQ)
# =============================================================================

class VisionReader:
    """Read images from ZMQ stream."""

    def __init__(self, server_ip: str = "192.168.123.164", port: int = 5555,
                 img_shape: tuple = (480, 640, 3),  # Match training image size
                 source_bgr: bool = True):
        from multiprocessing import shared_memory
        from data_utils.vision_client import VisionClient

        self.img_shape = img_shape
        # VisionClient uses cv2.imdecode(..., IMREAD_COLOR) -> BGR.
        # We expose RGB to the rest of the code (policy expects RGB like training data).
        self.source_bgr = bool(source_bgr)
        shm_bytes = int(np.prod(img_shape) * np.uint8().itemsize)
        self.shm = shared_memory.SharedMemory(create=True, size=shm_bytes)
        self.image_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)

        self.client = VisionClient(
            server_address=server_ip,
            port=port,
            img_shape=img_shape,
            img_shm_name=self.shm.name,
            image_show=False,
            depth_show=False,
            unit_test=True,
        )

        import threading
        self.thread = threading.Thread(target=self.client.receive_process, daemon=True)
        self.thread.start()
        print(f"Vision client started: {server_ip}:{port}")

    def get_image(self) -> np.ndarray:
        """Get latest RGB image (H, W, 3)."""
        img = self.image_array.copy()
        if self.source_bgr and img.ndim == 3 and img.shape[2] == 3:
            # BGR -> RGB
            return img[:, :, ::-1].copy()
        return img

    def close(self):
        try:
            self.client.running = False
            self.thread.join(timeout=1.0)
            self.shm.unlink()
            self.shm.close()
        except:
            pass


# =============================================================================
# Keyboard Toggle (k/p semantics)
# =============================================================================

class KeyboardToggle:
    """Terminal keyboard toggles (like xrobot_teleop_to_robot_w_hand_keyboard.py).

    - k: toggle send_enabled. When disabled: publish safe idle action (and disable hold).
    - p: toggle hold_position. When enabled: keep publishing cached last action.
    """

    def __init__(self, enabled: bool, toggle_send_key: str = "k", hold_position_key: str = "p"):
        self.enabled = bool(enabled)
        self.toggle_send_key = (toggle_send_key or "k")[0]
        self.hold_position_key = (hold_position_key or "p")[0]

        self._send_enabled = True
        self._hold_position_enabled = False
        self._lock = threading.Lock()

        self._thread = None
        self._stop = threading.Event()
        self._stdin_fd = None
        self._stdin_old_termios = None

    def start(self):
        if not self.enabled:
            return
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass
        self._thread = None

    def get(self):
        with self._lock:
            return self._send_enabled, self._hold_position_enabled

    def _loop(self):
        try:
            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            self._stdin_old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            print(f"[Keyboard] Press '{self.toggle_send_key}' to toggle send_enabled (disabled => safe idle)")
            print(f"[Keyboard] Press '{self.hold_position_key}' to toggle hold_position (hold last action)")
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == self.toggle_send_key:
                    with self._lock:
                        self._send_enabled = not self._send_enabled
                        enabled = self._send_enabled
                        if not self._send_enabled:
                            self._hold_position_enabled = False
                    print(f"[Keyboard] send_enabled => {enabled}")
                elif ch == self.hold_position_key:
                    with self._lock:
                        if not self._send_enabled:
                            self._hold_position_enabled = False
                            enabled = self._hold_position_enabled
                        else:
                            self._hold_position_enabled = not self._hold_position_enabled
                            enabled = self._hold_position_enabled
                    print(f"[Keyboard] hold_position_enabled => {enabled}")
        except Exception as e:
            print(f"[Keyboard] Keyboard listener unavailable: {e}")
        finally:
            try:
                if self._stdin_fd is not None and self._stdin_old_termios is not None:
                    termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_termios)
            except Exception:
                pass


# =============================================================================
# Common helpers (safety publish)
# =============================================================================

def _get_safe_idle_body_35(robot_key: str) -> np.ndarray:
    """Safe idle 35D action for the given robot_key."""
    from data_utils.params import DEFAULT_MIMIC_OBS
    base = DEFAULT_MIMIC_OBS.get(robot_key, DEFAULT_MIMIC_OBS["unitree_g1"])
    return np.array(base[:35], dtype=np.float32)


def _publish_with_kp_safety(
    redis_io: "RedisIO",
    kb: KeyboardToggle,
    safe_idle_body_35: np.ndarray,
    cached_body: np.ndarray,
    cached_neck: np.ndarray,
    desired_body: np.ndarray,
    desired_neck: Optional[np.ndarray] = None,
):
    """
    Publish action with xrobot-like k/p semantics.

    - k (send_enabled=False): publish safe idle, set wuji mode=default
    - p (hold_enabled=True): publish cached action, set wuji mode=hold
    - else: publish desired action, update cache, set wuji mode=follow

    Returns:
        pub_body, pub_neck, cached_body, cached_neck, advance_step(bool)
    """
    send_enabled, hold_enabled = kb.get()

    if not send_enabled:
        pub_body = safe_idle_body_35
        pub_neck = cached_neck
        redis_io.set_wuji_hand_mode("default")
        advance = False
    elif hold_enabled:
        pub_body = cached_body
        pub_neck = cached_neck
        redis_io.set_wuji_hand_mode("hold")
        advance = False
    else:
        pub_body = np.asarray(desired_body, dtype=np.float32)
        pub_neck = np.asarray(desired_neck if desired_neck is not None else cached_neck, dtype=np.float32)
        cached_body = pub_body.copy()
        cached_neck = pub_neck.copy()
        redis_io.set_wuji_hand_mode("follow")
        advance = True

    redis_io.publish_action(pub_body, pub_neck)
    return pub_body, pub_neck, cached_body, cached_neck, advance


def _ease(alpha: float, ease: str = "cosine") -> float:
    """Match init_pose ramp easing."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if str(ease) == "linear":
        return a
    # cosine ease-in-out
    return 0.5 - 0.5 * float(np.cos(np.pi * a))


@dataclass
class _ToggleRamp:
    """Smooth transition on k/p toggles (same semantics as init_pose ramp)."""
    seconds: float = 0.0
    ease: str = "cosine"
    active: bool = False
    t0: float = 0.0
    from_body: np.ndarray = field(default_factory=lambda: np.zeros(35, dtype=np.float32))
    from_neck: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    to_body: np.ndarray = field(default_factory=lambda: np.zeros(35, dtype=np.float32))
    to_neck: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    target_mode: str = "follow"  # follow|hold|default

    def start(
        self,
        from_body: np.ndarray,
        from_neck: np.ndarray,
        to_body: np.ndarray,
        to_neck: np.ndarray,
        target_mode: str,
        seconds: float,
        ease: str,
    ):
        self.seconds = float(seconds)
        self.ease = str(ease)
        self.active = self.seconds > 1e-6
        self.t0 = time.time()
        self.from_body = np.asarray(from_body, dtype=np.float32).reshape(35).copy()
        self.from_neck = np.asarray(from_neck, dtype=np.float32).reshape(2).copy()
        self.to_body = np.asarray(to_body, dtype=np.float32).reshape(35).copy()
        self.to_neck = np.asarray(to_neck, dtype=np.float32).reshape(2).copy()
        self.target_mode = str(target_mode)

    def value(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        if not self.active:
            return self.to_body, self.to_neck, False
        alpha = (time.time() - self.t0) / max(1e-6, float(self.seconds))
        w = _ease(alpha, self.ease)
        body = self.from_body + w * (self.to_body - self.from_body)
        neck = self.from_neck + w * (self.to_neck - self.from_neck)
        done = float(alpha) >= 1.0 - 1e-6
        if done:
            self.active = False
        return body.astype(np.float32), neck.astype(np.float32), (not self.active)


# =============================================================================
# ACT Policy Wrapper
# =============================================================================

class ACTPolicyWrapper:
    """Wrapper for ACT policy inference."""

    def __init__(self, ckpt_dir: str, config: Config, policy_config: Optional[dict] = None):
        self.config = config
        self.device = 'cuda' if config.use_gpu else 'cpu'

        # Load stats (saved during training)
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        print(f"Loaded normalization stats from {stats_path}")
        print(f"  qpos_mean shape: {self.stats['qpos_mean'].shape}")
        print(f"  action_mean shape: {self.stats['action_mean'].shape}")

        # Load relative action settings from stats
        self.use_relative_actions = self.stats.get('use_relative_actions', False)
        self.state_body_dim = self.stats.get('state_body_dim', 34)
        if self.use_relative_actions:
            print(f"[Policy] Relative actions enabled (state_body_dim={self.state_body_dim})")

        # Load policy
        self.policy = self._load_policy(ckpt_dir, policy_config)
        self.policy.eval()

        # Temporal aggregation state
        self.temporal_agg = config.temporal_agg
        self.chunk_size = config.chunk_size
        self.all_time_actions = None
        self.t = 0

    def _load_policy(self, ckpt_dir: str, policy_config: Optional[dict] = None):
        """Load ACT policy from checkpoint."""
        import torch
        from policy import ACTPolicy

        # Default policy config - MUST match training config
        # These are the typical values used in ACT training
        if policy_config is None:
            policy_config = {
                'lr': 1e-5,                      # Not used at inference
                'lr_backbone': 1e-5,             # Not used at inference
                'num_queries': self.config.chunk_size,  # chunk_size
                'kl_weight': 10,                 # Not used at inference
                'hidden_dim': 512,               # Must match training
                'dim_feedforward': 3200,         # Must match training
                'backbone': 'resnet18',          # Must match training
                'enc_layers': 4,                 # Must match training
                'dec_layers': 7,                 # Must match training
                'nheads': 8,                     # Must match training
                'camera_names': ['head'],        # Must match training
            }

        print(f"Policy config: hidden_dim={policy_config['hidden_dim']}, "
              f"chunk_size={policy_config['num_queries']}")

        policy = ACTPolicy(policy_config)

        # Load weights
        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')

        state_dict = torch.load(ckpt_path, map_location=self.device)
        loading_status = policy.load_state_dict(state_dict)
        print(f"Loading status: {loading_status}")

        if self.config.use_gpu:
            policy.cuda()

        print(f"Loaded policy from {ckpt_path}")
        return policy

    def reset(self):
        """Reset temporal aggregation state."""
        self.t = 0
        self.all_time_actions = None

    def preprocess_qpos(self, qpos: np.ndarray) -> 'torch.Tensor':
        """Normalize qpos."""
        import torch
        qpos_norm = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0)
        if self.config.use_gpu:
            qpos_tensor = qpos_tensor.cuda()
        return qpos_tensor

    def preprocess_image(self, image: np.ndarray) -> 'torch.Tensor':
        """Normalize image to (1, 1, C, H, W) tensor."""
        import torch
        # (H, W, C) -> (1, 1, C, H, W)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW
        image = image[np.newaxis, np.newaxis, ...]  # (1, 1, C, H, W)
        image_tensor = torch.from_numpy(image).float()
        if self.config.use_gpu:
            image_tensor = image_tensor.cuda()
        return image_tensor

    def postprocess_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action."""
        return action * self.stats['action_std'] + self.stats['action_mean']

    def __call__(self, qpos: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Run policy inference.

        Args:
            qpos: (state_dim,) state - 51/54 (single hand) or 71/74 (both hands)
            image: (H, W, 3) RGB image

        Returns:
            action: (action_dim,) denormalized action (in absolute space)
                    55D for single hand, 75D for both hands
        """
        import torch

        qpos_tensor = self.preprocess_qpos(qpos)
        image_tensor = self.preprocess_image(image)

        with torch.inference_mode():
            if self.temporal_agg:
                return self._infer_with_temporal_agg(qpos_tensor, image_tensor, qpos_raw=qpos)
            else:
                return self._infer_simple(qpos_tensor, image_tensor, qpos_raw=qpos)

    def _infer_simple(self, qpos_tensor, image_tensor, qpos_raw: np.ndarray) -> np.ndarray:
        """Simple inference without temporal aggregation.
        
        Args:
            qpos_raw: (state_dim,) raw unnormalized qpos - needed for relative->absolute conversion
        """
        import torch

        # Query policy every chunk_size steps
        if self.t % self.config.chunk_size == 0:
            self._cached_actions = self.policy(qpos_tensor, image_tensor)  # (1, chunk_size, action_dim)
            # Cache the current qpos for this chunk (for relative action conversion)
            self._cached_qpos_raw = qpos_raw.copy()

        action_idx = self.t % self.config.chunk_size
        raw_action = self._cached_actions[0, action_idx].cpu().numpy()
        self.t += 1

        # Denormalize
        action = self.postprocess_action(raw_action)
        
        # Convert to absolute if using relative actions
        if self.use_relative_actions:
            action = convert_actions_to_absolute(
                action.reshape(1, -1), 
                self._cached_qpos_raw, 
                state_body_dim=self.state_body_dim
            )[0]
        
        return action

    def _infer_with_temporal_agg(self, qpos_tensor, image_tensor, qpos_raw: np.ndarray) -> np.ndarray:
        """Inference with temporal aggregation (exponential weighting).
        
        For relative actions (Scheme B):
          1. Policy outputs normalized relative actions for a chunk
          2. Denormalize the entire chunk
          3. Convert the entire chunk from relative to absolute space
          4. Store absolute actions in the buffer
          5. Temporal aggregation happens in absolute space (mathematically correct)
        
        Args:
            qpos_raw: (state_dim,) raw unnormalized qpos - needed for relative->absolute conversion
        """
        import torch

        # Buffer size for temporal aggregation. This should be larger than the
        # maximum episode length you expect during inference/offline eval.
        max_timesteps = 5000  # previously 1000

        # Initialize buffer on first call
        if self.all_time_actions is None:
            self.all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + self.chunk_size, self.config.action_dim]
            )
            if self.config.use_gpu:
                self.all_time_actions = self.all_time_actions.cuda()

        t = self.t

        # Safety: avoid indexing past the temporal buffer if an episode is longer
        # than max_timesteps. In that rare case we clamp t to the last valid row
        # and keep re-using it, and print a warning once.
        if t >= max_timesteps:
            if not hasattr(self, "_temporal_agg_clamp_warned") or not self._temporal_agg_clamp_warned:
                print(
                    f"[TemporalAgg] t={t} exceeds buffer size {max_timesteps}. "
                    f"Clamping to {max_timesteps-1}. Consider increasing max_timesteps "
                    f"if your episodes are this long."
                )
                self._temporal_agg_clamp_warned = True
            t = max_timesteps - 1

        # Query policy (every step for temporal_agg)
        all_actions = self.policy(qpos_tensor, image_tensor)  # (1, chunk_size, action_dim)

        # Convert to numpy and denormalize the entire chunk
        all_actions_np = all_actions[0].cpu().numpy()  # (chunk_size, action_dim)
        all_actions_denorm = self.postprocess_action(all_actions_np)  # (chunk_size, action_dim)
        
        # If using relative actions, convert the entire chunk to absolute space
        # This is mathematically correct: all relative actions are w.r.t. the same qpos_raw,
        # so we convert them all to absolute before aggregation
        if self.use_relative_actions:
            all_actions_denorm = convert_actions_to_absolute(
                all_actions_denorm, 
                qpos_raw, 
                state_body_dim=self.state_body_dim
            )
        
        # Convert back to tensor and store in buffer (now in absolute space)
        all_actions_abs_tensor = torch.from_numpy(all_actions_denorm).float()
        if self.config.use_gpu:
            all_actions_abs_tensor = all_actions_abs_tensor.cuda()
        
        # Store in buffer
        self.all_time_actions[t, t:t+self.chunk_size] = all_actions_abs_tensor

        # Get all actions that predict current timestep (all in absolute space)
        actions_for_curr_step = self.all_time_actions[:, t]
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]

        # Exponential weighting (newer predictions weighted more)
        # Aggregation in absolute space is mathematically correct
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).float().unsqueeze(1)
        if self.config.use_gpu:
            exp_weights = exp_weights.cuda()

        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0).cpu().numpy()
        self.t += 1

        # Return absolute action (already denormalized, already absolute)
        return raw_action


# =============================================================================
# HDF5 Dataset Reader
# =============================================================================

class DatasetReader:
    """Read episodes from HDF5 dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.episode_ids = self._get_episode_ids()
        print(f"Dataset: {dataset_path}")
        print(f"Episodes: {len(self.episode_ids)} ({self.episode_ids[:5]}...)")

    def _get_episode_ids(self) -> list:
        with h5py.File(self.dataset_path, 'r') as f:
            ids = [int(k.split('_')[1]) for k in f.keys() if k.startswith('episode_')]
        return sorted(ids)

    def load_episode(self, episode_id: int, load_observations: bool = False, hand_side: str = "left", state_body_31d: bool = False) -> dict:
        """
        Load a single episode.

        Args:
            episode_id: Episode index
            load_observations: If True, also load qpos and images for policy input
            hand_side: "left", "right", or "both" to select which hand data to load
            state_body_31d: If True and dataset has 34D state_body, extract [3:34] to get 31D

        Returns:
            dict with keys:
                - action_body: (T, 35) GT body actions
                - action_neck: (T, 2) GT neck actions
                - action_hand: (T, 20 or 40) GT Wuji hand target(s)
                       20D for single hand, 40D for both hands
                - num_timesteps: int
                - text: str
                - qpos: (T, state_dim) state observations (if load_observations=True)
                        state_dim = 51/54 (single) or 71/74 (both)
                - images: (T, H, W, 3) RGB images (if load_observations=True)
        """
        if hand_side not in ["left", "right", "both"]:
            raise ValueError(f"hand_side must be 'left', 'right', or 'both', got {hand_side}")

        with h5py.File(self.dataset_path, 'r') as f:
            key = f'episode_{episode_id:04d}'
            if key not in f:
                raise ValueError(f"Episode {episode_id} not found")

            ep = f[key]
            data = {
                'action_body': ep['action_body'][:],
                'action_neck': ep['action_neck'][:],
                'num_timesteps': ep.attrs['num_timesteps'],
                'text': ep.attrs.get('text', '{}'),
            }

            # Optional: hand 20D or 40D action target (for visualization/comparison)
            if hand_side == "both":
                # Load both hands and concatenate
                hand_left_key = 'action_wuji_qpos_target_left'
                hand_right_key = 'action_wuji_qpos_target_right'
                
                T = int(ep.attrs['num_timesteps'])
                if hand_left_key in ep:
                    action_hand_left = ep[hand_left_key][:]
                else:
                    action_hand_left = np.zeros((T, 20), dtype=np.float32)
                
                if hand_right_key in ep:
                    action_hand_right = ep[hand_right_key][:]
                else:
                    action_hand_right = np.zeros((T, 20), dtype=np.float32)
                
                data['action_hand'] = np.concatenate([action_hand_left, action_hand_right], axis=1)  # (T, 40)
            else:
                # Single hand
                hand_action_key = f'action_wuji_qpos_target_{hand_side}'
                if hand_action_key in ep:
                    data['action_hand'] = ep[hand_action_key][:]
                else:
                    # Keep backward compatibility with datasets that don't have hand targets
                    T = int(ep.attrs['num_timesteps'])
                    data['action_hand'] = np.zeros((T, 20), dtype=np.float32)

            if load_observations:
                # Load state: qpos = state_body + state_wuji_hand_{hand_side}(s)
                state_body = ep['state_body'][:]  # (T, 31 or 34)
                dataset_state_body_dim = state_body.shape[1]
                
                # Handle state_body dimension conversion
                if state_body_31d:
                    if dataset_state_body_dim == 34:
                        # Dataset is 34D, extract [3:34] to get 31D
                        state_body = state_body[:, 3:34]
                    elif dataset_state_body_dim == 31:
                        # Dataset is already 31D, use as-is
                        pass
                    else:
                        print(f"Warning: unexpected state_body dim {dataset_state_body_dim}")
                else:
                    if dataset_state_body_dim == 31:
                        print(f"Warning: dataset has 31D state_body but state_body_31d=False. Consider using --state_body_31d")
                
                if hand_side == "both":
                    # Load both hands
                    hand_left_key = 'state_wuji_hand_left'
                    hand_right_key = 'state_wuji_hand_right'
                    
                    T = int(ep.attrs['num_timesteps'])
                    if hand_left_key in ep:
                        state_hand_left = ep[hand_left_key][:]  # (T, 20)
                    else:
                        state_hand_left = np.zeros((T, 20), dtype=np.float32)
                    
                    if hand_right_key in ep:
                        state_hand_right = ep[hand_right_key][:]  # (T, 20)
                    else:
                        state_hand_right = np.zeros((T, 20), dtype=np.float32)
                    
                    data['qpos'] = np.concatenate([state_body, state_hand_left, state_hand_right], axis=1)  # (T, 71 or 74)
                else:
                    # Single hand
                    hand_state_key = f'state_wuji_hand_{hand_side}'
                    if hand_state_key in ep:
                        state_hand = ep[hand_state_key][:] # (T, 20)
                    else:
                        # Fallback to zeros if hand state not found
                        T = int(ep.attrs['num_timesteps'])
                        state_hand = np.zeros((T, 20), dtype=np.float32)
                    data['qpos'] = np.concatenate([state_body, state_hand], axis=1)  # (T, 51 or 54)

                # Load images
                head = ep['head'][:]
                # Some datasets store JPEG-encoded bytes per frame as an object array: shape (T,), dtype=object.
                # Decode to RGB uint8 (T, H, W, 3) for policy input.
                if isinstance(head, np.ndarray) and (head.dtype == object or head.ndim == 1):
                    try:
                        import cv2
                        decoded = []
                        fallback_hw = (480, 640)  # common default
                        for i in range(len(head)):
                            buf = head[i]
                            # buf can be bytes or a 1D uint8 array
                            if isinstance(buf, (bytes, bytearray)):
                                arr = np.frombuffer(buf, dtype=np.uint8)
                            else:
                                arr = np.asarray(buf, dtype=np.uint8).reshape(-1)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if bgr is None:
                                # Decode failed; use a black frame
                                h, w = fallback_hw
                                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                            else:
                                rgb = bgr[:, :, ::-1]  # BGR -> RGB
                                fallback_hw = (rgb.shape[0], rgb.shape[1])
                            decoded.append(rgb)
                        data['images'] = np.stack(decoded, axis=0)
                    except Exception as e:
                        # If OpenCV is unavailable or decoding fails, keep raw bytes and let caller handle it.
                        print(f"[DatasetReader] Warning: failed to decode images from 'head': {e}")
                        data['images'] = head
                else:
                    # Already stored as raw RGB frames (T, H, W, 3)
                    data['images'] = head

        return data

    def random_episode_id(self) -> int:
        return np.random.choice(self.episode_ids)


# =============================================================================
# Replay Mode
# =============================================================================

def replay_episode(
    dataset_path: str,
    episode_id: Optional[int] = None,
    config: Optional[Config] = None,
    sim_only: bool = False,
    output_video: str = "replay_gt.mp4",
    vision_ip: str = "192.168.123.164",
    vision_port: int = 5555,
    rgb_stream: bool = False,
    sim_stream: bool = False,
    sim_save_vid: Optional[str] = None,
    sim_hand: bool = True,
    keyboard_toggle_send: bool = False,
    toggle_send_key: str = "k",
    hold_position_key: str = "p",
    hand_side: str = "left",
    toggle_ramp_seconds: float = 0.0,
    toggle_ramp_ease: str = "cosine",
):
    """Replay an episode from the dataset."""
    config = config or Config()
    reader = DatasetReader(dataset_path)

    if episode_id is None:
        episode_id = reader.random_episode_id()

    print(f"\n{'='*60}")
    print(f"Replaying episode {episode_id} (hand: {hand_side})")
    print(f"{'='*60}")

    data = reader.load_episode(episode_id, hand_side=hand_side)
    action_body = data['action_body']
    action_neck = data['action_neck']
    action_hand_20 = data.get('action_hand', None)  # (T, 20) if available
    T = data['num_timesteps']

    try:
        text = json.loads(data['text'])
        print(f"Goal: {text.get('goal', 'N/A')}")
    except:
        pass

    print(f"Timesteps: {T}")

    if sim_only:
        # Visualize GT actions in sim (either save video, stream window, or both)
        print(f"\nGenerating sim visualization...")
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act', 'sim_viz'))
            from visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths, save_video

            # If user requested streaming and/or recording, do a step-wise loop so we can show a window
            if sim_stream or sim_save_vid:
                import cv2
                paths = get_default_paths()
                viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
                hand_viz = None
                hand_viz_right = None
                cached_hand = None
                cached_hand_right = None
                use_hand = bool(sim_hand)
                use_both_hands = (hand_side == "both")
                
                if use_hand:
                    if use_both_hands:
                        # Initialize both hand visualizers
                        hand_xml_key_left = 'left_hand_xml'
                        hand_xml_key_right = 'right_hand_xml'
                        if hand_xml_key_left not in paths or hand_xml_key_right not in paths:
                            print(f"[Sim] WARN: Hand XML paths not found; disabling hand viz. Available: {list(paths.keys())}")
                            use_hand = False
                        else:
                            hand_viz = HandVisualizer(paths[hand_xml_key_left], hand_side='left')
                            hand_viz_right = HandVisualizer(paths[hand_xml_key_right], hand_side='right')
                            cached_hand = np.zeros(20, dtype=np.float32)
                            cached_hand_right = np.zeros(20, dtype=np.float32)
                    else:
                        # Single hand
                        hand_xml_key = f'{hand_side}_hand_xml'
                        if hand_xml_key not in paths:
                            print(f"[Sim] WARN: Hand XML path for {hand_side} not found; disabling hand viz. Available: {list(paths.keys())}")
                            use_hand = False
                        else:
                            hand_viz = HandVisualizer(paths[hand_xml_key], hand_side=hand_side)
                            cached_hand = np.zeros(20, dtype=np.float32)

                if sim_stream:
                    cv2.namedWindow("Sim (Body)", cv2.WINDOW_NORMAL)

                writer = None
                if sim_save_vid:
                    if sim_save_vid == "__AUTO__":
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        sim_save_vid = f"replay_sim_{episode_id}_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    if use_both_hands:
                        out_w, out_h = (1920, 480)  # body + left hand + right hand
                    else:
                        out_w, out_h = (1280, 480) if use_hand else (640, 480)
                    writer = cv2.VideoWriter(sim_save_vid, fourcc, 30, (out_w, out_h))
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter failed to open: {sim_save_vid}")
                    print(f"[Sim] Recording enabled: {sim_save_vid}")

                print(f"Streaming {T} GT actions...")
                viz._reset()
                if T > 0:
                    # warmup to first pose
                    for _ in range(100):
                        viz.step(np.asarray(action_body[0], dtype=np.float32))
                        
                # Helper to add label
                def add_label(frame, label):
                    frame = frame.copy()
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    return frame
                    
                for t in range(T):
                    body_frame = viz.step(np.asarray(action_body[t], dtype=np.float32))
                    out_frame = add_label(body_frame, "Body")
                    
                    if hand_viz is not None:
                        # If dataset includes hand target, visualize it; else keep zeros
                        if action_hand_20 is not None:
                            if use_both_hands:
                                # Split into left (20D) and right (20D)
                                cached_hand = np.asarray(action_hand_20[t, :20], dtype=np.float32).reshape(-1)
                                cached_hand_right = np.asarray(action_hand_20[t, 20:], dtype=np.float32).reshape(-1)
                            else:
                                cached_hand = np.asarray(action_hand_20[t], dtype=np.float32).reshape(-1)
                        
                        hand_frame = hand_viz.step(cached_hand)
                        hand_frame = add_label(hand_frame, "L-Hand" if use_both_hands else f"{hand_side[0].upper()}-Hand")
                        
                        if use_both_hands and hand_viz_right is not None:
                            hand_frame_right = hand_viz_right.step(cached_hand_right)
                            hand_frame_right = add_label(hand_frame_right, "R-Hand")
                            out_frame = np.concatenate([out_frame, hand_frame, hand_frame_right], axis=1)
                        else:
                            out_frame = np.concatenate([out_frame, hand_frame], axis=1)
                            
                    if sim_stream:
                        cv2.imshow("Sim (Body)", out_frame[:, :, ::-1])  # RGB->BGR
                        cv2.waitKey(1)
                    if writer is not None:
                        writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

                if writer is not None:
                    writer.release()
                    print(f"[Sim] Saved video to {sim_save_vid}")
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            else:
                # Render all frames then save to output_video (supports --sim_hand side-by-side)
                paths = get_default_paths()
                viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
                hand_viz = None
                hand_viz_right = None
                cached_hand = None
                cached_hand_right = None
                use_hand = bool(sim_hand)
                use_both_hands = (hand_side == "both")
                
                if use_hand:
                    if use_both_hands:
                        # Initialize both hand visualizers
                        hand_xml_key_left = 'left_hand_xml'
                        hand_xml_key_right = 'right_hand_xml'
                        if hand_xml_key_left not in paths or hand_xml_key_right not in paths:
                            print(f"[Sim] WARN: Hand XML paths not found; disabling hand viz. Available: {list(paths.keys())}")
                            use_hand = False
                        else:
                            hand_viz = HandVisualizer(paths[hand_xml_key_left], hand_side='left')
                            hand_viz_right = HandVisualizer(paths[hand_xml_key_right], hand_side='right')
                            cached_hand = np.zeros(20, dtype=np.float32)
                            cached_hand_right = np.zeros(20, dtype=np.float32)
                    else:
                        # Single hand
                        hand_xml_key = f'{hand_side}_hand_xml'
                        if hand_xml_key not in paths:
                            print(f"[Sim] WARN: Hand XML path for {hand_side} not found; disabling hand viz. Available: {list(paths.keys())}")
                            use_hand = False
                        else:
                            hand_viz = HandVisualizer(paths[hand_xml_key], hand_side=hand_side)
                            cached_hand = np.zeros(20, dtype=np.float32)

                print(f"Visualizing {T} GT actions...")
                # warmup to first pose
                viz._reset()
                if T > 0:
                    for _ in range(100):
                        viz.step(np.asarray(action_body[0], dtype=np.float32))

                # Helper to add label
                def add_label(frame, label):
                    import cv2
                    frame = frame.copy()
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    return frame

                frames = []
                for t in range(T):
                    body_frame = viz.step(np.asarray(action_body[t], dtype=np.float32))
                    out_frame = add_label(body_frame, "Body")
                    
                    if hand_viz is not None:
                        if action_hand_20 is not None:
                            if use_both_hands:
                                # Split into left (20D) and right (20D)
                                cached_hand = np.asarray(action_hand_20[t, :20], dtype=np.float32).reshape(-1)
                                cached_hand_right = np.asarray(action_hand_20[t, 20:], dtype=np.float32).reshape(-1)
                            else:
                                cached_hand = np.asarray(action_hand_20[t], dtype=np.float32).reshape(-1)
                        
                        hand_frame = hand_viz.step(cached_hand)
                        hand_frame = add_label(hand_frame, "L-Hand" if use_both_hands else f"{hand_side[0].upper()}-Hand")
                        
                        if use_both_hands and hand_viz_right is not None:
                            hand_frame_right = hand_viz_right.step(cached_hand_right)
                            hand_frame_right = add_label(hand_frame_right, "R-Hand")
                            out_frame = np.concatenate([out_frame, hand_frame, hand_frame_right], axis=1)
                        else:
                            out_frame = np.concatenate([out_frame, hand_frame], axis=1)
                    
                    frames.append(out_frame)
                    if (t + 1) % 100 == 0:
                        print(f"  {t+1}/{T} frames")

                save_video(frames, output_video, fps=30)
                print(f"Saved video to {output_video}")

        except Exception as e:
            print(f"Sim visualization failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nReplay complete!")
        return

    # Publish to Redis
    redis_io = RedisIO(config)
    dt = 1.0 / config.frequency

    # Safe idle action (same semantics as xrobot/xdmocap teleop)
    safe_idle_body_35 = _get_safe_idle_body_35(config.robot_key)
    cached_body = safe_idle_body_35.copy()
    cached_neck = np.array([0.0, 0.0], dtype=np.float32)
    last_pub_body = cached_body.copy()
    last_pub_neck = cached_neck.copy()
    ramp = _ToggleRamp()

    # Keyboard toggles
    kb = KeyboardToggle(
        enabled=keyboard_toggle_send,
        toggle_send_key=toggle_send_key,
        hold_position_key=hold_position_key,
    )
    kb.start()
    last_send_enabled, last_hold_enabled = kb.get()

    # Optional windows
    vision = None
    if rgb_stream:
        try:
            import cv2
            vision = VisionReader(server_ip=vision_ip, port=vision_port)
            cv2.namedWindow("Robot RGB", cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[RGB] Failed to start RGB stream: {e}")
            vision = None

    sim_viz = None
    sim_writer = None
    sim_hand_viz = None
    sim_hand_viz_right = None
    sim_cached_hand = None
    sim_cached_hand_right = None
    use_both_hands = (hand_side == "both")
    
    if sim_stream or sim_save_vid:
        try:
            import cv2
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act', 'sim_viz'))
            from visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths
            paths = get_default_paths()
            sim_viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
            use_hand = bool(sim_hand)
            if use_hand:
                if use_both_hands:
                    # Initialize both hand visualizers
                    hand_xml_key_left = 'left_hand_xml'
                    hand_xml_key_right = 'right_hand_xml'
                    if hand_xml_key_left not in paths or hand_xml_key_right not in paths:
                        print(f"[Sim] WARN: Hand XML paths not found; disabling hand viz. Available: {list(paths.keys())}")
                        use_hand = False
                    else:
                        sim_hand_viz = HandVisualizer(paths[hand_xml_key_left], hand_side='left')
                        sim_hand_viz_right = HandVisualizer(paths[hand_xml_key_right], hand_side='right')
                        sim_cached_hand = np.zeros(20, dtype=np.float32)
                        sim_cached_hand_right = np.zeros(20, dtype=np.float32)
                else:
                    # Single hand
                    hand_xml_key = f'{hand_side}_hand_xml'
                    if hand_xml_key not in paths:
                        print(f"[Sim] WARN: Hand XML path for {hand_side} not found; disabling hand viz. Available: {list(paths.keys())}")
                        use_hand = False
                    else:
                        sim_hand_viz = HandVisualizer(paths[hand_xml_key], hand_side=hand_side)
                        sim_cached_hand = np.zeros(20, dtype=np.float32)
            if sim_stream:
                cv2.namedWindow("Sim (Body)", cv2.WINDOW_NORMAL)
            if sim_save_vid:
                if sim_save_vid == "__AUTO__":
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    sim_save_vid = f"replay_sim_{episode_id}_{ts}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if use_both_hands:
                    out_w, out_h = (1920, 480)  # body + left hand + right hand
                else:
                    out_w, out_h = (1280, 480) if use_hand else (640, 480)
                sim_writer = cv2.VideoWriter(sim_save_vid, fourcc, 30, (out_w, out_h))
                if not sim_writer.isOpened():
                    raise RuntimeError(f"VideoWriter failed to open: {sim_save_vid}")
                print(f"[Sim] Recording enabled: {sim_save_vid}")
        except Exception as e:
            print(f"[Sim] Failed to start sim visualization: {e}")
            sim_viz = None
            sim_writer = None
            sim_hand_viz = None
            sim_hand_viz_right = None
            sim_cached_hand = None
            sim_cached_hand_right = None

    print(f"\nPublishing to Redis... (Ctrl+C to stop)\n")

    try:
        t = 0
        while t < T:
            t0 = time.time()
            send_enabled, hold_enabled = kb.get()

            # Detect k/p state changes and start a smooth ramp (freeze timeline during ramp)
            if (send_enabled != last_send_enabled) or (hold_enabled != last_hold_enabled):
                # Determine target for the NEW mode
                if not send_enabled:
                    target_body = safe_idle_body_35
                    target_neck = cached_neck
                    target_mode = "default"
                elif hold_enabled:
                    target_body = cached_body
                    target_neck = cached_neck
                    target_mode = "hold"
                else:
                    target_body = np.asarray(action_body[t], dtype=np.float32)
                    target_neck = np.asarray(action_neck[t], dtype=np.float32)
                    target_mode = "follow"

                ramp.start(
                    from_body=last_pub_body,
                    from_neck=last_pub_neck,
                    to_body=target_body,
                    to_neck=target_neck,
                    target_mode=target_mode,
                    seconds=float(toggle_ramp_seconds),
                    ease=str(toggle_ramp_ease),
                )
                last_send_enabled, last_hold_enabled = send_enabled, hold_enabled

            if ramp.active:
                pub_body, pub_neck, _done = ramp.value()
                # During ramp, do not advance replay timeline; publish with target mode
                try:
                    redis_io.set_wuji_hand_mode(ramp.target_mode)
                except Exception:
                    pass
                redis_io.publish_action(pub_body, pub_neck)
                advance = False
            else:
                pub_body, pub_neck, cached_body, cached_neck, advance = _publish_with_kp_safety(
                    redis_io=redis_io,
                    kb=kb,
                    safe_idle_body_35=safe_idle_body_35,
                    cached_body=cached_body,
                    cached_neck=cached_neck,
                    desired_body=action_body[t],
                    desired_neck=action_neck[t],
                )

            last_pub_body = np.asarray(pub_body, dtype=np.float32).reshape(35).copy()
            last_pub_neck = np.asarray(pub_neck, dtype=np.float32).reshape(2).copy()

            # Publish Wuji hand 20D target (only when we advance the timeline).
            # In hold/default, the hand server will use mode to decide behavior.
            if advance and action_hand_20 is not None:
                try:
                    redis_io.publish_wuji_qpos_target(action_hand_20[t], hand_side=hand_side)
                except Exception:
                    pass

            if (t + 1) % 30 == 0 or t == T - 1:
                print(f"  [{t+1}/{T}] z={pub_body[2]:.3f}")

            # Windows update
            if vision is not None:
                import cv2
                img = vision.get_image()
                cv2.imshow("Robot RGB", img[:, :, ::-1])  # RGB->BGR for display
                cv2.waitKey(1)

            if sim_viz is not None:
                import cv2
                
                # Helper to add label
                def add_label(frame, label):
                    frame = frame.copy()
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    return frame
                
                frame_body = sim_viz.step(np.asarray(pub_body, dtype=np.float32))
                frame = add_label(frame_body, "Body")
                
                if sim_hand_viz is not None:
                    if action_hand_20 is not None:
                        if use_both_hands:
                            # Split into left (20D) and right (20D)
                            sim_cached_hand = np.asarray(action_hand_20[t, :20], dtype=np.float32).reshape(-1)
                            sim_cached_hand_right = np.asarray(action_hand_20[t, 20:], dtype=np.float32).reshape(-1)
                        else:
                            sim_cached_hand = np.asarray(action_hand_20[t], dtype=np.float32).reshape(-1)
                    
                    frame_hand = sim_hand_viz.step(sim_cached_hand)
                    frame_hand = add_label(frame_hand, "L-Hand" if use_both_hands else f"{hand_side[0].upper()}-Hand")
                    
                    if use_both_hands and sim_hand_viz_right is not None:
                        frame_hand_right = sim_hand_viz_right.step(sim_cached_hand_right)
                        frame_hand_right = add_label(frame_hand_right, "R-Hand")
                        frame = np.concatenate([frame, frame_hand, frame_hand_right], axis=1)
                    else:
                        frame = np.concatenate([frame, frame_hand], axis=1)
                        
                if sim_stream:
                    cv2.imshow("Sim (Body)", frame[:, :, ::-1])  # RGB->BGR
                    cv2.waitKey(1)
                if sim_writer is not None:
                    sim_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            # Only advance replay time when actually executing the episode timeline
            if advance:
                t += 1

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        kb.stop()
        if vision is not None:
            vision.close()
        try:
            import cv2
            if sim_writer is not None:
                sim_writer.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"Replay complete!")


# =============================================================================
# Offline Eval Mode (Offline evaluation with sim visualization)
# =============================================================================

def eval_offline(
    ckpt_dir: str,
    dataset_path: str,
    episode_id: Optional[int] = None,
    config: Optional[Config] = None,
    policy_config: Optional[dict] = None,
    output_video: Optional[str] = None,
    save_actions: bool = False,
    hand_side: str = "left",
):
    """
    Evaluate policy using observations from dataset, visualize in sim.

    Args:
        ckpt_dir: Path to checkpoint directory
        dataset_path: Path to HDF5 dataset
        episode_id: Episode to evaluate (None = random)
        config: Inference config
        policy_config: Policy architecture config (must match training)
        output_video: Output video path
        save_actions: If True, save predicted actions to .npy file
    """
    config = config or Config()

    # Load dataset
    reader = DatasetReader(dataset_path)
    if episode_id is None:
        episode_id = reader.random_episode_id()

    print(f"\n{'='*60}")
    print(f"Offline Evaluation - Episode {episode_id} (hand: {hand_side})")
    print(f"state_body_31d: {config.state_body_31d} (state_dim: {config.state_dim})")
    print(f"{'='*60}")

    data = reader.load_episode(episode_id, load_observations=True, hand_side=hand_side, state_body_31d=config.state_body_31d)
    qpos_all = data['qpos']          # (T, state_dim) - 51 if state_body_31d else 54
    images_all = data['images']      # (T, H, W, 3)
    gt_action_body = data['action_body']  # (T, 35) - for comparison
    gt_action_hand = data.get('action_hand', None)  # (T, 20) - for comparison/visualization
    T = data['num_timesteps']

    try:
        text = json.loads(data['text'])
        print(f"Goal: {text.get('goal', 'N/A')}")
    except:
        pass

    print(f"Timesteps: {T}")
    print(f"Observations: qpos={qpos_all.shape} (state_body_31d={config.state_body_31d}), images={images_all.shape}")

    # Default output path: save under ckpt_dir
    if output_video is None:
        output_video = os.path.join(ckpt_dir, f"eval_ep{episode_id}.mp4")

    # Load policy
    policy = ACTPolicyWrapper(ckpt_dir, config, policy_config)
    policy.reset()

    # Run inference on each timestep
    print(f"\nRunning policy inference...")
    predicted_actions_body = []
    predicted_actions_hand = []
    predicted_actions_full = []

    for t in range(T):
        qpos = qpos_all[t]      # (state_dim,)
        image = images_all[t]   # (H, W, 3)

        action = policy(qpos, image)  # (55,) or (75,) depending on hand_side
        action_body = action[:35]
        action_hand = action[35:]  # (20,) for single hand or (40,) for both hands
        predicted_actions_full.append(action)
        predicted_actions_body.append(action_body)
        predicted_actions_hand.append(action_hand)

        if (t + 1) % 50 == 0 or t == T - 1:
            print(f"  [{t+1}/{T}] pred_z={action_body[2]:.3f}, gt_z={gt_action_body[t, 2]:.3f}")

    predicted_actions_full = np.array(predicted_actions_full)  # (T, 55) or (T, 75)
    predicted_actions_body = np.array(predicted_actions_body)  # (T, 35)
    predicted_actions_hand = np.array(predicted_actions_hand)  # (T, 20) or (T, 40)
    print(f"\nPredicted actions shape: full={predicted_actions_full.shape}, body={predicted_actions_body.shape}, hand={predicted_actions_hand.shape}")

    # Compute error metrics
    mse_body = np.mean((predicted_actions_body - gt_action_body) ** 2)
    mae_body = np.mean(np.abs(predicted_actions_body - gt_action_body))
    print(f"Body  MSE: {mse_body:.6f}, MAE: {mae_body:.6f}")
    if gt_action_hand is not None and gt_action_hand.shape[1] == 20:
        mse_hand = np.mean((predicted_actions_hand - gt_action_hand) ** 2)
        mae_hand = np.mean(np.abs(predicted_actions_hand - gt_action_hand))
        print(f"Hand  MSE: {mse_hand:.6f}, MAE: {mae_hand:.6f}")

    # Save predicted actions
    if save_actions:
        # Backward compatible: body-only arrays
        actions_path = output_video.replace('.mp4', '_actions.npy')
        np.save(actions_path, predicted_actions_body)
        print(f"Saved predicted BODY actions to {actions_path}")

        gt_path = output_video.replace('.mp4', '_gt_actions.npy')
        np.save(gt_path, gt_action_body)
        print(f"Saved GT BODY actions to {gt_path}")

        # New: full + hand arrays
        actions_full_path = output_video.replace('.mp4', '_actions_full.npy')
        np.save(actions_full_path, predicted_actions_full)
        print(f"Saved predicted FULL actions to {actions_full_path}")

        hand_pred_path = output_video.replace('.mp4', f'_actions_hand_{hand_side}.npy')
        np.save(hand_pred_path, predicted_actions_hand)
        print(f"Saved predicted HAND actions to {hand_pred_path}")

        if gt_action_hand is not None:
            hand_gt_path = output_video.replace('.mp4', f'_gt_actions_hand_{hand_side}.npy')
            np.save(hand_gt_path, gt_action_hand)
            print(f"Saved GT HAND actions to {hand_gt_path}")

    # Visualize in sim
    print(f"\nGenerating sim visualization...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act', 'sim_viz'))
        import cv2
        from visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths, save_video

        paths = get_default_paths()
        viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])

        # Helper to add label on every frame
        def add_label(frame, label):
            frame = frame.copy()
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            return frame

        # Visualize predicted/GT BODY actions
        print("Visualizing BODY predicted actions...")
        pred_body_frames = viz.visualize(predicted_actions_body, warmup_steps=100, verbose=False)

        print("Visualizing BODY GT actions...")
        viz._reset()
        gt_body_frames = viz.visualize(gt_action_body, warmup_steps=100, verbose=False)

        # Visualize predicted/GT HAND actions
        if hand_side == "both":
            # Split predicted actions into left (20D) and right (20D)
            predicted_actions_hand_left = predicted_actions_hand[:, :20]   # (T, 20)
            predicted_actions_hand_right = predicted_actions_hand[:, 20:]  # (T, 20)
            
            # Split GT actions into left and right
            if gt_action_hand is not None and gt_action_hand.shape[1] == 40:
                gt_hand_left = gt_action_hand[:, :20]
                gt_hand_right = gt_action_hand[:, 20:]
            else:
                gt_hand_left = np.zeros((T, 20), dtype=np.float32)
                gt_hand_right = np.zeros((T, 20), dtype=np.float32)
            
            # Create left hand visualizer
            hand_xml_key_left = 'left_hand_xml'
            if hand_xml_key_left not in paths:
                raise ValueError(f"Hand XML path for left not found in paths. Available: {list(paths.keys())}")
            hand_viz_left = HandVisualizer(paths[hand_xml_key_left], hand_side='left')
            
            print(f"Visualizing LEFT HAND predicted actions...")
            pred_hand_left_frames = hand_viz_left.visualize(predicted_actions_hand_left, warmup_steps=50, verbose=False)
            
            print(f"Visualizing LEFT HAND GT actions...")
            hand_viz_left._reset()
            gt_hand_left_frames = hand_viz_left.visualize(gt_hand_left, warmup_steps=50, verbose=False)
            
            # Create right hand visualizer
            hand_xml_key_right = 'right_hand_xml'
            if hand_xml_key_right not in paths:
                raise ValueError(f"Hand XML path for right not found in paths. Available: {list(paths.keys())}")
            hand_viz_right = HandVisualizer(paths[hand_xml_key_right], hand_side='right')
            
            print(f"Visualizing RIGHT HAND predicted actions...")
            pred_hand_right_frames = hand_viz_right.visualize(predicted_actions_hand_right, warmup_steps=50, verbose=False)
            
            print(f"Visualizing RIGHT HAND GT actions...")
            hand_viz_right._reset()
            gt_hand_right_frames = hand_viz_right.visualize(gt_hand_right, warmup_steps=50, verbose=False)
            
            # Compose comparison video with both hands:
            #   [Body GT | Body Pred | L-Hand GT | L-Hand Pred | R-Hand GT | R-Hand Pred]
            print("Creating labeled comparison video with both hands...")
            comparison_frames = []
            h = pred_body_frames[0].shape[0]
            w = pred_body_frames[0].shape[1]

            for (bpred, bgt, hlpred, hlgt, hrpred, hrgt) in zip(
                pred_body_frames, gt_body_frames, 
                pred_hand_left_frames, gt_hand_left_frames,
                pred_hand_right_frames, gt_hand_right_frames
            ):
                # Resize hand frames to match body pane size for clean tiling
                hlpred_r = cv2.resize(hlpred, (w, h))
                hlgt_r = cv2.resize(hlgt, (w, h))
                hrpred_r = cv2.resize(hrpred, (w, h))
                hrgt_r = cv2.resize(hrgt, (w, h))

                # Create a 2-row grid: top row has body, bottom row has hands
                top_row = np.concatenate([
                    add_label(bgt, "Body GT"), 
                    add_label(bpred, "Body Pred"),
                    add_label(hlgt_r, "L-Hand GT")
                ], axis=1)
                bottom_row = np.concatenate([
                    add_label(hlpred_r, "L-Hand Pred"),
                    add_label(hrgt_r, "R-Hand GT"),
                    add_label(hrpred_r, "R-Hand Pred")
                ], axis=1)
                grid = np.concatenate([top_row, bottom_row], axis=0)
                comparison_frames.append(grid)
        else:
            # Single hand visualization (original logic)
            hand_xml_key = f'{hand_side}_hand_xml'
            if hand_xml_key not in paths:
                raise ValueError(f"Hand XML path for {hand_side} not found in paths. Available: {list(paths.keys())}")
            hand_viz = HandVisualizer(paths[hand_xml_key], hand_side=hand_side)
            print(f"Visualizing HAND predicted actions ({hand_side})...")
            pred_hand_frames = hand_viz.visualize(predicted_actions_hand, warmup_steps=50, verbose=False)

            print(f"Visualizing HAND GT actions ({hand_side})...")
            hand_viz._reset()
            # If dataset doesn't contain hand GT, this will be zeros (see DatasetReader)
            gt_hand = gt_action_hand if gt_action_hand is not None else np.zeros((T, 20), dtype=np.float32)
            gt_hand_frames = hand_viz.visualize(gt_hand, warmup_steps=50, verbose=False)

            # Compose comparison video as a 2x2 grid with labels:
            #   [Body GT | Body Pred]
            #   [Hand GT | Hand Pred]
            print("Creating labeled 2x2 comparison video...")
            comparison_frames = []
            h = pred_body_frames[0].shape[0]
            w = pred_body_frames[0].shape[1]

            for (bpred, bgt, hpred, hgt) in zip(pred_body_frames, gt_body_frames, pred_hand_frames, gt_hand_frames):
                # Resize hand frames to match body pane size for clean tiling
                hpred_r = cv2.resize(hpred, (w, h))
                hgt_r = cv2.resize(hgt, (w, h))

                top = np.concatenate([add_label(bgt, "Body GT"), add_label(bpred, "Body Pred")], axis=1)
                hand_label = hand_side.upper()[0]  # "L" or "R"
                bottom = np.concatenate([add_label(hgt_r, f"Hand GT ({hand_label})"), add_label(hpred_r, f"Hand Pred ({hand_label})")], axis=1)
                grid = np.concatenate([top, bottom], axis=0)
                comparison_frames.append(grid)

        save_video(comparison_frames, output_video, fps=30)
        print(f"Saved comparison video to {output_video}")

    except Exception as e:
        print(f"Sim visualization failed: {e}")
        print("You can visualize the saved .npy files manually using sim_viz/visualize_body_actions.py")

    print(f"\nEvaluation complete!")


# =============================================================================
# Inference Mode (Real-time)
# =============================================================================

def run_inference(
    ckpt_dir: str,
    config: Optional[Config] = None,
    policy_config: Optional[dict] = None,
    max_timesteps: int = 500,
    vision_ip: str = "192.168.123.164",
    vision_port: int = 5555,
    rgb_stream: bool = False,
    sim_stream: bool = False,
    sim_save_vid: Optional[str] = None,
    sim_hand: bool = False,
    keyboard_toggle_send: bool = False,
    toggle_send_key: str = "k",
    hold_position_key: str = "p",
    obs_source: str = "real",
    dataset_path: Optional[str] = None,
    episode_id: Optional[int] = None,
    start_timestep: int = 0,
    record_run: bool = False,
    record_images: bool = False,
    save_rgb_video: bool = False,
    rgb_video_path: Optional[str] = None,
    rgb_video_fps: float = 0.0,
    hand_side: str = "left",
    toggle_ramp_seconds: float = 0.0,
    toggle_ramp_ease: str = "cosine",
):
    """
    Run policy inference with real-time observations.

    Args:
        ckpt_dir: Path to checkpoint directory (contains policy_best.ckpt and dataset_stats.pkl)
        config: Configuration
        policy_config: Policy architecture config (must match training)
        max_timesteps: Maximum number of inference steps
        vision_ip: ZMQ vision server IP
        vision_port: ZMQ vision server port
    """
    config = config or Config()

    print(f"\n{'='*60}")
    print(f"Policy Inference")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Temporal aggregation: {config.temporal_agg}")
    print(f"Frequency: {config.frequency} Hz")
    print(f"Max timesteps: {max_timesteps}")
    print(f"state_body_31d: {config.state_body_31d} (state_dim: {config.state_dim})")

    # Initialize components
    redis_io = RedisIO(config)
    vision = VisionReader(server_ip=vision_ip, port=vision_port)
    policy = ACTPolicyWrapper(ckpt_dir, config, policy_config)

    dt = 1.0 / config.frequency

    # Safe idle action + cached last action (xrobot semantics)
    safe_idle_body_35 = _get_safe_idle_body_35(config.robot_key)
    cached_body = safe_idle_body_35.copy()
    cached_neck = np.array([0.0, 0.0], dtype=np.float32)
    last_pub_body = cached_body.copy()
    last_pub_neck = cached_neck.copy()
    ramp = _ToggleRamp()

    kb = KeyboardToggle(
        enabled=keyboard_toggle_send,
        toggle_send_key=toggle_send_key,
        hold_position_key=hold_position_key,
    )
    kb.start()
    last_send_enabled, last_hold_enabled = kb.get()

    # -----------------------------------------------------------------------------
    # Optional: record this inference run (qpos + policy actions + published actions)
    # -----------------------------------------------------------------------------
    class _InferRecorder:
        def __init__(self, run_dir: str, max_steps: int, state_dim: int = 54):
            self.run_dir = run_dir
            os.makedirs(self.run_dir, exist_ok=True)

            self.max_steps = int(max_steps)
            self.state_dim = int(state_dim)
            self.i = 0

            # Allocate arrays
            self.ts_ms = np.zeros((self.max_steps,), dtype=np.int64)
            self.send_enabled = np.zeros((self.max_steps,), dtype=np.int8)
            self.hold_enabled = np.zeros((self.max_steps,), dtype=np.int8)
            self.obs_source_idx = np.full((self.max_steps,), -1, dtype=np.int32)

            self.qpos = np.full((self.max_steps, self.state_dim), np.nan, dtype=np.float32)
            self.policy_action_55 = np.full((self.max_steps, 55), np.nan, dtype=np.float32)
            self.pub_body_35 = np.full((self.max_steps, 35), np.nan, dtype=np.float32)
            self.pub_hand_left_20 = np.full((self.max_steps, 20), np.nan, dtype=np.float32)

        def append(
            self,
            ts_ms: int,
            send_enabled: bool,
            hold_enabled: bool,
            qpos: Optional[np.ndarray],
            policy_action_55: Optional[np.ndarray],
            pub_body_35: np.ndarray,
            pub_hand_left_20: Optional[np.ndarray],
            obs_source_idx: int = -1,
        ):
            if self.i >= self.max_steps:
                return
            self.ts_ms[self.i] = int(ts_ms)
            self.send_enabled[self.i] = 1 if send_enabled else 0
            self.hold_enabled[self.i] = 1 if hold_enabled else 0
            self.obs_source_idx[self.i] = int(obs_source_idx)

            if qpos is not None:
                self.qpos[self.i] = np.asarray(qpos, dtype=np.float32).reshape(self.state_dim)
            if policy_action_55 is not None:
                self.policy_action_55[self.i] = np.asarray(policy_action_55, dtype=np.float32).reshape(55)

            self.pub_body_35[self.i] = np.asarray(pub_body_35, dtype=np.float32).reshape(35)
            if pub_hand_left_20 is not None:
                self.pub_hand_left_20[self.i] = np.asarray(pub_hand_left_20, dtype=np.float32).reshape(20)

            self.i += 1

        def maybe_save_image(self, img_rgb: np.ndarray):
            if not record_images:
                return
            try:
                import cv2
                img_bgr = img_rgb[:, :, ::-1]
                cv2.imwrite(os.path.join(self.run_dir, f"rgb_{self.i:06d}.jpg"), img_bgr)
            except Exception:
                pass

        def close(self, meta: dict):
            # Truncate to actual length and save
            n = self.i
            np.save(os.path.join(self.run_dir, "ts_ms.npy"), self.ts_ms[:n])
            np.save(os.path.join(self.run_dir, "send_enabled.npy"), self.send_enabled[:n])
            np.save(os.path.join(self.run_dir, "hold_enabled.npy"), self.hold_enabled[:n])
            np.save(os.path.join(self.run_dir, "obs_source_idx.npy"), self.obs_source_idx[:n])
            # Save qpos with state_dim suffix for clarity
            np.save(os.path.join(self.run_dir, f"qpos_{self.state_dim}.npy"), self.qpos[:n])
            np.save(os.path.join(self.run_dir, "policy_action_55.npy"), self.policy_action_55[:n])
            np.save(os.path.join(self.run_dir, "pub_body_35.npy"), self.pub_body_35[:n])
            np.save(os.path.join(self.run_dir, "pub_hand_left_20.npy"), self.pub_hand_left_20[:n])
            with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

    recorder = None
    run_dir = None
    if record_run:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(ckpt_dir, "eval", f"this_run_{ts}")
        print(f"[Record] Enabled. Saving to: {run_dir}")
        recorder = _InferRecorder(run_dir, max_steps=max_timesteps, state_dim=config.state_dim)

    # -----------------------------------------------------------------------------
    # Optional: save RealSense RGB stream to mp4 under ckpt_dir
    # -----------------------------------------------------------------------------
    rgb_writer = None
    rgb_save_path = None
    if save_rgb_video:
        try:
            import cv2
            ts = time.strftime("%Y%m%d_%H%M%S")
            # default: save under ckpt_dir/eval/
            out_dir = os.path.join(ckpt_dir, "eval")
            os.makedirs(out_dir, exist_ok=True)
            rgb_save_path = rgb_video_path
            if not rgb_save_path:
                rgb_save_path = os.path.join(out_dir, f"infer_rgb_{ts}.mp4")
            fps = float(rgb_video_fps) if float(rgb_video_fps) > 1e-6 else float(config.frequency)
            # VisionReader default is 480x640 RGB
            h, w = int(vision.img_shape[0]), int(vision.img_shape[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            rgb_writer = cv2.VideoWriter(rgb_save_path, fourcc, fps, (w, h))
            if not rgb_writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {rgb_save_path}")
            print(f"[RGB] Recording enabled: {rgb_save_path} (fps={fps:.2f}, size={w}x{h})")
        except Exception as e:
            print(f"[RGB] Failed to start video recording: {e}")
            rgb_writer = None
            rgb_save_path = None

    # Prepare observation source first (so we can see progress even if GUI init hangs)
    dataset_data = None
    dataset_T = None
    dataset_idx = None
    if obs_source == "dataset":
        if dataset_path is None:
            raise ValueError("--obs_source dataset requires --dataset")
        reader = DatasetReader(dataset_path)
        if episode_id is None:
            episode_id = reader.random_episode_id()
        dataset_data = reader.load_episode(episode_id, load_observations=True, hand_side=hand_side, state_body_31d=config.state_body_31d)
        dataset_T = int(dataset_data["num_timesteps"])
        dataset_idx = max(0, min(int(start_timestep), dataset_T - 1))
        print(f"\nUsing DATASET observations for inference:")
        print(f"  dataset={dataset_path}")
        print(f"  episode={episode_id}, T={dataset_T}, start_timestep={dataset_idx}")
        print(f"  state_body_31d={config.state_body_31d}, qpos_shape={dataset_data['qpos'].shape}")
    else:
        print(f"\nWaiting for state data... (state_body_31d={config.state_body_31d}, expected state_dim={config.state_dim})")
        # Wait for initial state
        while True:
            qpos = redis_io.read_state()
            if qpos is not None:
                print(f"Got initial state: shape={qpos.shape}")
                break
            time.sleep(0.1)

    # Optional windows
    # If running headless (no DISPLAY/WAYLAND), disable pop-up windows to avoid blocking.
    if (rgb_stream or sim_stream) and (os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None):
        print("[GUI] No DISPLAY/WAYLAND_DISPLAY detected; disabling --rgb_stream/--sim_stream (use --sim_save_vid instead).")
        rgb_stream = False
        sim_stream = False

    if rgb_stream:
        try:
            print("[RGB] Opening Robot RGB window...")
            import cv2
            cv2.namedWindow("Robot RGB", cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[RGB] Failed to open RGB window: {e}")
            rgb_stream = False

    sim_viz = None
    sim_writer = None
    hand_viz = None
    hand_viz_right = None
    cached_hand_20 = np.zeros(20, dtype=np.float32)
    cached_hand_20_right = np.zeros(20, dtype=np.float32)
    use_both_hands = (hand_side == "both")
    
    if sim_stream or sim_save_vid:
        try:
            print("[Sim] Initializing sim preview...")
            import cv2
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act', 'sim_viz'))
            from visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths
            paths = get_default_paths()
            sim_viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
            if sim_hand:
                if use_both_hands:
                    # Initialize both hand visualizers
                    hand_xml_key_left = 'left_hand_xml'
                    hand_xml_key_right = 'right_hand_xml'
                    if hand_xml_key_left not in paths or hand_xml_key_right not in paths:
                        raise ValueError(f"Hand XML paths not found in paths. Available: {list(paths.keys())}")
                    hand_viz = HandVisualizer(paths[hand_xml_key_left], hand_side='left')
                    hand_viz_right = HandVisualizer(paths[hand_xml_key_right], hand_side='right')
                else:
                    # Single hand
                    hand_xml_key = f'{hand_side}_hand_xml'
                    if hand_xml_key not in paths:
                        raise ValueError(f"Hand XML path for {hand_side} not found in paths. Available: {list(paths.keys())}")
                    hand_viz = HandVisualizer(paths[hand_xml_key], hand_side=hand_side)
            if sim_stream:
                print("[Sim] Opening Sim (Body) window...")
                cv2.namedWindow("Sim (Body)", cv2.WINDOW_NORMAL)
            if sim_save_vid:
                if sim_save_vid == "__AUTO__":
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    sim_save_vid = os.path.join(ckpt_dir, f"infer_sim_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if use_both_hands and sim_hand:
                    out_w, out_h = (1920, 480)  # body + left hand + right hand
                else:
                    out_w, out_h = (1280, 480) if sim_hand else (640, 480)
                sim_writer = cv2.VideoWriter(sim_save_vid, fourcc, 30, (out_w, out_h))
                if not sim_writer.isOpened():
                    raise RuntimeError(f"VideoWriter failed to open: {sim_save_vid}")
                print(f"[Sim] Recording enabled: {sim_save_vid}")
        except Exception as e:
            print(f"[Sim] Failed to start sim visualization: {e}")
            sim_viz = None
            sim_writer = None
            hand_viz = None
            hand_viz_right = None

    print(f"\nRunning inference... (Ctrl+C to stop)\n")

    try:
        for t in range(max_timesteps):
            t0 = time.time()

            # Always update RGB window from live vision stream
            live_image = vision.get_image()
            if rgb_stream:
                import cv2
                cv2.imshow("Robot RGB", live_image[:, :, ::-1])
                cv2.waitKey(1)
            if rgb_writer is not None:
                try:
                    import cv2
                    # VideoWriter expects BGR
                    rgb_writer.write(cv2.cvtColor(live_image, cv2.COLOR_RGB2BGR))
                except Exception:
                    pass

            send_enabled, hold_enabled = kb.get()

            # Detect k/p transitions and ramp like init_pose (freeze policy/time during ramp)
            if (send_enabled != last_send_enabled) or (hold_enabled != last_hold_enabled):
                # Determine target for NEW mode
                if not send_enabled:
                    target_body = safe_idle_body_35
                    target_neck = cached_neck
                    target_mode = "default"
                    # Also reset any policy temporal state when we go to idle (safer)
                    try:
                        policy.reset()
                    except Exception:
                        pass
                elif hold_enabled:
                    target_body = cached_body
                    target_neck = cached_neck
                    target_mode = "hold"
                else:
                    # Follow mode target: compute one policy step as the landing target
                    if obs_source == "dataset":
                        qpos_in = dataset_data["qpos"][dataset_idx]
                        img_in = dataset_data["images"][dataset_idx]
                        _a = policy(qpos_in, img_in)
                        target_body = _a[:35]
                        target_neck = cached_neck
                    else:
                        qpos_now = redis_io.read_state()
                        if qpos_now is None:
                            target_body = cached_body
                            target_neck = cached_neck
                        else:
                            _a = policy(qpos_now, live_image)
                            target_body = _a[:35]
                            target_neck = cached_neck
                    target_mode = "follow"

                ramp.start(
                    from_body=last_pub_body,
                    from_neck=last_pub_neck,
                    to_body=target_body,
                    to_neck=target_neck,
                    target_mode=target_mode,
                    seconds=float(toggle_ramp_seconds),
                    ease=str(toggle_ramp_ease),
                )
                last_send_enabled, last_hold_enabled = send_enabled, hold_enabled

            # If ramping, publish interpolated action and skip policy/time advance
            if ramp.active:
                pub_body, pub_neck, _done = ramp.value()
                try:
                    redis_io.set_wuji_hand_mode(ramp.target_mode)
                except Exception:
                    pass
                redis_io.publish_action(pub_body, pub_neck)
                last_pub_body = np.asarray(pub_body, dtype=np.float32).reshape(35).copy()
                last_pub_neck = np.asarray(pub_neck, dtype=np.float32).reshape(2).copy()

                # keep UI/recording alive but don't run policy / don't advance dataset_idx
                if recorder is not None:
                    now = int(time.time() * 1000)
                    recorder.append(
                        ts_ms=now,
                        send_enabled=send_enabled,
                        hold_enabled=hold_enabled,
                        qpos=None,
                        policy_action_55=None,
                        pub_body_35=pub_body,
                        pub_hand_left_20=cached_hand_20,
                        obs_source_idx=-1,
                    )
                    recorder.maybe_save_image(live_image)

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            # Only compute desired action if we are not in k/p override states
            desired_body = cached_body
            desired_neck = cached_neck
            desired_hand_20 = None
            policy_action_full_55 = None
            obs_qpos = None
            obs_idx_for_record = -1
            if send_enabled and (not hold_enabled):
                if obs_source == "dataset":
                    qpos_in = dataset_data["qpos"][dataset_idx]
                    img_in = dataset_data["images"][dataset_idx]
                    action = policy(qpos_in, img_in)  # (55,)
                    desired_body = action[:35]
                    desired_hand_20 = action[35:]
                    policy_action_full_55 = action
                    obs_qpos = qpos_in
                    obs_idx_for_record = int(dataset_idx)
                else:
                    qpos = redis_io.read_state()
                    if qpos is None:
                        print(f"  [{t}] Warning: no state data")
                        time.sleep(dt)
                        continue
                    action = policy(qpos, live_image)  # (55,)
                    desired_body = action[:35]
                    desired_hand_20 = action[35:]
                    policy_action_full_55 = action
                    obs_qpos = qpos

            pub_body, pub_neck, cached_body, cached_neck, advance = _publish_with_kp_safety(
                redis_io=redis_io,
                kb=kb,
                safe_idle_body_35=safe_idle_body_35,
                cached_body=cached_body,
                cached_neck=cached_neck,
                desired_body=desired_body,
                desired_neck=desired_neck,
            )

            if obs_source == "dataset" and advance:
                dataset_idx += 1
                if dataset_idx >= dataset_T:
                    dataset_idx = dataset_T - 1

            # Publish Wuji hand 20D target when we actually advanced/ran policy
            if advance and desired_hand_20 is not None:
                if use_both_hands:
                    # Split into left (20D) and right (20D)
                    hand_left = desired_hand_20[:20]
                    hand_right = desired_hand_20[20:]
                    redis_io.publish_wuji_qpos_target(hand_left, hand_side='left')
                    redis_io.publish_wuji_qpos_target(hand_right, hand_side='right')
                    cached_hand_20 = np.asarray(hand_left, dtype=np.float32).reshape(-1)
                    cached_hand_20_right = np.asarray(hand_right, dtype=np.float32).reshape(-1)
                else:
                    redis_io.publish_wuji_qpos_target(desired_hand_20, hand_side=hand_side)
                    cached_hand_20 = np.asarray(desired_hand_20, dtype=np.float32).reshape(-1)

            # Record (always record what we published; policy_action may be NaN if in k/p override)
            if recorder is not None:
                now = int(time.time() * 1000)
                recorder.append(
                    ts_ms=now,
                    send_enabled=send_enabled,
                    hold_enabled=hold_enabled,
                    qpos=obs_qpos,
                    policy_action_55=policy_action_full_55,
                    pub_body_35=pub_body,
                    pub_hand_left_20=cached_hand_20,
                    obs_source_idx=obs_idx_for_record,
                )
                recorder.maybe_save_image(live_image)

            # Sim preview
            if sim_viz is not None:
                import cv2
                
                # Helper to add label
                def add_label(frame, label):
                    frame = frame.copy()
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    return frame
                
                body_frame = sim_viz.step(np.asarray(pub_body, dtype=np.float32))
                out_frame = add_label(body_frame, "Body")
                
                if hand_viz is not None:
                    hand_frame = hand_viz.step(cached_hand_20)
                    hand_frame = add_label(hand_frame, "L-Hand" if use_both_hands else f"{hand_side[0].upper()}-Hand")
                    
                    if use_both_hands and hand_viz_right is not None:
                        hand_frame_right = hand_viz_right.step(cached_hand_20_right)
                        hand_frame_right = add_label(hand_frame_right, "R-Hand")
                        out_frame = np.concatenate([out_frame, hand_frame, hand_frame_right], axis=1)
                    else:
                        out_frame = np.concatenate([out_frame, hand_frame], axis=1)
                        
                if sim_stream:
                    cv2.imshow("Sim (Body)", out_frame[:, :, ::-1])
                    cv2.waitKey(1)
                if sim_writer is not None:
                    sim_writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

            # Progress
            if (t + 1) % 30 == 0:
                print(f"  [{t+1}/{max_timesteps}] z={pub_body[2]:.3f}")

            # Rate limiting
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            last_pub_body = np.asarray(pub_body, dtype=np.float32).reshape(35).copy()
            last_pub_neck = np.asarray(pub_neck, dtype=np.float32).reshape(2).copy()

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        kb.stop()
        vision.close()
        try:
            import cv2
            if rgb_writer is not None:
                rgb_writer.release()
                if rgb_save_path:
                    print(f"[RGB] Saved video to {rgb_save_path}")
            if sim_writer is not None:
                sim_writer.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

        if recorder is not None and run_dir is not None:
            meta = {
                "ckpt_dir": ckpt_dir,
                "obs_source": obs_source,
                "dataset": dataset_path,
                "episode": episode_id,
                "start_timestep": start_timestep,
                "frequency": config.frequency,
                "chunk_size": config.chunk_size,
                "temporal_agg": config.temporal_agg,
                "vision_ip": vision_ip,
                "vision_port": vision_port,
                "max_timesteps": max_timesteps,
                "saved_images": bool(record_images),
                "saved_rgb_video": str(rgb_save_path) if rgb_save_path else "",
                "state_body_31d": config.state_body_31d,
                "state_dim": config.state_dim,
            }
            recorder.close(meta)
            print(f"[Record] Saved {recorder.i} steps to: {run_dir}")

    print(f"Inference complete!")


def publish_initial_action_from_dataset(
    dataset_path: str,
    episode_id: Optional[int] = None,
    timestep: int = 0,
    config: Optional[Config] = None,
    keyboard_toggle_send: bool = True,
    toggle_send_key: str = "k",
    hold_position_key: str = "p",
    hand_side: str = "left",
    ramp_seconds: float = 0.0,
    ramp_ease: str = "cosine",
    ramp_from: str = "redis_action",
):
    """Continuously publish an initial (body+neck) action from dataset to move robot to initial pose.

    Intended usage: run this first, watch robot reach initial pose, Ctrl-C to stop,
    then run your real `infer` or `replay`.
    """
    config = config or Config()
    reader = DatasetReader(dataset_path)
    if episode_id is None:
        episode_id = reader.random_episode_id()
    data = reader.load_episode(episode_id, load_observations=False, hand_side=hand_side)
    T = int(data["num_timesteps"])

    timestep = int(timestep)
    if timestep < 0:
        timestep = 0
    if timestep >= T:
        timestep = T - 1

    body = np.asarray(data["action_body"][timestep], dtype=np.float32)
    neck = np.asarray(data["action_neck"][timestep], dtype=np.float32)
    hand_20 = None
    if "action_hand" in data:
        try:
            hand_20 = np.asarray(data["action_hand"][timestep], dtype=np.float32)
        except Exception:
            hand_20 = None

    print(f"\n{'='*60}")
    print("Publish initial action (hold) from dataset")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Episode: {episode_id} (T={T})")
    print(f"Timestep: {timestep}")
    print(f"Publishing to Redis at {config.redis_ip}:{config.redis_port} robot_key={config.robot_key}")
    if float(ramp_seconds) > 0.0:
        print(f"Ramp: enabled ({float(ramp_seconds):.2f}s, ease={ramp_ease}, from={ramp_from})")
    print("Ctrl-C to stop.")

    redis_io = RedisIO(config)
    dt = 1.0 / config.frequency

    safe_idle_body_35 = _get_safe_idle_body_35(config.robot_key)
    cached_body = body.copy()
    cached_neck = neck.copy()

    # ---------------------------
    # Optional: smooth ramp-in
    # ---------------------------
    def _ease(alpha: float) -> float:
        a = float(np.clip(alpha, 0.0, 1.0))
        if ramp_ease == "linear":
            return a
        # cosine ease-in-out
        return 0.5 - 0.5 * float(np.cos(np.pi * a))

    def _read_redis_json_array(key: str, expected_len: Optional[int] = None) -> Optional[np.ndarray]:
        try:
            raw = redis_io.client.get(key)
            arr = redis_io._safe_json_load(raw)
            if arr is None:
                return None
            out = np.asarray(arr, dtype=np.float32).reshape(-1)
            if expected_len is not None and out.shape[0] != int(expected_len):
                return None
            return out
        except Exception:
            return None

    start_body = None
    start_neck = None
    start_hand = None
    if float(ramp_seconds) > 0.0:
        if ramp_from == "redis_action":
            start_body = _read_redis_json_array(config.key_action_body, expected_len=35)
            start_neck = _read_redis_json_array(config.key_action_neck, expected_len=2)
            if hand_20 is not None:
                start_hand = _read_redis_json_array(
                    f"action_wuji_qpos_target_{hand_side}_{config.robot_key}",
                    expected_len=20,
                )

        # Fallbacks
        if start_body is None:
            start_body = safe_idle_body_35.copy()
        if start_neck is None:
            start_neck = np.array([0.0, 0.0], dtype=np.float32)
        if hand_20 is not None and start_hand is None:
            start_hand = np.zeros(20, dtype=np.float32)

    ramp_t0 = time.time()
    kb = KeyboardToggle(
        enabled=keyboard_toggle_send,
        toggle_send_key=toggle_send_key,
        hold_position_key=hold_position_key,
    )
    kb.start()

    try:
        while True:
            t0 = time.time()
            desired_body = body
            desired_neck = neck
            desired_hand = hand_20

            if float(ramp_seconds) > 0.0:
                alpha = (time.time() - ramp_t0) / max(1e-6, float(ramp_seconds))
                w = _ease(alpha)
                desired_body = start_body + w * (body - start_body)
                desired_neck = start_neck + w * (neck - start_neck)
                if hand_20 is not None and start_hand is not None:
                    desired_hand = start_hand + w * (hand_20 - start_hand)

            _pub_body, _pub_neck, cached_body, cached_neck, _advance = _publish_with_kp_safety(
                redis_io=redis_io,
                kb=kb,
                safe_idle_body_35=safe_idle_body_35,
                cached_body=cached_body,
                cached_neck=cached_neck,
                desired_body=desired_body,
                desired_neck=desired_neck,
            )
            if desired_hand is not None:
                redis_io.publish_wuji_qpos_target(desired_hand, hand_side=hand_side)
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        kb.stop()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HumDex Policy Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Replay mode - replay GT actions from dataset
    replay_parser = subparsers.add_parser('replay', help='Replay GT actions from dataset')
    replay_parser.add_argument('--dataset', required=True, help='Path to HDF5 dataset')
    replay_parser.add_argument('--episode', type=int, default=None, help='Episode ID (default: random)')
    replay_parser.add_argument('--output', default='replay_gt.mp4', help='Output video path (for --sim_only)')
    replay_parser.add_argument('--redis_ip', default='localhost', help='Redis IP')
    replay_parser.add_argument('--frequency', type=float, default=30.0, help='Publish frequency (Hz)')
    replay_parser.add_argument('--sim_only', action='store_true', help='Only visualize in sim')
    replay_parser.add_argument('--vision_ip', default='192.168.123.164', help='Vision server IP (for --rgb_stream)')
    replay_parser.add_argument('--vision_port', type=int, default=5555, help='Vision server port (for --rgb_stream)')
    replay_parser.add_argument('--rgb_stream', action='store_true', help='Open a window to stream robot RGB (live)')
    replay_parser.add_argument('--sim_stream', action='store_true', help='Open a window to stream sim preview (body)')
    replay_parser.add_argument('--sim_save_vid', nargs='?', const="__AUTO__", default=None,
                              help='Save sim preview mp4 (optionally provide path)')
    replay_sim_hand_group = replay_parser.add_mutually_exclusive_group()
    replay_sim_hand_group.add_argument('--sim_hand', dest='sim_hand', action='store_true',
                                       help='(Deprecated) Include hand (20D) visualization in sim preview/video (enabled by default).')
    replay_sim_hand_group.add_argument('--no_sim_hand', dest='sim_hand', action='store_false',
                                       help='Disable hand (20D) visualization in sim preview/video.')
    replay_parser.set_defaults(sim_hand=True)
    replay_parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                              help='Which hand to use: left, right, or both (default: left)')
    replay_parser.set_defaults(keyboard_toggle_send=True)
    replay_parser.add_argument('--no_keyboard_toggle_send', dest='keyboard_toggle_send', action='store_false',
                              help="Disable terminal keyboard safety toggles (NOT recommended on real robot)")
    replay_parser.add_argument('--toggle_send_key', type=str, default='k', help="Key to toggle send_enabled (default 'k')")
    replay_parser.add_argument('--hold_position_key', type=str, default='p', help="Key to toggle hold_position (default 'p')")
    replay_parser.add_argument('--toggle_ramp_seconds', type=float, default=0.0,
                              help="k/p 切换时的平滑插值时长（秒，0 关闭）。插值曲线与 init_pose 一致。")
    replay_parser.add_argument('--toggle_ramp_ease', type=str, default='cosine', choices=['linear', 'cosine'],
                              help="k/p 插值曲线（linear/cosine，默认 cosine）")

    # Eval mode - run policy with dataset observations, visualize in sim
    eval_parser = subparsers.add_parser('eval', help='Evaluate policy with dataset observations')
    eval_parser.add_argument('--ckpt_dir', required=True, help='Checkpoint directory')
    eval_parser.add_argument('--dataset', required=True, help='Path to HDF5 dataset')
    eval_parser.add_argument('--episode', type=int, default=None, help='Episode ID (default: random)')
    eval_parser.add_argument('--output', default=None, help='Output video path (default: ckpt_dir/eval_ep{episode}.mp4)')
    eval_parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    eval_parser.add_argument('--chunk_size', type=int, default=50, help='Action chunk size (must match training)')
    eval_parser.add_argument('--hidden_dim', type=int, default=512, help='Policy hidden dim (must match training)')
    eval_parser.add_argument('--dim_feedforward', type=int, default=3200,
                             help='Transformer FFN dim (must match training, e.g. 3200)')
    eval_parser.add_argument('--save_actions', action='store_true',
                             help='If set, save predicted/GT actions to .npy alongside the output video')
    eval_parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                             help='Which hand to use: left, right, or both (default: left)')
    eval_parser.add_argument('--state_body_31d', action='store_true',
                             help='Use 31D state_body (roll/pitch + joints) instead of 34D. '
                                  'If dataset has 34D, extract [3:34]. If dataset has 31D, use as-is.')
    eval_parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')

    # Infer mode - run policy with real-time observations
    infer_parser = subparsers.add_parser('infer', help='Run policy with real-time observations')
    infer_parser.add_argument('--ckpt_dir', required=True, help='Checkpoint directory')
    infer_parser.add_argument('--redis_ip', default='localhost', help='Redis IP')
    infer_parser.add_argument('--frequency', type=float, default=30.0, help='Control frequency (Hz)')
    infer_parser.add_argument('--max_timesteps', type=int, default=500, help='Max inference steps')
    infer_parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    infer_parser.add_argument('--chunk_size', type=int, default=50, help='Action chunk size (must match training)')
    infer_parser.add_argument('--hidden_dim', type=int, default=512, help='Policy hidden dim (must match training)')
    infer_parser.add_argument('--dim_feedforward', type=int, default=3200,
                              help='Transformer FFN dim (must match training, e.g. 3200)')
    infer_parser.add_argument('--vision_ip', default='192.168.123.164', help='Vision server IP')
    infer_parser.add_argument('--vision_port', type=int, default=5555, help='Vision server port')
    infer_parser.add_argument('--rgb_stream', action='store_true', help='Open a window to stream robot RGB')
    infer_parser.add_argument('--sim_stream', action='store_true', help='Open a window to stream sim preview (body)')
    infer_parser.add_argument('--sim_save_vid', nargs='?', const="__AUTO__", default=None,
                              help='Save sim preview mp4 (optionally provide path)')
    infer_parser.add_argument('--sim_hand', action='store_true', help='Include hand (20D) visualization in sim preview/video')
    infer_parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                              help='Which hand to use: left, right, or both (default: left)')
    infer_parser.add_argument('--state_body_31d', action='store_true',
                              help='Use 31D state_body (roll/pitch + joints) instead of 34D. '
                                   'For real robot obs: extract [3:34] from 34D state. '
                                   'For dataset obs: auto-handle based on dataset dim.')
    infer_parser.set_defaults(keyboard_toggle_send=True)
    infer_parser.add_argument('--no_keyboard_toggle_send', dest='keyboard_toggle_send', action='store_false',
                              help="Disable terminal keyboard safety toggles (NOT recommended on real robot)")
    infer_parser.add_argument('--toggle_send_key', type=str, default='k', help="Key to toggle send_enabled (default 'k')")
    infer_parser.add_argument('--hold_position_key', type=str, default='p', help="Key to toggle hold_position (default 'p')")
    infer_parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    infer_parser.add_argument('--obs_source', choices=['real', 'dataset'], default='real',
                              help="Observation source for policy input: real (Redis+Vision) or dataset (HDF5). Action is still published to Redis.")
    infer_parser.add_argument('--dataset', default=None, help='HDF5 dataset path (required if --obs_source dataset)')
    infer_parser.add_argument('--episode', type=int, default=None, help='Dataset episode (default: random) if --obs_source dataset')
    infer_parser.add_argument('--start_timestep', type=int, default=0, help='Start timestep if --obs_source dataset')
    infer_parser.add_argument('--record_run', action='store_true',
                              help='Record qpos + policy actions + published actions under ckpt_dir/eval/this_run_<timestamp>/')
    infer_parser.add_argument('--record_images', action='store_true',
                              help='If set with --record_run, also save live RGB frames as JPGs (can be large)')
    infer_parser.add_argument('--save_rgb_video', action='store_true',
                              help='Save live RGB (RealSense/vision) stream to mp4 under ckpt_dir/eval/')
    infer_parser.add_argument('--rgb_video_path', default=None,
                              help='Optional output path for RGB mp4 (default: ckpt_dir/eval/infer_rgb_<timestamp>.mp4)')
    infer_parser.add_argument('--rgb_video_fps', type=float, default=0.0,
                              help='FPS for saved RGB video (default: use --frequency)')
    infer_parser.add_argument('--toggle_ramp_seconds', type=float, default=0.0,
                              help="k/p 切换时的平滑插值时长（秒，0 关闭）。插值曲线与 init_pose 一致。")
    infer_parser.add_argument('--toggle_ramp_ease', type=str, default='cosine', choices=['linear', 'cosine'],
                              help="k/p 插值曲线（linear/cosine，默认 cosine）")

    # Init-pose mode - publish a fixed initial action from dataset until Ctrl-C
    init_parser = subparsers.add_parser('init_pose', help='Publish an initial dataset action to move robot to initial pose (hold)')
    init_parser.add_argument('--dataset', required=True, help='Path to HDF5 dataset')
    init_parser.add_argument('--episode', type=int, default=None, help='Episode ID (default: random)')
    init_parser.add_argument('--timestep', type=int, default=0, help='Timestep index to take initial action from (default: 0)')
    init_parser.add_argument('--redis_ip', default='localhost', help='Redis IP')
    init_parser.add_argument('--frequency', type=float, default=30.0, help='Publish frequency (Hz)')
    init_parser.add_argument('--robot_key', default='unitree_g1_with_hands', help='Robot key suffix for Redis keys')
    init_parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                             help='Which hand to use: left, right, or both (default: left)')
    init_parser.add_argument('--ramp_seconds', type=float, default=0.0,
                             help='Smoothly interpolate from current action to target over this duration in seconds (0 disables).')
    init_parser.add_argument('--ramp_ease', type=str, default='cosine', choices=['linear', 'cosine'],
                             help='Ramp easing curve (default: cosine).')
    init_parser.add_argument('--ramp_from', type=str, default='redis_action', choices=['redis_action', 'safe_idle'],
                             help='Ramp start source: redis_action (read current published action_*), or safe_idle.')
    init_parser.set_defaults(keyboard_toggle_send=True)
    init_parser.add_argument('--no_keyboard_toggle_send', dest='keyboard_toggle_send', action='store_false',
                             help="Disable terminal keyboard safety toggles (NOT recommended on real robot)")
    init_parser.add_argument('--toggle_send_key', type=str, default='k', help="Key to toggle send_enabled (default 'k')")
    init_parser.add_argument('--hold_position_key', type=str, default='p', help="Key to toggle hold_position (default 'p')")

    args = parser.parse_args()

    if args.mode == 'replay':
        config = Config(redis_ip=args.redis_ip, frequency=args.frequency)
        replay_episode(
            dataset_path=args.dataset,
            episode_id=args.episode,
            config=config,
            sim_only=args.sim_only,
            output_video=args.output,
            vision_ip=args.vision_ip,
            vision_port=args.vision_port,
            rgb_stream=args.rgb_stream,
            sim_stream=args.sim_stream,
            sim_save_vid=args.sim_save_vid,
            sim_hand=args.sim_hand,
            keyboard_toggle_send=args.keyboard_toggle_send,
            toggle_send_key=args.toggle_send_key,
            hold_position_key=args.hold_position_key,
            hand_side=args.hand_side,
            toggle_ramp_seconds=float(args.toggle_ramp_seconds),
            toggle_ramp_ease=str(args.toggle_ramp_ease),
        )

    elif args.mode == 'eval':
        config = Config(
            temporal_agg=args.temporal_agg,
            chunk_size=args.chunk_size,
            use_gpu=not args.cpu,
            state_body_31d=args.state_body_31d,
        )
        # Build policy config from args
        policy_config = {
            'lr': 1e-5,
            'lr_backbone': 1e-5,
            'num_queries': args.chunk_size,
            'kl_weight': 10,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,  # Must match training
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['head'],
        }
        eval_offline(
            ckpt_dir=args.ckpt_dir,
            dataset_path=args.dataset,
            episode_id=args.episode,
            config=config,
            policy_config=policy_config,
            output_video=args.output,
            save_actions=args.save_actions,
            hand_side=args.hand_side,
        )

    elif args.mode == 'infer':
        config = Config(
            redis_ip=args.redis_ip,
            frequency=args.frequency,
            temporal_agg=args.temporal_agg,
            chunk_size=args.chunk_size,
            use_gpu=not args.cpu,
            state_body_31d=args.state_body_31d,
        )
        # Build policy config from args
        policy_config = {
            'lr': 1e-5,
            'lr_backbone': 1e-5,
            'num_queries': args.chunk_size,
            'kl_weight': 10,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,  # Must match training
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['head'],
        }
        run_inference(
            ckpt_dir=args.ckpt_dir,
            config=config,
            policy_config=policy_config,
            max_timesteps=args.max_timesteps,
            vision_ip=args.vision_ip,
            vision_port=args.vision_port,
            rgb_stream=args.rgb_stream,
            sim_stream=args.sim_stream,
            sim_save_vid=args.sim_save_vid,
            sim_hand=args.sim_hand,
            keyboard_toggle_send=args.keyboard_toggle_send,
            toggle_send_key=args.toggle_send_key,
            hold_position_key=args.hold_position_key,
            obs_source=args.obs_source,
            dataset_path=args.dataset,
            episode_id=args.episode,
            start_timestep=args.start_timestep,
            record_run=args.record_run,
            record_images=args.record_images,
            save_rgb_video=bool(args.save_rgb_video),
            rgb_video_path=args.rgb_video_path,
            rgb_video_fps=float(args.rgb_video_fps),
            hand_side=args.hand_side,
            toggle_ramp_seconds=float(args.toggle_ramp_seconds),
            toggle_ramp_ease=str(args.toggle_ramp_ease),
        )

    elif args.mode == 'init_pose':
        config = Config(
            redis_ip=args.redis_ip,
            frequency=args.frequency,
            robot_key=args.robot_key,
        )
        publish_initial_action_from_dataset(
            dataset_path=args.dataset,
            episode_id=args.episode,
            timestep=args.timestep,
            config=config,
            keyboard_toggle_send=args.keyboard_toggle_send,
            toggle_send_key=args.toggle_send_key,
            hold_position_key=args.hold_position_key,
            hand_side=args.hand_side,
            ramp_seconds=args.ramp_seconds,
            ramp_ease=args.ramp_ease,
            ramp_from=args.ramp_from,
        )


if __name__ == "__main__":
    main()
