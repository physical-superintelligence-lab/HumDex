"""Redis I/O utilities for HumDex robot state reading and action publishing."""

import json
import time
from typing import Optional, Any

import numpy as np


class RedisIO:
    """Read state from and publish actions to Redis."""

    def __init__(self, config):
        """
        Args:
            config: Object with attributes: redis_ip, redis_port, robot_key,
                    hand_side, key_state_body, key_state_hand_left,
                    key_state_hand_right, key_action_body, key_action_neck,
                    key_action_hand_left, key_action_hand_right, key_t_action.
        """
        import redis
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
        import redis
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
                  - Single hand (left/right): state_body + state_hand (51D)
                  - Both hands: state_body + state_hand_left + state_hand_right (71D)
        """
        try:
            if self.config.hand_side == "both":
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
                hand_key = self.config.key_state_hand_left if self.config.hand_side == "left" else self.config.key_state_hand_right

                self.pipeline.get(self.config.key_state_body)
                self.pipeline.get(hand_key)
                results = self.pipeline.execute()

                state_body = self._safe_json_load(results[0])
                state_hand = self._safe_json_load(results[1])

                if state_body is None:
                    return None

                state_body = np.array(state_body, dtype=np.float32)

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
            action_hand_left: (7,) optional (defaults to zeros)
            action_hand_right: (7,) optional (defaults to zeros)
        """
        self.pipeline.set(self.config.key_action_body, json.dumps(action_body.tolist()))

        if action_neck is not None:
            self.pipeline.set(self.config.key_action_neck, json.dumps(action_neck.tolist()))
        else:
            self.pipeline.set(self.config.key_action_neck, json.dumps([0.0, 0.0]))

        if action_hand_left is None:
            action_hand_left = np.zeros(7, dtype=np.float32)
        if action_hand_right is None:
            action_hand_right = np.zeros(7, dtype=np.float32)
        self.pipeline.set(self.config.key_action_hand_left, json.dumps(np.asarray(action_hand_left, dtype=float).tolist()))
        self.pipeline.set(self.config.key_action_hand_right, json.dumps(np.asarray(action_hand_right, dtype=float).tolist()))

        self.pipeline.set(self.config.key_t_action, int(time.time() * 1000))
        self.pipeline.execute()

    def set_wuji_hand_mode(self, mode: str):
        """Set Wuji hand mode keys (best-effort).

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
        """Publish 20D Wuji left-hand joint target to Redis (best-effort)."""
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
        """Publish 20D Wuji right-hand joint target to Redis (best-effort)."""
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
