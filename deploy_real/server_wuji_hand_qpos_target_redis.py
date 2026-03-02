#!/usr/bin/env python3
"""
Wuji Hand Controller via Redis (DIRECT 20D qpos target)

IMPORTANT:
- This is a copy-style variant of `server_wuji_hand_redis.py`.
- We DO NOT modify the original file.

Behavior:
- Reads `wuji_hand_mode_{left/right}_{robot_key}`: follow | hold | default
- In follow mode: reads `action_wuji_qpos_target_{hand_side}_{robot_key}` (20D flattened list)
  and sends it directly to Wuji hand controller.
- In hold mode: repeats last_qpos.
- In default mode: goes to zero_pose.

It also writes back:
- `state_wuji_hand_{hand_side}_{robot_key}` (20D actual qpos)
- `t_state_wuji_hand_{hand_side}_{robot_key}` (ms)
"""

import argparse
import json
import time
import numpy as np
import redis
import signal
import sys
from typing import Optional


def now_ms() -> int:
    return int(time.time() * 1000)


try:
    import wujihandpy
except ImportError:
    print("[ERROR] Error: wujihandpy is not installed. Please install it first:")
    print("   pip install wujihandpy")
    sys.exit(1)


def smooth_move(hand, controller, target_qpos, duration=0.1, steps=10):
    """Smoothly interpolate to a 5x4 target_qpos."""
    target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(5, 4)
    try:
        cur = hand.read_joint_actual_position()
    except Exception:
        cur = np.zeros((5, 4), dtype=np.float32)
    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / max(steps, 1))


class WujiHandDirectQposRedisController:
    def __init__(
        self,
        redis_ip: str = "localhost",
        hand_side: str = "left",
        robot_key: str = "unitree_g1_with_hands",
        target_fps: int = 50,
        smooth_enabled: bool = True,
        smooth_steps: int = 5,
        serial_number: str = "",
    ):
        self.hand_side = hand_side.strip().lower()
        assert self.hand_side in ["left", "right"]
        self.robot_key = robot_key.strip()

        self.target_fps = int(target_fps)
        self.control_dt = 1.0 / float(self.target_fps)
        self.smooth_enabled = bool(smooth_enabled)
        self.smooth_steps = int(smooth_steps)
        self.serial_number = (serial_number or "").strip()

        # Redis connection
        print(f"[INFO] Connecting Redis: {redis_ip}")
        self.redis_client = redis.Redis(host=redis_ip, port=6379, decode_responses=False)
        self.redis_client.ping()
        print("[OK] Redis connected")

        # Redis keys (keep naming consistent with server_wuji_hand_redis.py)
        self.redis_key_action_wuji_qpos_target = f"action_wuji_qpos_target_{self.hand_side}_{self.robot_key}"
        self.redis_key_state_wuji_hand = f"state_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_state_wuji_hand = f"t_state_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_t_action_wuji_hand = f"t_action_wuji_hand_{self.hand_side}_{self.robot_key}"
        self.redis_key_wuji_mode = f"wuji_hand_mode_{self.hand_side}_{self.robot_key}"

        # Init Wuji hand
        print(f"[INFO] Initializing Wuji {self.hand_side} hand...")
        if self.serial_number:
            print(f"[INFO] Using serial_number: {self.serial_number}")
            self.hand = wujihandpy.Hand(serial_number=self.serial_number)
        else:
            self.hand = wujihandpy.Hand()

        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=10.0),
        )
        time.sleep(0.4)

        actual_pose = self.hand.read_joint_actual_position()
        self.zero_pose = np.zeros_like(actual_pose)
        self.last_qpos = self.zero_pose.copy()

        self.running = True
        self._stop_requested_by_signal = None

    def _read_mode(self) -> str:
        try:
            mode_raw = self.redis_client.get(self.redis_key_wuji_mode)
            if isinstance(mode_raw, bytes):
                mode_raw = mode_raw.decode("utf-8")
            mode = str(mode_raw) if mode_raw is not None else "follow"
        except Exception:
            mode = "follow"
        return mode.strip().lower()

    def _read_target_20d(self) -> Optional[np.ndarray]:
        raw = self.redis_client.get(self.redis_key_action_wuji_qpos_target)
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            arr = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
            if arr.shape[0] != 20:
                return None
            return arr.reshape(5, 4)
        except Exception:
            return None

    def _write_state(self):
        try:
            actual_qpos = self.hand.read_joint_actual_position()
            self.redis_client.set(self.redis_key_state_wuji_hand, json.dumps(actual_qpos.reshape(-1).tolist()))
            self.redis_client.set(self.redis_key_t_state_wuji_hand, now_ms())
        except Exception:
            pass

    def run(self):
        print(f"\n[INFO] Direct-qpos control loop ({self.hand_side}) @ {self.target_fps} Hz")
        print("Ctrl+C to exit\n")

        def _handle_signal(signum, _frame):
            self._stop_requested_by_signal = signum
            self.running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while self.running:
                loop_start = time.time()

                mode = self._read_mode()

                if mode == "default":
                    target = self.zero_pose
                elif mode == "hold":
                    target = self.last_qpos
                else:
                    target = self._read_target_20d()
                    if target is None:
                        # If no target yet, keep last (safer than jittering)
                        target = self.last_qpos

                # For debug: write timestamp for action
                try:
                    self.redis_client.set(self.redis_key_action_wuji_qpos_target, json.dumps(np.asarray(target).reshape(-1).tolist()))
                    self.redis_client.set(self.redis_key_t_action_wuji_hand, now_ms())
                except Exception:
                    pass

                if self.smooth_enabled:
                    smooth_move(self.hand, self.controller, target, duration=self.control_dt, steps=self.smooth_steps)
                else:
                    self.controller.set_joint_target_position(np.asarray(target, dtype=np.float32).reshape(5, 4))

                self.last_qpos = np.asarray(target, dtype=np.float32).reshape(5, 4).copy()
                self._write_state()

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, self.control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n[STOP] Shutting down Wuji controller...")
        try:
            # Move to zero pose
            if self._stop_requested_by_signal == signal.SIGTERM:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=0.2, steps=10)
            else:
                smooth_move(self.hand, self.controller, self.zero_pose, duration=1.0, steps=50)
        except Exception:
            pass
        try:
            self.controller.close()
            self.hand.write_joint_enabled(False)
        except Exception:
            pass
        print("[OK] Done.")


def parse_arguments():
    ap = argparse.ArgumentParser(description="Wuji hand direct-qpos controller via Redis")
    ap.add_argument("--redis_ip", type=str, default="localhost")
    ap.add_argument("--hand_side", type=str, choices=["left", "right"], default="left")
    ap.add_argument("--robot_key", type=str, default="unitree_g1_with_hands")
    ap.add_argument("--target_fps", type=int, default=50)
    ap.add_argument("--no_smooth", action="store_true", help="Disable smoothing")
    ap.add_argument("--smooth_steps", type=int, default=5)
    ap.add_argument("--serial_number", type=str, default="", help="Optional Wuji device serial number")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    ctrl = WujiHandDirectQposRedisController(
        redis_ip=args.redis_ip,
        hand_side=args.hand_side,
        robot_key=args.robot_key,
        target_fps=args.target_fps,
        smooth_enabled=(not args.no_smooth),
        smooth_steps=args.smooth_steps,
        serial_number=args.serial_number,
    )
    ctrl.run()


