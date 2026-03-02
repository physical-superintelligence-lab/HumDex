from __future__ import annotations

import json
from typing import Any, Dict, Optional


def publish_twist2_step(
    *,
    redis_client: Any,
    dry_run: bool,
    body_35: list[float],
    hand_tracking_left: Dict[str, Any],
    hand_tracking_right: Dict[str, Any],
    wuji_mode_left: str,
    wuji_mode_right: str,
    now_ms: int,
    bvh_left: Optional[Dict[str, Any]] = None,
    bvh_right: Optional[Dict[str, Any]] = None,
) -> None:
    robot_key = "unitree_g1_with_hands"
    key_action_body = f"action_body_{robot_key}"
    key_t_action = "t_action"
    key_ht_l = f"hand_tracking_left_{robot_key}"
    key_ht_r = f"hand_tracking_right_{robot_key}"
    key_bvh_l = f"hand_bvh_left_{robot_key}"
    key_bvh_r = f"hand_bvh_right_{robot_key}"
    key_wuji_mode_l = f"wuji_hand_mode_left_{robot_key}"
    key_wuji_mode_r = f"wuji_hand_mode_right_{robot_key}"

    pipe = redis_client.pipeline()
    pipe.set(key_action_body, json.dumps(body_35))
    pipe.set(key_ht_l, json.dumps(hand_tracking_left))
    pipe.set(key_ht_r, json.dumps(hand_tracking_right))
    if bvh_left is not None:
        pipe.set(key_bvh_l, json.dumps(bvh_left))
    if bvh_right is not None:
        pipe.set(key_bvh_r, json.dumps(bvh_right))
    pipe.set(key_wuji_mode_l, wuji_mode_left)
    pipe.set(key_wuji_mode_r, wuji_mode_right)
    pipe.set(key_t_action, int(now_ms))
    if not bool(dry_run):
        pipe.execute()
