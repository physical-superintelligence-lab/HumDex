#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click builder for Wuji / GeoRT supervised training data.

Input:
- deploy_real/humdex_demonstration/**/episode_*/data.json

Pipeline:
1. Read per frame:
   - action_wuji_qpos_target_{left/right}   (20D)
   - hand_tracking_{left/right}             (26D tracker dict)
2. Reuse the logic from server_wuji_hand_redis.py:
   hand_tracking(26D) -> mediapipe_21d(21x3, wrist-relative) -> apply_mediapipe_transformations()
3. Extract:
   - action_wuji_qpos_target
   - mediapipe_21d_transformed
   - fingertips_rel_wrist
4. Concatenate all valid episode / hand_side segments into one merged NPZ,
   compatible with SupervisedRetargetDataset:
   - fingertips_rel_wrist
   - qpos
   - optionally frame_idx / timestamp_ms / mediapipe_21d_transformed if aligned
   - meta info if available

Output:
- One merged NPZ file (program-2 style)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===================== PATH / CONFIG =====================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _repo_root()
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "deploy_real" / "humdex_demonstration"
DEFAULT_OUT_DIR = PROJECT_ROOT / "wuji_policy" / "data"

# source/output keys
HUMAN_KEY = "fingertips_rel_wrist"
ROBOT_IN_KEY = "action_wuji_qpos_target"
QPOS_KEY_OUT = "qpos"
QPOS_ALIAS_KEYS = ["robot_qpos", "joint", "joint_angle", "joint_angles"]
KEEP_ORIGINAL_ROBOT_KEY = True

META_KEYS = [
    "fingertip_names",
    "hand_side",
    "mediapipe_21d_keypoint_names",
]

OPTIONAL_CONCAT_KEYS = [
    "frame_idx",
    "timestamp_ms",
    "mediapipe_21d_transformed",
]

# ===================== HAND / MEDIAPIPE DEFINITIONS =====================

MEDIAPIPE_21_KEYPOINT_NAMES: List[str] = [
    "Wrist",
    "Thumb_CMC",
    "Thumb_MCP",
    "Thumb_IP",
    "Thumb_Tip",
    "Index_MCP",
    "Index_PIP",
    "Index_DIP",
    "Index_Tip",
    "Middle_MCP",
    "Middle_PIP",
    "Middle_DIP",
    "Middle_Tip",
    "Ring_MCP",
    "Ring_PIP",
    "Ring_DIP",
    "Ring_Tip",
    "Pinky_MCP",
    "Pinky_PIP",
    "Pinky_DIP",
    "Pinky_Tip",
]

FINGERTIP_NAMES: List[str] = [
    "Thumb_Tip",
    "Index_Tip",
    "Middle_Tip",
    "Ring_Tip",
    "Pinky_Tip",
]

FINGERTIP_INDICES: List[int] = [4, 8, 12, 16, 20]

HAND_JOINT_NAMES_26: List[str] = [
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

MEDIAPIPE_MAPPING_26_TO_21: List[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    20,
    21,
    22,
    23,
    25,
]


# ===================== UTIL =====================

def _iter_data_json_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("episode_*/data.json"))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _frames_from_root(root: dict) -> List[dict]:
    if isinstance(root, dict) and "data" in root and isinstance(root["data"], list):
        return root["data"]
    raise ValueError("Unsupported data.json format: expected a top-level dict with a list field named 'data'.")


def _safe_float32_array(x, expected_len: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"Dimension mismatch: expected {expected_len}, got {arr.size}")
    return arr


def _resolve_output_path(output_name: Optional[str]) -> Path:
    if output_name:
        name = output_name if output_name.endswith(".npz") else f"{output_name}.npz"
    else:
        name = "wuji_geort_merged.npz"
    return (DEFAULT_OUT_DIR / name).resolve()


def _load_apply_mediapipe_transformations():
    """
    Load only wuji-retargeting/wuji_retargeting/mediapipe.py
    to avoid importing the whole package and triggering heavy dependencies.
    """
    mediapipe_py = PROJECT_ROOT / "wuji-retargeting" / "wuji_retargeting" / "mediapipe.py"
    if not mediapipe_py.exists():
        raise FileNotFoundError(f"File not found: {mediapipe_py}")

    spec = importlib.util.spec_from_file_location("_wuji_retargeting_mediapipe", str(mediapipe_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {mediapipe_py}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    fn = getattr(mod, "apply_mediapipe_transformations", None)
    if fn is None:
        raise AttributeError("apply_mediapipe_transformations was not found in mediapipe.py")
    return fn


# ===================== 26D -> 21D =====================

def hand_26d_to_mediapipe_21d(
    hand_data_dict: Dict,
    hand_side: str = "left",
    print_distances: bool = False,
) -> np.ndarray:
    """
    Convert a 26-joint hand tracking dict to MediaPipe 21 keypoints (21,3),
    normalized to be wrist-relative.
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"

    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]
            joint_positions_26[i] = pos
        else:
            joint_positions_26[i] = [0.0, 0.0, 0.0]

    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]
    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos

    scale_factor = 1.0
    mediapipe_21d[1:] = mediapipe_21d[1:] * scale_factor

    if print_distances:
        pass

    return mediapipe_21d


# ===================== BUILD PER EPISODE / SIDE =====================

def _build_one_episode_side(
    frames: List[dict],
    side: str,
    apply_mediapipe_transformations,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      action: (N, 20)
      mediapipe_21d_transformed: (N, 21, 3)
      fingertips_rel_wrist: (N, 5, 3)
      frame_idx: (N,)
      timestamp_ms: (N,)
    """
    side = side.lower()
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got: {side}")

    k_action = f"action_wuji_qpos_target_{side}"
    k_tracking = f"hand_tracking_{side}"
    k_t_action = f"t_action_wuji_hand_{side}"

    actions: List[np.ndarray] = []
    mp21_all: List[np.ndarray] = []
    tips_all: List[np.ndarray] = []
    idx_all: List[int] = []
    ts_all: List[int] = []

    for fr in frames:
        action = fr.get(k_action, None)
        tracking = fr.get(k_tracking, None)
        if action is None or tracking is None:
            continue

        if not isinstance(tracking, dict):
            continue
        if not bool(tracking.get("is_active", True)):
            continue

        try:
            a = _safe_float32_array(action, expected_len=20)
            mp21 = hand_26d_to_mediapipe_21d(tracking, hand_side=side, print_distances=False)
            mp21 = np.asarray(mp21, dtype=np.float32).reshape(21, 3)
            mp21_t = apply_mediapipe_transformations(mp21, hand_type=side)
            mp21_t = np.asarray(mp21_t, dtype=np.float32).reshape(21, 3)
        except Exception:
            continue

        tips = mp21_t[FINGERTIP_INDICES, :].copy()

        actions.append(a)
        mp21_all.append(mp21_t)
        tips_all.append(tips)
        idx_all.append(int(fr.get("idx", len(idx_all))))

        ts = fr.get(k_t_action, None)
        if ts is None:
            ts = tracking.get("timestamp", None)
        ts_all.append(int(ts) if ts is not None else -1)

    if len(actions) == 0:
        return (
            np.zeros((0, 20), dtype=np.float32),
            np.zeros((0, 21, 3), dtype=np.float32),
            np.zeros((0, 5, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(actions, axis=0),
        np.stack(mp21_all, axis=0),
        np.stack(tips_all, axis=0),
        np.asarray(idx_all, dtype=np.int64),
        np.asarray(ts_all, dtype=np.int64),
    )


# ===================== MAIN =====================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build one merged GeoRT/Wuji supervised dataset directly from humdex_demonstration data.json files."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(DEFAULT_INPUT_ROOT),
        help="Root directory of humdex_demonstration.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Output NPZ file name, with or without the .npz suffix.",
    )
    parser.add_argument(
        "--hand_side",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="Which hand side to export: left, right, or both.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=-1,
        help="Maximum number of data.json files to process, useful for debugging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    out_path = _resolve_output_path(args.output_name.strip() or None)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"[ERROR] input_root does not exist: {input_root}", file=sys.stderr)
        return 1

    apply_mediapipe_transformations = _load_apply_mediapipe_transformations()

    data_json_files = _iter_data_json_files(input_root)
    if args.max_files > 0:
        data_json_files = data_json_files[:args.max_files]

    if not data_json_files:
        print(f"[ERROR] No data.json files found under: {input_root}", file=sys.stderr)
        return 1

    sides: List[str] = ["left", "right"] if args.hand_side == "both" else [args.hand_side]

    print(f"[INFO] Found {len(data_json_files)} data.json files under: {input_root}")
    print(f"[INFO] hand_side={args.hand_side}")

    human_chunks: List[np.ndarray] = []
    robot_chunks: List[np.ndarray] = []

    optional_chunks = {k: [] for k in OPTIONAL_CONCAT_KEYS}
    optional_presence = {k: 0 for k in OPTIONAL_CONCAT_KEYS}

    meta_store = {}
    source_paths = []
    segment_lengths = []
    segment_sides = []

    kept_segments = 0
    skipped_files = 0

    kp_names = np.asarray(MEDIAPIPE_21_KEYPOINT_NAMES)
    tip_names = np.asarray(FINGERTIP_NAMES)

    for p in data_json_files:
        try:
            root = _load_json(p)
            frames = _frames_from_root(root)
        except Exception as e:
            print(f"[WARN] Skip file {p}: {type(e).__name__}: {e}")
            skipped_files += 1
            continue

        for side in sides:
            try:
                action_arr, mp21_arr, tips_arr, idx_arr, ts_arr = _build_one_episode_side(
                    frames=frames,
                    side=side,
                    apply_mediapipe_transformations=apply_mediapipe_transformations,
                )

                T = action_arr.shape[0]
                if T <= 0:
                    continue

                human_chunks.append(tips_arr.astype(np.float32))
                robot_chunks.append(action_arr.astype(np.float32))

                optional_chunks["frame_idx"].append(idx_arr)
                optional_presence["frame_idx"] += 1

                optional_chunks["timestamp_ms"].append(ts_arr)
                optional_presence["timestamp_ms"] += 1

                optional_chunks["mediapipe_21d_transformed"].append(mp21_arr.astype(np.float32))
                optional_presence["mediapipe_21d_transformed"] += 1

                if "fingertip_names" not in meta_store:
                    meta_store["fingertip_names"] = tip_names
                if "mediapipe_21d_keypoint_names" not in meta_store:
                    meta_store["mediapipe_21d_keypoint_names"] = kp_names

                source_paths.append(str(p))
                segment_lengths.append(T)
                segment_sides.append(side)

                kept_segments += 1

            except Exception as e:
                print(f"[WARN] Skip segment {p} [{side}]: {type(e).__name__}: {e}")

    if kept_segments == 0:
        print("[ERROR] No valid segments were found. Failed to build dataset.", file=sys.stderr)
        return 1

    human_all = np.concatenate(human_chunks, axis=0)
    robot_all = np.concatenate(robot_chunks, axis=0)
    total_T = human_all.shape[0]

    out = {
        HUMAN_KEY: human_all,
        QPOS_KEY_OUT: robot_all,
    }

    for k in QPOS_ALIAS_KEYS:
        out[k] = robot_all

    if KEEP_ORIGINAL_ROBOT_KEY:
        out[ROBOT_IN_KEY] = robot_all

    for k in OPTIONAL_CONCAT_KEYS:
        if optional_presence[k] != kept_segments:
            if optional_presence[k] > 0:
                print(f"[INFO] Optional key '{k}' present in {optional_presence[k]}/{kept_segments} segments -> NOT saving.")
            continue
        try:
            arr_all = np.concatenate(optional_chunks[k], axis=0)
            if arr_all.shape[0] != total_T:
                print(f"[WARN] Optional key '{k}' total length {arr_all.shape[0]} != total_T {total_T} -> NOT saving.")
                continue
            out[k] = arr_all
        except Exception as e:
            print(f"[WARN] Cannot concatenate optional key '{k}': {type(e).__name__}: {e}")

    for k, v in meta_store.items():
        out[k] = v

    out["source_data_json_list"] = np.asarray(source_paths, dtype=object)
    out["segment_lengths"] = np.asarray(segment_lengths, dtype=np.int64)
    out["segment_hand_sides"] = np.asarray(segment_sides, dtype=object)

    np.savez_compressed(out_path, **out)

    print(f"[INFO] Saved merged dataset: {out_path}")
    print(f"[INFO] kept_segments={kept_segments}, skipped_files={skipped_files}, total_T={total_T}")
    print("[INFO] Saved keys & shapes:")
    for k in sorted(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray):
            print(f"  {k:28s} shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k:28s} type={type(v)}")

    print("[CHECK]", HUMAN_KEY, "OK" if HUMAN_KEY in out else "MISSING")
    print("[CHECK]", QPOS_KEY_OUT, "OK" if QPOS_KEY_OUT in out else "MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    