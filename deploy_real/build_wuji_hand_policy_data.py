#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ----- constants copied from previous conversion scripts -----
MEDIAPIPE_21_KEYPOINT_NAMES = [
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
FINGERTIP_NAMES = ["Thumb_Tip", "Index_Tip", "Middle_Tip", "Ring_Tip", "Pinky_Tip"]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
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
MEDIAPIPE_MAPPING_26_TO_21 = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_output_name(raw_name: str, hand_side: str) -> str:
    s = str(raw_name).strip()
    if s:
        return s[:-4] if s.endswith(".npz") else s
    return f"wuji_{hand_side}"


def _load_apply_mediapipe_transformations():
    repo_root = _repo_root()
    candidates = [
        repo_root / "wuji-retargeting" / "wuji_retargeting" / "mediapipe.py",
        repo_root / "wuji_retargeting" / "wuji_retargeting" / "mediapipe.py",
    ]
    mediapipe_py = None
    for cand in candidates:
        if cand.exists():
            mediapipe_py = cand
            break
    if mediapipe_py is None:
        raise FileNotFoundError("Cannot find wuji retarget mediapipe.py")
    spec = importlib.util.spec_from_file_location("_wuji_retargeting_mediapipe", str(mediapipe_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import spec: {mediapipe_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    fn = getattr(mod, "apply_mediapipe_transformations", None)
    if not callable(fn):
        raise AttributeError("apply_mediapipe_transformations not found")
    return fn


def hand_26d_to_mediapipe_21d(hand_data_dict: dict, hand_side: str) -> np.ndarray:
    prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = prefix + joint_name
        if key in hand_data_dict:
            joint_positions_26[i] = np.asarray(hand_data_dict[key][0], dtype=np.float32).reshape(3)
    mediapipe_21d = joint_positions_26[np.asarray(MEDIAPIPE_MAPPING_26_TO_21, dtype=np.int32)]
    wrist = mediapipe_21d[0].copy()
    return mediapipe_21d - wrist


def _iter_data_json_files(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("episode_*/data.json"))


def _frames_from_data_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    if isinstance(root, dict) and isinstance(root.get("data"), list):
        return root["data"]
    raise ValueError(f"Unsupported json format: {path}")


def _safe_float32_array(x: Any, expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"length mismatch, expect {expected_len}, got {arr.size}")
    return arr


def _sanitize_rel_name(input_root: Path, data_json_path: Path) -> str:
    rel = data_json_path.relative_to(input_root).as_posix()
    rel = rel.replace("/", "__").replace("data.json", "").strip("_")
    return rel


def _build_intermediate_for_side(
    frames: list[dict],
    side: str,
    apply_mediapipe_transformations,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_action = f"action_wuji_qpos_target_{side}"
    k_tracking = f"hand_tracking_{side}"
    k_t_action = f"t_action_wuji_hand_{side}"

    actions: list[np.ndarray] = []
    mp21_all: list[np.ndarray] = []
    tips_all: list[np.ndarray] = []
    idx_all: list[int] = []
    ts_all: list[int] = []

    for fr in frames:
        action = fr.get(k_action, None)
        tracking = fr.get(k_tracking, None)
        if action is None or not isinstance(tracking, dict):
            continue
        if not bool(tracking.get("is_active", True)):
            continue
        try:
            a = _safe_float32_array(action, expected_len=20)
            mp21 = hand_26d_to_mediapipe_21d(tracking, hand_side=side)
            mp21 = np.asarray(mp21, dtype=np.float32).reshape(21, 3)
            mp21_t = np.asarray(apply_mediapipe_transformations(mp21, hand_type=side), dtype=np.float32).reshape(21, 3)
            tips = mp21_t[FINGERTIP_INDICES, :].copy()
            if (not np.isfinite(a).all()) or (not np.isfinite(mp21_t).all()) or (not np.isfinite(tips).all()):
                continue
        except Exception:
            continue

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


def _write_intermediate(
    *,
    input_root: Path,
    intermediate_root: Path,
    hand_side: str,
    max_files: int,
    overwrite: bool,
) -> int:
    apply_mediapipe_transformations = _load_apply_mediapipe_transformations()
    data_json_files = _iter_data_json_files(input_root)
    if max_files > 0:
        data_json_files = data_json_files[:max_files]
    sides = ["left", "right"] if hand_side == "both" else [hand_side]
    intermediate_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for p in data_json_files:
        try:
            frames = _frames_from_data_json(p)
        except Exception as e:
            print(f"[WARN] skip unreadable json: {p} ({e})")
            continue
        base = _sanitize_rel_name(input_root, p)
        for side in sides:
            action_arr, mp21_arr, tips_arr, idx_arr, ts_arr = _build_intermediate_for_side(
                frames=frames,
                side=side,
                apply_mediapipe_transformations=apply_mediapipe_transformations,
            )
            if action_arr.shape[0] == 0:
                continue
            out_file = intermediate_root / f"{base}__{side}.npz"
            if out_file.exists() and (not overwrite):
                continue
            np.savez_compressed(
                out_file,
                action_wuji_qpos_target=action_arr,
                mediapipe_21d_transformed=mp21_arr,
                mediapipe_21d_keypoint_names=np.asarray(MEDIAPIPE_21_KEYPOINT_NAMES),
                fingertips_rel_wrist=tips_arr,
                fingertip_names=np.asarray(FINGERTIP_NAMES),
                frame_idx=idx_arr,
                timestamp_ms=ts_arr,
                hand_side=np.asarray([side]),
                source_data_json=np.asarray([str(p)]),
            )
            written += 1
    return written


def _normalize_human(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1:] == (5, 3):
        return arr
    if arr.ndim == 2 and arr.shape[1] == 15:
        return arr.reshape(arr.shape[0], 5, 3)
    raise ValueError(f"fingertips_rel_wrist unexpected shape: {arr.shape}")


def _normalize_robot(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"action_wuji_qpos_target unexpected shape: {arr.shape}")


def _collect_intermediate_files(intermediate_root: Path, hand_side: str) -> list[Path]:
    if hand_side == "left":
        patterns = ["*left.npz", "*left*.npz"]
    elif hand_side == "right":
        patterns = ["*right.npz", "*right*.npz"]
    else:
        patterns = ["*left*.npz", "*right*.npz"]
    files: list[Path] = []
    for pat in patterns:
        files.extend([p for p in intermediate_root.glob(pat) if p.is_file()])
    dedup = sorted({str(p.resolve()): p.resolve() for p in files}.values(), key=lambda p: str(p))
    return dedup


def _merge_intermediate(intermediate_root: Path, output_path: Path, hand_side: str) -> None:
    files = _collect_intermediate_files(intermediate_root, hand_side)
    if not files:
        raise RuntimeError(f"No intermediate npz found in {intermediate_root}")

    human_chunks: list[np.ndarray] = []
    robot_chunks: list[np.ndarray] = []
    optional_concat_keys = ["frame_idx", "timestamp_ms", "mediapipe_21d_transformed"]
    optional_chunks: dict[str, list[np.ndarray]] = {k: [] for k in optional_concat_keys}
    optional_presence: dict[str, int] = {k: 0 for k in optional_concat_keys}
    meta_keys = ["fingertip_names", "hand_side", "mediapipe_21d_keypoint_names", "source_data_json"]
    meta_store: dict[str, np.ndarray] = {}

    kept = 0
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        keys = set(z.files)
        if "fingertips_rel_wrist" not in keys or "action_wuji_qpos_target" not in keys:
            z.close()
            continue
        try:
            human = _normalize_human(z["fingertips_rel_wrist"])
            robot = _normalize_robot(z["action_wuji_qpos_target"])
            t = min(human.shape[0], robot.shape[0])
            if t <= 0:
                raise ValueError("empty segment")
            human = human[:t].astype(np.float32)
            robot = robot[:t].astype(np.float32)
            if (not np.isfinite(human).all()) or (not np.isfinite(robot).all()):
                raise ValueError("non-finite in intermediate file")
            human_chunks.append(human)
            robot_chunks.append(robot)

            for k in optional_concat_keys:
                if k in keys:
                    arr = np.asarray(z[k])
                    if arr.ndim >= 1 and arr.shape[0] >= t:
                        optional_chunks[k].append(arr[:t])
                        optional_presence[k] += 1

            for k in meta_keys:
                if k in keys and k not in meta_store:
                    meta_store[k] = z[k]
            kept += 1
        finally:
            z.close()

    if kept == 0:
        raise RuntimeError("No usable intermediate files")

    human_all = np.concatenate(human_chunks, axis=0)
    robot_all = np.concatenate(robot_chunks, axis=0)
    out: dict[str, Any] = {
        "fingertips_rel_wrist": human_all,
        "qpos": robot_all,
        "robot_qpos": robot_all,
        "joint": robot_all,
        "joint_angle": robot_all,
        "joint_angles": robot_all,
        "action_wuji_qpos_target": robot_all,
    }
    total_t = human_all.shape[0]
    for k in optional_concat_keys:
        if optional_presence[k] == kept:
            arr = np.concatenate(optional_chunks[k], axis=0)
            if arr.shape[0] == total_t:
                out[k] = arr
    for k, v in meta_store.items():
        out[k] = v

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    default_data_root = repo_root / "deploy_real" / "humdex_demonstration"
    parser = argparse.ArgumentParser(description="Build Wuji hand policy training NPZ in one self-contained script.")
    parser.add_argument("--session_dir_name", type=str, default="", help="Subdirectory under data_root.")
    parser.add_argument("--input_root", type=str, default="", help="Input root containing episode_*/data.json.")
    parser.add_argument("--data_root", type=str, default=str(default_data_root))
    parser.add_argument("--hand_side", type=str, default="left", choices=["left", "right", "both"])
    parser.add_argument("--output_dir", type=str, default=str(repo_root / "wuji_policy" / "data"))
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--intermediate_root", type=str, default="")
    parser.add_argument("--max_files", type=int, default=-1)
    parser.add_argument("--no_overwrite", action="store_true")
    parser.add_argument("--cleanup_intermediate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    data_root = Path(args.data_root).expanduser().resolve()
    if str(args.input_root).strip():
        input_root = Path(args.input_root).expanduser().resolve()
    else:
        session_name = str(args.session_dir_name).strip()
        if not session_name:
            print("[ERROR] Missing input target. Provide --input_root or --session_dir_name.", file=sys.stderr)
            return 1
        input_root = (data_root / session_name).resolve()
    if not input_root.exists():
        print(f"[ERROR] input_root not found: {input_root}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_name = _resolve_output_name(args.output_name, args.hand_side)
    output_path = (output_dir / f"{output_name}.npz").resolve()
    if str(args.intermediate_root).strip():
        intermediate_root = Path(args.intermediate_root).expanduser().resolve()
    else:
        intermediate_root = (
            repo_root / "deploy_real" / "wuji_hand_policy_dataset_tmp" / f"{output_name}_{args.hand_side}"
        ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] input_root     : {input_root}")
    print(f"[INFO] hand_side      : {args.hand_side}")
    print(f"[INFO] intermediate   : {intermediate_root}")
    print(f"[INFO] output_npz     : {output_path}")

    written = _write_intermediate(
        input_root=input_root,
        intermediate_root=intermediate_root,
        hand_side=args.hand_side,
        max_files=int(args.max_files),
        overwrite=(not bool(args.no_overwrite)),
    )
    print(f"[INFO] intermediate files written: {written}")

    _merge_intermediate(
        intermediate_root=intermediate_root,
        output_path=output_path,
        hand_side=args.hand_side,
    )
    print(f"[DONE] Dataset written to: {output_path}")

    if bool(args.cleanup_intermediate):
        shutil.rmtree(intermediate_root, ignore_errors=True)
        print(f"[INFO] intermediate removed: {intermediate_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
