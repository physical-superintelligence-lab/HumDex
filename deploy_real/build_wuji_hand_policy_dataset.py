#!/usr/bin/env python3
"""
离线构建 Wuji 灵巧手 policy 训练数据集：

从 deploy_real/humdex demonstration/**/episode_*/data.json 读取每帧：
- action_wuji_qpos_target_{left/right} (20维，Wuji手关节目标)
- hand_tracking_{left/right} (26D tracker dict)

并复用 deploy_real/server_wuji_hand_redis.py 的逻辑做转换：
hand_tracking(26D) -> mediapipe_21d(21x3, wrist为零点) -> apply_mediapipe_transformations()

最终只保存：
- action_wuji_qpos_target
- mediapipe_21d_transformed（21个关键点，带名称）
- 5个指尖相对手腕坐标（thumb/index/middle/ring/pinky tip）

输出格式：每个 episode/hand_side 一个 .npz 文件 + 一个 manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore


# ---- MediaPipe 21 keypoint names (和本仓库 26->21 映射保持一致) ----
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

FINGERTIP_NAMES: List[str] = ["Thumb_Tip", "Index_Tip", "Middle_Tip", "Ring_Tip", "Pinky_Tip"]
FINGERTIP_INDICES: List[int] = [4, 8, 12, 16, 20]


def _repo_root() -> Path:
    # .../HumDex/deploy_real/build_wuji_hand_policy_dataset.py -> .../HumDex
    return Path(__file__).resolve().parents[1]


def _ensure_wuji_retargeting_on_path() -> None:
    """
    与 deploy_real/server_wuji_hand_redis.py 的做法一致：
    把仓库内的 wuji-retargeting 或 legacy wuji_retargeting 加到 sys.path。
    """
    import sys

    project_root = _repo_root()
    for wuji_retargeting_path in [
        project_root / "wuji-retargeting",
        project_root / "wuji_retargeting",
    ]:
        if wuji_retargeting_path.exists() and str(wuji_retargeting_path) not in sys.path:
            sys.path.insert(0, str(wuji_retargeting_path))

# ---- 26D -> 21D 转换（与 deploy_real/server_wuji_hand_redis.py 保持一致，避免依赖 deploy_real 作为包导入）----
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

# MediaPipe: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26D:      [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
MEDIAPIPE_MAPPING_26_TO_21: List[int] = [
    1,  # 0: Wrist -> Wrist
    2,  # 1: ThumbMetacarpal -> Thumb CMC
    3,  # 2: ThumbProximal -> Thumb MCP
    4,  # 3: ThumbDistal -> Thumb IP
    5,  # 4: ThumbTip -> Thumb Tip
    6,  # 5: IndexMetacarpal -> Index MCP
    7,  # 6: IndexProximal -> Index PIP
    8,  # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (跳过 IndexDistal)
    11,  # 9: MiddleMetacarpal -> Middle MCP
    12,  # 10: MiddleProximal -> Middle PIP
    13,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (跳过 MiddleDistal)
    16,  # 13: RingMetacarpal -> Ring MCP
    17,  # 14: RingProximal -> Ring PIP
    18,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (跳过 RingDistal)
    21,  # 17: LittleMetacarpal -> Pinky MCP
    22,  # 18: LittleProximal -> Pinky PIP
    23,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (跳过 LittleDistal)
]


def hand_26d_to_mediapipe_21d(hand_data_dict: Dict, hand_side: str = "left", print_distances: bool = False) -> np.ndarray:
    """
    将26维手部追踪 dict 转换为21维 MediaPipe 格式 (21,3)，并以 wrist 为原点。

    hand_data_dict: {"LeftHandWrist": [[x,y,z], [qw,qx,qy,qz]], ...}（附带 is_active/timestamp 等字段也允许）
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"

    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]  # [x, y, z]
            joint_positions_26[i] = pos
        else:
            joint_positions_26[i] = [0.0, 0.0, 0.0]

    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]

    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos

    # 与线上保持一致：目前 scale_factor=1.0
    scale_factor = 1.0
    mediapipe_21d[1:] = mediapipe_21d[1:] * scale_factor

    if print_distances:
        # 仅保留调试能力（不在离线导出中使用）
        pass

    return mediapipe_21d


def _iter_data_json_files(input_root: Path) -> List[Path]:
    files = sorted(input_root.rglob("episode_*/data.json"))
    return files


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _frames_from_root(root: dict) -> List[dict]:
    if isinstance(root, dict) and "data" in root and isinstance(root["data"], list):
        return root["data"]
    raise ValueError("不支持的 data.json 结构：期望顶层为 dict 且包含 list 类型的 'data' 字段。")


def _safe_float32_array(x: List[float], expected_len: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"action 维度不匹配：期望 {expected_len}，实际 {arr.size}")
    return arr


def _build_one_episode_side(
    frames: List[dict],
    side: str,
    hand_26d_to_mediapipe_21d,
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
        raise ValueError(f"side 必须是 left/right，收到: {side}")

    k_action = f"action_wuji_qpos_target_{side}"
    k_tracking = f"hand_tracking_{side}"
    k_t_action = f"t_action_wuji_hand_{side}"

    actions: List[np.ndarray] = []
    mp21_all: List[np.ndarray] = []
    tips_all: List[np.ndarray] = []
    idx_all: List[int] = []
    ts_all: List[int] = []
    dropped_non_finite = 0

    for fr in frames:
        action = fr.get(k_action, None)
        tracking = fr.get(k_tracking, None)
        if action is None or tracking is None:
            continue

        # tracking 里包含 is_active / timestamp / 26个关节 key
        if isinstance(tracking, dict):
            if not bool(tracking.get("is_active", True)):
                continue
        else:
            continue

        try:
            a = _safe_float32_array(action, expected_len=20)
        except Exception:
            # 避免单帧坏数据导致整段失败
            continue

        # 按需求：这两个字段统一置零，保持 shape 一致。
        # mediapipe_21d_transformed: (21,3), fingertips_rel_wrist: (5,3)
        mp21_t = np.zeros((21, 3), dtype=np.float32)
        tips = np.zeros((5, 3), dtype=np.float32)

        if not np.isfinite(a).all():
            dropped_non_finite += 1
            continue

        actions.append(a)
        mp21_all.append(mp21_t)
        tips_all.append(tips)
        idx_all.append(int(fr.get("idx", len(idx_all))))

        # 时间戳优先用 t_action_wuji_hand_*；没有的话回退 tracking.timestamp；再没有给 -1
        ts = fr.get(k_t_action, None)
        if ts is None and isinstance(tracking, dict):
            ts = tracking.get("timestamp", None)
        ts_all.append(int(ts) if ts is not None else -1)

    if dropped_non_finite > 0:
        print(f"[INFO] [{side}] 过滤非有限值帧: {dropped_non_finite}")

    if len(actions) == 0:
        return (
            np.zeros((0, 20), dtype=np.float32),
            np.zeros((0, 21, 3), dtype=np.float32),
            np.zeros((0, 5, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    action_arr = np.stack(actions, axis=0)
    mp21_arr = np.stack(mp21_all, axis=0)
    tips_arr = np.stack(tips_all, axis=0)
    idx_arr = np.asarray(idx_all, dtype=np.int64)
    ts_arr = np.asarray(ts_all, dtype=np.int64)
    return action_arr, mp21_arr, tips_arr, idx_arr, ts_arr


def _sanitize_rel_name(input_root: Path, data_json_path: Path) -> str:
    rel = data_json_path.relative_to(input_root).as_posix()
    rel = rel.replace("/", "__")
    rel = rel.replace("data.json", "").strip("_")
    return rel


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Wuji 灵巧手 policy 训练数据集（离线导出）")
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(_repo_root() / "deploy_real" / "humdex demonstration"),
        help="humdex demonstration 根目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(_repo_root() / "deploy_real" / "wuji_hand_policy_dataset"),
        help="输出数据集目录",
    )
    parser.add_argument(
        "--hand_side",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="导出左手/右手/两只手",
    )
    parser.add_argument("--max_files", type=int, default=-1, help="最多处理多少个 data.json（用于测试）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出 npz")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    def _load_apply_mediapipe_transformations():
        """
        只加载 wuji_retargeting/wuji_retargeting/mediapipe.py，避免 import 整包触发 pinocchio 依赖。
        """
        import importlib.util

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
            raise FileNotFoundError("找不到 mediapipe.py，请检查 wuji-retargeting 路径。")

        spec = importlib.util.spec_from_file_location("_wuji_retargeting_mediapipe", str(mediapipe_py))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法为 {mediapipe_py} 创建 import spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        fn = getattr(mod, "apply_mediapipe_transformations", None)
        if fn is None:
            raise AttributeError("mediapipe.py 中未找到 apply_mediapipe_transformations")
        return fn

    apply_mediapipe_transformations = _load_apply_mediapipe_transformations()

    data_json_files = _iter_data_json_files(input_root)
    if args.max_files and args.max_files > 0:
        data_json_files = data_json_files[: args.max_files]

    sides: List[str] = ["left", "right"] if args.hand_side == "both" else [args.hand_side]

    manifest_path = output_root / "manifest.jsonl"
    kp_names = np.asarray(MEDIAPIPE_21_KEYPOINT_NAMES)
    tip_names = np.asarray(FINGERTIP_NAMES)

    total_written = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for p in data_json_files:
            try:
                root = _load_json(p)
                frames = _frames_from_root(root)
            except Exception as e:
                print(f"[跳过] 无法读取/解析: {p} ({e})")
                continue

            base = _sanitize_rel_name(input_root, p)

            for side in sides:
                action_arr, mp21_arr, tips_arr, idx_arr, ts_arr = _build_one_episode_side(
                    frames,
                    side=side,
                    hand_26d_to_mediapipe_21d=hand_26d_to_mediapipe_21d,
                    apply_mediapipe_transformations=apply_mediapipe_transformations,
                )

                if action_arr.shape[0] == 0:
                    continue

                out_file = output_root / f"{base}__{side}.npz"
                if out_file.exists() and not args.overwrite:
                    print(f"[跳过] 已存在: {out_file}")
                    continue

                np.savez_compressed(
                    out_file,
                    action_wuji_qpos_target=action_arr,  # (N,20)
                    mediapipe_21d_transformed=mp21_arr,  # (N,21,3)
                    mediapipe_21d_keypoint_names=kp_names,  # (21,)
                    fingertips_rel_wrist=tips_arr,  # (N,5,3)
                    fingertip_names=tip_names,  # (5,)
                    frame_idx=idx_arr,  # (N,)
                    timestamp_ms=ts_arr,  # (N,)
                    hand_side=np.asarray([side]),
                    source_data_json=np.asarray([str(p)]),
                )

                rec = {
                    "output_npz": str(out_file),
                    "source_data_json": str(p),
                    "hand_side": side,
                    "num_frames": int(action_arr.shape[0]),
                    "action_dim": int(action_arr.shape[1]),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"✅ 完成：写出 {total_written} 个 npz 到: {output_root}")
    print(f"🧾 manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


