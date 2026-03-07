from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SlimevrBodyConfig:
    vmc_ip: str = "0.0.0.0"
    vmc_port: int = 39539
    vmc_timeout_s: float = 0.5
    vmc_rot_mode: str = "local"
    vmc_use_fk: bool = True
    vmc_use_viewer_fk: bool = True
    vmc_fk_skeleton: str = "bvh"
    vmc_bvh_path: str = "bvh-recording.bvh"
    vmc_bvh_scale: float = 1.0


def _safe_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _normalize_vmc_name(name: str) -> str:
    return "".join([c.lower() for c in str(name) if c.isalnum()])


def _vmc_quat_xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return _safe_quat_wxyz(np.array([qw, qx, qy, qz], dtype=np.float32))


def _quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    q = _safe_quat_wxyz(np.asarray(q, dtype=np.float32).reshape(4))
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_to_mat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        q = (q / n).astype(np.float32)
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _axis_swap_flip_matrix(swap: str, flip: str) -> np.ndarray:
    swap = str(swap).lower()
    flip = str(flip).lower()
    axes = {"x": 0, "y": 1, "z": 2}
    if len(swap) != 3 or any(c not in axes for c in swap):
        swap = "xyz"
    idx = [axes[c] for c in swap]
    m = np.eye(3, dtype=np.float32)[:, idx]
    f = np.ones(3, dtype=np.float32)
    for c in flip:
        if c in axes:
            f[axes[c]] *= -1.0
    return m * f.reshape(1, 3)


def _parse_bvh_offsets(path: str) -> tuple[dict[str, Optional[str]], dict[str, np.ndarray]]:
    parents: dict[str, Optional[str]] = {}
    offsets: dict[str, np.ndarray] = {}
    stack: list[Optional[str]] = []
    current: Optional[str] = None
    in_end_site = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            head = tokens[0]
            if head in ["ROOT", "JOINT"]:
                name = tokens[1]
                current = name
                parent = stack[-1] if stack else None
                parents[name] = parent
                stack.append(name)
            elif head == "End" and len(tokens) >= 2 and tokens[1] == "Site":
                in_end_site = True
                stack.append(None)
            elif head == "OFFSET" and len(tokens) >= 4:
                if not in_end_site and current is not None:
                    offsets[current] = np.array(
                        [float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=np.float32
                    )
            elif head == "}":
                if stack:
                    top = stack.pop()
                    if top is None:
                        in_end_site = False
                current = stack[-1] if stack else None
            elif head == "MOTION":
                break
    return parents, offsets


def _std_fk_skeleton() -> tuple[dict[str, Optional[str]], dict[str, np.ndarray], np.ndarray]:
    offsets: dict[str, np.ndarray] = {}
    parents: dict[str, Optional[str]] = {}

    def add(name: str, parent: Optional[str], off: tuple[float, float, float]) -> None:
        parents[name] = parent
        offsets[name] = np.array(off, dtype=np.float32)

    add("Hips", None, (0.0, 0.0, 0.0))
    add("Spine", "Hips", (0.0, 0.10, 0.0))
    add("Chest", "Spine", (0.0, 0.15, 0.0))
    add("UpperChest", "Chest", (0.0, 0.15, 0.0))
    add("Neck", "UpperChest", (0.0, 0.15, 0.0))
    add("Head", "Neck", (0.0, 0.10, 0.0))
    add("LeftUpperLeg", "Hips", (-0.08, -0.05, 0.0))
    add("LeftLowerLeg", "LeftUpperLeg", (0.0, -0.42, 0.0))
    add("LeftFoot", "LeftLowerLeg", (0.0, -0.40, 0.0))
    add("RightUpperLeg", "Hips", (0.08, -0.05, 0.0))
    add("RightLowerLeg", "RightUpperLeg", (0.0, -0.42, 0.0))
    add("RightFoot", "RightLowerLeg", (0.0, -0.40, 0.0))
    add("LeftShoulder", "UpperChest", (-0.10, 0.10, 0.0))
    add("LeftUpperArm", "LeftShoulder", (-0.12, 0.0, 0.0))
    add("LeftLowerArm", "LeftUpperArm", (-0.28, 0.0, 0.0))
    add("LeftHand", "LeftLowerArm", (-0.25, 0.0, 0.0))
    add("RightShoulder", "UpperChest", (0.10, 0.10, 0.0))
    add("RightUpperArm", "RightShoulder", (0.12, 0.0, 0.0))
    add("RightLowerArm", "RightUpperArm", (0.28, 0.0, 0.0))
    add("RightHand", "RightLowerArm", (0.25, 0.0, 0.0))
    root_pos = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return parents, offsets, root_pos


def _build_fk_from_vmc(
    raw_map: dict[str, tuple[np.ndarray, np.ndarray]],
    parents: dict[str, Optional[str]],
    offsets: dict[str, np.ndarray],
    bvh_to_vmc: dict[str, str],
    scale: float,
    rot_mode: str,
    root_pos: Optional[np.ndarray],
    axis_m: Optional[np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pos_out: dict[str, np.ndarray] = {}
    rot_out: dict[str, np.ndarray] = {}

    def get_local_rot(joint: str) -> np.ndarray:
        vmc_name = bvh_to_vmc.get(joint)
        if vmc_name is None:
            return np.eye(3, dtype=np.float32)
        key = _normalize_vmc_name(vmc_name)
        v = raw_map.get(key)
        if v is None:
            return np.eye(3, dtype=np.float32)
        _p, q = v
        return _quat_to_mat_wxyz(q)

    def solve(joint: str) -> None:
        if joint in pos_out:
            return
        parent = parents.get(joint)
        if parent is None:
            local_rot = get_local_rot(joint)
            if root_pos is None:
                pos_out[joint] = np.zeros(3, dtype=np.float32)
            else:
                rp = np.asarray(root_pos, dtype=np.float32).reshape(3)
                pos_out[joint] = (axis_m @ rp) if axis_m is not None else rp
            rot_out[joint] = local_rot
            return
        solve(parent)
        parent_rot = rot_out[parent]
        local_rot = get_local_rot(joint)
        if rot_mode == "global":
            local_rot = parent_rot.T @ local_rot
        rot_out[joint] = parent_rot @ local_rot
        off = offsets.get(joint, np.zeros(3, dtype=np.float32))
        off = (axis_m @ off) if axis_m is not None else off
        pos_out[joint] = pos_out[parent] + parent_rot @ (off * float(scale))

    for j in parents.keys():
        solve(j)
    return pos_out, rot_out


def _fk_to_vmc_pose(
    fk_pos: dict[str, np.ndarray],
    fk_rot: dict[str, np.ndarray],
    bvh_to_vmc: dict[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for bvh_name, vmc_name in bvh_to_vmc.items():
        p = fk_pos.get(bvh_name)
        r = fk_rot.get(bvh_name)
        if p is None or r is None:
            continue
        out[_normalize_vmc_name(vmc_name)] = (np.asarray(p, dtype=np.float32).reshape(3), _mat_to_quat_wxyz(r))
    return out


def _build_fk_from_vmc_std(
    raw_xyzw: dict[str, tuple[np.ndarray, np.ndarray]],
    rot_mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    parents, offsets, root_pos = _std_fk_skeleton()

    def get_rot_xyzw(name: str) -> np.ndarray:
        key = _normalize_vmc_name(name)
        v = raw_xyzw.get(key)
        if v is None:
            return np.eye(3, dtype=np.float32)
        _p, q = v
        return _quat_to_mat_xyzw(q)

    pos: dict[str, np.ndarray] = {}
    rot: dict[str, np.ndarray] = {}
    hips_rot = get_rot_xyzw("Hips")
    pos["Hips"] = np.asarray(root_pos, dtype=np.float32).reshape(3)
    rot["Hips"] = hips_rot
    processed = {"Hips"}
    while len(processed) < len(parents):
        for bone, parent in parents.items():
            if bone in processed:
                continue
            if parent in processed:
                parent_pos = pos[parent]
                parent_rot = rot[parent]
                off = offsets.get(bone, np.zeros(3, dtype=np.float32))
                pos[bone] = parent_pos + parent_rot @ off
                bone_rot = get_rot_xyzw(bone)
                if rot_mode == "global":
                    local_rot = parent_rot.T @ bone_rot
                    rot[bone] = parent_rot @ local_rot
                else:
                    rot[bone] = parent_rot @ bone_rot
                processed.add(bone)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in parents.keys():
        if name in pos and name in rot:
            out[_normalize_vmc_name(name)] = (
                np.asarray(pos[name], dtype=np.float32).reshape(3),
                _mat_to_quat_wxyz(rot[name]),
            )
    return out


def _vmc_build_body_frame(
    bones: Dict[str, Tuple[np.ndarray, np.ndarray]],
    joint_names: list[str],
) -> Dict[str, Any]:
    def pick(cands: list[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        for c in cands:
            v = bones.get(c)
            if v is not None:
                return v
        return None

    def set_joint(out: Dict[str, Any], name: str, cands: list[str], fallback: Optional[str] = None) -> None:
        v = pick(cands)
        if v is None and fallback is not None:
            v = out.get(fallback)
        if v is None:
            return
        pos, quat = v
        out[str(name)] = [np.asarray(pos, dtype=np.float32).reshape(3), _safe_quat_wxyz(np.asarray(quat, dtype=np.float32).reshape(4))]

    hips = ["hips", "hip", "root", "pelvis", "hiptracker"]
    spine = ["spine", "waist"]
    chest = ["chest"]
    upper_chest = ["upperchest"]
    neck = ["neck"]
    head = ["head"]
    r_shoulder = ["rightshoulder", "rightuppershoulder"]
    r_upper_arm = ["rightupperarm", "rightarm"]
    r_lower_arm = ["rightlowerarm", "rightforearm"]
    r_hand = ["righthand"]
    l_shoulder = ["leftshoulder", "leftuppershoulder"]
    l_upper_arm = ["leftupperarm", "leftarm"]
    l_lower_arm = ["leftlowerarm", "leftforearm"]
    l_hand = ["lefthand"]
    r_upper_leg = ["rightupperleg", "rightupleg", "rightthigh", "righthip"]
    r_lower_leg = ["rightlowerleg", "rightleg", "rightcalf"]
    r_foot = ["rightfoot", "rightankle"]
    r_toe = ["righttoe", "righttoes", "righttoeend"]
    l_upper_leg = ["leftupperleg", "leftupleg", "leftthigh", "lefthip"]
    l_lower_leg = ["leftlowerleg", "leftleg", "leftcalf"]
    l_foot = ["leftfoot", "leftankle"]
    l_toe = ["lefttoe", "lefttoes", "lefttoeend"]

    out: Dict[str, Any] = {}
    set_joint(out, "Hips", hips)
    set_joint(out, "Spine", spine + chest + hips, fallback="Hips")
    set_joint(out, "Spine1", chest + spine, fallback="Spine")
    set_joint(out, "Spine2", upper_chest + chest, fallback="Spine1")
    set_joint(out, "Spine3", upper_chest + chest + neck, fallback="Spine2")
    set_joint(out, "Neck", neck + upper_chest + chest, fallback="Spine3")
    set_joint(out, "Head", head + neck, fallback="Neck")
    set_joint(out, "RightShoulder", r_shoulder + upper_chest + chest, fallback="Spine3")
    set_joint(out, "RightUpperArm", r_upper_arm, fallback="RightShoulder")
    set_joint(out, "RightLowerArm", r_lower_arm, fallback="RightUpperArm")
    set_joint(out, "RightHand", r_hand, fallback="RightLowerArm")
    set_joint(out, "LeftShoulder", l_shoulder + upper_chest + chest, fallback="Spine3")
    set_joint(out, "LeftUpperArm", l_upper_arm, fallback="LeftShoulder")
    set_joint(out, "LeftLowerArm", l_lower_arm, fallback="LeftUpperArm")
    set_joint(out, "LeftHand", l_hand, fallback="LeftLowerArm")
    set_joint(out, "RightUpperLeg", r_upper_leg, fallback="Hips")
    set_joint(out, "RightLowerLeg", r_lower_leg, fallback="RightUpperLeg")
    set_joint(out, "RightFoot", r_foot, fallback="RightLowerLeg")
    set_joint(out, "RightToe", r_toe + r_foot, fallback="RightFoot")
    set_joint(out, "LeftUpperLeg", l_upper_leg, fallback="Hips")
    set_joint(out, "LeftLowerLeg", l_lower_leg, fallback="LeftUpperLeg")
    set_joint(out, "LeftFoot", l_foot, fallback="LeftLowerLeg")
    set_joint(out, "LeftToe", l_toe + l_foot, fallback="LeftFoot")

    if joint_names:
        ordered: Dict[str, Any] = {}
        for n in joint_names:
            if n in out:
                ordered[n] = out[n]
        for k, v in out.items():
            if k not in ordered:
                ordered[k] = v
        return ordered
    return out


class VmcReceiver:
    def __init__(self, ip: str, port: int) -> None:
        try:
            from pythonosc import dispatcher as osc_dispatcher  # type: ignore
            from pythonosc import osc_server  # type: ignore
        except Exception as e:
            raise RuntimeError(f"python-osc unavailable: {e}") from e

        self._lock = threading.Lock()
        self._bones: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._bones_raw_xyzw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._last_ts: float = 0.0
        self._seq: int = 0
        disp = osc_dispatcher.Dispatcher()
        disp.map("/VMC/Ext/Bone/Pos", self._on_bone_pos)
        self._server = osc_server.ThreadingOSCUDPServer((str(ip), int(port)), disp)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _on_bone_pos(self, _addr: str, name: str, px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float) -> None:
        ts = time.time()
        pos = np.array([float(px), float(py), float(pz)], dtype=np.float32)
        qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
        quat = _vmc_quat_xyzw_to_wxyz(qx, qy, qz, qw)
        quat_raw = np.array([qx, qy, qz, qw], dtype=np.float32)
        key = _normalize_vmc_name(name)
        with self._lock:
            self._bones[key] = (pos, quat)
            self._bones_raw_xyzw[key] = (pos, quat_raw)
            self._last_ts = float(ts)
            self._seq += 1

    def snapshot(self, max_age_s: float) -> Tuple[Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]], int, float]:
        with self._lock:
            now = time.time()
            if self._last_ts <= 0 or (now - float(self._last_ts)) > float(max_age_s):
                return None, -1, 0.0
            return dict(self._bones), int(self._seq), float(self._last_ts)

    def snapshot_raw_xyzw(self, max_age_s: float) -> Tuple[Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]], int, float]:
        with self._lock:
            now = time.time()
            if self._last_ts <= 0 or (now - float(self._last_ts)) > float(max_age_s):
                return None, -1, 0.0
            return dict(self._bones_raw_xyzw), int(self._seq), float(self._last_ts)

    def close(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass


def _parse_bone_axis_override(s: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not str(s).strip():
        return out
    for token in [x.strip() for x in str(s).split(";") if x.strip()]:
        if ":" not in token:
            continue
        bone, expr = token.split(":", 1)
        cfg = {"swap": "", "mirror_x": False, "mirror_y": False, "mirror_z": False}
        for term in [e.strip() for e in expr.split(",") if e.strip()]:
            if term.startswith("swap="):
                cfg["swap"] = term.split("=", 1)[1].strip().lower()
            elif term.startswith("flip="):
                flips = term.split("=", 1)[1].strip().lower()
                cfg["mirror_x"] = ("x" in flips)
                cfg["mirror_y"] = ("y" in flips)
                cfg["mirror_z"] = ("z" in flips)
        out[str(bone).strip()] = cfg
    return out


_DEFAULT_VIEWER_BONE_AXIS_OVERRIDE_STR = (
    "Hips:flip=yz;Spine:flip=yz;Spine1:flip=yz;Neck:flip=yz;Head:flip=yz;"
    "LeftUpperArm:flip=x;RightUpperArm:flip=x;LeftLowerArm:flip=x;RightLowerArm:flip=x;"
    "LeftHand:flip=x;RightHand:flip=x;LeftUpperLeg:flip=x;RightUpperLeg:flip=x;"
    "LeftLowerLeg:flip=x;RightLowerLeg:flip=x;LeftFoot:flip=x;RightFoot:flip=x"
)
_DEFAULT_VIEWER_BONE_AXIS_OVERRIDE = _parse_bone_axis_override(_DEFAULT_VIEWER_BONE_AXIS_OVERRIDE_STR)


def _build_bvh_to_vmc_map() -> Dict[str, str]:
    return {
        "HIP": "Hips",
        "Hips": "Hips",
        "WAIST": "Spine",
        "Spine": "Spine",
        "CHEST": "Chest",
        "Chest": "Chest",
        "UPPER_CHEST": "UpperChest",
        "UpperChest": "UpperChest",
        "NECK": "Neck",
        "Neck": "Neck",
        "HEAD": "Head",
        "Head": "Head",
        "LEFT_UPPER_SHOULDER": "LeftShoulder",
        "LEFT_SHOULDER": "LeftShoulder",
        "LeftShoulder": "LeftShoulder",
        "LEFT_UPPER_ARM": "LeftUpperArm",
        "LeftUpperArm": "LeftUpperArm",
        "LEFT_LOWER_ARM": "LeftLowerArm",
        "LeftLowerArm": "LeftLowerArm",
        "LEFT_HAND": "LeftHand",
        "LeftHand": "LeftHand",
        "RIGHT_UPPER_SHOULDER": "RightShoulder",
        "RIGHT_SHOULDER": "RightShoulder",
        "RightShoulder": "RightShoulder",
        "RIGHT_UPPER_ARM": "RightUpperArm",
        "RightUpperArm": "RightUpperArm",
        "RIGHT_LOWER_ARM": "RightLowerArm",
        "RightLowerArm": "RightLowerArm",
        "RIGHT_HAND": "RightHand",
        "RightHand": "RightHand",
        "LEFT_HIP": "Hips",
        "LEFT_UPPER_LEG": "LeftUpperLeg",
        "LeftUpperLeg": "LeftUpperLeg",
        "LEFT_LOWER_LEG": "LeftLowerLeg",
        "LeftLowerLeg": "LeftLowerLeg",
        "LEFT_FOOT": "LeftFoot",
        "LeftFoot": "LeftFoot",
        "RIGHT_HIP": "Hips",
        "RIGHT_UPPER_LEG": "RightUpperLeg",
        "RightUpperLeg": "RightUpperLeg",
        "RIGHT_LOWER_LEG": "RightLowerLeg",
        "RightLowerLeg": "RightLowerLeg",
        "RIGHT_FOOT": "RightFoot",
        "RightFoot": "RightFoot",
    }


class SlimevrBodyReader:
    def __init__(self, cfg: SlimevrBodyConfig) -> None:
        self.cfg = cfg
        self._receiver: Any = None
        self._viewer_fk: Any = None
        self._frame_index = -1
        self._bvh_to_vmc: Dict[str, str] = _build_bvh_to_vmc_map()
        self._fk_parents: Dict[str, Optional[str]] = {}
        self._fk_offsets: Dict[str, np.ndarray] = {}
        self._axis_m: Optional[np.ndarray] = None
        self._initialized = False

    def initialize(self) -> None:
        need_raw_receiver = not (bool(self.cfg.vmc_use_fk) and bool(self.cfg.vmc_use_viewer_fk))
        if need_raw_receiver:
            self._receiver = VmcReceiver(
                ip=str(self.cfg.vmc_ip),
                port=int(self.cfg.vmc_port),
            )
        else:
            self._receiver = None
        if bool(self.cfg.vmc_use_fk) and str(self.cfg.vmc_fk_skeleton).lower() == "bvh":
            self._fk_parents, self._fk_offsets = _parse_bvh_offsets(str(self.cfg.vmc_bvh_path))
            self._axis_m = _axis_swap_flip_matrix("xyz", "")
        if bool(self.cfg.vmc_use_viewer_fk):
            from deploy_real import vmc_fk_viewer as vmc_viewer  # type: ignore

            # skeleton topology/offsets from vmc_bvh_path (scaled by vmc_bvh_scale).
            if str(self.cfg.vmc_fk_skeleton).lower() == "bvh":
                bvh_parents, bvh_offsets = _parse_bvh_offsets(str(self.cfg.vmc_bvh_path))
                vmc_viewer.STD_SKELETON = {}
                for name, parent in bvh_parents.items():
                    off = bvh_offsets.get(name, np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    vmc_viewer.STD_SKELETON[name] = (
                        parent,
                        (
                            float(off[0]) * float(self.cfg.vmc_bvh_scale),
                            float(off[1]) * float(self.cfg.vmc_bvh_scale),
                            float(off[2]) * float(self.cfg.vmc_bvh_scale),
                        ),
                    )
            elif hasattr(vmc_viewer, "STD_SKELETON_BASE"):
                vmc_viewer.STD_SKELETON = dict(vmc_viewer.STD_SKELETON_BASE)

            self._viewer_fk = vmc_viewer.VMCFKReceiver(str(self.cfg.vmc_ip), int(self.cfg.vmc_port))
            # Rebuild name mapping/rotation cache to match the (possibly replaced) skeleton.
            self._viewer_fk.raw_rots = {}
            self._viewer_fk.name_map = {}
            for bone in vmc_viewer.STD_SKELETON:
                self._viewer_fk.raw_rots[bone] = np.array([0.0, 0.0, 0.0, 1.0])
            vmc_to_bvh = vmc_viewer._build_vmc_to_bvh_map(set(vmc_viewer.STD_SKELETON.keys()))
            for vmc_name, bvh_name in vmc_to_bvh.items():
                self._viewer_fk.name_map[vmc_viewer._normalize_name(vmc_name)] = bvh_name
            self._viewer_fk.rot_mode = str(self.cfg.vmc_rot_mode)
            # Keep quaternion convention aligned with legacy slimevr_teleop.sh (--vmc_no_invert_zw).
            self._viewer_fk.invert_vmc_zw = False
            # Keep viewer-axis corrections identical to the previous teleop default config.
            self._viewer_fk.bone_axis_override = {k: dict(v) for k, v in _DEFAULT_VIEWER_BONE_AXIS_OVERRIDE.items()}
            self._viewer_fk.start()
        self._initialized = True

    def _viewer_fk_pose(self) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        if self._viewer_fk is None:
            return None
        try:
            pos_map = self._viewer_fk.solve_fk()
            rot_map = getattr(self._viewer_fk, "computed_rot", {})
            out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for name, p in pos_map.items():
                r = rot_map.get(name, None)
                if r is None:
                    continue
                out[_normalize_vmc_name(name)] = (
                    np.asarray(p, dtype=np.float32).reshape(3),
                    _mat_to_quat_wxyz(np.asarray(r, dtype=np.float32).reshape(3, 3)),
                )
            return out if len(out) > 0 else None
        except Exception:
            return None

    def read_frame(self) -> Dict[str, Any]:
        if not self._initialized:
            return {"ok": False, "reason": "not_initialized"}

        vmc_pose = None
        seq = -1
        if self._viewer_fk is not None and bool(self.cfg.vmc_use_fk):
            vmc_pose = self._viewer_fk_pose()
            seq = int(time.time() * 1000)
        else:
            if self._receiver is None:
                return {"ok": False, "reason": "no_reader"}
            bones, seq, _ts = self._receiver.snapshot(float(self.cfg.vmc_timeout_s))
            if bones is None or seq < 0:
                return {"ok": False, "reason": "no_update"}
            if bool(self.cfg.vmc_use_fk):
                if str(self.cfg.vmc_fk_skeleton).lower() == "std":
                    raw_xyzw, _seq2, _ts2 = self._receiver.snapshot_raw_xyzw(float(self.cfg.vmc_timeout_s))
                    if raw_xyzw is None:
                        return {"ok": False, "reason": "no_update"}
                    vmc_pose = _build_fk_from_vmc_std(raw_xyzw, str(self.cfg.vmc_rot_mode))
                else:
                    fk_pos, fk_rot = _build_fk_from_vmc(
                        bones,
                        self._fk_parents,
                        self._fk_offsets,
                        self._bvh_to_vmc,
                        float(self.cfg.vmc_bvh_scale),
                        str(self.cfg.vmc_rot_mode),
                        None,
                        self._axis_m,
                    )
                    vmc_pose = _fk_to_vmc_pose(fk_pos, fk_rot, self._bvh_to_vmc)
            else:
                vmc_pose = bones

        if vmc_pose is None:
            return {"ok": False, "reason": "no_update"}
        if int(seq) == int(self._frame_index):
            return {"ok": False, "reason": "no_update"}
        self._frame_index = int(seq)
        body_frame = _vmc_build_body_frame(vmc_pose, [])
        return {"ok": True, "frame_index": int(self._frame_index), "body_frame": body_frame}

    def close(self) -> None:
        try:
            if self._receiver is not None:
                self._receiver.close()
        except Exception:
            pass
        self._receiver = None
        self._viewer_fk = None
        self._initialized = False

