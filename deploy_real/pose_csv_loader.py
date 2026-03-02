#!/usr/bin/env python3
"""
Utilities to read "FK-ed pose CSV" like 20251226_115314_chunk000_pose.csv.

CSV columns pattern (examples):
- body_Hips_px, body_Hips_py, body_Hips_pz, body_Hips_qw, body_Hips_qx, body_Hips_qy, body_Hips_qz
- rhand_RightIndexFinger3_px, ... _qz
- lhand_LeftIndexFinger3_px, ... _qz

This loader returns frames compatible with our BVH loader output:
    frame[joint_name] = [pos_xyz(np.float32, shape(3)), quat_wxyz(np.float32, shape(4))]
where pos is in meters, quat is scalar-first (wxyz).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set

import numpy as np
import re
import json


# BVH loader in replay_bvh_body_to_redis.py applies a fixed coordinate transform:
# rotation_matrix = [[1,0,0],[0,0,-1],[0,1,0]]
# rotation_quat   = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
# We expose the same transform here so "pose CSV in world coords" can be converted
# into the same convention as BVH loader output (GMR expects this convention).
BVH_GMR_ROT_M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
# Precomputed quaternion for BVH_GMR_ROT_M in scalar-first (wxyz):
# R.from_matrix(BVH_GMR_ROT_M).as_quat(scalar_first=True) == [0.70710678, 0.70710678, 0, 0]
BVH_GMR_ROT_Q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)

# Empirically best (and common) mapping for XDMocap WS_Geo -> GMR/nokov convention:
# (x,y,z) -> (y,-x,z)  == Rz(-90deg)
GEO_XYZ_TO_NOKOV_R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

# Fixed mapping we observed between some motionData CSV world and BVH raw world:
# (x,y,z) -> (-x, z, y)
CSV_POS_NEGX_Z_Y_M = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Hamilton product for quaternions in wxyz order.
    """
    aw, ax, ay, az = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bw, bx, by, bz = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n > 1e-8:
        return q / n
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_rotate_vec_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q (wxyz) without scipy.
    """
    q = quat_normalize_wxyz(q)
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    vpx = vx + qw * tx + (qy * tz - qz * ty)
    vpy = vy + qw * ty + (qz * tx - qx * tz)
    vpz = vz + qw * tz + (qx * ty - qy * tx)
    return np.array([vpx, vpy, vpz], dtype=np.float32)


def rotmat_to_quat_wxyz(Rm: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix (3x3) to quaternion (wxyz) without scipy.
    """
    Rm = np.asarray(Rm, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(Rm))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (Rm[2, 1] - Rm[1, 2]) / S
        qy = (Rm[0, 2] - Rm[2, 0]) / S
        qz = (Rm[1, 0] - Rm[0, 1]) / S
    else:
        if Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
            S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2.0
            qw = (Rm[2, 1] - Rm[1, 2]) / S
            qx = 0.25 * S
            qy = (Rm[0, 1] + Rm[1, 0]) / S
            qz = (Rm[0, 2] + Rm[2, 0]) / S
        elif Rm[1, 1] > Rm[2, 2]:
            S = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2.0
            qw = (Rm[0, 2] - Rm[2, 0]) / S
            qx = (Rm[0, 1] + Rm[1, 0]) / S
            qy = 0.25 * S
            qz = (Rm[1, 2] + Rm[2, 1]) / S
        else:
            S = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2.0
            qw = (Rm[1, 0] - Rm[0, 1]) / S
            qx = (Rm[0, 2] + Rm[2, 0]) / S
            qy = (Rm[1, 2] + Rm[2, 1]) / S
            qz = 0.25 * S
    return quat_normalize_wxyz(np.array([qw, qx, qy, qz], dtype=np.float32))


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (wxyz) to rotation matrix (3x3) without scipy.
    """
    q = quat_normalize_wxyz(np.asarray(q, dtype=np.float32).reshape(4))
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def apply_axis_basis_change_xyz_to_xzy(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basis change: Geo XYZ -> BVH XZY (swap Y and Z) for both position and orientation.

    For positions: p' = P p, where P swaps y/z:
        (x,y,z) -> (x,z,y)
    For rotations: R' = P R P^T  (change of basis)
    """
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        Rp = P @ p
        Rm = quat_wxyz_to_rotmat(q)
        Rm2 = P @ Rm @ P.T
        q2 = rotmat_to_quat_wxyz(Rm2)
        out[k] = [Rp, q2]
    return out


def apply_axis_basis_change_xyz_to_xzy_pos_only(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basis change for positions ONLY: Geo XYZ -> BVH XZY (swap Y/Z):
        (x,y,z) -> (x,z,y)
    Quaternions are kept unchanged.
    Useful if you are ignoring CSV quats via --csv_quat_mode identity/from_positions.
    """
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = np.asarray(v[1], dtype=np.float32).reshape(4)
        out[k] = [P @ p, q]
    return out


def apply_geo_xyz_to_nokov(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a fixed global rotation to map XDMocap WS_Geo XYZ into GMR/nokov convention:
        (x,y,z) -> (y,-x,z)
    Applies to BOTH position and quaternion (active rotation).
    """
    return apply_global_rotation(frame, GEO_XYZ_TO_NOKOV_R)


def apply_geo_to_bvh_official(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Official Geo -> BVH coordinate conversion (as provided by vendor support).

    Position:
      bvh = (-x, z, y)   (units preserved; their exporter may multiply by 100 for cm)

    Quaternion (global, wxyz):
      qw' = qw
      qx' = -qx
      qy' = qz
      qz' = qy

    Notes:
    - This is NOT a generic basis-change helper; it is the exact mapping in their code.
    - Output is intended to be BVH *raw world* convention (before our BVH->GMR fixed rotation).
    """
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        p2 = np.array([-p[0], p[2], p[1]], dtype=np.float32)
        q2 = np.array([q[0], -q[1], q[3], q[2]], dtype=np.float32)  # w, -x, z, y
        out[k] = [p2, quat_normalize_wxyz(q2)]
    return out


def apply_geo_to_bvh_official_pos_only(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same as apply_geo_to_bvh_official, but positions only (quat unchanged).
    Useful for debugging.
    """
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = np.asarray(v[1], dtype=np.float32).reshape(4)
        out[k] = [np.array([-p[0], p[2], p[1]], dtype=np.float32), q]
    return out


def apply_quat_left_multiply(frame: Dict[str, Any], qL_wxyz: np.ndarray) -> Dict[str, Any]:
    """
    Apply a left-multiply quaternion to ALL joint quaternions:
        q' = qL  q
    Positions are kept unchanged.

    Useful when positions are already aligned but quaternion convention needs a global fix.
    """
    qL = quat_normalize_wxyz(np.asarray(qL_wxyz, dtype=np.float32).reshape(4))
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        out[k] = [p, quat_mul_wxyz(qL, q)]
    return out


def apply_quat_left_multiply_per_joint(
    frame: Dict[str, Any],
    qL_map_wxyz: Dict[str, np.ndarray],
    default_qL_wxyz: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Apply a (possibly different) left-multiply quaternion per joint:
        q'_j = qL_map[j]  q_j
    If a joint is not in qL_map, use default_qL_wxyz if provided; otherwise keep unchanged.
    Positions are unchanged.
    """
    default_q = None
    if default_qL_wxyz is not None:
        default_q = quat_normalize_wxyz(np.asarray(default_qL_wxyz, dtype=np.float32).reshape(4))

    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        qL = qL_map_wxyz.get(k, None)
        if qL is None:
            if default_q is None:
                out[k] = [p, q]
            else:
                out[k] = [p, quat_mul_wxyz(default_q, q)]
        else:
            qL = quat_normalize_wxyz(np.asarray(qL, dtype=np.float32).reshape(4))
            out[k] = [p, quat_mul_wxyz(qL, q)]
    return out


def apply_quat_right_multiply(frame: Dict[str, Any], qR_wxyz: np.ndarray) -> Dict[str, Any]:
    """
    Right-multiply quaternion to ALL joint quaternions:
        q' = q  qR
    Positions unchanged.
    """
    qR = quat_normalize_wxyz(np.asarray(qR_wxyz, dtype=np.float32).reshape(4))
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        out[k] = [p, quat_mul_wxyz(q, qR)]
    return out


def apply_quat_right_multiply_per_joint(
    frame: Dict[str, Any],
    qR_map_wxyz: Dict[str, np.ndarray],
    default_qR_wxyz: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Apply a (possibly different) right-multiply quaternion per joint:
        q'_j = q_j  qR_map[j]
    If joint not in map, use default_qR_wxyz if provided; otherwise keep unchanged.
    Positions unchanged.
    """
    default_q = None
    if default_qR_wxyz is not None:
        default_q = quat_normalize_wxyz(np.asarray(default_qR_wxyz, dtype=np.float32).reshape(4))

    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        qR = qR_map_wxyz.get(k, None)
        if qR is None:
            if default_q is None:
                out[k] = [p, q]
            else:
                out[k] = [p, quat_mul_wxyz(q, default_q)]
        else:
            qR = quat_normalize_wxyz(np.asarray(qR, dtype=np.float32).reshape(4))
            out[k] = [p, quat_mul_wxyz(q, qR)]
    return out

def convert_quat_order(frame: Dict[str, Any], order: str) -> Dict[str, Any]:
    """
    order:
    - 'wxyz' (no-op)
    - 'xyzw' (convert to wxyz)
    """
    order = str(order).lower().strip()
    if order == "wxyz":
        return dict(frame)
    if order != "xyzw":
        raise ValueError("order must be 'wxyz' or 'xyzw'")
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        pos = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = np.asarray(v[1], dtype=np.float32).reshape(4)
        # xyzw -> wxyz
        q2 = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        out[k] = [pos, quat_normalize_wxyz(q2)]
    return out


def default_parent_map_body() -> Dict[str, Optional[str]]:
    """
    Parent map for the 23-body joint set used by the CSV (matches xdmocap demo naming).
    """
    p: Dict[str, Optional[str]] = {}
    p["Hips"] = None
    # right leg
    p["RightUpperLeg"] = "Hips"
    p["RightLowerLeg"] = "RightUpperLeg"
    p["RightFoot"] = "RightLowerLeg"
    p["RightToe"] = "RightFoot"
    # left leg
    p["LeftUpperLeg"] = "Hips"
    p["LeftLowerLeg"] = "LeftUpperLeg"
    p["LeftFoot"] = "LeftLowerLeg"
    p["LeftToe"] = "LeftFoot"
    # spine/head
    p["Spine"] = "Hips"
    p["Spine1"] = "Spine"
    p["Spine2"] = "Spine1"
    p["Spine3"] = "Spine2"
    p["Neck"] = "Spine3"
    p["Head"] = "Neck"
    # Arm chain
    p["RightShoulder"] = "Spine3"
    p["RightUpperArm"] = "RightShoulder"
    p["RightLowerArm"] = "RightUpperArm"
    p["RightHand"] = "RightLowerArm"
    p["LeftShoulder"] = "Spine3"
    p["LeftUpperArm"] = "LeftShoulder"
    p["LeftLowerArm"] = "LeftUpperArm"
    p["LeftHand"] = "LeftLowerArm"
    return p


def quats_local_to_global(frame: Dict[str, Any], parent_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Convert local quats to global quats using parent_map.
    Assumes positions are already global (we don't recompute pos).
    """
    out = dict(frame)
    # build a stable topo order by repeated relaxation (small graph => fine)
    unresolved = set([k for k in out.keys() if k in parent_map])
    global_q: Dict[str, np.ndarray] = {}

    # seed roots
    for j, pj in parent_map.items():
        if j in out and pj is None:
            global_q[j] = quat_normalize_wxyz(np.asarray(out[j][1], dtype=np.float32).reshape(4))
            unresolved.discard(j)

    changed = True
    while unresolved and changed:
        changed = False
        for j in list(unresolved):
            pj = parent_map.get(j, None)
            if pj is None:
                global_q[j] = quat_normalize_wxyz(np.asarray(out[j][1], dtype=np.float32).reshape(4))
                unresolved.discard(j)
                changed = True
                continue
            if pj in global_q:
                q_local = quat_normalize_wxyz(np.asarray(out[j][1], dtype=np.float32).reshape(4))
                global_q[j] = quat_mul_wxyz(global_q[pj], q_local)
                unresolved.discard(j)
                changed = True
    # write back for those we computed
    for j, qg in global_q.items():
        if j in out:
            out[j] = [np.asarray(out[j][0], dtype=np.float32).reshape(3), quat_normalize_wxyz(qg)]
    return out


def estimate_canonical_rotation_from_frame(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Estimate a world->canonical rotation where:
    - +Z is up (from Hips->Spine2)
    - +Y is left  (from LeftShoulder->RightShoulder gives right; we invert)
    - +X is forward (left x up)
    Returns R such that p_c = R @ p_w.
    """
    need = ["Hips", "Spine2", "LeftShoulder", "RightShoulder"]
    for k in need:
        if k not in frame:
            return None
    hips = np.asarray(frame["Hips"][0], dtype=np.float32).reshape(3)
    spine2 = np.asarray(frame["Spine2"][0], dtype=np.float32).reshape(3)
    ls = np.asarray(frame["LeftShoulder"][0], dtype=np.float32).reshape(3)
    rs = np.asarray(frame["RightShoulder"][0], dtype=np.float32).reshape(3)
    up = spine2 - hips
    nu = float(np.linalg.norm(up))
    if nu < 1e-6:
        return None
    up = up / nu
    right = rs - ls
    nr = float(np.linalg.norm(right))
    if nr < 1e-6:
        return None
    right = right / nr
    left = -right
    fwd = np.cross(left, up)
    nf = float(np.linalg.norm(fwd))
    if nf < 1e-6:
        return None
    fwd = fwd / nf
    # re-orthonormalize left
    left = np.cross(up, fwd)
    left = left / max(1e-6, float(np.linalg.norm(left)))
    R = np.stack([fwd, left, up], axis=0).astype(np.float32)  # rows: basis vectors
    return R


def apply_global_rotation(frame: Dict[str, Any], R: np.ndarray) -> Dict[str, Any]:
    """
    Apply p' = R @ p and q' = qR  q to every joint in a frame.
    """
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    qR = rotmat_to_quat_wxyz(R)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        out[k] = [R @ p, quat_mul_wxyz(qR, q)]
    return out

def apply_bvh_like_coordinate_transform(
    frame: Dict[str, Any],
    *,
    pos_unit: str = "m",
    apply_rotation: bool = True,
    rot_mode: str = "global",
    rot_tweak: str = "",
    rot_tweak_order: str = "post",
    apply_pos_rotation: bool = True,
    apply_quat_rotation: bool = True,
) -> Dict[str, Any]:
    """
    Convert a pose CSV frame to the same coordinate convention as our BVH loader output.

    - **pos_unit**: unit of CSV positions ('m'|'cm'|'mm')
    - **apply_rotation**: whether to apply BVH_GMR_ROT_M / BVH_GMR_ROT_Q

    Returns a NEW dict, does not mutate input.
    """
    unit = str(pos_unit).lower().strip()
    if unit == "m":
        s = 1.0
    elif unit == "cm":
        s = 0.01
    elif unit == "mm":
        s = 0.001
    else:
        raise ValueError(f"Invalid pos_unit: {pos_unit} (expected 'm'|'cm'|'mm')")

    out: Dict[str, Any] = {}
    rot_m = BVH_GMR_ROT_M
    rot_q = BVH_GMR_ROT_Q
    mode = str(rot_mode).lower().strip()
    if mode not in ["global", "basis"]:
        raise ValueError(f"Invalid rot_mode: {rot_mode} (expected global|basis)")
    tweak = str(rot_tweak).lower().strip()
    order = str(rot_tweak_order).lower().strip()
    if apply_rotation and tweak and tweak != "none":
        if tweak in ["rx180", "x180"]:
            extra_m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            extra_q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        elif tweak in ["ry180", "y180"]:
            extra_m = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
            extra_q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        elif tweak in ["rz180", "z180"]:
            extra_m = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
            extra_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            raise ValueError(f"Invalid rot_tweak: {rot_tweak} (expected none|rx180|ry180|rz180)")
        if order not in ["pre", "post"]:
            raise ValueError(f"Invalid rot_tweak_order: {rot_tweak_order} (expected pre|post)")
        if order == "pre":
            # effective rot_m = BVH * extra
            rot_m = rot_m @ extra_m
            rot_q = quat_mul_wxyz(rot_q, extra_q)
        else:
            # effective rot_m = extra * BVH
            rot_m = extra_m @ rot_m
            rot_q = quat_mul_wxyz(extra_q, rot_q)
    for name, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        pos = np.asarray(v[0], dtype=np.float32).reshape(3) * float(s)
        quat = np.asarray(v[1], dtype=np.float32).reshape(4)
        if apply_rotation:
            # Convert coords to BVH/GMR convention.
            # - positions: p' = rot_m @ p  (equivalently p @ rot_m.T for 1D)
            # - rotations:
            #   - global mode: q' = qR  q  (rotate the world frame)
            #   - basis  mode: q' = qR  q  qR^{-1} (change of basis, keeps physical orientation)
            if bool(apply_pos_rotation):
                pos = pos @ rot_m.T
            if bool(apply_quat_rotation):
                if mode == "global":
                    quat = quat_mul_wxyz(rot_q, quat)
                else:
                    quat = quat_mul_wxyz(quat_mul_wxyz(rot_q, quat), quat_conj_wxyz(rot_q))
        out[name] = [pos, quat]
    return out


def _make_basis_from_x_and_up(x_axis: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """
    Build a right-handed rotation matrix whose +X aligns with x_axis.
    up_hint is used to resolve roll; if nearly colinear, we fall back.
    Returns R (3x3) with columns [x,y,z] in world frame.
    """
    x = np.asarray(x_axis, dtype=np.float32).reshape(3)
    nx = float(np.linalg.norm(x))
    if nx < 1e-6:
        return np.eye(3, dtype=np.float32)
    x = x / nx

    up = np.asarray(up_hint, dtype=np.float32).reshape(3)
    nu = float(np.linalg.norm(up))
    if nu < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up = up / nu

    # z is orthogonal to x but close to up
    z = up - x * float(np.dot(up, x))
    nz = float(np.linalg.norm(z))
    if nz < 1e-6:
        # pick an arbitrary up that isn't colinear with x
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, x))) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        z = fallback - x * float(np.dot(fallback, x))
        nz = float(np.linalg.norm(z))
        if nz < 1e-6:
            return np.eye(3, dtype=np.float32)
    z = z / nz

    y = np.cross(z, x)
    ny = float(np.linalg.norm(y))
    if ny < 1e-6:
        return np.eye(3, dtype=np.float32)
    y = y / ny

    R = np.stack([x, y, z], axis=1).astype(np.float32)  # columns
    return R


def set_all_quats_identity(frame: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        out[k] = [p, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
    return out


def synthesize_gmr_body_quats_from_positions(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    For GMR body retargeting, quats are heavily constrained in IK configs.
    If the incoming quats are from sensors / a different bone-frame convention,
    IK can look obviously wrong. This function rebuilds *global* quats so that
    each segment's +X axis roughly points along the bone direction.

    Expected names are already in GMR(bvh_*) convention (after gmr_rename_and_footmod),
    e.g. Hips, Spine2, LeftUpLeg, LeftLeg, LeftFootMod, LeftArm, LeftForeArm, LeftHand, etc.
    """
    out = dict(frame)

    def has(k: str) -> bool:
        return k in out and isinstance(out[k], (list, tuple)) and len(out[k]) >= 2

    def pos(k: str) -> np.ndarray:
        return np.asarray(out[k][0], dtype=np.float32).reshape(3)

    # global up hint
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if has("Hips") and has("Spine2"):
        v = pos("Spine2") - pos("Hips")
        if float(np.linalg.norm(v)) > 1e-6:
            up_hint = v / float(np.linalg.norm(v))

    # pelvis forward hint via hips left-right
    fwd_hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if has("LeftUpLeg") and has("RightUpLeg") and has("Hips"):
        left = pos("LeftUpLeg") - pos("RightUpLeg")
        nl = float(np.linalg.norm(left))
        if nl > 1e-6:
            left = left / nl
            fwd = np.cross(left, up_hint)
            nf = float(np.linalg.norm(fwd))
            if nf > 1e-6:
                fwd_hint = fwd / nf

    # helper: set quat for joint A using direction A->B
    def set_quat_from_dir(a: str, b: str, up: np.ndarray):
        if not (has(a) and has(b)):
            return
        d = pos(b) - pos(a)
        Rm = _make_basis_from_x_and_up(d, up)
        out[a] = [pos(a), rotmat_to_quat_wxyz(Rm)]

    # torso
    set_quat_from_dir("Hips", "Spine2", up_hint)
    if has("Spine2") and has("Hips"):
        set_quat_from_dir("Spine2", "Hips", up_hint)  # fallback, still defines a basis

    # Leg chain
    set_quat_from_dir("LeftUpLeg", "LeftLeg", up_hint)
    set_quat_from_dir("LeftLeg", "LeftFoot", up_hint)
    set_quat_from_dir("LeftFootMod", "LeftToe", up_hint)
    set_quat_from_dir("RightUpLeg", "RightLeg", up_hint)
    set_quat_from_dir("RightLeg", "RightFoot", up_hint)
    set_quat_from_dir("RightFootMod", "RightToe", up_hint)

    # arms: use pelvis forward as a stable up hint (prevents roll flips when arm aligns with global up)
    arm_up = np.cross(fwd_hint, up_hint)
    if float(np.linalg.norm(arm_up)) < 1e-6:
        arm_up = up_hint
    set_quat_from_dir("LeftArm", "LeftForeArm", arm_up)
    set_quat_from_dir("LeftForeArm", "LeftHand", arm_up)
    set_quat_from_dir("LeftHand", "LeftHand", arm_up)  # no-op basis if needed
    set_quat_from_dir("RightArm", "RightForeArm", arm_up)
    set_quat_from_dir("RightForeArm", "RightHand", arm_up)
    set_quat_from_dir("RightHand", "RightHand", arm_up)

    # for any missing quat keys, keep original (positions-only pipelines might still rely on them)
    return out


def apply_pos_matrix(frame: Dict[str, Any], M: np.ndarray) -> Dict[str, Any]:
    """
    Apply position transform p' = M @ p to all joints in a frame.
    Quaternions are kept as-is (use quat_mode=from_positions to rebuild quats if needed).
    """
    M = np.asarray(M, dtype=np.float32).reshape(3, 3)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = np.asarray(v[1], dtype=np.float32).reshape(4)
        out[k] = [M @ p, q]
    return out


def _as_float_matrix3x3(x: Any) -> np.ndarray:
    M = np.asarray(x, dtype=np.float32).reshape(3, 3)
    return M


def _as_float_vec3(x: Any) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(3)
    return v


def _as_float_quat_wxyz(x: Any) -> np.ndarray:
    q = np.asarray(x, dtype=np.float32).reshape(4)
    return quat_normalize_wxyz(q)


def load_csv_calib_json(calib_json_path: str) -> Dict[str, Any]:
    """
    Load a calibration json used to convert CSV poses into the same convention as BVH->GMR pipeline.
    This is intended to avoid requiring a BVH file at replay time.

    Expected schema (minimal):
    {
      "version": 1,
      "pos": { "matrix": [[...],[...],[...]] , "units": "m", "apply_bvh_like_rotation": true },
      "quat_fix": { "mode": "per_joint", "side": "right", "map_wxyz": { "Hips": [w,x,y,z], ... } }
    }
    """
    with open(calib_json_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    if not isinstance(calib, dict):
        raise ValueError("calib json must be an object")
    ver = int(calib.get("version", 1))
    if ver != 1:
        raise ValueError(f"unsupported calib version: {ver}")
    return calib


def apply_csv_calib_to_frame(
    frame: Dict[str, Any],
    calib: Dict[str, Any],
    *,
    fmt: Optional[str] = None,
    recompute_footmod: bool = False,
) -> Dict[str, Any]:
    """
    Apply a loaded calibration dict to a single frame.

    Order:
    - optional pos.matrix (positions only)
    - optional pos.apply_bvh_like_rotation (applies BVH_GMR_ROT to pos+quat, and units scaling)
    - optional quat_fix (left/right multiply, per_joint)
    - optional recompute FootMod (if fmt provided and recompute_footmod=True)
    """
    out = dict(frame)

    pos_cfg = calib.get("pos", {}) if isinstance(calib.get("pos", {}), dict) else {}
    units = str(pos_cfg.get("units", "m")).lower().strip()
    apply_bvh_like = bool(pos_cfg.get("apply_bvh_like_rotation", False))
    # Option A: matrix (positions only)
    if "matrix" in pos_cfg and pos_cfg["matrix"] is not None:
        M = _as_float_matrix3x3(pos_cfg["matrix"])
        out = apply_pos_matrix(out, M)
    # Option B: similarity (affects pos + quat)
    elif ("R" in pos_cfg) and ("t" in pos_cfg):
        Rm = _as_float_matrix3x3(pos_cfg["R"])
        t = _as_float_vec3(pos_cfg["t"])
        s = float(pos_cfg.get("s", 1.0))
        out = apply_similarity_transform_frame(out, s, Rm, t)

    if apply_bvh_like or units != "m":
        out = apply_bvh_like_coordinate_transform(out, pos_unit=units, apply_rotation=apply_bvh_like)

    quat_cfg = calib.get("quat_fix", {}) if isinstance(calib.get("quat_fix", {}), dict) else {}
    mode = str(quat_cfg.get("mode", "")).strip().lower()
    side = str(quat_cfg.get("side", "right")).strip().lower()
    qmap = quat_cfg.get("map_wxyz", {})
    if mode:
        if mode != "per_joint":
            raise ValueError(f"calib.quat_fix.mode currently only supports 'per_joint', got: {mode}")
        if side not in ("left", "right"):
            raise ValueError(f"calib.quat_fix.side must be 'left' or 'right', got: {side}")
        if not isinstance(qmap, dict) or not qmap:
            raise ValueError("calib.quat_fix.map_wxyz must be a non-empty dict")
        qfix_map = {str(k): _as_float_quat_wxyz(v) for k, v in qmap.items()}
        if side == "left":
            out = apply_quat_left_multiply_per_joint(out, qfix_map, default_qL_wxyz=None)
        else:
            out = apply_quat_right_multiply_per_joint(out, qfix_map, default_qR_wxyz=None)

    if recompute_footmod:
        if fmt is None:
            raise ValueError("fmt must be provided when recompute_footmod=True")
        out = gmr_rename_and_footmod(out, fmt=fmt)

    return out


def apply_csv_calib_to_frames(
    frames: List[Dict[str, Any]],
    calib: Dict[str, Any],
    *,
    fmt: Optional[str] = None,
    recompute_footmod: bool = False,
) -> List[Dict[str, Any]]:
    return [
        apply_csv_calib_to_frame(fr, calib, fmt=fmt, recompute_footmod=recompute_footmod)
        for fr in frames
    ]


def signed_permutation_matrices(allow_reflection: bool = True) -> List[np.ndarray]:
    """
    Generate all signed permutation matrices M (entries in {-1,0,1}, orthonormal),
    representing axis swaps and sign flips. If allow_reflection=False, only det=+1.
    """
    import itertools

    mats: List[np.ndarray] = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            M = np.zeros((3, 3), dtype=np.float32)
            for i, ax in enumerate(perm):
                M[i, ax] = float(signs[i])
            if not allow_reflection:
                if float(np.linalg.det(M)) < 0.0:
                    continue
            mats.append(M)
    return mats


def umeyama_similarity_transform(
    X: np.ndarray,
    Y: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate similarity transform (s, R, t) mapping X -> Y:
        Y ~= s * (R @ X) + t
    X, Y: (N,3)
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 3)
    if X.shape[0] != Y.shape[0] or X.shape[0] < 3:
        raise ValueError(f"need N>=3 correspondences, got X={X.shape}, Y={Y.shape}")

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    Sigma = (Yc.T @ Xc) / float(X.shape[0])
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    Rm = U @ S @ Vt

    if with_scale:
        var_x = float((Xc**2).sum() / float(X.shape[0]))
        s = 1.0 if var_x < 1e-12 else float(np.trace(np.diag(D) @ S) / var_x)
    else:
        s = 1.0

    t = mu_y - s * (Rm @ mu_x)
    return float(s), Rm.astype(np.float32), t.astype(np.float32)


def apply_similarity_transform_frame(
    frame: Dict[str, Any],
    s: float,
    Rm: np.ndarray,
    t: np.ndarray,
) -> Dict[str, Any]:
    """
    Apply similarity transform to a frame:
      p' = s * (R p) + t
      q' = qR  q
    """
    Rm = np.asarray(Rm, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t, dtype=np.float32).reshape(3)
    qR = rotmat_to_quat_wxyz(Rm)
    out: Dict[str, Any] = {}
    for k, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        p = np.asarray(v[0], dtype=np.float32).reshape(3)
        q = quat_normalize_wxyz(np.asarray(v[1], dtype=np.float32).reshape(4))
        out[k] = [float(s) * (Rm @ p) + t, quat_mul_wxyz(qR, q)]
    return out


def xdmocap_demo_initial_tpose_body_geo_frame() -> Dict[str, Any]:
    """
    Initial T-pose joint positions from vdmocap/DataRead_Python_Demo/main.py.
    WorldSpace: WS_Geo, units: meters.
    Returns a frame: joint_name -> [pos(3), quat(4=wxyz identity)].
    """
    names = [
        "Hips",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "RightToe",
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
        "LeftToe",
        "Spine",
        "Spine1",
        "Spine2",
        "Spine3",
        "Neck",
        "Head",
        "RightShoulder",
        "RightUpperArm",
        "RightLowerArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftLowerArm",
        "LeftHand",
    ]
    vals = [
        [0.0, 0.0, 1.022],
        [0.074, 0.0, 1.002],
        [0.097, 0.0, 0.593],
        [0.104, 0.0, 0.111],
        [0.114, 0.159, 0.005],
        [-0.074, 0.0, 1.002],
        [-0.097, 0.001, 0.593],
        [-0.104, 0.0, 0.111],
        [-0.114, 0.158, 0.004],
        [0.0, 0.033, 1.123],
        [0.0, 0.03, 1.246],
        [0.0, 0.014, 1.362],
        [0.0, -0.048, 1.475],
        [0.0, -0.048, 1.549],
        [0.0, -0.016, 1.682],
        [0.071, -0.061, 1.526],
        [0.178, -0.061, 1.526],
        [0.421, -0.061, 1.526],
        [0.682, -0.061, 1.526],
        [-0.071, -0.061, 1.526],
        [-0.178, -0.061, 1.526],
        [-0.421, -0.061, 1.526],
        [-0.682, -0.061, 1.526],
    ]
    qI = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out: Dict[str, Any] = {}
    for n, v in zip(names, vals):
        out[n] = [np.asarray(v, dtype=np.float32), qI.copy()]
    return out
@dataclass(frozen=True)
class PoseCSVMeta:
    fieldnames: List[str]
    fps: Optional[float] = None


def _infer_joint_names(fieldnames: Iterable[str], prefix: str) -> List[str]:
    """
    prefix: "body_", "lhand_", "rhand_"
    """
    names: Set[str] = set()
    for k in fieldnames:
        if not k.startswith(prefix):
            continue
        # expect: <prefix><JointName>_<px|py|pz|qw|qx|qy|qz>
        rest = k[len(prefix) :]
        if "_" not in rest:
            continue
        joint, suffix = rest.rsplit("_", 1)
        if suffix in ["px", "py", "pz", "qw", "qx", "qy", "qz"]:
            names.add(joint)
    return sorted(names)


def _infer_joint_names_motiondata(fieldnames: Iterable[str]) -> List[str]:
    """
    Infer joint names from "motionData_*.csv" style columns, e.g.:
      - "Hips position X(m)"
      - "Hips quaternion W"
    """
    names: Set[str] = set()
    for k in fieldnames:
        m = re.match(r"^(.*)\s+position\s+[XYZ]\(m\)$", str(k).strip())
        if m:
            names.add(m.group(1).strip())
            continue
        m = re.match(r"^(.*)\s+quaternion\s+[WXYZ]$", str(k).strip())
        if m:
            names.add(m.group(1).strip())
            continue
    return sorted(names)


def _is_finger_joint_name(joint: str) -> bool:
    j = str(joint)
    # Finger joints: *Finger*, ThumbFinger*
    return ("Finger" in j) or ("ThumbFinger" in j)


def _is_hand_base_joint_name(joint: str) -> bool:
    return str(joint) in ("LeftHand", "RightHand")


def _read_joint_pose_motiondata(
    row: Dict[str, str],
    joint: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one joint pose from motionData-style row.
    Positions are meters, quaternions are wxyz.
    """
    pos = np.array(
        [
            _get_float(row, f"{joint} position X(m)"),
            _get_float(row, f"{joint} position Y(m)"),
            _get_float(row, f"{joint} position Z(m)"),
        ],
        dtype=np.float32,
    )
    quat = np.array(
        [
            _get_float(row, f"{joint} quaternion W", 1.0),
            _get_float(row, f"{joint} quaternion X", 0.0),
            _get_float(row, f"{joint} quaternion Y", 0.0),
            _get_float(row, f"{joint} quaternion Z", 0.0),
        ],
        dtype=np.float32,
    )
    return pos, quat


def _get_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _read_joint_pose(
    row: Dict[str, str],
    prefix: str,
    joint: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array(
        [
            _get_float(row, f"{prefix}{joint}_px"),
            _get_float(row, f"{prefix}{joint}_py"),
            _get_float(row, f"{prefix}{joint}_pz"),
        ],
        dtype=np.float32,
    )
    quat = np.array(
        [
            _get_float(row, f"{prefix}{joint}_qw", 1.0),
            _get_float(row, f"{prefix}{joint}_qx", 0.0),
            _get_float(row, f"{prefix}{joint}_qy", 0.0),
            _get_float(row, f"{prefix}{joint}_qz", 0.0),
        ],
        dtype=np.float32,
    )
    return pos, quat


def load_pose_csv_frames(
    csv_path: str,
    include_body: bool = True,
    include_lhand: bool = True,
    include_rhand: bool = True,
    max_frames: int = -1,
) -> Tuple[List[Dict[str, Any]], PoseCSVMeta]:
    """
    Return (frames, meta).
    Each frame is a dict: joint_name -> [pos(3), quat(4)].
    Joint names are taken directly from CSV (e.g. "Hips", "RightIndexFinger3", "LeftIndexFinger3").
    """
    frames: List[Dict[str, Any]] = []
    fps: Optional[float] = None
    dt_ms: List[float] = []
    last_t_ms: Optional[float] = None

    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header")
        fieldnames = list(reader.fieldnames)

        has_prefix_format = any(k.startswith("body_") or k.startswith("lhand_") or k.startswith("rhand_") for k in fieldnames)
        has_motiondata_format = ("Hips position X(m)" in fieldnames) or any(" position X(m)" in k for k in fieldnames)

        if has_prefix_format:
            body_joints = _infer_joint_names(fieldnames, "body_") if include_body else []
            lhand_joints = _infer_joint_names(fieldnames, "lhand_") if include_lhand else []
            rhand_joints = _infer_joint_names(fieldnames, "rhand_") if include_rhand else []
            motiondata_joints: List[str] = []
        elif has_motiondata_format:
            # All joints are in a flat namespace (no body_/lhand_/rhand_ prefixes)
            motiondata_joints = _infer_joint_names_motiondata(fieldnames)
            body_joints, lhand_joints, rhand_joints = [], [], []
        else:
            # Unknown format: keep behavior as "no joints found" (caller will see empty keys)
            motiondata_joints = []
            body_joints, lhand_joints, rhand_joints = [], [], []

        for idx, row in enumerate(reader):
            if max_frames >= 0 and idx >= int(max_frames):
                break
            if fps is None:
                # optional, depends on column existence
                try:
                    fps = float(row.get("frequency", "")) if row.get("frequency", "") else None
                except Exception:
                    fps = None

            # motionData format may have only time(ms); estimate fps from deltas
            if has_motiondata_format and fps is None:
                if "time(ms)" in row:
                    try:
                        t_ms = float(row.get("time(ms)", ""))
                        if last_t_ms is not None:
                            dt = t_ms - last_t_ms
                            if dt > 0:
                                dt_ms.append(dt)
                        last_t_ms = t_ms
                    except Exception:
                        pass

            fr: Dict[str, Any] = {}
            if has_prefix_format:
                for j in body_joints:
                    pos, quat = _read_joint_pose(row, "body_", j)
                    fr[j] = [pos, quat]
                for j in lhand_joints:
                    pos, quat = _read_joint_pose(row, "lhand_", j)
                    fr[j] = [pos, quat]
                for j in rhand_joints:
                    pos, quat = _read_joint_pose(row, "rhand_", j)
                    fr[j] = [pos, quat]
            elif has_motiondata_format:
                for j in motiondata_joints:
                    is_left = str(j).startswith("Left")
                    is_right = str(j).startswith("Right")
                    is_finger = _is_finger_joint_name(j)
                    is_hand_base = _is_hand_base_joint_name(j)

                    if is_finger:
                        # Fingers are only included when the corresponding hand stream is enabled.
                        if is_left and not include_lhand:
                            continue
                        if is_right and not include_rhand:
                            continue
                        # Non-sided finger name (unlikely) requires at least one hand enabled.
                        if (not is_left and not is_right) and (not include_lhand and not include_rhand):
                            continue
                    elif is_hand_base:
                        # LeftHand/RightHand are needed by both body retargeting and hand retargeting.
                        if include_body:
                            pass
                        else:
                            if is_left and not include_lhand:
                                continue
                            if is_right and not include_rhand:
                                continue
                    else:
                        # Other joints are treated as body joints.
                        if not include_body:
                            continue
                    pos, quat = _read_joint_pose_motiondata(row, j)
                    fr[j] = [pos, quat]

            frames.append(fr)

    # Use a small threshold so even short clips can infer fps.
    if fps is None and len(dt_ms) >= 1:
        med = float(np.median(np.asarray(dt_ms, dtype=np.float64)))
        if med > 1e-6:
            fps = 1000.0 / med

    meta = PoseCSVMeta(fieldnames=fieldnames, fps=fps)
    return frames, meta


def gmr_rename_and_footmod(frame: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    """
    Apply the same rename aliases and FootMod construction used by our BVH loader,
    but on a frame already containing global pos/quat.
    """
    out = dict(frame)

    rename = {
        "LeftUpperLeg": "LeftUpLeg",
        "RightUpperLeg": "RightUpLeg",
        "LeftLowerLeg": "LeftLeg",
        "RightLowerLeg": "RightLeg",
        "LeftUpperArm": "LeftArm",
        "RightUpperArm": "RightArm",
        "LeftLowerArm": "LeftForeArm",
        "RightLowerArm": "RightForeArm",
    }
    for src, dst in rename.items():
        if src in out and dst not in out:
            out[dst] = out[src]

    # Toe / ToeBase aliases
    if "LeftToe" in out and "LeftToeBase" not in out:
        out["LeftToeBase"] = out["LeftToe"]
    if "RightToe" in out and "RightToeBase" not in out:
        out["RightToeBase"] = out["RightToe"]
    if "LeftToeBase" in out and "LeftToe" not in out:
        out["LeftToe"] = out["LeftToeBase"]
    if "RightToeBase" in out and "RightToe" not in out:
        out["RightToe"] = out["RightToeBase"]

    # FootMod selection: position from Foot, orientation from Toe or ToeBase
    if fmt == "lafan1":
        left_toe = "LeftToe" if "LeftToe" in out else "LeftToeBase"
        right_toe = "RightToe" if "RightToe" in out else "RightToeBase"
    elif fmt == "nokov":
        left_toe = "LeftToeBase" if "LeftToeBase" in out else "LeftToe"
        right_toe = "RightToeBase" if "RightToeBase" in out else "RightToe"
    else:
        raise ValueError(f"Invalid format: {fmt}")

    if "LeftFoot" in out and left_toe in out:
        out["LeftFootMod"] = [out["LeftFoot"][0], out[left_toe][1]]
    if "RightFoot" in out and right_toe in out:
        out["RightFootMod"] = [out["RightFoot"][0], out[right_toe][1]]

    return out


