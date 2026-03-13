#!/usr/bin/env python3
"""
VMC Forward Kinematics Viewer
-----------------------------
Use a built-in standard skeleton instead of SlimeVR VRM/bone length metadata.
Combine SlimeVR rotations with predefined bone lengths to reconstruct full-body motion.
"""
import argparse
import math
import time
import threading
import numpy as np
import os
from pythonosc import dispatcher, osc_server

# ==========================================
# 1. Define the standard skeleton (unit: meters).
# This is a generic human T-pose structure.
# ==========================================
# Format: { "bone_name": (parent_bone, (x_offset, y_offset, z_offset)) }
# Offset means child position relative to parent when parent rotation is identity.
STD_SKELETON = {
    "Hips":           (None,            (0.0, 0.0, 0.0)),  # Root node
    
    # Spine chain upward
    "Spine":          ("Hips",          (0.0,  0.10, 0.0)),
    "Chest":          ("Spine",         (0.0,  0.15, 0.0)),
    "UpperChest":     ("Chest",         (0.0,  0.15, 0.0)),
    "Neck":           ("UpperChest",    (0.0,  0.15, 0.0)),
    "Head":           ("Neck",          (0.0,  0.10, 0.0)),

    # Legs (downward from Hips)
    "LeftUpperLeg":   ("Hips",          (-0.08, -0.05, 0.0)),  # Slight left/down offset
    "LeftLowerLeg":   ("LeftUpperLeg",  (0.0,  -0.42, 0.0)),
    "LeftFoot":       ("LeftLowerLeg",  (0.0,  -0.40, 0.0)),
    
    "RightUpperLeg":  ("Hips",          (0.08, -0.05, 0.0)),
    "RightLowerLeg":  ("RightUpperLeg", (0.0,  -0.42, 0.0)),
    "RightFoot":      ("RightLowerLeg", (0.0,  -0.40, 0.0)),

    # Arms (extend from UpperChest to both sides)
    "LeftShoulder":   ("UpperChest",    (-0.10, 0.10, 0.0)),
    "LeftUpperArm":   ("LeftShoulder",  (-0.12, 0.0,  0.0)),  # Left in T-pose
    "LeftLowerArm":   ("LeftUpperArm",  (-0.28, 0.0,  0.0)),
    "LeftHand":       ("LeftLowerArm",  (-0.25, 0.0,  0.0)),

    "RightShoulder":  ("UpperChest",    (0.10, 0.10, 0.0)),
    "RightUpperArm":  ("RightShoulder", (0.12, 0.0,  0.0)),
    "RightLowerArm":  ("RightUpperArm", (0.28, 0.0,  0.0)),
    "RightHand":      ("RightLowerArm", (0.25, 0.0,  0.0)),
}

# Backup of standard skeleton for BVH axis alignment.
STD_SKELETON_BASE = dict(STD_SKELETON)

# Line segments used for visualization.
DRAW_LINES = [
    ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "UpperChest"),
    ("UpperChest", "Neck"), ("Neck", "Head"),
    ("UpperChest", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"), ("LeftUpperArm", "LeftLowerArm"), ("LeftLowerArm", "LeftHand"),
    ("UpperChest", "RightShoulder"), ("RightShoulder", "RightUpperArm"), ("RightUpperArm", "RightLowerArm"), ("RightLowerArm", "RightHand"),
    ("Hips", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"), ("LeftLowerLeg", "LeftFoot"),
    ("Hips", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"), ("RightLowerLeg", "RightFoot")
]

# Quaternion rotation helpers.
def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([x, y, z, w])

def q_rot_vec(q, v):
    # Rotate vector v by quaternion q.
    x, y, z = v
    qv = np.array([x, y, z, 0.0])
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    return q_mult(q_mult(q, qv), q_conj)[:3]


def _normalize_name(name: str) -> str:
    return "".join([c.lower() for c in str(name) if c.isalnum()])


def _quat_to_mat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        q = q / n
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _basis_matrix(cfg: dict) -> np.ndarray:
    swap = cfg.get("swap", "xyz")
    axes = {"x": 0, "y": 1, "z": 2}
    if len(str(swap)) != 3 or any(c not in axes for c in str(swap)):
        swap = "xyz"
    idx = [axes[c] for c in str(swap)]
    B = np.eye(3, dtype=float)[:, idx]
    f = np.ones(3, dtype=float)
    if cfg.get("mirror_x", False):
        f[0] *= -1.0
    if cfg.get("mirror_y", False):
        f[1] *= -1.0
    if cfg.get("mirror_z", False):
        f[2] *= -1.0
    B = B * f.reshape(1, 3)
    return B

def _rot_from_euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    rx = math.radians(float(rx))
    ry = math.radians(float(ry))
    rz = math.radians(float(rz))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0],
                     [0, cx, -sx],
                     [0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0, sy],
                     [0, 1, 0],
                     [-sy, 0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0],
                     [sz, cz, 0],
                     [0, 0, 1]], dtype=float)
    return rz_m @ ry_m @ rx_m

def _rot_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.eye(3, dtype=float)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 0.999999:
        return np.eye(3, dtype=float)
    if c < -0.999999:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-8)
        vx, vy, vz = v
        K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=float)
        return np.eye(3, dtype=float) + 2.0 * (K @ K)
    s = np.linalg.norm(v)
    vx, vy, vz = v / (s + 1e-8)
    K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=float)
    R = np.eye(3, dtype=float) + K + K @ K * ((1 - c) / (s * s + 1e-8))
    return R


def _parse_bvh_offsets(path: str):
    parents = {}
    offsets = {}
    stack = []
    current = None
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
                    offsets[current] = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=float)
            elif head == "}":
                if stack:
                    top = stack.pop()
                    if top is None:
                        in_end_site = False
                current = stack[-1] if stack else None
            elif head == "MOTION":
                break
    return parents, offsets


def _build_vmc_to_bvh_map(bvh_joints: set) -> dict:
    def pick(*cands):
        for c in cands:
            if c in bvh_joints:
                return c
        return None

    mapping = {
        "Hips": pick("HIP", "Hips", "HipsRoot"),
        "Spine": pick("WAIST", "Spine", "SPINE"),
        "Chest": pick("CHEST", "Chest", "SPINE2"),
        "UpperChest": pick("UPPER_CHEST", "UpperChest", "SPINE3"),
        "Neck": pick("NECK", "Neck"),
        "Head": pick("HEAD", "Head"),
        "LeftUpperLeg": pick("LEFT_UPPER_LEG", "LeftUpperLeg", "LeftUpLeg"),
        "LeftLowerLeg": pick("LEFT_LOWER_LEG", "LeftLowerLeg", "LeftLeg"),
        "LeftFoot": pick("LEFT_FOOT", "LeftFoot"),
        "RightUpperLeg": pick("RIGHT_UPPER_LEG", "RightUpperLeg", "RightUpLeg"),
        "RightLowerLeg": pick("RIGHT_LOWER_LEG", "RightLowerLeg", "RightLeg"),
        "RightFoot": pick("RIGHT_FOOT", "RightFoot"),
        "LeftShoulder": pick("LEFT_SHOULDER", "LEFT_UPPER_SHOULDER", "LeftShoulder"),
        "LeftUpperArm": pick("LEFT_UPPER_ARM", "LeftUpperArm", "LeftArm"),
        "LeftLowerArm": pick("LEFT_LOWER_ARM", "LeftLowerArm", "LeftForeArm"),
        "LeftHand": pick("LEFT_HAND", "LeftHand"),
        "RightShoulder": pick("RIGHT_SHOULDER", "RIGHT_UPPER_SHOULDER", "RightShoulder"),
        "RightUpperArm": pick("RIGHT_UPPER_ARM", "RightUpperArm", "RightArm"),
        "RightLowerArm": pick("RIGHT_LOWER_ARM", "RightLowerArm", "RightForeArm"),
        "RightHand": pick("RIGHT_HAND", "RightHand"),
    }
    return {k: v for k, v in mapping.items() if v is not None}

class VMCFKReceiver:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.raw_rots = {}  # Raw local rotations: {name: [x, y, z, w]}
        self.computed_pos = {}  # Computed global positions
        self.computed_rot = {}  # Computed global rotation matrices
        self._lock = threading.Lock()
        self.name_map = {}
        self.axis_cfg = {"swap": "", "mirror_x": False, "mirror_y": False, "mirror_z": False}
        self.rot_mode = "global"
        self.use_ref_pose = False
        self.ref_ready = False
        self.ref_delay_s = 1.0
        self._t_start = time.time()
        self.ref_global = {}
        self.ref_local = {}
        self.invert_vmc_zw = True
        self.bone_rot_offset = {}
        self.bone_rot_offset_mode = "post"
        self.bone_axis_override = {}
        self.align_bvh_axes_to_std = False
        self.bvh_axis_to_std = {}
        self.recv_count = 0
        self.last_recv_ts = 0.0
        
        # Initialize all bone rotations to identity.
        for bone in STD_SKELETON:
            self.raw_rots[bone] = np.array([0.0, 0.0, 0.0, 1.0])
            self.name_map[_normalize_name(bone)] = bone

    def start(self):
        disp = dispatcher.Dispatcher()
        disp.map("/VMC/Ext/Bone/Pos", self.on_bone_packet)
        print(f"Listening on {self.ip}:{self.port} (FK Mode)...")
        server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), disp)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def on_bone_packet(self, address, *args):
        # VMC: name, px, py, pz, qx, qy, qz, qw
        if len(args) < 8: return
        name = str(args[0])
        key = _normalize_name(name)
        if key not in self.name_map:
            return
        canon_name = self.name_map[key]
        # VMC packet fields (4,5,6,7) contain quaternion rotation.
        rot = np.array(args[4:8], dtype=float)
        if self.invert_vmc_zw:
            rot[2] = -rot[2]
            rot[3] = -rot[3]
        
        with self._lock:
            # Update latest rotation.
            self.raw_rots[canon_name] = rot
            self.recv_count += 1
            self.last_recv_ts = time.time()
    
    def solve_fk(self):
        """Core FK solver."""
        with self._lock:
            current_rots = self.raw_rots.copy()
            axis_cfg = dict(self.axis_cfg)
            rot_mode = str(self.rot_mode)
            use_ref_pose = bool(self.use_ref_pose)
            ref_ready = bool(self.ref_ready)
            ref_delay_s = float(self.ref_delay_s)
            t_start = float(self._t_start)

        # 1. Hips (root) is special.
        # If SlimeVR root position is unavailable, keep hips at a fixed position.
        global_positions = {}
        global_rotations = {}
        base_basis = _basis_matrix(axis_cfg)

        # Solve hierarchically so parent transforms are available first.
        
        # Build processing order (simple topological strategy).
        process_queue = ["Hips"]
        
        # Initialize hips.
        hips_rot = current_rots.get("Hips", np.array([0.,0.,0.,1.]))
        hips_rot_m = _quat_to_mat_xyzw(hips_rot)
        hips_basis = _basis_matrix(self.bone_axis_override.get("Hips", axis_cfg))
        hips_rot_m = hips_basis @ hips_rot_m @ hips_basis.T
        if self.align_bvh_axes_to_std and "Hips" in self.bvh_axis_to_std:
            R = self.bvh_axis_to_std["Hips"]
            hips_rot_m = R.T @ hips_rot_m @ R
        global_positions["Hips"] = np.array([0.0, 1.0, 0.0])  # Place root at 1m height.
        global_rotations["Hips"] = hips_rot_m

        processed = {"Hips"}
        
        # Iterate until all bones are solved.
        while len(processed) < len(STD_SKELETON):
            # Pick a bone whose parent has already been solved.
            for bone, (parent, offset) in STD_SKELETON.items():
                if bone in processed: continue
                if parent in processed:
                    # Parent is ready, solve current bone.
                    parent_pos = global_positions[parent]
                    parent_rot = global_rotations[parent]
                    
                    # Core formula:
                    # child_world_pos = parent_world_pos + parent_world_rot * local_offset
                    offset_vec = np.array(offset)
                    rotated_offset = parent_rot @ offset_vec
                    curr_pos = parent_pos + rotated_offset
                    
                    # Compute current bone global rotation.
                    bone_rot = current_rots.get(bone, np.array([0.,0.,0.,1.]))
                    bone_rot_m = _quat_to_mat_xyzw(bone_rot)
                    bone_basis = _basis_matrix(self.bone_axis_override.get(bone, axis_cfg))
                    bone_rot_m = bone_basis @ bone_rot_m @ bone_basis.T
                    if self.align_bvh_axes_to_std and bone in self.bvh_axis_to_std:
                        R = self.bvh_axis_to_std[bone]
                        bone_rot_m = R.T @ bone_rot_m @ R

                    if bone in self.bone_rot_offset:
                        offset_m = self.bone_rot_offset[bone]
                        if self.bone_rot_offset_mode == "pre":
                            bone_rot_m = offset_m @ bone_rot_m
                        else:
                            bone_rot_m = bone_rot_m @ offset_m

                    # capture reference pose (per-bone)
                    if use_ref_pose and (not ref_ready) and (time.time() - t_start >= ref_delay_s):
                        if rot_mode == "global":
                            self.ref_global[bone] = bone_rot_m
                        else:
                            self.ref_local[bone] = bone_rot_m

                    if rot_mode == "global":
                        # VMC gives global rotation: local = parent_global.T * global
                        if use_ref_pose and bone in self.ref_global:
                            bone_rot_m = self.ref_global[bone].T @ bone_rot_m
                        local_rot = parent_rot.T @ bone_rot_m
                        bone_global_rot_m = parent_rot @ local_rot
                    else:
                        # VMC gives local rotation
                        if use_ref_pose and bone in self.ref_local:
                            bone_rot_m = self.ref_local[bone].T @ bone_rot_m
                        bone_global_rot_m = parent_rot @ bone_rot_m
                    
                    global_positions[bone] = curr_pos
                    global_rotations[bone] = bone_global_rot_m
                    
                    processed.add(bone)

        if use_ref_pose and (not ref_ready) and (time.time() - t_start >= ref_delay_s):
            self.ref_ready = True
        self.computed_pos = global_positions
        self.computed_rot = global_rotations
        return global_positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=39539)
    parser.add_argument("--mirror_x", action="store_true", help="Invert X axis in rotation space")
    parser.add_argument("--mirror_y", action="store_true", help="Invert Y axis in rotation space")
    parser.add_argument("--mirror_z", action="store_true", help="Invert Z axis in rotation space")
    parser.add_argument("--swap_lr", action="store_true", help="Swap Left/Right bone names from VMC")
    parser.add_argument("--swap_xy", action="store_true", help="Swap X and Y axis")
    parser.add_argument("--swap_xz", action="store_true", help="Swap X and Z axis")
    parser.add_argument("--swap_yz", action="store_true", help="Swap Y and Z axis")
    parser.add_argument("--auto_cycle", action="store_true", help="Auto cycle axis configs")
    parser.add_argument("--cycle_seconds", type=float, default=2.0, help="Seconds per config when auto_cycle")
    parser.add_argument("--cycle_once", action="store_true", help="Cycle through configs once then stop")
    parser.add_argument("--axis_swap", type=str, default="xyz", help="Axis swap order (e.g. xzy, yxz)")
    parser.add_argument("--axis_flip", type=str, default="", help="Axis flips (e.g. x, yz)")
    parser.add_argument("--use_bvh_skeleton", action="store_true", help="Use BVH offsets as skeleton")
    parser.add_argument("--use_bvh_lengths_only", action="store_true", help="Use BVH bone lengths but keep STD directions")
    parser.add_argument("--bvh_path", type=str, default="", help="BVH path for offsets")
    parser.add_argument("--bvh_scale", type=float, default=1.0, help="Scale BVH offsets")
    parser.add_argument("--mpl_backend", type=str, default="", help="Matplotlib backend override (e.g. TkAgg)")
    parser.add_argument("--save_png", action="store_true", help="Save frames as PNG when no GUI")
    parser.add_argument("--save_dir", type=str, default="/tmp/vmc_fk_viewer", help="Directory for PNG dumps")
    parser.add_argument("--save_every", type=int, default=30, help="Save every N frames")
    parser.add_argument("--save_even_empty", action="store_true", help="Save PNG even if no pose data")
    parser.add_argument("--warn_no_data_every", type=int, default=100, help="Warn if no pose data every N frames")
    parser.add_argument("--headless", action="store_true", help="Force Agg backend (no GUI)")
    parser.add_argument("--log_every", type=int, default=60, help="Log heartbeat every N frames")
    parser.add_argument("--auto_bounds", action="store_true", help="Auto fit view bounds to skeleton")
    parser.add_argument("--bound_margin", type=float, default=0.2, help="Auto bounds margin (meters)")
    parser.add_argument("--fixed_bounds", type=float, default=2.0, help="Half-size for fixed bounds")
    parser.add_argument("--pos_scale", type=float, default=1.0, help="Scale positions for view only")
    parser.add_argument("--dump_joints", action="store_true", help="Print key joint positions")
    parser.add_argument("--bone_rot_offset", type=str, default="", help="Per-bone rot offsets: Bone:rx,ry,rz;...")
    parser.add_argument("--bone_rot_offset_mode", choices=["pre", "post"], default="post", help="Offset multiply mode")
    parser.add_argument("--bone_axis_override", type=str, default="", help="Per-bone axis override: Bone:swap=xyz,flip=xy;...")
    parser.add_argument("--align_bvh_axes_to_std", action="store_true", help="Align BVH bone axes to STD offsets")
    parser.add_argument("--align_bvh_axes_to_std_bones", type=str, default="", help="Comma bones for axis align; empty=all")
    parser.add_argument("--align_bvh_axes_use_frame", action="store_true", help="Use bone frame (primary+secondary) for axis align")
    parser.add_argument("--auto_cycle_upperarm", action="store_true", help="Auto cycle axis overrides for UpperArm")
    parser.add_argument("--upperarm_cycle_seconds", type=float, default=2.0, help="Seconds per UpperArm config")
    parser.add_argument("--upperarm_cycle_once", action="store_true", help="Cycle UpperArm configs once then stop")
    parser.add_argument("--rot_mode", choices=["global", "local"], default="local", help="VMC rotations are global or local")
    parser.add_argument("--use_ref_pose", action="store_true", help="Use first pose as reference to remove rest offset")
    parser.add_argument("--ref_delay_s", type=float, default=1.0, help="Delay before capturing reference pose")
    parser.add_argument("--invert_vmc_zw", action="store_true", help="Invert VMC quaternion z/w (Unity coords)")
    parser.add_argument("--no_invert_vmc_zw", action="store_true", help="Do not invert VMC quaternion z/w")
    parser.add_argument("--clean_start", action="store_true", help="Disable all extra transforms and overrides")
    args = parser.parse_args()

    if bool(args.clean_start):
        args.mirror_x = False
        args.mirror_y = False
        args.mirror_z = False
        args.swap_lr = False
        args.swap_xy = False
        args.swap_xz = False
        args.swap_yz = False
        args.axis_swap = "xyz"
        args.axis_flip = ""
        args.auto_cycle = False
        args.use_ref_pose = False
        args.invert_vmc_zw = False
        args.no_invert_vmc_zw = True
        args.bone_rot_offset = ""
        args.bone_axis_override = ""
        args.align_bvh_axes_to_std = False
        args.align_bvh_axes_to_std_bones = ""
        args.align_bvh_axes_use_frame = False
        args.auto_cycle_upperarm = False

    backend = str(args.mpl_backend).strip()
    if bool(args.headless):
        backend = "Agg"
    if backend:
        os.environ["MPLBACKEND"] = backend
    import matplotlib
    if backend:
        matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    backend_used = str(plt.get_backend())
    print(f"[Viewer] DISPLAY={os.environ.get('DISPLAY', '')} backend={backend_used}", flush=True)
    if backend and backend.lower() not in backend_used.lower():
            print(f"[Viewer] WARNING: requested backend={backend} but got {backend_used}", flush=True)

    receiver = VMCFKReceiver(args.ip, args.port)
    if bool(args.use_bvh_skeleton) or bool(args.use_bvh_lengths_only):
        bvh_parents, bvh_offsets = _parse_bvh_offsets(str(args.bvh_path))
        # Replace STD_SKELETON with BVH offsets
        global STD_SKELETON
        if bool(args.use_bvh_lengths_only):
            STD_SKELETON = dict(STD_SKELETON_BASE)
            for bone, (parent, off_std) in list(STD_SKELETON.items()):
                if bone not in bvh_offsets:
                    continue
                off_bvh = np.array(bvh_offsets[bone], dtype=float)
                len_bvh = float(np.linalg.norm(off_bvh)) * float(args.bvh_scale)
                dir_std = np.array(off_std, dtype=float)
                n = float(np.linalg.norm(dir_std))
                if n < 1e-8:
                    continue
                dir_std = dir_std / n
                STD_SKELETON[bone] = (parent, (dir_std[0] * len_bvh, dir_std[1] * len_bvh, dir_std[2] * len_bvh))
        else:
            STD_SKELETON = {}
            for name, parent in bvh_parents.items():
                off = bvh_offsets.get(name, np.array([0.0, 0.0, 0.0], dtype=float))
                STD_SKELETON[name] = (parent, (float(off[0]) * float(args.bvh_scale), float(off[1]) * float(args.bvh_scale), float(off[2]) * float(args.bvh_scale)))
        if bool(args.align_bvh_axes_to_std):
            bone_whitelist = None
            if str(args.align_bvh_axes_to_std_bones).strip():
                bone_whitelist = set([b.strip() for b in str(args.align_bvh_axes_to_std_bones).split(",") if b.strip()])
            bvh_axis_to_std = {}
            if bool(args.align_bvh_axes_use_frame):
                bvh_children = _build_children_map(STD_SKELETON)
                std_children = _build_children_map(STD_SKELETON_BASE)
            for bone, (_, off_bvh) in STD_SKELETON.items():
                if bone_whitelist is not None and bone not in bone_whitelist:
                    continue
                if bone not in STD_SKELETON_BASE:
                    continue
                off_std = STD_SKELETON_BASE[bone][1]
                if bool(args.align_bvh_axes_use_frame):
                    if not bvh_children.get(bone) or not std_children.get(bone):
                        continue
                    child_bvh = bvh_children[bone][0]
                    child_std = std_children[bone][0]
                    off_bvh_primary = np.array(STD_SKELETON[child_bvh][1], dtype=float)
                    off_std_primary = np.array(STD_SKELETON_BASE[child_std][1], dtype=float)
                    off_bvh_secondary = np.array(off_bvh, dtype=float)
                    off_std_secondary = np.array(off_std, dtype=float)
                    Fb = _make_frame(off_bvh_primary, off_bvh_secondary)
                    Fs = _make_frame(off_std_primary, off_std_secondary)
                    R = Fs @ Fb.T
                else:
                    R = _rot_from_vectors(np.array(off_bvh), np.array(off_std))
                bvh_axis_to_std[bone] = R
            receiver.bvh_axis_to_std = bvh_axis_to_std
            receiver.align_bvh_axes_to_std = True
            print(f"[Viewer] align_bvh_axes_to_std bones={len(bvh_axis_to_std)}", flush=True)
        # Reset name_map and rots to match BVH skeleton
        receiver.raw_rots = {}
        receiver.name_map = {}
        for bone in STD_SKELETON:
            receiver.raw_rots[bone] = np.array([0.0, 0.0, 0.0, 1.0])
        # Build VMC->BVH mapping so VMC names drive BVH joints
        vmc_to_bvh = _build_vmc_to_bvh_map(set(STD_SKELETON.keys()))
        for vmc_name, bvh_name in vmc_to_bvh.items():
            receiver.name_map[_normalize_name(vmc_name)] = bvh_name
        if not vmc_to_bvh:
            print("[Viewer] WARNING: no VMC->BVH mapping found; check BVH joint names", flush=True)
        else:
            print(f"[Viewer] VMC->BVH mapping size={len(vmc_to_bvh)}", flush=True)
    receiver.rot_mode = str(args.rot_mode)
    receiver.use_ref_pose = bool(args.use_ref_pose)
    receiver.ref_delay_s = float(args.ref_delay_s)
    if bool(args.no_invert_vmc_zw):
        receiver.invert_vmc_zw = False
    elif bool(args.invert_vmc_zw):
        receiver.invert_vmc_zw = True
    if str(args.bone_axis_override).strip():
        overrides = {}
        items = [s.strip() for s in str(args.bone_axis_override).split(";") if s.strip()]
        for item in items:
            if ":" not in item:
                continue
            bone_name, cfg_str = item.split(":", 1)
            bone_name = bone_name.strip()
            cfg = {"swap": "xyz", "mirror_x": False, "mirror_y": False, "mirror_z": False}
            parts = [p.strip() for p in cfg_str.split(",") if p.strip()]
            for p in parts:
                if p.startswith("swap="):
                    cfg["swap"] = p.split("=", 1)[1].strip()
                elif p.startswith("flip="):
                    flips = p.split("=", 1)[1].strip()
                    cfg["mirror_x"] = "x" in flips
                    cfg["mirror_y"] = "y" in flips
                    cfg["mirror_z"] = "z" in flips
            overrides[bone_name] = cfg
        receiver.bone_axis_override = overrides

    base_bone_axis_override = dict(receiver.bone_axis_override)
    upperarm_cycle_cfgs = []
    if bool(args.auto_cycle_upperarm):
        swaps = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        mirror_flags = [
            {"mirror_x": False, "mirror_y": False, "mirror_z": False},
            {"mirror_x": True, "mirror_y": False, "mirror_z": False},
            {"mirror_x": False, "mirror_y": True, "mirror_z": False},
            {"mirror_x": False, "mirror_y": False, "mirror_z": True},
            {"mirror_x": True, "mirror_y": True, "mirror_z": False},
            {"mirror_x": True, "mirror_y": False, "mirror_z": True},
            {"mirror_x": False, "mirror_y": True, "mirror_z": True},
            {"mirror_x": True, "mirror_y": True, "mirror_z": True},
        ]
        for s in swaps:
            for m in mirror_flags:
                upperarm_cycle_cfgs.append({"swap": s, **m})
    receiver.bone_rot_offset_mode = str(args.bone_rot_offset_mode)
    if str(args.bone_rot_offset).strip():
        offsets = {}
        items = [s.strip() for s in str(args.bone_rot_offset).split(";") if s.strip()]
        for item in items:
            if ":" not in item:
                continue
            bone_name, angles = item.split(":", 1)
            bone_name = bone_name.strip()
            parts = [p.strip() for p in angles.split(",")]
            if len(parts) != 3:
                continue
            try:
                offsets[bone_name] = _rot_from_euler_xyz_deg(parts[0], parts[1], parts[2])
            except Exception:
                continue
        receiver.bone_rot_offset = offsets
    base_name_map = dict(receiver.name_map)
    if bool(args.swap_lr):
        for k, v in list(receiver.name_map.items()):
            if "left" in k:
                receiver.name_map[k] = v.replace("Left", "Right")
            elif "right" in k:
                receiver.name_map[k] = v.replace("Right", "Left")
    receiver.start()

    # Plot setup
    if not bool(args.headless):
        plt.ion()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    if not bool(args.headless):
        plt.show(block=False)
    if bool(args.save_png):
        os.makedirs(str(args.save_dir), exist_ok=True)
        print(f"[Viewer] save_dir={str(args.save_dir)}", flush=True)
        # Save initial empty frame to verify write access
        try:
            init_out = os.path.join(str(args.save_dir), "frame_init.png")
            fig.canvas.draw()
            fig.savefig(init_out)
            print(f"[Viewer] saved {init_out}", flush=True)
        except Exception as e:
            print(f"[Viewer] [WARN] Save failed: {e}", flush=True)
    
    print("FK Viewer Started. Waiting for rotation data...")

    cycle_cfgs = []
    if bool(args.auto_cycle):
        swaps = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        mirror_flags = [
            {"mirror_x": False, "mirror_y": False, "mirror_z": False},
            {"mirror_x": True, "mirror_y": False, "mirror_z": False},
            {"mirror_x": False, "mirror_y": True, "mirror_z": False},
            {"mirror_x": False, "mirror_y": False, "mirror_z": True},
            {"mirror_x": True, "mirror_y": True, "mirror_z": False},
            {"mirror_x": True, "mirror_y": False, "mirror_z": True},
            {"mirror_x": False, "mirror_y": True, "mirror_z": True},
            {"mirror_x": True, "mirror_y": True, "mirror_z": True},
        ]
        lr_opts = [False, True]
        for lr in lr_opts:
            for s in swaps:
                for m in mirror_flags:
                    cfg = {"swap": s, "swap_lr": lr, **m}
                    cycle_cfgs.append(cfg)
    cycle_idx = 0
    last_cycle_t = time.time()

    try:
        frame_idx = 0
        last_heartbeat = time.time()
        upperarm_cycle_idx = 0
        last_upperarm_cycle_t = time.time() - float(args.upperarm_cycle_seconds)
        while True:
            # 1) Solve FK.
            positions = receiver.solve_fk()
            if bool(args.log_every) and (frame_idx % int(args.log_every) == 0):
                print(f"[Viewer] frame={frame_idx} joints={len(positions)} recv={receiver.recv_count}", flush=True)
            if bool(args.dump_joints) and (frame_idx % int(args.log_every) == 0):
                dump_names = ["Hips", "Spine", "LeftUpperArm", "RightUpperArm", "LeftFoot", "RightFoot"]
                print(f"[vmc_fk_viewer t={time.time():.3f}]", flush=True)
                for n in dump_names:
                    p = positions.get(n)
                    if p is None:
                        continue
                    print(f"  {n}: {np.asarray(p, dtype=np.float32).round(4)}", flush=True)
            if time.time() - last_heartbeat >= 2.0:
                print(f"[Viewer] heartbeat frame={frame_idx}")
                last_heartbeat = time.time()
            
            # 2) Draw frame.
            ax.cla()
            ax.set_xlabel('X'); ax.set_ylabel('Z (Depth)'); ax.set_zlabel('Y (Up)')
            
            if not positions:
                if bool(args.warn_no_data_every) and (frame_idx % int(args.warn_no_data_every) == 0):
                    print("[Viewer] no pose data yet", flush=True)
                if bool(args.save_png) and bool(args.save_even_empty) and int(args.save_every) > 0:
                    if (frame_idx % int(args.save_every)) == 0:
                        out = os.path.join(str(args.save_dir), f"frame_{frame_idx:06d}.png")
                        try:
                            fig.canvas.draw()
                            fig.savefig(out)
                            print(f"[Viewer] saved {out}", flush=True)
                        except Exception as e:
                            print(f"[Viewer] [WARN] Save failed: {e}", flush=True)
                frame_idx += 1
                if bool(args.headless):
                    time.sleep(0.01)
                else:
                    plt.pause(0.1)
                continue

            # Draw skeleton edges.
            if bool(args.auto_cycle) and cycle_cfgs:
                now = time.time()
                if (now - last_cycle_t) >= float(args.cycle_seconds):
                    last_cycle_t = now
                    cycle_idx = (cycle_idx + 1) % len(cycle_cfgs)
                    if bool(args.cycle_once) and cycle_idx == 0:
                        print("[OK] Completed one full axis-cycle pass. Stopping.")
                        return
                    cfg = cycle_cfgs[cycle_idx]
                    receiver.name_map = dict(base_name_map)
                    if bool(cfg.get("swap_lr", False)):
                        for k, v in list(receiver.name_map.items()):
                            if "left" in k:
                                receiver.name_map[k] = v.replace("Left", "Right")
                            elif "right" in k:
                                receiver.name_map[k] = v.replace("Right", "Left")
                    receiver.axis_cfg = {
                        "swap": cfg.get("swap", ""),
                        "mirror_x": bool(cfg.get("mirror_x", False)),
                        "mirror_y": bool(cfg.get("mirror_y", False)),
                        "mirror_z": bool(cfg.get("mirror_z", False)),
                    }
                    print(f"[cycle] swap={cfg.get('swap','xyz')} mirror=({int(cfg.get('mirror_x',0))},{int(cfg.get('mirror_y',0))},{int(cfg.get('mirror_z',0))}) swap_lr={int(cfg.get('swap_lr',0))}")
                cfg = cycle_cfgs[cycle_idx]
            else:
                swap = str(args.axis_swap)
                if bool(args.swap_xy):
                    swap = "yxz"
                if bool(args.swap_xz):
                    swap = "z yx".replace(" ", "")
                if bool(args.swap_yz):
                    swap = "xzy"
                cfg = {
                    "swap": swap,
                    "mirror_x": bool(args.mirror_x) or ("x" in str(args.axis_flip)),
                    "mirror_y": bool(args.mirror_y) or ("y" in str(args.axis_flip)),
                    "mirror_z": bool(args.mirror_z) or ("z" in str(args.axis_flip)),
                    "swap_lr": bool(args.swap_lr),
                }
                receiver.axis_cfg = {
                    "swap": cfg.get("swap", ""),
                    "mirror_x": bool(cfg.get("mirror_x", False)),
                    "mirror_y": bool(cfg.get("mirror_y", False)),
                    "mirror_z": bool(cfg.get("mirror_z", False)),
                }
            if bool(args.auto_cycle_upperarm) and upperarm_cycle_cfgs:
                now = time.time()
                if (now - last_upperarm_cycle_t) >= float(args.upperarm_cycle_seconds):
                    last_upperarm_cycle_t = now
                    upperarm_cycle_idx = (upperarm_cycle_idx + 1) % len(upperarm_cycle_cfgs)
                    if bool(args.upperarm_cycle_once) and upperarm_cycle_idx == 0:
                        print("[OK] Completed one full UpperArm axis-cycle pass. Stopping.")
                        return
                    cfg = upperarm_cycle_cfgs[upperarm_cycle_idx]
                    receiver.bone_axis_override = dict(base_bone_axis_override)
                    receiver.bone_axis_override["LeftUpperArm"] = dict(cfg)
                    receiver.bone_axis_override["RightUpperArm"] = dict(cfg)
                    print(f"[upperarm] swap={cfg.get('swap','xyz')} flip=({int(cfg.get('mirror_x',0))},{int(cfg.get('mirror_y',0))},{int(cfg.get('mirror_z',0))})")
            for p1, p2 in DRAW_LINES:
                if p1 in positions and p2 in positions:
                    pos1 = positions[p1] * float(args.pos_scale)
                    pos2 = positions[p2] * float(args.pos_scale)
                    # Matplotlib axis mapping for readability:
                    # x->x, z->y(depth), y->z(height)
                    ax.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], [pos1[1], pos2[1]], c='r', marker='o')

            # bounds
            if bool(args.auto_bounds) and positions:
                pts = np.stack([v for v in positions.values()], axis=0) * float(args.pos_scale)
                mn = pts.min(axis=0) - float(args.bound_margin)
                mx = pts.max(axis=0) + float(args.bound_margin)
                ax.set_xlim(mn[0], mx[0])
                ax.set_ylim(mn[2], mx[2])  # depth uses Z
                ax.set_zlim(mn[1], mx[1])  # up uses Y
            else:
                b = float(args.fixed_bounds)
                ax.set_xlim(-b, b); ax.set_ylim(-b, b); ax.set_zlim(0, b)
            
            if bool(args.save_png) and int(args.save_every) > 0:
                if (frame_idx % int(args.save_every)) == 0:
                    out = os.path.join(str(args.save_dir), f"frame_{frame_idx:06d}.png")
                    try:
                        fig.canvas.draw()
                        fig.savefig(out)
                        print(f"[Viewer] saved {out}", flush=True)
                    except Exception as e:
                        print(f"[Viewer] [WARN] Save failed: {e}", flush=True)
            frame_idx += 1
            if bool(args.headless):
                time.sleep(0.01)
            else:
                plt.pause(0.01)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()