import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def quatToEuler(quat):
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec



def quat_rotate_inverse(q, v):
    """
    Rotate vector v by the inverse of quaternion q.
    This implementation matches the original script for consistency.
    """
    q = np.asarray(q)
    v = np.asarray(v)

    q_w = q[:, -1]      # quaternion w (scalar) component
    q_vec = q[:, :3]    # x, y, z

    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * (2.0 * q_w)[:, np.newaxis]
    dot = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * (2.0 * dot)

    return a - b + c

def quat_rotate_inverse_torch(q, v, scalar_first=True):
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def quat_rotate_inverse_np(q, v, scalar_first=True):
    q = np.asarray(q)
    v = np.asarray(v)
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c

def euler_from_quaternion_torch(quat_angle, scalar_first=True):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    if scalar_first:
        quat_angle = quat_angle[..., [1, 2, 3, 0]]
    else:
        quat_angle = quat_angle[..., [0, 1, 2, 3]]
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def euler_from_quaternion_np(quat, scalar_first=True):
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]
    else:
        quat = quat[..., [0, 1, 2, 3]]
    
    x = quat[:,0]; y = quat[:,1]; z = quat[:,2]; w = quat[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


def quat_diff_np(q1, q2, scalar_first=True):
    # Ensure quaternions are numpy arrays
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Convert to scipy Rotation object (scalar-first)
    r1 = R.from_quat(q1, scalar_first=scalar_first)
    r2 = R.from_quat(q2, scalar_first=scalar_first)

    # Relative rotation
    r_rel = r2 * r1.inv()

    # Rotation vector (axis * angle)
    rotvec = r_rel.as_rotvec()  # returns angle * axis vector

    return rotvec
