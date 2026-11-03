"""Standalone 29-joint motion interchange format; quaternions are WXYZ."""
import numpy as np
from scipy.spatial.transform import Rotation

BM_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def body_joint_axes(urdf):
    axes = np.zeros((len(BM_BODY_JOINT_NAMES), 3), dtype=np.float32)
    for i, name in enumerate(BM_BODY_JOINT_NAMES):
        axis = urdf.joint_map[name].axis
        axis = np.asarray(axis, dtype=np.float32) if axis is not None else np.array([0, 0, 1.])
        norm = np.linalg.norm(axis)
        axes[i] = axis / norm if norm > 1e-8 else np.array([0, 0, 1.])
    return axes


def compute_pose_aa(dof, base_wxyz, joint_axes):
    pose = np.zeros((len(dof), 30, 3), dtype=np.float32)
    pose[:, 0] = Rotation.from_quat(base_wxyz[:, [1, 2, 3, 0]]).as_rotvec()
    pose[:, 1:] = dof[:, :, None] * joint_axes[None, :, :]
    return pose
