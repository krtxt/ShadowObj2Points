from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_ROT_TYPE_TO_DIM = {
    "quat": 4,
    "r6d": 6,
}


@dataclass(frozen=True)
class HandPoseLayout:
    translation_slice: slice
    joint_slice: slice
    rotation_slice: slice
    n_dofs: int
    rot_type: str
    rot_dim: int

    @property
    def pose_dim(self) -> int:
        return 3 + self.n_dofs + self.rot_dim


_DEFAULT_LAYOUT = HandPoseLayout(
    translation_slice=slice(0, 3),
    joint_slice=slice(3, 27),
    rotation_slice=slice(27, 33),
    n_dofs=24,
    rot_type="r6d",
    rot_dim=_ROT_TYPE_TO_DIM["r6d"],
)

_CURRENT_LAYOUT: HandPoseLayout = _DEFAULT_LAYOUT

# Public module-level mirrors (read-only, updated via configure_hand_pose)
TRANSLATION_SLICE: slice = _CURRENT_LAYOUT.translation_slice
QPOS_SLICE: slice = _CURRENT_LAYOUT.joint_slice
ROTATION_SLICE: slice = _CURRENT_LAYOUT.rotation_slice
JOINT_ANGLE_DIM: int = _CURRENT_LAYOUT.n_dofs
POSE_DIM: int = _CURRENT_LAYOUT.pose_dim


def _resolve_rot_dim(rot_type: str) -> int:
    try:
        return _ROT_TYPE_TO_DIM[rot_type]
    except KeyError:
        raise ValueError(
            f"Unsupported rot_type '{rot_type}'. Expected one of {list(_ROT_TYPE_TO_DIM.keys())}."
        ) from None


def configure_hand_pose(n_dofs: int, rot_type: str) -> HandPoseLayout:
    """
    Configure global pose layout. Must be called once training/inference configuration is known.
    """
    global _CURRENT_LAYOUT, TRANSLATION_SLICE, QPOS_SLICE, ROTATION_SLICE, JOINT_ANGLE_DIM, POSE_DIM

    if n_dofs <= 0:
        raise ValueError(f"n_dofs must be positive, got {n_dofs}.")

    rot_dim = _resolve_rot_dim(rot_type)
    translation_slice = slice(0, 3)
    qpos_slice = slice(3, 3 + n_dofs)
    rotation_slice = slice(3 + n_dofs, 3 + n_dofs + rot_dim)

    _CURRENT_LAYOUT = HandPoseLayout(
        translation_slice=translation_slice,
        joint_slice=qpos_slice,
        rotation_slice=rotation_slice,
        n_dofs=n_dofs,
        rot_type=rot_type,
        rot_dim=rot_dim,
    )

    TRANSLATION_SLICE = translation_slice
    QPOS_SLICE = qpos_slice
    ROTATION_SLICE = rotation_slice
    JOINT_ANGLE_DIM = n_dofs
    POSE_DIM = _CURRENT_LAYOUT.pose_dim

    return _CURRENT_LAYOUT


def get_hand_pose_layout() -> HandPoseLayout:
    """
    Retrieve the currently active pose layout.
    """
    return _CURRENT_LAYOUT


def get_joint_dim() -> int:
    return _CURRENT_LAYOUT.n_dofs


def get_rot_dim() -> int:
    return _CURRENT_LAYOUT.rot_dim


def get_pose_dim() -> int:
    return _CURRENT_LAYOUT.pose_dim
