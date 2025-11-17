"""
旋转表示规范设计模式

提供不同旋转表示（四元数、6D、欧拉角、轴角）的统一接口
代码来源: PathSpec pattern from fm_lightning.py
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from pytorch3d import transforms

_ROTATION_SPEC_CACHE = {}
logger = logging.getLogger(__name__)


@dataclass
class RotationSpec:
    """旋转表示规范"""
    name: str
    dim: int
    is_continuous: bool
    needs_normalization: bool
    to_matrix_fn: Callable[[torch.Tensor], torch.Tensor]
    from_matrix_fn: Callable[[torch.Tensor], torch.Tensor]
    normalize_fn: Callable[[torch.Tensor], torch.Tensor]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    geodesic_distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    labels: list


def normalize_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
    """对6D旋转表示进行Gram-Schmidt正交化 [..., 6] -> [..., 6]"""
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    return torch.cat([b1, b2], dim=-1)


def _calculate_euler_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算欧拉角旋转损失"""
    return (pred - target).abs().sum(-1).mean()


def _calculate_quat_loss(pred_raw: torch.Tensor, target_unit: torch.Tensor) -> torch.Tensor:
    """计算四元数旋转损失（测量角度差异）[0, 1]：0=完美对齐，1=90°差异"""
    pred_norm = torch.linalg.norm(pred_raw, dim=-1, keepdim=True)
    pred_unit = pred_raw / (pred_norm + 1e-8)
    dot = (pred_unit * target_unit).sum(dim=-1)
    return (1.0 - dot.abs()).mean()


def _calculate_axis_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算轴角旋转损失（MSE）"""
    return F.mse_loss(pred, target)


def _calculate_r6d_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算6D旋转表示损失（MSE）"""
    return F.mse_loss(pred, target)


def _rotation_geodesic_distance(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """计算SO(3)测地线距离（旋转角度误差，弧度）[..., 3, 3] -> [...]"""
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_gt)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.acos(cos_angle)


def _create_geodesic_fn_from_spec(spec: RotationSpec) -> Callable:
    """为特定旋转类型创建测地线距离函数"""
    def geodesic_distance(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
        if spec.needs_normalization:
            rot1, rot2 = spec.normalize_fn(rot1), spec.normalize_fn(rot2)
        R1, R2 = spec.to_matrix_fn(rot1), spec.to_matrix_fn(rot2)
        return _rotation_geodesic_distance(R1, R2)
    return geodesic_distance


# 旋转类型配置
_ROT_TYPE_CONFIGS = {
    'quat': {
        'dim': 4, 'is_continuous': True, 'needs_normalization': True,
        'to_matrix_fn': transforms.quaternion_to_matrix,
        'from_matrix_fn': transforms.matrix_to_quaternion,
        'normalize_fn': lambda x: F.normalize(x, p=2, dim=-1, eps=1e-8),
        'loss_fn': _calculate_quat_loss,
        'labels': ["quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"],
    },
    'r6d': {
        'dim': 6, 'is_continuous': True, 'needs_normalization': False,
        'to_matrix_fn': transforms.rotation_6d_to_matrix,
        'from_matrix_fn': transforms.matrix_to_rotation_6d,
        'normalize_fn': normalize_rot6d,
        'loss_fn': _calculate_r6d_loss,
        'labels': [f"r6d_dim_{i}" for i in range(6)],
    },
    'euler': {
        'dim': 3, 'is_continuous': False, 'needs_normalization': False,
        'to_matrix_fn': lambda x: transforms.euler_angles_to_matrix(x, "XYZ"),
        'from_matrix_fn': lambda x: transforms.matrix_to_euler_angles(x, "XYZ"),
        'normalize_fn': lambda x: x,
        'loss_fn': _calculate_euler_loss,
        'labels': ["euler_x", "euler_y", "euler_z"],
    },
    'axis': {
        'dim': 3, 'is_continuous': True, 'needs_normalization': False,
        'to_matrix_fn': transforms.axis_angle_to_matrix,
        'from_matrix_fn': transforms.matrix_to_axis_angle,
        'normalize_fn': lambda x: x,
        'loss_fn': _calculate_axis_loss,
        'labels': [f"axis_angle_dim_{i}" for i in range(3)],
    },
}


def _create_spec(rot_type: str, debug: bool = False) -> RotationSpec:
    """创建RotationSpec实例"""
    rt = (rot_type or "").lower().strip()
    if rt not in _ROT_TYPE_CONFIGS:
        raise ValueError(f"Unsupported rot_type: '{rot_type}'. Supported: {list(_ROT_TYPE_CONFIGS.keys())}")
    
    if debug:
        logger.info(f"[RotationSpec] Creating spec for rot_type='{rt}'")
    
    config = _ROT_TYPE_CONFIGS[rt]
    spec = RotationSpec(
        name=rt,
        dim=config['dim'],
        is_continuous=config['is_continuous'],
        needs_normalization=config['needs_normalization'],
        to_matrix_fn=config['to_matrix_fn'],
        from_matrix_fn=config['from_matrix_fn'],
        normalize_fn=config['normalize_fn'],
        loss_fn=config['loss_fn'],
        geodesic_distance_fn=None,  # 将在后面设置
        labels=config['labels'],
    )
    spec.geodesic_distance_fn = _create_geodesic_fn_from_spec(spec)
    
    if debug:
        logger.info(f"[RotationSpec] Created: name={spec.name}, dim={spec.dim}, "
                   f"continuous={spec.is_continuous}, needs_norm={spec.needs_normalization}")
    return spec


def get_rotation_spec(rot_type: str, debug: bool = False) -> RotationSpec:
    """获取RotationSpec实例（带缓存）"""
    if rot_type not in _ROTATION_SPEC_CACHE:
        _ROTATION_SPEC_CACHE[rot_type] = _create_spec(rot_type, debug)
    return _ROTATION_SPEC_CACHE[rot_type]


def clear_cache():
    """清空RotationSpec缓存（用于测试）"""
    _ROTATION_SPEC_CACHE.clear()


def get_rot_dim(rot_type: str) -> int:
    """获取旋转表示的维度"""
    return get_rotation_spec(rot_type).dim


def get_rot_labels(rot_type: str) -> list:
    """获取旋转表示的标签"""
    return get_rotation_spec(rot_type).labels


def is_continuous_rotation(rot_type: str) -> bool:
    """判断是否为连续旋转表示"""
    return get_rotation_spec(rot_type).is_continuous
