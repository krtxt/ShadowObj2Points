"""
Rotation Specification Design Pattern

This module provides a unified interface for different rotation representations
(quaternion, 6D, euler, axis-angle) following the PathSpec pattern from fm_lightning.py.

Usage:
    from utils.rotation_spec import get_rotation_spec
    
    spec = get_rotation_spec('quat')
    rot_matrix = spec.to_matrix_fn(quaternion)
    loss = spec.loss_fn(pred, target)
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from pytorch3d import transforms

# 全局缓存，避免重复创建 RotationSpec
_ROTATION_SPEC_CACHE = {}

logger = logging.getLogger(__name__)


@dataclass
class RotationSpec:
    """
    旋转表示规范
    
    Attributes:
        name: 旋转类型名称 ('quat', 'r6d', 'euler', 'axis')
        dim: 旋转参数维度 (4, 6, 3, 3)
        is_continuous: 是否连续表示（r6d/quat 是，euler 不是）
        needs_normalization: 是否需要归一化（quat 需要）
        to_matrix_fn: 转换为旋转矩阵的函数
        from_matrix_fn: 从旋转矩阵转换的函数
        normalize_fn: 归一化函数
        loss_fn: 专用损失函数
        geodesic_distance_fn: 测地线距离计算函数
        labels: 用于日志和可视化的维度标签
    """
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


# ============================================================================
# 辅助函数
# ============================================================================

def normalize_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
    """
    对 6D 旋转表示进行 Gram-Schmidt 正交化
    
    Args:
        rot6d: [..., 6] 6D 旋转表示
    
    Returns:
        正交化后的 6D 表示 [..., 6]
    """
    # 提取前两个 3D 向量
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    # Gram-Schmidt 正交化
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    
    return torch.cat([b1, b2], dim=-1)


def _identity_normalize(x: torch.Tensor) -> torch.Tensor:
    """身份归一化函数（不做任何操作）"""
    return x


def _quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    """四元数归一化到单位四元数"""
    return F.normalize(quat, p=2, dim=-1, eps=1e-8)


# ============================================================================
# 损失函数
# ============================================================================

def _calculate_euler_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算欧拉角旋转损失"""
    error = (prediction - target).abs().sum(-1).mean()
    return error


def _calculate_quat_loss(prediction_raw: torch.Tensor, target_unit: torch.Tensor) -> torch.Tensor:
    """
    计算四元数旋转损失（测量角度差异）
    
    Args:
        prediction_raw: 原始四元数预测 (K, 4) - 不一定是单位四元数
        target_unit: 目标单位四元数 (K, 4) - 已归一化
    
    Returns:
        损失值，范围 [0, 1]：0 = 完美对齐，1 = 90° 差异
    """
    # 归一化预测为单位四元数
    epsilon = 1e-8
    prediction_norm = torch.linalg.norm(prediction_raw, dim=-1, keepdim=True)
    normalized_prediction = prediction_raw / (prediction_norm + epsilon)
    
    # 计算点积：对于单位四元数 p 和 q，p·q = cos(theta/2)，theta 是旋转角
    dot_product = (normalized_prediction * target_unit).sum(dim=-1)
    
    # 计算损失：1 - |dot_product|
    # abs() 处理 q 和 -q 表示相同旋转的情况
    rotation_loss = 1.0 - dot_product.abs()
    mean_rotation_loss = rotation_loss.mean()
    
    return mean_rotation_loss


def _calculate_axis_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算轴角旋转损失（MSE）"""
    return F.mse_loss(prediction, target)


def _calculate_r6d_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算 6D 旋转表示损失（MSE）"""
    return F.mse_loss(prediction, target)


# ============================================================================
# 测地线距离函数
# ============================================================================

def _rotation_geodesic_distance(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    计算 SO(3) 测地线距离（旋转角度误差）
    
    这是 SO(3) 流形上的真实距离度量，与旋转表示（quat/r6d）无关。
    
    Args:
        R_pred: 预测的旋转矩阵 [..., 3, 3]
        R_gt: 目标旋转矩阵 [..., 3, 3]
    
    Returns:
        angle: 弧度制的角度误差 [...]，范围 [0, π]
    
    数学原理：
        对于两个旋转矩阵 R1 和 R2，相对旋转为 R_rel = R1^T @ R2
        旋转角度 θ 满足：trace(R_rel) = 1 + 2*cos(θ)
        因此：θ = arccos((trace(R_rel) - 1) / 2)
    """
    # 计算相对旋转矩阵
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_gt)
    
    # 计算迹（对角线元素之和）
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    # 从迹恢复旋转角度
    # 数值稳定性处理：clamp 到有效范围 [-1, 1]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    # 计算角度（弧度）
    angle = torch.acos(cos_angle)
    
    return angle


def _create_geodesic_fn_from_spec(spec: 'RotationSpec') -> Callable:
    """
    为特定旋转类型创建测地线距离函数
    
    Args:
        spec: RotationSpec 实例
    
    Returns:
        计算两个旋转表示之间测地线距离的函数
    """
    def geodesic_distance(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
        """
        计算两个旋转表示之间的测地线距离
        
        Args:
            rot1, rot2: 旋转表示张量 [..., rot_dim]
        
        Returns:
            角度误差（弧度）[...]
        """
        # 转换为旋转矩阵
        if spec.needs_normalization:
            rot1 = spec.normalize_fn(rot1)
            rot2 = spec.normalize_fn(rot2)
        
        R1 = spec.to_matrix_fn(rot1)
        R2 = spec.to_matrix_fn(rot2)
        
        # 计算测地线距离
        return _rotation_geodesic_distance(R1, R2)
    
    return geodesic_distance


# ============================================================================
# RotationSpecFactory 工厂类
# ============================================================================

class RotationSpecFactory:
    """旋转配置工厂类"""
    
    @staticmethod
    def create(rot_type: str, debug: bool = False) -> RotationSpec:
        """
        创建 RotationSpec 实例
        
        Args:
            rot_type: 旋转类型 ('quat', 'r6d', 'euler', 'axis')
            debug: 是否输出调试信息
        
        Returns:
            RotationSpec 实例
        
        Raises:
            ValueError: 不支持的旋转类型
        """
        rt = (rot_type or "").lower().strip()
        
        if debug:
            logger.info(f"[RotationSpec] Creating spec for rot_type='{rt}'")
        
        if rt == 'quat':
            spec = RotationSpec(
                name='quat',
                dim=4,
                is_continuous=True,
                needs_normalization=True,
                to_matrix_fn=transforms.quaternion_to_matrix,
                from_matrix_fn=transforms.matrix_to_quaternion,
                normalize_fn=_quat_normalize,
                loss_fn=_calculate_quat_loss,
                geodesic_distance_fn=None,  # 将在后面设置
                labels=["quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"],
            )
        
        elif rt == 'r6d':
            spec = RotationSpec(
                name='r6d',
                dim=6,
                is_continuous=True,
                needs_normalization=False,
                to_matrix_fn=transforms.rotation_6d_to_matrix,
                from_matrix_fn=transforms.matrix_to_rotation_6d,
                normalize_fn=normalize_rot6d,
                loss_fn=_calculate_r6d_loss,
                geodesic_distance_fn=None,
                labels=[f"r6d_dim_{i}" for i in range(6)],
            )
        
        elif rt == 'euler':
            spec = RotationSpec(
                name='euler',
                dim=3,
                is_continuous=False,
                needs_normalization=False,
                to_matrix_fn=lambda x: transforms.euler_angles_to_matrix(x, "XYZ"),
                from_matrix_fn=lambda x: transforms.matrix_to_euler_angles(x, "XYZ"),
                normalize_fn=_identity_normalize,
                loss_fn=_calculate_euler_loss,
                geodesic_distance_fn=None,
                labels=["euler_x", "euler_y", "euler_z"],
            )
        
        elif rt == 'axis':
            spec = RotationSpec(
                name='axis',
                dim=3,
                is_continuous=True,
                needs_normalization=False,
                to_matrix_fn=transforms.axis_angle_to_matrix,
                from_matrix_fn=transforms.matrix_to_axis_angle,
                normalize_fn=_identity_normalize,
                loss_fn=_calculate_axis_loss,
                geodesic_distance_fn=None,
                labels=[f"axis_angle_dim_{i}" for i in range(3)],
            )
        
        else:
            raise ValueError(
                f"Unsupported rot_type: '{rot_type}'. "
                f"Supported types: 'quat', 'r6d', 'euler', 'axis'"
            )
        
        # 为 spec 创建测地线距离函数
        spec.geodesic_distance_fn = _create_geodesic_fn_from_spec(spec)
        
        if debug:
            logger.info(
                f"[RotationSpec] Created: name={spec.name}, dim={spec.dim}, "
                f"continuous={spec.is_continuous}, needs_norm={spec.needs_normalization}"
            )
        
        return spec


# ============================================================================
# 公共接口
# ============================================================================

def get_rotation_spec(rot_type: str, debug: bool = False) -> RotationSpec:
    """
    获取 RotationSpec 实例（带缓存）
    
    Args:
        rot_type: 旋转类型
        debug: 是否输出调试信息
    
    Returns:
        RotationSpec 实例
    """
    if rot_type not in _ROTATION_SPEC_CACHE:
        _ROTATION_SPEC_CACHE[rot_type] = RotationSpecFactory.create(rot_type, debug=debug)
    return _ROTATION_SPEC_CACHE[rot_type]


def clear_cache():
    """清空 RotationSpec 缓存（用于测试）"""
    global _ROTATION_SPEC_CACHE
    _ROTATION_SPEC_CACHE.clear()


# ============================================================================
# 便捷函数（向后兼容）
# ============================================================================

def get_rot_dim(rot_type: str) -> int:
    """获取旋转表示的维度"""
    spec = get_rotation_spec(rot_type)
    return spec.dim


def get_rot_labels(rot_type: str) -> list:
    """获取旋转表示的标签"""
    spec = get_rotation_spec(rot_type)
    return spec.labels


def is_continuous_rotation(rot_type: str) -> bool:
    """判断是否为连续旋转表示"""
    spec = get_rotation_spec(rot_type)
    return spec.is_continuous

