"""
6D旋转表示到旋转矩阵的转换工具

代码来源: https://github.com/papagina/RotationContinuity
On the Continuity of Rotation Representations in Neural Networks, Zhou et al. CVPR19
https://zhouyisjtu.github.io/project_rotation/rotation.html
"""

import torch


def compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """
    将6D旋转表示转换为旋转矩阵
    
    Args:
        poses: [B, 6] 或 [B, num_grasps, 6] - 6D旋转表示
    Returns:
        [B, 3, 3] 或 [B, num_grasps, 3, 3] - 旋转矩阵
    """
    if poses.dim() == 3:
        B, num_grasps, _ = poses.shape
        poses = poses.view(B * num_grasps, 6)
        matrices = _compute_rotation_matrix_from_ortho6d_2d(poses)
        return matrices.view(B, num_grasps, 3, 3)
    elif poses.dim() == 2:
        return _compute_rotation_matrix_from_ortho6d_2d(poses)
    else:
        raise ValueError(f"Unsupported dimension: {poses.dim()}. Expected 2 or 3.")


def _compute_rotation_matrix_from_ortho6d_2d(poses: torch.Tensor) -> torch.Tensor:
    """处理单抓取格式的6D旋转表示转换 [B, 6] -> [B, 3, 3]"""
    x = normalize_vector(poses[:, 0:3])
    z = normalize_vector(cross_product(x, poses[:, 3:6]))
    y = cross_product(z, x)
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)


def robust_compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """
    将6D旋转表示转换为旋转矩阵（鲁棒版本）
    创建考虑两个预测方向相等的基，而不是简单正交化
    
    Args:
        poses: [B, 6] 或 [B, num_grasps, 6] - 6D旋转表示
    Returns:
        [B, 3, 3] 或 [B, num_grasps, 3, 3] - 旋转矩阵
    """
    if poses.dim() == 3:
        B, num_grasps, _ = poses.shape
        poses = poses.view(B * num_grasps, 6)
        matrices = _robust_compute_rotation_matrix_from_ortho6d_2d(poses)
        return matrices.view(B, num_grasps, 3, 3)
    elif poses.dim() == 2:
        return _robust_compute_rotation_matrix_from_ortho6d_2d(poses)
    else:
        raise ValueError(f"Unsupported dimension: {poses.dim()}. Expected 2 or 3.")


def _robust_compute_rotation_matrix_from_ortho6d_2d(poses: torch.Tensor) -> torch.Tensor:
    """处理单抓取格式的鲁棒6D旋转表示转换 [B, 6] -> [B, 3, 3]"""
    x, y = normalize_vector(poses[:, 0:3]), normalize_vector(poses[:, 3:6])
    middle, orthmid = normalize_vector(x + y), normalize_vector(x - y)
    x, y = normalize_vector(middle + orthmid), normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)


def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """向量归一化 [B, 3] -> [B, 3]"""
    return v / torch.clamp(torch.norm(v, dim=1, keepdim=True), min=1e-8)


def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """向量叉积 [B, 3] × [B, 3] -> [B, 3]"""
    return torch.cross(u, v, dim=1)


# 向后兼容性别名函数
def compute_rotation_matrix_from_ortho6d_legacy(poses: torch.Tensor) -> torch.Tensor:
    """向后兼容函数，仅支持2D输入 [B, 6] -> [B, 3, 3]"""
    if poses.dim() != 2:
        raise ValueError("Legacy function only supports 2D input [B, 6]")
    return _compute_rotation_matrix_from_ortho6d_2d(poses)


def robust_compute_rotation_matrix_from_ortho6d_legacy(poses: torch.Tensor) -> torch.Tensor:
    """向后兼容鲁棒函数，仅支持2D输入 [B, 6] -> [B, 3, 3]"""
    if poses.dim() != 2:
        raise ValueError("Legacy function only supports 2D input [B, 6]")
    return _robust_compute_rotation_matrix_from_ortho6d_2d(poses)


def rot_to_ortho6d(rot: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵转换为6D旋转表示 [B, 3, 3] -> [B, 6]"""
    return rot.transpose(1, 2)[:, :2].reshape(-1, 6)


# 向后兼容性：保留旧的拼写错误函数名
rot_to_orthod6d = rot_to_ortho6d
