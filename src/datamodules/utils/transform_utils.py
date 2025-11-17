"""
Coordinate Transformation Utilities for SceneLeapPro Dataset

This module provides utility functions for coordinate transformations, including
hand pose transformations, LEAP format conversions, and SE(3) matrix operations.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (matrix_to_quaternion, quaternion_invert,
                                  quaternion_multiply, quaternion_to_matrix)


def transform_hand_poses_to_object_centric_frame(
    raw_hand_poses_gw: torch.Tensor, object_pose_in_gw_7d: torch.Tensor
) -> torch.Tensor:
    """
    Transform hand poses from grasp world to object-centric frame.

    Args:
        raw_hand_poses_gw: Hand poses in grasp world frame (N, 23) [P_gw, Q_gw_wxyz, Joints]
        object_pose_in_gw_7d: Object pose in grasp world (7,) [t_obj_gw, q_obj_gw_wxyz]

    Returns:
        Hand poses in object-centric frame (N, 23)
    """
    N = raw_hand_poses_gw.shape[0]
    device = raw_hand_poses_gw.device

    # Extract components
    P_h_gw = raw_hand_poses_gw[:, :3]  # (N, 3) - hand positions
    Q_h_gw = raw_hand_poses_gw[:, 3:7]  # (N, 4) - hand orientations [w,x,y,z]
    Joints = raw_hand_poses_gw[:, 7:]  # (N, 16) - joint angles

    # Expand object pose for batch processing
    t_obj_gw = object_pose_in_gw_7d[:3].unsqueeze(0).expand(N, -1)  # (N, 3)
    q_obj_gw = object_pose_in_gw_7d[3:7].unsqueeze(0).expand(N, -1)  # (N, 4)

    # Compute inverse object transformation
    q_obj_gw_inv = quaternion_invert(q_obj_gw)  # (N, 4)
    R_obj_gw_inv = quaternion_to_matrix(q_obj_gw_inv)  # (N, 3, 3)

    # Transform hand positions: P_h_model = R_obj_gw_inv @ (P_h_gw - t_obj_gw)
    P_h_model = torch.bmm(R_obj_gw_inv, (P_h_gw - t_obj_gw).unsqueeze(-1)).squeeze(
        -1
    )  # (N, 3)

    # Transform hand orientations: Q_h_model = q_obj_gw_inv * Q_h_gw
    Q_h_model = quaternion_multiply(q_obj_gw_inv, Q_h_gw)  # (N, 4)

    return torch.cat([P_h_model, Q_h_model, Joints], dim=-1)


def revert_leap_qpos_static(qpos_decomposed: torch.Tensor) -> torch.Tensor:
    """
    Convert LEAP hand pose back to original format.

    Supports both single poses (23,) and batched poses (N, 23).

    Args:
        qpos_decomposed: LEAP format hand poses [P, Q_wxyz, Joints]

    Returns:
        Original format hand poses with same batch structure
    """
    # Ensure input is float and handle batching
    qpos_decomposed_tensor = qpos_decomposed.float()
    is_batched = qpos_decomposed_tensor.dim() > 1

    if not is_batched:
        qpos_decomposed_tensor = qpos_decomposed_tensor.unsqueeze(0)

    # Handle empty batch case
    if qpos_decomposed_tensor.shape[0] == 0:
        empty_shape = (0, 23) if is_batched else (23,)
        return torch.zeros(
            empty_shape, dtype=torch.float32, device=qpos_decomposed_tensor.device
        )

    B = qpos_decomposed_tensor.shape[0]
    device = qpos_decomposed_tensor.device

    # Extract components
    P_new = qpos_decomposed_tensor[..., :3]  # (B, 3) - positions
    quat_new = qpos_decomposed_tensor[..., 3:7]  # (B, 4) - quaternions [w,x,y,z]
    joint_angles = qpos_decomposed_tensor[..., 7:]  # (B, 16) - joint angles

    # Convert quaternions to rotation matrices
    R_new = quaternion_to_matrix(quat_new)  # (B, 3, 3)

    # Apply LEAP-specific transformations
    delta_rot_quat = torch.tensor(
        [0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32
    )
    delta_rot_quat = delta_rot_quat.view(1, 4).expand(B, -1)
    DeltaR = quaternion_to_matrix(delta_rot_quat)  # (B, 3, 3)

    # Apply offset transformation
    T_offset_local = torch.tensor([0.0, 0.0, 0.1], device=device, dtype=torch.float32)
    T_offset_local = T_offset_local.view(1, 3, 1).expand(B, -1, -1)

    Offset_world = torch.matmul(R_new, T_offset_local).squeeze(-1)  # (B, 3)
    P_orig = P_new + Offset_world

    # Apply rotation transformation
    R_orig = torch.matmul(R_new, DeltaR)  # (B, 3, 3)
    quat_orig = matrix_to_quaternion(R_orig)  # (B, 4) [w,x,y,z]

    # Reconstruct original pose
    qpos_original_tensor = torch.cat([P_orig, quat_orig, joint_angles], dim=-1)

    # Return with original batch structure
    if not is_batched:
        qpos_original_tensor = qpos_original_tensor.squeeze(0)

    return qpos_original_tensor


def transform_hand_poses_omf_to_cf(
    hand_poses_omf: torch.Tensor, R_omf_to_cf_np: np.ndarray, t_omf_to_cf_np: np.ndarray
) -> torch.Tensor:
    """
    Transform hand poses from Object Model Frame (OMF) to Camera Frame (CF).

    Args:
        hand_poses_omf: Hand poses in OMF (N, 23) or (23,) [P_omf, Q_omf_wxyz, Joints]
        R_omf_to_cf_np: Rotation matrix from OMF to CF (3, 3)
        t_omf_to_cf_np: Translation vector from OMF to CF (3,)

    Returns:
        Hand poses in CF with same batch structure [P_cf, Q_cf_wxyz, Joints]
    """
    if not isinstance(hand_poses_omf, torch.Tensor) or hand_poses_omf.numel() == 0:
        return hand_poses_omf

    # Handle batching
    is_batched = hand_poses_omf.dim() > 1
    temp_hand_poses_omf = hand_poses_omf if is_batched else hand_poses_omf.unsqueeze(0)

    N = temp_hand_poses_omf.shape[0]
    device = temp_hand_poses_omf.device

    # Extract components
    P_omf = temp_hand_poses_omf[:, :3]  # (N, 3) - positions
    Q_omf_wxyz = temp_hand_poses_omf[:, 3:7]  # (N, 4) - quaternions [w,x,y,z]
    Joints = temp_hand_poses_omf[:, 7:]  # (N, 16) - joint angles

    # Convert transformation to tensors
    R_omf_to_cf = torch.from_numpy(R_omf_to_cf_np.astype(np.float32)).to(
        device
    )  # (3, 3)
    t_omf_to_cf = (
        torch.from_numpy(t_omf_to_cf_np.astype(np.float32)).to(device).unsqueeze(0)
    )  # (1, 3)

    # Transform positions: P_cf = P_omf @ R_omf_to_cf.T + t_omf_to_cf
    P_cf = torch.matmul(P_omf, R_omf_to_cf.T) + t_omf_to_cf  # (N, 3)

    # Transform orientations: Q_cf = Q_R_omf_to_cf * Q_omf_wxyz
    Q_R_omf_to_cf = (
        matrix_to_quaternion(R_omf_to_cf).unsqueeze(0).expand(N, -1)
    )  # (N, 4)
    Q_cf_wxyz = quaternion_multiply(Q_R_omf_to_cf, Q_omf_wxyz)  # (N, 4)

    # Reconstruct hand poses
    hand_poses_cf = torch.cat([P_cf, Q_cf_wxyz, Joints], dim=-1)

    return hand_poses_cf if is_batched else hand_poses_cf.squeeze(0)


def create_se3_matrix_from_pose(
    position: torch.Tensor, quaternion: torch.Tensor
) -> torch.Tensor:
    """
    Create SE(3) transformation matrix from position and quaternion.

    Args:
        position: Translation vector (3,)
        quaternion: Quaternion [w,x,y,z] (4,)

    Returns:
        SE(3) transformation matrix (4, 4)
    """
    se3_matrix = torch.eye(4, device=position.device, dtype=position.dtype)

    # Handle zero quaternion case
    current_Q_for_R = quaternion
    if torch.norm(quaternion) < 1e-6:
        current_Q_for_R = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=position.device, dtype=position.dtype
        )

    R_matrix = quaternion_to_matrix(current_Q_for_R)
    se3_matrix[:3, :3] = R_matrix
    se3_matrix[:3, 3] = position

    return se3_matrix


def reorder_hand_pose_components(hand_pose: torch.Tensor) -> torch.Tensor:
    """
    Reorder hand pose components from [P, Q, Joints] to [P, Joints, Q] format.

    Args:
        hand_pose: Hand pose tensor (23,) in format [P, Q_wxyz, Joints]

    Returns:
        Reordered hand pose tensor (23,) in format [P, Joints, Q_wxyz]
    """
    if hand_pose.shape[0] != 23:
        return torch.zeros(23, dtype=torch.float32)

    P = hand_pose[:3]  # Position (3,)
    Q_wxyz = hand_pose[3:7]  # Quaternion (4,)
    Joints = hand_pose[7:]  # Joint angles (16,)

    return torch.cat([P, Joints, Q_wxyz], dim=-1)


def extract_object_name_from_code(object_code: str) -> str:
    """
    Extract object name from object code.

    Args:
        object_code: Object code in format "name_uid" where uid is a long hex string

    Returns:
        Extracted object name
    """
    if "_" not in object_code:
        return object_code

    # Split by underscore and remove the last part (UID)
    parts = object_code.split("_")
    if len(parts) <= 1:
        return object_code

    # The UID is typically a long hex string, so we remove the last part
    # and rejoin the rest as the object name
    return "_".join(parts[:-1])


def validate_hand_pose_data(
    raw_qpos: Optional[np.ndarray], obj_pose: Optional[np.ndarray]
) -> bool:
    """
    Validate hand pose data shapes and types.

    Args:
        raw_qpos: Raw hand pose data
        obj_pose: Object pose data

    Returns:
        True if data is valid, False otherwise
    """
    return (
        raw_qpos is not None
        and raw_qpos.ndim == 2
        and raw_qpos.shape[1] == 23
        and obj_pose is not None
        and obj_pose.shape == (7,)
    )


def transform_point_cloud_with_se3(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    """
    Transform point cloud using SE(3) transformation.

    Args:
        points: Point cloud coordinates (N, 3)
        rotation_matrix: Rotation matrix (3, 3)
        translation_vector: Translation vector (3,)
        inverse: Whether to apply inverse transformation

    Returns:
        Transformed point cloud (N, 3)
    """
    if points.shape[0] == 0:
        return points

    if inverse:
        # Apply inverse transformation: R^T @ (p - t)
        transformed_points = np.dot(
            points - translation_vector.reshape(1, 3), rotation_matrix.T
        )
    else:
        # Apply forward transformation: R @ p + t
        transformed_points = np.dot(
            points, rotation_matrix.T
        ) + translation_vector.reshape(1, 3)

    return transformed_points


def generate_negative_prompts(
    scene_collision_info: list, current_obj_name: str, num_neg_prompts: int = 4
) -> list:
    """
    Generate negative prompts from other objects in the scene.

    Args:
        scene_collision_info: List of collision-free grasp info for the scene
        current_obj_name: Name of the current target object
        num_neg_prompts: Number of negative prompts to generate

    Returns:
        List of negative prompt strings
    """
    # Collect all other object names in the scene
    other_object_names = []

    for obj_info in scene_collision_info:
        obj_name = obj_info.get("object_name")
        if obj_name and obj_name != current_obj_name:
            other_object_names.append(obj_name)

    # Remove duplicates while preserving order
    unique_object_names = []
    seen_names = set()
    for name in other_object_names:
        if name not in seen_names:
            unique_object_names.append(name)
            seen_names.add(name)

    # Generate negative prompts list according to required count
    if len(unique_object_names) == 0:
        # No other objects in scene - use empty strings
        negative_prompts = [""] * num_neg_prompts
    elif len(unique_object_names) < num_neg_prompts:
        # Fewer objects than required - pad with last object name
        negative_prompts = unique_object_names + [unique_object_names[-1]] * (
            num_neg_prompts - len(unique_object_names)
        )
    else:
        # More objects than required - take first N objects
        negative_prompts = unique_object_names[:num_neg_prompts]

    return negative_prompts


def get_camera_transform(scene_gt_for_view: list, target_obj_id: int) -> tuple:
    """
    Extract camera transformation for target object from scene ground truth.

    Args:
        scene_gt_for_view: Scene ground truth data for specific view
        target_obj_id: Target object ID to find transformation for

    Returns:
        Tuple of (rotation_matrix, translation_vector) or (None, None) if not found
    """
    for obj_gt in scene_gt_for_view:
        if obj_gt.get("obj_id") == target_obj_id:
            cam_R_m2c_list = obj_gt.get("cam_R_m2c")
            cam_t_m2c_list = obj_gt.get("cam_t_m2c")
            if cam_R_m2c_list is None or cam_t_m2c_list is None:
                return None, None
            try:
                cam_R_m2c = np.array(cam_R_m2c_list).reshape(3, 3)
                cam_t_m2c = np.array(cam_t_m2c_list) / 1000.0  # Convert from mm to m
                return cam_R_m2c, cam_t_m2c
            except ValueError:
                return None, None
    return None, None


def get_specific_hand_pose(
    hand_pose_data: dict,
    scene_id: str,
    object_code: str,
    grasp_npy_idx: int,
    default_pose_dim: int = 23,
) -> torch.Tensor:
    """
    Get specific hand pose from hand pose data.

    Args:
        hand_pose_data: Dictionary containing hand pose data
        scene_id: Scene identifier
        object_code: Object code identifier
        grasp_npy_idx: Grasp index
        default_pose_dim: Default pose dimension

    Returns:
        Hand pose tensor or zero tensor if not found
    """
    all_reverted_poses_tensor = hand_pose_data.get(scene_id, {}).get(object_code)

    if all_reverted_poses_tensor is None:
        return torch.zeros(default_pose_dim, dtype=torch.float32)

    if not isinstance(all_reverted_poses_tensor, torch.Tensor):
        return torch.zeros(default_pose_dim, dtype=torch.float32)

    if (
        all_reverted_poses_tensor.ndim != 2
        or all_reverted_poses_tensor.shape[1] != default_pose_dim
    ):
        return torch.zeros(default_pose_dim, dtype=torch.float32)

    num_available_poses = all_reverted_poses_tensor.shape[0]
    if 0 <= grasp_npy_idx < num_available_poses:
        return all_reverted_poses_tensor[grasp_npy_idx]
    else:
        return torch.zeros(default_pose_dim, dtype=torch.float32)


def apply_object_pose_to_vertices(
    vertices: torch.Tensor, object_pose: torch.Tensor
) -> torch.Tensor:
    """
    Transform vertices from object model frame to grasp/world frame.

    Args:
        vertices: Vertex positions in object model frame (V, 3)
        object_pose: Object pose (7,) [t_obj(3), q_obj(4)] in target frame

    Returns:
        Transformed vertices in target frame (V, 3)
    """
    if vertices.numel() == 0:
        return vertices

    t_obj = object_pose[:3]
    q_obj = object_pose[3:7]
    R_obj = quaternion_to_matrix(q_obj)
    return torch.matmul(vertices, R_obj.T) + t_obj.unsqueeze(0)


def center_points_xy(points: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    Center points on the XY plane by subtracting the center translation.

    Args:
        points: Input points (N, 3)
        center: Center translation (3,)

    Returns:
        Centered points (N, 3) with original Z preserved
    """
    if points.numel() == 0:
        return points

    offset = torch.zeros(3, dtype=points.dtype, device=points.device)
    offset[:2] = center[:2].to(device=points.device, dtype=points.dtype)
    return points - offset.unsqueeze(0)


def center_hand_poses_xy(
    hand_poses: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """
    Center hand poses on the XY plane.

    Args:
        hand_poses: Hand pose tensor (N, 23) [P, Q, Joints]
        center: Center translation (3,)

    Returns:
        Centered hand pose tensor (N, 23)
    """
    if hand_poses.numel() == 0:
        return hand_poses

    positions = hand_poses[:, :3]
    offset = torch.zeros(3, dtype=hand_poses.dtype, device=hand_poses.device)
    offset[:2] = center[:2].to(device=hand_poses.device, dtype=hand_poses.dtype)
    centered_positions = positions - offset.unsqueeze(0)
    return torch.cat([centered_positions, hand_poses[:, 3:]], dim=1)


def reorder_hand_pose_batch(hand_poses: torch.Tensor) -> torch.Tensor:
    """
    Reorder batched hand poses from [P, Q, Joints] to [P, Joints, Q].

    Args:
        hand_poses: Hand pose tensor (N, 23) in [P, Q, Joints] format

    Returns:
        Reordered hand pose tensor (N, 23) in [P, Joints, Q] format
    """
    if (
        not isinstance(hand_poses, torch.Tensor)
        or hand_poses.ndim != 2
        or hand_poses.shape[1] != 23
    ):
        batch_dim = (
            hand_poses.shape[0]
            if isinstance(hand_poses, torch.Tensor) and hand_poses.ndim > 0
            else 0
        )
        dtype = (
            hand_poses.dtype if isinstance(hand_poses, torch.Tensor) else torch.float32
        )
        device = (
            hand_poses.device
            if isinstance(hand_poses, torch.Tensor)
            else torch.device("cpu")
        )
        return torch.zeros((batch_dim, 23), dtype=dtype, device=device)

    positions = hand_poses[:, :3]
    quaternions = hand_poses[:, 3:7]
    joints = hand_poses[:, 7:]
    return torch.cat([positions, joints, quaternions], dim=1)


def create_se3_matrices_from_pose_batch(
    positions: torch.Tensor, quaternions: torch.Tensor
) -> torch.Tensor:
    """
    Create batched SE(3) matrices from positions and quaternions.

    Args:
        positions: Translation vectors (N, 3)
        quaternions: Quaternions (N, 4) in wxyz convention

    Returns:
        SE(3) matrices (N, 4, 4)
    """
    if positions.shape[0] == 0:
        return torch.zeros((0, 4, 4), dtype=positions.dtype, device=positions.device)

    corrected_quats = quaternions.clone()
    norms = torch.norm(corrected_quats, dim=1, keepdim=True)
    zero_mask = norms.squeeze(1) < 1e-6
    if zero_mask.any():
        identity_q = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=corrected_quats.dtype,
            device=corrected_quats.device,
        )
        corrected_quats[zero_mask] = identity_q

    rotation = quaternion_to_matrix(corrected_quats)
    se3 = torch.zeros(
        (positions.shape[0], 4, 4),
        dtype=positions.dtype,
        device=positions.device,
    )
    se3[:, :3, :3] = rotation
    se3[:, :3, 3] = positions
    se3[:, 3, 3] = 1.0
    return se3


def quaternion_to_rotation_6d(quaternions: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为 6D 旋转表示（旋转矩阵的前两列）。
    
    Args:
        quaternions: (N, 4) 四元数 [w, x, y, z] 或 [x, y, z, w]，根据 pytorch3d 约定
    
    Returns:
        rotation_6d: (N, 6) 6D 旋转表示
    """
    rotation_matrices = quaternion_to_matrix(quaternions)  # (N, 3, 3)
    rotation_6d = torch.cat([rotation_matrices[:, :, 0], rotation_matrices[:, :, 1]], dim=-1)  # (N, 6)
    return rotation_6d


def rotation_6d_to_quaternion(rotation_6d: torch.Tensor) -> torch.Tensor:
    """
    将 6D 旋转表示转换回四元数。
    
    Args:
        rotation_6d: (N, 6) 6D 旋转表示
    
    Returns:
        quaternions: (N, 4) 四元数
    """
    # 提取前两列
    col0 = rotation_6d[:, :3]  # (N, 3)
    col1 = rotation_6d[:, 3:6]  # (N, 3)
    
    # 正交化：Gram-Schmidt
    col0_normalized = col0 / (torch.norm(col0, dim=-1, keepdim=True) + 1e-8)
    col1_orthogonal = col1 - (torch.sum(col0_normalized * col1, dim=-1, keepdim=True) * col0_normalized)
    col1_normalized = col1_orthogonal / (torch.norm(col1_orthogonal, dim=-1, keepdim=True) + 1e-8)
    
    # 第三列通过叉积得到
    col2_normalized = torch.cross(col0_normalized, col1_normalized, dim=-1)
    
    # 组装旋转矩阵
    rotation_matrices = torch.stack([col0_normalized, col1_normalized, col2_normalized], dim=-1)  # (N, 3, 3)
    
    # 转换为四元数
    quaternions = matrix_to_quaternion(rotation_matrices)  # (N, 4)
    return quaternions
