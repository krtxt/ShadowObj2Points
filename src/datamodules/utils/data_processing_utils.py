"""
Data Processing Utilities for SceneLeapPro Dataset

This module provides utility functions for data processing operations
including data index building, validation, and batch processing.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .transform_utils import (revert_leap_qpos_static,
                              transform_hand_poses_to_object_centric_frame,
                              validate_hand_pose_data)


def build_data_index(
    scene_dirs: List[str],
    collision_free_grasp_info: Dict[str, List],
    hand_pose_data: Dict[str, Dict],
    max_grasps_per_object: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build data index for dataset iteration.

    Args:
        scene_dirs: List of scene directory paths
        collision_free_grasp_info: Dictionary of collision-free grasp information
        hand_pose_data: Dictionary of hand pose data
        max_grasps_per_object: Maximum number of grasps per object

    Returns:
        List of data index entries
    """
    data_index = []

    for scene_dir_path_local in scene_dirs:
        scene_id = os.path.basename(scene_dir_path_local)
        current_scene_collision_info = collision_free_grasp_info.get(scene_id, [])

        # Get available depth view indices
        depth_view_indices = get_depth_view_indices_from_scene(scene_dir_path_local)
        if not depth_view_indices:
            continue

        for obj_grasp_entry in current_scene_collision_info:
            obj_name = obj_grasp_entry.get("object_name")
            obj_uid = obj_grasp_entry.get("uid")
            category_id_for_masking = obj_grasp_entry.get("object_index")

            if not obj_name or not obj_uid or category_id_for_masking is None:
                continue

            object_code = f"{obj_name}_{obj_uid}"
            if hand_pose_data.get(scene_id, {}).get(object_code) is None:
                continue

            collision_free_indices_for_obj = obj_grasp_entry.get(
                "collision_free_indices", []
            )
            if not collision_free_indices_for_obj:
                continue

            # Limit grasps per object if specified
            if (
                max_grasps_per_object is not None
                and len(collision_free_indices_for_obj) > max_grasps_per_object
            ):
                collision_free_indices_for_obj = collision_free_indices_for_obj[
                    :max_grasps_per_object
                ]

            # Add entries for all combinations of grasps and views
            for grasp_npy_idx in collision_free_indices_for_obj:
                for depth_view_idx in depth_view_indices:
                    data_index.append(
                        {
                            "scene_id": scene_id,
                            "object_code": object_code,
                            "category_id_for_masking": category_id_for_masking,
                            "depth_view_index": depth_view_idx,
                            "grasp_npy_idx": grasp_npy_idx,
                        }
                    )

    if not data_index:
        print("Warning: SceneLeapDataset data_index is empty after build.")

    return data_index


def get_depth_view_indices_from_scene(scene_dir_path: str) -> List[int]:
    """
    Get depth view indices from scene directory.

    Args:
        scene_dir_path: Path to scene directory

    Returns:
        List of depth view indices
    """
    depth_view_indices = []
    depth_dir_for_scan = os.path.join(scene_dir_path, "train_pbr/000000/depth/")

    if os.path.exists(depth_dir_for_scan) and os.path.isdir(depth_dir_for_scan):
        for f_name in sorted(os.listdir(depth_dir_for_scan)):
            if f_name.endswith(".png"):
                try:
                    depth_view_indices.append(int(os.path.splitext(f_name)[0]))
                except ValueError:
                    pass

    return depth_view_indices


def load_and_process_hand_pose_data(
    scene_dirs: List[str],
    collision_free_grasp_info: Dict[str, List],
    succ_grasp_dir: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load and process hand pose data for all scenes.

    Args:
        scene_dirs: List of scene directory paths
        collision_free_grasp_info: Dictionary of collision-free grasp information
        succ_grasp_dir: Directory containing successful grasp data

    Returns:
        Dictionary of processed hand pose data
    """
    hand_pose_data = {}

    for scene_dir_path in scene_dirs:
        scene_id = os.path.basename(scene_dir_path)
        hand_pose_data[scene_id] = {}

        for obj_info in collision_free_grasp_info.get(scene_id, []):
            obj_code = f"{obj_info['object_name']}_{obj_info['uid']}"
            hand_pose_path = os.path.join(succ_grasp_dir, f"{obj_code}.npy")

            if os.path.exists(hand_pose_path):
                try:
                    grasp_data_dict = np.load(hand_pose_path, allow_pickle=True).item()
                    raw_qpos_in_grasp_world_np = grasp_data_dict.get("grasp_qpos")
                    obj_pose_7d_np = grasp_data_dict.get("obj_pose")

                    if validate_hand_pose_data(
                        raw_qpos_in_grasp_world_np, obj_pose_7d_np
                    ):
                        qpos_gw_tensor = torch.from_numpy(
                            raw_qpos_in_grasp_world_np.astype(np.float32)
                        )
                        obj_pose_tensor = torch.from_numpy(
                            obj_pose_7d_np.astype(np.float32)
                        )

                        qpos_object_centric_tensor = (
                            transform_hand_poses_to_object_centric_frame(
                                qpos_gw_tensor, obj_pose_tensor
                            )
                        )
                        reverted_qpos = revert_leap_qpos_static(
                            qpos_object_centric_tensor
                        )
                        hand_pose_data[scene_id][obj_code] = reverted_qpos
                    else:
                        hand_pose_data[scene_id][obj_code] = None

                except Exception as e:
                    hand_pose_data[scene_id][obj_code] = None
            else:
                hand_pose_data[scene_id][obj_code] = None

    return hand_pose_data


def load_objectcentric_hand_pose_data(
    succ_grasp_dir: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load grasp poses and object poses for object-centric datasets.

    Args:
        succ_grasp_dir: Directory containing successful grasp .npy files

    Returns:
        Tuple of (hand_pose_data, obj_pose_data) dictionaries keyed by object_code
    """
    hand_pose_data: Dict[str, torch.Tensor] = {}
    obj_pose_data: Dict[str, torch.Tensor] = {}

    if not os.path.exists(succ_grasp_dir):
        raise FileNotFoundError(
            f"Successful grasp directory not found: {succ_grasp_dir}"
        )

    npy_files = [f for f in os.listdir(succ_grasp_dir) if f.endswith(".npy")]
    if not npy_files:
        raise ValueError(f"No .npy files found in {succ_grasp_dir}")

    for npy_file in npy_files:
        object_code = os.path.splitext(npy_file)[0]
        npy_path = os.path.join(succ_grasp_dir, npy_file)

        try:
            grasp_data_dict = np.load(npy_path, allow_pickle=True).item()
            raw_qpos_in_grasp_world_np = grasp_data_dict.get("grasp_qpos")
            obj_pose_7d_np = grasp_data_dict.get("obj_pose")

            if not validate_hand_pose_data(raw_qpos_in_grasp_world_np, obj_pose_7d_np):
                continue

            qpos_gw_tensor = torch.from_numpy(
                raw_qpos_in_grasp_world_np.astype(np.float32)
            )
            obj_pose_tensor = torch.from_numpy(obj_pose_7d_np.astype(np.float32))
            reverted_qpos = revert_leap_qpos_static(qpos_gw_tensor)

            hand_pose_data[object_code] = reverted_qpos
            obj_pose_data[object_code] = obj_pose_tensor
        except Exception:
            continue

    if not hand_pose_data:
        raise ValueError("No valid hand pose data loaded")

    return hand_pose_data, obj_pose_data


def load_bodex_hand_pose_data(
    succ_grasp_dir: str,
) -> Dict[str, torch.Tensor]:
    """
    Load Bodex shadow hand grasp poses from hierarchical directory structure.

    Bodex数据结构:
    - succ_grasp_dir/{object_name}/floating/scale{xxx}.npy
    - 每个npy包含: grasp_qpos (N_grasps, 29) [trans(3), quat(4), joints(22)]
    
    Args:
        succ_grasp_dir: Root directory containing Bodex grasp data
                       (e.g., data/bodex/bodex_shadow/succ_collect)
    
    Returns:
        Dictionary mapping scene_id -> tensor(N, 29)
        scene_id格式: "{object_name}_scale{scale_value}"
    
    Raises:
        FileNotFoundError: If succ_grasp_dir does not exist
        ValueError: If no valid grasp data is found
    """
    import logging
    logger = logging.getLogger(__name__)
    
    hand_pose_data: Dict[str, torch.Tensor] = {}
    
    if not os.path.exists(succ_grasp_dir):
        raise FileNotFoundError(
            f"Bodex grasp directory not found: {succ_grasp_dir}"
        )
    
    # 遍历所有物体目录
    object_dirs = [d for d in os.listdir(succ_grasp_dir) 
                   if os.path.isdir(os.path.join(succ_grasp_dir, d))]
    
    if not object_dirs:
        raise ValueError(f"No object directories found in {succ_grasp_dir}")
    
    total_scenes = 0
    total_grasps = 0
    
    for object_name in object_dirs:
        floating_dir = os.path.join(succ_grasp_dir, object_name, "floating")
        
        if not os.path.exists(floating_dir):
            logger.debug(f"Skipping {object_name}: floating dir not found")
            continue
        
        # 查找所有scale*.npy文件
        npy_files = [f for f in os.listdir(floating_dir) 
                     if f.startswith("scale") and f.endswith(".npy")]
        
        for npy_file in npy_files:
            npy_path = os.path.join(floating_dir, npy_file)
            
            try:
                # 加载npy文件
                grasp_data_dict = np.load(npy_path, allow_pickle=True).item()
                grasp_qpos_np = grasp_data_dict.get("grasp_qpos")
                
                if grasp_qpos_np is None:
                    logger.warning(f"No grasp_qpos in {npy_path}")
                    continue
                
                # 验证数据形状
                if not isinstance(grasp_qpos_np, np.ndarray) or grasp_qpos_np.ndim != 2:
                    logger.warning(f"Invalid grasp_qpos shape in {npy_path}: {grasp_qpos_np.shape if isinstance(grasp_qpos_np, np.ndarray) else type(grasp_qpos_np)}")
                    continue
                
                if grasp_qpos_np.shape[1] != 29:
                    logger.warning(f"Unexpected grasp_qpos dimension in {npy_path}: expected 29, got {grasp_qpos_np.shape[1]}")
                    continue
                
                # 提取scale值: scale010.npy -> 0.10
                scale_str = npy_file.replace("scale", "").replace(".npy", "")
                try:
                    scale_int = int(scale_str)
                    scale_value = scale_int / 100.0
                except ValueError:
                    logger.warning(f"Failed to parse scale from filename: {npy_file}")
                    continue
                
                # 构建scene_id
                scene_id = f"{object_name}_scale{scale_value}"
                
                # 转换为tensor（保持原始顺序: [trans(3), quat(4), joints(22)]）
                grasp_qpos_tensor = torch.from_numpy(grasp_qpos_np.astype(np.float32))
                hand_pose_data[scene_id] = grasp_qpos_tensor
                
                total_scenes += 1
                total_grasps += grasp_qpos_tensor.shape[0]
                
            except Exception as e:
                logger.warning(f"Failed to load {npy_path}: {e}")
                continue
    
    if not hand_pose_data:
        raise ValueError(f"No valid Bodex hand pose data loaded from {succ_grasp_dir}")
    
    logger.info(f"Loaded Bodex data: {total_scenes} scenes, {total_grasps} total grasps")
    
    return hand_pose_data


def load_scene_metadata(scene_dirs: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    Load scene metadata including instance maps, scene ground truth, and collision info.

    Args:
        scene_dirs: List of scene directory paths

    Returns:
        Tuple of (instance_maps, scene_gt, collision_free_grasp_info)
    """
    instance_maps = {}
    scene_gt = {}
    collision_free_grasp_info = {}

    for scene_dir_path in scene_dirs:
        scene_id = os.path.basename(scene_dir_path)

        # Load instance attribute maps
        instance_map_path = os.path.join(scene_dir_path, "instance_attribute_maps.json")
        if os.path.exists(instance_map_path):
            instance_maps[scene_id] = json.load(open(instance_map_path, "r"))
        else:
            instance_maps[scene_id] = {}

        # Load scene ground truth
        scene_gt_path = os.path.join(
            scene_dir_path, "train_pbr/000000", "scene_gt.json"
        )
        if os.path.exists(scene_gt_path):
            scene_gt[scene_id] = json.load(open(scene_gt_path, "r"))
        else:
            scene_gt[scene_id] = {}

        # Load collision-free grasp information
        collision_free_grasp_path = os.path.join(
            scene_dir_path, "collision_free_grasp_indices.json"
        )
        if os.path.exists(collision_free_grasp_path):
            collision_free_grasp_info[scene_id] = json.load(
                open(collision_free_grasp_path, "r")
            )
        else:
            collision_free_grasp_info[scene_id] = []

    return instance_maps, scene_gt, collision_free_grasp_info


def validate_dataset_configuration(
    root_dir: str, succ_grasp_dir: str, obj_root_dir: str, mode: str
) -> bool:
    """
    Validate dataset configuration parameters.

    Args:
        root_dir: Root directory path
        succ_grasp_dir: Successful grasp directory path
        obj_root_dir: Object root directory path
        mode: Coordinate system mode

    Returns:
        True if configuration is valid, False otherwise
    """
    # Validate directories exist
    if not os.path.exists(root_dir):
        print(f"Error: Root directory does not exist: {root_dir}")
        return False

    if not os.path.exists(succ_grasp_dir):
        print(f"Error: Successful grasp directory does not exist: {succ_grasp_dir}")
        return False

    if not os.path.exists(obj_root_dir):
        print(f"Error: Object root directory does not exist: {obj_root_dir}")
        return False

    # Validate mode
    valid_modes = [
        "object_centric",
        "camera_centric",
        "camera_centric_obj_mean_normalized",
        "camera_centric_scene_mean_normalized",
    ]
    if mode not in valid_modes:
        print(f"Error: Invalid mode '{mode}'. Must be one of {valid_modes}")
        return False

    return True


def collate_batch_data(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch data for DataLoader.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch dictionary
    """
    if not batch:
        return {}

    input_dict = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            try:
                input_dict[k] = torch.stack([sample[k] for sample in batch])
            except RuntimeError:
                input_dict[k] = torch.nn.utils.rnn.pad_sequence(
                    [sample[k] for sample in batch], batch_first=True, padding_value=0
                )
        else:
            input_dict[k] = [sample[k] for sample in batch]

    return input_dict


def collate_variable_grasps_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Advanced collate function for batches with variable number of grasps.

    This function handles the complex collation logic needed for ForMatchSceneLeapProDataset,
    where 'hand_model_pose' and 'se3' fields contain variable numbers of grasps that need
    to be padded to create dense tensors.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch dictionary with padded tensors
    """
    if not batch:
        return {}

    # Filter out None items or non-dict items from the batch
    batch = [item for item in batch if isinstance(item, dict)]
    if not batch:
        return {}

    # Determine fallback dtype and device from the batch
    fallback_dtype = torch.float32
    fallback_device = "cpu"

    for item_dict in batch:
        found_ref = False
        for val in item_dict.values():
            if isinstance(val, torch.Tensor) and val.numel() > 0:
                fallback_dtype = val.dtype
                fallback_device = val.device
                found_ref = True
                break
        if found_ref:
            break

    # Collect all keys from all items
    all_keys = set()
    for item_dict in batch:
        all_keys.update(item_dict.keys())

    collated_output = {}

    for key in all_keys:
        current_key_items = [item_dict.get(key) for item_dict in batch]

        if key == "hand_model_pose":  # Target: (batch_size, max_N, 23)
            collated_output[key] = _collate_hand_model_pose(
                current_key_items, fallback_dtype, fallback_device, len(batch)
            )
        elif key == "se3":  # Target: (batch_size, max_N, 4, 4)
            collated_output[key] = _collate_se3_matrices(
                current_key_items, fallback_dtype, fallback_device, len(batch)
            )
        elif key in [
            "object_mask",
            "obj_verts",
            "obj_faces",
            "positive_prompt",
            "negative_prompts",
            "error",
        ]:
            # Keep these as lists
            collated_output[key] = current_key_items
        else:
            # Default collation for other keys
            try:
                collated_output[key] = torch.utils.data.dataloader.default_collate(
                    current_key_items
                )
            except (RuntimeError, TypeError, AttributeError):
                collated_output[key] = current_key_items  # Fallback to list

    return collated_output


def _collate_hand_model_pose(
    items: List[Any], fallback_dtype: torch.dtype, fallback_device: str, batch_size: int
) -> torch.Tensor:
    """
    Collate hand_model_pose tensors with padding.

    Args:
        items: List of hand_model_pose tensors (each can be (N, D) or None)
        fallback_dtype: Fallback data type
        fallback_device: Fallback device
        batch_size: Size of the batch

    Returns:
        Padded tensor of shape (batch_size, max_N, D)
    """
    pose_dim = None
    for t in items:
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            pose_dim = t.shape[1]
            break
    if pose_dim is None:
        pose_dim = 23

    # Find maximum number of grasps
    max_n = 0
    for t in items:
        if (
            isinstance(t, torch.Tensor)
            and t.ndim == 2
            and t.shape[1] == pose_dim
        ):
            max_n = max(max_n, t.shape[0])

    padded_tensors = []
    for t in items:
        item_dtype = fallback_dtype
        item_device = fallback_device
        if isinstance(t, torch.Tensor):
            item_dtype = t.dtype
            item_device = t.device

        if (
            isinstance(t, torch.Tensor)
            and t.ndim == 2
            and t.shape[1] == pose_dim
        ):
            num_grasps = t.shape[0]
            if num_grasps == max_n:
                padded_tensors.append(t)
            else:  # num_grasps < max_n
                padding_size = max_n - num_grasps
                padding = torch.zeros(
                    (padding_size, pose_dim), dtype=item_dtype, device=item_device
                )
                padded_tensors.append(torch.cat([t, padding], dim=0))
        else:  # t is None, or not a tensor, or shape mismatch. Pad to max_n.
            padded_tensors.append(
                torch.zeros((max_n, pose_dim), dtype=item_dtype, device=item_device)
            )

    if padded_tensors:
        return torch.stack(padded_tensors)
    elif batch_size > 0:
        # Batch was non-empty but all tensors were empty -> maintain pose_dim.
        return torch.empty(
            (batch_size, 0, pose_dim), dtype=fallback_dtype, device=fallback_device
        )
    else:  # batch is empty
        return torch.empty(0)


def _collate_se3_matrices(
    items: List[Any], fallback_dtype: torch.dtype, fallback_device: str, batch_size: int
) -> torch.Tensor:
    """
    Collate SE3 matrices with padding.

    Args:
        items: List of SE3 tensors (each can be (N, 4, 4) or None)
        fallback_dtype: Fallback data type
        fallback_device: Fallback device
        batch_size: Size of the batch

    Returns:
        Padded tensor of shape (batch_size, max_N, 4, 4)
    """
    # Find maximum number of grasps
    max_n = 0
    for t in items:
        if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[1:] == (4, 4):
            max_n = max(max_n, t.shape[0])

    padded_tensors = []
    for t in items:
        item_dtype = fallback_dtype
        item_device = fallback_device
        if isinstance(t, torch.Tensor):
            item_dtype = t.dtype
            item_device = t.device

        if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[1:] == (4, 4):
            num_grasps = t.shape[0]
            if num_grasps == max_n:
                padded_tensors.append(t)
            else:  # num_grasps < max_n
                padding_size = max_n - num_grasps
                padding = torch.zeros(
                    (padding_size, 4, 4), dtype=item_dtype, device=item_device
                )
                padded_tensors.append(torch.cat([t, padding], dim=0))
        else:  # t is None, or not a tensor, or wrong shape. Pad to max_n.
            padded_tensors.append(
                torch.zeros((max_n, 4, 4), dtype=item_dtype, device=item_device)
            )

    if padded_tensors:
        return torch.stack(padded_tensors)
    elif batch_size > 0:
        return torch.empty(
            (batch_size, 0, 4, 4), dtype=fallback_dtype, device=fallback_device
        )
    else:  # batch is empty
        return torch.empty(0)
