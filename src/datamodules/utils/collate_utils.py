"""
Collation utilities for SceneLeapPro datasets.

This module provides reusable functions for tensor padding, batch collation,
and device/dtype inference to eliminate code duplication in dataset collate_fn methods.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .dataset_config import CONFIG


def infer_batch_dtype_device(
    batch: List[Dict[str, Any]]
) -> Tuple[torch.dtype, torch.device]:
    """
    Infer dtype and device from a batch of data items.

    Searches through all tensor values in the batch to find a reference
    tensor with non-zero elements to determine the appropriate dtype and device.

    Args:
        batch: List of data dictionaries

    Returns:
        Tuple[torch.dtype, torch.device]: Inferred dtype and device
    """
    fallback_dtype = CONFIG.DEFAULT_DTYPE
    fallback_device = torch.device(CONFIG.DEFAULT_DEVICE)

    for item_dict in batch:
        for val in item_dict.values():
            if isinstance(val, torch.Tensor) and val.numel() > 0:
                return val.dtype, val.device

    return fallback_dtype, fallback_device


def pad_tensor_batch(
    tensors: List[torch.Tensor],
    target_shape: Tuple[int, ...],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad a list of tensors to the same shape and stack them.

    Args:
        tensors: List of tensors to pad and stack
        target_shape: Target shape for padding (excluding batch dimension)
        padding_value: Value to use for padding

    Returns:
        torch.Tensor: Stacked tensor with shape (batch_size, *target_shape)
    """
    if not tensors:
        return torch.empty(0)

    # Get reference tensor for dtype and device
    ref_tensor = tensors[0]
    dtype = ref_tensor.dtype
    device = ref_tensor.device

    padded_tensors = []
    for tensor in tensors:
        if tensor.shape == target_shape:
            padded_tensors.append(tensor)
        else:
            # Create padding tensor
            padding_tensor = torch.full(
                target_shape, padding_value, dtype=dtype, device=device
            )

            # Copy original data to the beginning of padding tensor
            if tensor.numel() > 0:
                # Handle different dimensionalities
                if len(target_shape) == 1:  # 1D case
                    min_size = min(tensor.shape[0], target_shape[0])
                    padding_tensor[:min_size] = tensor[:min_size]
                elif len(target_shape) == 2:  # 2D case
                    min_rows = min(tensor.shape[0], target_shape[0])
                    min_cols = min(tensor.shape[1], target_shape[1])
                    padding_tensor[:min_rows, :min_cols] = tensor[:min_rows, :min_cols]
                elif len(target_shape) == 3:  # 3D case
                    min_d0 = min(tensor.shape[0], target_shape[0])
                    min_d1 = min(tensor.shape[1], target_shape[1])
                    min_d2 = min(tensor.shape[2], target_shape[2])
                    padding_tensor[:min_d0, :min_d1, :min_d2] = tensor[
                        :min_d0, :min_d1, :min_d2
                    ]

            padded_tensors.append(padding_tensor)

    return torch.stack(padded_tensors)


def collate_variable_length_tensors(
    tensors: List[Optional[Any]],
    expected_suffix_shape: Tuple[int, ...],
    fallback_dtype: torch.dtype = None,
    fallback_device: torch.device = None,
) -> torch.Tensor:
    """
    Collate tensors with variable first dimension but fixed suffix dimensions.

    This function handles the common pattern in SceneLeapPro datasets where
    tensors have shape (N, *suffix_shape) with variable N but fixed suffix.

    Args:
        tensors: List of tensors with shape (N_i, *suffix_shape) or None
        expected_suffix_shape: Expected shape for all dimensions except the first
        fallback_dtype: Fallback dtype if no valid tensors found
        fallback_device: Fallback device if no valid tensors found

    Returns:
        torch.Tensor: Collated tensor with shape (batch_size, max_N, *suffix_shape)
    """
    if fallback_dtype is None:
        fallback_dtype = CONFIG.DEFAULT_DTYPE
    if fallback_device is None:
        fallback_device = torch.device(CONFIG.DEFAULT_DEVICE)

    # Helper to convert inputs to torch.Tensor when possible
    def to_tensor(x: Any) -> Optional[torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            try:
                return torch.from_numpy(x)
            except Exception:
                return None
        return None

    tensor_list: List[Optional[torch.Tensor]] = [to_tensor(t) for t in tensors]

    # Filter valid tensors and find max first dimension
    valid_tensors: List[Optional[torch.Tensor]] = []
    max_n = 0

    for tensor in tensor_list:
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == len(expected_suffix_shape) + 1
            and tuple(int(x) for x in tensor.shape[1:]) == tuple(expected_suffix_shape)
        ):
            valid_tensors.append(tensor)
            max_n = max(max_n, int(tensor.shape[0]))
        else:
            valid_tensors.append(None)

    # Handle empty case
    if max_n == 0:
        # Attempt dynamic suffix detection from data
        suffix_counts: Dict[Tuple[int, ...], int] = {}
        for t in tensor_list:
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                suf = tuple(int(x) for x in t.shape[1:])
                suffix_counts[suf] = suffix_counts.get(suf, 0) + 1

        if suffix_counts:
            detected_suffix = max(suffix_counts.items(), key=lambda kv: kv[1])[0]

            valid_tensors = []
            max_n = 0
            for t in tensor_list:
                if (
                    isinstance(t, torch.Tensor)
                    and t.ndim == len(detected_suffix) + 1
                    and tuple(int(x) for x in t.shape[1:]) == detected_suffix
                ):
                    valid_tensors.append(t)
                    max_n = max(max_n, int(t.shape[0]))
                else:
                    valid_tensors.append(None)

            if max_n == 0:
                batch_size = len(tensors)
                empty_shape = (batch_size, 0, *detected_suffix)
                return torch.empty(empty_shape, dtype=fallback_dtype, device=fallback_device)

            expected_suffix_shape = detected_suffix
        else:
            batch_size = len(tensors)
            empty_shape = (batch_size, 0, *expected_suffix_shape)
            return torch.empty(empty_shape, dtype=fallback_dtype, device=fallback_device)

    # Pad tensors to max_n
    padded_tensors = []
    for i, tensor in enumerate(valid_tensors):
        if tensor is not None:
            current_n = int(tensor.shape[0])
            if current_n == max_n:
                padded_tensors.append(tensor)
            else:
                # Pad to max_n
                padding_shape = (max_n - current_n, *expected_suffix_shape)
                padding = torch.zeros(
                    padding_shape, dtype=tensor.dtype, device=tensor.device
                )
                padded_tensor = torch.cat([tensor, padding], dim=0)
                padded_tensors.append(padded_tensor)
        else:
            # Create zero tensor with max_n size
            zero_shape = (max_n, *expected_suffix_shape)
            zero_tensor = torch.zeros(
                zero_shape, dtype=fallback_dtype, device=fallback_device
            )
            padded_tensors.append(zero_tensor)

    return torch.stack(padded_tensors)


def collate_batch_data(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    General collation function for SceneLeapPro dataset batches.

    Handles special cases for 'hand_model_pose' and 'se3' tensors with variable
    first dimensions, while using standard collation for other fields.

    Args:
        batch: List of data dictionaries from dataset

    Returns:
        Dict[str, Any]: Collated batch data
    """
    if not batch:
        return {}

    # Filter out invalid items
    batch = [item for item in batch if isinstance(item, dict)]
    if not batch:
        return {}

    # Infer fallback dtype and device
    fallback_dtype, fallback_device = infer_batch_dtype_device(batch)

    # Get all keys from batch
    all_keys = set()
    for item_dict in batch:
        all_keys.update(item_dict.keys())

    collated_output = {}

    for key in all_keys:
        current_key_items = [item_dict.get(key) for item_dict in batch]

        if key == "hand_model_pose":
            # Handle variable-length hand poses with dynamic pose dimension.
            inferred_suffix_shape = None
            for tensor in current_key_items:
                # Support both torch.Tensor and numpy.ndarray
                if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                    inferred_suffix_shape = tuple(int(x) for x in tensor.shape[1:])
                    break
                try:
                    import numpy as _np
                    if isinstance(tensor, _np.ndarray) and tensor.ndim >= 2:
                        inferred_suffix_shape = tuple(int(x) for x in tensor.shape[1:])
                        break
                except Exception:
                    pass
            if inferred_suffix_shape is None:
                inferred_suffix_shape = CONFIG.BATCH_POSE_SHAPE_SUFFIX

            collated_output[key] = collate_variable_length_tensors(
                current_key_items,
                inferred_suffix_shape,
                fallback_dtype,
                fallback_device,
            )

        elif key == "se3":
            # Handle variable-length SE3 matrices: (N, 4, 4) -> (batch_size, max_N, 4, 4)
            collated_output[key] = collate_variable_length_tensors(
                current_key_items,
                CONFIG.SE3_MATRIX_SHAPE,
                fallback_dtype,
                fallback_device,
            )

        elif key in CONFIG.COLLATE_LIST_KEYS:
            # Keep these as lists
            collated_output[key] = current_key_items

        else:
            # Use default PyTorch collation for other keys
            try:
                collated_output[key] = torch.utils.data.dataloader.default_collate(
                    current_key_items
                )
            except (RuntimeError, TypeError, AttributeError):
                # Fallback to list if default collation fails
                collated_output[key] = current_key_items

    return collated_output


class BatchCollator:
    """
    Handles batch collation for training.

    This class provides static methods for collating batches of data items
    from cached datasets, ensuring identical behavior to the original implementation.
    """

    @staticmethod
    def collate_scene_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch for SceneLeapPro training.

        Handles proper batching of negative_prompts and other fields for training.
        Must preserve exact collation behavior from original implementation.

        Args:
            batch: List of data items from SceneLeapProDatasetCached

        Returns:
            dict: Collated batch with properly formatted tensors and lists
        """
        if not batch:
            return {}

        # Filter out None items or non-dict items from the batch
        batch = [item for item in batch if isinstance(item, dict)]
        if not batch:
            return {}

        # Determine a general dtype and device from the batch for fallback
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

        all_keys = set()
        for item_dict in batch:
            all_keys.update(item_dict.keys())

        collated_output = {}

        for key in all_keys:
            current_key_items = [item_dict.get(key) for item_dict in batch]

            if key == "scene_pc":
                # Stack scene_pc tensors for batch processing
                try:
                    # Try to stack scene_pc tensors (should work if all have same shape)
                    collated_output[key] = torch.stack(current_key_items)
                except RuntimeError:
                    # If stacking fails due to different shapes, use padding
                    collated_output[key] = torch.nn.utils.rnn.pad_sequence(
                        current_key_items, batch_first=True, padding_value=0
                    )

            elif key == "hand_model_pose":
                # Stack hand_model_pose tensors for batch processing
                try:
                    collated_output[key] = torch.stack(current_key_items)
                except RuntimeError:
                    # If stacking fails, use padding
                    collated_output[key] = torch.nn.utils.rnn.pad_sequence(
                        current_key_items, batch_first=True, padding_value=0
                    )

            elif key == "se3":
                # Stack SE3 transformation matrices
                try:
                    collated_output[key] = torch.stack(current_key_items)
                except RuntimeError:
                    # If stacking fails, use padding
                    collated_output[key] = torch.nn.utils.rnn.pad_sequence(
                        current_key_items, batch_first=True, padding_value=0
                    )

            elif key in ["positive_prompt", "negative_prompts", "error"]:
                # Keep text fields as lists - this preserves the correct structure
                # negative_prompts will be: [[sample1_negs], [sample2_negs], ...]
                # which is the correct format for TextEncoder.encode_negative
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

    @staticmethod
    def collate_formatch_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch for ForMatch training.

        'hand_model_pose' and 'se3' are padded to become dense tensors.
        Other tensor fields are handled with standard collation or kept as lists.
        Must preserve exact padding and stacking logic from original implementation.

        Args:
            batch: List of data items from ForMatchSceneLeapProDatasetCached

        Returns:
            dict: Collated batch with properly formatted tensors and lists
        """
        if not batch:
            return {}

        # Filter out None items or non-dict items from the batch
        batch = [item for item in batch if isinstance(item, dict)]
        if not batch:
            return {}

        # Determine a general dtype and device from the batch for fallback padding
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

        all_keys = set()
        for item_dict in batch:
            all_keys.update(item_dict.keys())

        collated_output = {}

        for key in all_keys:
            current_key_items = [item_dict.get(key) for item_dict in batch]

            if key == "hand_model_pose":  # Target: (batch_size, max_N, 23)
                max_n = 0
                for t in current_key_items:
                    if (isinstance(t, torch.Tensor) and t.ndim == 2 
                        and t.shape[1] == 23):
                        max_n = max(max_n, t.shape[0])

                padded_tensors = []
                for t in current_key_items:
                    item_dtype = fallback_dtype
                    item_device = fallback_device
                    if isinstance(
                        t, torch.Tensor
                    ):  # Handles valid tensors and empty tensors e.g. (0,23)
                        item_dtype = t.dtype
                        item_device = t.device

                    if isinstance(t, torch.Tensor) and t.ndim == 2 and t.shape[1] == 23:
                        num_grasps = t.shape[0]
                        if num_grasps == max_n:
                            padded_tensors.append(t)
                        else:  # num_grasps < max_n
                            padding_size = max_n - num_grasps
                            padding = torch.zeros(
                                (padding_size, 23), dtype=item_dtype, device=item_device
                            )
                            padded_tensors.append(torch.cat([t, padding], dim=0))
                    else:  # t is None, or not a tensor, or wrong shape. Pad to max_n.
                        padded_tensors.append(
                            torch.zeros(
                                (max_n, 23), dtype=item_dtype, device=item_device
                            )
                        )

                if padded_tensors:
                    collated_output[key] = torch.stack(padded_tensors)
                elif (
                    batch
                ):  # Only if batch was non-empty but all items for key resulted in no tensors for max_n=0
                    collated_output[key] = torch.empty(
                        (len(batch), 0, 23),
                        dtype=fallback_dtype,
                        device=fallback_device,
                    )
                else:  # batch is empty
                    collated_output[key] = torch.empty(0)

            elif key == "se3":  # Target: (batch_size, max_N, 4, 4)
                max_n = 0
                for t in current_key_items:
                    if (isinstance(t, torch.Tensor) and t.ndim == 3 
                        and t.shape[1:] == (4, 4)):
                        max_n = max(max_n, t.shape[0])

                padded_tensors = []
                for t in current_key_items:
                    item_dtype = fallback_dtype
                    item_device = fallback_device
                    if isinstance(t, torch.Tensor):
                        item_dtype = t.dtype
                        item_device = t.device

                    if (isinstance(t, torch.Tensor) and t.ndim == 3 
                        and t.shape[1:] == (4, 4)):
                        num_grasps = t.shape[0]
                        if num_grasps == max_n:
                            padded_tensors.append(t)
                        else:  # num_grasps < max_n
                            padding_size = max_n - num_grasps
                            padding = torch.zeros(
                                (padding_size, 4, 4),
                                dtype=item_dtype,
                                device=item_device,
                            )
                            padded_tensors.append(torch.cat([t, padding], dim=0))
                    else:  # t is None, or not a tensor, or wrong shape. Pad to max_n.
                        padded_tensors.append(
                            torch.zeros(
                                (max_n, 4, 4), dtype=item_dtype, device=item_device
                            )
                        )

                if padded_tensors:
                    collated_output[key] = torch.stack(padded_tensors)
                elif batch:
                    collated_output[key] = torch.empty(
                        (len(batch), 0, 4, 4),
                        dtype=fallback_dtype,
                        device=fallback_device,
                    )
                else:  # batch is empty
                    collated_output[key] = torch.empty(0)

            elif key in ["obj_verts", "obj_faces", "positive_prompt", 
                        "negative_prompts", "error"]:
                collated_output[key] = current_key_items  # Keep as list

            elif key == "scene_pc":  # Stack scene_pc tensors for batch processing
                try:
                    # Try to stack scene_pc tensors (should work if all have same shape)
                    collated_output[key] = torch.stack(current_key_items)
                except RuntimeError:
                    # If stacking fails due to different shapes, use padding
                    collated_output[key] = torch.nn.utils.rnn.pad_sequence(
                        current_key_items, batch_first=True, padding_value=0
                    )

            elif key in ["obj_code", "scene_id", "category_id_from_object_index", 
                        "depth_view_index"]:
                # ID fields - keep as list for metadata
                collated_output[key] = current_key_items  # Keep as list

            else:  # Default collation for other keys
                try:
                    collated_output[key] = torch.utils.data.dataloader.default_collate(
                        current_key_items
                    )
                except (RuntimeError, TypeError, AttributeError):
                    collated_output[key] = current_key_items  # Fallback to list

        return collated_output
