"""
Dataset configuration module for SceneLeapPro datasets.

This module contains configuration classes for SceneLeapPro dataset implementations.
Constants and default values are now centralized in constants.py.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .constants import (  # Dimensions; Default values; Data types; Validation; Error handling
    ALL_CACHE_KEYS, BATCH_POSE_SHAPE_SUFFIX, DEFAULT_BOOL_DTYPE,
    DEFAULT_CACHE_MODE, DEFAULT_CACHE_VERSION, DEFAULT_DEVICE, DEFAULT_DTYPE,
    DEFAULT_ENABLE_CROPPING, DEFAULT_LONG_DTYPE, DEFAULT_MAX_GRASPS_PER_OBJECT,
    DEFAULT_MAX_POINTS, DEFAULT_MESH_SCALE, DEFAULT_MODE,
    DEFAULT_NUM_NEG_PROMPTS, IDENTITY_QUATERNION, JOINTS_DIM, MIN_POSE_SHAPE,
    PADDING_VALUE, PC_RGB_DIM, PC_XYZ_DIM, PC_XYZRGB_DIM, POSE_DIM,
    POSITION_DIM, QUATERNION_DIM, QUATERNION_NORM_THRESHOLD, SE3_MATRIX_SHAPE,
    SE3_SHAPE, VALID_CACHE_MODES, VALID_COORDINATE_MODES)


@dataclass
class DatasetConfig:
    """
    Configuration class containing all constants and default values for SceneLeapPro datasets.

    This centralizes all hardcoded values and makes them easily configurable.
    """

    # Use constants from constants.py - dimensions
    POSE_DIM: int = POSE_DIM
    SE3_SHAPE: tuple = SE3_SHAPE
    POSITION_DIM: int = POSITION_DIM
    QUATERNION_DIM: int = QUATERNION_DIM
    JOINTS_DIM: int = JOINTS_DIM

    # Point cloud dimensions
    PC_XYZ_DIM: int = PC_XYZ_DIM
    PC_RGB_DIM: int = PC_RGB_DIM
    PC_XYZRGB_DIM: int = PC_XYZRGB_DIM

    # Default values for dataset initialization
    DEFAULT_MODE: str = DEFAULT_MODE
    DEFAULT_MAX_GRASPS_PER_OBJECT: Optional[int] = DEFAULT_MAX_GRASPS_PER_OBJECT
    DEFAULT_MESH_SCALE: float = DEFAULT_MESH_SCALE
    DEFAULT_NUM_NEG_PROMPTS: int = DEFAULT_NUM_NEG_PROMPTS
    DEFAULT_ENABLE_CROPPING: bool = DEFAULT_ENABLE_CROPPING
    DEFAULT_MAX_POINTS: int = DEFAULT_MAX_POINTS

    # Data types and devices
    DEFAULT_DTYPE: torch.dtype = DEFAULT_DTYPE
    DEFAULT_DEVICE: str = DEFAULT_DEVICE
    DEFAULT_LONG_DTYPE: torch.dtype = DEFAULT_LONG_DTYPE
    DEFAULT_BOOL_DTYPE: torch.dtype = DEFAULT_BOOL_DTYPE

    # Coordinate transformation modes
    VALID_MODES: List[str] = field(
        default_factory=lambda: VALID_COORDINATE_MODES.copy()
    )

    # Tensor shape validation
    MIN_POSE_SHAPE: tuple = MIN_POSE_SHAPE
    BATCH_POSE_SHAPE_SUFFIX: tuple = BATCH_POSE_SHAPE_SUFFIX
    SE3_MATRIX_SHAPE: tuple = SE3_MATRIX_SHAPE

    # Padding and fill values
    PADDING_VALUE: float = PADDING_VALUE
    IDENTITY_QUATERNION: List[float] = field(
        default_factory=lambda: IDENTITY_QUATERNION.copy()
    )
    QUATERNION_NORM_THRESHOLD: float = QUATERNION_NORM_THRESHOLD

    # Error handling
    ERROR_RETURN_KEYS: List[str] = field(default_factory=lambda: ALL_CACHE_KEYS.copy())

    # Collation configuration
    COLLATE_SPECIAL_KEYS: List[str] = field(
        default_factory=lambda: ["hand_model_pose", "se3"]
    )
    COLLATE_LIST_KEYS: List[str] = field(
        default_factory=lambda: [
            "object_mask",
            "obj_verts",
            "obj_faces",
            "positive_prompt",
            "negative_prompts",
            "error",
        ]
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> bool:
        """
        Validate the configuration settings.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate dimensions
        if self.POSE_DIM != self.POSITION_DIM + self.QUATERNION_DIM + self.JOINTS_DIM:
            raise ValueError(
                f"POSE_DIM ({self.POSE_DIM}) must equal "
                f"POSITION_DIM + QUATERNION_DIM + JOINTS_DIM "
                f"({self.POSITION_DIM + self.QUATERNION_DIM + self.JOINTS_DIM})"
            )

        if self.PC_XYZRGB_DIM != self.PC_XYZ_DIM + self.PC_RGB_DIM:
            raise ValueError(
                f"PC_XYZRGB_DIM ({self.PC_XYZRGB_DIM}) must equal "
                f"PC_XYZ_DIM + PC_RGB_DIM ({self.PC_XYZ_DIM + self.PC_RGB_DIM})"
            )

        # Validate mode
        if self.DEFAULT_MODE not in self.VALID_MODES:
            raise ValueError(
                f"DEFAULT_MODE '{self.DEFAULT_MODE}' not in VALID_MODES {self.VALID_MODES}"
            )

        # Validate positive values
        if self.DEFAULT_MAX_POINTS <= 0:
            raise ValueError(
                f"DEFAULT_MAX_POINTS must be positive, got {self.DEFAULT_MAX_POINTS}"
            )

        if self.DEFAULT_NUM_NEG_PROMPTS < 0:
            raise ValueError(
                f"DEFAULT_NUM_NEG_PROMPTS must be non-negative, got {self.DEFAULT_NUM_NEG_PROMPTS}"
            )

        if self.DEFAULT_MESH_SCALE <= 0:
            raise ValueError(
                f"DEFAULT_MESH_SCALE must be positive, got {self.DEFAULT_MESH_SCALE}"
            )

        # Validate quaternion
        if len(self.IDENTITY_QUATERNION) != self.QUATERNION_DIM:
            raise ValueError(
                f"IDENTITY_QUATERNION length ({len(self.IDENTITY_QUATERNION)}) "
                f"must equal QUATERNION_DIM ({self.QUATERNION_DIM})"
            )

        return True

    def get_error_return_template(
        self,
        obj_code: str = "unknown",
        scene_id: str = "unknown",
        category_id: int = -1,
        depth_view_index: int = -1,
        error_msg: str = "Unknown error",
    ) -> Dict[str, Any]:
        """
        Get a template dictionary for error returns.

        Args:
            obj_code: Object code
            scene_id: Scene identifier
            category_id: Category ID
            depth_view_index: Depth view index
            error_msg: Error message

        Returns:
            Dict[str, Any]: Error return template
        """
        return {
            "obj_code": obj_code,
            "scene_pc": torch.zeros((0, self.PC_XYZRGB_DIM), dtype=self.DEFAULT_DTYPE),
            "object_mask": torch.zeros((0,), dtype=self.DEFAULT_BOOL_DTYPE),
            "hand_model_pose": torch.zeros(
                (0, self.POSE_DIM), dtype=self.DEFAULT_DTYPE
            ),
            "se3": torch.zeros((0, *self.SE3_SHAPE), dtype=self.DEFAULT_DTYPE),
            "scene_id": scene_id,
            "category_id_from_object_index": category_id,
            "depth_view_index": depth_view_index,
            "obj_verts": torch.zeros((0, self.PC_XYZ_DIM), dtype=self.DEFAULT_DTYPE),
            "obj_faces": torch.zeros(
                (0, self.PC_XYZ_DIM), dtype=self.DEFAULT_LONG_DTYPE
            ),
            "positive_prompt": "",
            "negative_prompts": [],
            "error": error_msg,
        }

    def get_identity_quaternion_tensor(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Get identity quaternion as tensor.

        Args:
            device: Target device
            dtype: Target dtype

        Returns:
            torch.Tensor: Identity quaternion tensor
        """
        if device is None:
            device = self.DEFAULT_DEVICE
        if dtype is None:
            dtype = self.DEFAULT_DTYPE

        return torch.tensor(self.IDENTITY_QUATERNION, device=device, dtype=dtype)


# Global configuration instance
CONFIG = DatasetConfig()


def validate_dataset_configuration(
    root_dir: str, succ_grasp_dir: str, obj_root_dir: str, mode: str
) -> bool:
    """
    Validate dataset configuration parameters.

    Args:
        root_dir: Root directory path
        succ_grasp_dir: Successful grasp directory path
        obj_root_dir: Object root directory path
        mode: Coordinate transformation mode

    Returns:
        bool: True if configuration is valid
    """
    import os

    # Check if directories exist
    if not os.path.exists(root_dir):
        return False
    if not os.path.exists(succ_grasp_dir):
        return False
    if not os.path.exists(obj_root_dir):
        return False

    # Check if mode is valid
    if mode not in CONFIG.VALID_MODES:
        return False

    return True


@dataclass
class CachedDatasetConfig:
    """
    Configuration for cached datasets with validation and serialization.

    This class manages all configuration parameters for cached SceneLeapPro datasets,
    including cache file naming, hash generation, and parameter validation.
    """

    root_dir: str
    succ_grasp_dir: str
    obj_root_dir: str
    mode: str = DEFAULT_MODE
    max_grasps_per_object: Optional[int] = DEFAULT_MAX_GRASPS_PER_OBJECT
    mesh_scale: float = DEFAULT_MESH_SCALE
    num_neg_prompts: int = DEFAULT_NUM_NEG_PROMPTS
    enable_cropping: bool = DEFAULT_ENABLE_CROPPING
    max_points: int = DEFAULT_MAX_POINTS
    cache_version: str = DEFAULT_CACHE_VERSION
    cache_mode: str = DEFAULT_CACHE_MODE  # For ForMatch: "val" or "test"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> bool:
        """
        Validate the configuration settings.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate directories exist
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        if not os.path.exists(self.succ_grasp_dir):
            raise ValueError(
                f"Success grasp directory does not exist: {self.succ_grasp_dir}"
            )
        if not os.path.exists(self.obj_root_dir):
            raise ValueError(
                f"Object root directory does not exist: {self.obj_root_dir}"
            )

        # Validate mode
        if self.mode not in CONFIG.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Valid modes: {CONFIG.VALID_MODES}"
            )

        # Validate positive values
        if self.max_points <= 0:
            raise ValueError(f"max_points must be positive, got {self.max_points}")

        if self.num_neg_prompts < 0:
            raise ValueError(
                f"num_neg_prompts must be non-negative, got {self.num_neg_prompts}"
            )

        if self.mesh_scale <= 0:
            raise ValueError(f"mesh_scale must be positive, got {self.mesh_scale}")

        # Validate cache_mode
        valid_cache_modes = ["train", "val", "test"]
        if self.cache_mode not in valid_cache_modes:
            raise ValueError(
                f"Invalid cache_mode '{self.cache_mode}'. Valid modes: {valid_cache_modes}"
            )

        return True

    def generate_cache_hash(
        self,
        num_grasps: Optional[int] = None,
        grasp_sampling_strategy: Optional[str] = None,
        use_exhaustive_sampling: bool = False,
        exhaustive_sampling_strategy: Optional[str] = None,
    ) -> str:
        """
        Generate MD5 hash for cache filename.

        Args:
            num_grasps: Number of grasps per sample (for SceneLeapPlus)
            grasp_sampling_strategy: Grasp sampling strategy (for SceneLeapPlus)
            use_exhaustive_sampling: Whether to use exhaustive sampling
            exhaustive_sampling_strategy: Exhaustive sampling strategy

        Returns:
            str: MD5 hash string
        """
        if self.cache_mode == "train":
            # Use the original SceneLeapProDatasetCached hash generation logic
            params_string = (
                f"root_dir={os.path.abspath(self.root_dir)},"
                f"mode={self.mode},"
                f"max_points={self.max_points},"
                f"enable_cropping={self.enable_cropping},"
                f"coordinate_system_mode={self.mode},"
                f"num_neg_prompts={self.num_neg_prompts},"
                f"max_grasps_per_scene={self.max_grasps_per_object},"
                f"cache_version={self.cache_version}"
            )
        else:
            # Use the ForMatch hash generation logic
            params_string = (
                f"{self.root_dir}_{self.succ_grasp_dir}_{self.obj_root_dir}_{self.mode}_"
                f"{self.max_grasps_per_object}_{self.mesh_scale}_{self.num_neg_prompts}_"
                f"{self.enable_cropping}_{self.max_points}_{self.cache_version}_{self.cache_mode}"
            )

        # Add SceneLeapPlus specific parameters if provided
        if num_grasps is not None:
            params_string += f",num_grasps={num_grasps}"
        if grasp_sampling_strategy is not None:
            params_string += f",grasp_sampling_strategy={grasp_sampling_strategy}"
        if use_exhaustive_sampling:
            params_string += f",use_exhaustive_sampling={use_exhaustive_sampling}"
        if exhaustive_sampling_strategy is not None:
            params_string += (
                f",exhaustive_sampling_strategy={exhaustive_sampling_strategy}"
            )

        return hashlib.md5(params_string.encode("utf-8")).hexdigest()

    def generate_cache_filename(
        self,
        dataset_type: str = "sceneleappro",
        num_grasps: Optional[int] = None,
        grasp_sampling_strategy: Optional[str] = None,
        use_exhaustive_sampling: bool = False,
        exhaustive_sampling_strategy: Optional[str] = None,
    ) -> str:
        """
        Generate cache filename maintaining existing conventions.

        Args:
            dataset_type: Type of dataset ("sceneleappro" for normal, "formatch" for ForMatch)
            num_grasps: Number of grasps per sample (for SceneLeapPlus)
            grasp_sampling_strategy: Grasp sampling strategy (for SceneLeapPlus)
            use_exhaustive_sampling: Whether to use exhaustive sampling
            exhaustive_sampling_strategy: Exhaustive sampling strategy

        Returns:
            str: Cache filename
        """
        cache_hash = self.generate_cache_hash(
            num_grasps=num_grasps,
            grasp_sampling_strategy=grasp_sampling_strategy,
            use_exhaustive_sampling=use_exhaustive_sampling,
            exhaustive_sampling_strategy=exhaustive_sampling_strategy,
        )

        if self.cache_mode == "train":
            # Original SceneLeapProDatasetCached naming pattern
            return (
                f"sceneleappro_{cache_hash}_{self.mode}_{self.max_grasps_per_object}.h5"
            )
        else:
            # ForMatch naming pattern
            return f"sceneleappro_{self.cache_mode}_{cache_hash}_{self.mode}_{self.max_grasps_per_object}.h5"
