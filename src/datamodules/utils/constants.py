"""
Constants and Configuration Values for SceneLeapPro Datasets

This module centralizes all magic numbers, default values, and configuration constants
used across SceneLeapPro dataset implementations to improve maintainability and
make configuration easily modifiable.
"""

from typing import Any, Dict, List

import torch

# =============================================================================
# Cache Configuration Constants
# =============================================================================

# Cache health check and monitoring
CACHE_HEALTH_CHECK_INTERVAL = 1000  # Check cache health every N items
CACHE_CREATION_TIMEOUT = 300  # Timeout for cache creation in seconds
CACHE_FILE_CHECK_INTERVAL = 1.0  # Check interval for file creation in seconds
CACHE_VALIDATION_TEST_INDEX = 100  # Index used for cache health testing

# Cache version defaults
DEFAULT_CACHE_VERSION = "v2.0_train_only"  # Default cache version for training
DEFAULT_FORMATCH_CACHE_VERSION = "v1.0_formatch"  # Default cache version for ForMatch

# Cache modes
VALID_CACHE_MODES = ["train", "val", "test"]
DEFAULT_CACHE_MODE = "train"

# =============================================================================
# Dataset Configuration Constants
# =============================================================================

# Default dataset parameters
DEFAULT_MODE = "camera_centric"
DEFAULT_MAX_GRASPS_PER_OBJECT = 200
DEFAULT_MESH_SCALE = 0.1
DEFAULT_NUM_NEG_PROMPTS = 4
DEFAULT_ENABLE_CROPPING = True
DEFAULT_MAX_POINTS = 10000

# Valid coordinate transformation modes
VALID_COORDINATE_MODES = [
    "object_centric",
    "camera_centric",
    "camera_centric_obj_mean_normalized",
    "camera_centric_scene_mean_normalized",
]

# =============================================================================
# Data Dimensions and Shapes
# =============================================================================

# Hand pose and transformation dimensions
POSE_DIM = 23  # Hand pose dimension: P(3) + Q_wxyz(4) + Joints(16)
POSITION_DIM = 3  # Position dimension (x, y, z)
QUATERNION_DIM = 4  # Quaternion dimension (w, x, y, z)
JOINTS_DIM = 16  # Joint angles dimension
SE3_SHAPE = (4, 4)  # SE(3) transformation matrix shape

# Point cloud dimensions
PC_XYZ_DIM = 3  # Point cloud XYZ dimension
PC_RGB_DIM = 3  # Point cloud RGB dimension
PC_XYZRGB_DIM = 6  # Point cloud XYZ+RGB dimension (for PointNet2)
PC_XYZMASK_DIM = 4  # Point cloud XYZ+mask dimension (legacy)

# =============================================================================
# Data Types and Device Configuration
# =============================================================================

# Default tensor types
DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = "cpu"
DEFAULT_LONG_DTYPE = torch.long
DEFAULT_BOOL_DTYPE = torch.bool

# =============================================================================
# Error Handling and Default Values
# =============================================================================

# Padding and fill values
PADDING_VALUE = 0.0
IDENTITY_QUATERNION = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
QUATERNION_NORM_THRESHOLD = 1e-6

# Default error response values
DEFAULT_ERROR_OBJECT_NAME = "unknown_object"
DEFAULT_ERROR_PROMPT = "unknown_object"
DEFAULT_EMPTY_PROMPT = ""

# Error tensor shapes (for creating empty tensors)
ERROR_SCENE_PC_SHAPE = (0, PC_XYZRGB_DIM)
ERROR_POSE_SHAPE = (0, POSE_DIM)
ERROR_SE3_SHAPE = (0, SE3_SHAPE[0], SE3_SHAPE[1])
ERROR_MESH_VERTS_SHAPE = (0, 3)
ERROR_MESH_FACES_SHAPE = (0, 3)

# =============================================================================
# Cache Keys and Field Names
# =============================================================================

# Standard cache keys for training (SceneLeapPro)
STANDARD_CACHE_KEYS = [
    "scene_pc",
    "hand_model_pose",
    "se3",
    "positive_prompt",
    "negative_prompts",
]

# Extended cache keys for ForMatch validation (7 fields)
FORMATCH_VAL_CACHE_KEYS = STANDARD_CACHE_KEYS + ["obj_verts", "obj_faces"]

# Extended cache keys for ForMatch testing (11 fields with IDs)
FORMATCH_TEST_CACHE_KEYS = FORMATCH_VAL_CACHE_KEYS + [
    "scene_id",
    "object_id",
    "grasp_id",
    "object_code",
]

# All possible cache keys (for validation)
ALL_CACHE_KEYS = [
    "scene_pc",
    "hand_model_pose",
    "se3",
    "positive_prompt",
    "negative_prompts",
    "object_mask",
    "obj_verts",
    "obj_faces",
    "scene_id",
    "object_id",
    "grasp_id",
    "object_code",
    "error",
]

# =============================================================================
# Validation and Shape Constraints
# =============================================================================

# Minimum required shapes for validation
MIN_POSE_SHAPE = (POSE_DIM,)  # Minimum pose shape for single grasp
BATCH_POSE_SHAPE_SUFFIX = (POSE_DIM,)  # Last dimension for batch poses
SE3_MATRIX_SHAPE = SE3_SHAPE  # SE3 matrix shape validation

# Point cloud constraints
MIN_POINTS_PER_CLOUD = 1  # Minimum points required in point cloud
MAX_POINTS_UPPER_LIMIT = 100000  # Upper limit for max_points parameter

# Prompt constraints
MAX_PROMPT_LENGTH = 1000  # Maximum length for prompts
MIN_NEG_PROMPTS = 0  # Minimum number of negative prompts
MAX_NEG_PROMPTS = 20  # Maximum number of negative prompts

# =============================================================================
# File and Directory Configuration
# =============================================================================

# Cache file naming patterns
CACHE_FILENAME_PATTERN = "{prefix}_{hash}_{mode}_{max_grasps}.h5"
FORMATCH_CACHE_FILENAME_PATTERN = (
    "sceneleappro_{cache_mode}_{hash}_{mode}_{max_grasps}.h5"
)
STANDARD_CACHE_FILENAME_PATTERN = "sceneleappro_{hash}_{mode}_{max_grasps}.h5"

# HDF5 configuration
HDF5_COMPRESSION = "gzip"  # Default compression for HDF5 datasets
HDF5_ITEM_GROUP_PREFIX = "item_"  # Prefix for item groups in HDF5

# =============================================================================
# Logging and Monitoring Configuration
# =============================================================================

# Logging levels and intervals
CACHE_STATUS_LOG_INTERVAL = 100  # Log cache status every N items
PERFORMANCE_LOG_THRESHOLD = (
    5.0  # Log performance warnings if operations take > N seconds
)
MEMORY_WARNING_THRESHOLD_MB = 1000  # Warn if memory usage exceeds N MB

# Progress reporting
PROGRESS_BAR_UPDATE_INTERVAL = 10  # Update progress bar every N items
TQDM_DESCRIPTION_MAX_LENGTH = 50  # Maximum length for tqdm descriptions

# =============================================================================
# Utility Functions for Constants
# =============================================================================


def get_cache_keys_for_mode(cache_mode: str) -> List[str]:
    """
    Get appropriate cache keys based on cache mode.

    Args:
        cache_mode: Cache mode ("train", "val", or "test")

    Returns:
        List of cache keys for the specified mode

    Raises:
        ValueError: If cache_mode is invalid
    """
    if cache_mode == "train":
        return STANDARD_CACHE_KEYS.copy()
    elif cache_mode == "val":
        return FORMATCH_VAL_CACHE_KEYS.copy()
    elif cache_mode == "test":
        return FORMATCH_TEST_CACHE_KEYS.copy()
    else:
        raise ValueError(
            f"Invalid cache_mode '{cache_mode}'. Valid modes: {VALID_CACHE_MODES}"
        )


def get_default_error_values(num_neg_prompts: int) -> Dict[str, Any]:
    """
    Get default error values for dataset error responses.

    Args:
        num_neg_prompts: Number of negative prompts to generate

    Returns:
        Dictionary of default error values
    """
    return {
        "scene_pc": torch.zeros(ERROR_SCENE_PC_SHAPE, dtype=DEFAULT_DTYPE),
        "hand_model_pose": torch.zeros(ERROR_POSE_SHAPE, dtype=DEFAULT_DTYPE),
        "se3": torch.zeros(ERROR_SE3_SHAPE, dtype=DEFAULT_DTYPE),
        "positive_prompt": DEFAULT_ERROR_PROMPT,
        "negative_prompts": [DEFAULT_EMPTY_PROMPT] * num_neg_prompts,
        "obj_verts": torch.zeros(ERROR_MESH_VERTS_SHAPE, dtype=DEFAULT_DTYPE),
        "obj_faces": torch.zeros(ERROR_MESH_FACES_SHAPE, dtype=DEFAULT_LONG_DTYPE),
    }


def validate_constants() -> bool:
    """
    Validate that all constants are consistent and valid.

    Returns:
        True if all constants are valid

    Raises:
        ValueError: If any constants are invalid
    """
    # Validate dimensions
    if POSE_DIM != POSITION_DIM + QUATERNION_DIM + JOINTS_DIM:
        raise ValueError(
            f"POSE_DIM ({POSE_DIM}) must equal POSITION_DIM + QUATERNION_DIM + JOINTS_DIM"
        )

    if PC_XYZRGB_DIM != PC_XYZ_DIM + PC_RGB_DIM:
        raise ValueError(
            f"PC_XYZRGB_DIM ({PC_XYZRGB_DIM}) must equal PC_XYZ_DIM + PC_RGB_DIM"
        )

    # Validate default values
    if DEFAULT_MAX_POINTS <= 0:
        raise ValueError(f"DEFAULT_MAX_POINTS must be positive")

    if DEFAULT_NUM_NEG_PROMPTS < 0:
        raise ValueError(f"DEFAULT_NUM_NEG_PROMPTS must be non-negative")

    if DEFAULT_MESH_SCALE <= 0:
        raise ValueError(f"DEFAULT_MESH_SCALE must be positive")

    # Validate cache modes
    if DEFAULT_CACHE_MODE not in VALID_CACHE_MODES:
        raise ValueError(f"DEFAULT_CACHE_MODE must be in VALID_CACHE_MODES")

    # Validate coordinate modes
    if DEFAULT_MODE not in VALID_COORDINATE_MODES:
        raise ValueError(f"DEFAULT_MODE must be in VALID_COORDINATE_MODES")

    return True


# Validate constants on import
validate_constants()
