import json
import logging
import math
import os
from typing import Optional
import numpy as np
import pytorch3d.transforms as transforms
import torch
import torch.nn.functional as F
from pytorchse3.se3 import se3_log_map, se3_exp_map

from lightning.pytorch.utilities.rank_zero import rank_zero_info

from . import hand_pose_config
from .rotation_spec import get_rotation_spec
from .anchor_config import get_palm_center_margin

JSON_STATS_FILE_PATH = (
    "assets/formatch_overall_hand_pose_dimension_statistics_by_mode.json"
)
POSE_STATS = None

DEX_NORMALIZATION_MODES = {"dexgraspanything_base", "dexgraspanything_palm_center", "dexgraspnet", "bodexshadow"}
DEX_STATS_FILE_PATH = "assets/grasp_anyting_normalization_stats.pt"
DEX_STATS_JSON_FILE_PATH = "assets/grasp_anyting_normalization_stats.json"
# DEXNET_STATS_BASE_PATH = "assets/mydexgraspnet_normalization_stats_full"
unified_normalization_stats_path = "assets/unified_normalization_stats"
DEX_STATS_JSON_FILE_PATH = unified_normalization_stats_path
DEXNET_STATS_BASE_PATH = "assets/unified_normalization_stats"

BODEXSHADOW_STATS_BASE_PATH = "assets/unified_normalization_stats"

PALM_CENTER_MARGIN = 0.25
# 基于 mode 的缓存：每个模式各自缓存一次
# BODEXSHADOW_STATS_BASE_PATH = "assets/mybodexshadow_normalization_stats"

DEX_STATS_CACHE_MAP = {}

# 当前 Dex 反归一化所用的 anchor（由数据集在 process_hand_pose_test/process_hand_pose 中写入）
_CURRENT_DEX_ANCHOR_FOR_DENORM = None
# _CURRENT_DEX_ANCHOR_FOR_DENORM = palm_center

def set_current_dex_anchor_for_denorm(anchor: str):
    global _CURRENT_DEX_ANCHOR_FOR_DENORM
    try:
        a = (anchor or "").strip().lower()
        if a in ("base", "palm_center", "mjcf"):
            _CURRENT_DEX_ANCHOR_FOR_DENORM = a
    except Exception:
        pass

def get_current_dex_anchor_for_denorm() -> str:
    return _CURRENT_DEX_ANCHOR_FOR_DENORM

DEBUG_SHAPES = False
_DEBUG_BYPASS_NORMALIZED = {
    "dexgraspanything",
    "dexgraspanything_virtual",
    "dexgraspanythingvirtual",
}


def _normalize_mode(mode: str) -> str:
    return (mode or "").strip().lower()


def _is_debug_bypass_mode(mode: str) -> bool:
    return _normalize_mode(mode) in _DEBUG_BYPASS_NORMALIZED


def _is_dex_mode(mode: str) -> bool:
    return _normalize_mode(mode) in DEX_NORMALIZATION_MODES

# -----------------------------
# Public helpers (exposed)
# -----------------------------
def normalize_mode(mode: str) -> str:
    return _normalize_mode(mode)


def is_debug_bypass_mode(mode: str) -> bool:
    return _is_debug_bypass_mode(mode)


def is_dex_mode(mode: str) -> bool:
    return _is_dex_mode(mode)


# 在特定模式下忽略的关节角维度（相对于 qpos 片段的局部索引，从 0 开始）
# DexGraspNet: 数据集中有 2 个自由度恒为 0，训练与归一化中应忽略，
# 仅在送入手模型（FK/几何）前再补回（通常补 0）。
# 如与实际维度位置不一致，请告知我具体索引，我会更新此映射。

def get_disabled_qpos_indices(mode: str):
    m = _normalize_mode(mode)
    # 仅当全局关节维为24时，对 Dex/Bodex 的前两维做禁用；22维时不禁用任何维度
    if hand_pose_config.JOINT_ANGLE_DIM == 24 and m in ("dexgraspnet", "bodexshadow"):
        return (0, 1)
    return ()


def _load_dex_stats_if_needed(mode: str = None):
    """
    加载 Dex 模式（DexGraspAnything / DexGraspNet）的归一化统计，并基于 mode 做缓存。
    返回的字典包含可选键："hand_trans"、"palm_center"、"joint_angles"。
    """
    global DEX_STATS_CACHE_MAP

    mode_n = _normalize_mode(mode)
    if mode_n in DEX_STATS_CACHE_MAP:
        return DEX_STATS_CACHE_MAP[mode_n]

    # 依 mode 选择候选统计文件（按优先级尝试）
    candidates = []
    if mode_n in ("dexgraspanything_base", "dexgraspanything_palm_center"):
        candidates = [("json", DEX_STATS_JSON_FILE_PATH), ("pt", DEX_STATS_FILE_PATH)]
    elif mode_n == "dexgraspnet":
        base = DEXNET_STATS_BASE_PATH
        candidates = [
            ("json", base),
            ("json", base + ".json"),
            ("pt", base),
            ("pt", base + ".pt"),
        ]
    elif mode_n == "bodexshadow":
        base = BODEXSHADOW_STATS_BASE_PATH
        candidates = [
            ("json", base),
            ("json", base + ".json"),
            ("pt", base),
            ("pt", base + ".pt"),
        ]
    else:
        # 不识别的 mode，退化到 DexGraspAnything 的默认统计
        candidates = [("json", DEX_STATS_JSON_FILE_PATH), ("pt", DEX_STATS_FILE_PATH)]

    try:
        logging.getLogger(__name__).info(
            "[hand_helper] Dex 统计候选(mode=%s): %s",
            mode_n,
            [(k, p, (bool(p) and os.path.exists(p))) for k, p in candidates],
        )
    except Exception:
        pass

    raw = None
    for kind, path in candidates:
        try:
            if not path:
                continue
            if os.path.exists(path):
                if kind == "json":
                    with open(path, "r") as f:
                        raw = json.load(f)
                else:
                    raw = torch.load(path, map_location="cpu")
                if raw is not None:
                    logging.getLogger(__name__).info(
                        "[hand_helper] 已加载 Dex 归一化统计: path=%s kind=%s",
                        path,
                        kind,
                    )
                    break
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[hand_helper] 读取 Dex 归一化统计失败: %s (%s)", path, exc
            )
            raw = None
            continue

    if raw is None:
        logging.getLogger(__name__).warning(
            "[hand_helper] 未找到 Dex 归一化统计文件（mode=%s）", mode_n
        )
        DEX_STATS_CACHE_MAP[mode_n] = {}
        return DEX_STATS_CACHE_MAP[mode_n]

    stats = {}

    def _extract_bounds(entry: dict, prefer_margin: bool = True):
        if not isinstance(entry, dict):
            return None
        min_key = "min_with_margin" if prefer_margin and "min_with_margin" in entry else "min"
        max_key = "max_with_margin" if prefer_margin and "max_with_margin" in entry else "max"
        if min_key not in entry or max_key not in entry:
            return None
        min_tensor = torch.as_tensor(entry[min_key], dtype=torch.float32)
        max_tensor = torch.as_tensor(entry[max_key], dtype=torch.float32)
        # 不再添加额外的 margin，直接使用统计文件中的边界
        return {"min": min_tensor, "max": max_tensor}

    if isinstance(raw, dict):
        # 优先读取嵌套字典形式
        hand_trans_bounds = _extract_bounds(raw.get("hand_trans"))
        palm_center_bounds = _extract_bounds(raw.get("palm_center"))
        joint_entry = raw.get("joint_angles")

        # 兼容顶层扁平键（由统计脚本导出）：hand_trans_min/_max/_min_with_margin/_max_with_margin
        def _from_flat(prefix: str):
            min_key = f"{prefix}_min_with_margin" if f"{prefix}_min_with_margin" in raw else f"{prefix}_min"
            max_key = f"{prefix}_max_with_margin" if f"{prefix}_max_with_margin" in raw else f"{prefix}_max"
            if min_key in raw and max_key in raw:
                min_tensor = torch.as_tensor(raw[min_key], dtype=torch.float32)
                max_tensor = torch.as_tensor(raw[max_key], dtype=torch.float32)
                # 不再添加额外的 margin，直接使用统计文件中的边界
                return {"min": min_tensor, "max": max_tensor}
            return None

        if hand_trans_bounds is None:
            hand_trans_bounds = _from_flat("hand_trans")
        if palm_center_bounds is None:
            palm_center_bounds = _from_flat("palm_center")

        # joints - 使用 _extract_bounds 优先选择 min_with_margin/max_with_margin，与数据集保持一致
        joint_bounds = _extract_bounds(joint_entry)
        if joint_bounds is None:
            # 兼容扁平键 joint_angles_min/max
            min_t = torch.as_tensor(raw.get("joint_angles_min", []), dtype=torch.float32)
            max_t = torch.as_tensor(raw.get("joint_angles_max", []), dtype=torch.float32)
            if min_t.numel() == 0 or max_t.numel() == 0:
                joint_bounds = None
            else:
                joint_bounds = {"min": min_t, "max": max_t}

        if hand_trans_bounds is not None:
            stats["hand_trans"] = hand_trans_bounds
        if palm_center_bounds is not None:
            stats["palm_center"] = palm_center_bounds
        if joint_bounds is not None:
            stats["joint_angles"] = joint_bounds

    # 不在加载阶段派生 palm_center；统一在 norm/denorm 使用处从 hand_trans 即时派生，避免在 stats 中维护多份边界

    DEX_STATS_CACHE_MAP[mode_n] = stats
    return DEX_STATS_CACHE_MAP[mode_n]


def _denorm_from_bounds_torch(values: torch.Tensor, bounds: dict) -> torch.Tensor:
    if bounds is None:
        return values
    min_val = bounds["min"].to(values.device, values.dtype)
    max_val = bounds["max"].to(values.device, values.dtype)
    range_val = torch.clamp(max_val - min_val, min=1e-8)
    half = values.new_tensor(0.5)
    one = values.new_tensor(1.0)
    denorm = (values + one) * half * range_val + min_val
    return torch.clamp(denorm, min_val, max_val)


def _denorm_from_bounds_numpy(values: np.ndarray, bounds: dict) -> np.ndarray:
    if bounds is None:
        return values
    min_val = bounds["min"].cpu().numpy()
    max_val = bounds["max"].cpu().numpy()
    range_val = np.clip(max_val - min_val, 1e-8, None)
    denorm = (values + 1.0) * 0.5 * range_val + min_val
    return np.clip(denorm, min_val, max_val)


def _prepare_joint_bounds_for_denorm(joint_bounds: Optional[dict], target_dim: int, mode_normalized: str) -> Optional[dict]:
    """
    Ensure joint bounds align with the expected pose dimension.

    DexGraspNet / BodexShadow datasets pad the first two wrist DOFs with zeros.
    If stats contain 24 dims, we zero out the first two so they remain fixed.
    If stats contain only 22 dims, we embed them into a 24-dim vector with leading zeros.
    """
    if joint_bounds is None:
        return None

    min_val = joint_bounds.get("min")
    max_val = joint_bounds.get("max")
    if min_val is None or max_val is None:
        return joint_bounds

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return torch.as_tensor(x, dtype=torch.float32)

    min_tensor = _to_tensor(min_val)
    max_tensor = _to_tensor(max_val)

    dim = min_tensor.shape[-1]
    # 情况A：统计与目标维一致
    if dim == target_dim:
        # 仅当目标是24维且为 Dex/Bodex 模式时，将腕部两维固定为0（保持训练假设）
        if target_dim == 24 and mode_normalized in ("dexgraspnet", "bodexshadow"):
            min_tensor = min_tensor.clone()
            max_tensor = max_tensor.clone()
            min_tensor[:2] = 0.0
            max_tensor[:2] = 0.0
        return {"min": min_tensor, "max": max_tensor}

    # 情况B：统计少2维（22）而目标24维 -> 前置补两维0
    if dim == target_dim - 2:
        pad_lo = torch.zeros(target_dim, dtype=min_tensor.dtype)
        pad_hi = torch.zeros(target_dim, dtype=max_tensor.dtype)
        pad_lo[2:] = min_tensor
        pad_hi[2:] = max_tensor
        return {"min": pad_lo, "max": pad_hi}

    # 情况C：统计多2维（24）而目标22维 -> 去掉前两维（腕部），对齐到22维
    if dim == target_dim + 2:
        return {"min": min_tensor[2:].clone(), "max": max_tensor[2:].clone()}

    logging.getLogger(__name__).warning(
        "[hand_helper] Unexpected joint bounds dimension %d (expected %d); skipping alignment.",
        dim,
        target_dim,
    )
    return {"min": min_tensor, "max": max_tensor}


def normalize_with_bounds(values: torch.Tensor, bounds: dict) -> torch.Tensor:
    """
    使用给定的 min/max bounds 将张量线性缩放到 [-1, 1]。
    """
    if bounds is None:
        return values
    min_val = bounds["min"].to(values.device, values.dtype)
    max_val = bounds["max"].to(values.device, values.dtype)
    range_val = torch.clamp(max_val - min_val, min=1e-8)
    return 2.0 * (values - min_val) / range_val - 1.0


def denormalize_with_bounds(values: torch.Tensor, bounds: dict) -> torch.Tensor:
    """
    从 [-1, 1] 反归一化回给定的 bounds。
    """
    return _denorm_from_bounds_torch(values, bounds)


def _ensure_qpos_field(data: dict) -> None:
    if not isinstance(data, dict):
        return
    if "qpos" in data:
        return
    hand_pose = data.get("hand_model_pose")
    if hand_pose is None:
        return

    if isinstance(hand_pose, torch.Tensor):
        data["qpos"] = hand_pose[..., hand_pose_config.QPOS_SLICE]
    elif isinstance(hand_pose, np.ndarray):
        data["qpos"] = hand_pose[..., hand_pose_config.QPOS_SLICE]
    elif isinstance(hand_pose, list):
        qpos_list = []
        for pose in hand_pose:
            if isinstance(pose, torch.Tensor):
                qpos_list.append(pose[..., hand_pose_config.QPOS_SLICE])
            elif isinstance(pose, np.ndarray):
                qpos_list.append(pose[..., hand_pose_config.QPOS_SLICE])
            else:
                return
        if qpos_list:
            data["qpos"] = qpos_list


def _make_batched_eye(batch_shape, device, dtype):
    if not batch_shape:
        return torch.eye(4, device=device, dtype=dtype)
    if any(dim == 0 for dim in batch_shape):
        return torch.zeros(*batch_shape, 4, 4, device=device, dtype=dtype)
    view_shape = (1,) * len(batch_shape) + (4, 4)
    eye = torch.eye(4, device=device, dtype=dtype)
    return eye.view(view_shape).expand(*batch_shape, 4, 4).clone()


def _build_se3_from_pose_tensor(pose_tensor: torch.Tensor, rot_type: str) -> torch.Tensor:
    if not isinstance(pose_tensor, torch.Tensor):
        raise TypeError(f"pose_tensor must be a torch.Tensor, got {type(pose_tensor)}")

    if pose_tensor.ndim == 1:
        pose_tensor = pose_tensor.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    batch_shape = pose_tensor.shape[:-1]
    rot_repr = pose_tensor[..., hand_pose_config.ROTATION_SLICE]
    flat_rot = rot_repr.reshape(-1, rot_repr.shape[-1])

    if flat_rot.numel() == 0:
        se3 = _make_batched_eye(batch_shape, pose_tensor.device, pose_tensor.dtype)
        if squeeze_back:
            se3 = se3.squeeze(0)
        return se3

    rot_type_lower = (rot_type or "").lower()
    if rot_type_lower == "quat":
        flat_rot = F.normalize(flat_rot, dim=-1, eps=1e-8)
        rot_mats_flat = transforms.quaternion_to_matrix(flat_rot)
    elif rot_type_lower == "r6d":
        rot_mats_flat = transforms.rotation_6d_to_matrix(flat_rot)
    elif rot_type_lower == "axis":
        rot_mats_flat = transforms.axis_angle_to_matrix(flat_rot)
    elif rot_type_lower == "euler":
        rot_mats_flat = transforms.euler_angles_to_matrix(flat_rot, convention="XYZ")
    elif rot_type_lower == "map":
        rot_mats_flat = transforms.axis_angle_to_matrix(flat_rot)
    else:
        raise ValueError(f"Unsupported rot_type '{rot_type}' for Dex passthrough.")

    rot_mats = rot_mats_flat.view(*batch_shape, 3, 3)
    se3 = _make_batched_eye(batch_shape, pose_tensor.device, pose_tensor.dtype)
    if se3.numel() > 0:
        se3[..., :3, :3] = rot_mats

    if squeeze_back:
        se3 = se3.squeeze(0)
    return se3


def _passthrough_dex_mode(data, rot_type):
    if not isinstance(data, dict):
        return data

    # 记录当前样本使用的 anchor，供 DexGraspNet 反归一化选择边界
    try:
        anchor = str(data.get("trans_anchor", "")).strip().lower()
        if anchor in ("base", "palm_center", "mjcf"):
            set_current_dex_anchor_for_denorm(anchor)
    except Exception:
        pass

    if "norm_pose" not in data and "hand_model_pose" in data:
        data["norm_pose"] = data["hand_model_pose"]
    norm_pose = data.get("norm_pose")

    if isinstance(norm_pose, torch.Tensor):
        se3_seed = _build_se3_from_pose_tensor(norm_pose, rot_type)
        _assign_norm_qpos_and_se3(data, norm_pose, se3_seed)
        return data

    if isinstance(norm_pose, np.ndarray):
        norm_pose_tensor = torch.from_numpy(norm_pose)
        se3_seed = _build_se3_from_pose_tensor(norm_pose_tensor, rot_type)
        _assign_norm_qpos_and_se3(data, norm_pose_tensor, se3_seed)
        data["norm_pose"] = norm_pose_tensor
        return data

    if isinstance(norm_pose, list) and norm_pose:
        pose_tensors = []
        se3_list = []
        for pose_item in norm_pose:
            if isinstance(pose_item, torch.Tensor):
                pose_tensors.append(pose_item)
                se3_list.append(_build_se3_from_pose_tensor(pose_item, rot_type))
            elif isinstance(pose_item, np.ndarray):
                pose_tensor = torch.from_numpy(pose_item)
                pose_tensors.append(pose_tensor)
                se3_list.append(_build_se3_from_pose_tensor(pose_tensor, rot_type))
            else:
                logging.getLogger(__name__).warning(
                    "[hand_helper] Dex 模式遇到不支持的 norm_pose 类型: %s", type(pose_item)
                )
                _ensure_qpos_field(data)
                return data
        data["norm_pose"] = pose_tensors
        _assign_norm_qpos_and_se3(data, pose_tensors, se3_list)
        return data

    _ensure_qpos_field(data)
    return data


def set_debug_shapes(flag: bool) -> None:
    """
    Enable or disable verbose shape logging for hand pose helpers.
    """
    global DEBUG_SHAPES
    DEBUG_SHAPES = bool(flag)
    logging.getLogger(__name__).info(
        "[hand_helper] debug_shapes %s",
        "enabled" if DEBUG_SHAPES else "disabled",
    )


def _debug_log(message: str, *args) -> None:
    if DEBUG_SHAPES:
        logging.getLogger(__name__).debug(message, *args)


def _shape_repr(tensor_like) -> str:
    if hasattr(tensor_like, "shape"):
        try:
            return str(tuple(tensor_like.shape))
        except Exception:
            return f"shape_error({type(tensor_like)})"
    if isinstance(tensor_like, (list, tuple)):
        return f"{type(tensor_like).__name__}(len={len(tensor_like)})"
    return type(tensor_like).__name__


def _assign_norm_qpos_and_se3(data, norm_pose, se3):
    """
    Inject normalized qpos (from norm_pose) and overwrite SE(3) translation
    with normalized translation values.
    """
    if isinstance(norm_pose, list):
        qpos_list = [pose[..., 3 : 3 + hand_pose_config.JOINT_ANGLE_DIM] for pose in norm_pose]
        data['qpos'] = qpos_list

        if not isinstance(se3, list):
            raise TypeError("Expected 'se3' to be a list when 'norm_pose' is a list.")

        se3_norm_list = []
        for pose_item, se3_item in zip(norm_pose, se3):
            se3_clone = se3_item.clone()
            se3_clone[..., :3, 3] = pose_item[..., :3].to(se3_clone.dtype)
            se3_norm_list.append(se3_clone)
        data['se3'] = se3_norm_list
        return

    if not isinstance(norm_pose, torch.Tensor):
        raise TypeError(f"norm_pose must be Tensor or list, got {type(norm_pose)}")

    norm_trans = norm_pose[..., :3]
    norm_qpos = norm_pose[..., 3 : 3 + hand_pose_config.JOINT_ANGLE_DIM]
    data['qpos'] = norm_qpos

    if isinstance(se3, torch.Tensor):
        se3_norm = se3.clone().to(norm_pose.dtype)
        se3_norm[..., :3, 3] = norm_trans.to(se3_norm.dtype)
        data['se3'] = se3_norm
    else:
        raise TypeError(f"Unsupported se3 type for normalized assignment: {type(se3)}")


def load_pose_stats_if_needed():
    """Loads pose statistics from the JSON file if not already loaded."""
    global POSE_STATS
    if POSE_STATS is None:
        try:
            with open(JSON_STATS_FILE_PATH, "r") as f:
                raw_stats = json.load(f)
            POSE_STATS = {}
            for mode_key, stats_list in raw_stats.items():
                POSE_STATS[mode_key] = {
                    item["dimension_label"]: item for item in stats_list
                }
        except FileNotFoundError:
            print(
                f"CRITICAL WARNING: Statistics file {JSON_STATS_FILE_PATH} not found. Normalization will likely fail or be incorrect."
            )
            POSE_STATS = {}
        except json.JSONDecodeError:
            print(
                f"CRITICAL WARNING: Error decoding JSON from {JSON_STATS_FILE_PATH}. Normalization will be incorrect."
            )
            POSE_STATS = {}
        except Exception as e:
            print(
                f"CRITICAL WARNING: An unexpected error occurred while loading statistics: {e}"
            )
            POSE_STATS = {}


load_pose_stats_if_needed()


# ============================================================================
# 新的统一归一化方法（ObjectCentric 数据集专用）
# ============================================================================

OBJECTCENTRIC_NORM_PARAMS_PATH = "assets/objectcentric_normalization_stats_val.pt"
OBJECTCENTRIC_NORM_PARAMS = None

def load_objectcentric_norm_params(mode: str = None):
    """加载 ObjectCentric 数据集的归一化参数（仅加载一次）"""
    if mode == "DexGraspAnythingVirtual":
        return

    global OBJECTCENTRIC_NORM_PARAMS
    if OBJECTCENTRIC_NORM_PARAMS is None:
        try:
            OBJECTCENTRIC_NORM_PARAMS = torch.load(OBJECTCENTRIC_NORM_PARAMS_PATH)
            print(f"Loaded ObjectCentric normalization params from {OBJECTCENTRIC_NORM_PARAMS_PATH}")
            print(f"  Spatial bbox: min={OBJECTCENTRIC_NORM_PARAMS['spatial_bbox_min'].numpy()}, "
                  f"max={OBJECTCENTRIC_NORM_PARAMS['spatial_bbox_max'].numpy()}")
        except FileNotFoundError:
            raise RuntimeError(
                f"ObjectCentric normalization params not found at {OBJECTCENTRIC_NORM_PARAMS_PATH}. "
                f"Please run tests/compute_objectcentric_normalization_stats.py first."
            )


def normalize_spatial_objectcentric(coords, use_margin=True):
    """
    使用全局 bbox 统一归一化空间坐标（用于 scene_pc 和 hand_trans）

    Args:
        coords: (*, 3) 空间坐标
        use_margin: 是否使用带 margin 的 bbox（推荐）

    Returns:
        coords_norm: 归一化到 [-1, 1]³ 的坐标
    """
    load_objectcentric_norm_params()
    params = OBJECTCENTRIC_NORM_PARAMS

    if use_margin:
        bbox_min = params['spatial_bbox_min_with_margin'].to(coords.device, coords.dtype)
        bbox_max = params['spatial_bbox_max_with_margin'].to(coords.device, coords.dtype)
    else:
        bbox_min = params['spatial_bbox_min'].to(coords.device, coords.dtype)
        bbox_max = params['spatial_bbox_max'].to(coords.device, coords.dtype)

    # Min-Max 归一化到 [-1, 1]
    bbox_range = bbox_max - bbox_min
    bbox_range = torch.clamp(bbox_range, min=1e-8)  # 防止除零

    coords_norm = 2.0 * (coords - bbox_min) / bbox_range - 1.0

    return coords_norm


def denormalize_spatial_objectcentric(coords_norm, use_margin=True):
    """
    反归一化空间坐标

    Args:
        coords_norm: (*, 3) 归一化后的坐标
        use_margin: 是否使用带 margin 的 bbox（需与归一化时一致）

    Returns:
        coords: 反归一化后的坐标
    """
    load_objectcentric_norm_params()
    params = OBJECTCENTRIC_NORM_PARAMS

    if use_margin:
        bbox_min = params['spatial_bbox_min_with_margin'].to(coords_norm.device, coords_norm.dtype)
        bbox_max = params['spatial_bbox_max_with_margin'].to(coords_norm.device, coords_norm.dtype)
    else:
        bbox_min = params['spatial_bbox_min'].to(coords_norm.device, coords_norm.dtype)
        bbox_max = params['spatial_bbox_max'].to(coords_norm.device, coords_norm.dtype)

    # 反归一化
    bbox_range = bbox_max - bbox_min
    bbox_range = torch.clamp(bbox_range, min=1e-8)

    coords = (coords_norm + 1.0) / 2.0 * bbox_range + bbox_min

    return coords


def normalize_joints_objectcentric(joint_angles):
    """
    归一化关节角度（16维）

    Args:
        joint_angles: (*, 16) 关节角度

    Returns:
        joint_angles_norm: 归一化到 [-1, 1] 的关节角度
    """
    load_objectcentric_norm_params()
    params = OBJECTCENTRIC_NORM_PARAMS

    joint_min = params['joint_angles_min'].to(joint_angles.device, joint_angles.dtype)
    joint_max = params['joint_angles_max'].to(joint_angles.device, joint_angles.dtype)

    # Min-Max 归一化到 [-1, 1]
    joint_range = joint_max - joint_min
    joint_range = torch.clamp(joint_range, min=1e-8)

    joint_angles_norm = 2.0 * (joint_angles - joint_min) / joint_range - 1.0

    return joint_angles_norm


def denormalize_joints_objectcentric(joint_angles_norm):
    """
    反归一化关节角度

    Args:
        joint_angles_norm: (*, 16) 归一化后的关节角度

    Returns:
        joint_angles: 反归一化后的关节角度
    """
    load_objectcentric_norm_params()
    params = OBJECTCENTRIC_NORM_PARAMS

    joint_min = params['joint_angles_min'].to(joint_angles_norm.device, joint_angles_norm.dtype)
    joint_max = params['joint_angles_max'].to(joint_angles_norm.device, joint_angles_norm.dtype)

    # 反归一化
    joint_range = joint_max - joint_min
    joint_range = torch.clamp(joint_range, min=1e-8)

    joint_angles = (joint_angles_norm + 1.0) / 2.0 * joint_range + joint_min

    return joint_angles


# ============================================================================
# ObjectCentric 统一归一化的处理函数
# ============================================================================

def process_hand_pose_objectcentric(data, rot_type):
    """
    使用统一归一化策略处理手部姿态（ObjectCentric 数据集）

    关键特性：
    1. 点云和 hand_trans 使用同一 bbox 归一化
    2. 关节角度使用独立归一化
    3. 旋转只做正交化，不做 min-max 归一化

    Args:
        data (dict): 包含以下键：
            - 'scene_pc' or 'pointclouds': (B, N, 3 或 6) 点云
            - 'hand_model_pose': (B, M, 23) 手部姿态
            - 'se3': (B, M, 4, 4) SE(3) 变换矩阵
        rot_type (str): 旋转表示类型 ('quat', 'r6d', 等)

    Returns:
        dict: 更新后的数据字典，包含：
            - 'norm_scene_pc': 归一化后的场景点云（xyz 归一化，rgb 保持）
            - 'norm_pose': 归一化后的手部姿态
            - 'hand_model_pose': 处理后的手部姿态（用于反归一化）
    """
    # 获取点云数据
    if 'pointclouds' in data:
        pointclouds = data['pointclouds']  # (B, N, 3 或 6)
    elif 'scene_pc' in data:
        pointclouds = data['scene_pc']  # (B, N, 3 或 6)
    else:
        raise KeyError("数据中必须包含 'pointclouds' 或 'scene_pc'")

    # 提取 xyz 坐标
    if pointclouds.shape[-1] == 6:  # xyz + rgb
        scene_xyz = pointclouds[..., :3]  # (B, N, 3)
        scene_rgb = pointclouds[..., 3:]  # (B, N, 3)
    else:
        scene_xyz = pointclouds  # (B, N, 3)
        scene_rgb = None

    # 归一化点云坐标
    B, N, _ = scene_xyz.shape
    scene_xyz_flat = scene_xyz.reshape(-1, 3)  # (B*N, 3)
    scene_xyz_norm_flat = normalize_spatial_objectcentric(scene_xyz_flat)
    scene_xyz_norm = scene_xyz_norm_flat.reshape(B, N, 3)

    # 重新组装点云
    if scene_rgb is not None:
        pointclouds_norm = torch.cat([scene_xyz_norm, scene_rgb], dim=-1)
    else:
        pointclouds_norm = scene_xyz_norm

    # 归一化后的场景点云写入统一键名
    data['norm_scene_pc'] = pointclouds_norm

    # 处理手部姿态
    se3 = data['se3']  # (B, M, 4, 4)
    hand_model_pose = data['hand_model_pose']  # (B, M, 23)

    B, M, pose_dim = hand_model_pose.shape

    # 提取各部分
    hand_trans = hand_model_pose[..., :3]  # (B, M, 3)
    hand_joints = hand_model_pose[..., 3:19]  # (B, M, 16)
    hand_quat = hand_model_pose[..., 19:23]  # (B, M, 4)

    # 从 SE(3) 提取旋转矩阵并转换
    se3_flat = se3.reshape(B * M, 4, 4)
    rmat = se3_flat[:, :3, :3]  # (B*M, 3, 3)

    if rot_type == 'quat':
        rot_representation = transforms.matrix_to_quaternion(rmat)  # (B*M, 4)
        # Quaternion 不归一化（已经是单位四元数）
        rot_norm = rot_representation
    elif rot_type == 'r6d':
        rot_representation = transforms.matrix_to_rotation_6d(rmat)  # (B*M, 6)
        # r6d 只做 Gram-Schmidt 正交化
        rot_norm = normalize_rot6d(rot_representation)
    else:
        raise ValueError(f"Unsupported rot_type: {rot_type}")

    # 归一化 trans 和 joints
    hand_trans_flat = hand_trans.reshape(B * M, 3)
    hand_joints_flat = hand_joints.reshape(B * M, 16)

    hand_trans_norm_flat = normalize_spatial_objectcentric(hand_trans_flat)
    hand_joints_norm_flat = normalize_joints_objectcentric(hand_joints_flat)

    # 重塑回原始形状
    hand_trans_norm = hand_trans_norm_flat.reshape(B, M, 3)
    hand_joints_norm = hand_joints_norm_flat.reshape(B, M, 16)
    rot_norm = rot_norm.reshape(B, M, -1)

    # 拼接归一化后的 pose
    norm_pose = torch.cat([hand_trans_norm, hand_joints_norm, rot_norm], dim=-1)

    # 保存归一化后的 pose
    data['norm_pose'] = norm_pose

    # 更新 hand_model_pose 为处理后的版本（包含新的旋转表示）
    rot_representation = rot_representation.reshape(B, M, -1)
    processed_hand_model_pose = torch.cat([hand_trans, hand_joints, rot_representation], dim=-1)
    data['hand_model_pose'] = processed_hand_model_pose

    # 将归一化信息注入 qpos / se3
    _assign_norm_qpos_and_se3(data, norm_pose, se3)

    return data


def process_hand_pose_test_objectcentric(data, rot_type):
    """
    测试时使用统一归一化策略处理手部姿态（ObjectCentric 数据集）

    与 process_hand_pose_objectcentric 相同的处理逻辑
    """
    return process_hand_pose_objectcentric(data, rot_type)


def denorm_hand_pose_objectcentric(hand_pose, rot_type):
    """
    使用统一归一化策略反归一化手部姿态（ObjectCentric 数据集）

    Args:
        hand_pose: (*, D) 归一化后的手部姿态
            - 对于 r6d: D = 3 + 16 + 6 = 25
            - 对于 quat: D = 3 + 16 + 4 = 23
        rot_type (str): 旋转表示类型

    Returns:
        hand_pose_denorm: 反归一化后的手部姿态（与输入形状相同）
    """
    if rot_type == 'r6d':
        trans_dim, joint_dim, rot_dim = 3, 16, 6
    elif rot_type == 'quat':
        trans_dim, joint_dim, rot_dim = 3, 16, 4
    else:
        raise ValueError(f"Unsupported rot_type: {rot_type}")

    orig_shape = hand_pose.shape

    # 处理多维输入
    if len(orig_shape) == 3:  # (B, M, D)
        hand_pose_flat = hand_pose.reshape(-1, orig_shape[-1])
    else:
        hand_pose_flat = hand_pose

    # 分离各部分
    hand_trans_norm = hand_pose_flat[:, :trans_dim]
    hand_joints_norm = hand_pose_flat[:, trans_dim:trans_dim+joint_dim]
    hand_rot_norm = hand_pose_flat[:, trans_dim+joint_dim:trans_dim+joint_dim+rot_dim]

    # 反归一化 trans 和 joints
    hand_trans_denorm = denormalize_spatial_objectcentric(hand_trans_norm)
    hand_joints_denorm = denormalize_joints_objectcentric(hand_joints_norm)

    # 旋转部分处理
    if rot_type == 'r6d':
        # r6d 需要重新正交化（确保是有效的 r6d）
        hand_rot_denorm = normalize_rot6d(hand_rot_norm)
    elif rot_type == 'quat':
        # Quaternion 归一化到单位四元数
        hand_rot_denorm = F.normalize(hand_rot_norm, p=2, dim=-1)
    else:
        hand_rot_denorm = hand_rot_norm

    # 重新拼接
    hand_pose_denorm_flat = torch.cat([hand_trans_denorm, hand_joints_denorm, hand_rot_denorm], dim=-1)

    # 恢复原始形状
    if len(orig_shape) == 3:
        hand_pose_denorm = hand_pose_denorm_flat.reshape(orig_shape)
    else:
        hand_pose_denorm = hand_pose_denorm_flat

    return hand_pose_denorm


def norm_hand_pose_objectcentric(hand_pose, rot_type):
    """
    使用统一归一化策略归一化手部姿态（ObjectCentric 数据集）
    """
    if not isinstance(hand_pose, torch.Tensor):
        raise TypeError("ObjectCentric 模式仅支持 torch.Tensor 形式的 hand_pose。")

    if rot_type == 'r6d':
        rot_dim = 6
    elif rot_type == 'quat':
        rot_dim = 4
    else:
        raise ValueError(f"Unsupported rot_type for ObjectCentric normalization: {rot_type}")

    orig_shape = hand_pose.shape
    if len(orig_shape) == 3:
        hand_pose_flat = hand_pose.reshape(-1, orig_shape[-1])
    else:
        hand_pose_flat = hand_pose

    hand_trans = hand_pose_flat[:, :3]
    hand_joints = hand_pose_flat[:, 3 : 3 + hand_pose_config.JOINT_ANGLE_DIM]
    hand_rot = hand_pose_flat[:, 3 + hand_pose_config.JOINT_ANGLE_DIM : 3 + hand_pose_config.JOINT_ANGLE_DIM + rot_dim]

    hand_trans_norm = normalize_spatial_objectcentric(hand_trans)
    hand_joints_norm = normalize_joints_objectcentric(hand_joints)

    if rot_type == 'r6d':
        hand_rot_norm = normalize_rot6d(hand_rot)
    elif rot_type == 'quat':
        hand_rot_norm = F.normalize(hand_rot, p=2, dim=-1)
    else:
        hand_rot_norm = hand_rot

    hand_pose_norm_flat = torch.cat(
        [hand_trans_norm, hand_joints_norm, hand_rot_norm], dim=-1
    )

    if len(orig_shape) == 3:
        hand_pose_norm = hand_pose_norm_flat.reshape(orig_shape)
    else:
        hand_pose_norm = hand_pose_norm_flat

    return hand_pose_norm


def denormalize_pointcloud_objectcentric(pointclouds_norm):
    """
    反归一化点云（ObjectCentric 数据集）

    Args:
        pointclouds_norm: (*, N, 3 或 6) 归一化后的点云

    Returns:
        pointclouds: 反归一化后的点云
    """
    orig_shape = pointclouds_norm.shape

    # 提取 xyz
    if orig_shape[-1] == 6:
        scene_xyz_norm = pointclouds_norm[..., :3]
        scene_rgb = pointclouds_norm[..., 3:]
    else:
        scene_xyz_norm = pointclouds_norm
        scene_rgb = None

    # 反归一化 xyz
    scene_xyz_norm_flat = scene_xyz_norm.reshape(-1, 3)
    scene_xyz_denorm_flat = denormalize_spatial_objectcentric(scene_xyz_norm_flat)
    scene_xyz_denorm = scene_xyz_denorm_flat.reshape(*orig_shape[:-1], 3)

    # 重新组装
    if scene_rgb is not None:
        pointclouds = torch.cat([scene_xyz_denorm, scene_rgb], dim=-1)
    else:
        pointclouds = scene_xyz_denorm

    return pointclouds


# ============================================================================
# 旧的归一化方法（保持向后兼容）
# ============================================================================

def get_min_max_from_stats(
    mode, dimension_labels, device, dtype, tensor_type=torch.Tensor
):
    """Helper function to retrieve min and max values for given dimension labels from loaded stats."""
    if POSE_STATS is None or not POSE_STATS:
        raise RuntimeError(
            "Pose statistics are not loaded. Cannot proceed with normalization."
        )
    if mode not in POSE_STATS:
        raise ValueError(
            f"Mode '{mode}' not found in loaded statistics. Available modes: {list(POSE_STATS.keys())}"
        )

    mins = []
    maxs = []
    for label in dimension_labels:
        if label not in POSE_STATS[mode]:
            raise ValueError(
                f"Dimension label '{label}' not found in mode '{mode}'. Check statistics file."
            )
        mins.append(POSE_STATS[mode][label]["min"])
        maxs.append(POSE_STATS[mode][label]["max"])

    if tensor_type == torch.Tensor:
        return torch.tensor(mins, device=device, dtype=dtype), torch.tensor(
            maxs, device=device, dtype=dtype
        )
    elif tensor_type == np.ndarray:
        return np.array(mins, dtype=dtype), np.array(maxs, dtype=dtype)
    else:
        raise TypeError(f"Unsupported tensor_type: {tensor_type}")


# Rotation normalization functions
def normalize_rot6d_torch(rot):
    if rot.shape[-1] == 3:
        unflatten = True
        rot = rot.flatten(-2, -1)
    else:
        unflatten = False
    a1, a2 = rot[..., :3], rot[..., 3:]
    b1 = F.normalize(a1, p=2, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, p=2, dim=-1)
    rot = torch.cat([b1, b2], dim=-1)
    if unflatten:
        rot = rot.unflatten(-1, (2, 3))
    return rot


def normalize_np(x):
    x_n = np.linalg.norm(x, axis=-1, keepdims=True)
    x_n = x_n.clip(min=1e-8)
    x = x / x_n
    return x


def normalize_rot6d_numpy(rot):
    if rot.shape[-1] == 3:
        unflatten = True
        undim = True
        ori_shape = rot.shape[:-2]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    elif len(rot.shape) > 2:
        unflatten = False
        undim = True
        ori_shape = rot.shape[:-1]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    else:
        unflatten = False
        undim = False
        ori_shape = None
    a1, a2 = rot[:, :3], rot[:, 3:]
    b1 = normalize_np(a1)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = normalize_np(b2)
    rot = np.concatenate([b1, b2], axis=-1)
    if unflatten:
        rot = rot.reshape(ori_shape + (2, 3))
    elif undim:
        rot = rot.reshape(ori_shape + (6,))
    return rot


def normalize_rot6d(rot):
    if isinstance(rot, torch.Tensor):
        return normalize_rot6d_torch(rot)
    elif isinstance(rot, np.ndarray):
        return normalize_rot6d_numpy(rot)
    else:
        raise NotImplementedError


# Constants
NORM_UPPER = 1.0
NORM_LOWER = -1.0

ROT_DIM_DICT = {
    "quat": 4,
    "axis": 3,
    "euler": 3,
    "r6d": 6,
    "map": 3,
}


# -----------------------------
# Generic pose utilities (pure functions)
# -----------------------------
def split_pose(pose: torch.Tensor, rot_type: str):
    """
    将 processed hand pose 拆分为 (trans, qpos, rot)。
    pose: (..., D)
    返回: trans (..., 3), qpos (..., JOINT_ANGLE_DIM), rot (..., rot_dim)
    """
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    trans = pose[..., :3]
    qpos = pose[..., 3 : 3 + hand_pose_config.JOINT_ANGLE_DIM]
    rot = pose[..., 3 + hand_pose_config.JOINT_ANGLE_DIM : 3 + hand_pose_config.JOINT_ANGLE_DIM + rot_dim]
    return trans, qpos, rot


def join_pose(trans: torch.Tensor, qpos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """
    将 (trans, qpos, rot) 拼接为 processed hand pose。
    """
    return torch.cat([trans, qpos, rot], dim=-1)


def rot_matrix_to_repr(R: torch.Tensor, rot_type: str) -> torch.Tensor:
    """使用 RotationSpec 统一接口将旋转矩阵转换为指定表示"""
    rot_type_lower = (rot_type or "").lower()
    # 对于 'map' 需要 se3 才能取对数映射，使用 rot_from_se3
    if rot_type_lower == "map":
        raise ValueError(f"rot_matrix_to_repr 不支持 rot_type '{rot_type}'. 如需 'map'，请使用 rot_from_se3。")

    spec = get_rotation_spec(rot_type_lower)
    return spec.from_matrix_fn(R)


def repr_to_rot_matrix(rot_repr: torch.Tensor, rot_type: str) -> torch.Tensor:
    """使用 RotationSpec 统一接口将旋转表示转换为旋转矩阵"""
    rot_type_lower = (rot_type or "").lower()
    if rot_type_lower == "map":
        raise ValueError(f"repr_to_rot_matrix 不支持 rot_type '{rot_type}'. 如需 'map'，请使用 se3_exp_map。")

    spec = get_rotation_spec(rot_type_lower)
    if spec.needs_normalization:
        rot_repr = spec.normalize_fn(rot_repr)
    return spec.to_matrix_fn(rot_repr)


def rot_from_se3(se3: torch.Tensor, rot_type: str) -> torch.Tensor:
    """
    从 SE(3) 矩阵中提取旋转并转换为指定表示。
    - 若 rot_type == 'map'，返回 se3_log_map 的旋转部分（axis-angle）。
    - 否则基于 R = se3[:3,:3] 转换。
    """
    rot_type_lower = (rot_type or "").lower()
    if rot_type_lower == "map":
        log_map = se3_log_map(se3.view(-1, 4, 4)).view(*se3.shape[:-2], 6)
        return log_map[..., 3:]
    R = se3[..., :3, :3]
    return rot_matrix_to_repr(R, rot_type)


def normalize_rot_component(rot: torch.Tensor, rot_type: str) -> torch.Tensor:
    """
    对旋转分量做归一化/正交化：
    - quat: 单位化
    - r6d: Gram-Schmidt 正交化
    - 其他: 原样返回
    """
    rot_type_lower = (rot_type or "").lower()
    if rot_type_lower == "quat":
        return F.normalize(rot, p=2, dim=-1)
    if rot_type_lower == "r6d":
        return normalize_rot6d(rot)
    return rot


def _flatten_reshape(x: torch.Tensor, fn):
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else x
    y_flat = fn(x_flat)
    if x.ndim >= 2:
        return y_flat.reshape(*orig_shape)
    return y_flat


def normalize_pose_objectcentric_components(trans: torch.Tensor, qpos: torch.Tensor, rot: torch.Tensor, rot_type: str):
    """
    对 (trans, qpos, rot) 逐组件进行 ObjectCentric 归一化。
    返回: (trans_norm, qpos_norm, rot_norm)
    """
    trans_flat = trans.reshape(-1, 3)
    qpos_flat = qpos.reshape(-1, hand_pose_config.JOINT_ANGLE_DIM)
    rot_flat = rot.reshape(-1, rot.shape[-1])

    trans_n = normalize_spatial_objectcentric(trans_flat).reshape_as(trans)
    qpos_n = normalize_joints_objectcentric(qpos_flat).reshape_as(qpos)
    rot_n = normalize_rot_component(rot_flat, rot_type).reshape_as(rot)
    return trans_n, qpos_n, rot_n


def normalize_pose_dex_components(trans: torch.Tensor, qpos: torch.Tensor, rot: torch.Tensor, rot_type: str, mode: str):
    """
    对 (trans, qpos) 使用 DexGraspAnything 统计进行 [-1,1] 归一化；rot 保持不变（保持与历史实现一致）。
    若统计缺失则透传。
    返回: (trans_norm, qpos_norm, rot_out)
    """
    stats = _load_dex_stats_if_needed(mode)
    if not stats:
        return trans, qpos, rot
    mode_n = _normalize_mode(mode)
    trans_key = "hand_trans" if mode_n in ("dexgraspanything_base", "dexgraspnet") else "palm_center"
    t_bounds = stats.get(trans_key)
    j_bounds = stats.get("joint_angles")
    if t_bounds is None or j_bounds is None:
        return trans, qpos, rot

    def _apply_bounds(x: torch.Tensor, bounds: dict):
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = normalize_with_bounds(x_flat, bounds)
        return y_flat.reshape_as(x)

    trans_n = _apply_bounds(trans, t_bounds)
    qpos_n = _apply_bounds(qpos, j_bounds)
    # 与历史实现保持一致：不归一化 rot；但若是 quat，确保单位化更安全
    if (rot_type or "").lower() == "quat":
        rot_out = F.normalize(rot, p=2, dim=-1)
    else:
        rot_out = rot
    return trans_n, qpos_n, rot_out


# Translation normalization functions
def normalize_trans_torch(hand_t, mode, rot_type_for_map_distinction="not_map"):
    if rot_type_for_map_distinction == "map":
        labels = [f"se3_log_map_dim_{i}" for i in range(3)]
    else:
        labels = ["translation_x", "translation_y", "translation_z"]
    t_min, t_max = get_min_max_from_stats(mode, labels, hand_t.device, hand_t.dtype)

    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8

    t_scaled_0_1 = torch.div((hand_t - t_min), t_range)
    t_final_normalized = t_scaled_0_1 * (NORM_UPPER - NORM_LOWER) + NORM_LOWER

    t = t_final_normalized

    return t


def denormalize_trans_torch(hand_t, mode, rot_type_for_map_distinction="not_map"):
    if rot_type_for_map_distinction == "map":
        labels = [f"se3_log_map_dim_{i}" for i in range(3)]
    else:
        labels = ["translation_x", "translation_y", "translation_z"]
    t_min, t_max = get_min_max_from_stats(mode, labels, hand_t.device, hand_t.dtype)

    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8

    t = hand_t + ((NORM_UPPER - NORM_LOWER) / 2.0)
    t /= NORM_UPPER - NORM_LOWER
    t = t * t_range + t_min

    return t


def normalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction="not_map"):
    if rot_type_for_map_distinction == "map":
        labels = [f"se3_log_map_dim_{i}" for i in range(3)]
    else:
        labels = ["translation_x", "translation_y", "translation_z"]
    t_min, t_max = get_min_max_from_stats(
        mode, labels, None, hand_t.dtype, tensor_type=np.ndarray
    )

    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8

    t = (hand_t - t_min) / t_range
    t = t * (NORM_UPPER - NORM_LOWER) + NORM_LOWER
    return t


def denormalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction="not_map"):
    if rot_type_for_map_distinction == "map":
        labels = [f"se3_log_map_dim_{i}" for i in range(3)]
    else:
        labels = ["translation_x", "translation_y", "translation_z"]
    t_min, t_max = get_min_max_from_stats(
        mode, labels, None, hand_t.dtype, tensor_type=np.ndarray
    )

    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8

    t = hand_t + ((NORM_UPPER - NORM_LOWER) / 2.0)
    t /= NORM_UPPER - NORM_LOWER
    t = t * t_range + t_min
    return t


# Joint parameter normalization functions
def normalize_param_torch(hand_param, mode):
    labels = [f"joint_angle_{i}" for i in range(hand_pose_config.JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(
        mode, labels, hand_param.device, hand_param.dtype
    )

    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = torch.div((hand_param - p_min), p_range)
    p = p * (NORM_UPPER - NORM_LOWER) + NORM_LOWER

    # 在 dexgraspnet 模式下，忽略（置 0）恒为 0 的 qpos 维度在归一化后的贡献
    disabled = get_disabled_qpos_indices(mode)
    if disabled:
        idx = list(disabled)
        p[..., idx] = 0.0

    return p


def denormalize_param_torch(hand_param, mode):
    labels = [f"joint_angle_{i}" for i in range(hand_pose_config.JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(
        mode, labels, hand_param.device, hand_param.dtype
    )

    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = hand_param + ((NORM_UPPER - NORM_LOWER) / 2.0)
    p /= NORM_UPPER - NORM_LOWER
    p = p * p_range + p_min

    # 在 dexgraspnet 模式下，直接将被忽略的维度置 0（数值域内）
    disabled = get_disabled_qpos_indices(mode)
    if disabled:
        idx = list(disabled)
        p[..., idx] = 0.0

    return p


def normalize_param_numpy(hand_param, mode):
    labels = [f"joint_angle_{i}" for i in range(hand_pose_config.JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(
        mode, labels, None, hand_param.dtype, tensor_type=np.ndarray
    )

    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = (hand_param - p_min) / p_range
    p = p * (NORM_UPPER - NORM_LOWER) + NORM_LOWER

    # numpy 版本同样忽略 dexgraspnet 下的恒零维度
    disabled = get_disabled_qpos_indices(mode)
    if disabled:
        idx = list(disabled)
        p[..., idx] = 0.0
    return p


def denormalize_param_numpy(hand_param, mode):
    labels = [f"joint_angle_{i}" for i in range(hand_pose_config.JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(
        mode, labels, None, hand_param.dtype, tensor_type=np.ndarray
    )

    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = hand_param + ((NORM_UPPER - NORM_LOWER) / 2.0)
    p /= NORM_UPPER - NORM_LOWER
    p = p * p_range + p_min

    disabled = get_disabled_qpos_indices(mode)
    if disabled:
        idx = list(disabled)
        p[..., idx] = 0.0
    return p


# Rotation normalization functions
def _get_rot_labels(rot_type):
    """获取旋转表示的维度标签（使用 RotationSpec）"""
    # 特殊处理 'map' 类型（不在 RotationSpec 中）
    if rot_type == "map":
        return [f"se3_log_map_dim_{i}" for i in range(3, 6)]

    # 对于标准旋转类型，使用 RotationSpec
    try:
        spec = get_rotation_spec(rot_type)
        return spec.labels
    except ValueError:
        raise ValueError(
            f"Unsupported rotation type for generating dimension labels: {rot_type}"
        )


def normalize_rot_torch(hand_r, rot_type, mode):
    """归一化旋转表示（使用 RotationSpec 获取标签）"""
    if rot_type == "quat":
        return hand_r

    spec = get_rotation_spec(rot_type)
    labels = spec.labels
    r_min, r_max = get_min_max_from_stats(mode, labels, hand_r.device, hand_r.dtype)

    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8

    r = torch.div((hand_r - r_min), r_range)
    r = r * (NORM_UPPER - NORM_LOWER) + NORM_LOWER

    return r


def denormalize_rot_torch(hand_r, rot_type, mode):
    """反归一化旋转表示（使用 RotationSpec 获取标签）"""
    if rot_type == "quat":
        return hand_r

    spec = get_rotation_spec(rot_type)
    labels = spec.labels
    r_min, r_max = get_min_max_from_stats(mode, labels, hand_r.device, hand_r.dtype)

    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8

    r = hand_r + ((NORM_UPPER - NORM_LOWER) / 2.0)
    r /= NORM_UPPER - NORM_LOWER
    r = r * r_range + r_min

    return r


def normalize_rot_numpy(hand_r, rot_type, mode):
    """归一化旋转表示（numpy 版本，使用 RotationSpec 获取标签）"""
    if rot_type == "quat":
        return hand_r

    spec = get_rotation_spec(rot_type)
    labels = spec.labels
    r_min, r_max = get_min_max_from_stats(
        mode, labels, None, hand_r.dtype, tensor_type=np.ndarray
    )

    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8

    r = (hand_r - r_min) / r_range
    r = r * (NORM_UPPER - NORM_LOWER) + NORM_LOWER
    return r


def denormalize_rot_numpy(hand_r, rot_type, mode):
    """反归一化旋转表示（numpy 版本，使用 RotationSpec 获取标签）"""
    if rot_type == "quat":
        return hand_r

    spec = get_rotation_spec(rot_type)
    labels = spec.labels
    r_min, r_max = get_min_max_from_stats(
        mode, labels, None, hand_r.dtype, tensor_type=np.ndarray
    )

    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8

    r = hand_r + ((NORM_UPPER - NORM_LOWER) / 2.0)
    r /= NORM_UPPER - NORM_LOWER
    r = r * r_range + r_min
    return r


# Main pose normalization functions
def norm_hand_pose(hand_pose, rot_type, mode):
    """Normalize hand pose parameters"""
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    if mode == "objectcentric_single_object":
        return norm_hand_pose_objectcentric(hand_pose, rot_type)
    if isinstance(hand_pose, torch.Tensor):
        hand_t, hand_param, hand_r = hand_pose.split(
            (3, hand_pose_config.JOINT_ANGLE_DIM, rot_dim), dim=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t = normalize_trans_torch(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r = normalize_rot_torch(hand_r, rot_type, mode)
        hand_param = normalize_param_torch(hand_param, mode)
        hand = torch.cat([hand_t, hand_param, hand_r], dim=-1)
    elif isinstance(hand_pose, np.ndarray):
        hand_t, hand_param, hand_r = np.split(
            hand_pose, [3, 3 + hand_pose_config.JOINT_ANGLE_DIM], axis=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t = normalize_trans_numpy(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r = normalize_rot_numpy(hand_r, rot_type, mode)
        hand_param = normalize_param_numpy(hand_param, mode)
        hand = np.concatenate([hand_t, hand_param, hand_r], axis=-1)
    else:
        raise NotImplementedError

    return hand


def denorm_hand_pose(hand_pose, rot_type, mode):
    """Denormalize hand pose parameters"""
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    if mode == "objectcentric_single_object":
        if not isinstance(hand_pose, torch.Tensor):
            raise TypeError("ObjectCentric 模式仅支持 torch.Tensor 形式的 hand_pose。")
        return denorm_hand_pose_objectcentric(hand_pose, rot_type)
    if isinstance(hand_pose, torch.Tensor):
        hand_t, hand_param, hand_r = hand_pose.split(
            (3, hand_pose_config.JOINT_ANGLE_DIM, rot_dim), dim=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t = denormalize_trans_torch(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r = denormalize_rot_torch(hand_r, rot_type, mode)
        hand_param = denormalize_param_torch(hand_param, mode)
        hand = torch.cat([hand_t, hand_param, hand_r], dim=-1)
    elif isinstance(hand_pose, np.ndarray):
        hand_t, hand_param, hand_r = np.split(
            hand_pose, [3, 3 + hand_pose_config.JOINT_ANGLE_DIM], axis=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t = denormalize_trans_numpy(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r = denormalize_rot_numpy(hand_r, rot_type, mode)
        hand_param = denormalize_param_numpy(hand_param, mode)
        hand = np.concatenate([hand_t, hand_param, hand_r], axis=-1)
    else:
        raise NotImplementedError

    return hand


def norm_hand_pose_robust(hand_pose, rot_type, mode):
    """
    Normalize hand pose parameters. Supports both single and multi-grasp formats:
    - Single grasp: [B, pose_dim]
    - Multi grasp: [B, num_grasps, pose_dim]
    """
    rot_dim = ROT_DIM_DICT[rot_type.lower()]

    if _is_dex_mode(mode):
        # DexGraspAnything 模式使用基于 .pt 的统计进行归一化
        return _norm_hand_pose_dex(hand_pose, rot_type, _normalize_mode(mode))
    if mode == "objectcentric_single_object":
        return norm_hand_pose_objectcentric(hand_pose, rot_type)

    if isinstance(hand_pose, torch.Tensor):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose

        hand_t, hand_param, hand_r = hand_pose_reshaped.split(
            (3, hand_pose_config.JOINT_ANGLE_DIM, rot_dim), dim=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t_norm = normalize_trans_torch(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r_norm = normalize_rot_torch(hand_r, rot_type, mode)
        hand_param_norm = normalize_param_torch(hand_param, mode)
        hand_norm_reshaped = torch.cat(
            [hand_t_norm, hand_param_norm, hand_r_norm], dim=-1
        )

        if len(orig_shape) == 3:
            hand = hand_norm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_norm_reshaped

    elif isinstance(hand_pose, np.ndarray):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose

        hand_t, hand_param, hand_r = np.split(
            hand_pose_reshaped, [3, 3 + hand_pose_config.JOINT_ANGLE_DIM], axis=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t_norm = normalize_trans_numpy(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r_norm = normalize_rot_numpy(hand_r, rot_type, mode)
        hand_param_norm = normalize_param_numpy(hand_param, mode)
        hand_norm_reshaped = np.concatenate(
            [hand_t_norm, hand_param_norm, hand_r_norm], axis=-1
        )

        if len(orig_shape) == 3:
            hand = hand_norm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_norm_reshaped
    else:
        raise NotImplementedError

    return hand


def _denorm_hand_pose_dex(hand_pose, rot_type, mode_normalized: str):
    stats = _load_dex_stats_if_needed(mode_normalized)
    if not stats:
        logging.getLogger(__name__).warning(
            "[hand_helper] DexGraspAnything 模式缺少归一化统计，直接透传 hand_pose (mode=%s, shape=%s)",
            mode_normalized,
            getattr(hand_pose, "shape", "unknown"),
        )
        return hand_pose

    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    # 依据 mode 选择对应的 trans 边界（与数据集统一：dexgraspnet / bodexshadow 均使用 palm_center anchor）
    anchor_val = get_current_dex_anchor_for_denorm()
    anchor_n = (anchor_val or "").strip().lower()
    if mode_normalized in ("dexgraspnet", "bodexshadow"):
        if anchor_n in ("base", "mjcf"):
            raise NotImplementedError(
                f"[hand_helper] dex 模式当前仅支持 palm_center anchor，检测到 trans_anchor='{anchor_n}' 暂未实现"
            )
        trans_bounds_key = "palm_center"
    else:
        if mode_normalized == "dexgraspanything_base":
            trans_bounds_key = "hand_trans"
        else:
            trans_bounds_key = "palm_center"

    # 固定 palm_center anchor：在此处即从 hand_trans 即时派生 palm_center 边界；不在 stats 中维护派生结果
    # 若 hand_trans 缺失将直接报错，避免静默退化或多 anchor 造成混乱

    # Anchor 强约束：dexgraspnet/bodexshadow 仅允许 palm_center
    if mode_normalized in ("dexgraspnet", "bodexshadow") and trans_bounds_key != "palm_center":
        raise ValueError(f"[hand_helper] 仅支持 palm_center anchor (mode={mode_normalized})")

    # 仅从统计文件读取 hand_trans，并即时从 hand_trans 内缩派生 palm_center 边界（不在 stats 中维护 palm_center）
    if mode_normalized in ("dexgraspnet", "bodexshadow"):
        ht = stats.get("hand_trans")
        if ht is None:
            raise ValueError(f"[hand_helper] 缺少 hand_trans 边界 (mode={mode_normalized})，无法派生 palm_center。")
        lo = ht["min"]
        hi = ht["max"]
        if isinstance(lo, np.ndarray):
            lo = torch.from_numpy(lo).float()
        elif not isinstance(lo, torch.Tensor):
            lo = torch.as_tensor(lo, dtype=torch.float32)
        if isinstance(hi, np.ndarray):
            hi = torch.from_numpy(hi).float()
        elif not isinstance(hi, torch.Tensor):
            hi = torch.as_tensor(hi, dtype=torch.float32)
        # margin = float(get_palm_center_margin())
        margin = PALM_CENTER_MARGIN
        diff = hi - lo
        mask = diff > (2.0 * margin)
        lo_out = torch.where(mask, lo + margin, lo)
        hi_out = torch.where(mask, hi - margin, hi)
        trans_bounds = {"min": lo_out, "max": hi_out}
    else:
        trans_bounds = stats.get("hand_trans")

    joint_bounds = stats.get("joint_angles")
    joint_bounds = _prepare_joint_bounds_for_denorm(joint_bounds, hand_pose_config.JOINT_ANGLE_DIM, mode_normalized)

    # Debug: 打印关节边界信息（仅首次）
    if mode_normalized in ("dexgraspnet", "bodexshadow") and not getattr(_denorm_hand_pose_dex, "_debug_joint_bounds_printed", False):
        if joint_bounds is not None:
            min_j = joint_bounds.get("min")
            max_j = joint_bounds.get("max")
            if min_j is not None and max_j is not None:
                min_j_np = min_j.cpu().numpy() if isinstance(min_j, torch.Tensor) else min_j
                max_j_np = max_j.cpu().numpy() if isinstance(max_j, torch.Tensor) else max_j
                logging.getLogger(__name__).info(
                    "[hand_helper][denorm] joint_bounds shape=%s, min[:5]=%s, max[:5]=%s",
                    min_j_np.shape, min_j_np[:5].tolist(), max_j_np[:5].tolist()
                )
                _denorm_hand_pose_dex._debug_joint_bounds_printed = True

    if trans_bounds is None or joint_bounds is None:
        logging.getLogger(__name__).warning(
            "[hand_helper] DexGraspAnything 模式归一化统计不完整，直接透传 hand_pose (mode=%s, trans=%s, joint=%s, shape=%s)",
            mode_normalized,
            "available" if trans_bounds is not None else "missing",
            "available" if joint_bounds is not None else "missing",
            getattr(hand_pose, "shape", "unknown"),
        )
        return hand_pose

    if isinstance(hand_pose, torch.Tensor):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            pose_flat = hand_pose.reshape(-1, orig_shape[-1])
        else:
            pose_flat = hand_pose


        # Debug:        
        if not getattr(_denorm_hand_pose_dex, "_debug_once", False):
            try:
                tdbg = pose_flat.detach()
                finite = torch.isfinite(tdbg)
                tmask = torch.where(finite, tdbg, torch.zeros_like(tdbg))
                in_mn = tmask.min().item() if tmask.numel() > 0 else 0.0
                in_mx = tmask.max().item() if tmask.numel() > 0 else 0.0
                in_me = tmask.float().mean().item() if tmask.numel() > 0 else 0.0
                logging.getLogger(__name__).info(
                    "[hand_helper][denorm] input(norm_pose) shape=%s min=%.4f max=%.4f mean=%.4f",
                    tuple(pose_flat.shape), in_mn, in_mx, in_me,
                )
            except Exception:
                pass

        hand_t, hand_param, hand_r = pose_flat.split(
            (3, hand_pose_config.JOINT_ANGLE_DIM, rot_dim), dim=-1
        )
        hand_t_denorm = _denorm_from_bounds_torch(hand_t, trans_bounds)
        hand_param_denorm = _denorm_from_bounds_torch(hand_param, joint_bounds)
        hand_denorm_flat = torch.cat([hand_t_denorm, hand_param_denorm, hand_r], dim=-1)


        # Debug: 仅首次打印反归一化后范围（torch）
        if not getattr(_denorm_hand_pose_dex, "_debug_once", False):
            try:
                outdbg = hand_denorm_flat.detach()
                finite = torch.isfinite(outdbg)
                omask = torch.where(finite, outdbg, torch.zeros_like(outdbg))
                out_mn = omask.min().item() if omask.numel() > 0 else 0.0
                out_mx = omask.max().item() if omask.numel() > 0 else 0.0
                out_me = omask.float().mean().item() if omask.numel() > 0 else 0.0
                logging.getLogger(__name__).info(
                    "[hand_helper][denorm] output(hand_pose) shape=%s min=%.4f max=%.4f mean=%.4f",
                    tuple(hand_denorm_flat.shape), out_mn, out_mx, out_me,
                )
                _denorm_hand_pose_dex._debug_once = True
            except Exception:
                pass

        if len(orig_shape) == 3:
            return hand_denorm_flat.reshape(orig_shape)
        return hand_denorm_flat

    if isinstance(hand_pose, np.ndarray):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            pose_flat = hand_pose.reshape(-1, orig_shape[-1])
        else:
            pose_flat = hand_pose

        split_qpos = 3 + hand_pose_config.JOINT_ANGLE_DIM
        hand_t = pose_flat[:, :3]
        hand_param = pose_flat[:, 3:split_qpos]
        hand_r = pose_flat[:, split_qpos:split_qpos + rot_dim]
        hand_t_denorm = _denorm_from_bounds_numpy(hand_t, trans_bounds)


        hand_param_denorm = _denorm_from_bounds_numpy(hand_param, joint_bounds)
        hand_denorm_flat = np.concatenate([hand_t_denorm, hand_param_denorm, hand_r], axis=-1)

        if len(orig_shape) == 3:
            return hand_denorm_flat.reshape(orig_shape)
        return hand_denorm_flat

    return hand_pose


def _norm_hand_pose_dex(hand_pose, rot_type, mode_normalized: str):
    stats = _load_dex_stats_if_needed(mode_normalized)
    if not stats:
        logging.getLogger(__name__).warning(
            "[hand_helper] DexGraspAnything 模式缺少归一化统计，直接透传 hand_pose。",
        )
        return hand_pose
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    anchor_val = get_current_dex_anchor_for_denorm()
    anchor_n = (anchor_val or "").strip().lower()
    if mode_normalized in ("dexgraspnet", "bodexshadow"):
        if anchor_n in ("base", "mjcf"):
            raise NotImplementedError(
                f"[hand_helper] dex 模式当前仅支持 palm_center anchor，检测到 trans_anchor='{anchor_n}' 暂未实现"
            )
        # 仅从 hand_trans 读取并即时派生 palm_center
        ht = stats.get("hand_trans")
        if ht is None:
            logging.getLogger(__name__).warning(
                "[hand_helper] 归一化缺少 hand_trans 边界 (mode=%s)，直接透传 hand_pose。",
                mode_normalized,
            )
            return hand_pose
        lo = ht["min"]; hi = ht["max"]
        if isinstance(lo, np.ndarray):
            lo = torch.from_numpy(lo).float()
        elif not isinstance(lo, torch.Tensor):
            lo = torch.as_tensor(lo, dtype=torch.float32)
        if isinstance(hi, np.ndarray):
            hi = torch.from_numpy(hi).float()
        elif not isinstance(hi, torch.Tensor):
            hi = torch.as_tensor(hi, dtype=torch.float32)
        # margin = float(get_palm_center_margin())
        margin = PALM_CENTER_MARGIN
        diff = hi - lo
        mask = diff > (2.0 * margin)
        lo_out = torch.where(mask, lo + margin, lo)
        hi_out = torch.where(mask, hi - margin, hi)
        trans_bounds = {"min": lo_out, "max": hi_out}
    else:
        trans_bounds = stats.get("hand_trans") if mode_normalized == "dexgraspanything_base" else stats.get("palm_center")
    joint_bounds = stats.get("joint_angles")
    # 对关节边界进行维度对齐（22->24，或 24->24 并将禁用维置 0）
    joint_bounds = _prepare_joint_bounds_for_denorm(joint_bounds, hand_pose_config.JOINT_ANGLE_DIM, mode_normalized)
    if trans_bounds is None or joint_bounds is None:
        logging.getLogger(__name__).warning(
            "[hand_helper] DexGraspAnything 模式归一化统计不完整（trans=%s, joint=%s），直接透传 hand_pose。",
            "available" if trans_bounds is not None else "missing",
            "available" if joint_bounds is not None else "missing",
        )
        return hand_pose

    def _norm_from_bounds_torch(values: torch.Tensor, bounds: dict) -> torch.Tensor:
        min_val = bounds["min"].to(values.device, values.dtype)
        max_val = bounds["max"].to(values.device, values.dtype)
        range_val = torch.clamp(max_val - min_val, min=1e-8)
        return 2.0 * (values - min_val) / range_val - 1.0

    def _norm_from_bounds_numpy(values: np.ndarray, bounds: dict) -> np.ndarray:
        min_val = bounds["min"].cpu().numpy()
        max_val = bounds["max"].cpu().numpy()
        range_val = np.clip(max_val - min_val, 1e-8, None)
        return 2.0 * (values - min_val) / range_val - 1.0

    if isinstance(hand_pose, torch.Tensor):
        orig_shape = hand_pose.shape
        pose_flat = hand_pose.reshape(-1, orig_shape[-1]) if hand_pose.ndim == 3 else hand_pose
        hand_t, hand_param, hand_r = pose_flat.split((3, hand_pose_config.JOINT_ANGLE_DIM, rot_dim), dim=-1)
        hand_t_norm = _norm_from_bounds_torch(hand_t, trans_bounds)
        hand_param_norm = _norm_from_bounds_torch(hand_param, joint_bounds)
        # 在 Dex 模式下，将被禁用的 qpos 维度（如前两维）强制置 0，保持与训练假设一致
        disabled = get_disabled_qpos_indices(mode_normalized)
        if disabled:
            idx = list(disabled)
            hand_param_norm[..., idx] = 0.0
        hand_norm_flat = torch.cat([hand_t_norm, hand_param_norm, hand_r], dim=-1)
        if hand_pose.ndim == 3:
            return hand_norm_flat.reshape(orig_shape[0], orig_shape[1], -1)
        return hand_norm_flat

    if isinstance(hand_pose, np.ndarray):
        orig_shape = hand_pose.shape
        pose_flat = hand_pose.reshape(-1, orig_shape[-1]) if hand_pose.ndim == 3 else hand_pose
        split_qpos = 3 + hand_pose_config.JOINT_ANGLE_DIM
        hand_t = pose_flat[:, :3]
        hand_param = pose_flat[:, 3:split_qpos]
        hand_r = pose_flat[:, split_qpos:split_qpos + rot_dim]
        hand_t_norm = _norm_from_bounds_numpy(hand_t, trans_bounds)
        hand_param_norm = _norm_from_bounds_numpy(hand_param, joint_bounds)
        # 同样对 numpy 分支置 0 被禁用的维度
        disabled = get_disabled_qpos_indices(mode_normalized)
        if disabled:
            idx = list(disabled)
            hand_param_norm[..., idx] = 0.0
        hand_norm_flat = np.concatenate([hand_t_norm, hand_param_norm, hand_r], axis=-1)
        if hand_pose.ndim == 3:
            return hand_norm_flat.reshape(orig_shape[0], orig_shape[1], -1)
        return hand_norm_flat

    return hand_pose

def denorm_hand_pose_robust(hand_pose, rot_type, mode):
    """
    Denormalize hand pose parameters. Supports both single and multi-grasp formats:
    - Single grasp: [B, pose_dim]
    - Multi grasp: [B, num_grasps, pose_dim]
    """
    # logging.info("denorm_hand_pose_robust(): mode=%s", mode)
    rot_dim = ROT_DIM_DICT[rot_type.lower()]

    if _is_debug_bypass_mode(mode):
        logger = logging.getLogger(__name__)
        rank_zero_info(f"[hand_helper][debug] mode={mode} 走调试透传路径，跳过 denorm_hand_pose_robust")
        target_dim = 33 if rot_type.lower() == "r6d" else 31

        if isinstance(hand_pose, torch.Tensor):
            current_dim = hand_pose.shape[-1] if hand_pose.ndim >= 1 else target_dim
            if current_dim == target_dim:
                return hand_pose
            base_shape = hand_pose.shape[:-1]
            placeholder = torch.zeros(
                *base_shape, target_dim, device=hand_pose.device, dtype=hand_pose.dtype
            )
            return placeholder
        if isinstance(hand_pose, np.ndarray):
            current_dim = hand_pose.shape[-1] if hand_pose.ndim >= 1 else target_dim
            if current_dim == target_dim:
                return hand_pose
            base_shape = hand_pose.shape[:-1]
            placeholder = np.zeros((*base_shape, target_dim), dtype=hand_pose.dtype)
            return placeholder
        return hand_pose

    if mode == "objectcentric_single_object":
        if not isinstance(hand_pose, torch.Tensor):
            raise TypeError("ObjectCentric 模式仅支持 torch.Tensor 形式的 hand_pose。")
        return denorm_hand_pose_objectcentric(hand_pose, rot_type)
    if _is_dex_mode(mode):
        return _denorm_hand_pose_dex(hand_pose, rot_type, _normalize_mode(mode))

    if isinstance(hand_pose, torch.Tensor):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose

        hand_t, hand_param, hand_r = hand_pose_reshaped.split(
            (3, hand_pose_config.JOINT_ANGLE_DIM, rot_dim), dim=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t_denorm = denormalize_trans_torch(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r_denorm = denormalize_rot_torch(hand_r, rot_type, mode)
        hand_param_denorm = denormalize_param_torch(hand_param, mode)
        hand_denorm_reshaped = torch.cat(
            [hand_t_denorm, hand_param_denorm, hand_r_denorm], dim=-1
        )

        if len(orig_shape) == 3:
            hand = hand_denorm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_denorm_reshaped

    elif isinstance(hand_pose, np.ndarray):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose

        hand_t, hand_param, hand_r = np.split(
            hand_pose_reshaped, [3, 3 + hand_pose_config.JOINT_ANGLE_DIM], axis=-1
        )

        map_distinction = "map" if rot_type == "map" else "not_map"
        hand_t_denorm = denormalize_trans_numpy(
            hand_t, mode, rot_type_for_map_distinction=map_distinction
        )
        hand_r_denorm = denormalize_rot_numpy(hand_r, rot_type, mode)
        hand_param_denorm = denormalize_param_numpy(hand_param, mode)
        hand_denorm_reshaped = np.concatenate(
            [hand_t_denorm, hand_param_denorm, hand_r_denorm], axis=-1
        )

        if len(orig_shape) == 3:
            hand = hand_denorm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_denorm_reshaped
    else:
        raise NotImplementedError

    return hand


# Batch processing functions
def _process_batch_pose_logic(se3_batch, hand_model_pose_batch, rot_type, mode):
    """
    Processes a batch of SE(3) poses and hand model poses, unifying single and multi-grasp formats.
    - Single grasp input: hand_model_pose_batch [B, 23], se3_batch [B, 4, 4]
    - Multi grasp input: hand_model_pose_batch [B, num_grasps, 23], se3_batch [B, num_grasps, 4, 4]
    Returns consistently shaped 3D tensors: [B, num_grasps, pose_dim]
    """
    _debug_log(
        "[_process_batch_pose_logic] se3_batch=%s hand_model_pose_batch=%s rot_type=%s mode=%s joint_dim=%d",
        _shape_repr(se3_batch),
        _shape_repr(hand_model_pose_batch),
        rot_type,
        mode,
        hand_pose_config.JOINT_ANGLE_DIM,
    )

    # Unify input shapes to [B, num_grasps, ...]
    if hand_model_pose_batch.dim() == 2:
        hand_model_pose_batch = hand_model_pose_batch.unsqueeze(1)
        se3_batch = se3_batch.unsqueeze(1)
        _debug_log(
            "[_process_batch_pose_logic] expanded to multi-grasp se3=%s hand_pose=%s",
            _shape_repr(se3_batch),
            _shape_repr(hand_model_pose_batch),
        )

    B, num_grasps = hand_model_pose_batch.shape[:2]
    _debug_log(
        "[_process_batch_pose_logic] batch_size=%d num_grasps=%d",
        B,
        num_grasps,
    )

    # Reshape for batch processing
    se3_flat = se3_batch.view(B * num_grasps, 4, 4)
    pose_flat = hand_model_pose_batch.view(B * num_grasps, -1)
    _debug_log(
        "[_process_batch_pose_logic] se3_flat=%s pose_flat=%s",
        _shape_repr(se3_flat),
        _shape_repr(pose_flat),
    )

    # Process using existing logic
    matrix_batch = se3_flat[:, :3, :3]

    if rot_type == "quat":
        rot_representation = transforms.matrix_to_quaternion(matrix_batch)
    elif rot_type == "r6d":
        rot_representation = transforms.matrix_to_rotation_6d(matrix_batch)
    elif rot_type == "axis":
        rot_representation = transforms.matrix_to_axis_angle(matrix_batch)
    elif rot_type == "euler":
        rot_representation = transforms.matrix_to_euler_angles(
            matrix_batch, convention="XYZ"
        )
    elif rot_type == "map":
        log_map_full = se3_log_map(se3_flat)
        rot_representation = log_map_full[:, 3:]
    else:
        raise ValueError(f"Unsupported rotation type: {rot_type}")

    input_translation_part = pose_flat[:, :3]
    input_joint_angle_part = pose_flat[:, 3 : 3 + hand_pose_config.JOINT_ANGLE_DIM]
    _debug_log(
        "[_process_batch_pose_logic] translation_part=%s joint_part=%s rot_repr=%s",
        _shape_repr(input_translation_part),
        _shape_repr(input_joint_angle_part),
        _shape_repr(rot_representation),
    )

    if rot_type == "map":
        processed_translation_part = log_map_full[:, :3]
    else:
        processed_translation_part = input_translation_part

    processed_hand_pose_flat = torch.cat(
        [processed_translation_part, input_joint_angle_part, rot_representation], dim=1
    )
    _debug_log(
        "[_process_batch_pose_logic] processed_hand_pose_flat=%s",
        _shape_repr(processed_hand_pose_flat),
    )

    # Normalize using robust function
    norm_pose_flat = norm_hand_pose_robust(
        processed_hand_pose_flat, rot_type=rot_type, mode=mode
    )
    _debug_log(
        "[_process_batch_pose_logic] norm_pose_flat=%s",
        _shape_repr(norm_pose_flat),
    )

    # Reshape back to unified 3D format
    pose_dim = processed_hand_pose_flat.shape[-1]
    processed_hand_pose = processed_hand_pose_flat.view(B, num_grasps, pose_dim)
    norm_pose = norm_pose_flat.view(B, num_grasps, pose_dim)
    _debug_log(
        "[_process_batch_pose_logic] processed_hand_pose=%s norm_pose=%s",
        _shape_repr(processed_hand_pose),
        _shape_repr(norm_pose),
    )

    return norm_pose, processed_hand_pose


def process_hand_pose_train(data, rot_type, mode):
    se3 = data["se3"]
    hand_model_pose_input = data["hand_model_pose"]

    norm_pose, processed_hand_pose = _process_batch_pose_logic(
        se3, hand_model_pose_input, rot_type, mode
    )

    data["norm_pose"] = norm_pose
    data["hand_model_pose"] = processed_hand_pose

    _assign_norm_qpos_and_se3(data, norm_pose, se3)

    return data


def process_hand_pose_single(data, rot_type, mode):
    """Process single hand pose data
    Args:
        data (dict): Dictionary containing:
            - se3 (torch.Tensor): Shape (4, 4) transformation matrix
            - hand_model_pose (torch.Tensor): Shape (23,) hand pose parameters (T, 16J, 4Q)
        rot_type (str): Rotation representation type
        mode (str): Normalization statistics mode
    Returns:
        dict: Processed data with normalized pose
    """
    se3_single = data["se3"].unsqueeze(0)
    hand_model_pose_single = data["hand_model_pose"].unsqueeze(0)

    norm_pose_batched, processed_hand_pose_batched = _process_batch_pose_logic(
        se3_single, hand_model_pose_single, rot_type, mode
    )

    norm_pose_single = norm_pose_batched.squeeze(0)
    processed_hand_pose_single = processed_hand_pose_batched.squeeze(0)
    data["norm_pose"] = norm_pose_single
    data["hand_model_pose"] = processed_hand_pose_single

    _assign_norm_qpos_and_se3(data, norm_pose_single, data["se3"])

    return data


def process_hand_pose_test(data, rot_type, mode):
    """
    Process data returned from ForMatchSceneLeapDataset, performing normalization and rotation representation transformation.
    Supports both list format and tensor format for multi-grasp data.

    Args:
        data (dict): Dictionary containing:
            - hand_model_pose: [B, num_grasps, 23] tensor or list of tensors
            - se3: [B, num_grasps, 4, 4] tensor or list of tensors
        rot_type (str): Rotation representation type
        mode (str): Processing mode

    Returns:
        dict: Processed data dictionary containing:
            - hand_model_pose: Processed hand pose
            - norm_pose: Normalized pose
    """
    # logging.info("process_hand_pose_test(): mode=%s", mode)
    if _is_debug_bypass_mode(mode):
        rank_zero_info(f"[hand_helper][debug] mode={mode} 走调试透传路径（test），保持 hand_model_pose 原样")
        if isinstance(data, dict) and 'hand_model_pose' in data and 'norm_pose' not in data:
            data['norm_pose'] = data['hand_model_pose']
        return data
    if _is_dex_mode(mode):
        return _passthrough_dex_mode(data, rot_type)

    if mode == "objectcentric_single_object":
        return process_hand_pose_test_objectcentric(data, rot_type)
    # 非 ObjectCentric 模式强制报错（待实现）
    raise NotImplementedError(
        "process_hand_pose_test: 非 objectcentric_single_object 模式暂未实现，请先在数据侧/归一化侧补齐实现后再使用。"
    )

    if isinstance(data.get("hand_model_pose"), list):
        # Handle list format (legacy support)
        processed_data = []
        for i, (poses, se3s) in enumerate(zip(data["hand_model_pose"], data["se3"])):
            # poses: [num_grasps, 23], se3s: [num_grasps, 4, 4]
            item_data = {"hand_model_pose": poses, "se3": se3s}
            processed_item = process_hand_pose(item_data, rot_type, mode)
            processed_data.append(processed_item)

        # Reorganize data
        data["norm_pose"] = [item["norm_pose"] for item in processed_data]
        data["hand_model_pose"] = [item["hand_model_pose"] for item in processed_data]

    else:
        # Standard tensor format: [B, num_grasps, 23] and [B, num_grasps, 4, 4]
        if "se3" not in data or "hand_model_pose" not in data:
            return data

        # Calculate valid mask for non-zero poses
        valid_mask = data["hand_model_pose"].abs().sum(dim=-1) > 0

        # Process using updated _process_batch_pose_logic that supports multi-grasp
        norm_pose_processed, processed_hand_pose_full = _process_batch_pose_logic(
            data["se3"], data["hand_model_pose"], rot_type, mode
        )

        # Apply valid mask to filter out zero poses
        valid_mask = valid_mask.unsqueeze(-1)
        processed_hand_pose_final = processed_hand_pose_full * valid_mask
        norm_pose_final = norm_pose_processed * valid_mask

        data["hand_model_pose"] = processed_hand_pose_final
        data["norm_pose"] = norm_pose_final

        _assign_norm_qpos_and_se3(data, norm_pose_final, data["se3"])

    return data


def _process_batch_pose(se3, hand_model_pose, rot_type, mode):
    """
    Helper for process_hand_pose when input is a standard batch.
    hand_model_pose is the new 23-dim pose.
    """
    return _process_batch_pose_logic(se3, hand_model_pose, rot_type, mode)


def process_hand_pose(data, rot_type, mode):
    """
    Process a batch of data from DataLoader.
    Supports both single and multi-grasp formats:
    - Single grasp: se3 [B, 4, 4], hand_model_pose [B, 23]
    - Multi grasp: se3 [B, num_grasps, 4, 4], hand_model_pose [B, num_grasps, 23]
    - List format: hand_model_pose as list of tensors

    Automatically selects processing method based on input data type.

    If mode is 'objectcentric_single_object', uses unified normalization strategy.
    """
    # logging.info("process_hand_pose(): mode=%s", mode)
    if _is_debug_bypass_mode(mode):
        logger = logging.getLogger(__name__)
        rank_zero_info(f"[hand_helper][debug] mode={mode} 走调试透传路径，保持 hand_model_pose 原样")
        if not isinstance(data, dict) or 'hand_model_pose' not in data:
            return data

        hand_pose = data['hand_model_pose']
        if 'norm_pose' not in data:
            data['norm_pose'] = hand_pose

        if isinstance(hand_pose, torch.Tensor):
            data.setdefault(
                'qpos',
                hand_pose[..., hand_pose_config.QPOS_SLICE],
            )
            if 'se3' not in data:
                B, G = hand_pose.shape[:2]
                identity = torch.eye(4, device=hand_pose.device, dtype=hand_pose.dtype)
                data['se3'] = identity.view(1, 1, 4, 4).expand(B, G, 4, 4).clone()
                logger.warning(
                    "[hand_helper][debug] mode=%s 缺少 se3，使用单位矩阵占位。",
                    mode,
                )
        else:
            if 'qpos' not in data:
                logger.warning(
                    "[hand_helper][debug] mode=%s hand_model_pose 非 Tensor，无法自动解析 qpos。",
                    mode,
                )
            if 'se3' not in data:
                logger.warning(
                    "[hand_helper][debug] mode=%s 无 se3 信息，后续路径规划可能失败。",
                    mode,
                )
        return data
    if _is_dex_mode(mode):
        return _passthrough_dex_mode(data, rot_type)

    # Use ObjectCentric unified normalization for objectcentric_single_object mode
    if mode == 'objectcentric_single_object':
        return process_hand_pose_objectcentric(data, rot_type)
    # 非 ObjectCentric 模式强制报错（待实现）
    raise NotImplementedError(
        "process_hand_pose: 非 objectcentric_single_object 模式暂未实现，请先在数据侧/归一化侧补齐实现后再使用。"
    )

    if isinstance(data, dict) and not isinstance(data.get("hand_model_pose"), list):
        if "se3" not in data or "hand_model_pose" not in data:
            logging.warning(
                "Missing required keys 'se3' or 'hand_model_pose' in input data. Skipping pose processing."
            )
            return data

        se3 = data["se3"]
        hand_model_pose_input = data["hand_model_pose"]

        norm_pose, processed_hand_pose = _process_batch_pose(
            se3, hand_model_pose_input, rot_type, mode
        )

        data["norm_pose"] = norm_pose
        data["hand_model_pose"] = processed_hand_pose

        _assign_norm_qpos_and_se3(data, norm_pose, se3)

    elif isinstance(data, dict) and isinstance(data.get("hand_model_pose"), list):
        num_items = len(data["hand_model_pose"])
        if num_items == 0:
            data["norm_pose"] = []
            data["hand_model_pose"] = []
            return data

        norm_poses_list = [None] * num_items
        processed_hand_poses_list = [None] * num_items

        for i in range(num_items):
            se3_i = data["se3"][i]
            hand_model_pose_i = data["hand_model_pose"][i]

            norm_pose_i_batched, processed_hand_pose_i_batched = _process_batch_pose(
                se3_i, hand_model_pose_i, rot_type, mode
            )
            norm_poses_list[i] = norm_pose_i_batched
            processed_hand_poses_list[i] = processed_hand_pose_i_batched

        data["norm_pose"] = norm_poses_list
        data["hand_model_pose"] = processed_hand_poses_list

        _assign_norm_qpos_and_se3(data, norm_poses_list, data["se3"])

    else:
        raise ValueError(
            f"Unsupported data type for process_hand_pose: {type(data)}. Expected dict or dict of lists."
        )

    return data


def process_se3(data, rot_type, mode):
    matrix = data["se3"][:, :3, :3]

    if rot_type == "quat":
        rot_representation = transforms.matrix_to_quaternion(matrix)
    elif rot_type == "r6d":
        rot_representation = transforms.matrix_to_rotation_6d(matrix)
    elif rot_type == "axis":
        rot_representation = transforms.matrix_to_axis_angle(matrix)
    elif rot_type == "euler":
        rot_representation = transforms.matrix_to_euler_angles(matrix, convention="XYZ")
    elif rot_type == "map":
        se3_full_log_map = se3_log_map(data["se3"])
    else:
        raise ValueError(f"Unsupported rotation type: {rot_type}")

    if rot_type == "map":
        se3_rep_for_norm = se3_full_log_map
    else:
        trans = data["se3"][:, :3, 3]
        se3_rep_for_norm = torch.cat([trans, rot_representation], dim=-1)

    trans_to_norm = se3_rep_for_norm[..., :3]
    rot_to_norm = se3_rep_for_norm[..., 3:]

    map_distinction_for_trans = "map" if rot_type == "map" else "not_map"
    trans_norm = normalize_trans_torch(
        trans_to_norm, mode, rot_type_for_map_distinction=map_distinction_for_trans
    )

    rot_norm = normalize_rot_torch(rot_to_norm, rot_type, mode)

    se3_rep_norm = torch.cat([trans_norm, rot_norm], dim=-1)
    data["se3_rep_norm"] = se3_rep_norm

    return data


def decompose_hand_pose(hand_pose, rot_type="quat"):
    """
    Decompose hand pose into global translation, global rotation, and joint angles.
    Input hand_pose is the "processed_hand_pose" format: T (3) | J (hand_pose_config.JOINT_ANGLE_DIM) | R (rot_dim)
    """
    global_translation = hand_pose[:, :3]

    qpos = hand_pose[:, 3 : 3 + hand_pose_config.JOINT_ANGLE_DIM]

    rot_part_start_idx = 3 + hand_pose_config.JOINT_ANGLE_DIM

    if rot_type == "quat":
        quat = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 4]
        global_rotation = transforms.quaternion_to_matrix(quat)
    elif rot_type == "r6d":
        r6d = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 6]
        global_rotation = transforms.rotation_6d_to_matrix(r6d)
    elif rot_type == "axis":
        axis_angle = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 3]
        global_rotation = transforms.axis_angle_to_matrix(axis_angle)
    elif rot_type == "euler":
        euler = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 3]
        global_rotation = transforms.euler_angles_to_matrix(euler, convention="XYZ")
    elif rot_type == "map":
        axis_angle_from_map = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 3]
        global_rotation = transforms.axis_angle_to_matrix(axis_angle_from_map)
    else:
        raise ValueError(f"Unsupported rotation type: {rot_type}")

    return global_translation, global_rotation, qpos
