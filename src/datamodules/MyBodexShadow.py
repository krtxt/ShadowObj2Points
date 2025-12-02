"""Bodex Shadow Hand dataset with flattened data structure.

This dataset loads Bodex shadow hand grasp data from hierarchical directory structure.
Data format: {object_name}/floating/scale{xxx}.npy containing grasp_qpos (N, 29).
Format: [trans(3), quat(4), joints(22)]

Features:
- Flattened data structure: each item returns one hand_model_pose and norm_pose
- Support for multiple rotation representations (quat, r6d, euler, axis)
- Optional stats-based normalization of translations/joints
- Multiple translation anchor options (mjcf, base, palm_center)
- Optional object-level train/test splits via metadata json
"""
import json
import logging
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import pickle
import trimesh
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
)
from torch.utils.data import Dataset

from .utils.collate_utils import collate_batch_data
from .utils.data_processing_utils import load_bodex_hand_pose_data

logger = logging.getLogger(__name__)
MJCF_ANCHOR_TRANSLATION = torch.tensor([0.0, -0.01, 0.213], dtype=torch.float32)
PALM_CENTER_ANCHOR_TRANSLATION = torch.tensor([0.008, -0.013, 0.283], dtype=torch.float32)
PALM_CENTER_MARGIN = 0.25


class MyBodexShadow(Dataset):
    """PyTorch Dataset for Bodex Shadow Hand data with flattened structure.

    Args:
        asset_dir: Root directory (e.g., data/bodex/bodex_shadow).
        succ_grasp_subdir: Subdirectory name for grasp data (default: 'succ_collect').
        phase: 'train' | 'test' | 'all' (reserved, currently loads all data).
        rot_type: Rotation representation: 'quat' | 'r6d' | 'euler' | 'axis'.
        trans_anchor: 'mjcf' (default, data native) | 'base' (URDF) | 'palm_center'.
        use_stats_normalization: If True, apply stats-based normalization.
        normalization_stats_path: Path or stem for stats.
        split_file_name: Optional json file listing train/test/all object splits.
        debug_mode: When True, read original succ_grasp npy files; otherwise use packed pt file.
    """
    TRANS_DIM = 3
    JOINT_DIM = 24  # Align with hand_pose_config (two wrist DOFs + 22 joints)

    def __init__(self, asset_dir='data/bodex/bodex_shadow', succ_grasp_subdir='succ_collect',
                 phase='all', rot_type='quat',
                 trans_anchor: str = 'mjcf',
                 use_stats_normalization: bool = False,
                 normalization_stats_path: Optional[str] = None,
                 split_file_name: Optional[str] = None,
                 debug_mode: bool = False,
                 max_points: int = 4096,
                 point_clouds_file_name: Optional[str] = None,
                 use_scene_normals: bool = False):
        super().__init__()

        self.name = 'BodexShadow'
        self.asset_dir = asset_dir
        self.succ_grasp_subdir = succ_grasp_subdir
        self.phase = phase
        self.rot_type = rot_type
        self.trans_anchor = str(trans_anchor)
        if self.trans_anchor == 'anchor':
            self.trans_anchor = 'palm_center'
        valid_anchors = {'mjcf', 'base', 'palm_center'}
        if self.trans_anchor not in valid_anchors:
            raise ValueError(f"Unsupported trans_anchor '{self.trans_anchor}'. Expected one of {sorted(valid_anchors)}.")
        self._object_splits: Optional[Dict[str, set]] = None

        self.use_stats_normalization = use_stats_normalization
        self.normalization_stats_path = normalization_stats_path or os.path.join('assets', 'mybodexshadow_normalization_stats')
        self._stats_bounds: Dict[str, Dict[str, np.ndarray]] = {}

        self.debug_mode = debug_mode

        self.max_points = int(max_points) if int(max_points) > 0 else 4096
        self.point_clouds_file_name = point_clouds_file_name
        self.use_scene_normals = bool(use_scene_normals)

        # Split file defaults to dataset root
        if split_file_name is None:
            self.split_file_path = os.path.join(self.asset_dir, 'bodex_split.json')
        elif os.path.isabs(split_file_name):
            self.split_file_path = split_file_name
        else:
            self.split_file_path = os.path.join(self.asset_dir, split_file_name)

        self._init_rotation_config()
        self._init_paths()
        self._validate_init_params()

        self.scene_data: Dict[str, Dict[str, Any]] = {}
        self.scene_ids: List[str] = []
        self.scene_pcds: Dict[str, np.ndarray] = {}
        self._scene_pc_cache: Dict[str, torch.Tensor] = {}

        self._load_data()
        self._apply_object_split_filter()
        self._load_point_clouds()
        self._filter_missing_pointclouds()
        if self.use_stats_normalization:
            self._try_load_normalization_stats()

        # Build flattened data index - each item is a single grasp
        self.data = self._build_flattened_data_index()
        logger.info("MyBodexShadow: Using flattened data with %d items.", len(self.data))

        logger.info("MyBodexShadow initialized: %d scenes loaded", len(self.scene_ids))

    def _init_rotation_config(self) -> None:
        """Initialize rotation representation configuration."""
        converter_map = {
            'quat': (4, matrix_to_quaternion, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)),
            'r6d': (6, matrix_to_rotation_6d, torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)),
            'euler': (3, partial(matrix_to_euler_angles, convention="XYZ"), torch.zeros(3, dtype=torch.float32)),
            'axis': (3, matrix_to_axis_angle, torch.zeros(3, dtype=torch.float32)),
        }
        if self.rot_type not in converter_map:
            raise ValueError(f"rot_type must be one of {list(converter_map.keys())}, got '{self.rot_type}'")

        rot_dim, converter, identity = converter_map[self.rot_type]
        self.ROT_DIM = rot_dim
        self.POSE_DIM = self.TRANS_DIM + self.JOINT_DIM + self.ROT_DIM
        self._rotation_converter = converter
        self._rot_identity_base = identity

    def _init_paths(self) -> None:
        """Initialize file paths for dataset."""
        self.succ_grasp_dir = os.path.join(self.asset_dir, self.succ_grasp_subdir)
        self.pt_dataset_path = os.path.join(self.asset_dir, 'bodex_shadowhand.pt')

        dexgraspnet_root = './data/DexGraspNet'
        if self.point_clouds_file_name:
            self.point_cloud_path = os.path.join(dexgraspnet_root, self.point_clouds_file_name)
        else:
            self.point_cloud_path = os.path.join(dexgraspnet_root, 'object_pcds_nors.pkl')
        self.mesh_base_path = os.path.join(dexgraspnet_root, 'meshdata')

        self._load_object_splits()

    def _validate_init_params(self) -> None:
        """Validate initialization parameters."""
        pass

    def _load_object_splits(self) -> None:
        split_path = self.split_file_path
        if not split_path:
            return
        if not os.path.exists(split_path):
            logger.warning("MyBodexShadow: split file not found at %s", split_path)
            return
        try:
            with open(split_path, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("MyBodexShadow: failed to load split file %s: %s", split_path, exc)
            return

        splits: Dict[str, set] = {}
        for key_alias, phase_name in (
            ('_train_split', 'train'),
            ('_test_split', 'test'),
            ('_val_split', 'val'),
            ('_all_split', 'all'),
        ):
            if key_alias in data:
                splits[phase_name] = set(data[key_alias])
        if 'all' not in splits:
            combined = set()
            for phase, obj_set in splits.items():
                if phase != 'all':
                    combined |= obj_set
            splits['all'] = combined
        self._object_splits = splits
        logger.info("MyBodexShadow: loaded object splits from %s (keys=%s)", split_path, list(splits.keys()))

    def _get_allowed_object_names(self) -> Optional[set]:
        if not self._object_splits:
            return None
        phase_key = self.phase.lower()
        if phase_key in self._object_splits:
            return self._object_splits[phase_key]
        if phase_key == 'all':
            return self._object_splits.get('all')
        logger.warning("MyBodexShadow: phase '%s' not found in split file, using 'all'", self.phase)
        return self._object_splits.get('all')

    def _apply_object_split_filter(self) -> None:
        if self.phase.lower() == 'all':
            self.scene_ids = list(self.scene_data.keys())
            return
        allowed = self._get_allowed_object_names()
        if allowed is None or not allowed:
            return
        removed = 0
        for scene_id in list(self.scene_data.keys()):
            obj = self.scene_data[scene_id]['object_name']
            if obj not in allowed:
                del self.scene_data[scene_id]
                removed += 1
        if removed:
            logger.info("MyBodexShadow: filtered %d scenes using phase '%s' split", removed, self.phase)
        self.scene_ids = list(self.scene_data.keys())

    def _filter_missing_pointclouds(self) -> None:
        if not self.scene_pcds:
            logger.warning("MyBodexShadow: no point cloud dictionary loaded; all scenes will be dropped.")
            self.scene_data.clear()
            self.scene_ids = []
            return

        removed = 0
        missing_objects = set()
        for scene_id in list(self.scene_data.keys()):
            obj_name = self.scene_data[scene_id]['object_name']
            pcd_obj_name = obj_name.replace('_', '-')
            pc = self.scene_pcds.get(pcd_obj_name)
            if pc is None:
                pc = self.scene_pcds.get(obj_name)
            if pc is None or pc.shape[0] == 0 or pc.shape[1] < 3:
                del self.scene_data[scene_id]
                removed += 1
                missing_objects.add(obj_name)
        if removed > 0:
            sample = ", ".join(sorted(list(missing_objects))[:5])
            logger.warning(
                "MyBodexShadow: removed %d scenes due to missing point clouds (objects: %s%s)",
                removed,
                sample,
                "..." if len(missing_objects) > 5 else "",
            )
        self.scene_ids = list(self.scene_data.keys())

    def _try_load_normalization_stats(self) -> None:
        base_path = self.normalization_stats_path
        if base_path.endswith('.json'):
            json_path = base_path
            pt_path = base_path[:-5] + '.pt'
        elif base_path.endswith('.pt'):
            pt_path = base_path
            json_path = base_path[:-3] + '.json'
        else:
            json_path = base_path + '.json'
            pt_path = base_path + '.pt'

        stats_obj = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    stats_obj = json.load(f)
                logger.info("MyBodexShadow: loaded normalization stats JSON from %s", json_path)
            except Exception as exc:
                logger.warning("MyBodexShadow: failed to read JSON stats %s: %s", json_path, exc)

        if stats_obj is None and os.path.exists(pt_path):
            try:
                stats_obj = torch.load(pt_path, map_location='cpu')
                logger.info("MyBodexShadow: loaded normalization stats PT from %s", pt_path)
            except Exception as exc:
                logger.warning("MyBodexShadow: failed to read PT stats %s: %s", pt_path, exc)

        if stats_obj is None:
            logger.warning("MyBodexShadow: normalization stats not found (base=%s); disabling normalization.", self.normalization_stats_path)
            self.use_stats_normalization = False
            return

        def pick_bounds(block: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
            mn = block.get('min_with_margin', block.get('min'))
            mx = block.get('max_with_margin', block.get('max'))
            return np.asarray(mn, dtype=np.float32), np.asarray(mx, dtype=np.float32)

        bounds: Dict[str, Dict[str, np.ndarray]] = {}
        if 'hand_trans' in stats_obj:
            lo, hi = pick_bounds(stats_obj['hand_trans'])
            bounds['hand_trans'] = {'min': lo, 'max': hi}
            diff = hi - lo
            mjcf_margin = 0.2
            palm_margin = PALM_CENTER_MARGIN

            shrink_mask_mjcf = diff > (2 * mjcf_margin)
            shrunk_mjcf_lo = np.where(shrink_mask_mjcf, lo + mjcf_margin, lo)
            shrunk_mjcf_hi = np.where(shrink_mask_mjcf, hi - mjcf_margin, hi)
            bounds['mjcf_trans'] = {'min': shrunk_mjcf_lo, 'max': shrunk_mjcf_hi}

            shrink_mask_palm = diff > (2 * palm_margin)
            shrunk_palm_lo = np.where(shrink_mask_palm, lo + palm_margin, lo)
            shrunk_palm_hi = np.where(shrink_mask_palm, hi - palm_margin, hi)
            bounds['palm_center'] = {'min': shrunk_palm_lo, 'max': shrunk_palm_hi}
        else:
            for key in ('mjcf_trans', 'palm_center'):
                if key in stats_obj:
                    lo, hi = pick_bounds(stats_obj[key])
                    bounds[key] = {'min': lo, 'max': hi}
        if 'joint_angles' in stats_obj:
            lo, hi = pick_bounds(stats_obj['joint_angles'])
            lo, hi = self._align_joint_bounds(lo, hi)
            bounds['joint_angles'] = {'min': lo, 'max': hi}

        self._stats_bounds = bounds
        logger.info("MyBodexShadow: loaded normalization bounds for keys=%s", list(bounds.keys()))

    def _normalize_pose(self, hand_poses: torch.Tensor) -> np.ndarray:
        """Normalize hand pose to [-1, 1] range."""
        hand_poses_np = hand_poses.cpu().numpy() if isinstance(hand_poses, torch.Tensor) else hand_poses
        norm_trans = hand_poses_np[:, :3].astype(np.float32)
        norm_qpos = hand_poses_np[:, 3:3 + self.JOINT_DIM].astype(np.float32)
        rot_tail = hand_poses_np[:, 3 + self.JOINT_DIM:].astype(np.float32)

        if self.use_stats_normalization and self._stats_bounds:
            # Get translation bounds based on anchor
            t_bounds = self._stats_bounds.get('mjcf_trans' if self.trans_anchor == 'mjcf' else
                                             'palm_center' if self.trans_anchor == 'palm_center' else
                                             'hand_trans')
            if t_bounds:
                lo, hi = t_bounds['min'].reshape(1, 3), t_bounds['max'].reshape(1, 3)
                norm_trans = self._normalize_by_bounds(hand_poses_np[:, :3], lo, hi)

            # Get joint bounds
            j_bounds = self._stats_bounds.get('joint_angles')
            if j_bounds:
                lo_j, hi_j = j_bounds['min'].reshape(1, self.JOINT_DIM), j_bounds['max'].reshape(1, self.JOINT_DIM)
                norm_qpos = self._normalize_by_bounds(hand_poses_np[:, 3:3 + self.JOINT_DIM], lo_j, hi_j)
                norm_qpos[..., :2] = 0.0  # Force disabled DOF to zero

        norm_trans = np.clip(norm_trans, -1.0, 1.0)
        norm_qpos = np.clip(norm_qpos, -1.0, 1.0)
        return np.concatenate([norm_trans, norm_qpos, rot_tail], axis=1).astype(np.float32)

    @staticmethod
    def _normalize_by_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize values to [-1, 1] with per-dimension bounds."""
        y = (x - lo) / np.maximum(hi - lo, eps)
        return (y * 2.0 - 1.0).astype(np.float32)

    def _align_joint_bounds(self, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        target_dim = self.JOINT_DIM
        if lo.shape[-1] == target_dim:
            lo = lo.copy()
            hi = hi.copy()
            lo[..., :2] = 0.0
            hi[..., :2] = 0.0
            return lo, hi
        if lo.shape[-1] == target_dim - 2:
            padded_lo = np.zeros(target_dim, dtype=lo.dtype)
            padded_hi = np.zeros(target_dim, dtype=hi.dtype)
            padded_lo[2:] = lo
            padded_hi[2:] = hi
            return padded_lo, padded_hi
        logger.warning("MyBodexShadow: unexpected joint bounds dim=%d (target=%d); returning raw values", lo.shape[-1], target_dim)
        return lo, hi

    def _build_flattened_data_index(self) -> List[Dict[str, Any]]:
        """Build flattened data index where each item represents a single grasp."""
        flattened_data = []
        total_grasps = 0

        for scene_id in self.scene_ids:
            poses = self.scene_data[scene_id]['grasps']
            num_grasps = poses.shape[0]
            total_grasps += num_grasps
            
            for grasp_idx in range(num_grasps):
                flattened_data.append({
                    "scene_id": scene_id,
                    "grasp_index": grasp_idx
                })

        logger.info("Flattened data stats: Total grasps=%d across %d scenes", total_grasps, len(self.scene_ids))
        return flattened_data

    def _empty_scene_record(self) -> Dict[str, Any]:
        return {
            'object_name': None,
            'scale': None,
            'grasps': [],
            'rot_mats': [],
            'trans': [],
            'joint_angles': [],
        }

    def _load_data(self) -> None:
        """Load Bodex grasp data from either pt file or raw npy files."""
        if self.debug_mode:
            logger.info("MyBodexShadow: debug_mode=True, loading raw npy files from %s", self.succ_grasp_dir)
            scene_data_dict = self._load_data_from_hierarchical_dir()
        else:
            logger.info("MyBodexShadow: debug_mode=False, loading packed pt file from %s", self.pt_dataset_path)
            scene_data_dict = self._load_data_from_pt_file()
        self._finalize_scene_data(scene_data_dict)

    def _load_data_from_pt_file(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.pt_dataset_path):
            raise FileNotFoundError(
                f"Packed Bodex dataset not found: {self.pt_dataset_path}. "
                "Generate it via scripts/generate_bodex_pt_file.py."
            )

        dataset_blob = torch.load(self.pt_dataset_path, map_location='cpu')
        metadata = dataset_blob.get('metadata')
        if not metadata:
            raise ValueError(f"Invalid Bodex pt file: missing metadata at {self.pt_dataset_path}")

        scene_data_dict: Dict[str, Dict[str, Any]] = defaultdict(self._empty_scene_record)

        for idx, item in enumerate(metadata):
            object_name = item.get('object_name')
            scale_value_raw = item.get('scale')
            rot_mat = item.get('rotations')
            joint_angle = item.get('joint_positions')
            trans_world = item.get('translations')

            if object_name is None or scale_value_raw is None:
                logger.warning("MyBodexShadow: metadata entry %d missing object or scale, skipping", idx)
                continue

            try:
                if isinstance(scale_value_raw, torch.Tensor):
                    scale_value = float(scale_value_raw.item())
                else:
                    scale_value = float(scale_value_raw)
            except (TypeError, ValueError):
                logger.warning("MyBodexShadow: invalid scale value in metadata entry %d, skipping", idx)
                continue

            scene_id = f"{object_name}_scale{scale_value}"

            def _to_tensor(x):
                return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

            rot_mat_tensor = _to_tensor(rot_mat).to(dtype=torch.float32)
            joint_tensor = _to_tensor(joint_angle).to(dtype=torch.float32)
            trans_tensor = _to_tensor(trans_world).to(dtype=torch.float32)

            if rot_mat_tensor.shape != (3, 3):
                logger.warning("MyBodexShadow: invalid rotation matrix shape in entry %d, skipping", idx)
                continue
            if joint_tensor.shape[0] != self.JOINT_DIM:
                logger.warning("MyBodexShadow: invalid joint dimension in entry %d, skipping", idx)
                continue
            if trans_tensor.shape[0] != self.TRANS_DIM:
                logger.warning("MyBodexShadow: invalid translation dimension in entry %d, skipping", idx)
                continue

            pose = self._build_hand_pose(rot_mat_tensor, joint_tensor, trans_tensor)

            scene_bucket = scene_data_dict[scene_id]
            scene_bucket['object_name'] = object_name
            scene_bucket['scale'] = scale_value
            scene_bucket['grasps'].append(pose)
            scene_bucket['rot_mats'].append(rot_mat_tensor)
            scene_bucket['trans'].append(trans_tensor)
            scene_bucket['joint_angles'].append(joint_tensor)

        return scene_data_dict

    def _load_data_from_hierarchical_dir(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.succ_grasp_dir):
            raise FileNotFoundError(f"Grasp directory not found: {self.succ_grasp_dir}")

        raw_hand_pose_data = load_bodex_hand_pose_data(self.succ_grasp_dir)
        scene_data_dict: Dict[str, Dict[str, Any]] = defaultdict(self._empty_scene_record)

        for scene_id, qpos_29d_batch in raw_hand_pose_data.items():
            parts = scene_id.rsplit('_scale', 1)
            if len(parts) != 2:
                logger.warning(f"Invalid scene_id format: {scene_id}, skipping")
                continue

            object_name = parts[0]
            try:
                scale = float(parts[1])
            except ValueError:
                logger.warning(f"Invalid scale in scene_id: {scene_id}, skipping")
                continue

            scene_data_dict[scene_id]['object_name'] = object_name
            scene_data_dict[scene_id]['scale'] = scale

            for i in range(qpos_29d_batch.shape[0]):
                qpos_29d = qpos_29d_batch[i]
                try:
                    pose, rot_mat, trans_world, joint_angle = self._process_grasp_29d(qpos_29d)
                    scene_data_dict[scene_id]['grasps'].append(pose)
                    scene_data_dict[scene_id]['rot_mats'].append(rot_mat)
                    scene_data_dict[scene_id]['trans'].append(trans_world)
                    scene_data_dict[scene_id]['joint_angles'].append(joint_angle)
                except Exception as e:
                    logger.warning(f"Failed to process grasp {i} in {scene_id}: {e}")
                    continue

        return scene_data_dict

    def _finalize_scene_data(self, scene_data_dict: Dict[str, Dict[str, Any]]) -> None:
        all_scene_ids = list(scene_data_dict.keys())
        for scene_id in all_scene_ids:
            scene_data = scene_data_dict[scene_id]
            if not scene_data['grasps']:
                logger.warning(f"No valid grasps for scene {scene_id}, removing")
                del scene_data_dict[scene_id]
                continue

            for key in ('grasps', 'rot_mats', 'trans', 'joint_angles'):
                scene_data[key] = torch.stack(scene_data[key], dim=0)

        self.scene_data = dict(scene_data_dict)
        self.scene_ids = list(self.scene_data.keys())

        logger.info(
            "Loaded %d scenes from %s (rot_type: %s, pose_dim: %d)",
            len(self.scene_ids), self.name, self.rot_type, self.POSE_DIM,
        )

        try:
            counts = [int(self.scene_data[sid]['grasps'].shape[0]) for sid in self.scene_ids]
            if counts:
                cmin, cmax = min(counts), max(counts)
                cmean = sum(counts) / max(1, len(counts))
                czero = sum(1 for c in counts if c == 0)
                logger.info(
                    "MyBodexShadow: grasps/scene stats -> min=%d max=%d mean=%.1f zeros=%d (scenes=%d)",
                    cmin, cmax, cmean, czero, len(counts)
                )
        except Exception as _e:
            logger.debug("MyBodexShadow: failed to log dataset statistics: %s", _e)

    def _process_grasp_29d(self, qpos_29d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process 29-dimensional Bodex grasp data.

        Input format: [trans(3), quat(4), joints(22)]
        不做任何坐标系外参修正，直接返回 trans + joints + rot_repr。

        Returns:
            pose: hand_model_pose (POSE_DIM,) [trans, joints, rot_repr]
            rot_mat: rotation matrix (3, 3)
            trans_world: translation (3,)
            joint_angle: joint angles (24,)
        """
        # Extract components (new order)
        trans_mj = qpos_29d[:3].clone().detach().float()
        quat = qpos_29d[3:7].clone().detach().float()
        joint_angle = torch.zeros(self.JOINT_DIM, dtype=torch.float32, device=qpos_29d.device)
        joint_angle[2:] = qpos_29d[7:29].clone().detach().float()

        # Convert quaternion to rotation matrix
        # Normalize quaternion first
        quat_norm = quat / (torch.norm(quat) + 1e-8)
        rot_mat = quaternion_to_matrix(quat_norm.unsqueeze(0)).squeeze(0)  # (3,3)
        trans_world = trans_mj

        # Build hand_model_pose
        pose = self._build_hand_pose(rot_mat, joint_angle, trans_world)

        return pose, rot_mat, trans_world, joint_angle

    def _build_hand_pose(self, hand_rot_mat: torch.Tensor, joint_angle: torch.Tensor, global_trans: torch.Tensor) -> torch.Tensor:
        """Build hand_model_pose from rotation matrix, joints, and translation.

        Args:
            hand_rot_mat: rotation matrix (3, 3)
            joint_angle: joint angles (22,)
            global_trans: global translation (3,)

        Returns:
            hand_model_pose: (POSE_DIM,) [trans, joints, rot_repr]
            Note: We keep the historical order (trans, joints, rot_repr) in the dataset
            output for backward compatibility, but the values are already aligned for
            HandModel usage.
        """
        rot_repr = self._rotation_converter(hand_rot_mat.unsqueeze(0)).squeeze(0)
        hand_model_pose = torch.zeros(self.POSE_DIM, dtype=joint_angle.dtype, device=joint_angle.device)
        hand_model_pose[:self.TRANS_DIM] = global_trans
        hand_model_pose[self.TRANS_DIM:self.TRANS_DIM + self.JOINT_DIM] = joint_angle
        hand_model_pose[self.TRANS_DIM + self.JOINT_DIM:] = rot_repr
        return hand_model_pose

    def _convert_mjcf_to_base(self, rot_mats: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """Convert MJCF anchor translation to URDF base translation."""
        if trans.shape[0] == 0:
            return trans
        offset = self._apply_rotation_offset(rot_mats, MJCF_ANCHOR_TRANSLATION)
        return trans - offset

    def _convert_base_to_palm_center(self, rot_mats: torch.Tensor, base_trans: torch.Tensor) -> torch.Tensor:
        """Convert base anchor translation to palm center translation."""
        if base_trans.shape[0] == 0:
            return base_trans
        offset = self._apply_rotation_offset(rot_mats, PALM_CENTER_ANCHOR_TRANSLATION)
        return base_trans + offset

    @staticmethod
    def _apply_rotation_offset(rot_mats: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Apply rotation to offset vector."""
        t_off = offset.to(device=rot_mats.device, dtype=rot_mats.dtype)
        t_off = t_off.view(1, 3, 1).expand(rot_mats.shape[0], -1, -1)
        return torch.bmm(rot_mats, t_off).squeeze(-1)

    def _build_se3_matrices(self, rot_mats: torch.Tensor, trans: torch.Tensor, num_grasps: int) -> torch.Tensor:
        """Build SE(3) transformation matrices."""
        se3 = torch.eye(4, device=rot_mats.device, dtype=rot_mats.dtype).unsqueeze(0).repeat(num_grasps, 1, 1)
        se3[:, :3, :3] = rot_mats
        se3[:, :3, 3] = trans
        return se3

    def _load_point_clouds(self) -> None:
        if not hasattr(self, "point_cloud_path"):
            self.scene_pcds = {}
            return
        if not os.path.exists(self.point_cloud_path):
            logger.warning("Point cloud file not found: %s", self.point_cloud_path)
            self.scene_pcds = {}
            return
        try:
            with open(self.point_cloud_path, "rb") as handle:
                self.scene_pcds = pickle.load(handle)
        except Exception as exc:
            logger.warning("Failed to load point clouds from %s: %s", self.point_cloud_path, exc)
            self.scene_pcds = {}

    def _load_point_cloud(self, object_name: str, scale: float) -> torch.Tensor:
        pcd_object_name = object_name.replace("_", "-")
        scene_pc = self.scene_pcds.get(pcd_object_name)
        if scene_pc is None:
            scene_pc = self.scene_pcds.get(object_name)
        if (
            scene_pc is None
            or not isinstance(scene_pc, np.ndarray)
            or scene_pc.ndim < 2
            or scene_pc.shape[1] < 3
        ):
            out_dim = 6 if self.use_scene_normals else 3
            return torch.zeros((0, out_dim), dtype=torch.float32)

        # Determine output dimension based on use_scene_normals
        has_normals = scene_pc.shape[1] >= 6
        if self.use_scene_normals and has_normals:
            # Return xyz + normals (6D), scale only affects xyz
            scene_pc_out = scene_pc[:, :6].copy().astype(np.float32)
            scene_pc_out[:, :3] *= float(scale)  # Scale xyz only, normals remain unchanged
        else:
            # Return xyz only (3D)
            scene_pc_out = (scene_pc[:, :3] * float(scale)).astype(np.float32)

        if len(scene_pc_out) > self.max_points:
            if self.phase != "train":
                np.random.seed(0)
            resample_indices = np.random.permutation(len(scene_pc_out))
            scene_pc_out = scene_pc_out[resample_indices[: self.max_points]]

        return torch.from_numpy(scene_pc_out)

    def _get_scene_point_cloud(self, scene_id: str, object_name: str, scale: float) -> torch.Tensor:
        if scene_id in self._scene_pc_cache:
            return self._scene_pc_cache[scene_id]
        pc = self._load_point_cloud(object_name, scale)
        self._scene_pc_cache[scene_id] = pc
        return pc

    def _load_mesh(self, object_name: str, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        mesh_object_name = object_name.replace("_", "-")
        mesh_path = os.path.join(self.mesh_base_path, mesh_object_name, "coacd", "decomposed.obj")
        if not os.path.exists(mesh_path):
            logger.warning("Mesh file not found: %s", mesh_path)
            return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.long)
        try:
            obj_mesh = trimesh.load(mesh_path)
            obj_mesh.apply_scale(scale)
            return torch.from_numpy(obj_mesh.vertices).float(), torch.from_numpy(obj_mesh.faces).long()
        except Exception as exc:
            logger.warning("Failed to load mesh %s: %s", mesh_path, exc)
            return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.long)

    def _apply_anchor_transform(
        self,
        hand_poses: torch.Tensor,
        rot_mats: torch.Tensor,
        trans: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adjust translation in hand_poses / SE(3) according to target anchor."""
        if hand_poses.shape[0] == 0:
            return hand_poses, trans

        if self.trans_anchor == 'mjcf':
            return hand_poses, trans

        base_trans = self._convert_mjcf_to_base(rot_mats, trans)
        if self.trans_anchor == 'base':
            updated_poses = hand_poses.clone()
            updated_poses[:, :3] = base_trans
            return updated_poses, base_trans

        if self.trans_anchor == 'palm_center':
            palm_centers = self._convert_base_to_palm_center(rot_mats, base_trans)
            updated_poses = hand_poses.clone()
            updated_poses[:, :3] = palm_centers
            return updated_poses, palm_centers

        return hand_poses, trans

    def __len__(self) -> int:
        return len(self.data) if hasattr(self, 'data') else len(self.scene_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset (one grasp per item)."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.data)}")

        item = self.data[idx]
        scene_id = item['scene_id']
        grasp_index = item['grasp_index']

        scene_info = self.scene_data[scene_id]
        object_name = scene_info['object_name']
        scale = scene_info['scale']

        # Get single grasp
        all_poses = self.scene_data[scene_id]['grasps']
        all_rot_mats = self.scene_data[scene_id]['rot_mats']
        all_trans = self.scene_data[scene_id]['trans']
        
        hand_pose = all_poses[grasp_index:grasp_index+1]  # Keep as 2D for consistency
        rot_mat = all_rot_mats[grasp_index:grasp_index+1]
        trans = all_trans[grasp_index:grasp_index+1]
        
        hand_pose, anchor_trans = self._apply_anchor_transform(hand_pose, rot_mat, trans)

        # Enforce disabled DOF: first two qpos must be zero (24-d with leading zeros)
        hand_pose = hand_pose.clone()
        hand_pose[:, 3:5] = 0.0

        # Build SE(3) matrix
        se3_matrix = self._build_se3_matrices(rot_mat, anchor_trans, 1)

        scene_pc = self._get_scene_point_cloud(scene_id, object_name, scale)

        # Normalization (pose only)
        norm_pose = self._normalize_pose(hand_pose)
        # Convert numpy array to torch.Tensor for consistency with PyTorch dataset
        if isinstance(norm_pose, np.ndarray):
            norm_pose = torch.from_numpy(norm_pose).to(dtype=hand_pose.dtype, device=hand_pose.device)

        result = {
            'obj_code': object_name,
            'scene_id': scene_id,
            'hand_model_pose': hand_pose.squeeze(0),  # Return 1D tensor
            'norm_pose': norm_pose.squeeze(0),  # Return 1D tensor
            'se3': se3_matrix.squeeze(0),  # Return 1 SE(3) matrix
            'trans_anchor': self.trans_anchor,
            'rot_type': self.rot_type,
            'scene_pc': scene_pc,
        }

        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        return collate_batch_data(batch)
