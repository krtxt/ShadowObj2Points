"""DexGraspNet dataset with flattened data structure.

This class wraps preprocessed DexGraspNet assets and provides:
- flattened data structure: each item returns one hand_model_pose and norm_pose
- optional stats-based normalization of translations/joints
- multiple rotation representations (quat, r6d, euler, axis)
- multiple translation anchor options (base, palm_center, mjcf)
"""
import json
import logging
import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
)
from torch.utils.data import Dataset

from .utils.collate_utils import collate_batch_data


logger = logging.getLogger(__name__)
MJCF_ANCHOR_TRANSLATION = torch.tensor([0.0, -0.01, 0.213], dtype=torch.float32)
PALM_CENTER_ANCHOR_TRANSLATION = torch.tensor([0.008, -0.013, 0.283], dtype=torch.float32)
PALM_CENTER_MARGIN = 0.25


def load_from_json(input_file: str) -> Tuple[List[str], List[str], List[str]]:
    """Load dataset splits from a JSON file.

    The file must contain keys: '_train_split', '_test_split', '_all_split'.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data["_train_split"], data["_test_split"], data["_all_split"]


class MyDexGraspNet(Dataset):
    """PyTorch Dataset for DexGraspNet with flattened structure.

    Args:
        asset_dir: Root directory of prepared DexGraspNet assets.
        phase: 'train' | 'test' | 'all'.
        rot_type: Rotation representation: 'quat' | 'r6d' | 'euler' | 'axis'.
        trans_anchor: 'base' | 'palm_center' | 'mjcf' (alias 'anchor' -> 'palm_center').
        use_stats_normalization: If True, apply stats-based [-1, 1] normalization.
        normalization_stats_path: Path or stem for stats (.json or .pt).
        pt_file_name: Optional explicit .pt dataset file name.
        split_file_name: Optional explicit split json file name.
        debug_mode: When True, use debug data files; otherwise use normal data files.
    """
    TRANS_DIM = 3
    JOINT_DIM = 24

    def __init__(self, asset_dir='data/DexGraspNet', phase='all', rot_type='r6d',
                 trans_anchor: str = 'base',
                 use_stats_normalization: bool = True,
                 normalization_stats_path: Optional[str] = None,
                 pt_file_name: Optional[str] = None,
                 split_file_name: Optional[str] = None,
                 max_points: int = 4096,
                 point_clouds_file_name: Optional[str] = None,
                 debug_mode: bool = False,
                 use_scene_normals: bool = False):
        super().__init__()

        self.name = 'DexGraspNet'
        self.asset_dir = asset_dir
        self.phase = phase
        self.rot_type = rot_type
        self.trans_anchor = str(trans_anchor)
        if self.trans_anchor == 'anchor':
            self.trans_anchor = 'palm_center'
        valid_anchors = {'base', 'palm_center', 'mjcf'}
        if self.trans_anchor not in valid_anchors:
            raise ValueError(f"Unsupported trans_anchor '{self.trans_anchor}'. Expected one of {sorted(valid_anchors)}.")
        self.use_stats_normalization = use_stats_normalization
        self.normalization_stats_path = normalization_stats_path or os.path.join('assets', 'mydexgraspnet_normalization_stats_debug')
        self._stats_bounds: Dict[str, Dict[str, np.ndarray]] = {}

        self.debug_mode = debug_mode
        self.max_points = int(max_points) if int(max_points) > 0 else 4096
        self.point_clouds_file_name = point_clouds_file_name
        self.use_scene_normals = bool(use_scene_normals)

        # Optional explicit file names (relative to asset_dir); if provided, they take precedence
        self.pt_file_name = pt_file_name
        self.split_file_name = split_file_name

        self._init_rotation_config()
        self._init_paths()
        self._load_splits()
        self._validate_init_params()

        self.scene_data: Dict[str, Dict[str, Any]] = {}
        self.scene_ids: List[str] = []
        self.scene_pcds: Dict[str, np.ndarray] = {}
        self._scene_pc_cache: Dict[str, torch.Tensor] = {}
        self._missing_pcd_objects: set = set()

        self._load_data()
        self._load_point_clouds()

        # Build flattened data index - each item is a single grasp
        self.data = self._build_flattened_data_index()
        logger.info("MyDexGraspNet: Using flattened data with %d items.", len(self.data))

        if self.use_stats_normalization:
            self._try_load_normalization_stats()

    def _init_rotation_config(self) -> None:
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
        # Determine if explicit file names are provided; if so, they take precedence
        explicit = bool(self.pt_file_name or self.split_file_name)

        debug_defaults = {
            'pt': 'debug_grasp_data_128.pt',
            'split': 'debug_grasp_128.json',
            'pcd': 'debug_point_clouds_128.pkl',
        }
        normal_defaults = {
            'pt': 'dexgraspnet_shadowhand_downsample.pt',
            'split': 'grasp.json',
            'pcd': 'object_pcds_nors.pkl',
        }

        if explicit:
            pt_name = self.pt_file_name or debug_defaults['pt']
            split_name = self.split_file_name or debug_defaults['split']
            pcd_name = self.point_clouds_file_name or (debug_defaults['pcd'] if self.debug_mode else normal_defaults['pcd'])
        else:
            if self.debug_mode:
                pt_name = debug_defaults['pt']
                split_name = debug_defaults['split']
                pcd_name = debug_defaults['pcd']
            else:
                pt_name = normal_defaults['pt']
                split_name = normal_defaults['split']
                pcd_name = normal_defaults['pcd']

        if self.point_clouds_file_name is not None:
            pcd_name = self.point_clouds_file_name

        self.pt_folder = os.path.join(self.asset_dir, pt_name)
        self._split_file = os.path.join(self.asset_dir, split_name)
        self.point_cloud_path = os.path.join(self.asset_dir, pcd_name)
        self.point_cloud_path = os.path.join(self.asset_dir, pcd_name)

    def _load_splits(self) -> None:
        self._train_split, self._test_split, self._all_split = load_from_json(self._split_file)
        phase_map = {'train': self._train_split, 'test': self._test_split, 'all': self._all_split}
        if self.phase not in phase_map:
            raise ValueError(f"Unsupported phase '{self.phase}'. Expected one of {list(phase_map.keys())}.")
        self.split = phase_map[self.phase]

    def _validate_init_params(self) -> None:
        """Validate initialization parameters."""
        pass

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

    def _load_data(self) -> None:
        if not os.path.exists(self.pt_folder):
            raise FileNotFoundError(f"Data file not found: {self.pt_folder}")

        try:
            grasp_dataset = torch.load(self.pt_folder, weights_only=False)
        except TypeError:
            grasp_dataset = torch.load(self.pt_folder)
        scene_data_dict = defaultdict(lambda: {
            'object_name': None,
            'scale': None,
            'grasps': [],
            'rot_mats': [],
            'trans': [],
            'joint_angles': [],
        })

        for mdata in grasp_dataset['metadata']:
            self._process_grasp(mdata, scene_data_dict)

        all_scene_ids = list(scene_data_dict.keys())
        for scene_id in all_scene_ids:
            scene_data = scene_data_dict[scene_id]
            for key in ('grasps', 'rot_mats', 'trans', 'joint_angles'):
                scene_data[key] = torch.stack(scene_data[key], dim=0)

        self.scene_data = dict(scene_data_dict)
        self.scene_ids = [sid for sid in all_scene_ids if scene_data_dict[sid]['object_name'] in self.split]
        logger.info(
            "Loaded %d scenes from %s (phase: %s, split size: %d, rot_type: %s, pose_dim: %d)",
            len(self.scene_ids), self.name, self.phase, len(self.split), self.rot_type, self.POSE_DIM,
        )
        # Additional statistics to help diagnose data scale and distribution
        try:
            counts = [int(self.scene_data[sid]['grasps'].shape[0]) for sid in self.scene_ids]
            if counts:
                cmin, cmax = min(counts), max(counts)
                cmean = sum(counts) / max(1, len(counts))
                czero = sum(1 for c in counts if c == 0)
                # logger.info(
                #     "MyDexGraspNet: grasps/scene stats -> min=%d max=%d mean=%.1f zeros=%d (scenes=%d)",
                #     cmin, cmax, cmean, czero, len(counts)
                # )
            # Translation magnitude range
            trans_all = torch.cat([self.scene_data[sid]['trans'] for sid in self.scene_ids], dim=0)
            tmin = trans_all.min().item() if trans_all.numel() > 0 else 0.0
            tmax = trans_all.max().item() if trans_all.numel() > 0 else 0.0
            tmean = trans_all.float().mean().item() if trans_all.numel() > 0 else 0.0
            # logger.info("MyDexGraspNet: trans range -> min=%.4f max=%.4f mean=%.4f", tmin, tmax, tmean)
        except Exception as _e:
            logger.debug("MyDexGraspNet: failed to log dataset statistics: %s", _e)

    def _process_grasp(self, mdata: Dict[str, Any], scene_data_dict: Dict[str, Dict[str, Any]]) -> str:
        hand_rot_mat = mdata['rotations'].clone().detach().float()
        joint_angle = mdata['joint_positions'].clone().detach().float()
        global_trans = mdata['translations'].clone().detach().float()
        object_name = mdata['object_name']
        scale = self._get_scale(mdata)
        scene_id = f"{object_name}_scale{scale}"
        trans_transformed = torch.matmul(hand_rot_mat, global_trans)

        scene_data_dict[scene_id]['object_name'] = object_name
        scene_data_dict[scene_id]['scale'] = scale
        scene_data_dict[scene_id]['grasps'].append(self._build_hand_pose(hand_rot_mat, joint_angle, global_trans))
        scene_data_dict[scene_id]['rot_mats'].append(hand_rot_mat)
        scene_data_dict[scene_id]['trans'].append(trans_transformed)
        scene_data_dict[scene_id]['joint_angles'].append(joint_angle)
        return scene_id

    def _get_scale(self, mdata: Dict[str, Any]) -> float:
        return 1 / mdata['scale'] if 'UniDexGrasp' in self.pt_folder else mdata['scale']

    def _build_hand_pose(self, hand_rot_mat: torch.Tensor, joint_angle: torch.Tensor, global_trans: torch.Tensor) -> torch.Tensor:
        trans_transformed = torch.matmul(hand_rot_mat, global_trans)
        rot_repr = self._rotation_converter(hand_rot_mat.unsqueeze(0)).squeeze(0)
        hand_model_pose = torch.zeros(self.POSE_DIM, dtype=joint_angle.dtype, device=joint_angle.device)
        hand_model_pose[:self.TRANS_DIM] = trans_transformed
        hand_model_pose[self.TRANS_DIM:self.TRANS_DIM + self.JOINT_DIM] = joint_angle
        hand_model_pose[self.TRANS_DIM + self.JOINT_DIM:] = rot_repr
        return hand_model_pose

    def __len__(self) -> int:
        return len(self.data) if hasattr(self, 'data') else len(self.scene_ids)

    def _try_load_normalization_stats(self) -> None:
        base_path = self.normalization_stats_path

        # Auto add extension if needed
        if base_path.endswith('.pt'):
            path_pt = base_path
            path_json = base_path[:-3] + '.json'
        elif base_path.endswith('.json'):
            path_json = base_path
            path_pt = base_path[:-5] + '.pt'
        else:
            # No extension provided; try both
            path_pt = base_path + '.pt'
            path_json = base_path + '.json'

        stats_obj = None

        # Prefer JSON first
        if os.path.exists(path_json):
            try:
                with open(path_json, 'r') as f:
                    stats_obj = json.load(f)
                # logger.info("Loaded normalization stats from JSON: %s", path_json)
            except Exception as e:
                logger.warning("Failed to load JSON: %s", e)
                stats_obj = None

        # Fallback to PT if JSON fails
        if stats_obj is None and os.path.exists(path_pt):
            try:
                stats_obj = torch.load(path_pt, map_location='cpu')
                logger.info("Loaded normalization stats from PT: %s", path_pt)
            except Exception as e:
                logger.warning("Failed to load PT: %s", e)
                stats_obj = None

        if stats_obj is None:
            logger.warning("Normalization stats not found: %s", self.normalization_stats_path)
            self.use_stats_normalization = False
            return

        def pick_bounds(block: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
            mn = block.get('min_with_margin', block.get('min'))
            mx = block.get('max_with_margin', block.get('max'))
            return np.asarray(mn, dtype=np.float32), np.asarray(mx, dtype=np.float32)

        bounds: Dict[str, Dict[str, np.ndarray]] = {}
        mjcf_margin = 0.2
        palm_center_margin = PALM_CENTER_MARGIN

        if 'hand_trans' in stats_obj:
            lo, hi = pick_bounds(stats_obj['hand_trans'])
            bounds['hand_trans'] = {'min': lo, 'max': hi}
            diff = hi - lo

            shrink_mask_mjcf = diff > (2 * mjcf_margin)
            shrunk_mjcf_lo = np.where(shrink_mask_mjcf, lo + mjcf_margin, lo)
            shrunk_mjcf_hi = np.where(shrink_mask_mjcf, hi - mjcf_margin, hi)
            bounds['mjcf_trans'] = {'min': shrunk_mjcf_lo, 'max': shrunk_mjcf_hi}

            shrink_mask_palm = diff > (2 * palm_center_margin)
            shrunk_palm_lo = np.where(shrink_mask_palm, lo + palm_center_margin, lo)
            shrunk_palm_hi = np.where(shrink_mask_palm, hi - palm_center_margin, hi)
            bounds['palm_center'] = {'min': shrunk_palm_lo, 'max': shrunk_palm_hi}
        if 'joint_angles' in stats_obj:
            lo, hi = pick_bounds(stats_obj['joint_angles'])
            # Handle dimension mismatch: if 22-dim, pad with zeros at the front to get 24-dim
            # Ensure 1D arrays for dimension checking
            lo_flat = lo.flatten() if lo.ndim > 1 else lo
            hi_flat = hi.flatten() if hi.ndim > 1 else hi
            lo_dim = lo_flat.shape[0]
            hi_dim = hi_flat.shape[0]

            if lo_dim == 22 and hi_dim == 22:
                # Pad with 2 zeros at the front to get 24 dimensions
                lo = np.concatenate([np.zeros(2, dtype=np.float32), lo_flat])
                hi = np.concatenate([np.zeros(2, dtype=np.float32), hi_flat])
                # logger.info("MyDexGraspNet: joint_angles bounds padded from 22-dim to 24-dim (added 2 zeros at front)")
            elif lo_dim == 24 and hi_dim == 24:
                # Already 24-dim, use as-is
                lo = lo_flat
                hi = hi_flat
                logger.info("MyDexGraspNet: joint_angles bounds are 24-dim, using as-is")
            else:
                # Unexpected dimension, log warning but proceed
                logger.warning(
                    "MyDexGraspNet: joint_angles bounds have unexpected dimensions: lo_dim=%d, hi_dim=%d. Expected 22 or 24. Proceeding without padding.",
                    lo_dim, hi_dim
                )
                lo = lo_flat
                hi = hi_flat

            bounds['joint_angles'] = {'min': lo, 'max': hi}

        self._stats_bounds = bounds
        try:
            if 'hand_trans' in bounds:
                lo = bounds['hand_trans']['min']; hi = bounds['hand_trans']['max']
                # logger.info("MyDexGraspNet: hand_trans bounds min=%s max=%s", np.asarray(lo).tolist(), np.asarray(hi).tolist())
            if 'mjcf_trans' in bounds:
                lo = bounds['mjcf_trans']['min']; hi = bounds['mjcf_trans']['max']
                # logger.info("MyDexGraspNet: mjcf_trans bounds min=%s max=%s", np.asarray(lo).tolist(), np.asarray(hi).tolist())
            if 'palm_center' in bounds:
                lo = bounds['palm_center']['min']; hi = bounds['palm_center']['max']
                # logger.info("MyDexGraspNet: palm_center bounds min=%s max=%s", np.asarray(lo).tolist(), np.asarray(hi).tolist())
            if 'joint_angles' in bounds:
                lo = bounds['joint_angles']['min']; hi = bounds['joint_angles']['max']
                # logger.info("MyDexGraspNet: joint_angles bounds min(mean)=%.4f max(mean)=%.4f", float(np.mean(lo)), float(np.mean(hi)))
        except Exception as _e:
            logger.debug("MyDexGraspNet: failed to log normalization bounds: %s", _e)

    def _build_se3_matrices(self, rot_mats: torch.Tensor, trans: torch.Tensor, num_grasps: int) -> torch.Tensor:
        """Build SE(3) transformation matrices."""
        se3 = torch.eye(4, device=rot_mats.device, dtype=rot_mats.dtype).unsqueeze(0).repeat(num_grasps, 1, 1)
        se3[:, :3, :3] = rot_mats
        if num_grasps > 0:
            se3[:, :3, 3] = trans.to(device=rot_mats.device, dtype=rot_mats.dtype)
        return se3

    def _normalize_pose(self, hand_poses: torch.Tensor) -> np.ndarray:
        """Normalize hand pose to [-1, 1] range."""
        hand_poses_np = hand_poses.cpu().numpy() if isinstance(hand_poses, torch.Tensor) else hand_poses
        norm_trans = hand_poses_np[:, :3].astype(np.float32)
        norm_qpos = hand_poses_np[:, 3:27].astype(np.float32)
        rot_tail = hand_poses_np[:, 27:].astype(np.float32)

        if self.use_stats_normalization and self._stats_bounds:
            # Get translation bounds based on anchor
            if self.trans_anchor == 'palm_center' and 'palm_center' in self._stats_bounds:
                t_lo, t_hi = self._stats_bounds['palm_center']['min'], self._stats_bounds['palm_center']['max']
            elif self.trans_anchor == 'mjcf' and 'mjcf_trans' in self._stats_bounds:
                t_lo, t_hi = self._stats_bounds['mjcf_trans']['min'], self._stats_bounds['mjcf_trans']['max']
            else:
                t_bounds = self._stats_bounds.get('hand_trans', {})
                t_lo, t_hi = t_bounds.get('min'), t_bounds.get('max')

            if t_lo is not None and t_hi is not None:
                norm_trans = self._normalize_by_bounds(hand_poses_np[:, :3], t_lo.reshape(1, 3), t_hi.reshape(1, 3))

            # Get joint bounds
            j_bounds = self._stats_bounds.get('joint_angles', {})
            j_lo, j_hi = j_bounds.get('min'), j_bounds.get('max')
            if j_lo is not None and j_hi is not None:
                norm_qpos = self._normalize_by_bounds(hand_poses_np[:, 3:27], j_lo.reshape(1, 24), j_hi.reshape(1, 24))
                norm_qpos[..., :2] = 0.0  # Force disabled DOF to zero

        norm_trans = np.clip(norm_trans, -1.0, 1.0)
        norm_qpos = np.clip(norm_qpos, -1.0, 1.0)
        return np.concatenate([norm_trans, norm_qpos, rot_tail], axis=1).astype(np.float32)

    @staticmethod
    def _normalize_by_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize values to [-1, 1] with per-dimension bounds."""
        y = (x - lo) / np.maximum(hi - lo, eps)
        return (y * 2.0 - 1.0).astype(np.float32)

    def _convert_base_to_mjcf(self, rot_mats: torch.Tensor, base_trans: torch.Tensor) -> torch.Tensor:
        """Convert base anchor translation to MJCF anchor translation."""
        if base_trans.shape[0] == 0:
            return base_trans
        offset = self._apply_rotation_offset(rot_mats, MJCF_ANCHOR_TRANSLATION)
        return base_trans + offset

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.data)}")

        item = self.data[idx]
        scene_id = item['scene_id']
        grasp_index = item['grasp_index']

        scene_info = self.scene_data[scene_id]
        object_name = scene_info['object_name']

        # Get single grasp
        all_poses = self.scene_data[scene_id]['grasps']
        all_rot_mats = self.scene_data[scene_id]['rot_mats']
        all_trans = self.scene_data[scene_id]['trans']
        
        hand_pose = all_poses[grasp_index:grasp_index+1]  # Keep as 2D for consistency
        rot_mat = all_rot_mats[grasp_index:grasp_index+1]
        trans = all_trans[grasp_index:grasp_index+1]

        # Apply anchor transformation
        if self.trans_anchor == 'mjcf':
            anchor_trans_tensor = self._convert_base_to_mjcf(rot_mat, trans)
        elif self.trans_anchor == 'palm_center':
            anchor_trans_tensor = self._convert_base_to_palm_center(rot_mat, trans)
        else:
            anchor_trans_tensor = trans

        if self.trans_anchor != 'base':
            hand_pose = hand_pose.clone()
            hand_pose[:, :3] = anchor_trans_tensor

        # Enforce disabled DOF: first two qpos must be zero (24-d with leading zeros)
        hand_pose = hand_pose.clone()
        hand_pose[:, 3:5] = 0.0

        se3_matrix = self._build_se3_matrices(rot_mat, anchor_trans_tensor, 1)

        scene_pc = self._get_scene_point_cloud(scene_id, object_name, scene_info['scale'])

        # Normalization processing (pose only)
        norm_pose = self._normalize_pose(hand_pose)

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

    def _load_point_clouds(self) -> None:
        if not os.path.exists(self.point_cloud_path):
            logger.warning("Point cloud file not found: %s", self.point_cloud_path)
            self.scene_pcds = {}
            return
        try:
            with open(self.point_cloud_path, 'rb') as handle:
                self.scene_pcds = pickle.load(handle)
        except Exception as exc:
            logger.warning("Failed to load point clouds from %s: %s", self.point_cloud_path, exc)
            self.scene_pcds = {}

    def _load_point_cloud(self, object_name: str, scale: float) -> torch.Tensor:
        scene_pc = self.scene_pcds.get(object_name)
        if scene_pc is None or not isinstance(scene_pc, np.ndarray) or scene_pc.ndim < 2 or scene_pc.shape[1] < 3:
            if not hasattr(self, "_missing_pcd_objects"):
                self._missing_pcd_objects = set()
            if object_name not in self._missing_pcd_objects:
                logger.warning(
                    "MyDexGraspNet: no valid point cloud found for object '%s' in %s; returning empty point cloud.",
                    object_name,
                    self.point_cloud_path,
                )
                self._missing_pcd_objects.add(object_name)
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
            if self.phase != 'train':
                np.random.seed(0)
            resample_indices = np.random.permutation(len(scene_pc_out))
            scene_pc_out = scene_pc_out[resample_indices[:self.max_points]]

        return torch.from_numpy(scene_pc_out)

    def _get_scene_point_cloud(self, scene_id: str, object_name: str, scale: float) -> torch.Tensor:
        if scene_id in self._scene_pc_cache:
            return self._scene_pc_cache[scene_id]
        pc = self._load_point_cloud(object_name, scale)
        self._scene_pc_cache[scene_id] = pc
        return pc

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_batch_data(batch)
