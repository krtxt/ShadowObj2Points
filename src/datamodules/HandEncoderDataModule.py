import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import lightning.pytorch as L
from torch.utils.data import Dataset, DataLoader

# Optional OmegaConf support
try:  # pragma: no cover
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore

# Internal imports (Project specific)
from .MyDexGraspNet import MyDexGraspNet
from .MyBodexShadow import MyBodexShadow
from .CachedHandKeypointDataset import HDF5HandKeypointDataset, HDF5PointCloudCache
from utils.shadown_hand_model import (
    HandModel,
    PALM_CENTER_ANCHOR_TRANSLATION,
    MJCF_ANCHOR_TRANSLATION,
)

logger = logging.getLogger(__name__)


FINGER_PREFIX_MAP = {
    'th': 0, 'ff': 1, 'mf': 2, 'rf': 3, 'lf': 4,
}

JOINT_TYPE_ORDER = [
    'knuckle', 'proximal', 'middle', 'distal', 'tip',
    'metacarpal', 'base', 'hub', 'palm', 'wrist', 'forearm'
]
JOINT_TYPE_TO_ID = {name: i for i, name in enumerate(JOINT_TYPE_ORDER)}

MJCF_ANCHOR_MARGIN = 0.2

FINGER_CHAINS = {
    'ff': ['ffknuckle', 'ffproximal', 'ffmiddle', 'ffdistal', 'fftip'],
    'mf': ['mfknuckle', 'mfproximal', 'mfmiddle', 'mfdistal', 'mftip'],
    'rf': ['rfknuckle', 'rfproximal', 'rfmiddle', 'rfdistal', 'rftip'],
    'lf': ['lfmetacarpal', 'lfknuckle', 'lfproximal', 'lfmiddle', 'lfdistal', 'lftip'],
    'th': ['thbase', 'thproximal', 'thhub', 'thmiddle', 'thdistal', 'thtip'],
}

def _resolve_finger_id(link_name: str) -> int:
    ln = link_name.lower()
    for pre, fid in FINGER_PREFIX_MAP.items():
        if ln.startswith(pre):
            return fid
    return 5

def _resolve_joint_type_id(link_name: str) -> int:
    ln = link_name.lower()
    for typ in JOINT_TYPE_ORDER:
        if ln.endswith(typ):
            return JOINT_TYPE_TO_ID[typ]
    if ln in ('palm', 'wrist', 'forearm'):
        return JOINT_TYPE_TO_ID.get(ln, len(JOINT_TYPE_TO_ID))
    return JOINT_TYPE_TO_ID['knuckle']

def _build_ids_from_primary(primary: Dict[int, str], N: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    finger_ids = np.zeros(N, dtype=np.int64)
    joint_ids = np.zeros(N, dtype=np.int64)
    for i in range(N):
        lk = primary.get(i, 'unknown')
        finger_ids[i] = _resolve_finger_id(lk)
        joint_ids[i] = _resolve_joint_type_id(lk)
    
    num_fingers = int(finger_ids.max()) + 1 if N > 0 else 1
    num_joint_types = int(joint_ids.max()) + 1 if N > 0 else 1
    return finger_ids, joint_ids, num_fingers, num_joint_types

def _build_unique_names_and_primary_map(
    link_to_unique: Dict[str, List[int]],
    link_order: List[str],
) -> Tuple[List[str], Dict[int, str]]:
    """
    Selects a primary link name for each unique index based on suffix priority.
    """
    if not link_to_unique:
        return [], {}

    # Determine max unique index
    N = 0
    for idxs in link_to_unique.values():
        if idxs:
            N = max(N, max(int(i) for i in idxs) + 1)

    # Map unique ID back to list of candidate link names
    unique_to_links: Dict[int, List[str]] = defaultdict(list)
    for lk, idxs in link_to_unique.items():
        for u in idxs:
            unique_to_links[int(u)].append(lk)

    # Priority mapping (lower value = higher priority)
    suffix_priority_map = {
        'middle': 0, 'proximal': 1, 'knuckle': 2, 'metacarpal': 3,
        'base': 4, 'hub': 5, 'palm': 6, 'wrist': 7, 'forearm': 8,
        'distal': 9, 'tip': 10,
    }

    def _get_priority(lk: str) -> int:
        for suf, pr in suffix_priority_map.items():
            if lk.endswith(suf):
                return pr
        return 100

    primary: Dict[int, str] = {}
    names: List[Optional[str]] = [None] * N
    counters: Dict[str, int] = defaultdict(int)

    for u in range(N):
        cands = unique_to_links.get(u, [])
        if not cands:
            primary[u] = 'unknown'
        else:
            primary[u] = min(cands, key=_get_priority)

        # Generate unique name
        lk = primary[u]
        names[u] = f"{lk}_{counters[lk]}"
        counters[lk] += 1

    final_names = [n if n is not None else f"u{i}" for i, n in enumerate(names)]
    return final_names, primary

def _add_middle_distal_bridges_dynamic(
    link_to_unique: Dict[str, List[int]], 
    edges: List[Tuple[int, int, int]]
) -> None:
    """
    Dynamically adds middle<->distal directed edges (type=3).
    Includes logic for resolving aliases (e.g., ffmiddle vs ifmiddle).
    """
    def pick(idxs: List[int], mode: str = 'first') -> Optional[int]:
        if not idxs:
            return None
        return int(idxs[-1] if mode == 'last' else idxs[0])

    finger_sets = [
        (['ffmiddle', 'ifmiddle'], ['ffdistal', 'ifdistal']),
        (['mfmiddle'], ['mfdistal']),
        (['rfmiddle'], ['rfdistal']),
        (['lfmiddle'], ['lfdistal']),
        (['thmiddle'], ['thdistal']),
    ]
    
    # Specific override for picking the last index for certain links
    middle_pick_mode = {
        k: 'last' for k in ['ffmiddle', 'ifmiddle', 'mfmiddle', 'lfmiddle', 'rfmiddle', 'thmiddle']
    }

    for mids, dists in finger_sets:
        u, v = None, None
        
        # Resolve 'middle' joint index
        for nm in mids:
            if nm in link_to_unique and link_to_unique[nm]:
                mode = middle_pick_mode.get(nm, 'first')
                u = pick(link_to_unique[nm], mode)
                if u is not None:
                    break
        
        # Resolve 'distal' joint index
        for nm in dists:
            if nm in link_to_unique and link_to_unique[nm]:
                v = pick(link_to_unique[nm], 'first')
                if v is not None:
                    break
                    
        if u is not None and v is not None and u != v:
            edges.append((u, v, 3))
            edges.append((v, u, 3))

def _build_rigid_edges_unique(
    link_to_unique: Dict[str, List[int]],
    use_palm_star: bool = False,
    connect_forearm: bool = True,
    add_middle_distal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the graph edge index and edge types based on connectivity rules.
    """
    edges: List[Tuple[int, int, int]] = []

    # 1. Intra-link edges (Type 0: Rigid body connectivity)
    for idxs in link_to_unique.values():
        if not idxs or len(idxs) < 2:
            continue
        # Deduplicate indices while preserving order
        unique_seq = sorted(set(int(x) for x in idxs), key=lambda x: list(map(int, idxs)).index(x))
        for a, b in zip(unique_seq[:-1], unique_seq[1:]):
            if a != b:
                edges.extend([(a, b, 0), (b, a, 0)])

    # 2. Palm Star (Type 1: Kinematic tree roots)
    if use_palm_star and link_to_unique.get('palm'):
        palm_root = int(link_to_unique['palm'][0])
        for chain in FINGER_CHAINS.values():
            target = None
            for lk in chain:
                if link_to_unique.get(lk):
                    target = int(link_to_unique[lk][0])
                    break
            if target is not None and target != palm_root:
                edges.extend([(palm_root, target, 1), (target, palm_root, 1)])

    # 3. Forearm <-> Wrist (Type 2)
    if connect_forearm and link_to_unique.get('forearm') and link_to_unique.get('wrist'):
        u = int(link_to_unique['forearm'][0])
        v = int(link_to_unique['wrist'][0])
        if u != v:
            edges.extend([(u, v, 2), (v, u, 2)])

    # 4. Middle <-> Distal Bridges (Type 3)
    if add_middle_distal:
        _add_middle_distal_bridges_dynamic(link_to_unique, edges)

    # 5. Specific Kinematic Connections & Pruning
    def _get_idx(name: str, k: int = 0) -> Optional[int]:
        if name in link_to_unique and len(link_to_unique[name]) > k:
            return int(link_to_unique[name][k])
        return None
    
    def _get_idx_any(names: List[str], k: int = 0) -> Optional[int]:
        for nm in names:
            if nm in link_to_unique and len(link_to_unique[nm]) > k:
                return int(link_to_unique[nm][k])
        return None

    palm0 = _get_idx('palm', 0)
    thprox0 = _get_idx('thproximal', 0)
    ffprox0 = _get_idx('ffproximal', 0)
    lfmeta0 = _get_idx('lfmetacarpal', 0)
    ifmeta0 = _get_idx_any(['ifmetacarpal', 'ffmetacarpal', 'ffknuckle'], 0)

    # Add specific kinematic links
    for u, v in [
        (palm0, ifmeta0), (palm0, lfmeta0), (palm0, thprox0), (ffprox0, thprox0)
    ]:
        if u is not None and v is not None and u != v:
            edges.extend([(u, v, 1), (v, u, 1)])

    # Remove specific conflicting pairs
    remove_pairs = set()
    if palm0 is not None and ffprox0 is not None and palm0 != ffprox0:
        remove_pairs.update([(palm0, ffprox0), (ffprox0, palm0)])
    
    if ifmeta0 is not None and thprox0 is not None and ifmeta0 != thprox0:
        # Avoid removing valid ffprox-thprox if alias collision occurs
        if not (ffprox0 is not None and ifmeta0 == ffprox0):
            remove_pairs.update([(ifmeta0, thprox0), (thprox0, ifmeta0)])
            
    if lfmeta0 is not None and thprox0 is not None and lfmeta0 != thprox0:
        remove_pairs.update([(lfmeta0, thprox0), (thprox0, lfmeta0)])

    if remove_pairs:
        edges = [e for e in edges if (e[0], e[1]) not in remove_pairs]

    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    # Build final arrays
    uniq_edges = sorted(set(edges))
    edge_index = np.array([[u for u, _, _ in uniq_edges], [v for _, v, _ in uniq_edges]], dtype=np.int64)
    edge_type = np.array([t for _, _, t in uniq_edges], dtype=np.int64)
    return edge_index, edge_type

def _build_rigid_groups_from_link_map(link_to_unique: Dict[str, List[int]]) -> List[List[int]]:
    groups: List[List[int]] = []
    for idxs in link_to_unique.values():
        if not idxs:
            continue
        # Preserve order, remove duplicates
        seen = []
        for raw in idxs:
            val = int(raw)
            if val not in seen:
                seen.append(val)
        if seen:
            groups.append(seen)
    return groups

def _build_manual_rigid_groups_by_index(N: int) -> List[List[int]]:
    """
    Hardcoded rigid groups matching the observed hardware layout.
    """
    manual_groups = [
        [0, 1], [1, 5, 4, 3, 2, 6], [6, 20], [20, 21], [21, 22],
        [2, 7], [7, 8], [8, 9],
        [3, 10], [10, 11], [11, 12],
        [4, 13], [13, 14], [14, 15],
        [5, 16], [16, 17], [17, 18], [18, 19],
    ]
    return [g for g in manual_groups if all(0 <= i < N for i in g)]

def _compute_rest_lengths(
    hand_model: HandModel,
    edge_index_t: torch.Tensor,
    q_rest: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        xyz_rest = hand_model.get_joint_keypoints_unique(q=q_rest)[0]
    
    i, j = edge_index_t[0], edge_index_t[1]
    diff = xyz_rest[:, i, :] - xyz_rest[:, j, :]
    # Add epsilon for numerical stability
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9)
    return dist.squeeze(0)


class _HandEncoderPreparedDataset(Dataset):
    """
    Internal dataset wrapper that handles:
    1. Kinematic Forward Pass (World & Local)
    2. Caching integration
    3. Point cloud transformation & Normalization
    """
    def __init__(
        self,
        base_dataset: Dataset,
        hand_model: HandModel,
        canonical_link_to_unique: Dict[str, List[int]],
        finger_ids: torch.Tensor,
        joint_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_rest_lengths: torch.Tensor,
        scale: float = 1.0,
        norm_xyz_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_local_pose_only: bool = False,
        return_norm: bool = False,
        scene_pc_return_mode: Optional[str] = None,
        cache_dataset: Optional[Dataset] = None,
        cached_stored_anchor: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.cache = cache_dataset
        self.hand_model = hand_model
        self.cached_stored_anchor = (cached_stored_anchor or "base").lower()
        
        # Graph constants
        self.link_to_unique = canonical_link_to_unique
        self.finger_ids = finger_ids
        self.joint_type_ids = joint_type_ids
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_rest_lengths = edge_rest_lengths
        self.N = int(finger_ids.shape[0])
        
        self.scale = float(scale)
        self.use_local_pose_only = bool(use_local_pose_only)
        self.return_norm = bool(return_norm)

        # Scene PC Mode Configuration
        self.scene_pc_return_mode = str(scene_pc_return_mode).lower() if scene_pc_return_mode else ("norm" if self.return_norm else "raw")
        if self.scene_pc_return_mode not in ("raw", "norm", "both"):
            raise ValueError(f"Invalid scene_pc_return_mode: {self.scene_pc_return_mode}")
        if self.scene_pc_return_mode in ("norm", "both") and not self.return_norm:
            raise ValueError("scene_pc_return_mode requires return_norm=True for 'norm' or 'both'.")

        # Cache Integrity Check
        if self.cache is not None and len(self.cache) != len(self.base):
            raise RuntimeError(f"Cache/Base length mismatch: {len(self.cache)} vs {len(self.base)}")

        # Normalization Bounds
        self._norm_xyz_min: Optional[torch.Tensor] = None
        self._norm_xyz_max: Optional[torch.Tensor] = None
        self._norm_eps = 1e-6
        if self.return_norm and norm_xyz_bounds is not None:
            lo, hi = norm_xyz_bounds
            self._norm_xyz_min = lo.to(device=self.hand_model.device, dtype=torch.float32).view(1, 3)
            self._norm_xyz_max = hi.to(device=self.hand_model.device, dtype=torch.float32).view(1, 3)

    def __len__(self) -> int:
        return len(self.base)

    def _rotation_identity(self, device: torch.device) -> torch.Tensor:
        rot_type = getattr(self.hand_model, 'rot_type', 'quat')
        if rot_type == 'quat':
            vals = [1.0, 0.0, 0.0, 0.0]
        elif rot_type == 'r6d':
            vals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        elif rot_type in ('euler', 'axis'):
            vals = [0.0, 0.0, 0.0]
        else:
            vals = [1.0, 0.0, 0.0, 0.0]
        return torch.tensor(vals, dtype=torch.float32, device=device)

    @torch.no_grad()
    def _build_xyz_from_pose(self, q: torch.Tensor) -> torch.Tensor:
        """Computes keypoints in World Frame using the HandModel."""
        self.hand_model.update_kinematics(q)
        xyz = torch.zeros((self.N, 3), dtype=torch.float32, device=self.hand_model.device)
        filled = torch.zeros((self.N,), dtype=torch.bool, device=self.hand_model.device)
        
        # Map model name to status name if necessary
        mapper = getattr(self.hand_model, '_map_to_current_status_name', lambda x: x)
        
        for link_name, pts_local in self.hand_model.joint_key_points.items():
            if not pts_local:
                continue
            
            mapped_name = mapper(link_name) or link_name
            if mapped_name not in self.hand_model.current_status:
                continue
                
            # Transform local points -> world
            pts_t = torch.tensor(pts_local, device=self.hand_model.device, dtype=torch.float32)
            kp = self.hand_model.current_status[mapped_name].transform_points(pts_t)
            # Apply global transform
            kp = kp.expand(self.hand_model.batch_size, -1, -1)
            kp = torch.bmm(kp, self.hand_model.global_rotation.transpose(1, 2)) 
            kp = kp + self.hand_model.global_translation.unsqueeze(1)
            kp = kp * self.hand_model.scale
            
            # Fill unique buffer
            idxs = self.link_to_unique.get(link_name, [])
            for k, u in enumerate(idxs):
                u_idx = int(u)
                if 0 <= u_idx < self.N and not filled[u_idx]:
                    xyz[u_idx] = kp[0, k]
                    filled[u_idx] = True
                    
        if not torch.all(filled):
            xyz[~filled] = 0.0
        return xyz

    @torch.no_grad()
    def _build_xyz_from_pose_local(self, q: torch.Tensor) -> torch.Tensor:
        """Computes keypoints in Local/Canonical Frame (Root aligned)."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(device=self.hand_model.device, dtype=torch.float32)
        
        # Reset rigid pose (Translation=0, Rotation=Identity)
        q_local = q.clone()
        q_local[:, :3] = 0.0
        
        rot_id = self._rotation_identity(q_local.device)
        rot_dim = rot_id.shape[0]
        rot_start = max(0, q_local.shape[1] - rot_dim)
        if rot_start + rot_dim <= q_local.shape[1]:
            q_local[:, rot_start:rot_start + rot_dim] = rot_id.view(1, -1)
            
        # Update kinematics with reset pose
        self.hand_model.update_kinematics(q_local)
        
        # (Exact logic copy from _build_xyz_from_pose regarding point filling)
        xyz = torch.zeros((self.N, 3), dtype=torch.float32, device=self.hand_model.device)
        filled = torch.zeros((self.N,), dtype=torch.bool, device=self.hand_model.device)
        mapper = getattr(self.hand_model, '_map_to_current_status_name', lambda x: x)

        for link_name, pts_local in self.hand_model.joint_key_points.items():
            if not pts_local:
                continue
            mapped_name = mapper(link_name) or link_name
            if mapped_name not in self.hand_model.current_status:
                continue
            pts_t = torch.tensor(pts_local, device=self.hand_model.device, dtype=torch.float32)
            kp = self.hand_model.current_status[mapped_name].transform_points(pts_t)
            kp = kp.expand(self.hand_model.batch_size, -1, -1)
            # Note: Global rotation/translation is identity/zero here due to q_local, except scale
            kp = kp * self.hand_model.scale
            
            idxs = self.link_to_unique.get(link_name, [])
            for k, u in enumerate(idxs):
                u_idx = int(u)
                if 0 <= u_idx < self.N and not filled[u_idx]:
                    xyz[u_idx] = kp[0, k]
                    filled[u_idx] = True

        if not torch.all(filled):
            xyz[~filled] = 0.0
            
        # Re-apply anchor translation if it exists (for centering logic)
        anchor_trans = getattr(self.hand_model, 'global_translation', None)
        if anchor_trans is not None and anchor_trans.numel() >= 3:
            xyz = xyz + anchor_trans[0].view(1, 3)
            
        return xyz

    def _normalize_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        if self._norm_xyz_min is None or self._norm_xyz_max is None:
            raise RuntimeError("Normalization bounds not initialized.")
        denom = torch.clamp(self._norm_xyz_max - self._norm_xyz_min, min=self._norm_eps)
        norm = (xyz - self._norm_xyz_min) / denom
        return torch.clamp(norm * 2.0 - 1.0, -1.0, 1.0)

    def _denormalize_xyz(self, norm_xyz: torch.Tensor) -> torch.Tensor:
        if self._norm_xyz_min is None or self._norm_xyz_max is None:
            raise RuntimeError("Normalization bounds not initialized.")
        norm = torch.clamp(norm_xyz, -1.0, 1.0)
        unscaled = (norm + 1.0) * 0.5
        denom = torch.clamp(self._norm_xyz_max - self._norm_xyz_min, min=self._norm_eps)
        return unscaled * denom + self._norm_xyz_min

    def _xyz_from_cache(self, xyz_local: Any, se3: Any) -> torch.Tensor:
        xyz_t = torch.as_tensor(xyz_local, dtype=torch.float32)
        se3_t = torch.as_tensor(se3, dtype=torch.float32)
        rot, trans = se3_t[:3, :3], se3_t[:3, 3]
        return torch.matmul(xyz_t, rot.transpose(0, 1)) + trans.unsqueeze(0)

    def _anchor_offset(self, anchor: str, device: Optional[torch.device] = None) -> torch.Tensor:
        a = (anchor or "base").lower()
        if a == "palm_center":
            off = PALM_CENTER_ANCHOR_TRANSLATION
        elif a == "mjcf":
            off = MJCF_ANCHOR_TRANSLATION
        else:
            off = torch.zeros(3, dtype=torch.float32)
        return off.to(device=device or self.hand_model.device, dtype=torch.float32)

    @torch.no_grad()
    def _scene_pc_world_to_local(
        self,
        scene_pc: torch.Tensor,
        R_world: torch.Tensor,
        t_world: torch.Tensor,
        t_local: torch.Tensor,
    ) -> torch.Tensor:
        if scene_pc.ndim != 2 or scene_pc.shape[1] < 3: return scene_pc
        if R_world.shape != (3, 3) or t_world.shape != (3,) or t_local.shape != (3,): return scene_pc

        pts = scene_pc[:, :3].to(dtype=torch.float32)
        # Transform: (X_world - T_world) @ R_world + T_local
        local = (pts - t_world.view(1, 3)) @ R_world.float() + t_local.view(1, 3).float()
        
        if scene_pc.shape[1] > 3:
            return torch.cat([local.to(dtype=scene_pc.dtype), scene_pc[:, 3:]], dim=1)
        return local.to(dtype=scene_pc.dtype)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        hand_pose_t = torch.as_tensor(sample['hand_model_pose'], dtype=torch.float32)
        
        # Retrieve Cache
        if self.cache is not None:
            c_sample = self.cache[idx]
            cached_xyz, cached_se3 = c_sample.get('cached_xyz_local'), c_sample.get('cached_se3')
        else:
            cached_xyz, cached_se3 = sample.get('cached_xyz_local'), sample.get('cached_se3')

        # Handle Scene Point Cloud Input
        scene_pc_raw = sample.get('scene_pc')
        if isinstance(scene_pc_raw, (torch.Tensor, np.ndarray)):
            scene_pc = torch.as_tensor(scene_pc_raw, dtype=torch.float32)
            if scene_pc.ndim < 2 or scene_pc.shape[1] < 3:
                scene_pc = torch.zeros((0, 3), dtype=torch.float32)
        else:
            scene_pc = torch.zeros((0, 3), dtype=torch.float32)

        # Compute XYZ and Scene PC
        if self.use_local_pose_only:
            xyz = None
            target_anchor = getattr(self.hand_model, "anchor", "base")
            stored_anchor = self.cached_stored_anchor or "base"
            stored_off = self._anchor_offset(stored_anchor, device=self.hand_model.device)
            target_off = self._anchor_offset(target_anchor, device=self.hand_model.device)

            if cached_xyz is not None:
                xyz = torch.as_tensor(cached_xyz, dtype=torch.float32, device=self.hand_model.device)
                xyz = xyz + (stored_off - target_off)

            # Scene point cloud: use cached SE3 if available, otherwise fall back to FK
            if scene_pc.numel() > 0 and cached_se3 is not None:
                se3_t = torch.as_tensor(cached_se3, dtype=torch.float32)
                R_world = se3_t[:3, :3]
                t_world = se3_t[:3, 3]
                t_local = (stored_off - target_off)
                scene_pc = self._scene_pc_world_to_local(scene_pc, R_world, t_world, t_local)

            if xyz is None or (scene_pc.numel() > 0 and cached_se3 is None):
                q = hand_pose_t.unsqueeze(0).to(self.hand_model.device)
                _ = self._build_xyz_from_pose(q)
                R_world = getattr(self.hand_model, "global_rotation", None)
                t_world = getattr(self.hand_model, "global_translation", None)
                
                xyz = self._build_xyz_from_pose_local(q).to(torch.float32)
                t_local = getattr(self.hand_model, "global_translation", None)
                
                if (scene_pc.numel() > 0 and 
                    all(x is not None and x.numel() > 0 for x in [R_world, t_world, t_local])):
                    scene_pc = self._scene_pc_world_to_local(scene_pc, R_world[0], t_world[0], t_local[0])
        else:
            # Global Mode: Use Cache or Compute
            if cached_xyz is not None and cached_se3 is not None:
                xyz = self._xyz_from_cache(cached_xyz, cached_se3).to(torch.float32)
            elif cached_xyz is not None:
                xyz = torch.as_tensor(cached_xyz, dtype=torch.float32)
            else:
                q = hand_pose_t.unsqueeze(0).to(self.hand_model.device)
                xyz = self._build_xyz_from_pose(q).to(torch.float32)

        # Build Output Dictionary
        xyz_cpu = xyz.cpu()
        norm_xyz = self._normalize_xyz(xyz).cpu() if self.return_norm else xyz_cpu
        
        # Handle Norm Pose (Precomputed or Raw)
        norm_pose = sample.get('norm_pose')
        if norm_pose is not None:
            norm_pose = torch.as_tensor(norm_pose).cpu()

        # Handle Local Pose Output
        out_pose = hand_pose_t
        if self.use_local_pose_only:
            hp = hand_pose_t.clone()
            if hp.numel() >= 6:
                hp[:3] = 0.0
                rot_id = self._rotation_identity(hp.device).cpu()
                tail_len = rot_id.shape[0]
                rot_start = max(0, hp.shape[0] - tail_len)
                if rot_start + tail_len <= hp.shape[0]:
                    hp[rot_start:rot_start + tail_len] = rot_id.to(hp.dtype)
            out_pose = hp.cpu()

        result = {
            'xyz': xyz_cpu,
            'norm_xyz': norm_xyz,
            'hand_model_pose': out_pose,
            'norm_pose': norm_pose,
            'obj_code': sample.get('obj_code'),
            'scene_id': sample.get('scene_id'),
        }

        # Process Scene PC Output (Raw/Norm/Both)
        if scene_pc.numel() > 0:
            norm_pc = None
            if self.return_norm:
                xyz_part = scene_pc[:, :3]
                norm_xyz_part = self._normalize_xyz(xyz_part)
                if scene_pc.shape[1] > 3:
                    norm_pc = torch.cat([norm_xyz_part, scene_pc[:, 3:]], dim=1)
                else:
                    norm_pc = norm_xyz_part
            
            if self.scene_pc_return_mode in ("raw", "both"):
                result['scene_pc'] = scene_pc.cpu()
            if self.scene_pc_return_mode in ("norm", "both") and norm_pc is not None:
                result['norm_scene_pc'] = norm_pc.cpu()

        return result


def _collate_for_hand_encoder(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}
        
    # Stack standard tensors
    out = {
        'xyz': torch.stack([b['xyz'] for b in batch], dim=0),
        'norm_xyz': torch.stack([b['norm_xyz'] for b in batch], dim=0),
        'hand_model_pose': torch.stack([b['hand_model_pose'] for b in batch], dim=0),
        'obj_code': [b.get('obj_code') for b in batch],
        'scene_id': [b.get('scene_id') for b in batch],
    }
    
    if all(b.get('norm_pose') is not None for b in batch):
        out['norm_pose'] = torch.stack([b['norm_pose'] for b in batch], dim=0)
    else:
        out['norm_pose'] = None

    # Handle Variable Length Point Clouds (Padding)
    pc_keys = []
    if any('scene_pc' in b for b in batch): pc_keys.append('scene_pc')
    if any('norm_scene_pc' in b for b in batch): pc_keys.append('norm_scene_pc')

    if pc_keys:
        # Determine max N and feature dim
        max_n = 0
        feat_dim = None
        ref_device = None
        ref_dtype = None

        for b in batch:
            for k in pc_keys:
                t = b.get(k)
                if isinstance(t, (torch.Tensor, np.ndarray)):
                    t = torch.as_tensor(t)
                    if t.ndim == 2:
                        max_n = max(max_n, int(t.shape[0]))
                        if feat_dim is None: 
                            feat_dim = int(t.shape[1])
                            ref_device = t.device
                            ref_dtype = t.dtype
        
        if max_n > 0 and feat_dim is not None:
            batch_size = len(batch)
            for k in pc_keys:
                padded = torch.zeros((batch_size, max_n, feat_dim), dtype=ref_dtype, device=ref_device)
                for i, b in enumerate(batch):
                    t = b.get(k)
                    if t is not None:
                        t = torch.as_tensor(t)
                        if t.ndim == 2:
                            n_i = min(int(t.shape[0]), max_n)
                            padded[i, :n_i, :t.shape[1]] = t[:n_i]
                out[k] = padded

    return out


class HandEncoderDataModule(L.LightningDataModule if L is not None else object):
    def __init__(
        self,
        name: Optional[str] = None,
        mode: str = "dexgraspnet",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        rot_type: str = "quat",
        trans_anchor: str = "palm_center",
        hand_scale: float = 1.0,
        urdf_assets_meta_path: Optional[str] = None,
        use_palm_star: bool = False,
        connect_forearm: bool = True,
        add_middle_distal: bool = True,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        use_local_pose_only: bool = False,
        train_phase: Optional[str] = None,
        val_phase: Optional[str] = None,
        test_phase: Optional[str] = None,
        use_cached_keypoints: bool = False,
        cache_root: Optional[str] = None,
        cache_max_shards_in_memory: Optional[Any] = None,
        cache_max_shards: Optional[Any] = 2,
        norm_xyz_stats_path: Optional[str] = None,
        xyz_per_point_stats_path: Optional[str] = None,
        cache_preload_to_ram: bool = False,
        cache_show_progress: bool = True,
        prefetch_factor: int = 1,
        persistent_workers: Optional[bool] = None,
        data_cfg: Optional[Any] = None,  # Legacy ignored
        return_norm: bool = False,
        scene_pc_return_mode: Optional[str] = None,
        # New options for cache-only mode
        cache_only_mode: bool = False,
        point_cloud_hdf5_path: Optional[str] = None,
        point_cloud_max_cache_size: int = 200,
        max_points: int = 4096,
    ) -> None:
        super().__init__()
        
        # Basic Config
        self.name = name
        self.mode = str(mode or "dexgraspnet").lower()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers) if persistent_workers is not None else (self.num_workers > 0)
        
        # Dataset Config
        if OmegaConf is not None and isinstance(dataset_kwargs, DictConfig):  # type: ignore
            dataset_kwargs = OmegaConf.to_container(dataset_kwargs, resolve=True)
        self.dataset_kwargs = dict(dataset_kwargs or {})

        # Hand/Graph Config
        self.rot_type = str(rot_type or 'quat')
        self.trans_anchor = str(trans_anchor or 'palm_center')
        self.hand_scale = float(hand_scale)
        self.urdf_assets_meta_path = urdf_assets_meta_path
        self.use_palm_star = bool(use_palm_star)
        self.connect_forearm = bool(connect_forearm)
        self.add_middle_distal = bool(add_middle_distal)

        # Caching Config
        self.use_cached_keypoints = bool(use_cached_keypoints)
        self.cache_root = cache_root
        self.cache_preload_to_ram = bool(cache_preload_to_ram)
        self.cache_show_progress = bool(cache_show_progress)
        self._cache_mode_dir: Optional[Path] = None
        self._cache_meta_checked = False
        self.cache_max_shards = self._parse_cache_max(cache_max_shards_in_memory) or self._parse_cache_max(cache_max_shards) or 2
        self._cache_usage_logged: bool = False
        self._cache_meta: Optional[Dict[str, Any]] = None
        
        # Cache-only mode config (no base dataset dependency)
        self.cache_only_mode = bool(cache_only_mode)
        self.point_cloud_hdf5_path = str(point_cloud_hdf5_path) if point_cloud_hdf5_path else None
        self.point_cloud_max_cache_size = int(point_cloud_max_cache_size)
        self.max_points = int(max_points)
        self._point_cloud_cache = None  # Lazy initialized

        # Normalization Config
        self.return_norm = bool(return_norm)
        self.use_local_pose_only = bool(use_local_pose_only)
        self.scene_pc_return_mode = str(scene_pc_return_mode).lower() if scene_pc_return_mode else ("norm" if self.return_norm else "raw")
        
        # Stats Paths
        self.norm_xyz_stats_path = str(norm_xyz_stats_path) if norm_xyz_stats_path else None
        self.xyz_per_point_stats_path = str(xyz_per_point_stats_path) if xyz_per_point_stats_path else None
        if self.dataset_kwargs.get('normalization_stats_path'):
            self.normalization_stats_path = str(self.dataset_kwargs['normalization_stats_path'])
        else:
            self.normalization_stats_path = None

        # Phases
        self.train_phase = train_phase
        self.val_phase = val_phase
        self.test_phase = test_phase

        # State State
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self._prepared_constants: Optional[Dict[str, Any]] = None
        self._norm_xyz_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @staticmethod
    def _parse_cache_max(v: Any) -> Optional[Union[int, str]]:
        if v is None: return None
        if isinstance(v, str):
            if v.strip().lower() in ('all', 'full', 'infinite', 'inf', 'max'):
                return 'all'
            try: return int(v)
            except Exception: return None
        try: return int(v)
        except Exception: return None

    def prepare_data(self) -> None:
        pass

    def _resolve_path(self, configured_path: Optional[str], default_relpath: str) -> Path:
        proj_root = Path(__file__).resolve().parents[2]
        if configured_path:
            p = Path(configured_path)
            return p if p.is_absolute() else proj_root / p
        return proj_root / default_relpath

    def _load_handencoder_scene_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        json_path = self._resolve_path(self.norm_xyz_stats_path, "data/DexGraspNet/handencoder_scene_xyz_bounds.json")
        if not json_path.exists():
            raise RuntimeError(f"Norm stats missing: {json_path}")
        
        with open(json_path, "r") as f:
            stats = json.load(f).get("stats", {}).get("combined", {})
        
        mins = np.asarray(stats.get("min"), dtype=np.float32).reshape(-1)
        maxs = np.asarray(stats.get("max"), dtype=np.float32).reshape(-1)
        if mins.shape[0] != 3 or maxs.shape[0] != 3:
            raise RuntimeError(f"Invalid 3D bounds in {json_path}")
            
        return torch.from_numpy(mins).view(1, 3), torch.from_numpy(maxs).view(1, 3)

    def _load_handencoder_per_point_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        json_path = self._resolve_path(self.xyz_per_point_stats_path, "data/DexGraspNet/handencoder_xyz_per_point_stats.json")
        if not json_path.exists():
            raise RuntimeError(f"Per-point stats missing: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f).get("stats", {}).get("xyz_per_point", {}).get("per_point", [])
            
        if not data:
            raise RuntimeError("Empty per-point stats data.")
            
        can_xyz = torch.tensor([p["canonical_xyz"] for p in data], dtype=torch.float32)
        approx_std = torch.tensor([p["approx_std"] for p in data], dtype=torch.float32)
        return can_xyz, approx_std

    def _ensure_norm_xyz_bounds(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.return_norm:
            return None
        if not self.use_local_pose_only or self.trans_anchor != "palm_center":
            raise RuntimeError("Normalization requires use_local_pose_only=True and trans_anchor='palm_center'.")
            
        if self._norm_xyz_bounds is None:
            lo, hi = self._load_handencoder_scene_bounds()
            self._norm_xyz_bounds = (lo.to(device), hi.to(device))
            logger.info(f"HandEncoderDataModule: Loaded bounds min={lo.tolist()}, max={hi.tolist()}")
        return self._norm_xyz_bounds

    def _clone_norm_bounds(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return (self._norm_xyz_bounds[0].clone(), self._norm_xyz_bounds[1].clone()) if self._norm_xyz_bounds else None

    def _load_urdf_assets(self) -> Tuple[str, str]:
        proj_root = Path(__file__).resolve().parents[2]
        meta_path = Path(self.urdf_assets_meta_path) if self.urdf_assets_meta_path else proj_root / 'assets/urdf/urdf_assets_meta.json'
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return str(proj_root / meta['urdf_path']['shadowhand']), str(proj_root / meta['meshes_path']['shadowhand'])

    def _resolve_cache_dir(self) -> Path:
        if self._cache_mode_dir:
            return self._cache_mode_dir
        if not self.cache_root:
            raise ValueError("cache_root required for use_cached_keypoints=True")
            
        base = Path(self.cache_root)
        base = base if base.is_absolute() else Path.cwd() / base
        
        mode_dir = base / (self.mode or 'default')
        if not mode_dir.exists():
            mode_dir = base if base.exists() else None
            
        if not mode_dir or not mode_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {mode_dir or base}")
            
        self._cache_mode_dir = mode_dir
        logger.info(f"HandEncoderDataModule: using cached keypoints from {mode_dir}")
        return mode_dir

    def _make_base_dataset(self, phase: str) -> Dataset:
        kwargs = {**self.dataset_kwargs, 'phase': phase, 'rot_type': self.rot_type, 'trans_anchor': self.trans_anchor}
        if self.mode in ('dexgraspnet', 'dex'):
            return MyDexGraspNet(**kwargs)
        if self.mode in ('bodexshadow', 'bodex', 'bodexhshadow'):
            return MyBodexShadow(**kwargs)
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _get_point_cloud_cache(self) -> Optional[HDF5PointCloudCache]:
        """Get or create point cloud cache (lazy initialization)."""
        if self._point_cloud_cache is not None:
            return self._point_cloud_cache
        
        if not self.point_cloud_hdf5_path:
            return None
        
        pc_path = Path(self.point_cloud_hdf5_path)
        if not pc_path.is_absolute():
            proj_root = Path(__file__).resolve().parents[2]
            pc_path = proj_root / pc_path
        
        if not pc_path.exists():
            logger.warning(f"Point cloud HDF5 not found: {pc_path}")
            return None
        
        self._point_cloud_cache = HDF5PointCloudCache(
            str(pc_path),
            max_cache_size=self.point_cloud_max_cache_size,
            max_points=self.max_points,
        )
        logger.info(f"HandEncoderDataModule: point cloud cache enabled from {pc_path}")
        return self._point_cloud_cache

    def _make_cached_dataset(self, phase: str) -> Dataset:
        cache_dir = self._resolve_cache_dir()
        h5_path = cache_dir / f"{phase}_cache.h5"
        
        if not h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 cache file not found for phase '{phase}': {h5_path}"
            )
        
        # Get point cloud cache if configured
        pc_cache = self._get_point_cloud_cache() if self.cache_only_mode else None
        
        ds = HDF5HandKeypointDataset(
            str(h5_path),
            phase=phase,
            show_progress=self.cache_show_progress,
            point_cloud_cache=pc_cache,
        )
        logger.info("HandEncoderDataModule: cache enabled (phase=%s) using HDF5 file: %s", phase, h5_path)
        self._cache_usage_logged = True

        if not self._cache_meta_checked:
            meta = ds.meta
            anchor = meta.get('stored_anchor')
            logger.info(f"Cache Meta: { {k: meta.get(k) for k in ('stored_anchor', 'source_anchor', 'hand_scale')} }")
            if anchor and anchor.lower() != 'base':
                logger.warning(f"Cache stored with anchor '{anchor}', expected 'base'.")
            self._cache_meta_checked = True
            self._cache_meta = meta
        return ds

    @torch.no_grad()
    def _build_canonical(self, hand_model: HandModel) -> Dict[str, Any]:
        """Constructs the canonical graph structure using the HandModel."""
        # Setup Identity Rest Pose
        rot_len = 6 if self.rot_type == 'r6d' else (3 if self.rot_type in ('euler', 'axis') else 4)
        rot_tail = torch.tensor([1.0, 0.0, 0.0, 0.0] if rot_len == 4 else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] if rot_len == 6 else [0.0]*3, device=hand_model.device).unsqueeze(0)
        
        q_rest = torch.zeros((1, 3 + 24 + rot_len), device=hand_model.device)
        if rot_len > 3:
            q_rest[:, 27:27 + rot_len] = rot_tail

        # Extract Topology
        xyz_u, link_to_unique = hand_model.get_joint_keypoints_unique(q=q_rest)
        link_order = [lk for lk in hand_model.joint_key_points.keys() if lk in link_to_unique]
        names, primary = _build_unique_names_and_primary_map(link_to_unique, link_order)
        f_ids, j_ids, n_fing, n_joint = _build_ids_from_primary(primary, len(names))
        
        edge_idx, edge_typ = _build_rigid_edges_unique(
            link_to_unique, 
            use_palm_star=self.use_palm_star, 
            connect_forearm=self.connect_forearm, 
            add_middle_distal=self.add_middle_distal
        )

        # To Tensor
        dev = hand_model.device
        return {
            'link_to_unique': link_to_unique,
            'finger_ids': torch.from_numpy(f_ids).to(dev, dtype=torch.long),
            'joint_type_ids': torch.from_numpy(j_ids).to(dev, dtype=torch.long),
            'edge_index': torch.from_numpy(edge_idx).to(dev, dtype=torch.long),
            'edge_type': torch.from_numpy(edge_typ).to(dev, dtype=torch.long),
            'edge_rest_lengths': _compute_rest_lengths(hand_model, torch.from_numpy(edge_idx).to(dev), q_rest).float(),
            'template_xyz': xyz_u.squeeze(0).float(),
            'rigid_groups': _build_manual_rigid_groups_by_index(len(names)),
            'num_fingers': n_fing, 'num_joint_types': n_joint,
            'N': int(f_ids.shape[0]), 'E': int(edge_idx.shape[1]),
        }

    def _ensure_prepared_constants(self) -> None:
        if self._prepared_constants is not None:
            return

        urdf, meshes = self._load_urdf_assets()
        device = torch.device('cpu')
        
        hand_model = HandModel(
            robot_name='shadowhand', urdf_filename=urdf, mesh_path=meshes, 
            batch_size=1, device=device, hand_scale=self.hand_scale, 
            rot_type=self.rot_type, anchor=self.trans_anchor, mesh_source='urdf'
        )
        
        canon = self._build_canonical(hand_model)
        c_xyz, c_std = self._load_handencoder_per_point_stats()
        
        if c_xyz.shape[0] != canon['N']:
            raise RuntimeError(f"Stats mismatch: expected {canon['N']} points, got {c_xyz.shape[0]}")

        self._prepared_constants = {
            'hand_model': hand_model,
            **canon,
            'canonical_xyz': c_xyz.to(device),
            'approx_std': c_std.to(device),
        }
        self._ensure_norm_xyz_bounds(device)

    def setup(self, stage: Optional[str] = None) -> None:
        self._ensure_prepared_constants()
        hm = self._prepared_constants['hand_model']
        c = self._prepared_constants
        
        logger.info(
            "HandEncoderDataModule setup: mode=%s, rot_type=%s, trans_anchor=%s, "
            "batch_size=%d, num_workers=%d, pin_memory=%s, prefetch_factor=%d, persistent_workers=%s, "
            "use_local_pose_only=%s, return_norm=%s, scene_pc_mode=%s, use_cached_keypoints=%s, cache_only_mode=%s",
            self.mode, self.rot_type, self.trans_anchor,
            self.batch_size, self.num_workers, self.pin_memory, self.prefetch_factor, self.persistent_workers,
            self.use_local_pose_only, self.return_norm, self.scene_pc_return_mode, self.use_cached_keypoints,
            self.cache_only_mode,
        )
        if not self.use_cached_keypoints and not self._cache_usage_logged:
            logger.info("HandEncoderDataModule: cache disabled (use_cached_keypoints=False); computing keypoints on the fly.")
            self._cache_usage_logged = True
        
        if self.cache_only_mode:
            logger.info("HandEncoderDataModule: CACHE-ONLY MODE - no base dataset dependency")

        common_args = dict(
            hand_model=hm, canonical_link_to_unique=c['link_to_unique'],
            finger_ids=c['finger_ids'], joint_type_ids=c['joint_type_ids'],
            edge_index=c['edge_index'], edge_type=c['edge_type'],
            edge_rest_lengths=c['edge_rest_lengths'], scale=self.hand_scale,
            norm_xyz_bounds=self._clone_norm_bounds(), use_local_pose_only=self.use_local_pose_only,
            return_norm=self.return_norm, scene_pc_return_mode=self.scene_pc_return_mode
        )

        # Train
        if stage in (None, 'fit') and self.train_dataset is None:
            ph = self.train_phase or self.dataset_kwargs.get('phase', 'all')
            
            if self.cache_only_mode:
                # Cache-only mode: use cache as the primary dataset
                cache = self._make_cached_dataset(ph)
                if len(cache) == 0:
                    raise RuntimeError(f"Training cache empty for phase {ph}.")
                cache_anchor = cache.meta.get('stored_anchor') if hasattr(cache, 'meta') else None
                self.train_dataset = _HandEncoderPreparedDataset(
                    cache, cache_dataset=None, cached_stored_anchor=cache_anchor, **common_args
                )
                logger.info("Train dataset (cache-only): phase=%s, size=%d", ph, len(cache))
            else:
                # Normal mode: base + optional cache
                base = self._make_base_dataset(ph)
                if len(base) == 0:
                    raise RuntimeError(f"Training dataset empty for phase {ph}. Check split files/assets.")
                cache = self._make_cached_dataset(ph) if self.use_cached_keypoints else None
                cache_anchor = None
                if cache is not None:
                    try:
                        cache_anchor = cache.meta.get('stored_anchor')
                    except Exception:
                        cache_anchor = None
                self.train_dataset = _HandEncoderPreparedDataset(
                    base, cache_dataset=cache, cached_stored_anchor=cache_anchor, **common_args
                )
                logger.info("Train dataset: phase=%s, size=%d", ph, len(base))
            
            logger.info(
                "Train DataLoader: batch_size=%d, num_workers=%d, pin_memory=%s, prefetch_factor=%d, "
                "persistent_workers=%s, drop_last=%s",
                self.batch_size, self.num_workers, self.pin_memory, self.prefetch_factor,
                self.persistent_workers, (len(self.train_dataset) > self.batch_size),
            )

        # Validation
        if stage in (None, 'fit', 'validate') and self.val_phase and self.val_dataset is None:
            if self.cache_only_mode:
                cache = self._make_cached_dataset(self.val_phase)
                cache_anchor = cache.meta.get('stored_anchor') if hasattr(cache, 'meta') else None
                self.val_dataset = _HandEncoderPreparedDataset(
                    cache, cache_dataset=None, cached_stored_anchor=cache_anchor, **common_args
                )
                logger.info("Val dataset (cache-only): phase=%s, size=%d", self.val_phase, len(cache))
            else:
                base = self._make_base_dataset(self.val_phase)
                cache = self._make_cached_dataset(self.val_phase) if self.use_cached_keypoints else None
                cache_anchor = None
                if cache is not None:
                    try:
                        cache_anchor = cache.meta.get('stored_anchor')
                    except Exception:
                        cache_anchor = None
                self.val_dataset = _HandEncoderPreparedDataset(
                    base, cache_dataset=cache, cached_stored_anchor=cache_anchor, **common_args
                )
                logger.info("Val dataset: phase=%s, size=%d", self.val_phase, len(base))

        # Test
        if stage in (None, 'test') and self.test_phase and self.test_dataset is None:
            if self.cache_only_mode:
                cache = self._make_cached_dataset(self.test_phase)
                cache_anchor = cache.meta.get('stored_anchor') if hasattr(cache, 'meta') else None
                self.test_dataset = _HandEncoderPreparedDataset(
                    cache, cache_dataset=None, cached_stored_anchor=cache_anchor, **common_args
                )
                logger.info("Test dataset (cache-only): phase=%s, size=%d", self.test_phase, len(cache))
            else:
                base = self._make_base_dataset(self.test_phase)
                cache = self._make_cached_dataset(self.test_phase) if self.use_cached_keypoints else None
                cache_anchor = None
                if cache is not None:
                    try:
                        cache_anchor = cache.meta.get('stored_anchor')
                    except Exception:
                        cache_anchor = None
                self.test_dataset = _HandEncoderPreparedDataset(
                    base, cache_dataset=cache, cached_stored_anchor=cache_anchor, **common_args
                )
                logger.info("Test dataset: phase=%s, size=%d", self.test_phase, len(base))

    def get_graph_constants(self) -> Dict[str, torch.Tensor]:
        if self._prepared_constants is None:
            raise RuntimeError("Setup must be called before accessing graph constants.")
            
        c = self._prepared_constants
        consts = {
            k: c[k].detach().cpu() 
            for k in ['finger_ids', 'joint_type_ids', 'edge_index', 'edge_type', 'edge_rest_lengths', 
                      'template_xyz', 'canonical_xyz', 'approx_std']
        }
        
        if self._norm_xyz_bounds:
            consts['norm_min'] = self._norm_xyz_bounds[0].detach().cpu().view(1, 1, 3)
            consts['norm_max'] = self._norm_xyz_bounds[1].detach().cpu().view(1, 1, 3)

        consts['rigid_groups'] = [
            (g.clone().detach().long() if isinstance(g, torch.Tensor) else torch.tensor(g, dtype=torch.long))
            for g in c.get('rigid_groups', []) if g is not None and len(g) > 0
        ]
        
        return consts

    def _dataloader(self, ds: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=_collate_for_hand_encoder,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=(shuffle and len(ds) > self.batch_size)
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.val_dataset, shuffle=False) if self.val_dataset else None

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.test_dataset, shuffle=False) if self.test_dataset else None
