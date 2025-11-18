import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore

# Optional omegaconf support for DictConfig-based configuration
try:  # pragma: no cover
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore

from .MyDexGraspNet import MyDexGraspNet
from .MyBodexShadow import MyBodexShadow
from .CachedHandKeypointDataset import CachedHandKeypointDataset, HDF5HandKeypointDataset
from utils.shadown_hand_model import HandModel

logger = logging.getLogger(__name__)


FINGER_PREFIX_MAP = {
    'th': 0,
    'ff': 1,
    'mf': 2,
    'rf': 3,
    'lf': 4,
}

JOINT_TYPE_ORDER = [
    'knuckle', 'proximal', 'middle', 'distal', 'tip',
    'metacarpal', 'base', 'hub', 'palm', 'wrist', 'forearm'
]
JOINT_TYPE_TO_ID = {name: i for i, name in enumerate(JOINT_TYPE_ORDER)}
MJCF_ANCHOR_MARGIN = 0.2


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


FINGER_CHAINS = {
    'ff': ['ffknuckle', 'ffproximal', 'ffmiddle', 'ffdistal', 'fftip'],
    'mf': ['mfknuckle', 'mfproximal', 'mfmiddle', 'mfdistal', 'mftip'],
    'rf': ['rfknuckle', 'rfproximal', 'rfmiddle', 'rfdistal', 'rftip'],
    'lf': ['lfmetacarpal', 'lfknuckle', 'lfproximal', 'lfmiddle', 'lfdistal', 'lftip'],
    'th': ['thbase', 'thproximal', 'thhub', 'thmiddle', 'thdistal', 'thtip'],
}
MIDDLE_DISTAL_PAIRS = [
    ('ffmiddle', 'ffdistal'),
    ('mfmiddle', 'mfdistal'),
    ('rfmiddle', 'rfdistal'),
    ('lfmiddle', 'lfdistal'),
    ('thmiddle', 'thdistal'),
]


def _build_unique_names_and_primary_map(
    link_to_unique: Dict[str, List[int]],
    link_order: List[str],
) -> Tuple[List[str], Dict[int, str]]:
    """
    Select a primary link name for each unique index using suffix-priority,
    so that e.g. '*middle' wins over '*proximal' when the same unique point
    is shared by multiple links. This matches the notebook behavior.
    """
    if not link_to_unique:
        return [], {}

    # Count unique points
    N = 0
    for idxs in link_to_unique.values():
        if not idxs:
            continue
        N = max(N, max(int(i) for i in idxs) + 1)

    # Build candidates per unique id
    unique_to_links: Dict[int, List[str]] = defaultdict(list)
    for lk, idxs in link_to_unique.items():
        for u in idxs:
            unique_to_links[int(u)].append(lk)

    # Suffix priority (lower is stronger)
    SUFFIX_PRIORITY = {
        'middle': 0,
        'proximal': 1,
        'knuckle': 2,
        'metacarpal': 3,
        'base': 4,
        'hub': 5,
        'palm': 6,
        'wrist': 7,
        'forearm': 8,
        'distal': 9,
        'tip': 10,
    }

    def suffix_priority(lk: str) -> int:
        for suf, pr in SUFFIX_PRIORITY.items():
            if lk.endswith(suf):
                return pr
        return 100

    # Choose primary by priority
    primary: Dict[int, str] = {}
    for u in range(N):
        cands = unique_to_links.get(u, [])
        if not cands:
            primary[u] = 'unknown'
        else:
            primary[u] = min(cands, key=suffix_priority)

    # Assign names with running counters
    names: List[Optional[str]] = [None] * N
    counters: Dict[str, int] = defaultdict(int)
    for u in range(N):
        lk = primary[u]
        k = counters[lk]
        names[u] = f"{lk}_{k}"
        counters[lk] += 1

    return [n if n is not None else f"u{i}" for i, n in enumerate(names)], primary


def _add_middle_distal_bridges_dynamic(link_to_unique: Dict[str, List[int]], edges: List[Tuple[int, int, int]]) -> None:
    """
    Add middle<->distal directed edges (type=3) per finger dynamically.
    - Supports ff/if prefix for index finger.
    - Picks the last index for lfmiddle/rfmiddle (e.g., *_middle_1) if present,
      otherwise falls back to the first.
    - Distal always uses its first index.
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

    middle_pick_mode = {
        'ffmiddle': 'last',
        'ifmiddle': 'last',
        'mfmiddle': 'last',
        'lfmiddle': 'last',
        'rfmiddle': 'last',
        'thmiddle': 'last',
    }

    for mids, dists in finger_sets:
        u = v = None
        for nm in mids:
            if nm in link_to_unique and len(link_to_unique[nm]) > 0:
                mode = middle_pick_mode.get(nm, 'first')
                u = pick(link_to_unique[nm], mode)
                if u is not None:
                    break
        for nm in dists:
            if nm in link_to_unique and len(link_to_unique[nm]) > 0:
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
    edges: List[Tuple[int, int, int]] = []
    for _, idxs in link_to_unique.items():
        if not idxs or len(idxs) < 2:
            continue
        seen = set()
        seq: List[int] = []
        for w in idxs:
            w = int(w)
            if w not in seen:
                seen.add(w)
                seq.append(w)
        for a, b in zip(seq[:-1], seq[1:]):
            if a != b:
                edges.append((a, b, 0))
                edges.append((b, a, 0))
    if use_palm_star and 'palm' in link_to_unique and len(link_to_unique['palm']) > 0:
        palm_root = int(link_to_unique['palm'][0])
        for _, chain in FINGER_CHAINS.items():
            target = None
            for lk in chain:
                if lk in link_to_unique and len(link_to_unique[lk]) > 0:
                    target = int(link_to_unique[lk][0])
                    break
            if target is not None and target != palm_root:
                edges.append((palm_root, target, 1))
                edges.append((target, palm_root, 1))
    if connect_forearm and ('forearm' in link_to_unique) and ('wrist' in link_to_unique):
        if len(link_to_unique['forearm']) > 0 and len(link_to_unique['wrist']) > 0:
            u = int(link_to_unique['forearm'][0])
            v = int(link_to_unique['wrist'][0])
            if u != v:
                edges.append((u, v, 2))
                edges.append((v, u, 2))
    if add_middle_distal:
        _add_middle_distal_bridges_dynamic(link_to_unique, edges)
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
    ifmeta0 = _get_idx_any(['ifmetacarpal', 'ffmetacarpal', 'ffknuckle'], 0)
    ffprox0 = _get_idx('ffproximal', 0)
    lfmeta0 = _get_idx('lfmetacarpal', 0)
    def _add_pair(u: Optional[int], v: Optional[int], t: int = 1) -> None:
        if u is not None and v is not None and u != v:
            edges.append((u, v, t))
            edges.append((v, u, t))
    _add_pair(palm0, ifmeta0, 1)
    _add_pair(palm0, lfmeta0, 1)
    _add_pair(palm0, thprox0, 1)
    _add_pair(ffprox0, thprox0, 1)
    remove_pairs = set()
    if palm0 is not None and ffprox0 is not None and palm0 != ffprox0:
        remove_pairs.add((palm0, ffprox0))
        remove_pairs.add((ffprox0, palm0))
    if ifmeta0 is not None and thprox0 is not None and ifmeta0 != thprox0:
        # If ifmeta0 aliases ffprox0 (due to shared unique index), do not remove ffprox0↔thprox0
        if not (ffprox0 is not None and ifmeta0 == ffprox0):
            remove_pairs.add((ifmeta0, thprox0))
            remove_pairs.add((thprox0, ifmeta0))
    if lfmeta0 is not None and thprox0 is not None and lfmeta0 != thprox0:
        remove_pairs.add((lfmeta0, thprox0))
        remove_pairs.add((thprox0, lfmeta0))
    if remove_pairs:
        edges = [e for e in edges if (e[0], e[1]) not in remove_pairs]
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    uniq = sorted(set(edges))
    edge_index = np.array([[u for (u, v, t) in uniq], [v for (u, v, t) in uniq]], dtype=np.int64)
    edge_type = np.array([t for (u, v, t) in uniq], dtype=np.int64)
    return edge_index, edge_type


def _build_rigid_groups_from_link_map(link_to_unique: Dict[str, List[int]]) -> List[List[int]]:
    """Each rigid body link forms one group of unique point indices."""

    groups: List[List[int]] = []
    for _, idxs in link_to_unique.items():
        if not idxs:
            continue
        seen: List[int] = []
        for raw_idx in idxs:
            idx = int(raw_idx)
            if idx not in seen:
                seen.append(idx)
        if seen:
            groups.append(seen)
    return groups


def _compute_rest_lengths(
    hand_model: HandModel,
    edge_index_t: torch.Tensor,
    q_rest: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        xyz_rest = hand_model.get_joint_keypoints_unique(q=q_rest)[0]
    i, j = edge_index_t[0], edge_index_t[1]
    diff = xyz_rest[:, i, :] - xyz_rest[:, j, :]
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9)
    return dist.squeeze(0)


class _HandEncoderPreparedDataset(Dataset):
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
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.hand_model = hand_model
        self.link_to_unique = canonical_link_to_unique
        self.finger_ids = finger_ids
        self.joint_type_ids = joint_type_ids
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_rest_lengths = edge_rest_lengths
        self.N = int(finger_ids.shape[0])
        self.scale = float(scale)
        self._norm_xyz_min: Optional[torch.Tensor] = None
        self._norm_xyz_max: Optional[torch.Tensor] = None
        self._norm_eps = 1e-6
        self.use_local_pose_only = bool(use_local_pose_only)
        if norm_xyz_bounds is not None:
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
        self.hand_model.update_kinematics(q)
        xyz = torch.zeros((self.N, 3), dtype=torch.float32, device=self.hand_model.device)
        filled = torch.zeros((self.N,), dtype=torch.bool, device=self.hand_model.device)
        for link_name in self.hand_model.joint_key_points:
            pts_local = self.hand_model.joint_key_points[link_name]
            if len(pts_local) == 0:
                continue
            mapped_name = getattr(self.hand_model, '_map_to_current_status_name', lambda x: x)(link_name) or link_name
            if mapped_name not in self.hand_model.current_status:
                continue
            kp = self.hand_model.current_status[mapped_name].transform_points(
                torch.tensor(pts_local, device=self.hand_model.device, dtype=torch.float32)
            ).expand(self.hand_model.batch_size, -1, -1)
            kp = torch.bmm(kp, self.hand_model.global_rotation.transpose(1, 2)) + self.hand_model.global_translation.unsqueeze(1)
            kp = kp * self.hand_model.scale
            idxs = self.link_to_unique.get(link_name, [])
            for k, u in enumerate(idxs):
                u = int(u)
                if 0 <= u < self.N and not filled[u]:
                    xyz[u] = kp[0, k]
                    filled[u] = True
        if not torch.all(filled):
            missing = (~filled).nonzero(as_tuple=False).view(-1)
            if missing.numel() > 0:
                xyz[missing] = 0.0
        return xyz

    @torch.no_grad()
    def _build_xyz_from_pose_local(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(device=self.hand_model.device, dtype=torch.float32)
        q_local = q.clone()
        q_local[:, :3] = 0.0
        rot_id = self._rotation_identity(q_local.device)
        rot_dim = rot_id.shape[0]
        rot_start = max(0, q_local.shape[1] - rot_dim)
        if rot_start + rot_dim <= q_local.shape[1]:
            q_local[:, rot_start:rot_start + rot_dim] = rot_id.view(1, -1)
        self.hand_model.update_kinematics(q_local)
        xyz = torch.zeros((self.N, 3), dtype=torch.float32, device=self.hand_model.device)
        filled = torch.zeros((self.N,), dtype=torch.bool, device=self.hand_model.device)
        for link_name in self.hand_model.joint_key_points:
            pts_local = self.hand_model.joint_key_points[link_name]
            if len(pts_local) == 0:
                continue
            mapped_name = getattr(self.hand_model, '_map_to_current_status_name', lambda x: x)(link_name) or link_name
            if mapped_name not in self.hand_model.current_status:
                continue
            kp = self.hand_model.current_status[mapped_name].transform_points(
                torch.tensor(pts_local, device=self.hand_model.device, dtype=torch.float32)
            ).expand(self.hand_model.batch_size, -1, -1)
            kp = kp * self.hand_model.scale
            idxs = self.link_to_unique.get(link_name, [])
            for k, u in enumerate(idxs):
                u = int(u)
                if 0 <= u < self.N and not filled[u]:
                    xyz[u] = kp[0, k]
                    filled[u] = True
        if not torch.all(filled):
            missing = (~filled).nonzero(as_tuple=False).view(-1)
            if missing.numel() > 0:
                xyz[missing] = 0.0
        # Anchor translation should still be reflected even when rigid pose is reset
        anchor_translation = getattr(self.hand_model, 'global_translation', None)
        if anchor_translation is not None and anchor_translation.numel() >= 3:
            xyz = xyz + anchor_translation[0].view(1, 3)
        return xyz

    def _normalize_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        if self._norm_xyz_min is None or self._norm_xyz_max is None:
            return xyz.clone()
        denom = torch.clamp(self._norm_xyz_max - self._norm_xyz_min, min=self._norm_eps)
        norm = (xyz - self._norm_xyz_min) / denom
        norm = norm * 2.0 - 1.0
        return torch.clamp(norm, -1.0, 1.0)

    def _denormalize_xyz(self, norm_xyz: torch.Tensor) -> torch.Tensor:
        if self._norm_xyz_min is None or self._norm_xyz_max is None:
            return norm_xyz.clone()
        norm = torch.clamp(norm_xyz, -1.0, 1.0)
        unscaled = (norm + 1.0) * 0.5
        denom = torch.clamp(self._norm_xyz_max - self._norm_xyz_min, min=self._norm_eps)
        xyz = unscaled * denom + self._norm_xyz_min
        return xyz

    def _xyz_from_cache(self, xyz_local: Any, se3: Any) -> torch.Tensor:
        if not isinstance(xyz_local, torch.Tensor):
            xyz_local_t = torch.tensor(xyz_local, dtype=torch.float32)
        else:
            xyz_local_t = xyz_local.to(torch.float32)
        if isinstance(se3, torch.Tensor):
            se3_t = se3.to(torch.float32)
        else:
            se3_t = torch.tensor(se3, dtype=torch.float32)
        rot = se3_t[:3, :3]
        trans = se3_t[:3, 3]
        world = torch.matmul(xyz_local_t, rot.transpose(0, 1)) + trans.unsqueeze(0)
        return world

    @torch.no_grad()
    def _scene_pc_world_to_local(
        self,
        scene_pc: torch.Tensor,
        R_world: torch.Tensor,
        t_world: torch.Tensor,
        t_local: torch.Tensor,
    ) -> torch.Tensor:
        if scene_pc.ndim != 2 or scene_pc.shape[1] < 3:
            return scene_pc
        if R_world.ndim != 2 or R_world.shape != (3, 3):
            return scene_pc
        if t_world.ndim != 1 or t_world.shape[0] != 3:
            return scene_pc
        if t_local.ndim != 1 or t_local.shape[0] != 3:
            return scene_pc
        sp = scene_pc.to(dtype=torch.float32)
        pts = sp[:, :3]
        t_w = t_world.view(1, 3).to(dtype=torch.float32)
        t_l = t_local.view(1, 3).to(dtype=torch.float32)
        R = R_world.to(dtype=torch.float32)
        # World (object-centric) -> local anchor frame used by _build_xyz_from_pose_local
        # x_local = (x_world - t_world) @ R_world + t_local
        local = (pts - t_w) @ R + t_l
        if scene_pc.shape[1] > 3:
            tail = scene_pc[:, 3:].to(dtype=scene_pc.dtype)
            return torch.cat([local.to(dtype=scene_pc.dtype), tail], dim=1)
        return local.to(dtype=scene_pc.dtype)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        hand_pose = sample['hand_model_pose']
        if not isinstance(hand_pose, torch.Tensor):
            hand_pose_t = torch.tensor(hand_pose, dtype=torch.float32)
        else:
            hand_pose_t = hand_pose.to(torch.float32)

        cached_xyz = sample.get('cached_xyz_local')
        cached_se3 = sample.get('cached_se3')

        scene_pc_raw = sample.get('scene_pc')
        scene_pc: torch.Tensor
        if isinstance(scene_pc_raw, torch.Tensor):
            scene_pc = scene_pc_raw.to(torch.float32)
        elif isinstance(scene_pc_raw, np.ndarray):
            arr = scene_pc_raw
            if arr.ndim >= 2 and arr.shape[1] >= 3:
                scene_pc = torch.from_numpy(arr.astype(np.float32))
            else:
                scene_pc = torch.zeros((0, 3), dtype=torch.float32)
        else:
            scene_pc = torch.zeros((0, 3), dtype=torch.float32)

        xyz_world: Optional[torch.Tensor] = None

        if self.use_local_pose_only:
            q = hand_pose_t.unsqueeze(0).to(self.hand_model.device)
            # First compute world-frame keypoints to obtain the hand's global SE(3)
            _ = self._build_xyz_from_pose(q).to(torch.float32)
            R_world = getattr(self.hand_model, "global_rotation", None)
            t_world = getattr(self.hand_model, "global_translation", None)
            # Then build local xyz with rigid pose reset (this call overwrites global_translation to t_local)
            xyz = self._build_xyz_from_pose_local(q).to(torch.float32)
            t_local = getattr(self.hand_model, "global_translation", None)
            if (
                scene_pc.numel() > 0
                and isinstance(R_world, torch.Tensor)
                and isinstance(t_world, torch.Tensor)
                and isinstance(t_local, torch.Tensor)
                and R_world.shape[0] > 0
                and t_world.shape[0] > 0
                and t_local.shape[0] > 0
            ):
                scene_pc = self._scene_pc_world_to_local(scene_pc, R_world[0], t_world[0], t_local[0])
        else:
            xyz: Optional[torch.Tensor] = None
            if cached_xyz is not None and cached_se3 is not None:
                xyz = self._xyz_from_cache(cached_xyz, cached_se3).to(torch.float32)
            elif cached_xyz is not None:
                xyz = (cached_xyz if isinstance(cached_xyz, torch.Tensor) else torch.tensor(cached_xyz, dtype=torch.float32)).to(torch.float32)
            if xyz is None:
                q = hand_pose_t.unsqueeze(0).to(self.hand_model.device)
                xyz = self._build_xyz_from_pose(q).to(torch.float32)
        norm_xyz = self._normalize_xyz(xyz)
        norm_pose = sample.get('norm_pose')
        if norm_pose is not None:
            if isinstance(norm_pose, np.ndarray):
                norm_pose = torch.from_numpy(norm_pose)
            if isinstance(norm_pose, torch.Tensor):
                norm_pose = norm_pose.cpu()
        hand_pose_out = hand_pose_t
        if self.use_local_pose_only:
            hp = hand_pose_t.clone()
            if hp.numel() >= 6:  # ensure pose has rigid part
                hp[:3] = 0.0
                rot_id = self._rotation_identity(hp.device)
                tail_len = rot_id.shape[0]
                rot_start = max(0, hp.shape[0] - tail_len)
                if rot_start + tail_len <= hp.shape[0]:
                    hp[rot_start:rot_start + tail_len] = rot_id.to(hp.dtype).cpu()
            hand_pose_out = hp.cpu()
        out = {
            'xyz': xyz.cpu(),
            'norm_xyz': norm_xyz.cpu(),
            'hand_model_pose': hand_pose_out,
            'norm_pose': norm_pose,
            'obj_code': sample.get('obj_code'),
            'scene_id': sample.get('scene_id'),
        }
        if scene_pc.numel() > 0:
            out['scene_pc'] = scene_pc.cpu()
        return out


def _collate_for_hand_encoder(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}
    xyz = torch.stack([b['xyz'] for b in batch], dim=0)
    norm_xyz = torch.stack([b['norm_xyz'] for b in batch], dim=0)
    hand_model_pose = torch.stack([b['hand_model_pose'] for b in batch], dim=0)
    norm_pose = None
    if all(b.get('norm_pose') is not None for b in batch):
        norm_pose = torch.stack([b['norm_pose'] for b in batch], dim=0)
    obj_code = [b.get('obj_code') for b in batch]
    scene_id = [b.get('scene_id') for b in batch]
    scene_pcs_raw = [b.get('scene_pc') for b in batch]
    scene_pc = None
    if any(isinstance(pc, (torch.Tensor, np.ndarray)) for pc in scene_pcs_raw):
        scene_list: List[Optional[torch.Tensor]] = []
        max_n = 0
        feat_dim = None
        ref_tensor: Optional[torch.Tensor] = None
        for pc in scene_pcs_raw:
            tensor_pc: Optional[torch.Tensor] = None
            if isinstance(pc, torch.Tensor):
                tensor_pc = pc
            elif isinstance(pc, np.ndarray):
                try:
                    tensor_pc = torch.from_numpy(pc)
                except Exception:
                    tensor_pc = None
            scene_list.append(tensor_pc)
            if tensor_pc is not None and tensor_pc.ndim == 2:
                if ref_tensor is None:
                    ref_tensor = tensor_pc
                max_n = max(max_n, int(tensor_pc.shape[0]))
                if feat_dim is None and tensor_pc.shape[1] > 0:
                    feat_dim = int(tensor_pc.shape[1])
        if ref_tensor is not None and feat_dim is not None and max_n > 0:
            dtype = ref_tensor.dtype
            device = ref_tensor.device
            batch_size = len(scene_list)
            scene_pc = torch.zeros((batch_size, max_n, feat_dim), dtype=dtype, device=device)
            for i, pc in enumerate(scene_list):
                if pc is None or pc.ndim != 2:
                    continue
                n_i = min(int(pc.shape[0]), max_n)
                scene_pc[i, :n_i, :pc.shape[1]] = pc[:n_i]
    result: Dict[str, Any] = {
        'xyz': xyz,
        'hand_model_pose': hand_model_pose,
        'norm_xyz': norm_xyz,
        'norm_pose': norm_pose,
        'obj_code': obj_code,
        'scene_id': scene_id,
    }
    if scene_pc is not None:
        result['scene_pc'] = scene_pc
    return result


class HandEncoderDataModule(pl.LightningDataModule if pl is not None else object):
    def __init__(
        self,
        data_cfg: "DictConfig" = None,
        # Direct params for Hydra instantiate compatibility (optional)
        name: Optional[str] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        rot_type: Optional[str] = None,
        trans_anchor: Optional[str] = None,
        hand_scale: Optional[float] = None,
        urdf_assets_meta_path: Optional[str] = None,
        use_palm_star: Optional[bool] = None,
        connect_forearm: Optional[bool] = None,
        add_middle_distal: Optional[bool] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        use_local_pose_only: Optional[bool] = None,
        # Optional split phases
        train_phase: Optional[str] = None,
        val_phase: Optional[str] = None,
        test_phase: Optional[str] = None,
    ) -> None:
        """
        Initialize from a Hydra/OmegaConf DictConfig or direct keyword args.

        Expected keys (see configs/datamodule/handencoder_*.yaml):
          - mode, batch_size, num_workers, pin_memory
          - rot_type, trans_anchor, hand_scale, urdf_assets_meta_path
          - use_palm_star, connect_forearm, add_middle_distal
          - dataset_kwargs (mapping passed to underlying dataset)
          - optional train/val/test phases
        """
        super().__init__()

        # Helper to prioritize direct kwargs over DictConfig
        params = {
            'name': name,
            'mode': mode,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'rot_type': rot_type,
            'trans_anchor': trans_anchor,
            'hand_scale': hand_scale,
            'urdf_assets_meta_path': urdf_assets_meta_path,
            'use_palm_star': use_palm_star,
            'connect_forearm': connect_forearm,
            'add_middle_distal': add_middle_distal,
            'dataset_kwargs': dataset_kwargs,
            'use_local_pose_only': use_local_pose_only,
            'train_phase': train_phase,
            'val_phase': val_phase,
            'test_phase': test_phase,
        }

        def _get(key: str, default: Any = None):
            if key in params and params[key] is not None:
                return params[key]
            return getattr(data_cfg, key, default) if data_cfg is not None else default

        # Basic settings
        self.name = _get('name', None)
        self.mode = str(_get('mode', 'dexgraspnet') or 'dexgraspnet').lower()
        self.batch_size = int(_get('batch_size', 32) or 32)
        self.num_workers = int(_get('num_workers', 4) or 4)
        self.pin_memory = bool(_get('pin_memory', True) if _get('pin_memory', True) is not None else True)

        # Dataset kwargs may be nested; convert to plain dict if OmegaConf
        dk = _get('dataset_kwargs', {})
        if OmegaConf is not None and isinstance(dk, DictConfig):  # type: ignore
            dk = OmegaConf.to_container(dk, resolve=True)
        self.dataset_kwargs = dict(dk or {})
        self.use_cached_keypoints = bool(_get('use_cached_keypoints', False) or False)
        self.cache_root = _get('cache_root', None)
        # Respect both keys: cache_max_shards_in_memory (preferred) and cache_max_shards (legacy)
        _cmsi = _get('cache_max_shards_in_memory', None)
        _cms = _get('cache_max_shards', None)

        def _parse_cache_max(v: Any):
            if v is None:
                return None
            if isinstance(v, str):
                key = v.strip().lower()
                if key in ('all', 'full', 'infinite', 'inf', 'max'):
                    return 'all'
                try:
                    return int(v)
                except Exception:
                    return None
            try:
                iv = int(v)
                return iv
            except Exception:
                return None

        parsed = _parse_cache_max(_cmsi)
        if parsed is None:
            parsed = _parse_cache_max(_cms)
        self.cache_max_shards = parsed if parsed is not None else 2
        self._cache_mode_dir: Optional[Path] = None
        self._cache_meta_checked = False
        # Cached dataset options
        _cptr = _get('cache_preload_to_ram', None)
        self.cache_preload_to_ram = bool(_cptr) if (_cptr is not None) else False
        _csp = _get('cache_show_progress', None)
        self.cache_show_progress = bool(_csp) if (_csp is not None) else True
        # DataLoader performance knobs (safe defaults)
        _pf = _get('prefetch_factor', None)
        self.prefetch_factor = int(_pf) if (_pf is not None) else 1
        _pw = _get('persistent_workers', None)
        self.persistent_workers = bool(_pw) if (_pw is not None) else (self.num_workers > 0)
        norm_stats_path = self.dataset_kwargs.get('normalization_stats_path')
        if norm_stats_path is not None:
            self.normalization_stats_path = str(norm_stats_path)
            self.dataset_kwargs['normalization_stats_path'] = self.normalization_stats_path
        else:
            self.normalization_stats_path = None
        # Optional stats path for xyz normalization (use_local_pose_only 专用，可按 anchor 区分)
        self.norm_xyz_stats_path = _get('norm_xyz_stats_path', None)

        # Hand/rotation config
        self.rot_type = str(_get('rot_type', 'quat') or 'quat')
        self.trans_anchor = str(_get('trans_anchor', 'palm_center') or 'palm_center')
        self.hand_scale = float(_get('hand_scale', 1.0) or 1.0)
        self.urdf_assets_meta_path = _get('urdf_assets_meta_path', None)

        # Graph options
        self.use_palm_star = bool(_get('use_palm_star', False) or False)
        self.connect_forearm = bool(_get('connect_forearm', True) or True)
        self.add_middle_distal = bool(_get('add_middle_distal', True) or True)

        # Optional phases
        self.train_phase = _get('train_phase', None)
        self.val_phase = _get('val_phase', None)
        self.test_phase = _get('test_phase', None)

        # Local-only pose option
        self.use_local_pose_only = bool(_get('use_local_pose_only', False) or False)

        # Datasets placeholders
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self._prepared_constants: Optional[Dict[str, Any]] = None
        self._norm_xyz_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # Track setup to avoid expensive re-initialization
        self._setup_done_stages: set = set()

    def prepare_data(self) -> None:
        pass

    def _load_normalization_stats(self) -> Optional[Dict[str, Any]]:
        base_path = self.normalization_stats_path
        if not base_path:
            logger.warning("HandEncoderDataModule: normalization_stats_path is not set; norm_xyz will be raw xyz.")
            return None

        if base_path.endswith('.json'):
            json_path = base_path
            pt_path = base_path[:-5] + '.pt'
        elif base_path.endswith('.pt'):
            pt_path = base_path
            json_path = base_path[:-3] + '.json'
        else:
            json_path = base_path + '.json'
            pt_path = base_path + '.pt'

        stats_obj: Optional[Dict[str, Any]] = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    stats_obj = json.load(f)
            except Exception as exc:
                logger.warning("HandEncoderDataModule: failed to read normalization JSON %s: %s", json_path, exc)
        if stats_obj is None and os.path.exists(pt_path):
            try:
                stats_obj = torch.load(pt_path, map_location='cpu')
            except Exception as exc:
                logger.warning("HandEncoderDataModule: failed to read normalization PT %s: %s", pt_path, exc)
        if stats_obj is None:
            logger.warning("HandEncoderDataModule: normalization stats not found at base path %s", base_path)
        return stats_obj

    @staticmethod
    def _pick_stat_bounds(block: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        mn = block.get('min_with_margin', block.get('min'))
        mx = block.get('max_with_margin', block.get('max'))
        return np.asarray(mn, dtype=np.float32), np.asarray(mx, dtype=np.float32)

    def _prepare_norm_xyz_bounds(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        stats_obj = self._load_normalization_stats()
        if stats_obj is None:
            return None
        if 'hand_trans' not in stats_obj:
            logger.warning("HandEncoderDataModule: normalization stats missing 'hand_trans'; norm_xyz will be raw xyz.")
            return None

        lo, hi = self._pick_stat_bounds(stats_obj['hand_trans'])
        lo = lo.reshape(-1)
        hi = hi.reshape(-1)
        if lo.shape[0] != 3 or hi.shape[0] != 3:
            logger.warning("HandEncoderDataModule: expected 3D hand_trans bounds, got %s/%s", lo.shape, hi.shape)
            return None

        diff = hi - lo
        shrink_mask = diff > (2 * MJCF_ANCHOR_MARGIN)
        shrunk_lo = np.where(shrink_mask, lo + MJCF_ANCHOR_MARGIN, lo)
        shrunk_hi = np.where(shrink_mask, hi - MJCF_ANCHOR_MARGIN, hi)

        lo_t = torch.from_numpy(shrunk_lo.astype(np.float32)).to(device=device, dtype=torch.float32).view(1, 3)
        hi_t = torch.from_numpy(shrunk_hi.astype(np.float32)).to(device=device, dtype=torch.float32).view(1, 3)
        logger.info("HandEncoderDataModule: prepared norm_xyz bounds (mjcf shrink) min=%s max=%s", shrunk_lo.tolist(), shrunk_hi.tolist())
        return lo_t, hi_t

    def _load_norm_xyz_stats_local(self) -> Optional[Dict[str, Any]]:
        """Load xyz statistics for use_local_pose_only, optionally anchor-specific.

        优先使用 norm_xyz_stats_path；若未设置，则默认从数据集根目录下的
        "norm_xyz_stats_use_local_anchor-{anchor}.json" 读取，其中 anchor 取自
        self.trans_anchor。
        """
        base_path = self.norm_xyz_stats_path
        if not base_path:
            asset_dir = self.dataset_kwargs.get('asset_dir')
            if not asset_dir:
                logger.warning("HandEncoderDataModule: dataset_kwargs.asset_dir 未设置，无法推断 norm_xyz 统计文件路径。")
                return None
            if not os.path.isabs(asset_dir):
                proj_root = Path(__file__).resolve().parents[2]
                asset_dir = str((proj_root / asset_dir).resolve())
            base_path = os.path.join(asset_dir, f"norm_xyz_stats_use_local_anchor-{self.trans_anchor}")

        if base_path.endswith('.json'):
            json_path = base_path
            pt_path = base_path[:-5] + '.pt'
        elif base_path.endswith('.pt'):
            pt_path = base_path
            json_path = base_path[:-3] + '.json'
        else:
            json_path = base_path + '.json'
            pt_path = base_path + '.pt'

        stats_obj: Optional[Dict[str, Any]] = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    stats_obj = json.load(f)
            except Exception as exc:
                logger.warning("HandEncoderDataModule: failed to read norm_xyz JSON %s: %s", json_path, exc)
        if stats_obj is None and os.path.exists(pt_path):
            try:
                stats_obj = torch.load(pt_path, map_location='cpu')
            except Exception as exc:
                logger.warning("HandEncoderDataModule: failed to read norm_xyz PT %s: %s", pt_path, exc)
        if stats_obj is None:
            logger.warning("HandEncoderDataModule: norm_xyz stats not found at base path %s", base_path)
        return stats_obj

    def _prepare_norm_xyz_bounds_local(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Prepare xyz normalization bounds for use_local_pose_only based on xyz stats.

        该逻辑假定 xyz 分布统计是针对 use_local_pose_only=true 且特定 anchor
        下的 joint keypoints，在此基础上增加一定比例的 margin，避免采样不完全
        带来的截断。
        """
        stats_obj = self._load_norm_xyz_stats_local()
        if stats_obj is None:
            return None
        block = stats_obj.get('stats') or {}
        mins = block.get('min')
        maxs = block.get('max')
        if mins is None or maxs is None:
            logger.warning("HandEncoderDataModule: norm_xyz stats missing 'min'/'max'; norm_xyz 将退回 raw xyz。")
            return None
        lo = np.asarray(mins, dtype=np.float32).reshape(-1)
        hi = np.asarray(maxs, dtype=np.float32).reshape(-1)
        if lo.shape[0] != 3 or hi.shape[0] != 3:
            logger.warning("HandEncoderDataModule: expected 3D norm_xyz bounds, got %s/%s", lo.shape, hi.shape)
            return None

        script_args = stats_obj.get('script_args') or {}
        max_grasps_sampled = int(script_args.get('max_grasps', 0) or 0)
        margin_ratio = 0.05
        # 如果统计脚本使用了 max_grasps 进行抽样，则采用更大的 margin，
        # 以降低尾部截断风险。
        if max_grasps_sampled > 0:
            margin_ratio = 0.25

        diff = hi - lo
        margin = diff * float(margin_ratio)
        expanded_lo = lo - margin
        expanded_hi = hi + margin

        lo_t = torch.from_numpy(expanded_lo.astype(np.float32)).to(device=device, dtype=torch.float32).view(1, 3)
        hi_t = torch.from_numpy(expanded_hi.astype(np.float32)).to(device=device, dtype=torch.float32).view(1, 3)
        logger.info(
            "HandEncoderDataModule: prepared norm_xyz bounds (local, anchor=%s, margin=%.3f) min=%s max=%s",
            self.trans_anchor,
            margin_ratio,
            expanded_lo.tolist(),
            expanded_hi.tolist(),
        )
        return lo_t, hi_t

    def _ensure_norm_xyz_bounds(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._norm_xyz_bounds is None:
            if self.use_local_pose_only:
                self._norm_xyz_bounds = self._prepare_norm_xyz_bounds_local(device)
                if self._norm_xyz_bounds is None:
                    self._norm_xyz_bounds = self._prepare_norm_xyz_bounds(device)
            else:
                self._norm_xyz_bounds = self._prepare_norm_xyz_bounds(device)
        return self._norm_xyz_bounds

    def _clone_norm_bounds(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._norm_xyz_bounds is None:
            return None
        lo, hi = self._norm_xyz_bounds
        return (lo.clone(), hi.clone())

    def _load_urdf_assets(self) -> Tuple[str, str]:
        proj_root = Path(__file__).resolve().parents[2]
        meta_path = Path(self.urdf_assets_meta_path) if self.urdf_assets_meta_path else (proj_root / 'assets' / 'urdf' / 'urdf_assets_meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        urdf_path = proj_root / meta['urdf_path']['shadowhand']
        meshes_path = proj_root / meta['meshes_path']['shadowhand']
        return str(urdf_path), str(meshes_path)

    def _make_base_dataset(self, phase: str) -> Dataset:
        if self.use_cached_keypoints:
            return self._make_cached_dataset(phase)

        kwargs = dict(self.dataset_kwargs)
        kwargs['phase'] = phase
        kwargs['rot_type'] = self.rot_type
        kwargs['trans_anchor'] = self.trans_anchor
        if self.mode in ('dexgraspnet', 'dex'):
            return MyDexGraspNet(**kwargs)
        elif self.mode in ('bodexshadow', 'bodex', 'bodexhshadow'):
            return MyBodexShadow(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _make_cached_dataset(self, phase: str) -> Dataset:
        cache_dir = self._resolve_cache_dir()
        # Prefer HDF5 cache if present: <phase>_cache.h5
        h5_path = cache_dir / f"{phase}_cache.h5"
        if h5_path.exists():
            dataset = HDF5HandKeypointDataset(
                file_path=str(h5_path),
                phase=phase,
                show_progress=self.cache_show_progress,
            )
        else:
            # Backward-compat: if deprecated cache_preload_to_ram is True, map to 'all'
            max_shards_arg: Any = self.cache_max_shards
            if getattr(self, 'cache_preload_to_ram', False):
                logger.warning("cache_preload_to_ram is deprecated; use cache_max_shards_in_memory: 'all'. Applying preload.")
                max_shards_arg = 'all'
            dataset = CachedHandKeypointDataset(
                cache_dir=str(cache_dir),
                phase=phase,
                max_shards_in_memory=max_shards_arg,
                show_progress=self.cache_show_progress,
            )
        if not self._cache_meta_checked:
            meta = dataset.meta
            stored_anchor = meta.get('stored_anchor')
            logger.info(
                "HandEncoderDataModule: cache meta=%s",
                {k: meta.get(k) for k in ('stored_anchor', 'source_anchor', 'hand_scale')},
            )
            if stored_anchor is not None and stored_anchor.lower() != 'base':
                logger.warning(
                    "Cached keypoints stored with anchor '%s'; expected 'base'.",
                    stored_anchor,
                )
            self._cache_meta_checked = True
        return dataset

    def _resolve_cache_dir(self) -> Path:
        if self._cache_mode_dir is not None:
            return self._cache_mode_dir
        if not self.cache_root:
            raise ValueError("cache_root must be set when use_cached_keypoints=True")
        base = Path(self.cache_root)
        if not base.is_absolute():
            base = Path.cwd() / base
        mode_dir = base / (self.mode or 'default')
        if not mode_dir.exists():
            if base.exists():
                mode_dir = base
            else:
                raise FileNotFoundError(f"Cache root directory not found: {base}")
        if not mode_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {mode_dir}")
        self._cache_mode_dir = mode_dir
        logger.info("HandEncoderDataModule: using cached keypoints from %s", mode_dir)
        return mode_dir

    @torch.no_grad()
    def _build_canonical(self, hand_model: HandModel) -> Dict[str, Any]:
        if self.rot_type == 'quat':
            rot_tail = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=hand_model.device)
        elif self.rot_type == 'r6d':
            rot_tail = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=hand_model.device)
        elif self.rot_type in ('euler', 'axis'):
            rot_tail = torch.zeros((1, 3), device=hand_model.device)
        else:
            raise ValueError(f"Unsupported rot_type: {self.rot_type}")
        POSE_DIM = 3 + 24 + int(rot_tail.shape[1])
        q_rest = torch.zeros((1, POSE_DIM), device=hand_model.device)
        q_rest[:, 27:27 + rot_tail.shape[1]] = rot_tail
        xyz_u, link_to_unique = hand_model.get_joint_keypoints_unique(q=q_rest)
        link_order = [lk for lk in hand_model.joint_key_points.keys() if lk in link_to_unique]
        names, primary = _build_unique_names_and_primary_map(link_to_unique, link_order)
        finger_ids_np, joint_ids_np, num_fingers, num_joint_types = _build_ids_from_primary(primary, len(names))
        edge_index_np, edge_type_np = _build_rigid_edges_unique(
            link_to_unique,
            use_palm_star=self.use_palm_star,
            connect_forearm=self.connect_forearm,
            add_middle_distal=self.add_middle_distal,
        )
        device = hand_model.device
        edge_index_t = torch.from_numpy(edge_index_np).to(device=device, dtype=torch.long)
        edge_type_t = torch.from_numpy(edge_type_np).to(device=device, dtype=torch.long)
        rest_lengths_t = _compute_rest_lengths(hand_model, edge_index_t, q_rest).to(dtype=torch.float32)
        finger_ids_t = torch.from_numpy(finger_ids_np).to(device=device, dtype=torch.long)
        joint_ids_t = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
        template_xyz = xyz_u.squeeze(0).to(device=device, dtype=torch.float32)
        rigid_groups = _build_rigid_groups_from_link_map(link_to_unique)
        return {
            'link_to_unique': link_to_unique,
            'finger_ids': finger_ids_t,
            'joint_type_ids': joint_ids_t,
            'edge_index': edge_index_t,
            'edge_type': edge_type_t,
            'edge_rest_lengths': rest_lengths_t,
            'template_xyz': template_xyz,
            'rigid_groups': rigid_groups,
            'num_fingers': num_fingers,
            'num_joint_types': num_joint_types,
            'N': int(finger_ids_t.shape[0]),
            'E': int(edge_index_t.shape[1]),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        # Build constants and hand_model once
        if self._prepared_constants is None:
            urdf_path, meshes_path = self._load_urdf_assets()
            device = torch.device('cpu')
            hand_model = HandModel(
                robot_name='shadowhand',
                urdf_filename=urdf_path,
                mesh_path=meshes_path,
                batch_size=1,
                device=device,
                hand_scale=self.hand_scale,
                rot_type=self.rot_type,
                anchor=self.trans_anchor,
                mesh_source='urdf',
            )
            canon = self._build_canonical(hand_model)
            self._prepared_constants = {
                'hand_model': hand_model,
                **canon,
            }
            self._ensure_norm_xyz_bounds(device)

        # Resolve what to (ensure) build for this stage
        want_train = (stage is None) or (stage == 'fit')
        want_val = ((stage is None) or (stage in ('fit', 'validate'))) and (self.val_phase is not None)
        want_test = ((stage is None) or (stage == 'test')) and (self.test_phase is not None)

        hand_model = self._prepared_constants['hand_model']
        canon = self._prepared_constants

        # Train dataset
        if want_train and self.train_dataset is None:
            train_phase = self.train_phase or self.dataset_kwargs.get('phase', 'all')
            base_train = self._make_base_dataset(phase=train_phase)
            self.train_dataset = _HandEncoderPreparedDataset(
                base_train,
                hand_model,
                canon['link_to_unique'],
                canon['finger_ids'],
                canon['joint_type_ids'],
                canon['edge_index'],
                canon['edge_type'],
                canon['edge_rest_lengths'],
                scale=self.hand_scale,
                norm_xyz_bounds=self._clone_norm_bounds(),
                use_local_pose_only=self.use_local_pose_only,
            )

        # Validation dataset
        if want_val and self.val_dataset is None:
            base_val = self._make_base_dataset(phase=self.val_phase)
            self.val_dataset = _HandEncoderPreparedDataset(
                base_val,
                hand_model,
                canon['link_to_unique'],
                canon['finger_ids'],
                canon['joint_type_ids'],
                canon['edge_index'],
                canon['edge_type'],
                canon['edge_rest_lengths'],
                scale=self.hand_scale,
                norm_xyz_bounds=self._clone_norm_bounds(),
                use_local_pose_only=self.use_local_pose_only,
            )

        # Test dataset
        if want_test and self.test_dataset is None:
            base_test = self._make_base_dataset(phase=self.test_phase)
            self.test_dataset = _HandEncoderPreparedDataset(
                base_test,
                hand_model,
                canon['link_to_unique'],
                canon['finger_ids'],
                canon['joint_type_ids'],
                canon['edge_index'],
                canon['edge_type'],
                canon['edge_rest_lengths'],
                scale=self.hand_scale,
                norm_xyz_bounds=self._clone_norm_bounds(),
                use_local_pose_only=self.use_local_pose_only,
            )

    def get_graph_constants(self) -> Dict[str, torch.Tensor]:
        if self._prepared_constants is None:
            raise RuntimeError("HandEncoderDataModule.setup must be called before accessing graph constants")
        tensor_keys = ['finger_ids', 'joint_type_ids', 'edge_index', 'edge_type', 'edge_rest_lengths']
        consts: Dict[str, Any] = {}
        for key in tensor_keys:
            tensor = self._prepared_constants.get(key)
            if tensor is None:
                raise KeyError(f"Missing graph constant '{key}' in prepared constants")
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor)
            consts[key] = tensor.detach().cpu()

        template_xyz = self._prepared_constants.get('template_xyz')
        if template_xyz is None:
            raise KeyError("Missing 'template_xyz' in prepared constants")
        consts['template_xyz'] = template_xyz.detach().cpu()

        rigid_groups_raw = self._prepared_constants.get('rigid_groups', []) or []
        consts['rigid_groups'] = [
            (grp.clone().detach().long() if isinstance(grp, torch.Tensor)
             else torch.tensor(grp, dtype=torch.long))
            for grp in rigid_groups_raw
            if grp is not None and len(grp) > 0
        ]

        return consts

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        _kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate_for_hand_encoder,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            _kwargs['prefetch_factor'] = int(self.prefetch_factor)
        return DataLoader(self.train_dataset, **_kwargs)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        _kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate_for_hand_encoder,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            _kwargs['prefetch_factor'] = int(self.prefetch_factor)
        return DataLoader(self.val_dataset, **_kwargs)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        _kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate_for_hand_encoder,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            _kwargs['prefetch_factor'] = int(self.prefetch_factor)
        return DataLoader(self.test_dataset, **_kwargs)
