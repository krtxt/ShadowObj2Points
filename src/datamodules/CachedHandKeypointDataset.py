import bisect
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

logger = logging.getLogger(__name__)


class HDF5PointCloudCache:
    """Lazy-loading point cloud cache from HDF5 file.
    
    This class provides efficient point cloud access with:
    - Lazy file opening (only opens when first accessed)
    - Per-object caching with configurable cache size
    - Scale extraction from scene_id
    """
    
    def __init__(
        self,
        file_path: str,
        max_cache_size: int = 100,
        max_points: int = 4096,
    ) -> None:
        self.file_path = Path(file_path)
        self.max_cache_size = max_cache_size
        self.max_points = max_points
        self._file: Optional[h5py.File] = None
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._available_objects: Optional[set] = None
    
    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Point cloud HDF5 not found: {self.file_path}")
            self._file = h5py.File(self.file_path, "r")
            if "point_clouds" not in self._file:
                raise KeyError(f"HDF5 missing 'point_clouds' group: {self.file_path}")
            self._available_objects = set(self._file["point_clouds"].keys())
        return self._file
    
    def _extract_scale(self, scene_id: str) -> float:
        """Extract scale from scene_id format: '{obj_code}_scale{value}'"""
        match = re.search(r"_scale([\d.]+)$", scene_id)
        return float(match.group(1)) if match else 1.0
    
    def get(self, obj_code: str, scene_id: str) -> Optional[torch.Tensor]:
        """Get point cloud for object, applying scale from scene_id.
        
        Returns only XYZ coordinates (first 3 columns), shape (N, 3).
        """
        f = self._ensure_open()
        
        if obj_code not in self._available_objects:
            return None
        
        # Check cache
        cache_key = f"{obj_code}_{scene_id}"
        if cache_key in self._cache:
            return torch.from_numpy(self._cache[cache_key].copy())
        
        # Load from HDF5
        try:
            pc = f["point_clouds"][obj_code][:]
            scale = self._extract_scale(scene_id)
            
            # Apply scale to XYZ (first 3 columns)
            pc = pc.astype(np.float32)
            pc[:, :3] *= scale
            
            # Subsample if needed
            valid_mask = np.any(pc[:, :3] != 0, axis=1)
            pc = pc[valid_mask]
            
            if len(pc) > self.max_points:
                indices = np.random.permutation(len(pc))[:self.max_points]
                pc = pc[indices]
            
            # Only keep XYZ (first 3 columns)
            pc = pc[:, :3]
            
            # Update cache with LRU eviction
            if len(self._cache) >= self.max_cache_size:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
            
            self._cache[cache_key] = pc
            self._cache_order.append(cache_key)
            
            return torch.from_numpy(pc.copy())
        except Exception as e:
            logger.warning(f"Failed to load point cloud for {obj_code}: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Explicitly clear the LRU cache to free memory."""
        self._cache.clear()
        self._cache_order.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics for monitoring."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.max_cache_size,
            "cache_memory_mb": sum(pc.nbytes for pc in self._cache.values()) / 1024 / 1024,
        }
    
    def close(self) -> None:
        """Close file and clear cache."""
        self.clear_cache()
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
    
    def __del__(self) -> None:
        self.close()


class CachedHandKeypointDataset(Dataset):
    """Deprecated PT-shard based cache dataset.

    This class is kept for backward compatibility but is no longer supported.
    Only HDF5HandKeypointDataset should be used for hand keypoint caches.
    """

    def __init__(
        self,
        cache_dir: str,
        phase: str,
        shard_pattern: Optional[str] = None,
        max_shards_in_memory: Union[int, str] = 2,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        raise RuntimeError(
            "CachedHandKeypointDataset (PT shards) is deprecated and disabled. "
            "Please use HDF5HandKeypointDataset with '<phase>_cache.h5' files instead."
        )


class HDF5HandKeypointDataset(Dataset):
    """Dataset that reads hand keypoint cache from a single HDF5 file.

    Expected HDF5 layout (datasets):
      - xyz_local: (N, P, 3) float32
      - se3: (N, 4, 4) float32
      - hand_model_pose: (N, D) float32
      - norm_pose: (N, D_norm) float32 (optional)
      - scene_id: (N,) string
      - obj_code: (N,) string
      - grasp_index: (N,) int64

    File-level attrs should contain the same keys as the original PT cache
    "meta" dict (e.g., stored_anchor, source_anchor, hand_scale, rot_type).
    
    Optionally supports loading point clouds from a separate HDF5 file via
    point_cloud_cache parameter.
    """

    def __init__(
        self,
        file_path: str,
        phase: Optional[str] = None,
        show_progress: bool = True,
        point_cloud_cache: Optional[HDF5PointCloudCache] = None,
    ) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.phase = phase
        self.show_progress = bool(show_progress)
        self.point_cloud_cache = point_cloud_cache
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 cache file not found: {self.file_path}")

        self._file: Optional[h5py.File] = None
        self._length: Optional[int] = None
        self._meta_cache: Optional[Dict[str, Any]] = None

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
            if "hand_model_pose" not in self._file:
                raise KeyError(f"HDF5 cache missing 'hand_model_pose' dataset: {self.file_path}")
            self._length = int(self._file["hand_model_pose"].shape[0])
        return self._file

    @property
    def meta(self) -> Dict[str, Any]:
        f = self._ensure_open()
        if self._meta_cache is None:
            meta: Dict[str, Any] = {}
            for k, v in f.attrs.items():
                if isinstance(v, (bytes, bytearray)):
                    meta[k] = v.decode("utf-8")
                else:
                    meta[k] = v
            self._meta_cache = meta
        return dict(self._meta_cache or {})

    def __len__(self) -> int:
        self._ensure_open()
        return int(self._length or 0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        f = self._ensure_open()
        if idx < 0 or idx >= (self._length or 0):
            raise IndexError(f"Index {idx} out of bounds (total={self._length})")

        xyz_local = f["xyz_local"][idx]
        se3 = f["se3"][idx]
        hand_model_pose = f["hand_model_pose"][idx]
        norm_pose = f["norm_pose"][idx] if "norm_pose" in f else None

        scene_id_raw = f["scene_id"][idx]
        obj_code_raw = f["obj_code"][idx]
        grasp_index_raw = f["grasp_index"][idx]

        def _decode_str(x: Any) -> str:
            if isinstance(x, (bytes, bytearray)):
                return x.decode("utf-8")
            return str(x)

        scene_id = _decode_str(scene_id_raw)
        obj_code = _decode_str(obj_code_raw)
        grasp_index = int(grasp_index_raw)

        sample: Dict[str, Any] = {
            "scene_id": scene_id,
            "obj_code": obj_code,
            "grasp_index": grasp_index,
            "cached_xyz_local": torch.from_numpy(np.asarray(xyz_local, dtype=np.float32)),
            "cached_se3": torch.from_numpy(np.asarray(se3, dtype=np.float32)),
            "hand_model_pose": torch.from_numpy(np.asarray(hand_model_pose, dtype=np.float32)),
        }
        if norm_pose is not None and np.asarray(norm_pose).size > 0:
            sample["norm_pose"] = torch.from_numpy(np.asarray(norm_pose, dtype=np.float32))
        
        # Load point cloud if cache is available
        if self.point_cloud_cache is not None:
            pc = self.point_cloud_cache.get(obj_code, scene_id)
            if pc is not None:
                sample["scene_pc"] = pc
            else:
                sample["scene_pc"] = torch.zeros((0, 3), dtype=torch.float32)
        
        return sample

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            if getattr(self, "_file", None) is not None:
                self._file.close()
        except Exception:
            pass
