import bisect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CachedHandKeypointDataset(Dataset):
    """Dataset that streams precomputed hand keypoint caches from disk."""

    def __init__(
        self,
        cache_dir: str,
        phase: str,
        shard_pattern: Optional[str] = None,
        max_shards_in_memory: Union[int, str] = 2,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.phase = phase
        self.shard_pattern = shard_pattern or f"{phase}_cache_*.pt"
        self._max_shards_arg = max_shards_in_memory
        self.max_shards_in_memory = 2  # finalized after shard scan param parse
        self.show_progress = bool(show_progress)
        self.preload_all = False

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        self.shard_paths = sorted(self.cache_dir.glob(self.shard_pattern))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No cache shards matching '{self.shard_pattern}' in {self.cache_dir}"
            )
        # finalize memory policy from max_shards_in_memory argument
        arg = self._max_shards_arg
        if isinstance(arg, str):
            key = arg.strip().lower()
            if key in ("all", "full", "infinite", "inf", "max"):
                self.preload_all = True
                self.max_shards_in_memory = len(self.shard_paths)
            else:
                try:
                    self.max_shards_in_memory = max(1, int(arg))
                except Exception:
                    self.max_shards_in_memory = 2
        else:
            try:
                v = int(arg)
                if v <= 0:
                    self.preload_all = True
                    self.max_shards_in_memory = len(self.shard_paths)
                else:
                    self.max_shards_in_memory = max(1, v)
            except Exception:
                self.max_shards_in_memory = 2

        self._meta: Optional[Dict[str, Any]] = None
        self._shard_lengths: List[int] = []
        self._cumulative: List[int] = []
        total = 0
        self._in_memory_samples: Optional[List[Dict[str, Any]]] = None

        def _is_main_proc() -> bool:
            try:
                return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                return True

        use_progress = self.show_progress and _is_main_proc()
        progress_iter = None
        if self.preload_all and use_progress:
            try:
                from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
                progress = Progress(
                    "[bold blue]Scanning cache shards",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "•",
                    TimeElapsedColumn(),
                    "|",
                    TimeRemainingColumn(),
                    transient=True,
                    refresh_per_second=5,
                )
                with progress:
                    task = progress.add_task(f"{self.phase}: {len(self.shard_paths)} shards", total=len(self.shard_paths))
                    self._in_memory_samples = []
                    for idx, path in enumerate(self.shard_paths):
                        payload = torch.load(path, map_location="cpu")
                        samples: Sequence[Dict[str, Any]] = payload.get("samples", [])
                        length = len(samples)
                        self._shard_lengths.append(length)
                        total += length
                        self._cumulative.append(total)
                        if self._meta is None:
                            self._meta = payload.get("meta", {})
                        # Normalize keys upfront for speed parity with lazy path
                        for s in samples:
                            if 'xyz_local' in s and 'cached_xyz_local' not in s:
                                s['cached_xyz_local'] = s.pop('xyz_local')
                            if 'se3' in s and 'cached_se3' not in s:
                                s['cached_se3'] = s.pop('se3')
                            self._in_memory_samples.append(s)
                        del samples
                        del payload
                        progress.advance(task)
            except Exception:
                # Fallback to simple logging
                self._in_memory_samples = []
                for idx, path in enumerate(self.shard_paths):
                    payload = torch.load(path, map_location="cpu")
                    samples: Sequence[Dict[str, Any]] = payload.get("samples", [])
                    length = len(samples)
                    self._shard_lengths.append(length)
                    total += length
                    self._cumulative.append(total)
                    if self._meta is None:
                        self._meta = payload.get("meta", {})
                    for s in samples:
                        if 'xyz_local' in s and 'cached_xyz_local' not in s:
                            s['cached_xyz_local'] = s.pop('xyz_local')
                        if 'se3' in s and 'cached_se3' not in s:
                            s['cached_se3'] = s.pop('se3')
                        self._in_memory_samples.append(s)
                    del samples
                    del payload
                    if (idx + 1) % 25 == 0:
                        logger.info("Scanning cache shards %d/%d... (cum=%d)", idx + 1, len(self.shard_paths), total)
        else:
            for idx, path in enumerate(self.shard_paths):
                payload = torch.load(path, map_location="cpu")
                samples: Sequence[Dict[str, Any]] = payload.get("samples", [])
                length = len(samples)
                self._shard_lengths.append(length)
                total += length
                self._cumulative.append(total)
                if self._meta is None:
                    self._meta = payload.get("meta", {})
                if self.preload_all:
                    if self._in_memory_samples is None:
                        self._in_memory_samples = []
                    for s in samples:
                        if 'xyz_local' in s and 'cached_xyz_local' not in s:
                            s['cached_xyz_local'] = s.pop('xyz_local')
                        if 'se3' in s and 'cached_se3' not in s:
                            s['cached_se3'] = s.pop('se3')
                        self._in_memory_samples.append(s)
                del samples
                del payload
                if self.show_progress and (idx + 1) % 50 == 0:
                    logger.info("Scanning cache shards %d/%d... (cum=%d)", idx + 1, len(self.shard_paths), total)

        if total == 0:
            raise RuntimeError(
                f"Cache directory {self.cache_dir} does not contain any samples for phase '{phase}'."
            )

        self._total = total
        self._loaded_shards: Dict[int, List[Dict[str, Any]]] = {}
        self._lru: List[int] = []
        if self.preload_all:
            logger.info(
                "CachedHandKeypointDataset initialized: phase=%s, shards=%d, total_samples=%d (preloaded to RAM)",
                phase,
                len(self.shard_paths),
                self._total,
            )
        else:
            logger.info(
                "CachedHandKeypointDataset initialized: phase=%s, shards=%d, total_samples=%d",
                phase,
                len(self.shard_paths),
                self._total,
            )

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta or {})

    def __len__(self) -> int:
        return self._total

    def _locate_shard(self, idx: int) -> int:
        shard_idx = bisect.bisect_right(self._cumulative, idx)
        if shard_idx >= len(self.shard_paths):
            raise IndexError(f"Index {idx} out of bounds (total={self._total})")
        return shard_idx

    def _load_shard(self, shard_idx: int) -> List[Dict[str, Any]]:
        if shard_idx in self._loaded_shards:
            return self._loaded_shards[shard_idx]

        payload = torch.load(self.shard_paths[shard_idx], map_location="cpu")
        samples: List[Dict[str, Any]] = payload.get("samples", [])
        self._loaded_shards[shard_idx] = samples
        self._lru.append(shard_idx)
        if len(self._lru) > self.max_shards_in_memory:
            drop_idx = self._lru.pop(0)
            if drop_idx in self._loaded_shards:
                del self._loaded_shards[drop_idx]
        return samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of bounds (total={self._total})")
        if self._in_memory_samples is not None:
            sample = dict(self._in_memory_samples[idx])
            return sample
        shard_idx = self._locate_shard(idx)
        prev_cum = self._cumulative[shard_idx - 1] if shard_idx > 0 else 0
        local_idx = idx - prev_cum
        shard_samples = self._load_shard(shard_idx)
        sample = dict(shard_samples[local_idx])
        if 'xyz_local' in sample and 'cached_xyz_local' not in sample:
            sample['cached_xyz_local'] = sample.pop('xyz_local')
        if 'se3' in sample and 'cached_se3' not in sample:
            sample['cached_se3'] = sample.pop('se3')
        return sample


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
    """

    def __init__(self, file_path: str, phase: Optional[str] = None, show_progress: bool = True) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.phase = phase
        self.show_progress = bool(show_progress)
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
        return sample

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            if getattr(self, "_file", None) is not None:
                self._file.close()
        except Exception:
            pass
