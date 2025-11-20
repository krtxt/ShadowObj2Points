from __future__ import annotations

from typing import Any, Dict

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class LightDeviceStatsMonitor(Callback):
    """Lightweight replacement for Lightning's DeviceStatsMonitor.

    - Logs only a few CUDA memory stats.
    - Logs at a configurable batch interval instead of every batch.
    """

    def __init__(
        self,
        log_every_n_train_batches: int = 50,
        log_every_n_val_batches: int = 50,
        log_every_n_test_batches: int = 50,
        prefix: str = "device",
    ) -> None:
        super().__init__()
        self.log_every_n_train_batches = max(1, int(log_every_n_train_batches))
        self.log_every_n_val_batches = max(1, int(log_every_n_val_batches))
        self.log_every_n_test_batches = max(1, int(log_every_n_test_batches))
        self.prefix = prefix

    def _get_device_stats(self, device: torch.device) -> Dict[str, float]:
        if not torch.cuda.is_available() or device.type != "cuda":
            return {}

        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        total = float(props.total_memory)
        allocated = float(torch.cuda.memory_allocated(index))
        reserved = float(torch.cuda.memory_reserved(index))
        used_ratio = allocated / total if total > 0 else 0.0

        gb = 1024.0 ** 3
        return {
            "memory_allocated_gb": allocated / gb,
            "memory_reserved_gb": reserved / gb,
            "memory_used_ratio": used_ratio,
        }

    @rank_zero_only
    def _log_device_stats(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        tag: str,
        batch_idx: int,
    ) -> None:
        logger = trainer.logger
        if logger is None or not hasattr(logger, "log_metrics"):
            return

        stats = self._get_device_stats(pl_module.device)
        if not stats:
            return

        metrics: Dict[str, float] = {
            f"{self.prefix}/{tag}/{name}": value for name, value in stats.items()
        }
        logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % self.log_every_n_train_batches != 0:
            return
        self._log_device_stats(trainer, pl_module, tag="train", batch_idx=batch_idx)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if (batch_idx + 1) % self.log_every_n_val_batches != 0:
            return
        self._log_device_stats(trainer, pl_module, tag="val", batch_idx=batch_idx)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if (batch_idx + 1) % self.log_every_n_test_batches != 0:
            return
        self._log_device_stats(trainer, pl_module, tag="test", batch_idx=batch_idx)
