"""Training ETA (Estimated Time of Arrival) callback for time estimation.

Stores ETA info in pl_module.run_info["eta"] for ValidationSummaryCallback to display.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback


class TrainingETACallback(Callback):
    """Estimate training completion time and store in pl_module.run_info["eta"].

    Stores ETA data for ValidationSummaryCallback to display, rather than
    printing directly. This provides cleaner, unified output during validation.

    Args:
        datetime_format: Format string for completion datetime display.
    """

    def __init__(self, datetime_format: str = "%Y-%m-%d %H:%M:%S") -> None:
        super().__init__()
        self.datetime_format = datetime_format
        self._start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._epoch_times: list = []

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 0:
            return "N/A"
        td = timedelta(seconds=int(seconds))
        days, hours = td.days, td.seconds // 3600
        minutes, secs = (td.seconds % 3600) // 60, td.seconds % 60
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or days > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def _ensure_run_info(self, pl_module: L.LightningModule) -> Dict[str, Any]:
        """Ensure pl_module has run_info dict."""
        if not hasattr(pl_module, "run_info"):
            pl_module.run_info = {}
        return pl_module.run_info

    def _update_eta_info(self, pl_module: L.LightningModule, current_epoch: int, max_epochs: int) -> None:
        """Calculate and store ETA info in pl_module.run_info["eta"]."""
        run_info = self._ensure_run_info(pl_module)

        elapsed = time.time() - self._start_time if self._start_time else 0.0
        avg_epoch = sum(self._epoch_times) / len(self._epoch_times) if self._epoch_times else 0.0

        if current_epoch > 0 and max_epochs > 0:
            progress = current_epoch / max_epochs
            remaining = (elapsed / progress - elapsed) if progress > 0 else 0.0
        else:
            remaining = avg_epoch * max_epochs if avg_epoch > 0 else 0.0

        completion = datetime.now() + timedelta(seconds=remaining)
        progress_pct = (current_epoch / max_epochs * 100) if max_epochs > 0 else 0

        run_info["eta"] = {
            "elapsed": elapsed,
            "remaining": remaining,
            "completion": completion.strftime(self.datetime_format),
            "progress_pct": progress_pct,
            "avg_epoch": avg_epoch,
            "current_epoch": current_epoch,
            "max_epochs": max_epochs,
        }

        # Log to tensorboard/wandb
        pl_module.log("train/eta_remaining_hours", remaining / 3600, on_epoch=True, logger=True)
        pl_module.log("train/elapsed_hours", elapsed / 3600, on_epoch=True, logger=True)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record training start time."""
        self._start_time = time.time()
        self._epoch_times = []

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record epoch start time."""
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Calculate epoch time and update ETA info."""
        if self._epoch_start_time is not None:
            self._epoch_times.append(time.time() - self._epoch_start_time)

        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or 1
        self._update_eta_info(pl_module, current_epoch, max_epochs)
