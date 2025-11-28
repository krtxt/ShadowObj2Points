"""Training ETA (Estimated Time of Arrival) callback for time estimation."""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

log = logging.getLogger(__name__)


class TrainingETACallback(Callback):
    """Estimate and display training completion time.
    
    This callback tracks training progress and provides:
    - Elapsed time since training started
    - Estimated remaining time
    - Predicted completion datetime
    - Average time per epoch/step
    
    Args:
        log_every_n_epochs: Log ETA every N epochs (default: 1).
        log_every_n_steps: Log ETA every N steps (0 to disable step-level logging).
        show_epoch_time: Whether to show average epoch time.
        datetime_format: Format string for completion datetime display.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        log_every_n_steps: int = 0,
        show_epoch_time: bool = True,
        datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.show_epoch_time = show_epoch_time
        self.datetime_format = datetime_format

        self._start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._epoch_times: list = []
        self._step_times: list = []
        self._last_step_time: Optional[float] = None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 0:
            return "N/A"
        
        td = timedelta(seconds=int(seconds))
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or days > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        
        return " ".join(parts)

    def _calculate_eta(
        self, current_epoch: int, max_epochs: int
    ) -> tuple[float, float, datetime]:
        """Calculate ETA based on epoch progress.
        
        Returns:
            Tuple of (elapsed_seconds, remaining_seconds, completion_datetime)
        """
        if self._start_time is None:
            return 0.0, 0.0, datetime.now()
        
        elapsed = time.time() - self._start_time
        
        if current_epoch == 0:
            # Use average epoch time if available
            if self._epoch_times:
                avg_epoch_time = sum(self._epoch_times) / len(self._epoch_times)
                remaining_epochs = max_epochs - current_epoch
                remaining = avg_epoch_time * remaining_epochs
            else:
                remaining = 0.0
        else:
            # Linear extrapolation based on progress
            progress = current_epoch / max_epochs
            if progress > 0:
                total_estimated = elapsed / progress
                remaining = total_estimated - elapsed
            else:
                remaining = 0.0
        
        completion = datetime.now() + timedelta(seconds=remaining)
        return elapsed, remaining, completion

    @rank_zero_only
    def _log_eta(
        self,
        pl_module: L.LightningModule,
        current_epoch: int,
        max_epochs: int,
        prefix: str = "",
    ) -> None:
        """Log ETA information."""
        elapsed, remaining, completion = self._calculate_eta(current_epoch, max_epochs)
        
        progress_pct = (current_epoch / max_epochs * 100) if max_epochs > 0 else 0
        
        msg_parts = [f"{prefix}Progress: {progress_pct:.1f}%"]
        msg_parts.append(f"Elapsed: {self._format_duration(elapsed)}")
        msg_parts.append(f"Remaining: {self._format_duration(remaining)}")
        msg_parts.append(f"ETA: {completion.strftime(self.datetime_format)}")
        
        if self.show_epoch_time and self._epoch_times:
            avg_epoch = sum(self._epoch_times[-10:]) / len(self._epoch_times[-10:])
            msg_parts.append(f"Avg epoch: {self._format_duration(avg_epoch)}")
        
        log.info(" | ".join(msg_parts))
        
        # Also log to trainer's logger for tensorboard/wandb
        pl_module.log("train/eta_remaining_hours", remaining / 3600, on_epoch=True, logger=True)
        pl_module.log("train/elapsed_hours", elapsed / 3600, on_epoch=True, logger=True)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record training start time."""
        self._start_time = time.time()
        self._epoch_times = []
        self._step_times = []
        log.info(f"Training started at {datetime.now().strftime(self.datetime_format)}")

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Record epoch start time."""
        self._epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Calculate epoch time and log ETA."""
        if self._epoch_start_time is not None:
            epoch_time = time.time() - self._epoch_start_time
            self._epoch_times.append(epoch_time)
        
        current_epoch = trainer.current_epoch + 1  # +1 because epoch just ended
        max_epochs = trainer.max_epochs or 1
        
        if self.log_every_n_epochs > 0 and current_epoch % self.log_every_n_epochs == 0:
            self._log_eta(pl_module, current_epoch, max_epochs, prefix="[ETA] ")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Track step-level timing."""
        current_time = time.time()
        
        if self._last_step_time is not None:
            step_time = current_time - self._last_step_time
            self._step_times.append(step_time)
            # Keep only last 100 step times
            if len(self._step_times) > 100:
                self._step_times = self._step_times[-100:]
        
        self._last_step_time = current_time
        
        # Step-level ETA logging (if enabled)
        if self.log_every_n_steps > 0:
            global_step = trainer.global_step
            if global_step > 0 and global_step % self.log_every_n_steps == 0:
                if self._step_times:
                    avg_step = sum(self._step_times) / len(self._step_times)
                    pl_module.log("train/avg_step_time_ms", avg_step * 1000, on_step=True, logger=True)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log final training duration."""
        if self._start_time is not None:
            total_time = time.time() - self._start_time
            log.info(
                f"Training completed. Total time: {self._format_duration(total_time)} "
                f"({total_time / 3600:.2f} hours)"
            )
            if self._epoch_times:
                avg_epoch = sum(self._epoch_times) / len(self._epoch_times)
                log.info(f"Average epoch time: {self._format_duration(avg_epoch)}")
