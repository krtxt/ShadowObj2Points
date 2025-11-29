"""DDP Watchdog Callback - Detects deadlocks and zombie processes in distributed training."""

import logging
import os
import signal
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


class DDPWatchdog(Callback):
    """Watchdog callback to detect DDP deadlocks and stuck training.
    
    Features:
    - Monitors training step duration and detects hangs
    - Periodic heartbeat to detect unresponsive processes
    - Optional barrier timeout monitoring
    - Zombie process detection and cleanup
    - Emergency checkpoint saving on timeout
    
    Args:
        step_timeout: Max seconds per training step before warning (default: 300)
        heartbeat_interval: Seconds between heartbeat checks (default: 60)
        max_consecutive_timeouts: Stop training after N consecutive timeouts (default: 3)
        barrier_timeout: Timeout for DDP barrier operations in seconds (default: 1800)
        kill_zombies: Whether to attempt killing zombie child processes (default: True)
        save_on_timeout: Save emergency checkpoint on timeout (default: True)
        log_gpu_memory: Log GPU memory usage on timeout (default: True)
    """

    def __init__(
        self,
        step_timeout: float = 300.0,
        heartbeat_interval: float = 60.0,
        max_consecutive_timeouts: int = 3,
        barrier_timeout: float = 1800.0,
        kill_zombies: bool = True,
        save_on_timeout: bool = True,
        log_gpu_memory: bool = True,
    ) -> None:
        super().__init__()
        self.step_timeout = step_timeout
        self.heartbeat_interval = heartbeat_interval
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.barrier_timeout = barrier_timeout
        self.kill_zombies = kill_zombies
        self.save_on_timeout = save_on_timeout
        self.log_gpu_memory = log_gpu_memory

        self._log = logging.getLogger(self.__class__.__name__)
        self._last_step_time: Optional[float] = None
        self._last_heartbeat: Optional[float] = None
        self._consecutive_timeouts: int = 0
        self._step_count: int = 0
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_watchdog: threading.Event = threading.Event()
        self._trainer: Optional[Trainer] = None
        self._warned_this_step: bool = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup watchdog when training starts."""
        self._trainer = trainer
        
        if stage == "fit" and trainer.world_size > 1:
            self._configure_ddp_timeout(trainer)
            rank_zero_info(
                f"[DDPWatchdog] Initialized: step_timeout={self.step_timeout}s, "
                f"barrier_timeout={self.barrier_timeout}s, world_size={trainer.world_size}"
            )

    def _configure_ddp_timeout(self, trainer: Trainer) -> None:
        """Configure DDP process group timeout if possible."""
        if not dist.is_initialized():
            return

        try:
            timeout = timedelta(seconds=self.barrier_timeout)
            if hasattr(dist, "distributed_c10d") and hasattr(dist.distributed_c10d, "_get_default_group"):
                pg = dist.distributed_c10d._get_default_group()
                if hasattr(pg, "options") and hasattr(pg.options, "timeout"):
                    rank_zero_info(f"[DDPWatchdog] Setting process group timeout to {self.barrier_timeout}s")
        except Exception as e:
            self._log.debug(f"Could not set process group timeout: {e}")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start watchdog thread when training begins."""
        self._stop_watchdog.clear()
        self._last_heartbeat = time.time()
        
        if trainer.world_size > 1:
            self._start_watchdog_thread()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Stop watchdog thread when training ends."""
        self._stop_watchdog_thread()

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Record batch start time."""
        self._last_step_time = time.time()
        self._last_heartbeat = time.time()
        self._warned_this_step = False

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check step duration and reset counters."""
        if self._last_step_time is None:
            return

        elapsed = time.time() - self._last_step_time
        self._step_count += 1
        self._last_heartbeat = time.time()

        if elapsed > self.step_timeout:
            self._consecutive_timeouts += 1
            rank_zero_warn(
                f"[DDPWatchdog] Step {batch_idx} took {elapsed:.1f}s "
                f"(timeout: {self.step_timeout}s) - consecutive: {self._consecutive_timeouts}"
            )
            
            if self.log_gpu_memory:
                self._log_gpu_status()

            if self._consecutive_timeouts >= self.max_consecutive_timeouts:
                self._handle_timeout_limit(trainer, pl_module)
        else:
            self._consecutive_timeouts = 0

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Handle exceptions - cleanup and log status."""
        self._stop_watchdog_thread()
        
        if self.log_gpu_memory:
            self._log_gpu_status()
        
        if self.kill_zombies:
            self._cleanup_zombie_processes()

    def _start_watchdog_thread(self) -> None:
        """Start background watchdog thread."""
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="DDPWatchdog"
        )
        self._watchdog_thread.start()
        rank_zero_info("[DDPWatchdog] Background watchdog thread started")

    def _stop_watchdog_thread(self) -> None:
        """Stop background watchdog thread."""
        self._stop_watchdog.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5.0)
            self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        """Background loop to check for hangs."""
        while not self._stop_watchdog.wait(self.heartbeat_interval):
            if self._last_heartbeat is None:
                continue

            elapsed = time.time() - self._last_heartbeat
            
            if elapsed > self.step_timeout and not self._warned_this_step:
                self._warned_this_step = True
                self._log.warning(
                    f"[DDPWatchdog] No heartbeat for {elapsed:.1f}s - possible hang detected! "
                    f"(rank={self._get_rank()})"
                )
                self._log_gpu_status()

    def _handle_timeout_limit(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Handle reaching the consecutive timeout limit."""
        rank_zero_warn(
            f"[DDPWatchdog] Reached {self.max_consecutive_timeouts} consecutive timeouts. "
            "Stopping training."
        )

        if self.save_on_timeout:
            self._save_emergency_checkpoint(trainer)

        if self.kill_zombies:
            self._cleanup_zombie_processes()

        trainer.should_stop = True

    def _save_emergency_checkpoint(self, trainer: Trainer) -> None:
        """Save emergency checkpoint on timeout."""
        try:
            from pathlib import Path
            
            output_dir = trainer.default_root_dir or "."
            ckpt_dir = Path(output_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = ckpt_dir / f"emergency_timeout_{timestamp}.ckpt"
            
            trainer.save_checkpoint(str(ckpt_path))
            rank_zero_info(f"[DDPWatchdog] Emergency checkpoint saved: {ckpt_path}")
        except Exception as e:
            self._log.error(f"[DDPWatchdog] Failed to save emergency checkpoint: {e}")

    def _log_gpu_status(self) -> None:
        """Log GPU memory status for debugging."""
        if not torch.cuda.is_available():
            return

        try:
            rank = self._get_rank()
            device = torch.cuda.current_device()
            
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            
            self._log.warning(
                f"[DDPWatchdog] GPU {device} (rank {rank}): "
                f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={max_allocated:.2f}GB"
            )
        except Exception as e:
            self._log.debug(f"Could not get GPU status: {e}")

    def _cleanup_zombie_processes(self) -> None:
        """Attempt to cleanup zombie child processes."""
        try:
            import subprocess
            
            pid = os.getpid()
            result = subprocess.run(
                ["ps", "--ppid", str(pid), "-o", "pid,stat,cmd", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            zombie_count = 0
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2 and "Z" in parts[1]:
                    zombie_pid = int(parts[0])
                    zombie_count += 1
                    self._log.warning(f"[DDPWatchdog] Found zombie process: PID {zombie_pid}")
                    try:
                        os.waitpid(zombie_pid, os.WNOHANG)
                    except ChildProcessError:
                        pass

            if zombie_count > 0:
                rank_zero_warn(f"[DDPWatchdog] Cleaned up {zombie_count} zombie processes")
                
        except Exception as e:
            self._log.debug(f"Could not check for zombies: {e}")

    def _get_rank(self) -> int:
        """Get current process rank."""
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def state_key(self) -> str:
        return self.__class__.__name__


class DDPBarrierMonitor(Callback):
    """Monitor DDP barrier operations for potential deadlocks.
    
    This callback wraps barrier calls with timeout detection.
    Use in combination with DDPWatchdog for comprehensive monitoring.
    
    Args:
        barrier_timeout: Timeout for barrier in seconds (default: 600)
        check_interval: How often to log waiting status (default: 30)
    """

    def __init__(
        self,
        barrier_timeout: float = 600.0,
        check_interval: float = 30.0,
    ) -> None:
        super().__init__()
        self.barrier_timeout = barrier_timeout
        self.check_interval = check_interval
        self._log = logging.getLogger(self.__class__.__name__)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Monitor barrier at epoch end."""
        if trainer.world_size <= 1 or not dist.is_initialized():
            return

        self._timed_barrier(f"epoch_{trainer.current_epoch}_end")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Monitor barrier at validation end."""
        if trainer.world_size <= 1 or not dist.is_initialized():
            return

        self._timed_barrier(f"val_epoch_{trainer.current_epoch}_end")

    def _timed_barrier(self, name: str) -> None:
        """Execute barrier with timeout monitoring."""
        start = time.time()
        rank = dist.get_rank()
        
        def timeout_warning():
            elapsed = 0
            while elapsed < self.barrier_timeout:
                time.sleep(self.check_interval)
                elapsed = time.time() - start
                if elapsed >= self.check_interval:
                    self._log.warning(
                        f"[DDPBarrierMonitor] Rank {rank} waiting at barrier '{name}' "
                        f"for {elapsed:.1f}s..."
                    )

        warning_thread = threading.Thread(target=timeout_warning, daemon=True)
        warning_thread.start()

        try:
            dist.barrier()
            elapsed = time.time() - start
            if elapsed > 5.0:
                rank_zero_info(f"[DDPBarrierMonitor] Barrier '{name}' completed in {elapsed:.1f}s")
        except Exception as e:
            self._log.error(f"[DDPBarrierMonitor] Barrier '{name}' failed on rank {rank}: {e}")
            raise
