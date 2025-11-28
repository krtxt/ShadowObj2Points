import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 0,
    total_epochs: int = 100,
    eta_min: float = 0.0,
    warmup_start_factor: float = 1e-3,
    resume_epoch: int = 0,
    skip_warmup_on_resume: bool = True,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a warmup + cosine scheduler using standard PyTorch schedulers.

    Supports resuming from checkpoint with optional warmup skip.

    Intended for use via Hydra:

    scheduler:
      _target_: src.models.components.schedulers.build_warmup_cosine_scheduler
      warmup_epochs: 10
      total_epochs: ${trainer.max_epochs}
      eta_min: 1e-6
      warmup_start_factor: 1e-3
      skip_warmup_on_resume: true
      
    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of training epochs.
        eta_min: Minimum learning rate for cosine annealing.
        warmup_start_factor: Starting factor for warmup (lr * factor).
        resume_epoch: Current epoch when resuming (set by model.configure_optimizers).
        skip_warmup_on_resume: If True and resume_epoch > 0, skip warmup phase.
    """
    warmup_epochs = int(warmup_epochs)
    total_epochs = int(total_epochs)
    eta_min = float(eta_min)
    warmup_start_factor = float(warmup_start_factor)
    resume_epoch = int(resume_epoch)

    # Check if we should skip warmup when resuming
    if skip_warmup_on_resume and resume_epoch >= warmup_epochs and warmup_epochs > 0:
        log.info(
            f"Skipping warmup phase: resume_epoch={resume_epoch} >= warmup_epochs={warmup_epochs}"
        )
        # Just use cosine scheduler from current position
        remaining_epochs = max(1, total_epochs - resume_epoch)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs,
            eta_min=eta_min,
        )

    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=eta_min,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(1e-8, warmup_start_factor),
        end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    
    # If resuming within warmup phase, advance the scheduler
    if resume_epoch > 0 and resume_epoch < warmup_epochs:
        log.info(f"Resuming within warmup phase: advancing scheduler to epoch {resume_epoch}")
        for _ in range(resume_epoch):
            scheduler.step()
    
    return scheduler


class WarmupSkipSchedulerWrapper:
    """Wrapper that helps models skip warmup on resume.
    
    This wrapper tracks the current epoch and can be used to build
    schedulers with proper resume support.
    
    Usage in LightningModule.configure_optimizers():
    
        def configure_optimizers(self):
            optimizer = ...
            scheduler_cfg = self.optimizer_cfg.get("scheduler", {})
            
            # Inject resume epoch if available
            if self.trainer and self.trainer.current_epoch > 0:
                scheduler_cfg["resume_epoch"] = self.trainer.current_epoch
            
            scheduler = hydra_instantiate(scheduler_cfg, optimizer=optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
    """
    
    @staticmethod
    def get_resume_epoch(trainer) -> int:
        """Get the current epoch from trainer for resume support."""
        if trainer is None:
            return 0
        # During fit, current_epoch is the epoch being trained
        # When resuming, it will be set to the checkpoint's epoch
        return getattr(trainer, "current_epoch", 0)

