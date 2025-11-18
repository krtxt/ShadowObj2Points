from typing import Optional

import torch


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 0,
    total_epochs: int = 100,
    eta_min: float = 0.0,
    warmup_start_factor: float = 1e-3,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a warmup + cosine scheduler using standard PyTorch schedulers.

    Intended for use via Hydra:

    scheduler:
      _target_: src.models.components.schedulers.build_warmup_cosine_scheduler
      warmup_epochs: 10
      total_epochs: ${trainer.max_epochs}
      eta_min: 1e-6
      warmup_start_factor: 1e-3
    """
    warmup_epochs = int(warmup_epochs)
    total_epochs = int(total_epochs)
    eta_min = float(eta_min)
    warmup_start_factor = float(warmup_start_factor)

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
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

