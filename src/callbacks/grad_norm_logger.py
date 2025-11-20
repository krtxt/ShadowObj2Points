import logging
from typing import Optional

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class GradNormLogger(Callback):
    """Log gradient norm to the Lightning logger.

    This re-introduces the old ``Trainer(track_grad_norm=...)`` behavior for
    newer Lightning versions where the argument was removed.
    """

    def __init__(
        self,
        norm_type: float = 2.0,
        log_every_n_steps: int = 1,
        metric_key: str = "grad/norm",
    ) -> None:
        super().__init__()
        self.norm_type = float(norm_type)
        self.log_every_n_steps = int(log_every_n_steps)
        self.metric_key = str(metric_key)

    def on_after_backward(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
    ) -> None:
        """Compute and log gradient norm after backprop."""
        # Respect logging frequency
        global_step = getattr(trainer, "global_step", 0)
        if self.log_every_n_steps > 1 and (global_step + 1) % self.log_every_n_steps != 0:
            return

        params = [p for p in pl_module.parameters() if p.grad is not None]
        if not params:
            return

        with torch.no_grad():
            # Compute per-parameter norm and aggregate with the same p-norm.
            norms = [p.grad.detach().data.norm(self.norm_type) for p in params]
            if not norms:
                return
            total_norm = torch.norm(torch.stack(norms), self.norm_type)

        # Use Lightning logging API (on_step to mimic Trainer.track_grad_norm)
        pl_module.log(
            self.metric_key,
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

