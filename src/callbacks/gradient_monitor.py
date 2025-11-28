"""Gradient health monitoring callback for detecting NaN/Inf values."""

import logging
from typing import Dict, List, Optional, Set

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

log = logging.getLogger(__name__)


class GradientMonitor(Callback):
    """Monitor gradients for NaN/Inf values and optionally halt training.

    This callback checks gradients after each backward pass and can:
    - Log warnings when NaN/Inf gradients are detected
    - Track which parameters have problematic gradients
    - Optionally stop training after too many consecutive bad batches
    - Log gradient statistics for debugging
    
    Args:
        check_nan: Whether to check for NaN gradients.
        check_inf: Whether to check for Inf gradients.
        halt_on_nan: Stop training immediately when NaN is detected.
        halt_on_inf: Stop training immediately when Inf is detected.
        max_consecutive_bad_batches: Stop training after this many consecutive
            batches with bad gradients. Set to -1 to disable.
        log_param_names: Log the names of parameters with bad gradients.
        log_every_n_steps: How often to log gradient statistics (0 to disable).
        raise_exception: If True, raise an exception instead of stopping trainer.
    """

    def __init__(
        self,
        check_nan: bool = True,
        check_inf: bool = True,
        halt_on_nan: bool = False,
        halt_on_inf: bool = False,
        max_consecutive_bad_batches: int = 10,
        log_param_names: bool = True,
        log_every_n_steps: int = 0,
        raise_exception: bool = False,
    ) -> None:
        super().__init__()
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.halt_on_nan = halt_on_nan
        self.halt_on_inf = halt_on_inf
        self.max_consecutive_bad_batches = max_consecutive_bad_batches
        self.log_param_names = log_param_names
        self.log_every_n_steps = log_every_n_steps
        self.raise_exception = raise_exception

        self._consecutive_bad_batches = 0
        self._total_nan_batches = 0
        self._total_inf_batches = 0
        self._bad_param_names: Set[str] = set()

    def _check_gradients(
        self, pl_module: L.LightningModule
    ) -> Dict[str, List[str]]:
        """Check all gradients for NaN/Inf values.
        
        Returns:
            Dict with 'nan' and 'inf' keys containing lists of parameter names.
        """
        nan_params: List[str] = []
        inf_params: List[str] = []

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            
            if self.check_nan and torch.isnan(grad).any():
                nan_params.append(name)
            if self.check_inf and torch.isinf(grad).any():
                inf_params.append(name)

        return {"nan": nan_params, "inf": inf_params}

    def _compute_grad_stats(
        self, pl_module: L.LightningModule
    ) -> Dict[str, float]:
        """Compute gradient statistics for logging."""
        grads = []
        for param in pl_module.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().view(-1))
        
        if not grads:
            return {}
        
        all_grads = torch.cat(grads)
        return {
            "grad/mean": all_grads.mean().item(),
            "grad/std": all_grads.std().item(),
            "grad/max": all_grads.abs().max().item(),
            "grad/min": all_grads.abs().min().item(),
        }

    @rank_zero_only
    def _log_warning(self, message: str) -> None:
        log.warning(message)

    @rank_zero_only
    def _log_error(self, message: str) -> None:
        log.error(message)

    def on_after_backward(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Check gradients after backward pass."""
        bad_grads = self._check_gradients(pl_module)
        has_nan = len(bad_grads["nan"]) > 0
        has_inf = len(bad_grads["inf"]) > 0
        has_bad = has_nan or has_inf

        if has_bad:
            self._consecutive_bad_batches += 1
            if has_nan:
                self._total_nan_batches += 1
            if has_inf:
                self._total_inf_batches += 1

            # Build warning message
            step = trainer.global_step
            epoch = trainer.current_epoch
            msg_parts = [f"[Epoch {epoch}, Step {step}] Gradient anomaly detected:"]
            
            if has_nan:
                msg_parts.append(f" NaN in {len(bad_grads['nan'])} params")
                if self.log_param_names and bad_grads["nan"]:
                    # Only show first 5 to avoid log spam
                    sample = bad_grads["nan"][:5]
                    msg_parts.append(f" ({', '.join(sample)}{'...' if len(bad_grads['nan']) > 5 else ''})")
                    self._bad_param_names.update(bad_grads["nan"])

            if has_inf:
                msg_parts.append(f" Inf in {len(bad_grads['inf'])} params")
                if self.log_param_names and bad_grads["inf"]:
                    sample = bad_grads["inf"][:5]
                    msg_parts.append(f" ({', '.join(sample)}{'...' if len(bad_grads['inf']) > 5 else ''})")
                    self._bad_param_names.update(bad_grads["inf"])

            msg_parts.append(f" | Consecutive bad batches: {self._consecutive_bad_batches}")
            self._log_warning("".join(msg_parts))

            # Log to trainer's logger
            pl_module.log("grad/nan_count", float(len(bad_grads["nan"])), on_step=True, logger=True)
            pl_module.log("grad/inf_count", float(len(bad_grads["inf"])), on_step=True, logger=True)

            # Check halt conditions
            should_halt = False
            halt_reason = ""

            if self.halt_on_nan and has_nan:
                should_halt = True
                halt_reason = "NaN gradients detected (halt_on_nan=True)"
            elif self.halt_on_inf and has_inf:
                should_halt = True
                halt_reason = "Inf gradients detected (halt_on_inf=True)"
            elif (
                self.max_consecutive_bad_batches > 0
                and self._consecutive_bad_batches >= self.max_consecutive_bad_batches
            ):
                should_halt = True
                halt_reason = f"Exceeded max consecutive bad batches ({self.max_consecutive_bad_batches})"

            if should_halt:
                self._log_error(f"Training halted: {halt_reason}")
                if self.raise_exception:
                    raise RuntimeError(f"GradientMonitor: {halt_reason}")
                else:
                    trainer.should_stop = True
        else:
            # Reset consecutive counter on good batch
            self._consecutive_bad_batches = 0

        # Periodically log gradient statistics
        if self.log_every_n_steps > 0:
            if (trainer.global_step + 1) % self.log_every_n_steps == 0:
                stats = self._compute_grad_stats(pl_module)
                for key, value in stats.items():
                    pl_module.log(key, value, on_step=True, logger=True)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log summary of gradient issues at end of training."""
        if self._total_nan_batches > 0 or self._total_inf_batches > 0:
            self._log_warning(
                f"Training completed with gradient issues: "
                f"NaN batches={self._total_nan_batches}, Inf batches={self._total_inf_batches}"
            )
            if self._bad_param_names:
                self._log_warning(
                    f"Parameters with gradient issues: {', '.join(sorted(self._bad_param_names)[:20])}"
                )
