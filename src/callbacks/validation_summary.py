import math
from typing import Any, Dict, Optional

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback

from utils.logging_utils import log_validation_summary, console


class ValidationSummaryCallback(Callback):
    """Callback to aggregate validation metrics and pretty-print a summary.

    This callback assumes the model (pl_module) optionally exposes:
      - _val_step_losses: List[Tensor] of per-batch validation losses
    and that metrics are logged via pl_module.log with keys such as:
      - "val_loss"
      - "val/flow_*" for flow matching metrics
      - "val/recon_*" for reconstruction metrics
    """

    def on_validation_epoch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
    ) -> None:
        # Aggregate basic loss statistics from per-step losses, if available
        losses_tensor: Optional[torch.Tensor] = None
        if hasattr(pl_module, "_val_step_losses") and pl_module._val_step_losses:
            try:
                losses_tensor = torch.stack(list(pl_module._val_step_losses)).float().detach().cpu()
            except Exception:
                losses_tensor = None

        if losses_tensor is not None and losses_tensor.numel() > 0:
            avg_loss = float(losses_tensor.mean().item())
            loss_std = float(losses_tensor.std(unbiased=False).item())
            loss_min = float(losses_tensor.min().item())
            loss_max = float(losses_tensor.max().item())
            num_batches = int(losses_tensor.numel())
        else:
            avg_loss = float("nan")
            loss_std = float("nan")
            loss_min = float("nan")
            loss_max = float("nan")
            num_batches = 0

        metrics: Dict[str, Any] = dict(trainer.callback_metrics)

        def _get_metric(name: str) -> Optional[float]:
            value = metrics.get(name)
            if value is None:
                return None
            try:
                if hasattr(value, "item"):
                    value = value.item()
                return float(value)
            except Exception:
                return None

        # Detailed loss breakdown: collect all validation loss-like metrics
        # - Always include "val_loss" if available
        # - Additionally include any metrics whose key starts with "val/loss"
        #   so different models can expose their own loss components without
        #   requiring hard-coded names here.
        val_detailed_loss: Dict[str, float] = {}

        for name in metrics.keys():
            if name == "val_loss" or name.startswith("val/loss"):
                v = _get_metric(name)
                if v is None or not math.isfinite(v):
                    continue

                # Normalize display key: strip "val/" prefix if present so
                # components like "val/loss_pos" become "loss_pos".
                display_key = name.split("/", 1)[-1] if "/" in name else name

                # Later metrics with the same display_key should not
                # overwrite earlier ones (e.g., keep the primary val_loss).
                if display_key not in val_detailed_loss:
                    val_detailed_loss[display_key] = v

        # Flow metrics group
        flow_metric_keys = [
            "val/flow_loss",
            "val/flow_loss_fm",
            "val/flow_loss_tangent",
            "val/flow_edge_len_err",
            "val/flow_total",
        ]
        val_flow_metrics: Dict[str, float] = {}
        for key in flow_metric_keys:
            v = _get_metric(key)
            if v is None:
                continue
            short_name = key.split("/", 1)[-1]
            val_flow_metrics[short_name] = v

        # Reconstruction / quality metrics group
        recon_metric_keys = [
            "val/recon_smooth_l1",
            "val/recon_chamfer",
            "val/recon_direction",
            "val/recon_edge_len_err",
            "val/recon_total",
        ]
        val_quality_metrics: Dict[str, float] = {}
        for key in recon_metric_keys:
            v = _get_metric(key)
            if v is None:
                continue
            short_name = key.split("/", 1)[-1]
            val_quality_metrics[short_name] = v

        # Currently we do not track set/diversity metrics here
        val_set_metrics: Dict[str, float] = {}
        val_diversity_metrics: Dict[str, float] = {}

        epoch = int(getattr(trainer, "current_epoch", 0))

        run_meta = getattr(pl_module, "run_meta", None)
        if getattr(trainer, "is_global_zero", True):
            if isinstance(run_meta, dict) and run_meta:
                exp_name = run_meta.get("experiment_name")
                velocity_mode = run_meta.get("velocity_mode")
                backbone_name = run_meta.get("backbone_name")

                console.print("")
                console.rule("[header]🧪 Run Meta Information[/header]", style="cyan")
                if exp_name is not None:
                    console.print(f"[metric]Experiment[/metric]: [value]{exp_name}[/value]")
                if velocity_mode is not None:
                    console.print(f"[metric]Velocity Strategy[/metric]: [value]{velocity_mode}[/value]")
                if backbone_name is not None:
                    console.print(f"[metric]Backbone[/metric]: [value]{backbone_name}[/value]")

        log_validation_summary(
            epoch=epoch,
            num_batches=num_batches,
            avg_loss=avg_loss,
            loss_std=loss_std,
            loss_min=loss_min,
            loss_max=loss_max,
            val_detailed_loss=val_detailed_loss,
            val_set_metrics=val_set_metrics,
            val_quality_metrics=val_quality_metrics,
            val_diversity_metrics=val_diversity_metrics,
            val_flow_metrics=val_flow_metrics,
        )

        # Reset per-step validation loss buffer to avoid cross-loop accumulation
        if hasattr(pl_module, "_val_step_losses"):
            try:
                pl_module._val_step_losses = []
            except Exception:
                # Best-effort; do not crash callback on failure
                pass
