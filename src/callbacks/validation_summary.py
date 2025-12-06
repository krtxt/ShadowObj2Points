"""Validation summary callback for unified training info display.

Aggregates and displays training information from pl_module.run_info:
- eta: Training progress and time estimates (from TrainingETACallback)
- curriculum: Current curriculum stage (from model, if applicable)
- meta: Run metadata (experiment name, backbone, etc.)
- memory: RAM and GPU memory usage (collected here)
"""

import gc
import math
from typing import Any, Dict, Optional

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from utils.logging_utils import log_validation_summary, console

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class ValidationSummaryCallback(Callback):
    """Callback to display unified training info and validation metrics summary.

    Reads from pl_module.run_info dict which can contain:
    - "eta": dict with elapsed, remaining, completion, progress_pct, avg_epoch
    - "curriculum": dict with stage, epoch
    - "meta": dict with experiment_name, velocity_mode, backbone_name
    """

    def _get_run_info(self, pl_module: L.LightningModule) -> Dict[str, Any]:
        """Get run_info dict from pl_module, creating if needed."""
        if not hasattr(pl_module, "run_info"):
            pl_module.run_info = {}
        return pl_module.run_info

    @rank_zero_only
    def _collect_memory(self, pl_module: L.LightningModule) -> Dict[str, float]:
        """Collect and store memory usage in run_info["memory"]."""
        gc.collect()
        mem_info = {}

        if HAS_PSUTIL:
            mem_info["ram_gb"] = psutil.Process().memory_info().rss / 1024**3

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                mem_info[f"gpu{i}_allocated_gb"] = torch.cuda.memory_allocated(i) / 1024**3
                mem_info[f"gpu{i}_reserved_gb"] = torch.cuda.memory_reserved(i) / 1024**3

        run_info = self._get_run_info(pl_module)
        run_info["memory"] = mem_info
        return mem_info

    @rank_zero_only
    def _print_run_info(self, pl_module: L.LightningModule, epoch: int) -> None:
        """Print unified run info section: ETA, curriculum, memory, meta."""
        from callbacks.training_eta import TrainingETACallback

        run_info = self._get_run_info(pl_module)
        has_content = any(run_info.get(k) for k in ("eta", "curriculum", "memory", "meta"))
        if not has_content:
            return

        console.print("")
        console.rule("[header]📊 Run Information[/header]", style="cyan")

        # ETA info
        if eta := run_info.get("eta"):
            fmt = TrainingETACallback.format_duration
            parts = [
                f"[metric]Progress[/metric]: [value]{eta.get('progress_pct', 0):.1f}%[/value]",
                f"[metric]Elapsed[/metric]: [value]{fmt(eta.get('elapsed', 0))}[/value]",
                f"[metric]Remaining[/metric]: [value]{fmt(eta.get('remaining', 0))}[/value]",
                f"[metric]ETA[/metric]: [value]{eta.get('completion', 'N/A')}[/value]",
            ]
            if avg := eta.get("avg_epoch"):
                parts.append(f"[metric]Avg Epoch[/metric]: [value]{fmt(avg)}[/value]")
            console.print(" | ".join(parts))

        # Curriculum stage
        if curr := run_info.get("curriculum"):
            console.print(
                f"[metric]Curriculum Stage[/metric]: [value]{curr.get('stage', 'N/A')}[/value]"
            )

        # Memory usage
        if mem := run_info.get("memory"):
            parts = []
            if "ram_gb" in mem:
                parts.append(f"RAM: {mem['ram_gb']:.2f} GB")
            for i in range(8):  # Support up to 8 GPUs
                if f"gpu{i}_allocated_gb" in mem:
                    alloc = mem[f"gpu{i}_allocated_gb"]
                    reserved = mem.get(f"gpu{i}_reserved_gb", alloc)
                    parts.append(f"GPU{i}: {alloc:.2f}/{reserved:.2f} GB")
            if parts:
                console.print(f"[metric]Memory[/metric]: [value]{' | '.join(parts)}[/value]")

        # Run meta info
        if meta := run_info.get("meta"):
            meta_parts = []
            if exp := meta.get("experiment_name"):
                meta_parts.append(f"Exp: {exp}")
            if vel := meta.get("velocity_mode"):
                meta_parts.append(f"Velocity: {vel}")
            if bb := meta.get("backbone_name"):
                meta_parts.append(f"Backbone: {bb}")
            if pt := meta.get("prediction_target"):
                meta_parts.append(f"PredTarget: {pt}")
            if meta_parts:
                console.print(f"[metric]Meta[/metric]: [value]{' | '.join(meta_parts)}[/value]")
        
        # Sampling frequency info (read directly from model)
        sampling_parts = []
        if hasattr(pl_module, "val_num_sample_batches"):
            val_sample = pl_module.val_num_sample_batches
            sampling_parts.append(f"Recon: {val_sample if val_sample else 'all'}")
        if hasattr(pl_module, "trajectory_num_sample_batches"):
            traj_sample = pl_module.trajectory_num_sample_batches
            sampling_parts.append(f"Traj: {traj_sample if traj_sample else 'all'}")
        if hasattr(pl_module, "train_num_sample_batches_for_reg_loss"):
            train_sample = pl_module.train_num_sample_batches_for_reg_loss
            sampling_parts.append(f"TrainReg: {train_sample if train_sample else 'all'}")
        if sampling_parts:
            console.print(f"[metric]Sampling[/metric]: [value]{' | '.join(sampling_parts)}[/value]")

    def _collect_metrics(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> Dict[str, Any]:
        """Collect validation metrics from trainer and model."""
        # Aggregate loss statistics from per-step losses
        avg_loss, loss_std, loss_min, loss_max, num_batches = (
            float("nan"), float("nan"), float("nan"), float("nan"), 0
        )

        if hasattr(pl_module, "_val_step_losses") and pl_module._val_step_losses:
            losses = pl_module._val_step_losses
            try:
                if isinstance(losses[0], (int, float)):
                    import numpy as np
                    arr = np.array(losses, dtype=np.float32)
                    avg_loss, loss_std = float(np.mean(arr)), float(np.std(arr))
                    loss_min, loss_max = float(np.min(arr)), float(np.max(arr))
                    num_batches = len(arr)
                else:
                    t = torch.stack(list(losses)).float().detach().cpu()
                    avg_loss, loss_std = float(t.mean()), float(t.std(unbiased=False))
                    loss_min, loss_max = float(t.min()), float(t.max())
                    num_batches = int(t.numel())
            except Exception:
                pass

        metrics = dict(trainer.callback_metrics)

        def get_metric(name: str) -> Optional[float]:
            v = metrics.get(name)
            if v is None:
                return None
            try:
                return float(v.item() if hasattr(v, "item") else v)
            except Exception:
                return None

        # Detailed loss breakdown
        val_detailed_loss = {}
        for name in metrics:
            if name == "val_loss" or name.startswith("val/loss"):
                if (v := get_metric(name)) is not None and math.isfinite(v):
                    key = name.split("/", 1)[-1] if "/" in name else name
                    if key not in val_detailed_loss:
                        val_detailed_loss[key] = v

        # Flow metrics (including trajectory quality metrics)
        val_flow_metrics = {}
        for key in (
            "val/flow_loss", "val/flow_loss_fm", "val/flow_loss_tangent",
            "val/flow_edge_len_err", "val/flow_total",
            # Trajectory quality metrics
            "val/flow_plr",                  # Path Length Ratio (arc_length / euclidean)
            "val/flow_transport_cost",       # Kinetic energy (integral of ||v||^2)
            "val/flow_velocity_consistency", # Avg cosine similarity between adjacent v
        ):
            if (v := get_metric(key)) is not None:
                val_flow_metrics[key.split("/", 1)[-1]] = v

        # Quality metrics (recon)
        val_quality_metrics = {}
        for key in (
            "val/recon_l1",
            "val/recon_chamfer",
            "val/recon_direction",
            "val/recon_edge_len",  # actual logged name
            "val/recon_edge_len_err",  # backward compatibility if ever used
            "val/recon_collision",  # collision with scene (cm space)
            "val/recon_total",
        ):
            if (v := get_metric(key)) is not None:
                val_quality_metrics[key.split("/", 1)[-1]] = v

        # Regression loss metrics (when lambda_regression > 0)
        val_regression_metrics = {}
        for key in (
            "val/loss_regression",
            "val/reg_l1",
            "val/reg_chamfer",
            "val/reg_direction",
            "val/reg_edge_len",
            "val/reg_bone",
            "val/reg_collision",
            "val/reg_fingertip_contact",
            "val/reg_active_points_repulsion",
        ):
            if (v := get_metric(key)) is not None:
                val_regression_metrics[key.split("/", 1)[-1]] = v

        return {
            "num_batches": num_batches,
            "avg_loss": avg_loss,
            "loss_std": loss_std,
            "loss_min": loss_min,
            "loss_max": loss_max,
            "val_detailed_loss": val_detailed_loss,
            "val_flow_metrics": val_flow_metrics,
            "val_quality_metrics": val_quality_metrics,
            "val_regression_metrics": val_regression_metrics,
            "val_set_metrics": {},
            "val_diversity_metrics": {},
        }

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0))

        # Collect memory and log to tensorboard
        mem_info = self._collect_memory(pl_module)
        for key, value in (mem_info or {}).items():
            pl_module.log(f"memory/{key}", value, on_epoch=True, sync_dist=False)

        # Print unified run info
        if getattr(trainer, "is_global_zero", True):
            self._print_run_info(pl_module, epoch)

        # Collect and print metrics
        m = self._collect_metrics(trainer, pl_module)
        log_validation_summary(
            epoch=epoch,
            num_batches=m["num_batches"],
            avg_loss=m["avg_loss"],
            loss_std=m["loss_std"],
            loss_min=m["loss_min"],
            loss_max=m["loss_max"],
            val_detailed_loss=m["val_detailed_loss"],
            val_set_metrics=m["val_set_metrics"],
            val_quality_metrics=m["val_quality_metrics"],
            val_diversity_metrics=m["val_diversity_metrics"],
            val_flow_metrics=m["val_flow_metrics"],
            val_regression_metrics=m["val_regression_metrics"],
        )

        # Reset validation loss buffer
        if hasattr(pl_module, "_val_step_losses"):
            try:
                pl_module._val_step_losses = []
            except Exception:
                pass
