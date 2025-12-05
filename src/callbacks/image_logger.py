
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as L
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)


class ImageLogger(Callback):
    """Logs validation samples (GT vs Predicted) to the logger and/or disk."""

    def __init__(
        self,
        num_samples: int = 5,
        log_key: str = "val/hand_samples",
        sample_steps: Optional[int] = None,
        save_to_logger: bool = False,
        save_to_disk: bool = True,
        output_dir: Optional[str] = None,
        html_subdir: str = "val_visualizations",
        scene_pc_max_points: int = 2048,
    ):
        """
        Args:
            num_samples: Number of samples to visualize per validation epoch.
            log_key: Key/tag for logging images.
            sample_steps: Number of ODE steps for sampling. If None, uses model default.
            save_to_logger: Whether to log rendered figures to the configured logger.
            save_to_disk: Whether to save Plotly HTML files locally.
            output_dir: Base directory for saving HTML files. Defaults to trainer/Logger dir.
            html_subdir: Subdirectory (under output_dir) for validation visualizations.
            scene_pc_max_points: Max number of scene points to plot (downsamples for speed).
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_key = log_key
        self.sample_steps = sample_steps
        self.save_to_logger = save_to_logger
        self.save_to_disk = save_to_disk
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.html_subdir = html_subdir
        self.scene_pc_max_points = scene_pc_max_points
        self._val_vis_batches: List[Dict[str, torch.Tensor]] = []
        self._fixed_samples: List[Dict[str, torch.Tensor]] = []

    def on_validation_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        """Reset the visualization buffer at the start of validation."""
        if not self._fixed_samples:
            self._val_vis_batches = []

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect a few batches for visualization."""
        if self._fixed_samples or len(self._val_vis_batches) >= self.num_samples:
            return

        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        # Assuming batch is a dict with 'xyz' and optionally 'scene_pc'
        if not isinstance(batch, dict) or "xyz" not in batch:
            return

        with torch.no_grad():
            use_norm_data = bool(getattr(pl_module, "use_norm_data", False))
            xyz_key = "norm_xyz" if use_norm_data and "norm_xyz" in batch else "xyz"
            scene_key = (
                "norm_scene_pc" if use_norm_data and "norm_scene_pc" in batch else "scene_pc"
            )

            xyz = batch[xyz_key].detach().cpu()
            scene_pc = batch.get(scene_key, None)
            if scene_pc is not None:
                scene_pc = scene_pc.detach().cpu()
            self._val_vis_batches.append({"xyz": xyz, "scene_pc": scene_pc})

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        """Generate samples and log images."""
        if not self.save_to_disk and not self.save_to_logger:
            return

        if not self._val_vis_batches and not self._fixed_samples:
            return

        logger = trainer.logger
        if logger is None:
            return

        samples_to_vis = self._get_samples_to_visualize()

        if not samples_to_vis:
            return

        device = pl_module.device
        images: List[np.ndarray] = []
        figs: List[go.Figure] = []
        captions: List[str] = []

        edge_index = getattr(pl_module, "edge_index", None)
        if edge_index is None:
            log.warning("Model does not have 'edge_index', cannot plot edges.")
            edges = np.empty((2, 0), dtype=int)
        else:
            edges = edge_index.detach().cpu().numpy()

        pl_module.eval()
        use_norm_data = bool(getattr(pl_module, "use_norm_data", False))

        for idx, item in enumerate(samples_to_vis):
            gt_xyz = item["xyz"].to(device).unsqueeze(0)  # (1, N, 3)
            scene_pc = item["scene_pc"]
            
            if scene_pc is None:
                scene_pc = torch.zeros(1, 0, 3, device=device, dtype=gt_xyz.dtype)
            else:
                scene_pc = scene_pc.to(device).unsqueeze(0)

            # Call model.sample()
            # Check if model has sample method
            if not hasattr(pl_module, "sample"):
                log.warning("Model does not implement sample(), skipping visualization.")
                return

            try:
                with torch.no_grad():
                    kwargs = {}
                    if self.sample_steps is not None:
                        kwargs["num_steps"] = self.sample_steps
                    
                    pred_xyz = pl_module.sample(scene_pc, **kwargs).detach().cpu()
                    gt_xyz_cpu = gt_xyz.detach().cpu()
            except Exception as e:
                log.error(f"Error sampling visualization: {e}")
                continue

            scene_pc_cpu = scene_pc.detach().cpu()[0] if scene_pc is not None else None
            pred_vis = pred_xyz[0]
            gt_vis = gt_xyz_cpu[0]

            if use_norm_data:
                pred_vis = self._normalize_points(pred_vis, pl_module)
                gt_vis = self._normalize_points(gt_vis, pl_module)
                if scene_pc_cpu is not None:
                    scene_pc_cpu = self._normalize_points(scene_pc_cpu, pl_module)

            if scene_pc_cpu is not None:
                scene_pc_cpu = self._prepare_scene_pc(scene_pc_cpu)

            fig = self._render_hand_pair_figure(
                gt_vis,
                pred_vis,
                edges,
                scene_pc=scene_pc_cpu,
                title_suffix=f"S{idx}",
            )
            figs.append(fig)
            captions.append(f"epoch={trainer.current_epoch}, sample={idx}")

            if self.save_to_logger:
                img = self._figure_to_image(fig)
                if img is not None:
                    images.append(img)

        epoch_dir = self._get_epoch_dir(trainer)
        if self.save_to_disk and figs:
            self._save_figures_as_html(figs, epoch_dir)

        if self.save_to_logger and images:
            self._log_images(logger, images, captions, trainer.current_epoch)

    def _get_samples_to_visualize(self) -> List[Dict[str, torch.Tensor]]:
        """Select and cache samples for visualization (fixed across epochs)."""
        if self._fixed_samples:
            return self._fixed_samples

        samples_to_vis: List[Dict[str, torch.Tensor]] = []
        for entry in self._val_vis_batches:
            B = entry["xyz"].shape[0]
            for i in range(B):
                samples_to_vis.append(
                    {
                        "xyz": entry["xyz"][i],
                        "scene_pc": entry["scene_pc"][i] if entry["scene_pc"] is not None else None,
                    }
                )
                if len(samples_to_vis) >= self.num_samples:
                    break
            if len(samples_to_vis) >= self.num_samples:
                break

        if len(samples_to_vis) == self.num_samples:
            self._fixed_samples = samples_to_vis
            self._val_vis_batches = []
        else:
            log.warning(
                "ImageLogger collected %d/%d samples; will try again next epoch.",
                len(samples_to_vis),
                self.num_samples,
            )
            return []
        return samples_to_vis

    def _render_hand_pair_figure(
        self,
        gt_xyz: torch.Tensor,
        pred_xyz: torch.Tensor,
        edges: np.ndarray,
        title_suffix: str = "",
        scene_pc: Optional[torch.Tensor] = None,
    ) -> go.Figure:
        """Render a side-by-side Plotly figure of GT vs predicted hand keypoints."""
        gt = gt_xyz.numpy()
        pred = pred_xyz.numpy()
        scene_np = scene_pc.numpy() if scene_pc is not None and scene_pc.numel() > 0 else None

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=(f"GT {title_suffix}", f"Pred {title_suffix}"),
        )

        def _edge_lines(pts: np.ndarray) -> Dict[str, List[float]]:
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            if edges.shape[1] == 0:
                return {"x": xs, "y": ys, "z": zs}
            for e in edges.T:
                i, j = int(e[0]), int(e[1])
                if i < pts.shape[0] and j < pts.shape[0]:
                    xs.extend([pts[i, 0], pts[j, 0], None])
                    ys.extend([pts[i, 1], pts[j, 1], None])
                    zs.extend([pts[i, 2], pts[j, 2], None])
            return {"x": xs, "y": ys, "z": zs}

        def _add_scene(col: int, pts: np.ndarray, color: str) -> None:
            lines = _edge_lines(pts)
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=4, color=color),
                    name="keypoints",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            if lines["x"]:
                fig.add_trace(
                    go.Scatter3d(
                        **lines,
                        mode="lines",
                        line=dict(color=color, width=3),
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
            if scene_np is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=scene_np[:, 0],
                        y=scene_np[:, 1],
                        z=scene_np[:, 2],
                        mode="markers",
                        marker=dict(size=2, color="#999999", opacity=0.35),
                        name="scene",
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )

        _add_scene(1, gt, "#1f77b4")
        _add_scene(2, pred, "#d62728")

        fig.update_layout(
            height=520,
            width=1100,
            margin=dict(l=10, r=10, b=10, t=40),
            scene=dict(aspectmode="data"),
            scene2=dict(aspectmode="data"),
            template="plotly_white",
        )
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        return fig

    def _figure_to_image(self, fig: go.Figure) -> Optional[np.ndarray]:
        """Convert Plotly figure to HWC uint8 image for logger."""
        try:
            from PIL import Image
        except Exception as e:
            log.warning(f"Pillow not available, skip logging images: {e}")
            return None

    def _normalize_points(self, pts: torch.Tensor, pl_module: "L.LightningModule") -> torch.Tensor:
        """Normalize xyz coordinates to [-1, 1] using model bounds (keeps extras)."""
        norm_min = getattr(pl_module, "norm_min", None)
        norm_max = getattr(pl_module, "norm_max", None)
        if norm_min is None or norm_max is None:
            return pts

        view_shape = [1] * pts.dim()
        view_shape[-1] = 3
        nm = norm_min.view(view_shape).to(device=pts.device, dtype=pts.dtype)
        sc = (2.0 / torch.clamp(norm_max - norm_min, min=1e-6)).view(view_shape).to(
            device=pts.device, dtype=pts.dtype
        )
        xyz = pts[..., :3]
        norm = (xyz - nm) * sc - 1.0
        if pts.shape[-1] > 3:
            return torch.cat([norm, pts[..., 3:]], dim=-1)
        return norm

    def _prepare_scene_pc(self, scene_pc: torch.Tensor) -> torch.Tensor:
        """Downsample scene point cloud for faster plotting."""
        if self.scene_pc_max_points and scene_pc.shape[0] > self.scene_pc_max_points:
            idx = torch.randperm(scene_pc.shape[0])[: self.scene_pc_max_points]
            scene_pc = scene_pc[idx]
        return scene_pc
        try:
            img_bytes = pio.to_image(fig, format="png")
            with BytesIO(img_bytes) as buf:
                arr = np.array(Image.open(buf).convert("RGB"))
            return arr
        except Exception as e:
            log.warning(f"Failed to convert Plotly figure to image: {e}")
            return None

    def _get_epoch_dir(self, trainer: "L.Trainer") -> Path:
        """Resolve directory for saving current epoch visualizations."""
        if self.output_dir is not None:
            base_dir = self.output_dir
        else:
            logger_dir = getattr(trainer.logger, "log_dir", None) or getattr(
                trainer.logger, "save_dir", None
            )
            if logger_dir:
                base_dir = Path(logger_dir)
            else:
                base_dir = Path(getattr(trainer, "default_root_dir", Path.cwd()))
        epoch_dir = Path(base_dir) / self.html_subdir / f"epoch_{trainer.current_epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        return epoch_dir

    def _save_figures_as_html(self, figs: List[go.Figure], epoch_dir: Path) -> None:
        """Save Plotly figures as standalone HTML files."""
        for idx, fig in enumerate(figs):
            path = epoch_dir / f"sample_{idx:02d}.html"
            try:
                fig.write_html(path, include_plotlyjs="cdn", full_html=True)
            except Exception as e:
                log.error(f"Failed to save visualization HTML {path}: {e}")

    def _log_images(
        self,
        logger: Logger,
        images: List[np.ndarray],
        captions: List[str],
        step: int,
    ) -> None:
        """Dispatch logging to specific logger implementations."""
        # WandB
        if isinstance(logger, WandbLogger) or hasattr(logger, "log_image"):
            # WandB usually expects a list of images or single image
            # Some interfaces differ.
            try:
                logger.log_image(key=self.log_key, images=images, caption=captions, step=step)
            except AttributeError:
                # Fallback for loggers that might have slightly different API
                pass
        
        # TensorBoard
        elif isinstance(logger, TensorBoardLogger):
            for i, img in enumerate(images):
                # TensorBoard expects CHW usually? add_image doc says:
                # img_tensor: Default is (3, H, W). You can use dataformats="HWC"
                logger.experiment.add_image(
                    f"{self.log_key}/{i}",
                    img,
                    global_step=step,
                    dataformats="HWC",
                )
