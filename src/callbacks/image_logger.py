
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

log = logging.getLogger(__name__)


class ImageLogger(Callback):
    """Logs validation samples (GT vs Predicted) to the logger."""

    def __init__(
        self,
        num_samples: int = 4,
        log_key: str = "val/hand_samples",
        sample_steps: Optional[int] = None,
    ):
        """
        Args:
            num_samples: Number of samples to visualize per validation epoch.
            log_key: Key/tag for logging images.
            sample_steps: Number of ODE steps for sampling. If None, uses model default.
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_key = log_key
        self.sample_steps = sample_steps
        self._val_vis_batches: List[Dict[str, torch.Tensor]] = []

    def on_validation_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        """Reset the visualization buffer at the start of validation."""
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
        if len(self._val_vis_batches) >= self.num_samples:
            return

        # Assuming batch is a dict with 'xyz' and optionally 'scene_pc'
        if not isinstance(batch, dict) or "xyz" not in batch:
            return

        with torch.no_grad():
            xyz = batch["xyz"].detach().cpu()
            scene_pc = batch.get("scene_pc", None)
            if scene_pc is not None:
                scene_pc = scene_pc.detach().cpu()
            
            # Store individual samples to reach exactly num_samples
            # But here we store the whole batch to process later for efficiency
            # or just store the first item?
            # To keep it simple, let's just append the first item of each batch
            # until we have enough.
            
            # Actually, let's store the batch tensors and slice later.
            self._val_vis_batches.append({"xyz": xyz, "scene_pc": scene_pc})

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        """Generate samples and log images."""
        if not self._val_vis_batches:
            return

        logger = trainer.logger
        if logger is None:
            return

        # Flatten batches into a list of samples
        samples_to_vis = []
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

        if not samples_to_vis:
            return

        device = pl_module.device
        images: List[np.ndarray] = []
        captions: List[str] = []

        # Get edge_index for plotting
        # Assuming model has edge_index buffer or attribute
        edge_index = getattr(pl_module, "edge_index", None)
        if edge_index is None:
            log.warning("Model does not have 'edge_index', cannot plot edges.")
            edges = np.empty((2, 0), dtype=int)
        else:
            edges = edge_index.detach().cpu().numpy()

        # Sampling
        # We need to move samples to device for inference
        pl_module.eval()
        
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
                    # Handle optional args for sample
                    kwargs = {}
                    if self.sample_steps is not None:
                        kwargs["num_steps"] = self.sample_steps
                    
                    pred_xyz = pl_module.sample(scene_pc, **kwargs).detach().cpu()
                    gt_xyz_cpu = gt_xyz.detach().cpu()
            except Exception as e:
                log.error(f"Error sampling visualization: {e}")
                continue

            # Render
            img = self._render_hand_pair_image(
                gt_xyz_cpu[0], pred_xyz[0], edges, title_suffix=f"S{idx}"
            )
            images.append(img)
            captions.append(f"epoch={trainer.current_epoch}, sample={idx}")

        # Log images
        self._log_images(logger, images, captions, trainer.current_epoch)

    def _render_hand_pair_image(
        self,
        gt_xyz: torch.Tensor,
        pred_xyz: torch.Tensor,
        edges: np.ndarray,
        title_suffix: str = "",
    ) -> np.ndarray:
        """Render a side-by-side image of GT vs predicted hand keypoints."""
        gt = gt_xyz.numpy()
        pred = pred_xyz.numpy()

        fig = plt.figure(figsize=(6, 3))

        def _plot(subplot_idx: int, pts: np.ndarray, title: str, color_points: str, color_edges: str):
            ax = fig.add_subplot(1, 2, subplot_idx, projection="3d")
            # Center view
            # ax.view_init(elev=..., azim=...)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color_points, s=10)
            
            if edges.shape[1] > 0:
                for e in edges.T:
                    i, j = int(e[0]), int(e[1])
                    # Simple bounds check
                    if i < pts.shape[0] and j < pts.shape[0]:
                        xs = [pts[i, 0], pts[j, 0]]
                        ys = [pts[i, 1], pts[j, 1]]
                        zs = [pts[i, 2], pts[j, 2]]
                        ax.plot(xs, ys, zs, color=color_edges, linewidth=0.5)
            
            ax.set_title(title)
            ax.set_axis_off()

        _plot(1, gt, f"GT {title_suffix}", "#1f77b4", "#1f77b4")
        _plot(2, pred, f"Pred {title_suffix}", "#d62728", "#d62728")
        
        plt.tight_layout()
        fig.canvas.draw()

        # Convert to numpy (backend-agnostic)
        width, height = fig.canvas.get_width_height()
        buffer = fig.canvas.buffer_rgba()
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        plt.close(fig)
        return image

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
