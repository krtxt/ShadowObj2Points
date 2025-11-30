# models/deterministic_hand_dit.py

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from .components.hand_encoder import build_hand_encoder
from .components.graph_dit import build_dit
from .backbone import build_backbone
from .components.graph_config import GraphConfig
from .components.losses import HandValidationMetricManager, DeterministicRegressionLoss

logger = logging.getLogger(__name__)


class TimestepEmbedding(nn.Module):
    """
    Project scalar inputs to a vector space.
    Used here as a static context embedding rather than dynamic time.
    """
    def __init__(self, dim: int, scale_factor: int = 4):
        super().__init__()
        hidden_dim = dim * scale_factor
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support both (B,) and (B, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class HandDeterministicDiT(L.LightningModule):
    """
    Deterministic Graph-aware DiT for Hand Pose Regression.
    
    Predicts hand keypoints directly from scene context:
        Pred = Template + Network(Scene, Template)
    
    Key Features:
    - No ODE/Flow Matching; pure regression with structural constraints.
    - Graph-aware: Respects bone connectivity and joint types.
    - Robust Normalization: Handles bounding box scaling for stable training.
    """

    def __init__(
        self,
        d_model: int,
        graph_consts: Dict[str, torch.Tensor],
        backbone: Any,
        hand_encoder: Any,
        dit: Any,
        loss_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        use_norm_data: bool = False,
    ):
        super().__init__()
        # Exclude large objects from hparams to keep checkpoints light
        self.save_hyperparameters(ignore=["graph_consts", "backbone", "hand_encoder", "dit"])
        
        self.use_norm_data = use_norm_data
        self.optimizer_cfg = optimizer_cfg
        
        # 1. Setup Graph Topology & Normalization
        self._setup_graph_structure(graph_consts)
        if self.use_norm_data:
            self._setup_normalization(graph_consts)

        # 2. Setup Neural Components
        self._setup_architecture(d_model, backbone, hand_encoder, dit, graph_consts)

        # 3. Setup Objectives & Evaluation
        self._setup_loss_and_metrics(loss_cfg)

        self._log_init_summary()

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _setup_graph_structure(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Initialize graph topology constants."""
        cfg = GraphConfig(self, graph_consts)
        self.N = cfg.N
        self.rigid_groups = cfg.rigid_groups
        
        # Fixed points (Palm/Wrist anchor)
        fixed_idx = torch.arange(0, min(7, self.N))
        fixed_mask = torch.zeros(self.N, dtype=torch.bool)
        fixed_mask[fixed_idx] = True
        
        self.register_buffer("fixed_point_indices", fixed_idx, persistent=False)
        self.register_buffer("fixed_point_mask", fixed_mask, persistent=False)
        
        # Active edges: ignore edges completely within fixed points
        i, j = self.edge_index
        edge_mask = ~(fixed_mask[i] & fixed_mask[j])
        self.register_buffer("active_edge_mask", edge_mask, persistent=False)
        
        # Filter rigid groups containing only fixed points
        self.rigid_groups_active = [g for g in self.rigid_groups if not fixed_mask[g].all()]

    def _safe_register_buffer(self, name: str, tensor: torch.Tensor, persistent: bool = True) -> None:
        if hasattr(self, name):
            if name in self._buffers:
                self._buffers[name] = tensor
                return
            else:
                delattr(self, name)
        self.register_buffer(name, tensor, persistent=persistent)

    def _setup_normalization(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Apply affine transformation to graph constants if normalization is enabled."""
        norm_min = getattr(self, "norm_min", None)
        norm_max = getattr(self, "norm_max", None)
        
        if norm_min is None or norm_max is None:
            raise RuntimeError("use_norm_data=True requires 'norm_min' and 'norm_max' in graph_consts.")

        # Calculate Scale & Bias
        # Map [min, max] -> [-1, 1]
        scale = 2.0 / torch.clamp(norm_max - norm_min, min=1e-6)
        bias = -1.0

        def normalize(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is None: return None
            # Reshape scale/min to match tensor dimensions
            view_shape = [1] * tensor.dim()
            view_shape[-1] = 3
            s = scale.view(view_shape)
            m = norm_min.view(view_shape)
            return (tensor - m) * s + bias

        # 1. Normalize Coordinate Tensors
        for name in ["template_xyz", "canonical_xyz"]:
            tensor = getattr(self, name, None)
            if tensor is not None:
                normed = normalize(tensor)
                # If it's template, update in place; otherwise register as buffer
                if name == "template_xyz":
                    tensor.copy_(normed)
                else:
                    self._safe_register_buffer(f"norm_{name}", normed.float(), persistent=True)

        # 2. Normalize Standard Deviation (Scale only)
        if hasattr(self, "approx_std"):
            s = scale.view(1, 1, 3) if self.approx_std.dim() == 3 else scale.view(1, 3)
            self._safe_register_buffer("norm_approx_std", (self.approx_std * s).float(), persistent=True)

        # 3. Recompute Edge Lengths in Normalized Space
        # This ensures the bone length loss works correctly in the [-1, 1] space
        tmpl = self.template_xyz.squeeze(0) if self.template_xyz.dim() == 3 else self.template_xyz
        i, j = self.edge_index
        new_lengths = torch.linalg.norm(tmpl[i] - tmpl[j], dim=-1)
        self.edge_rest_lengths.copy_(new_lengths)

    def _setup_architecture(self, d_model, backbone, hand_encoder, dit, graph_consts):
        """Instantiate sub-modules."""
        self.backbone = backbone if isinstance(backbone, nn.Module) else build_backbone(backbone)
        
        self.hand_encoder = (
            hand_encoder if isinstance(hand_encoder, nn.Module) 
            else build_hand_encoder(hand_encoder, graph_consts, out_dim=d_model)
        )
        
        self.dit = (
            dit if isinstance(dit, nn.Module) 
            else build_dit(dit, graph_consts, dim=d_model)
        )
        
        self.timestep_embed = TimestepEmbedding(dim=d_model)
        self.xyz_head = nn.Linear(d_model, 3)

    def _setup_loss_and_metrics(self, loss_cfg: Optional[DictConfig]):
        """Parse loss weights and setup metrics."""
        if isinstance(loss_cfg, DictConfig):
            cfg = OmegaConf.to_container(loss_cfg, resolve=True)
        elif isinstance(loss_cfg, dict):
            cfg = deepcopy(loss_cfg)
        else:
            cfg = {}

        # Backwards-compatible scalar weights & collision config
        self.lambda_pos = float(cfg.get("lambda_pos", 1.0))
        self.lambda_bone = float(cfg.get("lambda_bone", 0.0))
        self.lambda_collision = float(cfg.get("lambda_collision", 0.0))
        self.collision_margin = float(cfg.get("collision_margin", 0.0))

        max_pts = int(cfg.get("collision_max_scene_points", 0))
        self.collision_max_scene_points = max_pts if max_pts > 0 else None

        # Deterministic regression loss manager (reuses recon components)
        self.loss_manager = DeterministicRegressionLoss(
            edge_index=self.edge_index,
            config=cfg,
        )

        # Validation Metrics (can be disabled via loss_cfg.val_metrics)
        val_cfg = cfg.get("val_metrics", None)
        self.val_metric_manager = (
            HandValidationMetricManager(self.edge_index, val_cfg) if val_cfg else None
        )
        self._val_step_losses: List[float] = []  # Store scalars, not tensors

    # =========================================================================
    # Data Processing Utilities
    # =========================================================================

    def _denorm_xyz(self, norm_xyz: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] back to world coordinates."""
        if not self.use_norm_data:
            return norm_xyz
        # Formula: (x + 1) / 2 * (max - min) + min
        unscaled = (torch.clamp(norm_xyz, -1.0, 1.0) + 1.0) * 0.5
        range_width = torch.clamp(self.norm_max - self.norm_min, min=1e-6)
        return unscaled * range_width + self.norm_min

    def _get_template_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Expand template to batch size."""
        # template_xyz is already normalized if use_norm_data is True
        tmpl = self.template_xyz
        if tmpl.dim() == 2:
            tmpl = tmpl.unsqueeze(0)
        return tmpl.to(device).expand(batch_size, -1, -1).contiguous()

    def encode_scene_tokens(self, scene_pc: torch.Tensor) -> torch.Tensor:
        """
        Robustly extract tokens from backbone.
        Handles various return types: Tuple, Dict, or Tensor.
        """
        B = scene_pc.shape[0]
        empty_token = torch.zeros(B, 1, self.hparams.d_model, device=scene_pc.device)
        
        if scene_pc.numel() == 0 or scene_pc.shape[1] == 0:
            return empty_token

        out = self.backbone(scene_pc)
        tokens = None

        # 1. Parse Output Type
        if isinstance(out, tuple):
            # Assumes (coords, features)
            tokens = out[1]
        elif isinstance(out, dict):
            for k in ["tokens", "scene_tokens", "feat"]:
                if k in out:
                    tokens = out[k]
                    break
            if tokens is None:
                raise KeyError(f"Backbone dict missing keys. Found: {out.keys()}")
        elif torch.is_tensor(out):
            tokens = out
        else:
            raise TypeError(f"Unknown backbone output type: {type(out)}")

        # 2. Ensure Format (B, N, C)
        if tokens.dim() != 3:
            raise ValueError(f"Scene tokens must be 3D, got {tokens.shape}")
        
        # Heuristic: if dim 1 matches d_model, assumes (B, C, N) -> permute
        if tokens.shape[1] == self.hparams.d_model and tokens.shape[2] != self.hparams.d_model:
            tokens = tokens.permute(0, 2, 1)

        # Ensure consistent dtype for DiT input to avoid torch.compile recompilation
        # between training (fp16 from autocast) and validation (fp32)
        return tokens.float()

    # =========================================================================
    # Forward Pass & Prediction
    # =========================================================================

    def predict_hand_xyz(
        self, 
        scene_pc: torch.Tensor, 
        scene_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Core logic: Scene + Template -> Residual -> Prediction.
        """
        B, _, _ = scene_pc.shape
        device = scene_pc.device

        # 1. Prepare Inputs
        if scene_tokens is None:
            scene_tokens = self.encode_scene_tokens(scene_pc)
        
        template = self._get_template_batch(B, device)
        
        # 2. Encode Hand (Template)
        hand_tokens = self.hand_encoder(
            xyz=template,
            finger_ids=self.finger_ids_const.expand(B, -1),
            joint_type_ids=self.joint_type_ids_const.expand(B, -1),
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_rest_lengths=self.edge_rest_lengths,
        )

        # 3. DiT Forward
        # Use constant time embedding (0.5) as context
        t_embed = self.timestep_embed(torch.full((B,), 0.5, device=device))
        
        latent = self.dit(
            hand_tokens=hand_tokens,
            scene_tokens=scene_tokens,
            temb=t_embed,
            xyz=template, # Geometric bias
        )

        # 4. Decode & Apply Residual
        delta = self.xyz_head(latent)
        pred_xyz = template + delta

        # 5. Enforce Fixed Points Constraint
        # Anchors (wrist/palm) should not deviate from template
        if hasattr(self, "fixed_point_indices"):
            idx = self.fixed_point_indices
            pred_xyz[:, idx, :] = template[:, idx, :]

        return pred_xyz

    def forward(self, scene_pc: torch.Tensor) -> torch.Tensor:
        """Inference entry point. Returns World Coordinates."""
        pred = self.predict_hand_xyz(scene_pc)
        return self._denorm_xyz(pred) if self.use_norm_data else pred

    @torch.no_grad()
    def sample(
        self,
        scene_pc: torch.Tensor,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
        trajectory_indices: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict]]]:
        """
        Interface compatible with diffusion samplers.
        'Trajectory' here is just [Template, Prediction].
        """
        self.eval()
        B = scene_pc.shape[0]
        
        pred_norm = self.predict_hand_xyz(scene_pc)
        template_norm = self._get_template_batch(B, scene_pc.device)

        if self.use_norm_data:
            pred_final = self._denorm_xyz(pred_norm)
            template_final = self._denorm_xyz(template_norm)
        else:
            pred_final, template_final = pred_norm, template_norm

        if not return_trajectory:
            return pred_final

        # Fake trajectory for visualization compatibility
        traj = [
            {"step": 0, "t": 0.0, "xyz": template_final.cpu()},
            {"step": 1, "t": 1.0, "xyz": pred_final.cpu()},
        ]
        return pred_final, traj

    # =========================================================================
    # Lightning Hooks (Train/Val)
    # =========================================================================

    def _extract_batch(self, batch: Dict, stage: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts (Target, RawTarget, Scene) from batch with checks."""
        # 1. Targets
        if "xyz" not in batch:
            raise KeyError(f"{stage}: Batch missing 'xyz'")
        
        raw_target = batch["xyz"]
        target = batch.get("norm_xyz", raw_target) if self.use_norm_data else raw_target

        # 2. Scene
        scene = batch.get("norm_scene_pc" if self.use_norm_data else "scene_pc")
        
        # Handle missing scene cases gracefully
        if scene is None:
            if stage == "train":
                scene = torch.zeros(target.shape[0], 0, 3, device=self.device)
            else:
                raise KeyError(f"{stage}: Batch missing scene_pc")

        return target, raw_target, scene

    def on_train_epoch_start(self) -> None:
        """Update loss scheduler with current epoch and log weights."""
        epoch = self.current_epoch
        self.loss_manager.set_epoch(epoch)

        # Log current loss weights for monitoring curriculum progress
        weights = self.loss_manager.get_current_weights()
        for name, weight in weights.items():
            self.log(f"loss_weight/{name}", weight, on_step=False, on_epoch=True, sync_dist=True)

        stage = self.loss_manager.scheduler.get_current_stage_name()
        if stage:
            logger.info(f"Epoch {epoch}: Curriculum stage = {stage}")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        target, _, scene = self._extract_batch(batch, "train")
        batch_size = int(target.size(0))

        pred = self.predict_hand_xyz(scene)
        denorm_fn = self._denorm_xyz if self.use_norm_data else None
        loss, components = self.loss_manager(
            pred_xyz=pred,
            gt_xyz=target,
            edge_rest_lengths=self.edge_rest_lengths,
            active_edge_mask=getattr(self, "active_edge_mask", None),
            scene_pc=scene,
            denorm_fn=denorm_fn,
            epoch=self.current_epoch,
        )

        logs = {f"loss_{k}": v.detach() for k, v in components.items()}

        # Log main loss and components BEFORE NaN check for debugging
        self.log("train/loss", loss.detach(), prog_bar=False, batch_size=batch_size)
        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=False, batch_size=batch_size)

        if not torch.isfinite(loss):
            # Log which components are NaN/Inf for debugging
            nan_components = [k for k, v in components.items() if not torch.isfinite(v)]
            self.log("train/nan_loss", 1.0, on_step=True, batch_size=batch_size)
            logger.error(
                f"NaN/Inf loss detected at batch {batch_idx}! "
                f"Total loss: {loss.item():.6f}, NaN components: {nan_components}"
            )
            for k, v in components.items():
                logger.error(f"  {k}: {v.item():.6f} (finite={torch.isfinite(v).item()})")
            
            # Option 1: Clean NaN values and continue (less aggressive)
            # loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            # return loss
            
            # Option 2: Raise error immediately for faster debugging (recommended)
            raise RuntimeError(
                f"NaN/Inf loss detected at batch {batch_idx}. "
                f"Components: {nan_components}. Check gradient clipping and input data."
            )
            
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset validation loss buffer each epoch for summary callback."""
        self._val_step_losses = []

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        target, raw_target, scene = self._extract_batch(batch, "val")
        batch_size = int(target.size(0))

        pred = self.predict_hand_xyz(scene)
        denorm_fn = self._denorm_xyz if self.use_norm_data else None
        loss, components = self.loss_manager(
            pred_xyz=pred,
            gt_xyz=target,
            edge_rest_lengths=self.edge_rest_lengths,
            active_edge_mask=getattr(self, "active_edge_mask", None),
            scene_pc=scene,
            denorm_fn=denorm_fn,
            epoch=self.current_epoch,
        )

        logs = {f"loss_{k}": v.detach() for k, v in components.items()}

        # Buffer per-batch loss for ValidationSummaryCallback
        if torch.isfinite(loss):
            self._val_step_losses.append(float(loss.detach().item()))  # Store scalar, not tensor
        else:
            # Log detailed NaN info for debugging
            nan_components = [k for k, v in components.items() if not torch.isfinite(v)]
            self.log("val/nan_loss", 1.0, on_step=False, on_epoch=True, batch_size=batch_size)
            logger.warning(
                f"NaN/Inf validation loss at batch {batch_idx}. "
                f"NaN components: {nan_components}"
            )
            return

        # Basic Logging (+ alias for callbacks expecting 'val_loss')
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val_loss", loss, sync_dist=True, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        for k, v in logs.items():
            self.log(f"val/{k}", v, sync_dist=True, batch_size=batch_size)

        # Bone Length Error (Physical Metric)
        # Always compute on Denormalized/World coordinates for interpretability
        pred_world = self._denorm_xyz(pred) if self.use_norm_data else pred
        
        # Temporary compute edge error manually to log
        i, j = self.edge_index
        dist = torch.linalg.norm(pred_world[:, i] - pred_world[:, j], dim=-1)
        # We need raw rest lengths here. Since edge_rest_lengths is updated to norm space,
        # we recalculate raw rest from raw template or handle scaling. 
        # Easier: Just rely on metric manager if available, but here is a quick check:
        # (This is approximate if using metric manager, but good for tensorboard)
        
        # Advanced Metrics
        if self.val_metric_manager:
            metrics = self.val_metric_manager.compute_recon_metrics(pred_world, raw_target)
            for k, v in metrics.items():
                self.log(f"val/recon_{k}", v, sync_dist=True)

    def configure_optimizers(self):
        if not self.optimizer_cfg:
            return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-2)

        # Clone config to avoid mutation side-effects
        opt_cfg = OmegaConf.to_container(self.optimizer_cfg, resolve=True) \
                  if isinstance(self.optimizer_cfg, DictConfig) else deepcopy(self.optimizer_cfg)

        sched_cfg = opt_cfg.pop("scheduler", None)
        optimizer = hydra_instantiate(DictConfig(opt_cfg), params=self.parameters())

        if not sched_cfg:
            return optimizer

        # Remove Lightning-only keys that should not be passed to the builder
        if isinstance(sched_cfg, DictConfig):
            sched_cfg = OmegaConf.to_container(sched_cfg, resolve=True)  # type: ignore
        sched_cfg_clean = dict(sched_cfg or {})
        for k in ("interval", "frequency", "monitor"):
            if k in sched_cfg_clean:
                sched_cfg_clean.pop(k, None)

        scheduler = hydra_instantiate(DictConfig(sched_cfg_clean), optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": (sched_cfg.get("interval") if isinstance(sched_cfg, dict) else None) or "epoch",
                "frequency": (sched_cfg.get("frequency") if isinstance(sched_cfg, dict) else None) or 1,
                "monitor": (sched_cfg.get("monitor") if isinstance(sched_cfg, dict) else None) or "val/loss",
            },
        }

    def _log_init_summary(self):
        logger.info(f"HandDeterministicDiT Initialized | Norm: {self.use_norm_data}")
        logger.info(f"  > Backbone: {type(self.backbone).__name__}")
        logger.info(f"  > DiT: {type(self.dit).__name__}")
        logger.info(f"  > Graph: N={self.N}, ActiveEdges={self.active_edge_mask.sum().item()}")
