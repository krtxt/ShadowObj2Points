"""Flow Matching model with Graph-aware DiT for dexterous hand keypoint generation."""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
import numpy as np
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from .components.losses import FlowMatchingLoss, HandValidationMetricManager
from .components.hand_encoder import build_hand_encoder
from .components.graph_dit import build_dit
from .backbone import build_backbone
from .components.velocity_strategies import build_velocity_strategy
from .components.graph_config import GraphConfig
from .components.sampling_config import SamplingConfig
import logging


class TimestepEmbedding(nn.Module):
    """Maps scalar timesteps t ∈ [0,1] to embedding vectors (B, dim).
    
    Args:
        dim: Output embedding dimension
        scale_factor: Hidden layer expansion factor (default: 4)
    """
    def __init__(self, dim: int, scale_factor: int = 4):
        super().__init__()
        hidden_dim = dim * scale_factor
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) or (B, 1) timestep values in [0, 1]
            
        Returns:
            Timestep embeddings of shape (B, dim)
        """
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)
        return self.net(timesteps)

class HandFlowMatchingDiT(L.LightningModule):
    """Flow Matching model with Graph-aware DiT for hand keypoint generation.
    
    Generates hand keypoints conditioned on scene point clouds using rectified flow.
    
    Architecture:
        - Hand encoder: Encodes current state into per-point tokens (B, N, D)
        - Scene backbone: Processes point cloud into scene tokens (B, K, D)
        - Graph DiT: Self-attention with graph bias + cross-attention with scene
        - Velocity head: Maps tokens to velocity field (B, N, 3)
    
    Args:
        d_model: Token embedding dimension
        graph_consts: Dictionary containing graph structure
        backbone: Scene point cloud encoder module
        hand_encoder: Hand keypoint encoder module
        dit: Graph-aware DiT module
        loss_cfg: Loss configuration
        optimizer_cfg: Optimizer configuration
        sample_num_steps: Number of ODE steps for sampling
        sample_solver: Solver type ('euler', 'heun', 'rk4')
        sample_schedule: Time schedule type ('linear', 'cosine', 'shift')
        schedule_shift: Shift factor for shift schedule (default 1.0)
        proj_num_iters: Number of edge length projection iterations
        proj_max_corr: Maximum correction per projection step
    """
    def __init__(
        self,
        d_model: int,
        graph_consts: Dict[str, torch.Tensor],
        backbone: nn.Module,
        hand_encoder: Any,
        dit: Any,
        loss_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        # Sampling configuration
        sample_num_steps: int = 40,
        sample_solver: str = "euler",
        sample_schedule: str = "shift",
        schedule_shift: float = 1.0,
        proj_num_iters: int = 2,
        proj_max_corr: float = 0.2,
        velocity_mode: str = "direct",
        velocity_kwargs: Optional[Dict] = None,
        state_projection_mode: str = "pbd",
        state_projection_kwargs: Optional[Dict] = None,
        use_norm_data: bool = False,
        use_per_point_gaussian_noise: bool = False,
        # Validation sampling config
        val_num_sample_batches: int = 10,
        # Prediction target (JiT-style option)
        prediction_target: str = "v",
        tau_min: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["graph_consts", "backbone", "hand_encoder", "dit"])
        self.prediction_target = str(prediction_target).lower()
        self.use_norm_data = bool(use_norm_data)
        self.use_per_point_gaussian_noise = bool(use_per_point_gaussian_noise)
        self._setup_graph_structure(graph_consts)
        if self.use_norm_data:
            self._validate_norm_bounds()
            self._apply_norm_to_graph_consts()
        self._setup_core_modules(d_model, backbone, hand_encoder, dit, graph_consts)
        self._setup_velocity_strategy(d_model, velocity_mode, velocity_kwargs, prediction_target, tau_min)
        self._setup_sampling_config(
            sample_num_steps, sample_solver, sample_schedule, schedule_shift,
            state_projection_mode, state_projection_kwargs, proj_num_iters, proj_max_corr
        )
        self._setup_loss(loss_cfg)
        self._setup_val_metrics(loss_cfg)
        self.optimizer_cfg = optimizer_cfg
        self._val_step_losses: List[float] = []  # Store scalars, not tensors
        # Validation sampling config: how many batches to sample for recon metrics
        self.val_num_sample_batches: int = val_num_sample_batches
        self._val_total_batches: int = 0  # Discovered during first validation epoch
        self._val_sample_count: int = 0   # Counter for current epoch
        self._log_model_init_summary()

    def _validate_norm_bounds(self) -> None:
        """Ensure normalization bounds are available and correctly shaped."""
        norm_min = getattr(self, "norm_min", None)
        norm_max = getattr(self, "norm_max", None)
        if norm_min is None or norm_max is None:
            raise RuntimeError(
                "use_norm_data=True requires 'norm_min' and 'norm_max' to be supplied in graph_consts "
                "and registered via GraphConfig."
            )
        if norm_min.dim() != 3 or norm_max.dim() != 3 or norm_min.shape[-1] != 3 or norm_max.shape[-1] != 3:
            raise ValueError(f"Normalization bounds must have shape (1,1,3); got {tuple(norm_min.shape)} / {tuple(norm_max.shape)}")

    @torch.no_grad()
    def _apply_norm_to_graph_consts(self) -> None:
        """Recompute template and edge rest lengths in normalized space for consistency."""
        if self.norm_min is None or self.norm_max is None:
            raise RuntimeError("Normalization bounds not loaded.")
        # Broadcast min/max to match point tensor shape
        scale = 2.0 / torch.clamp(self.norm_max - self.norm_min, min=1e-6)  # (1,1,3)
        bias = -1.0

        def _norm_pts(x: torch.Tensor) -> torch.Tensor:
            # reshape scale/min to match dims of x
            view_shape = [1] * x.dim()
            view_shape[-1] = 3
            nm = self.norm_min.view(view_shape)
            sc = scale.view(view_shape)
            return (x - nm) * sc + bias

        tmpl_norm: Optional[torch.Tensor] = None
        if self.template_xyz is not None:
            tmpl_norm = _norm_pts(self.template_xyz)
            self.template_xyz.copy_(tmpl_norm.view_as(self.template_xyz))

        if self.canonical_xyz is not None:
            canon_norm = _norm_pts(self.canonical_xyz)
            if hasattr(self, "norm_canonical_xyz") and isinstance(self.norm_canonical_xyz, torch.Tensor):
                self.norm_canonical_xyz.copy_(canon_norm.view_as(self.canonical_xyz))
            else:
                if hasattr(self, "norm_canonical_xyz"):
                    delattr(self, "norm_canonical_xyz")
                self.register_buffer("norm_canonical_xyz", canon_norm.view_as(self.canonical_xyz).float())

        if self.approx_std is not None:
            view_shape = [1] * self.approx_std.dim()
            view_shape[-1] = 3
            sc = scale.view(view_shape)
            std_norm = self.approx_std * sc
            if hasattr(self, "norm_approx_std") and isinstance(self.norm_approx_std, torch.Tensor):
                self.norm_approx_std.copy_(std_norm.view_as(self.approx_std))
            else:
                if hasattr(self, "norm_approx_std"):
                    delattr(self, "norm_approx_std")
                self.register_buffer("norm_approx_std", std_norm.view_as(self.approx_std).float())

        if tmpl_norm is not None:
            # Ensure shape is (N,3)
            if tmpl_norm.dim() == 3 and tmpl_norm.shape[0] == 1:
                tmpl_norm_for_edge = tmpl_norm.squeeze(0)
            else:
                tmpl_norm_for_edge = tmpl_norm
            i, j = self.edge_index
            diff = tmpl_norm_for_edge[i, :] - tmpl_norm_for_edge[j, :]
            rest = torch.linalg.norm(diff, dim=-1)
            self.edge_rest_lengths.copy_(rest.view_as(self.edge_rest_lengths))
        else:
            # Fallback: isotropic scale using mean scale factor
            scalar_scale = float(scale.mean())
            self.edge_rest_lengths.mul_(scalar_scale)

    def _denorm_xyz(self, norm_xyz: torch.Tensor) -> torch.Tensor:
        """Map normalized xyz in [-1,1] back to original space."""
        if self.norm_min is None or self.norm_max is None:
            raise RuntimeError("Normalization bounds not loaded.")
        norm = torch.clamp(norm_xyz, -1.0, 1.0)
        unscaled = (norm + 1.0) * 0.5
        denom = torch.clamp(self.norm_max - self.norm_min, min=1e-6)
        return unscaled * denom + self.norm_min
    
    def _setup_graph_structure(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Parse and register graph structure constants."""
        graph_cfg = GraphConfig(self, graph_consts)
        self.N = graph_cfg.N
        self.num_fingers = graph_cfg.num_fingers
        self.num_joint_types = graph_cfg.num_joint_types
        self.num_edge_types = graph_cfg.num_edge_types
        self.rigid_groups = graph_cfg.rigid_groups
        fixed_idx = torch.arange(0, min(7, self.N))
        # Keep fixed-point indices on the model device for safe indexing on GPU.
        self.register_buffer("fixed_point_indices", fixed_idx, persistent=False)
        fixed_mask = torch.zeros(self.N, dtype=torch.bool)
        fixed_mask[fixed_idx] = True
        self.register_buffer("fixed_point_mask", fixed_mask, persistent=False)
        # Mask edges that are entirely between fixed points for optional use in metrics.
        edge_mask = ~(fixed_mask[self.edge_index[0]] & fixed_mask[self.edge_index[1]])
        self.register_buffer("active_edge_mask", edge_mask, persistent=False)
        # Cache rigid groups that involve at least one active point for projection purposes.
        self.rigid_groups_active: List[torch.Tensor] = [
            g for g in self.rigid_groups if not fixed_mask[g].all()
        ]
    
    def _setup_core_modules(
        self,
        d_model: int,
        backbone: Any,
        hand_encoder: Any,
        dit: Any,
        graph_consts: Dict[str, torch.Tensor],
    ) -> None:
        """Build core network modules (backbone, hand encoder, DiT, timestep embedding)."""
        if isinstance(backbone, nn.Module):
            self.backbone = backbone
        else:
            self.backbone = build_backbone(backbone)
        
        if isinstance(hand_encoder, nn.Module):
            self.hand_encoder = hand_encoder
        else:
            self.hand_encoder = build_hand_encoder(
                cfg=hand_encoder,
                graph_consts=graph_consts,
                out_dim=d_model,
            )
        
        if isinstance(dit, nn.Module):
            self.dit = dit
        else:
            self.dit = build_dit(
                cfg=dit,
                graph_consts=graph_consts,
                dim=d_model,
            )
        
        self.timestep_embed = TimestepEmbedding(dim=d_model)
    
    def _setup_velocity_strategy(
        self,
        d_model: int,
        velocity_mode: str,
        velocity_kwargs: Optional[Dict],
        prediction_target: str = "v",
        tau_min: float = 1e-3,
    ) -> None:
        """Build velocity prediction strategy."""
        self.velocity_mode = str(velocity_mode).lower()
        self.velocity_strategy = build_velocity_strategy(
            mode=self.velocity_mode,
            d_model=d_model,
            edge_index=self.edge_index,
            rest_lengths=self.edge_rest_lengths,
            template_xyz=self.template_xyz,
            groups=self.rigid_groups_active if self.rigid_groups_active else self.rigid_groups,
            kwargs=velocity_kwargs,
            prediction_target=prediction_target,
            tau_min=tau_min,
        )
    
    def _setup_sampling_config(
        self,
        num_steps: int,
        solver_name: str,
        schedule_name: str,
        schedule_shift: float,
        projector_mode: str,
        projector_kwargs: Optional[Dict],
        proj_num_iters: int,
        proj_max_corr: float,
    ) -> None:
        """Build sampling configuration (solver, time schedule, state projector)."""
        proj_kwargs = dict(projector_kwargs or {})
        if projector_mode == "pbd":
            proj_kwargs.setdefault("iters", max(2, proj_num_iters))
            proj_kwargs.setdefault("max_corr", proj_max_corr)
        
        self.sampling_config = SamplingConfig.from_params(
            num_steps=num_steps,
            solver_name=solver_name,
            schedule_name=schedule_name,
            schedule_shift=schedule_shift,
            projector_mode=projector_mode,
            projector_kwargs=proj_kwargs,
            edge_index=self.edge_index,
            rest_lengths=self.edge_rest_lengths,
            template_xyz=self.template_xyz,
            rigid_groups=self.rigid_groups_active if self.rigid_groups_active else self.rigid_groups,
        )
        
        self.proj_num_iters = proj_num_iters
        self.proj_max_corr = proj_max_corr
    
    def _setup_loss(self, loss_cfg: Optional[DictConfig]) -> None:
        """Setup loss function from configuration."""
        if loss_cfg is None:
            self.loss_fn = FlowMatchingLoss(edge_index=self.edge_index, lambda_tangent=1.0)
            self.lambda_tangent = 1.0
            return
        
        cfg_dict = (
            OmegaConf.to_container(loss_cfg, resolve=True)
            if isinstance(loss_cfg, DictConfig)
            else deepcopy(loss_cfg)
        )
        
        # Remove validation-only configuration before instantiating loss
        cfg_dict.pop("val_metrics", None)
        
        lambda_tangent = float(cfg_dict.get("lambda_tangent", 1.0))
        target = cfg_dict.pop("_target_", None)
        
        if target:
            hydra_cfg = OmegaConf.create({"_target_": target, **cfg_dict})
            self.loss_fn = hydra_instantiate(hydra_cfg, edge_index=self.edge_index)
        else:
            self.loss_fn = FlowMatchingLoss(edge_index=self.edge_index, lambda_tangent=lambda_tangent)
        
        self.lambda_tangent = float(getattr(self.loss_fn, "lambda_tangent", lambda_tangent))

    def _setup_val_metrics(self, loss_cfg: Optional[DictConfig]) -> None:
        """Setup validation metric manager from loss configuration."""
        self.val_metric_manager: Optional[HandValidationMetricManager] = None
        if loss_cfg is None:
            return

        if isinstance(loss_cfg, DictConfig):
            if "val_metrics" not in loss_cfg:
                return
            val_cfg = OmegaConf.to_container(loss_cfg.val_metrics, resolve=True)
        else:
            val_cfg = dict(loss_cfg.get("val_metrics", {})) if isinstance(loss_cfg, dict) else {}

        if not val_cfg:
            return

        self.val_metric_manager = HandValidationMetricManager(edge_index=self.edge_index, config=val_cfg)

    def encode_scene_tokens(self, scene_pc: torch.Tensor, return_xyz: bool = False):
        """Encode scene point cloud into tokens.

        Args:
            scene_pc: Scene point cloud of shape (B, P, C) where C >= 3.
                      C=3 for xyz only, C=6 for xyz+normals.
                      If backbone doesn't support normals, automatically truncates to xyz.
            return_xyz: If True, also return token xyz (when available).

        Returns:
            Scene tokens of shape (B, K, d_model); optionally token xyz.
        """
        if scene_pc.numel() == 0 or scene_pc.shape[1] == 0:
            B = scene_pc.shape[0]
            empty = torch.zeros(B, 1, self.hparams.d_model, device=scene_pc.device, dtype=scene_pc.dtype)
            return (empty, None) if return_xyz else empty
        
        # Check if backbone supports normals; if not, truncate to xyz only
        backbone_uses_normals = getattr(self.backbone, 'use_scene_normals', False)
        if not backbone_uses_normals and scene_pc.shape[-1] > 3:
            scene_pc = scene_pc[..., :3]
        
        backbone_out = self.backbone(scene_pc)
        scene_xyz = None
        if isinstance(backbone_out, tuple):
            xyz, feats = backbone_out[0], backbone_out[1]
            scene_xyz = xyz
            if not isinstance(feats, torch.Tensor) or feats.dim() != 3:
                raise TypeError("Backbone features must be a 3D tensor when returning a tuple (xyz, features)")
            if not isinstance(xyz, torch.Tensor) or xyz.dim() != 3:
                raise TypeError("Backbone coordinates must be a 3D tensor when returning a tuple (xyz, features)")
            B_xyz, K_xyz, _ = xyz.shape
            B_feat = feats.shape[0]
            if B_xyz != scene_pc.shape[0] or B_feat != scene_pc.shape[0]:
                raise ValueError("Backbone output batch dimension does not match scene_pc batch size")
            if feats.shape[1] == K_xyz and feats.shape[2] != K_xyz:
                scene_tokens = feats
            elif feats.shape[2] == K_xyz:
                scene_tokens = feats.permute(0, 2, 1)
            else:
                raise ValueError(
                    f"Cannot infer token dimension from backbone features with shape {feats.shape} "
                    f"and xyz shape {xyz.shape}"
                )
        elif isinstance(backbone_out, dict):
            scene_tokens = backbone_out
            for key in ["tokens", "scene_tokens", "feat"]:
                if key in scene_tokens:
                    scene_tokens = scene_tokens[key]
                    break
            else:
                raise KeyError(
                    f"Backbone returned dict without expected keys: {list(scene_tokens.keys())}"
                )
            if isinstance(scene_tokens, torch.Tensor) and scene_tokens.dim() == 3:
                if scene_tokens.shape[-1] == self.hparams.d_model:
                    pass
                elif scene_tokens.shape[1] == self.hparams.d_model:
                    scene_tokens = scene_tokens.permute(0, 2, 1)
        elif torch.is_tensor(backbone_out):
            scene_tokens = backbone_out
            if scene_tokens.dim() == 3:
                if scene_tokens.shape[-1] == self.hparams.d_model:
                    pass
                elif scene_tokens.shape[1] == self.hparams.d_model:
                    scene_tokens = scene_tokens.permute(0, 2, 1)
        else:
            raise TypeError(f"Unsupported backbone output type: {type(backbone_out)}")
        # Ensure consistent dtype for DiT input to avoid torch.compile recompilation
        # between training (fp16 from autocast) and validation (fp32)
        scene_tokens = scene_tokens.float()
        return (scene_tokens, scene_xyz) if return_xyz else scene_tokens

    def encode_hand_tokens(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Encode hand keypoints into per-point tokens.
        
        Args:
            keypoints: Hand keypoints of shape (B, N, 3)
            
        Returns:
            Hand tokens of shape (B, N, d_model)
        """
        B, N, _ = keypoints.shape
        if N != self.N:
            raise ValueError(f"Expected {self.N} keypoints, got {N}")
        finger_ids = self.finger_ids_const.unsqueeze(0).expand(B, -1)
        joint_type_ids = self.joint_type_ids_const.unsqueeze(0).expand(B, -1)
        hand_tokens = self.hand_encoder(
            xyz=keypoints,
            finger_ids=finger_ids,
            joint_type_ids=joint_type_ids,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_rest_lengths=self.edge_rest_lengths,
        )
        return hand_tokens

    def predict_velocity(
        self,
        keypoints: torch.Tensor,
        scene_pc: torch.Tensor,
        timesteps: torch.Tensor,
        scene_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity by the selected strategy.

        Keeps the original encoding pathway, but delegates velocity logic to strategy.
        Optionally accepts pre-computed ``scene_tokens`` to avoid re-encoding during
        sampling, where the scene does not change across ODE evaluations.
        """
        use_cross_bias = bool(getattr(self.dit, "hand_scene_bias", False))
        scene_xyz: Optional[torch.Tensor] = None

        # Reuse cached scene tokens if provided (e.g., inside sampling loop)
        if scene_tokens is None:
            if use_cross_bias:
                scene_tokens, scene_xyz = self.encode_scene_tokens(scene_pc, return_xyz=True)
            else:
                scene_tokens = self.encode_scene_tokens(scene_pc)
        elif use_cross_bias and isinstance(scene_tokens, tuple):
            scene_tokens, scene_xyz = scene_tokens
        hand_tokens = self.encode_hand_tokens(keypoints)
        time_emb = self.timestep_embed(timesteps)
        hand_tokens_out = self.dit(
            hand_tokens=hand_tokens,
            scene_tokens=scene_tokens,
            temb=time_emb,
            xyz=keypoints,
            scene_xyz=scene_xyz,
        )
        # Delegate:
        velocity = self.velocity_strategy.predict(
            model=self,
            keypoints=keypoints,
            timesteps=timesteps,
            hand_tokens_out=hand_tokens_out,
        )
        # Anchor fixed points by zeroing their velocity to prevent drift during integration
        if hasattr(self, "fixed_point_indices") and self.fixed_point_indices.numel() > 0:
            velocity[:, self.fixed_point_indices, :] = 0
        return velocity

    def _construct_flow_path(
        self,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct rectified flow path for training.
        
        Samples: y0 ~ N(0, I), t ~ U(0, 1)
        Constructs: y_t = (1 - t) * y0 + t * y1, v = y1 - y0
        
        Args:
            target: Target keypoints of shape (B, N, 3)
            
        Returns:
            Tuple of (source, timesteps, noisy_sample, target_velocity)
        """
        B = target.size(0)
        device = target.device
        dtype = target.dtype

        if self.use_per_point_gaussian_noise:
            mean, std = self._get_noise_stats(device=device, dtype=dtype)
            source = torch.randn_like(target) * std + mean
            # Fixed points: zero-velocity by anchoring source to target
            source[:, self.fixed_point_indices, :] = target[:, self.fixed_point_indices, :]
        else:
            source = torch.randn_like(target)

        timesteps = torch.rand(B, device=device)
        t_expanded = timesteps.view(B, 1, 1)
        noisy_sample = (1.0 - t_expanded) * source + t_expanded * target
        target_velocity = target - source
        return source, timesteps, noisy_sample, target_velocity

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for flow matching.

        Args:
            batch: Dictionary containing 'xyz' (B, N, 3) and optionally 'scene_pc' (B, P, 3)
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        logger = logging.getLogger(__name__)
        try:
            target, _, scene_pc = self._prepare_batch_data(batch, stage="train")
            batch_size = int(target.size(0))

            loss, loss_dict = self._compute_step_loss(target, scene_pc)

            # Log BEFORE NaN check for debugging
            self.log(
                "train/loss",
                loss.detach(),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "train/loss_fm",
                loss_dict["loss_fm"].detach(),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "train/loss_tangent",
                loss_dict["loss_tangent"].detach(),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

            if not torch.isfinite(loss):
                # Log which components are NaN/Inf for debugging
                nan_components = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
                self.log("train/nan_loss", 1.0, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
                logger.error(
                    f"NaN/Inf loss detected at batch {batch_idx}! "
                    f"Total loss: {loss.item():.6f}, NaN components: {nan_components}"
                )
                for k, v in loss_dict.items():
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
        except Exception as e:
            self.log("train/error_flag", 1.0, prog_bar=True, on_step=True, on_epoch=False)
            self.print(f"Training step error at batch {batch_idx}: {e}")
            raise

    def _prepare_batch_data(self, batch: Dict[str, Any], stage: str = "train") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data for training/validation/testing.
        
        Args:
            batch: Input batch dictionary
            stage: 'train', 'val', or 'test' for logging prefixes
            
        Returns:
            Tuple of (target, raw_target, scene_pc)
        """
        if "xyz" not in batch:
            self.log(f"{stage}/missing_xyz", 1.0, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
            raise KeyError("Batch is missing 'xyz' key")

        raw_target = batch["xyz"].to(self.device)
        target = raw_target
        if self.use_norm_data:
            if "norm_xyz" not in batch:
                raise KeyError("use_norm_data=True requires 'norm_xyz' in batch")
            target = batch["norm_xyz"].to(self.device)
        
        if target.dim() != 3 or target.size(-1) != 3:
            self.log(f"{stage}/invalid_xyz_shape", 1.0, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
            raise ValueError(f"Expected 'xyz' of shape (B, N, 3), got {tuple(target.shape)}")

        scene_pc = None
        if self.use_norm_data:
            scene_pc = batch.get("norm_scene_pc", None)
        if scene_pc is None:
            scene_pc = batch.get("scene_pc", None)
        
        if scene_pc is None:
            if stage == "train":
                B = target.shape[0]
                scene_pc = torch.zeros(B, 0, 3, device=self.device, dtype=target.dtype)
            else:
                self.log(f"{stage}/missing_scene_pc", 1.0, prog_bar=True, on_step=False, on_epoch=True)
                raise KeyError(
                    "Batch is missing 'scene_pc' / 'norm_scene_pc'. "
                    "This usually means the DataModule did not attach scene point clouds."
                )
        else:
            scene_pc = scene_pc.to(self.device)
            # Accept 3D (xyz only) or 6D (xyz + normals) point clouds
            valid_dims = (3, 6)
            if scene_pc.dim() != 3 or scene_pc.size(-1) not in valid_dims:
                self.log(f"{stage}/invalid_scene_shape", 1.0, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
                raise ValueError(f"Expected 'scene_pc' of shape (B, P, 3 or 6), got {tuple(scene_pc.shape)}")
            
            if stage != "train" and scene_pc.size(1) == 0:
                self.log(f"{stage}/empty_scene_pc", 1.0, prog_bar=True, on_step=False, on_epoch=True)
                raise ValueError(
                    f"'scene_pc' has zero points: shape={tuple(scene_pc.shape)}. "
                    "Check DexGraspNet point cloud files and scene_pc_return_mode."
                )
            
        return target, raw_target, scene_pc

    def _compute_step_loss(
        self, 
        target: torch.Tensor, 
        scene_pc: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute flow matching loss for a step.
        
        Returns:
            Tuple of (loss, loss_dict)
        """
        _, timesteps, noisy_sample, target_velocity = self._construct_flow_path(target)
        pred_velocity = self.predict_velocity(noisy_sample, scene_pc, timesteps)
        loss_dict = self.loss_fn(v_hat=pred_velocity, v_star=target_velocity, y_tau=noisy_sample)
        loss = loss_dict["loss"]
        return loss, loss_dict

    def _compute_val_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        flow_edge_len_err: torch.Tensor,
        scene_pc: torch.Tensor,
        raw_target: torch.Tensor,
        batch_size: int,
        batch_idx: int = 0,
    ) -> None:
        """Compute and log validation metrics.
        
        Note: Sampling-based recon metrics are only computed on first few batches
        to avoid excessive memory usage during validation.
        """
        if self.val_metric_manager is None:
            return

        with torch.no_grad():
            flow_metrics = self.val_metric_manager.compute_flow_metrics(
                loss_dict=loss_dict,
                flow_edge_len_err=flow_edge_len_err,
            )
        for name, value in flow_metrics.items():
            self.log(
                f"val/flow_{name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        # Sampling-based reconstruction metrics (full keypoint prediction)
        # Uniformly sample val_num_sample_batches across the validation set
        num_sample_batches = getattr(self, "val_num_sample_batches", 10)
        
        # Compute stride based on discovered total batches (from previous epoch)
        # First epoch: use batch_idx to estimate, sample more conservatively
        if self._val_total_batches > 0:
            # After first epoch: we know the total, compute uniform stride
            stride = max(1, self._val_total_batches // num_sample_batches)
            should_sample = (batch_idx % stride == 0) and (self._val_sample_count < num_sample_batches)
        else:
            # First epoch: sample first few batches, then every ~20th batch as fallback
            fallback_stride = 20
            should_sample = (batch_idx < 3) or (
                batch_idx % fallback_stride == 0 and self._val_sample_count < num_sample_batches
            )
        
        # Track max batch_idx to learn total batches for next epoch
        self._val_total_batches = max(self._val_total_batches, batch_idx + 1)
        
        if should_sample:
            self._val_sample_count += 1
            with torch.no_grad():
                pred_xyz = self.sample(scene_pc)
                recon_metrics = self.val_metric_manager.compute_recon_metrics(
                    pred_xyz=pred_xyz,
                    gt_xyz=raw_target,
                )
                # Explicitly delete to help GC
                del pred_xyz
            for name, value in recon_metrics.items():
                self.log(
                    f"val/recon_{name}",
                    value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

    def _record_trajectory_state(
        self, 
        step_idx: int, 
        t_scalar: float, 
        state: torch.Tensor, 
        trajectory_records: List[Dict[str, Any]]
    ) -> None:
        """Record a snapshot of the trajectory state."""
        snap = state
        if self.use_norm_data:
            snap = self._denorm_xyz(state)
        trajectory_records.append(
            {
                "step": int(step_idx),
                "t": float(t_scalar),
                "xyz": snap.detach().cpu(),
            }
        )

    def on_validation_epoch_start(self) -> None:
        """Reset validation loss buffer at the start of each epoch."""
        self._val_step_losses = []
        self._val_sample_count = 0  # Reset sample counter for this epoch

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step for flow matching and reconstruction metrics.

        Args:
            batch: Dictionary containing 'xyz' (B, N, 3) and optionally 'scene_pc' (B, P, 3)
            batch_idx: Batch index
        """
        logger = logging.getLogger(__name__)
        try:
            target, raw_target, scene_pc = self._prepare_batch_data(batch, stage="val")
            batch_size = int(target.size(0))

            loss, loss_dict = self._compute_step_loss(target, scene_pc)
            
            if not torch.isfinite(loss):
                # Log detailed NaN info for debugging
                nan_components = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
                self.log("val/nan_loss", 1.0, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
                logger.warning(
                    f"NaN/Inf validation loss at batch {batch_idx}. "
                    f"NaN components: {nan_components}"
                )
                return

            self._val_step_losses.append(float(loss.detach().item()))  # Store scalar, not tensor

            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log("val/loss_fm", loss_dict["loss_fm"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log(
                "val/loss_tangent",
                loss_dict["loss_tangent"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            
            with torch.no_grad():
                # Keep edge-length computation in the same space as edge_rest_lengths.
                edge_input = target  # already normalized when use_norm_data=True
                flow_edge_len_err = self._compute_edge_length_error(edge_input)
            
            # Backwards-compatible alias
            self.log(
                "val/edge_len_err",
                flow_edge_len_err,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

            self._compute_val_metrics(loss_dict, flow_edge_len_err, scene_pc, raw_target, batch_size, batch_idx)

        except Exception as e:
            self.log("val/error_flag", 1.0, prog_bar=True, on_step=False, on_epoch=True)
            self.print(f"Validation step error at batch {batch_idx}: {e}")
            raise

    def _apply_state_projection(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Apply the configured state projection during sampling."""
        if self.use_per_point_gaussian_noise:
            # Preserve fixed-point positions when using per-point Gaussian noise.
            fixed_backup = keypoints[:, self.fixed_point_indices, :].clone()

        if self.sampling_config.is_hybrid():
            x = self.sampling_config.rigid_projector(keypoints)
            x = self.sampling_config.pbd_projector(x)
            keypoints = x
        elif self.sampling_config.state_projector is not None:
            keypoints = self.sampling_config.state_projector(keypoints)
        # else: keep keypoints unchanged

        if self.use_per_point_gaussian_noise:
            keypoints[:, self.fixed_point_indices, :] = fixed_backup

        return keypoints

    def _project_edge_lengths(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Apply PBD-style edge length projection to satisfy kinematic constraints.
        
        Args:
            keypoints: Hand keypoints of shape (B, N, 3)
            
        Returns:
            Projected keypoints of shape (B, N, 3)
        """
        if self.proj_num_iters <= 0:
            return keypoints
        keypoints = keypoints.clone()
        i, j = self.edge_index
        rest_lengths = self.edge_rest_lengths.view(1, -1, 1).to(keypoints)
        for _ in range(self.proj_num_iters):
            diff = keypoints[:, i, :] - keypoints[:, j, :]
            dist = (diff.pow(2).sum(-1, keepdim=True) + 1e-9).sqrt()
            correction = (1.0 - rest_lengths / dist).clamp(-self.proj_max_corr, self.proj_max_corr)
            delta = correction * diff * 0.5
            keypoints.index_add_(1, i, -delta)
            keypoints.index_add_(1, j, delta)
        return keypoints

    def _compute_edge_length_error(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Compute mean relative edge length error for monitoring.
        
        Args:
            keypoints: Hand keypoints of shape (B, N, 3)
            
        Returns:
            Mean relative edge length error (scalar)
        """
        i, j = self.edge_index
        diff = keypoints[:, i, :] - keypoints[:, j, :]
        dist = (diff.pow(2).sum(-1) + 1e-9).sqrt()
        rest_lengths = self.edge_rest_lengths.view(1, -1).to(keypoints)
        rel_error = (dist - rest_lengths).abs() / (rest_lengths + 1e-6)
        mask = getattr(self, "active_edge_mask", None)
        if mask is not None and mask.numel() == rel_error.shape[1]:
            rel_error = rel_error[:, mask]
        if rel_error.numel() == 0:
            return torch.zeros((), device=keypoints.device, dtype=keypoints.dtype)
        return rel_error.mean()

    def _get_noise_stats(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch per-point Gaussian mean/std depending on normalization setting."""
        if self.use_norm_data:
            mean = getattr(self, "norm_canonical_xyz", None)
            std = getattr(self, "norm_approx_std", None)
        else:
            mean = getattr(self, "canonical_xyz", None)
            std = getattr(self, "approx_std", None)
        if mean is None or std is None:
            raise RuntimeError("canonical_xyz/approx_std (or normalized variants) are required for per-point noise.")
        mean = mean.to(device=device, dtype=dtype).view(1, self.N, 3)
        std = std.to(device=device, dtype=dtype).view(1, self.N, 3)
        return mean, std


    @torch.no_grad()
    def sample(
        self,
        scene_pc: torch.Tensor,
        num_steps: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        trajectory_indices: Optional[Sequence[int]] = None,
        trajectory_stride: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]]]:
        """Generate hand keypoints by integrating the learned flow.
        
        Args:
            scene_pc: Scene point cloud of shape (B, P, 3)
            num_steps: Number of ODE steps (default: self.sample_num_steps)
            initial_noise: Initial keypoints (B, N, 3), sampled from N(0, I) if None
            return_trajectory: If True, also return snapshots along the flow trajectory
            trajectory_indices: Optional explicit list of step indices (0..num_steps) to capture
            trajectory_stride: If provided, capture every N steps in addition to explicit indices
            
        Returns:
            Generated keypoints of shape (B, N, 3). If ``return_trajectory`` is True,
            also returns a list of dictionaries ``{\"step\": idx, \"t\": float, \"xyz\": Tensor}``
            describing captured intermediate states on CPU.
        """
        self.eval()
        device = scene_pc.device
        B = scene_pc.size(0)
        N = self.N
        
        num_steps = num_steps or self.sampling_config.num_steps
        
        if initial_noise is None:
            if self.use_per_point_gaussian_noise:
                mean, std = self._get_noise_stats(device=device, dtype=scene_pc.dtype)
                keypoints = torch.randn(B, N, 3, device=device, dtype=scene_pc.dtype) * std + mean
                keypoints[:, self.fixed_point_indices, :] = mean[:, self.fixed_point_indices, :]
            else:
                keypoints = torch.randn(B, N, 3, device=device)
        else:
            keypoints = initial_noise.to(device)
        
        # Scene tokens are static across ODE evaluations; cache once to avoid recompute.
        use_cross_bias = bool(getattr(self.dit, "hand_scene_bias", False))
        if use_cross_bias:
            scene_tokens_cached = self.encode_scene_tokens(scene_pc, return_xyz=True)
        else:
            scene_tokens_cached = self.encode_scene_tokens(scene_pc)
        timesteps = self.sampling_config.time_schedule.generate(num_steps, device)
        
        capture_steps: List[int] = []
        trajectory_records: List[Dict[str, Any]] = []

        if return_trajectory:
            capture_steps = self._build_trajectory_capture_steps(
                num_steps=num_steps,
                explicit_indices=trajectory_indices,
                stride=trajectory_stride,
            )
            if 0 in capture_steps:
                self._record_trajectory_state(0, timesteps[0].item(), keypoints, trajectory_records)
        
        for k in range(num_steps):
            t_curr = timesteps[k]
            t_next = timesteps[k + 1]
            
            keypoints = self.sampling_config.solver.step(
                keypoints=keypoints,
                velocity_fn=lambda kp, sc, ts: self.predict_velocity(
                    keypoints=kp,
                    scene_pc=sc,
                    timesteps=ts,
                    scene_tokens=scene_tokens_cached,
                ),
                t_curr=t_curr.item(),
                t_next=t_next.item(),
                scene_pc=scene_pc,
            )
            
            keypoints = self._apply_state_projection(keypoints)

            if return_trajectory:
                step_idx = k + 1
                if step_idx in capture_steps:
                    self._record_trajectory_state(step_idx, t_next.item(), keypoints, trajectory_records)
            
        if self.use_norm_data:
            keypoints = self._denorm_xyz(keypoints)
        if return_trajectory:
            return keypoints, trajectory_records
        return keypoints

    def _build_trajectory_capture_steps(
        self,
        num_steps: int,
        explicit_indices: Optional[Sequence[int]],
        stride: Optional[int],
    ) -> List[int]:
        """Resolve which integration steps to capture for trajectory visualization."""
        steps: List[int] = []
        if explicit_indices is not None:
            steps.extend(int(idx) for idx in explicit_indices)
        if stride is not None and stride > 0:
            steps.extend(range(0, num_steps + 1, int(stride)))
        if not steps:
            quarter = max(num_steps // 4, 1)
            steps = [0, min(quarter, num_steps), min(2 * quarter, num_steps), min(3 * quarter, num_steps), num_steps]
        # Always ensure final step is captured
        steps.append(num_steps)

        deduped = sorted({int(max(0, min(num_steps, idx))) for idx in steps})
        return deduped

    def _log_model_init_summary(self) -> None:
        """Log a concise model configuration summary after initialization."""
        logger = logging.getLogger(__name__)
        active_edges = int(self.active_edge_mask.sum().item()) if hasattr(self, "active_edge_mask") else int(self.edge_index.shape[1])
        total_edges = int(self.edge_index.shape[1])
        fixed_pts = int(self.fixed_point_indices.numel()) if hasattr(self, "fixed_point_indices") else 0
        rigid_groups = getattr(self, "rigid_groups_active", self.rigid_groups)
        logger.info(
            "FlowMatchingHandDiT init: use_per_point_gaussian_noise=%s, use_norm_data=%s, velocity_mode=%s",
            self.use_per_point_gaussian_noise,
            self.use_norm_data,
            self.velocity_mode,
        )
        logger.info(
            "Sampling config: num_steps=%d, solver=%s, schedule=%s, projector_mode=%s",
            self.sampling_config.num_steps,
            type(self.sampling_config.solver).__name__,
            type(self.sampling_config.time_schedule).__name__,
            self.sampling_config.state_projector.__class__.__name__
            if (self.sampling_config.state_projector is not None) else (
                "hybrid" if self.sampling_config.is_hybrid() else "none"
            ),
        )
        logger.info(
            "Graph: N=%d, edges=%d (active=%d), fixed_points=%d, rigid_groups=%d (active=%d)",
            self.N,
            total_edges,
            active_edges,
            fixed_pts,
            len(self.rigid_groups),
            len(rigid_groups) if rigid_groups is not None else 0,
        )
        logger.info(
            "Modules: backbone=%s, hand_encoder=%s, dit=%s",
            self.backbone.__class__.__name__,
            self.hand_encoder.__class__.__name__,
            self.dit.__class__.__name__,
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler via Hydra."""
        if self.optimizer_cfg is None:
            # Fallback: simple AdamW without scheduler (kept for backward compatibility)
            optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-2)
            return optimizer

        opt_cfg = deepcopy(self.optimizer_cfg)
        sched_cfg = None
        if "scheduler" in opt_cfg:
            sched_cfg = opt_cfg.scheduler
            # Drop scheduler from optimizer cfg before instantiation
            opt_cfg = deepcopy(opt_cfg)
            opt_cfg.pop("scheduler", None)

        optimizer = hydra_instantiate(opt_cfg, params=self.parameters())

        if sched_cfg is None:
            return optimizer

        sched_cfg = deepcopy(sched_cfg)
        interval = str(sched_cfg.pop("interval", "epoch"))
        frequency = int(sched_cfg.pop("frequency", 1))
        monitor = sched_cfg.pop("monitor", None)
        scheduler = hydra_instantiate(sched_cfg, optimizer=optimizer)

        scheduler_dict: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }
        if monitor:
            scheduler_dict["monitor"] = monitor
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
