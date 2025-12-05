"""Flow Matching model with Graph-aware DiT for dexterous hand keypoint generation."""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List, Sequence, Union
import logging

import torch
import torch.nn as nn
import lightning.pytorch as L
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from .components.losses import FlowMatchingLoss, HandValidationMetricManager
from .components.hand_encoder import build_hand_encoder
from .components.graph_dit import build_dit
from .components.embeddings import TimestepEmbedding
from .backbone import build_backbone
from .components.velocity_strategies import build_velocity_strategy
from .components.graph_config import GraphConfig
from .components.sampling_config import SamplingConfig
from .components.time_schedules import build_training_time_sampler
from .components.data_normalizer import SamplingFrequencyManager


# =============================================================================
# Main Model
# =============================================================================

class HandFlowMatchingDiT(L.LightningModule):
    """Flow Matching model with Graph-aware DiT for hand keypoint generation.
    
    Generates hand keypoints conditioned on scene point clouds using rectified flow.
    
    Architecture:
        - Hand encoder: Encodes current state into per-point tokens (B, N, D)
        - Scene backbone: Processes point cloud into scene tokens (B, K, D)
        - Graph DiT: Self-attention with graph bias + cross-attention with scene
        - Velocity head: Maps tokens to velocity field (B, N, 3)
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
        # Training time schedule
        train_time_schedule: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["graph_consts", "backbone", "hand_encoder", "dit"])
        
        # Core settings
        self.prediction_target = str(prediction_target).lower()
        self.use_norm_data = bool(use_norm_data)
        self.use_per_point_gaussian_noise = bool(use_per_point_gaussian_noise)
        self.optimizer_cfg = optimizer_cfg
        self.proj_num_iters = proj_num_iters
        self.proj_max_corr = proj_max_corr
        
        # Initialize components
        self._setup_graph_structure(graph_consts)
        self._setup_normalization(graph_consts)
        self._setup_core_modules(d_model, backbone, hand_encoder, dit, graph_consts)
        self._setup_velocity_strategy(d_model, velocity_mode, velocity_kwargs, prediction_target, tau_min)
        self._setup_training_time_sampler(train_time_schedule)
        self._setup_sampling_config(
            sample_num_steps, sample_solver, sample_schedule, schedule_shift,
            state_projection_mode, state_projection_kwargs, proj_num_iters, proj_max_corr
        )
        self._setup_loss(loss_cfg)
        self._setup_val_metrics(loss_cfg)
        self._setup_sampling_frequency(loss_cfg, val_num_sample_batches)
        
        self._val_step_losses: List[float] = []
        self._log_model_init_summary()

    # =========================================================================
    # Initialization Methods
    # =========================================================================
    
    def _setup_graph_structure(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Parse and register graph structure constants."""
        graph_cfg = GraphConfig(self, graph_consts)
        self.N = graph_cfg.N
        self.num_fingers = graph_cfg.num_fingers
        self.num_joint_types = graph_cfg.num_joint_types
        self.num_edge_types = graph_cfg.num_edge_types
        self.rigid_groups = graph_cfg.rigid_groups
        
        # Fixed point handling
        fixed_idx = torch.arange(0, min(7, self.N))
        self._safe_register_buffer("fixed_point_indices", fixed_idx, persistent=False)
        
        fixed_mask = torch.zeros(self.N, dtype=torch.bool)
        fixed_mask[fixed_idx] = True
        self._safe_register_buffer("fixed_point_mask", fixed_mask, persistent=False)
        
        # Active edge mask (excludes edges between fixed points)
        edge_mask = ~(fixed_mask[self.edge_index[0]] & fixed_mask[self.edge_index[1]])
        self._safe_register_buffer("active_edge_mask", edge_mask, persistent=False)
        
        # Cache active rigid groups
        self.rigid_groups_active: List[torch.Tensor] = [
            g for g in self.rigid_groups if not fixed_mask[g].all()
        ]
    
    def _setup_normalization(self, graph_consts: Dict[str, torch.Tensor]) -> None:
        """Setup data normalization if enabled."""
        if not self.use_norm_data:
            return
        
        # Validate bounds
        norm_min = getattr(self, "norm_min", None)
        norm_max = getattr(self, "norm_max", None)
        if norm_min is None or norm_max is None:
            raise RuntimeError(
                "use_norm_data=True requires 'norm_min' and 'norm_max' in graph_consts."
            )
        if norm_min.dim() != 3 or norm_max.dim() != 3:
            raise ValueError(f"Normalization bounds must have shape (1,1,3)")
        
        # Apply normalization to graph constants
        self._apply_norm_to_graph_consts()
    
    @torch.no_grad()
    def _apply_norm_to_graph_consts(self) -> None:
        """Recompute template and edge rest lengths in normalized space."""
        scale = 2.0 / torch.clamp(self.norm_max - self.norm_min, min=1e-6)
        bias = -1.0
        
        def _norm_pts(x: torch.Tensor) -> torch.Tensor:
            view_shape = [1] * x.dim()
            view_shape[-1] = 3
            nm = self.norm_min.view(view_shape)
            sc = scale.view(view_shape)
            return (x - nm) * sc + bias
        
        # Normalize template
        tmpl_norm: Optional[torch.Tensor] = None
        if self.template_xyz is not None:
            tmpl_norm = _norm_pts(self.template_xyz)
            self.template_xyz.copy_(tmpl_norm.view_as(self.template_xyz))
        
        # Normalize canonical xyz
        if self.canonical_xyz is not None:
            canon_norm = _norm_pts(self.canonical_xyz)
            self._safe_register_buffer("norm_canonical_xyz", canon_norm.view_as(self.canonical_xyz).float())
        
        # Normalize approx_std (scale only)
        if self.approx_std is not None:
            view_shape = [1] * self.approx_std.dim()
            view_shape[-1] = 3
            sc = scale.view(view_shape)
            std_norm = self.approx_std * sc
            self._safe_register_buffer("norm_approx_std", std_norm.view_as(self.approx_std).float())
        
        # Recompute edge rest lengths
        if tmpl_norm is not None:
            tmpl_for_edge = tmpl_norm.squeeze(0) if tmpl_norm.dim() == 3 and tmpl_norm.shape[0] == 1 else tmpl_norm
            i, j = self.edge_index
            diff = tmpl_for_edge[i, :] - tmpl_for_edge[j, :]
            rest = torch.linalg.norm(diff, dim=-1)
            self.edge_rest_lengths.copy_(rest.view_as(self.edge_rest_lengths))
        else:
            scalar_scale = float(scale.mean())
            self.edge_rest_lengths.mul_(scalar_scale)
    
    def _setup_core_modules(
        self, d_model: int, backbone: Any, hand_encoder: Any, dit: Any, graph_consts: Dict
    ) -> None:
        """Build core network modules."""
        self.backbone = backbone if isinstance(backbone, nn.Module) else build_backbone(backbone)
        
        self.hand_encoder = hand_encoder if isinstance(hand_encoder, nn.Module) else build_hand_encoder(
            cfg=hand_encoder, graph_consts=graph_consts, out_dim=d_model
        )
        
        self.dit = dit if isinstance(dit, nn.Module) else build_dit(
            cfg=dit, graph_consts=graph_consts, dim=d_model
        )
        
        self.timestep_embed = TimestepEmbedding(dim=d_model)
    
    def _setup_velocity_strategy(
        self, d_model: int, velocity_mode: str, velocity_kwargs: Optional[Dict],
        prediction_target: str, tau_min: float
    ) -> None:
        """Build velocity prediction strategy."""
        self.velocity_mode = str(velocity_mode).lower()
        self.velocity_strategy = build_velocity_strategy(
            mode=self.velocity_mode,
            d_model=d_model,
            edge_index=self.edge_index,
            rest_lengths=self.edge_rest_lengths,
            template_xyz=self.template_xyz,
            groups=self.rigid_groups_active or self.rigid_groups,
            kwargs=velocity_kwargs,
            prediction_target=prediction_target,
            tau_min=tau_min,
        )
    
    def _setup_training_time_sampler(self, train_time_schedule: Optional[Dict[str, Any]]) -> None:
        """Build stochastic timestep sampler for training."""
        cfg = dict(train_time_schedule or {})
        self.training_time_sampler = build_training_time_sampler(
            name=cfg.get("name", "uniform"),
            params=cfg.get("params", {}),
        )
    
    def _setup_sampling_config(
        self, num_steps: int, solver_name: str, schedule_name: str, schedule_shift: float,
        projector_mode: str, projector_kwargs: Optional[Dict], proj_num_iters: int, proj_max_corr: float
    ) -> None:
        """Build sampling configuration."""
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
            rigid_groups=self.rigid_groups_active or self.rigid_groups,
        )
    
    def _setup_loss(self, loss_cfg: Optional[DictConfig]) -> None:
        """Setup loss function from configuration."""
        from .components.losses import DeterministicRegressionLoss
        
        if loss_cfg is None:
            self.loss_fn = FlowMatchingLoss(edge_index=self.edge_index, lambda_tangent=1.0)
            self.lambda_tangent = 1.0
            self.regression_loss_fn = None
            self.lambda_regression = 0.0
            return
        
        cfg_dict = OmegaConf.to_container(loss_cfg, resolve=True) if isinstance(loss_cfg, DictConfig) else deepcopy(loss_cfg)
        
        # Remove validation-only config
        cfg_dict.pop("val_metrics", None)
        
        # Extract regression loss config
        regression_cfg = cfg_dict.pop("regression", None)
        self.lambda_regression = float(cfg_dict.pop("lambda_regression", 0.0))
        
        # Extract flow matching weights
        flow_weights = cfg_dict.pop("weights", None)
        lambda_tangent = float(cfg_dict.get("lambda_tangent", 1.0))
        loss_clamp = float(cfg_dict.get("loss_clamp", 100.0))
        target = cfg_dict.pop("_target_", None)
        
        if target:
            hydra_cfg = OmegaConf.create({"_target_": target, **cfg_dict})
            self.loss_fn = hydra_instantiate(hydra_cfg, edge_index=self.edge_index)
        else:
            self.loss_fn = FlowMatchingLoss(
                edge_index=self.edge_index, 
                lambda_tangent=lambda_tangent,
                loss_clamp=loss_clamp,
                weights=flow_weights,
            )
        
        self.lambda_tangent = float(getattr(self.loss_fn, "lambda_tangent", lambda_tangent))
        
        # Setup regression loss if enabled
        self.regression_loss_fn = None
        if self.lambda_regression > 0.0:
            if self.prediction_target != "x":
                logging.getLogger(__name__).warning(
                    f"lambda_regression={self.lambda_regression} requires prediction_target='x'. Disabling."
                )
                self.lambda_regression = 0.0
            else:
                reg_cfg = dict(regression_cfg or {})
                for key in ["collision_margin", "collision_max_scene_points", 
                            "collision_capsule_radius", "collision_capsule_circle_segments",
                            "collision_capsule_cap_segments"]:
                    if key in cfg_dict:
                        reg_cfg[key] = cfg_dict[key]
                self.regression_loss_fn = DeterministicRegressionLoss(
                    edge_index=self.edge_index,
                    config=reg_cfg,
                    fixed_point_indices=getattr(self, "fixed_point_indices", None),
                )
    
    def _setup_val_metrics(self, loss_cfg: Optional[DictConfig]) -> None:
        """Setup validation metric manager."""
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
        
        recon_scale = float(val_cfg.get("recon_scale", val_cfg.get("recon", {}).get("scale_factor", 1.0)))
        self.val_metric_manager = HandValidationMetricManager(
            edge_index=self.edge_index,
            config=val_cfg,
            fixed_point_indices=getattr(self, "fixed_point_indices", None),
            recon_scale=recon_scale,
        )
    
    def _setup_sampling_frequency(self, loss_cfg: Optional[DictConfig], val_fallback: int) -> None:
        """Setup sampling frequency managers."""
        # Validation sampling frequency (for recon metrics)
        val_sample = None if loss_cfg is None else loss_cfg.get("val_num_sample_batches", None)
        self.val_num_sample_batches = int(val_sample) if val_sample else val_fallback
        self._val_sample_mgr = SamplingFrequencyManager(self.val_num_sample_batches)
        
        # Training regression loss sampling frequency
        train_sample = None if loss_cfg is None else loss_cfg.get("train_num_sample_batches_for_reg_loss", None)
        self.train_num_sample_batches_for_reg_loss = int(train_sample) if train_sample else None
        self._train_reg_sample_mgr = SamplingFrequencyManager(self.train_num_sample_batches_for_reg_loss)
        
        # Trajectory metrics sampling frequency (can be independent from recon)
        traj_sample = None
        if loss_cfg is not None:
            val_metrics = loss_cfg.get("val_metrics", {})
            flow_cfg = val_metrics.get("flow", {}) if val_metrics else {}
            traj_sample = flow_cfg.get("trajectory_num_sample_batches", None)
        # Default: use val_num_sample_batches if not specified
        self.trajectory_num_sample_batches = int(traj_sample) if traj_sample else self.val_num_sample_batches
        self._traj_sample_mgr = SamplingFrequencyManager(self.trajectory_num_sample_batches)
    
    def _safe_register_buffer(self, name: str, tensor: torch.Tensor, persistent: bool = True) -> None:
        """Register buffer, replacing existing attributes safely."""
        if hasattr(self, name):
            if name in self._buffers:
                self._buffers[name] = tensor
                return
            delattr(self, name)
        self.register_buffer(name, tensor, persistent=persistent)

    # =========================================================================
    # Encoding Methods
    # =========================================================================
    
    def encode_scene_tokens(self, scene_pc: torch.Tensor, return_xyz: bool = False):
        """Encode scene point cloud into tokens."""
        if scene_pc.numel() == 0 or scene_pc.shape[1] == 0:
            B = scene_pc.shape[0]
            empty = torch.zeros(B, 1, self.hparams.d_model, device=scene_pc.device, dtype=scene_pc.dtype)
            return (empty, None) if return_xyz else empty
        
        # Check if backbone supports normals
        if not getattr(self.backbone, 'use_scene_normals', False) and scene_pc.shape[-1] > 3:
            scene_pc = scene_pc[..., :3]
        
        backbone_out = self.backbone(scene_pc)
        scene_xyz = None
        
        if isinstance(backbone_out, tuple):
            xyz, feats = backbone_out[0], backbone_out[1]
            scene_xyz = xyz
            scene_tokens = feats if feats.shape[1] == xyz.shape[1] else feats.permute(0, 2, 1)
        elif isinstance(backbone_out, dict):
            for key in ["tokens", "scene_tokens", "feat"]:
                if key in backbone_out:
                    scene_tokens = backbone_out[key]
                    break
            else:
                raise KeyError(f"Backbone dict missing expected keys: {list(backbone_out.keys())}")
        else:
            scene_tokens = backbone_out
        
        if scene_tokens.dim() == 3 and scene_tokens.shape[1] == self.hparams.d_model:
            scene_tokens = scene_tokens.permute(0, 2, 1)
        
        scene_tokens = scene_tokens.float()
        return (scene_tokens, scene_xyz) if return_xyz else scene_tokens
    
    def encode_hand_tokens(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Encode hand keypoints into per-point tokens."""
        B, N, _ = keypoints.shape
        if N != self.N:
            raise ValueError(f"Expected {self.N} keypoints, got {N}")
        
        finger_ids = self.finger_ids_const.unsqueeze(0).expand(B, -1)
        joint_type_ids = self.joint_type_ids_const.unsqueeze(0).expand(B, -1)
        
        return self.hand_encoder(
            xyz=keypoints,
            finger_ids=finger_ids,
            joint_type_ids=joint_type_ids,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_rest_lengths=self.edge_rest_lengths,
        )

    # =========================================================================
    # Forward Pass Methods
    # =========================================================================
    
    def predict_velocity(
        self,
        keypoints: torch.Tensor,
        scene_pc: torch.Tensor,
        timesteps: torch.Tensor,
        scene_tokens: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict velocity field."""
        use_cross_bias = bool(getattr(self.dit, "hand_scene_bias", False))
        scene_xyz: Optional[torch.Tensor] = None
        
        # Encode scene tokens if not cached
        if scene_tokens is None:
            if use_cross_bias:
                scene_tokens, scene_xyz = self.encode_scene_tokens(scene_pc, return_xyz=True)
            else:
                scene_tokens = self.encode_scene_tokens(scene_pc)
        elif use_cross_bias and isinstance(scene_tokens, tuple):
            scene_tokens, scene_xyz = scene_tokens
        
        # Encode hand tokens and timestep
        hand_tokens = self.encode_hand_tokens(keypoints)
        time_emb = self.timestep_embed(timesteps)
        
        # Forward through DiT
        hand_tokens_out = self.dit(
            hand_tokens=hand_tokens,
            scene_tokens=scene_tokens,
            temb=time_emb,
            xyz=keypoints,
            scene_xyz=scene_xyz,
        )
        
        # Predict velocity via strategy
        result = self.velocity_strategy.predict(
            model=self,
            keypoints=keypoints,
            timesteps=timesteps,
            hand_tokens_out=hand_tokens_out,
        )
        
        # Zero velocity for fixed points
        velocity = result["velocity"]
        if hasattr(self, "fixed_point_indices") and self.fixed_point_indices.numel() > 0:
            velocity[:, self.fixed_point_indices, :] = 0
            result["velocity"] = velocity
        
        return result if return_dict else velocity

    # =========================================================================
    # Training Methods
    # =========================================================================
    
    def _construct_flow_path(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct rectified flow path: y_t = (1-t)*y0 + t*y1, v = y1 - y0."""
        B = target.size(0)
        device, dtype = target.device, target.dtype
        
        # Sample source
        if self.use_per_point_gaussian_noise:
            mean, std = self._get_noise_stats(device=device, dtype=dtype)
            source = torch.randn_like(target) * std + mean
            source[:, self.fixed_point_indices, :] = target[:, self.fixed_point_indices, :]
        else:
            source = torch.randn_like(target)
        
        # Sample timesteps
        timesteps = self.training_time_sampler(batch_size=B, device=device, dtype=dtype)
        t_expanded = timesteps.view(B, 1, 1)
        
        # Interpolate
        noisy_sample = (1.0 - t_expanded) * source + t_expanded * target
        target_velocity = target - source
        
        return source, timesteps, noisy_sample, target_velocity
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for flow matching."""
        logger = logging.getLogger(__name__)
        try:
            target, _, scene_pc = self._prepare_batch_data(batch, stage="train")
            batch_size = int(target.size(0))
            
            loss, loss_dict = self._compute_step_loss(target, scene_pc, batch_idx, is_training=True)
            
            # Log metrics
            self._log_training_metrics(loss, loss_dict, batch_size)
            
            # Check for NaN
            if not torch.isfinite(loss):
                nan_components = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
                self.log("train/nan_loss", 1.0, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
                logger.error(f"NaN/Inf loss at batch {batch_idx}! NaN components: {nan_components}")
                raise RuntimeError(f"NaN/Inf loss at batch {batch_idx}. Components: {nan_components}")
            
            return loss
        except Exception as e:
            self.log("train/error_flag", 1.0, prog_bar=True, on_step=True, on_epoch=False)
            self.print(f"Training step error at batch {batch_idx}: {e}")
            raise
    
    def _log_training_metrics(self, loss: torch.Tensor, loss_dict: Dict[str, torch.Tensor], batch_size: int) -> None:
        """Log training metrics."""
        self.log("train/loss", loss.detach(), prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/loss_fm", loss_dict["loss_fm"].detach(), prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/loss_tangent", loss_dict["loss_tangent"].detach(), prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        
        if "loss_regression" in loss_dict:
            self.log("train/loss_regression", loss_dict["loss_regression"].detach(), prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
            for k, v in loss_dict.items():
                if k.startswith("reg_"):
                    self.log(f"train/{k}", v.detach(), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
    
    def on_train_epoch_start(self) -> None:
        """Reset training sample counter."""
        self._train_reg_sample_mgr.reset()

    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step for flow matching and reconstruction metrics."""
        logger = logging.getLogger(__name__)
        try:
            target, raw_target, scene_pc = self._prepare_batch_data(batch, stage="val")
            batch_size = int(target.size(0))
            
            loss, loss_dict = self._compute_step_loss(target, scene_pc, batch_idx, is_training=False)
            
            if not torch.isfinite(loss):
                nan_components = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
                self.log("val/nan_loss", 1.0, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
                logger.warning(f"NaN/Inf validation loss at batch {batch_idx}. NaN components: {nan_components}")
                return
            
            self._val_step_losses.append(float(loss.detach().item()))
            
            # Log metrics
            self._log_validation_metrics(loss, loss_dict, batch_size)
            
            # Compute edge length error
            with torch.no_grad():
                flow_edge_len_err = self._compute_edge_length_error(target)
            self.log("val/edge_len_err", flow_edge_len_err, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            
            # Compute validation metrics
            self._compute_val_metrics(loss_dict, flow_edge_len_err, scene_pc, raw_target, batch_size, batch_idx)
            
        except Exception as e:
            self.log("val/error_flag", 1.0, prog_bar=True, on_step=False, on_epoch=True)
            self.print(f"Validation step error at batch {batch_idx}: {e}")
            raise
    
    def _log_validation_metrics(self, loss: torch.Tensor, loss_dict: Dict[str, torch.Tensor], batch_size: int) -> None:
        """Log validation metrics."""
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val/loss_fm", loss_dict["loss_fm"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val/loss_tangent", loss_dict["loss_tangent"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        if "loss_regression" in loss_dict:
            self.log("val/loss_regression", loss_dict["loss_regression"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            for k, v in loss_dict.items():
                if k.startswith("reg_"):
                    self.log(f"val/{k}", v, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
    
    def on_validation_epoch_start(self) -> None:
        """Reset validation counters."""
        self._val_step_losses = []
        self._val_sample_mgr.reset()
        self._traj_sample_mgr.reset()
    
    def _compute_val_metrics(
        self, loss_dict: Dict[str, torch.Tensor], flow_edge_len_err: torch.Tensor,
        scene_pc: torch.Tensor, raw_target: torch.Tensor, batch_size: int, batch_idx: int
    ) -> None:
        """Compute and log validation metrics."""
        if self.val_metric_manager is None:
            return
        
        # Flow metrics
        with torch.no_grad():
            flow_metrics = self.val_metric_manager.compute_flow_metrics(
                loss_dict=loss_dict, flow_edge_len_err=flow_edge_len_err
            )
        for name, value in flow_metrics.items():
            self.log(f"val/flow_{name}", value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Recon and trajectory metrics (sampled batches, can have different frequencies)
        should_sample_recon = self._val_sample_mgr.should_sample(batch_idx)
        should_sample_traj = self._traj_sample_mgr.should_sample(batch_idx)
        
        if should_sample_recon or should_sample_traj:
            with torch.no_grad():
                # Sample with trajectory metrics only when needed
                pred_xyz, traj_stats = self.sample(
                    scene_pc, 
                    compute_trajectory_metrics=should_sample_traj
                )
                
                # Recon metrics (including collision if enabled)
                if should_sample_recon:
                    recon_metrics = self.val_metric_manager.compute_recon_metrics(
                        pred_xyz=pred_xyz, gt_xyz=raw_target, scene_pc=scene_pc
                    )
                    for name, value in recon_metrics.items():
                        self.log(f"val/recon_{name}", value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                
                # Trajectory quality metrics (PLR, transport cost, velocity consistency)
                if should_sample_traj and traj_stats is not None:
                    traj_metrics = self.val_metric_manager.compute_trajectory_metrics(traj_stats)
                    for name, value in traj_metrics.items():
                        self.log(f"val/flow_{name}", value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                
                del pred_xyz
                if traj_stats is not None:
                    del traj_stats

    # =========================================================================
    # Loss Computation
    # =========================================================================
    
    def _prepare_batch_data(self, batch: Dict[str, Any], stage: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data for training/validation."""
        if "xyz" not in batch:
            self.log(f"{stage}/missing_xyz", 1.0, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
            raise KeyError("Batch is missing 'xyz' key")
        
        raw_target = batch["xyz"].to(self.device)
        target = batch.get("norm_xyz", raw_target).to(self.device) if self.use_norm_data else raw_target
        
        if target.dim() != 3 or target.size(-1) != 3:
            raise ValueError(f"Expected 'xyz' of shape (B, N, 3), got {tuple(target.shape)}")
        
        # Get scene point cloud
        scene_pc = batch.get("norm_scene_pc" if self.use_norm_data else "scene_pc", batch.get("scene_pc"))
        if scene_pc is None:
            if stage == "train":
                scene_pc = torch.zeros(target.shape[0], 0, 3, device=self.device, dtype=target.dtype)
            else:
                raise KeyError("Batch is missing 'scene_pc'")
        else:
            scene_pc = scene_pc.to(self.device)
            if scene_pc.dim() != 3 or scene_pc.size(-1) not in (3, 6):
                raise ValueError(f"Expected 'scene_pc' of shape (B, P, 3 or 6), got {tuple(scene_pc.shape)}")
        
        return target, raw_target, scene_pc
    
    def _compute_step_loss(
        self, target: torch.Tensor, scene_pc: torch.Tensor, batch_idx: int, is_training: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute flow matching loss for a step."""
        _, timesteps, noisy_sample, target_velocity = self._construct_flow_path(target)
        
        # Check if we should compute regression loss
        should_compute_reg = (
            self.lambda_regression > 0.0 
            and self.regression_loss_fn is not None
            and (not is_training or self._train_reg_sample_mgr.should_sample(batch_idx))
        )
        
        # Predict velocity
        pred_result = self.predict_velocity(noisy_sample, scene_pc, timesteps, return_dict=should_compute_reg)
        
        if should_compute_reg:
            pred_velocity = pred_result["velocity"]
            x_pred = pred_result.get("x_pred")
        else:
            pred_velocity = pred_result if not isinstance(pred_result, dict) else pred_result["velocity"]
            x_pred = None
        
        # Flow matching loss
        loss_dict = self.loss_fn(v_hat=pred_velocity, v_star=target_velocity, y_tau=noisy_sample)
        loss = loss_dict["loss"]
        
        # Add regression loss
        if should_compute_reg and x_pred is not None:
            denorm_fn = self._denorm_xyz if self.use_norm_data else None
            scene_pc_xyz = scene_pc[..., :3] if scene_pc is not None else None
            reg_loss, reg_components = self.regression_loss_fn(
                pred_xyz=x_pred, gt_xyz=target, edge_rest_lengths=self.edge_rest_lengths,
                active_edge_mask=getattr(self, "active_edge_mask", None),
                scene_pc=scene_pc_xyz, denorm_fn=denorm_fn,
            )
            loss = loss + self.lambda_regression * reg_loss
            loss_dict["loss_regression"] = reg_loss
            for k, v in reg_components.items():
                loss_dict[f"reg_{k}"] = v
        
        loss_dict["loss"] = loss
        return loss, loss_dict

    # =========================================================================
    # Sampling Methods
    # =========================================================================
    
    @torch.no_grad()
    def sample(
        self,
        scene_pc: torch.Tensor,
        num_steps: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        trajectory_indices: Optional[Sequence[int]] = None,
        trajectory_stride: Optional[int] = None,
        compute_trajectory_metrics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]], Tuple[torch.Tensor, Dict[str, Any]]]:
        """Generate hand keypoints by integrating the learned flow.
        
        Args:
            compute_trajectory_metrics: If True, collect trajectory statistics for
                PLR, transport cost, and velocity consistency metrics. Returns
                (keypoints, traj_stats) tuple instead of just keypoints.
                Note: Adds one extra predict_velocity call per step for non-Euler solvers.
        """
        self.eval()
        device = scene_pc.device
        B, N = scene_pc.size(0), self.N
        num_steps = num_steps or self.sampling_config.num_steps
        
        # Initialize keypoints
        if initial_noise is None:
            if self.use_per_point_gaussian_noise:
                mean, std = self._get_noise_stats(device=device, dtype=scene_pc.dtype)
                keypoints = torch.randn(B, N, 3, device=device, dtype=scene_pc.dtype) * std + mean
                keypoints[:, self.fixed_point_indices, :] = mean[:, self.fixed_point_indices, :]
            else:
                keypoints = torch.randn(B, N, 3, device=device)
        else:
            keypoints = initial_noise.to(device)
        
        # Cache scene tokens
        use_cross_bias = bool(getattr(self.dit, "hand_scene_bias", False))
        scene_tokens_cached = self.encode_scene_tokens(scene_pc, return_xyz=use_cross_bias)
        
        # Generate timesteps
        timesteps = self.sampling_config.time_schedule.generate(num_steps, device)
        
        # Trajectory recording
        trajectory_records: List[Dict[str, Any]] = []
        capture_steps = self._build_trajectory_capture_steps(num_steps, trajectory_indices, trajectory_stride) if return_trajectory else []
        if return_trajectory and 0 in capture_steps:
            self._record_trajectory_state(0, timesteps[0].item(), keypoints, trajectory_records)
        
        # Trajectory metrics initialization (lightweight: only scalars per batch)
        if compute_trajectory_metrics:
            x0 = keypoints.clone()
            arc_length = torch.zeros(B, device=device, dtype=keypoints.dtype)
            transport_cost = torch.zeros(B, device=device, dtype=keypoints.dtype)
            cos_sim_sum = torch.zeros(B, device=device, dtype=keypoints.dtype)
            cos_sim_count = 0
            prev_velocity = None
        
        # Velocity function for solver
        def velocity_fn(kp, sc, ts):
            return self.predict_velocity(kp, sc, ts, scene_tokens=scene_tokens_cached)
        
        # Integration loop
        for k in range(num_steps):
            t_curr, t_next = timesteps[k], timesteps[k + 1]
            dt = t_next.item() - t_curr.item()
            
            # Collect trajectory metrics (one extra forward pass for non-Euler solvers)
            if compute_trajectory_metrics:
                t_batch = torch.full((B,), t_curr.item(), device=device, dtype=keypoints.dtype)
                velocity = self.predict_velocity(keypoints, scene_pc, t_batch, scene_tokens=scene_tokens_cached)
                
                # Accumulate metrics (cheap scalar operations)
                v_norm = velocity.norm(dim=-1).mean(dim=-1)  # (B,)
                arc_length = arc_length + v_norm * dt
                transport_cost = transport_cost + velocity.pow(2).sum(dim=-1).mean(dim=-1) * dt
                
                if prev_velocity is not None:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        velocity.flatten(1), prev_velocity.flatten(1), dim=-1
                    )
                    cos_sim_sum = cos_sim_sum + cos_sim
                    cos_sim_count += 1
                prev_velocity = velocity
            
            # Always use configured solver for integration
            keypoints = self.sampling_config.solver.step(
                keypoints=keypoints,
                velocity_fn=velocity_fn,
                t_curr=t_curr.item(),
                t_next=t_next.item(),
                scene_pc=scene_pc,
            )
            
            keypoints = self._apply_state_projection(keypoints)
            
            if return_trajectory and (k + 1) in capture_steps:
                self._record_trajectory_state(k + 1, t_next.item(), keypoints, trajectory_records)
        
        # Build trajectory stats dict
        traj_stats = None
        if compute_trajectory_metrics:
            traj_stats = {
                "arc_length": arc_length,
                "transport_cost": transport_cost,
                "cos_sim_sum": cos_sim_sum,
                "cos_sim_count": cos_sim_count,
                "x0": x0,
                "x1": keypoints.clone(),
            }
        
        if self.use_norm_data:
            keypoints = self._denorm_xyz(keypoints)
        
        if compute_trajectory_metrics:
            return keypoints, traj_stats
        return (keypoints, trajectory_records) if return_trajectory else keypoints
    
    def _apply_state_projection(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Apply state projection during sampling."""
        if self.use_per_point_gaussian_noise:
            fixed_backup = keypoints[:, self.fixed_point_indices, :].clone()
        
        if self.sampling_config.is_hybrid():
            keypoints = self.sampling_config.rigid_projector(keypoints)
            keypoints = self.sampling_config.pbd_projector(keypoints)
        elif self.sampling_config.state_projector is not None:
            keypoints = self.sampling_config.state_projector(keypoints)
        
        if self.use_per_point_gaussian_noise:
            keypoints[:, self.fixed_point_indices, :] = fixed_backup
        
        return keypoints
    
    def _build_trajectory_capture_steps(
        self, num_steps: int, explicit_indices: Optional[Sequence[int]], stride: Optional[int]
    ) -> List[int]:
        """Determine which steps to capture for trajectory visualization."""
        steps: List[int] = []
        if explicit_indices is not None:
            steps.extend(int(idx) for idx in explicit_indices)
        if stride and stride > 0:
            steps.extend(range(0, num_steps + 1, int(stride)))
        if not steps:
            quarter = max(num_steps // 4, 1)
            steps = [0, quarter, 2 * quarter, 3 * quarter, num_steps]
        steps.append(num_steps)
        return sorted({max(0, min(num_steps, idx)) for idx in steps})
    
    def _record_trajectory_state(
        self, step_idx: int, t_scalar: float, state: torch.Tensor, trajectory_records: List[Dict[str, Any]]
    ) -> None:
        """Record trajectory state snapshot."""
        snap = self._denorm_xyz(state) if self.use_norm_data else state
        trajectory_records.append({"step": int(step_idx), "t": float(t_scalar), "xyz": snap.detach().cpu()})

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _denorm_xyz(self, norm_xyz: torch.Tensor) -> torch.Tensor:
        """Map normalized xyz in [-1,1] back to original space."""
        if self.norm_min is None or self.norm_max is None:
            raise RuntimeError("Normalization bounds not loaded.")
        
        if norm_xyz.shape[-1] == 6:
            xyz_part = norm_xyz[..., :3]
            normals_part = norm_xyz[..., 3:]
            return torch.cat([self._denorm_xyz(xyz_part), normals_part], dim=-1)
        
        norm = torch.clamp(norm_xyz, -1.0, 1.0)
        unscaled = (norm + 1.0) * 0.5
        denom = torch.clamp(self.norm_max - self.norm_min, min=1e-6)
        return unscaled * denom + self.norm_min
    
    def _get_noise_stats(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-point Gaussian noise statistics."""
        if self.use_norm_data:
            mean = getattr(self, "norm_canonical_xyz", None)
            std = getattr(self, "norm_approx_std", None)
        else:
            mean = getattr(self, "canonical_xyz", None)
            std = getattr(self, "approx_std", None)
        
        if mean is None or std is None:
            raise RuntimeError("canonical_xyz/approx_std required for per-point noise.")
        
        return mean.to(device=device, dtype=dtype).view(1, self.N, 3), std.to(device=device, dtype=dtype).view(1, self.N, 3)
    
    def _compute_edge_length_error(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Compute mean relative edge length error."""
        i, j = self.edge_index
        diff = keypoints[:, i, :] - keypoints[:, j, :]
        dist = (diff.pow(2).sum(-1) + 1e-9).sqrt()
        rest_lengths = self.edge_rest_lengths.view(1, -1).to(keypoints)
        rel_error = (dist - rest_lengths).abs() / (rest_lengths + 1e-6)
        
        mask = getattr(self, "active_edge_mask", None)
        if mask is not None and mask.numel() == rel_error.shape[1]:
            rel_error = rel_error[:, mask]
        
        return rel_error.mean() if rel_error.numel() > 0 else torch.zeros((), device=keypoints.device, dtype=keypoints.dtype)
    
    def _project_edge_lengths(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Apply PBD-style edge length projection."""
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
    
    def _log_model_init_summary(self) -> None:
        """Log model configuration summary."""
        logger = logging.getLogger(__name__)
        active_edges = int(self.active_edge_mask.sum().item()) if hasattr(self, "active_edge_mask") else self.edge_index.shape[1]
        total_edges = self.edge_index.shape[1]
        fixed_pts = self.fixed_point_indices.numel() if hasattr(self, "fixed_point_indices") else 0
        rigid_groups = self.rigid_groups_active or self.rigid_groups
        
        logger.info(
            "FlowMatchingHandDiT: use_per_point_gaussian_noise=%s, use_norm_data=%s, velocity_mode=%s",
            self.use_per_point_gaussian_noise, self.use_norm_data, self.velocity_mode
        )
        logger.info(
            "Sampling: num_steps=%d, solver=%s, schedule=%s, projector=%s",
            self.sampling_config.num_steps,
            type(self.sampling_config.solver).__name__,
            type(self.sampling_config.time_schedule).__name__,
            "hybrid" if self.sampling_config.is_hybrid() else (
                self.sampling_config.state_projector.__class__.__name__ if self.sampling_config.state_projector else "none"
            ),
        )
        logger.info(
            "Graph: N=%d, edges=%d (active=%d), fixed=%d, rigid_groups=%d (active=%d)",
            self.N, total_edges, active_edges, fixed_pts, len(self.rigid_groups), len(rigid_groups) if rigid_groups else 0
        )
        logger.info(
            "Modules: backbone=%s, hand_encoder=%s, dit=%s",
            self.backbone.__class__.__name__, self.hand_encoder.__class__.__name__, self.dit.__class__.__name__
        )
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler via Hydra."""
        if self.optimizer_cfg is None:
            return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-2)
        
        opt_cfg = deepcopy(self.optimizer_cfg)
        sched_cfg = opt_cfg.pop("scheduler", None) if "scheduler" in opt_cfg else None
        
        optimizer = hydra_instantiate(opt_cfg, params=self.parameters())
        
        if sched_cfg is None:
            return optimizer
        
        sched_cfg = deepcopy(sched_cfg)
        interval = str(sched_cfg.pop("interval", "epoch"))
        frequency = int(sched_cfg.pop("frequency", 1))
        monitor = sched_cfg.pop("monitor", None)
        scheduler = hydra_instantiate(sched_cfg, optimizer=optimizer)
        
        scheduler_dict: Dict[str, Any] = {"scheduler": scheduler, "interval": interval, "frequency": frequency}
        if monitor:
            scheduler_dict["monitor"] = monitor
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
