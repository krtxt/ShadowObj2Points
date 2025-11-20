"""Flow Matching model with Graph-aware DiT for dexterous hand keypoint generation."""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
import numpy as np
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from .components.losses import FlowMatchingLoss
from .components.hand_encoder import build_hand_encoder
from .components.graph_dit import build_dit
from .backbone import build_backbone


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
        use_ema: Whether to use EMA
        ema_decay: EMA decay rate
        val_vis_num_samples: Number of samples to visualize during validation
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
        sample_schedule: str = "shift",  # [New] Time schedule type
        schedule_shift: float = 1.0,     # [New] Shift factor (SD3 uses 3.0)
        proj_num_iters: int = 2,
        proj_max_corr: float = 0.2,
        # EMA configuration
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_device: Optional[str] = None,
        # Validation visualization
        val_vis_num_samples: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["graph_consts", "backbone", "hand_encoder", "dit"])

        # Parse graph structure constants
        finger_ids = graph_consts["finger_ids"].long()
        joint_type_ids = graph_consts["joint_type_ids"].long()
        edge_index = graph_consts["edge_index"].long()
        edge_type = graph_consts["edge_type"].long()
        edge_rest_lengths = graph_consts["edge_rest_lengths"].float()

        N = finger_ids.shape[0]
        num_fingers = int(finger_ids.max().item()) + 1
        num_joint_types = int(joint_type_ids.max().item()) + 1
        num_edge_types = int(edge_type.max().item()) + 1

        self.N = N
        self.num_fingers = num_fingers
        self.num_joint_types = num_joint_types
        self.num_edge_types = num_edge_types

        # Validation visualization config
        self.val_vis_num_samples = int(val_vis_num_samples)
        self._val_vis_batches: List[Dict[str, Any]] = []

        # Register graph structure as buffers
        self.register_buffer("finger_ids_const", finger_ids)
        self.register_buffer("joint_type_ids_const", joint_type_ids)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)
        self.register_buffer("edge_rest_lengths", edge_rest_lengths)

        # Optional template keypoints
        template_xyz = graph_consts.get("template_xyz", None)
        if template_xyz is not None:
            self.register_buffer("template_xyz", template_xyz.float())
        else:
            self.template_xyz = None

        # Core modules
        if isinstance(backbone, nn.Module):
            self.backbone = backbone
        else:
            self.backbone = build_backbone(backbone)

        # Hand encoder: allow passing either a pre-built module or a config node
        if isinstance(hand_encoder, nn.Module):
            self.hand_encoder = hand_encoder
        else:
            self.hand_encoder = build_hand_encoder(
                cfg=hand_encoder,
                graph_consts=graph_consts,
                out_dim=d_model,
            )

        # DiT: similarly allow module or config
        if isinstance(dit, nn.Module):
            self.dit = dit
        else:
            self.dit = build_dit(
                cfg=dit,
                graph_consts=graph_consts,
                dim=d_model,
            )

        # Timestep embedding and velocity prediction head
        self.timestep_embed = TimestepEmbedding(dim=d_model)
        self.velocity_head = nn.Linear(d_model, 3)

        # Loss function
        self.loss_cfg = self._prepare_loss_config(loss_cfg)
        lambda_tangent = self._resolve_lambda(default_lambda=1.0)
        self.loss_fn = self._build_loss_fn(lambda_tangent)
        self.lambda_tangent = float(getattr(self.loss_fn, "lambda_tangent", lambda_tangent))

        # Optimizer configuration (Hydra-driven)
        self.optimizer_cfg = optimizer_cfg

        # Sampling configuration
        self.sample_num_steps = sample_num_steps
        self.sample_solver = str(sample_solver).lower()
        self.sample_schedule = str(sample_schedule).lower()
        self.schedule_shift = float(schedule_shift)
        self.proj_num_iters = proj_num_iters
        self.proj_max_corr = proj_max_corr

        # EMA & validation visualization
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_device = ema_device
        self.ema = self._build_ema() if self.use_ema else None

    def encode_scene_tokens(self, scene_pc: torch.Tensor) -> torch.Tensor:
        """Encode scene point cloud into tokens.

        Args:
            scene_pc: Scene point cloud of shape (B, P, 3). Returns zero tokens if empty.

        Returns:
            Scene tokens of shape (B, K, d_model)
        """
        if scene_pc.numel() == 0 or scene_pc.shape[1] == 0:
            B = scene_pc.shape[0]
            return torch.zeros(B, 1, self.hparams.d_model, device=scene_pc.device, dtype=scene_pc.dtype)
        backbone_out = self.backbone(scene_pc)
        if isinstance(backbone_out, tuple):
            xyz, feats = backbone_out[0], backbone_out[1]
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
        return scene_tokens

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
    ) -> torch.Tensor:
        """Predict velocity field conditioned on keypoints, scene, and timestep.
        
        Args:
            keypoints: Hand keypoints of shape (B, N, 3)
            scene_pc: Scene point cloud of shape (B, P, 3)
            timesteps: Timestep values of shape (B,) in [0, 1]
            
        Returns:
            Predicted velocity of shape (B, N, 3)
        """
        scene_tokens = self.encode_scene_tokens(scene_pc)
        hand_tokens = self.encode_hand_tokens(keypoints)
        time_emb = self.timestep_embed(timesteps)
        hand_tokens_out = self.dit(
            hand_tokens=hand_tokens,
            scene_tokens=scene_tokens,
            temb=time_emb,
            xyz=keypoints,
        )
        velocity = self.velocity_head(hand_tokens_out)
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
        try:
            if "xyz" not in batch:
                self.log("train/missing_xyz", 1.0, prog_bar=True, on_step=True, on_epoch=False)
                raise KeyError("Batch is missing 'xyz' key")

            target = batch["xyz"].to(self.device)
            if target.dim() != 3 or target.size(-1) != 3:
                self.log("train/invalid_xyz_shape", 1.0, prog_bar=True, on_step=True, on_epoch=False)
                raise ValueError(f"Expected 'xyz' of shape (B, N, 3), got {tuple(target.shape)}")

            scene_pc = batch.get("scene_pc", None)
            if scene_pc is None:
                B = target.shape[0]
                scene_pc = torch.zeros(B, 0, 3, device=self.device, dtype=target.dtype)
            else:
                scene_pc = scene_pc.to(self.device)
                if scene_pc.dim() != 3 or scene_pc.size(-1) != 3:
                    self.log("train/invalid_scene_shape", 1.0, prog_bar=True, on_step=True, on_epoch=False)
                    raise ValueError(f"Expected 'scene_pc' of shape (B, P, 3), got {tuple(scene_pc.shape)}")

            _, timesteps, noisy_sample, target_velocity = self._construct_flow_path(target)
            pred_velocity = self.predict_velocity(noisy_sample, scene_pc, timesteps)
            loss_dict = self.loss_fn(v_hat=pred_velocity, v_star=target_velocity, y_tau=noisy_sample)
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                self.log("train/nan_loss", 1.0, prog_bar=True, on_step=True, on_epoch=True)
                return None

            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/loss_fm", loss_dict["loss_fm"], prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/loss_tangent", loss_dict["loss_tangent"], prog_bar=False, on_step=True, on_epoch=True)
            return loss
        except Exception as e:
            self.log("train/error_flag", 1.0, prog_bar=True, on_step=True, on_epoch=False)
            self.print(f"Training step error at batch {batch_idx}: {e}")
            raise

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step for flow matching.
        
        Args:
            batch: Dictionary containing 'xyz' (B, N, 3) and optionally 'scene_pc' (B, P, 3)
            batch_idx: Batch index
        """
        try:
            if "xyz" not in batch:
                self.log("val/missing_xyz", 1.0, prog_bar=True, on_step=False, on_epoch=True)
                raise KeyError("Batch is missing 'xyz' key")

            target = batch["xyz"].to(self.device)
            if target.dim() != 3 or target.size(-1) != 3:
                self.log("val/invalid_xyz_shape", 1.0, prog_bar=True, on_step=False, on_epoch=True)
                raise ValueError(f"Expected 'xyz' of shape (B, N, 3), got {tuple(target.shape)}")

            scene_pc = batch.get("scene_pc", None)
            if scene_pc is None:
                B = target.shape[0]
                scene_pc = torch.zeros(B, 0, 3, device=self.device, dtype=target.dtype)
            else:
                scene_pc = scene_pc.to(self.device)
                if scene_pc.dim() != 3 or scene_pc.size(-1) != 3:
                    self.log("val/invalid_scene_shape", 1.0, prog_bar=True, on_step=False, on_epoch=True)
                    raise ValueError(f"Expected 'scene_pc' of shape (B, P, 3), got {tuple(scene_pc.shape)}")

            _, timesteps, noisy_sample, target_velocity = self._construct_flow_path(target)
            pred_velocity = self.predict_velocity(noisy_sample, scene_pc, timesteps)
            loss_dict = self.loss_fn(v_hat=pred_velocity, v_star=target_velocity, y_tau=noisy_sample)
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                self.log("val/nan_loss", 1.0, prog_bar=True, on_step=False, on_epoch=True)
                return

            # 使用无斜杠的别名 `val_loss` 以便 ModelCheckpoint/ EarlyStopping 监控和文件名格式化
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/loss_fm", loss_dict["loss_fm"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/loss_tangent", loss_dict["loss_tangent"], prog_bar=False, on_step=False, on_epoch=True)
            with torch.no_grad():
                edge_len_err = self._compute_edge_length_error(target)
            self.log("val/edge_len_err", edge_len_err, prog_bar=False, on_step=False, on_epoch=True)
            # Collect a few batches for qualitative visualization at epoch end
            self._maybe_collect_val_vis_batch(batch)
        except Exception as e:
            self.log("val/error_flag", 1.0, prog_bar=True, on_step=False, on_epoch=True)
            self.print(f"Validation step error at batch {batch_idx}: {e}")
            raise

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
        return rel_error.mean()

    def _generate_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Generate time steps based on the configured schedule.
        
        Returns:
            Timesteps tensor of shape (num_steps + 1,) from 0.0 to 1.0
        """
        # Base linear time [0, 1]
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        
        if self.sample_schedule == "linear":
            return timesteps
            
        elif self.sample_schedule == "cosine":
            # Standard cosine schedule used in diffusion models
            # Maps uniform t in [0,1] to cosine curve
            return 1.0 - torch.cos(timesteps * (torch.pi / 2))

        elif self.sample_schedule == "shift":
            # Time shift schedule (e.g., from Stable Diffusion 3 / Flux)
            # Formula: t' = (s * t) / (1 + (s - 1) * t)
            # s > 1 pushes steps towards t=1 (data), prioritizing low-noise refinement.
            s = self.schedule_shift
            return (s * timesteps) / (1 + (s - 1) * timesteps)
            
        else:
            # Default fallback
            return timesteps

    @torch.no_grad()
    def sample(
        self,
        scene_pc: torch.Tensor,
        num_steps: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate hand keypoints by integrating the learned flow.
        
        Args:
            scene_pc: Scene point cloud of shape (B, P, 3)
            num_steps: Number of ODE steps (default: self.sample_num_steps)
            initial_noise: Initial keypoints (B, N, 3), sampled from N(0, I) if None
            
        Returns:
            Generated keypoints of shape (B, N, 3)
        """
        self.eval()
        device = scene_pc.device
        B = scene_pc.size(0)
        N = self.N
        
        num_steps = num_steps or self.sample_num_steps
        
        if initial_noise is None:
            keypoints = torch.randn(B, N, 3, device=device)
        else:
            keypoints = initial_noise.to(device)
            
        # [Modified] Use the time schedule generator instead of linear space
        timesteps = self._generate_timesteps(num_steps, device)
        time_dtype = timesteps.dtype
        
        solver = str(getattr(self, "sample_solver", "euler")).lower()
        
        for k in range(num_steps):
            t_curr = timesteps[k]
            t_next = timesteps[k + 1]
            dt = t_next - t_curr  # delta t is now variable
            
            if solver in ("heun", "rk2"):
                # Heun's method (RK2): predictor-corrector
                t_batch_curr = torch.full((B,), t_curr.item(), device=device, dtype=time_dtype)
                v1 = self.predict_velocity(keypoints, scene_pc, t_batch_curr)
                
                # Predictor step
                keypoints_pred = keypoints + dt * v1
                
                # Corrector step
                t_batch_next = torch.full((B,), t_next.item(), device=device, dtype=time_dtype)
                v2 = self.predict_velocity(keypoints_pred, scene_pc, t_batch_next)
                
                keypoints = keypoints + 0.5 * dt * (v1 + v2)
                
            elif solver in ("rk4", "runge_kutta4"):
                # Standard RK4
                t_batch_curr = torch.full((B,), t_curr.item(), device=device, dtype=time_dtype)
                v1 = self.predict_velocity(keypoints, scene_pc, t_batch_curr)

                # Half step point logic for RK4 with variable time steps:
                # t_mid = t_curr + 0.5 * dt
                half_step_time = (t_curr + 0.5 * dt).item()
                t_batch_half = torch.full((B,), half_step_time, device=device, dtype=time_dtype)
                
                v2 = self.predict_velocity(keypoints + 0.5 * dt * v1, scene_pc, t_batch_half)
                v3 = self.predict_velocity(keypoints + 0.5 * dt * v2, scene_pc, t_batch_half)

                t_batch_next = torch.full((B,), t_next.item(), device=device, dtype=time_dtype)
                v4 = self.predict_velocity(keypoints + dt * v3, scene_pc, t_batch_next)

                keypoints = keypoints + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
                
            else:
                # Default: explicit Euler
                t_batch = torch.full((B,), t_curr.item(), device=device, dtype=time_dtype)
                velocity = self.predict_velocity(keypoints, scene_pc, t_batch)
                keypoints = keypoints + dt * velocity
                
            # Apply kinematic constraints (PBD)
            keypoints = self._project_edge_lengths(keypoints)
            
        return keypoints

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

    def _prepare_loss_config(self, loss_cfg: Optional[DictConfig]) -> Optional[Dict[str, Any]]:
        if loss_cfg is None:
            return None
        if isinstance(loss_cfg, DictConfig):
            return OmegaConf.to_container(loss_cfg, resolve=True)
        if isinstance(loss_cfg, dict):
            return deepcopy(loss_cfg)
        raise TypeError("loss_cfg must be a DictConfig or dict")

    def _resolve_lambda(self, default_lambda: float) -> float:
        if self.loss_cfg and "lambda_tangent" in self.loss_cfg:
            return float(self.loss_cfg["lambda_tangent"])
        return float(default_lambda)

    def _build_loss_fn(self, lambda_tangent: float) -> nn.Module:
        if self.loss_cfg:
            cfg = deepcopy(self.loss_cfg)
            target = cfg.pop("_target_", None)
            cfg.setdefault("lambda_tangent", lambda_tangent)
            if target:
                hydra_cfg = OmegaConf.create({"_target_": target, **cfg})
                return hydra_instantiate(hydra_cfg, edge_index=self.edge_index)
            lambda_val = float(cfg.get("lambda_tangent", lambda_tangent))
            return FlowMatchingLoss(edge_index=self.edge_index, lambda_tangent=lambda_val)
        return FlowMatchingLoss(edge_index=self.edge_index, lambda_tangent=lambda_tangent)

    # -------------------------------------------------------------------------
    # EMA utilities and validation visualization
    # -------------------------------------------------------------------------
    def _build_ema(self):
        """Create an EMA copy of the model (parameters only)."""
        with torch.no_grad():
            ema_model = deepcopy(self)
            # Drop EMA on the copy to avoid recursion
            ema_model.ema = None
            for p in ema_model.parameters():
                p.requires_grad_(False)
            if self.ema_device is not None:
                ema_model.to(self.ema_device)
        return _ModelEmaWrapper(ema_model, decay=self.ema_decay)

    def on_train_batch_end(self, outputs, batch, batch_idx: int = 0) -> None:
        if self.ema is not None:
            if (int(self.global_step) % 4) == 0:
                self.ema.update(self)

    def on_validation_start(self) -> None:
        # Reset visualization buffer
        self._val_vis_batches = []
        # Swap to EMA weights for evaluation, if enabled
        if self.ema is not None:
            self.ema.store(self)
            self.ema.copy_to(self)

    def on_validation_end(self) -> None:
        # Restore original weights after evaluation
        if self.ema is not None:
            self.ema.restore(self)

    def on_test_start(self) -> None:
        if self.ema is not None:
            self.ema.store(self)
            self.ema.copy_to(self)

    def on_test_end(self) -> None:
        if self.ema is not None:
            self.ema.restore(self)

    def _maybe_collect_val_vis_batch(self, batch: Dict[str, Any]) -> None:
        """Collect a few validation batches for qualitative visualization."""
        if len(self._val_vis_batches) >= self.val_vis_num_samples:
            return
        if "xyz" not in batch:
            return
        with torch.no_grad():
            xyz = batch["xyz"].detach().cpu()
            scene_pc = batch.get("scene_pc", None)
            if scene_pc is not None:
                scene_pc = scene_pc.detach().cpu()
            self._val_vis_batches.append({"xyz": xyz, "scene_pc": scene_pc})

    def on_validation_epoch_end(self) -> None:
        if not self._val_vis_batches:
            return
        logger: Optional[Logger] = getattr(self, "logger", None)
        if logger is None:
            return

        # Use at most val_vis_num_samples samples for visualization
        num_samples = min(self.val_vis_num_samples, len(self._val_vis_batches))
        images: List[np.ndarray] = []
        captions: List[str] = []

        device = self.device
        for idx in range(num_samples):
            entry = self._val_vis_batches[idx]
            gt_xyz = entry["xyz"].to(device)
            scene_pc = entry["scene_pc"]
            if scene_pc is None:
                B = gt_xyz.size(0)
                scene_pc = torch.zeros(B, 0, 3, device=device, dtype=gt_xyz.dtype)
            else:
                scene_pc = scene_pc.to(device)

            with torch.no_grad():
                pred_xyz = self.sample(scene_pc, num_steps=self.sample_num_steps).detach().cpu()
                gt_xyz_cpu = gt_xyz.detach().cpu()

            # Only visualize the first element in the batch for this entry
            img = self._render_hand_pair_image(gt_xyz_cpu[0], pred_xyz[0])
            images.append(img)
            captions.append(f"epoch={self.current_epoch}, sample={idx}")

        # Try logger.log_image if available (e.g., WandbLogger), otherwise fall back to TensorBoard
        if hasattr(logger, "log_image"):
            logger.log_image(key="val/hand_samples", images=images, caption=captions, step=self.current_epoch)
        elif isinstance(logger, TensorBoardLogger):
            for i, img in enumerate(images):
                logger.experiment.add_image(
                    f"val/hand_samples/{i}",
                    img,
                    global_step=self.current_epoch,
                    dataformats="HWC",
                )

    def _render_hand_pair_image(self, gt_xyz: torch.Tensor, pred_xyz: torch.Tensor) -> np.ndarray:
        """Render a side-by-side image of GT vs predicted hand keypoints."""
        gt = gt_xyz.detach().cpu().numpy()
        pred = pred_xyz.detach().cpu().numpy()
        edges = self.edge_index.detach().cpu().numpy()

        fig = plt.figure(figsize=(6, 3))

        def _plot(subplot_idx: int, pts: np.ndarray, title: str, color_points: str, color_edges: str):
            ax = fig.add_subplot(1, 2, subplot_idx, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color_points, s=10)
            for e in edges.T:
                i, j = int(e[0]), int(e[1])
                xs = [pts[i, 0], pts[j, 0]]
                ys = [pts[i, 1], pts[j, 1]]
                zs = [pts[i, 2], pts[j, 2]]
                ax.plot(xs, ys, zs, color=color_edges, linewidth=0.5)
            ax.set_title(title)
            ax.set_axis_off()

        _plot(1, gt, "GT", "#1f77b4", "#1f77b4")
        _plot(2, pred, "Pred", "#d62728", "#d62728")
        plt.tight_layout()

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = fig.canvas.buffer_rgba()
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        plt.close(fig)
        return image


class _ModelEmaWrapper:
    """Lightweight EMA wrapper storing a non-trainable copy of the model."""

    def __init__(self, ema_model: nn.Module, decay: float = 0.999):
        self.ema_model = ema_model
        self.decay = float(decay)
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for k, ema_v in ema_state.items():
            if k not in model_state:
                continue
            model_v = model_state[k].detach()
            if not torch.is_floating_point(ema_v):
                ema_v.copy_(model_v.to(ema_v.device))
                continue
            model_v = model_v.to(ema_v.device)
            ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.ema_model.state_dict(), strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            return
        model.load_state_dict(self._backup, strict=False)
        self._backup = None
