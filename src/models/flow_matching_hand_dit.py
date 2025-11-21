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

from .components.losses import FlowMatchingLoss
from .components.hand_encoder import build_hand_encoder
from .components.graph_dit import build_dit
from .backbone import build_backbone
from .components.velocity_strategies import (
    build_rigid_groups,
    build_velocity_strategy,
    build_state_projector,
)
from .components.solvers import build_solver


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
        # ===== added =====
        velocity_mode: str = "direct",     # ['direct','direct_tangent','goal_kabsch','group_rigid','pbd_corrected','phys_guided']
        velocity_kwargs: Optional[Dict] = None,
        state_projection_mode: str = "pbd",# ['none','pbd','rigid','hybrid']
        state_projection_kwargs: Optional[Dict] = None,
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

        # Timestep embedding
        self.timestep_embed = TimestepEmbedding(dim=d_model)

        # ====== rigid groups: prefer provided graph consts, otherwise fallback ======
        rigid_groups = graph_consts.get("rigid_groups", None)
        if rigid_groups is not None and len(rigid_groups) > 0:
            self.rigid_groups = [g.clone().detach().long() for g in rigid_groups]
        else:
            self.rigid_groups = build_rigid_groups(self.edge_index, self.N)

        # ====== velocity strategy: use registry/factory ======
        self.velocity_mode = str(velocity_mode).lower()
        self.velocity_strategy = build_velocity_strategy(
            mode=self.velocity_mode,
            d_model=d_model,
            edge_index=self.edge_index,
            rest_lengths=self.edge_rest_lengths,
            template_xyz=self.template_xyz,
            groups=self.rigid_groups,
            kwargs=velocity_kwargs,
        )

        # ====== state projection for sampling: use registry/factory ======
        self.state_projection_mode = str(state_projection_mode).lower()
        self.state_projection_kwargs = dict(state_projection_kwargs or {})
        
        # Merge legacy proj_num_iters and proj_max_corr into kwargs if not already set
        proj_kwargs = dict(self.state_projection_kwargs)
        if self.state_projection_mode == "pbd":
            proj_kwargs.setdefault("iters", max(2, proj_num_iters))
            proj_kwargs.setdefault("max_corr", proj_max_corr)
        
        projector_result = build_state_projector(
            mode=self.state_projection_mode,
            edge_index=self.edge_index,
            rest_lengths=self.edge_rest_lengths,
            template_xyz=self.template_xyz,
            groups=self.rigid_groups,
            kwargs=proj_kwargs,
        )
        
        # Handle hybrid mode specially (returns tuple)
        if isinstance(projector_result, tuple) and projector_result[0] == "hybrid":
            _, self.sample_rigid, self.sample_xpbd = projector_result
            self.sample_state_projector = None
        else:
            self.sample_state_projector = projector_result
            self.sample_rigid = None
            self.sample_xpbd = None

        # Loss function
        self.loss_cfg = self._prepare_loss_config(loss_cfg)
        lambda_tangent = self._resolve_lambda(default_lambda=1.0)
        self.loss_fn = self._build_loss_fn(lambda_tangent)
        self.lambda_tangent = float(getattr(self.loss_fn, "lambda_tangent", lambda_tangent))

        # Optimizer configuration (Hydra-driven)
        self.optimizer_cfg = optimizer_cfg

        # Sampling configuration
        self.sample_num_steps = sample_num_steps
        self.sample_solver_name = str(sample_solver).lower()
        self.solver = build_solver(self.sample_solver_name)
        self.sample_schedule = str(sample_schedule).lower()
        self.schedule_shift = float(schedule_shift)
        self.proj_num_iters = proj_num_iters
        self.proj_max_corr = proj_max_corr

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
        """Predict velocity by the selected strategy.

        Keeps the original encoding pathway, but delegates velocity logic to strategy.
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
        # Delegate:
        velocity = self.velocity_strategy.predict(
            model=self,
            keypoints=keypoints,
            timesteps=timesteps,
            hand_tokens_out=hand_tokens_out,
        )
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

            self.log("train/loss", loss, prog_bar=False, on_step=True, on_epoch=True)
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
        except Exception as e:
            self.log("val/error_flag", 1.0, prog_bar=True, on_step=False, on_epoch=True)
            self.print(f"Validation step error at batch {batch_idx}: {e}")
            raise

    def _apply_state_projection(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Apply the configured state projection during sampling."""
        # Hybrid mode: rigid first, then PBD smoothing
        if self.sample_rigid is not None and self.sample_xpbd is not None:
            x = self.sample_rigid(keypoints)
            x = self.sample_xpbd(x)
            return x
        
        # Other modes (pbd, rigid, none)
        if self.sample_state_projector is not None:
            return self.sample_state_projector(keypoints)
        
        # None mode or no projector
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
        
        num_steps = num_steps or self.sample_num_steps
        
        if initial_noise is None:
            keypoints = torch.randn(B, N, 3, device=device)
        else:
            keypoints = initial_noise.to(device)
            
        # [Modified] Use the time schedule generator instead of linear space
        timesteps = self._generate_timesteps(num_steps, device)
        time_dtype = timesteps.dtype
        
        capture_steps: List[int] = []
        trajectory_records: List[Dict[str, Any]] = []

        def _record_state(step_idx: int, t_scalar: torch.Tensor, state: torch.Tensor) -> None:
            trajectory_records.append(
                {
                    "step": int(step_idx),
                    "t": float(t_scalar.item()),
                    "xyz": state.detach().cpu(),
                }
            )

        if return_trajectory:
            capture_steps = self._build_trajectory_capture_steps(
                num_steps=num_steps,
                explicit_indices=trajectory_indices,
                stride=trajectory_stride,
            )
            if 0 in capture_steps:
                _record_state(0, timesteps[0], keypoints)
        
        for k in range(num_steps):
            t_curr = timesteps[k]
            t_next = timesteps[k + 1]
            
            # Use solver strategy to take one integration step
            keypoints = self.solver.step(
                keypoints=keypoints,
                velocity_fn=self.predict_velocity,
                t_curr=t_curr.item(),
                t_next=t_next.item(),
                scene_pc=scene_pc,
            )
                
            # Apply kinematic constraints (configurable)
            keypoints = self._apply_state_projection(keypoints)

            if return_trajectory:
                step_idx = k + 1
                if step_idx in capture_steps:
                    _record_state(step_idx, t_next, keypoints)
            
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
