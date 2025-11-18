"""Flow Matching model with Graph-aware DiT for dexterous hand keypoint generation."""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from .components.losses import FlowMatchingLoss


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

class HandFlowMatchingDiT(pl.LightningModule):
    """Flow Matching model with Graph-aware DiT for hand keypoint generation.
    
    Generates hand keypoints conditioned on scene point clouds using rectified flow.
    
    Architecture:
        - Hand encoder: Encodes current state into per-point tokens (B, N, D)
        - Scene backbone: Processes point cloud into scene tokens (B, K, D)
        - Graph DiT: Self-attention with graph bias + cross-attention with scene
        - Velocity head: Maps tokens to velocity field (B, N, 3)
    
    Args:
        d_model: Token embedding dimension
        graph_consts: Dictionary containing graph structure:
            - finger_ids: (N,) finger indices for each keypoint
            - joint_type_ids: (N,) joint type indices
            - edge_index: (2, E) edge connectivity
            - edge_type: (E,) edge type indices
            - edge_rest_lengths: (E,) rest lengths for each edge
            - template_xyz: (N, 3) optional template keypoints
        backbone: Scene point cloud encoder module
        hand_encoder: Hand keypoint encoder module
        dit: Graph-aware DiT module
        lambda_tangent: Weight for tangent loss term
        learning_rate: Optimizer learning rate
        weight_decay: Optimizer weight decay
        use_scheduler: Whether to use learning rate scheduler
        scheduler_T_max: Cosine scheduler period in epochs
        sample_num_steps: Number of ODE steps for sampling
        proj_num_iters: Number of edge length projection iterations
        proj_max_corr: Maximum correction per projection step
    """
    def __init__(
        self,
        d_model: int,
        graph_consts: Dict[str, torch.Tensor],
        backbone: nn.Module,
        hand_encoder: nn.Module,
        dit: nn.Module,
        loss_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        lambda_tangent: float = 1.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        use_scheduler: bool = True,
        scheduler_T_max: int = 100,
        sample_num_steps: int = 40,
        sample_solver: str = "euler",
        proj_num_iters: int = 2,
        proj_max_corr: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["graph_consts", "backbone", "hand_encoder", "dit", "loss_cfg", "optimizer_cfg"]
        )

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
        self.backbone = backbone
        self.hand_encoder = hand_encoder
        self.dit = dit

        # Timestep embedding and velocity prediction head
        self.timestep_embed = TimestepEmbedding(dim=d_model)
        self.velocity_head = nn.Linear(d_model, 3)

        # Loss function
        self.loss_cfg = self._prepare_loss_config(loss_cfg)
        self.lambda_tangent = self._resolve_lambda(lambda_tangent)
        self.loss_fn = self._build_loss_fn(self.lambda_tangent)
        self.lambda_tangent = float(getattr(self.loss_fn, "lambda_tangent", self.lambda_tangent))

        # Optimizer configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_T_max = scheduler_T_max
        self.optimizer_cfg = self._prepare_optimizer_config(optimizer_cfg)

        # Sampling configuration
        self.sample_num_steps = sample_num_steps
        self.sample_solver = str(sample_solver).lower()
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

            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/loss_fm", loss_dict["loss_fm"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/loss_tangent", loss_dict["loss_tangent"], prog_bar=False, on_step=False, on_epoch=True)
            with torch.no_grad():
                edge_len_err = self._compute_edge_length_error(target)
            self.log("val/edge_len_err", edge_len_err, prog_bar=False, on_step=False, on_epoch=True)
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
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        solver = getattr(self, "sample_solver", "euler")
        for k in range(num_steps):
            t_curr = timesteps[k]
            t_next = timesteps[k + 1]
            dt = t_next - t_curr
            if solver in ("heun", "rk2"):
                # Heun's method (RK2): predictor-corrector
                t_batch_curr = torch.full((B,), t_curr.item(), device=device)
                v1 = self.predict_velocity(keypoints, scene_pc, t_batch_curr)
                keypoints_pred = keypoints + dt * v1
                t_batch_next = torch.full((B,), t_next.item(), device=device)
                v2 = self.predict_velocity(keypoints_pred, scene_pc, t_batch_next)
                keypoints = keypoints + 0.5 * dt * (v1 + v2)
            else:
                # Default: explicit Euler
                t_batch = torch.full((B,), t_curr.item(), device=device)
                velocity = self.predict_velocity(keypoints, scene_pc, t_batch)
                keypoints = keypoints + dt * velocity
            keypoints = self._project_edge_lengths(keypoints)
        return keypoints

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.optimizer_cfg:
            optimizer, scheduler_dict = self._configure_from_hydra()
            if scheduler_dict is None:
                return optimizer
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if not self.use_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_T_max,
            eta_min=0.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }

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

    def _prepare_optimizer_config(self, optimizer_cfg: Optional[DictConfig]) -> Optional[Dict[str, Any]]:
        if optimizer_cfg is None:
            return None
        if isinstance(optimizer_cfg, DictConfig):
            return OmegaConf.to_container(optimizer_cfg, resolve=True)
        if isinstance(optimizer_cfg, dict):
            return deepcopy(optimizer_cfg)
        raise TypeError("optimizer_cfg must be a DictConfig or dict")

    def _configure_from_hydra(self):
        cfg = deepcopy(self.optimizer_cfg)
        scheduler_cfg = cfg.pop("scheduler", None)
        optimizer = self._instantiate_optimizer(cfg)
        scheduler_dict = self._instantiate_scheduler(optimizer, scheduler_cfg)
        return optimizer, scheduler_dict

    def _instantiate_optimizer(self, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        cfg = deepcopy(cfg)
        target = cfg.pop("_target_", None)
        opt_type = cfg.pop("type", None)
        if target:
            hydra_cfg = OmegaConf.create({"_target_": target, **cfg})
            return hydra_instantiate(hydra_cfg, params=self.parameters())
        opt_key = (opt_type or "adamw").lower()
        aliases = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if opt_key not in aliases:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        optimizer_cls = aliases[opt_key]
        return optimizer_cls(self.parameters(), **cfg)

    def _instantiate_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not scheduler_cfg:
            return None
        cfg = deepcopy(scheduler_cfg)
        interval = cfg.pop("interval", "epoch")
        frequency = int(cfg.pop("frequency", 1))
        monitor = cfg.pop("monitor", None)
        reduce_on_plateau_flag = bool(cfg.pop("reduce_on_plateau", False))
        target = cfg.pop("_target_", None)
        sched_type = cfg.pop("type", None)
        if target:
            hydra_cfg = OmegaConf.create({"_target_": target, **cfg})
            scheduler = hydra_instantiate(hydra_cfg, optimizer=optimizer)
        else:
            scheduler = self._build_legacy_scheduler(optimizer, sched_type, cfg)
        if scheduler is None:
            return None
        scheduler_dict = {"scheduler": scheduler, "interval": interval, "frequency": frequency}
        if monitor:
            scheduler_dict["monitor"] = monitor
        if reduce_on_plateau_flag or isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_dict["reduce_on_plateau"] = True
            if not monitor:
                scheduler_dict["monitor"] = "val/loss"
        return scheduler_dict

    def _build_legacy_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        sched_type: Optional[str],
        cfg: Dict[str, Any],
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if not sched_type or sched_type.lower() == "none":
            return None
        sched_key = sched_type.lower()
        if sched_key == "cosine":
            t_max = cfg.get("t_max") or cfg.get("T_max") or self.scheduler_T_max
            eta_min = cfg.get("eta_min", 0.0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(t_max)), eta_min=eta_min)
        if sched_key == "step":
            step_size = int(cfg.get("step_size", 30))
            gamma = float(cfg.get("gamma", 0.1))
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if sched_key == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.get("mode", "min"),
                factor=float(cfg.get("factor", 0.5)),
                patience=int(cfg.get("patience", 10)),
                threshold=float(cfg.get("threshold", 1e-4)),
                min_lr=float(cfg.get("min_lr", 0.0)),
            )
        if sched_key == "warmup_cosine":
            warmup_epochs = int(cfg.get("warmup_epochs", 0))
            total_epochs = int(cfg.get("total_epochs", self.scheduler_T_max))
            eta_min = float(cfg.get("eta_min", 0.0))
            start_factor = float(cfg.get("warmup_start_factor", cfg.get("start_factor", 1e-3)))
            if warmup_epochs <= 0:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_epochs),
                    eta_min=eta_min,
                )
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=max(1e-8, start_factor),
                end_factor=1.0,
                total_iters=max(1, warmup_epochs),
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_epochs - warmup_epochs),
                eta_min=eta_min,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
