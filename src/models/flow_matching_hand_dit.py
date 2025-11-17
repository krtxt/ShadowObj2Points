# models/flow_matching_hand_dit.py
# ------------------------------------------------------------
# Flow Matching + Graph-aware DiT for dexterous hand keypoints
# ------------------------------------------------------------
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.backbone import build_backbone
from .components.hand_encoder import (
    HandPointTokenEncoderTransformerBias,  # or HandPointTokenEncoderEGNNLite
)
from .components.graph_dit import (
    HandSceneGraphDiT,
    GraphAttentionBias,
)


# ------------------------------
# 小工具：时间嵌入、旋转等
# ------------------------------
class TimeEmbedding(nn.Module):
    """
    简单的标量时间嵌入： t ∈ [0,1] → (B, dim)

    结构：Linear(1→4*dim) + SiLU + Linear(4*dim→dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B,1) in [0,1]
        Returns:
            emb: (B, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,1)
        return self.net(t)


# ------------------------------
# Flow Matching LightningModule
# ------------------------------
class HandFlowMatchingDiT(pl.LightningModule):
    """
    Flow Matching + Graph-aware DiT 模型，用于：
      - 条件在物体点云 scene_pc 上
      - 生成手部 23 个关键点坐标 xyz ∈ ℝ^{B×N×3}

    设计：
      - 数据空间：直接在 xyz（世界坐标）空间做 flow matching
      - Hand encoder: 把当前状态 y_tau 编成 per-point tokens (B,N,D)
      - Scene backbone: 点云 → scene tokens (B,K,D)
      - Graph DiT: hand tokens 做 self-attn（带 graph bias），然后 cross-attn 融入 scene tokens
      - Velocity head: (B,N,D) → (B,N,3)，输出速度场 v_hat(y_tau, t; scene)

    输入 batch 约定：
      batch["xyz"]:      (B, N, 3)      # GT 关键点世界坐标
      batch["scene_pc"]: (B, P, 3)      # 物体点云

    graph_consts 约定（来自 HandEncoderDataModule.get_graph_constants()）：
      {
        "finger_ids": (N,),
        "joint_type_ids": (N,),
        "edge_index": (2, E),
        "edge_type": (E,),
        "edge_rest_lengths": (E,),
        "template_xyz": (N,3),   # 可选，用于额外几何先验
      }
    """

    def __init__(
        self,
        # 模型维度 & 结构
        d_model: int,
        dit_depth: int,
        dit_heads: int,
        # 图结构常量
        graph_consts: Dict[str, torch.Tensor],
        # 点云 backbone 配置（传给 build_backbone）
        backbone_cfg: Dict[str, Any],
        # Hand encoder 结构
        hand_encoder_type: str = "transformer",  # "transformer" or "egnn"
        hand_encoder_layers: int = 3,
        num_frequencies: int = 10,
        # Flow Matching 损失权重
        lambda_tangent: float = 1.0,
        # 训练优化参数
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        use_scheduler: bool = True,
        scheduler_T_max: int = 100,  # epoch 级 T_max
        # 采样/投影
        sample_num_steps: int = 40,
        proj_num_iters: int = 2,
        proj_max_corr: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["graph_consts"])

        # ------------------------------
        # 1. 解析图结构常量
        # ------------------------------
        finger_ids = graph_consts["finger_ids"].long()          # (N,)
        joint_type_ids = graph_consts["joint_type_ids"].long()  # (N,)
        edge_index = graph_consts["edge_index"].long()          # (2,E)
        edge_type = graph_consts["edge_type"].long()            # (E,)
        edge_rest_lengths = graph_consts["edge_rest_lengths"].float()  # (E,)

        N = finger_ids.shape[0]
        num_fingers = int(finger_ids.max().item()) + 1
        num_joint_types = int(joint_type_ids.max().item()) + 1
        num_edge_types = int(edge_type.max().item()) + 1

        self.N = N
        self.num_fingers = num_fingers
        self.num_joint_types = num_joint_types
        self.num_edge_types = num_edge_types

        # 这些 buffer 用于每个 batch 复制
        self.register_buffer("finger_ids_const", finger_ids, persistent=False)          # (N,)
        self.register_buffer("joint_type_ids_const", joint_type_ids, persistent=False)  # (N,)
        self.register_buffer("edge_index", edge_index, persistent=False)                # (2,E)
        self.register_buffer("edge_type", edge_type, persistent=False)                  # (E,)
        self.register_buffer("edge_rest_lengths", edge_rest_lengths, persistent=False)  # (E,)

        # 可选：模板关键点
        template_xyz = graph_consts.get("template_xyz", None)
        if template_xyz is not None:
            self.register_buffer("template_xyz", template_xyz.float(), persistent=False)
        else:
            self.template_xyz = None

        # ------------------------------
        # 2. 点云 backbone（物体点云 → scene tokens）
        # ------------------------------
        self.backbone = build_backbone(backbone_cfg)  # 约定 forward(scene_pc) -> (B,K,d_model) 或 dict

        # ------------------------------
        # 3. Hand encoder: (B,N,3) → (B,N,d_model)
        # ------------------------------
        if hand_encoder_type.lower() == "transformer":
            self.hand_encoder = HandPointTokenEncoderTransformerBias(
                num_fingers=num_fingers,
                num_joint_types=num_joint_types,
                num_edge_types=num_edge_types,
                d_model=d_model // 4,        # 中间 dim，可以调
                n_layers=hand_encoder_layers,
                n_heads=4,
                d_struct=8,
                num_frequencies=num_frequencies,
                out_dim=d_model,
                dropout=0.1,
            )
        elif hand_encoder_type.lower() == "egnn":
            from .components.hand_encoder import HandPointTokenEncoderEGNNLite
            self.hand_encoder = HandPointTokenEncoderEGNNLite(
                num_fingers=num_fingers,
                num_joint_types=num_joint_types,
                num_edge_types=num_edge_types,
                d_model=d_model // 4,
                n_layers=hand_encoder_layers,
                d_edge=64,
                d_struct=8,
                num_frequencies=num_frequencies,
                out_dim=d_model,
                dropout=0.1,
            )
        else:
            raise ValueError(f"Unknown hand_encoder_type: {hand_encoder_type}")

        # ------------------------------
        # 4. Graph attention bias + DiT
        # ------------------------------
        self.graph_bias = GraphAttentionBias(
            num_nodes=N,
            num_edge_types=num_edge_types,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_rest_lengths=edge_rest_lengths,
            d_struct=8,
        )

        self.dit = HandSceneGraphDiT(
            dim=d_model,
            depth=dit_depth,
            num_heads=dit_heads,
            cross_attention_dim=d_model,
            graph_bias=self.graph_bias,
            dropout=0.0,
            activation_fn="geglu",
            qk_norm=True,
        )

        # ------------------------------
        # 5. 时间嵌入 & 速度头
        # ------------------------------
        self.time_embed = TimeEmbedding(dim=d_model)
        self.vel_head = nn.Linear(d_model, 3)  # per-point 速度 (B,N,3)

        # ------------------------------
        # 6. Flow Matching 损失参数
        # ------------------------------
        self.lambda_tangent = lambda_tangent

        # ------------------------------
        # 7. 优化器 & 调度参数
        # ------------------------------
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_T_max = scheduler_T_max

        # ------------------------------
        # 8. 采样/投影参数
        # ------------------------------
        self.sample_num_steps = sample_num_steps
        self.proj_num_iters = proj_num_iters
        self.proj_max_corr = proj_max_corr

    # ------------------------------
    # 编码：scene tokens & hand tokens
    # ------------------------------
    def encode_scene_tokens(self, scene_pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_pc: (B, P, 3)
        Returns:
            scene_tokens: (B, K, d_model)
        """
        scene_tokens = self.backbone(scene_pc)
        # 允许 backbone 输出 dict
        if isinstance(scene_tokens, dict):
            # 尝试常见键名
            for key in ["tokens", "scene_tokens", "feat"]:
                if key in scene_tokens:
                    scene_tokens = scene_tokens[key]
                    break
            else:
                raise KeyError(
                    "Backbone returned a dict, but no 'tokens'/'scene_tokens'/'feat' key found."
                )
        return scene_tokens  # (B,K,d_model)

    def encode_hand_tokens(self, y_tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_tau: (B, N, 3) 当前状态（flow path 上的点）

        Returns:
            hand_tokens: (B, N, d_model)
        """
        B, N, _ = y_tau.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"

        finger_ids = self.finger_ids_const.unsqueeze(0).expand(B, -1)        # (B,N)
        joint_type_ids = self.joint_type_ids_const.unsqueeze(0).expand(B, -1)  # (B,N)

        hand_tokens = self.hand_encoder(
            xyz=y_tau,
            finger_ids=finger_ids,
            joint_type_ids=joint_type_ids,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_rest_lengths=self.edge_rest_lengths,
        )  # (B,N,d_model)
        return hand_tokens

    # ------------------------------
    # 核心：预测速度场 v_hat
    # ------------------------------
    def predict_velocity(
        self,
        y_tau: torch.Tensor,        # (B,N,3)
        scene_pc: torch.Tensor,     # (B,P,3)
        tau: torch.Tensor,          # (B,) in [0,1]
    ) -> torch.Tensor:
        """
        核心前向：给定 y_tau & scene_pc & 时间 tau，预测速度 v_hat(y_tau, tau; scene)。

        Returns:
            v_hat: (B, N, 3)
        """
        scene_tokens = self.encode_scene_tokens(scene_pc)  # (B,K,D)
        hand_tokens = self.encode_hand_tokens(y_tau)       # (B,N,D)
        temb = self.time_embed(tau)                        # (B,D)

        # Graph-aware DiT：self-attn on hand + cross-attn with scene
        hand_tokens_out = self.dit(
            hand_tokens=hand_tokens,
            scene_tokens=scene_tokens,
            temb=temb,
            xyz=y_tau,      # 用当前坐标构造 graph attention bias
        )  # (B,N,D)

        v_hat = self.vel_head(hand_tokens_out)  # (B,N,3)
        return v_hat

    # ------------------------------
    # Flow Matching 构造 & 损失
    # ------------------------------
    def _flow_matching_path(
        self,
        y1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构造 Rectified Flow 的路径：
          y0 ~ N(0,I)
          tau ~ U(0,1)
          y_tau = (1-tau) y0 + tau y1
          v_star = y1 - y0

        Args:
            y1: (B,N,3) GT 关键点

        Returns:
            y0:     (B,N,3)
            tau:    (B,)
            y_tau:  (B,N,3)
            v_star: (B,N,3)
        """
        B = y1.size(0)
        device = y1.device

        y0 = torch.randn_like(y1)
        tau = torch.rand(B, device=device)  # (B,)
        tau_ = tau.view(B, 1, 1)

        y_tau = (1.0 - tau_) * y0 + tau_ * y1
        v_star = y1 - y0
        return y0, tau, y_tau, v_star

    def _tangent_loss(
        self,
        y_tau: torch.Tensor,  # (B,N,3)
        v_hat: torch.Tensor,  # (B,N,3)
    ) -> torch.Tensor:
        """
        切向约束：对每条边 (i,j) 施加
            ((y_i - y_j) · (v_i - v_j))^2
        让速度尽量在约束流形的切空间上。
        """
        i, j = self.edge_index  # (E,)
        diff_y = y_tau[:, i, :] - y_tau[:, j, :]  # (B,E,3)
        diff_v = v_hat[:, i, :] - v_hat[:, j, :]  # (B,E,3)
        residual = (diff_y * diff_v).sum(-1)      # (B,E)
        return (residual ** 2).mean()

    # ------------------------------
    # Lightning: training / validation
    # ------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        预期 batch:
          batch["xyz"]      : (B,N,3)
          batch["scene_pc"] : (B,P,3)
        """
        y1 = batch["xyz"].to(self.device)            # (B,N,3)
        scene_pc = batch["scene_pc"].to(self.device) # (B,P,3)

        _, tau, y_tau, v_star = self._flow_matching_path(y1)  # tau: (B,)

        v_hat = self.predict_velocity(y_tau, scene_pc, tau)   # (B,N,3)

        loss_fm = F.mse_loss(v_hat, v_star)

        if self.lambda_tangent > 0:
            loss_tan = self._tangent_loss(y_tau, v_hat)
        else:
            loss_tan = torch.tensor(0.0, device=self.device)

        loss = loss_fm + self.lambda_tangent * loss_tan

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_fm", loss_fm, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss_tangent", loss_tan, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        y1 = batch["xyz"].to(self.device)
        scene_pc = batch["scene_pc"].to(self.device)

        _, tau, y_tau, v_star = self._flow_matching_path(y1)
        v_hat = self.predict_velocity(y_tau, scene_pc, tau)

        loss_fm = F.mse_loss(v_hat, v_star)

        if self.lambda_tangent > 0:
            loss_tan = self._tangent_loss(y_tau, v_hat)
        else:
            loss_tan = torch.tensor(0.0, device=self.device)

        loss = loss_fm + self.lambda_tangent * loss_tan

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/loss_fm", loss_fm, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/loss_tangent", loss_tan, prog_bar=False, on_step=False, on_epoch=True)

        # 可选：评估边长误差
        with torch.no_grad():
            edge_len_err = self._edge_length_error(y1)
        self.log("val/edge_len_err", edge_len_err, prog_bar=False, on_step=False, on_epoch=True)

    # ------------------------------
    # 约束投影（PBD-style） & 误差评估
    # ------------------------------
    def _edge_length_projection(self, y: torch.Tensor) -> torch.Tensor:
        """
        Position-Based Dynamics (PBD) 风格的边长投影，迭代 num_iters 次。

        Args:
            y: (B,N,3)

        Returns:
            y_proj: (B,N,3) 经过若干次投影后的坐标
        """
        if self.proj_num_iters <= 0:
            return y

        y = y.clone()
        i, j = self.edge_index  # (E,)
        rest = self.edge_rest_lengths.view(1, -1, 1).to(y)  # (1,E,1)

        for _ in range(self.proj_num_iters):
            diff = y[:, i, :] - y[:, j, :]          # (B,E,3)
            dist = (diff.pow(2).sum(-1, keepdim=True) + 1e-9).sqrt()  # (B,E,1)
            corr = (1.0 - rest / dist).clamp(-self.proj_max_corr, self.proj_max_corr)  # (B,E,1)
            delta = corr * diff * 0.5              # (B,E,3) 每个点分担一半

            # 把 delta 累加到节点 i、j
            y.index_add_(1, i, -delta)
            y.index_add_(1, j,  delta)

        return y

    def _edge_length_error(self, y: torch.Tensor) -> torch.Tensor:
        """
        计算当前 y 下边长相对误差的平均值，用于监控。
        """
        i, j = self.edge_index
        diff = y[:, i, :] - y[:, j, :]              # (B,E,3)
        dist = (diff.pow(2).sum(-1) + 1e-9).sqrt()  # (B,E)
        rest = self.edge_rest_lengths.view(1, -1).to(y)  # (1,E)
        rel_err = (dist - rest).abs() / (rest + 1e-6)    # (B,E)
        return rel_err.mean()

    # ------------------------------
    # 采样接口：给定 scene_pc 生成抓取
    # ------------------------------
    @torch.no_grad()
    def sample(
        self,
        scene_pc: torch.Tensor,  # (B,P,3)
        num_steps: Optional[int] = None,
        y_init: Optional[torch.Tensor] = None,  # (B,N,3)
    ) -> torch.Tensor:
        """
        从高斯噪声出发，沿着 learned flow 积分，生成抓取关键点 xyz_pred。

        Args:
            scene_pc: (B,P,3)
            num_steps: ODE 步数，默认用 self.sample_num_steps
            y_init:   可选初始点；不提供则 ~N(0,I)

        Returns:
            y: (B,N,3) 采样得到的关键点坐标
        """
        self.eval()
        device = scene_pc.device
        B = scene_pc.size(0)
        N = self.N

        num_steps = num_steps or self.sample_num_steps

        if y_init is None:
            y = torch.randn(B, N, 3, device=device)
        else:
            y = y_init.to(device)

        # 时间网格：0 -> 1
        t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for k in range(num_steps):
            t0 = t_vals[k]
            t1 = t_vals[k + 1]
            dt = t1 - t0

            tau = torch.full((B,), t0.item(), device=device)
            v = self.predict_velocity(y, scene_pc, tau)  # (B,N,3)

            # 简单 Euler 步
            y = y + dt * v

            # 每步做一次 PBD 投影，保持边长约束
            y = self._edge_length_projection(y)

        return y

    # ------------------------------
    # Optimizer & Scheduler
    # ------------------------------
    def configure_optimizers(self):
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
