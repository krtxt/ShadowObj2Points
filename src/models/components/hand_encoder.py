# hand_encoders.py
# ------------------------------------------------------------
# Hand encoders that map N hand keypoints to per-point tokens
# of dimension out_dim (default 512), with strong inductive
# biases for dexterous hands / kinematic trees.
# ------------------------------------------------------------
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse FourierPositionalEmbedding from the provided attachment
# (see embeddings.py)
from .embeddings import FourierPositionalEmbedding  # noqa: F401


# -----------------------------
# Small Shared Building Blocks
# -----------------------------
class HandPointEmbedding(nn.Module):
    """
    点级特征构建：
      Fourier(xyz) + finger_id Embedding + joint_type Embedding -> 投到 d_model

    Args:
        num_fingers: 手指数目（finger_id 取值范围大小）
        num_joint_types: 关节类型数（joint_type_id 取值范围大小）
        d_model: 中间特征维度（会再投到 out_dim）
        finger_dim: finger embedding 维度
        joint_dim: joint embedding 维度
        num_frequencies: Fourier 频率数
    """
    def __init__(
        self,
        num_fingers: int,
        num_joint_types: int,
        d_model: int = 128,
        finger_dim: int = 16,
        joint_dim: int = 16,
        num_frequencies: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fourier = FourierPositionalEmbedding(
            num_frequencies=num_frequencies,
            include_input=True,
            log_sampling=True,
        )
        self.finger_emb = nn.Embedding(num_fingers, finger_dim)
        self.joint_emb = nn.Embedding(num_joint_types, joint_dim)

        in_dim = self.fourier.output_dim + finger_dim + joint_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        finger_ids: torch.Tensor,
        joint_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)
            finger_ids: (B, N)
            joint_type_ids: (B, N)

        Returns:
            h: (B, N, d_model)
        """
        f_xyz = self.fourier(xyz)                    # (B, N, Df)
        f_finger = self.finger_emb(finger_ids)       # (B, N, df)
        f_joint = self.joint_emb(joint_type_ids)     # (B, N, dj)
        feats = torch.cat([f_xyz, f_finger, f_joint], dim=-1)
        h = self.proj(feats)
        return h


class PMA1(nn.Module):
    """
    Set Transformer 风格的 PMA（k=1）：将 N 个点级 token 汇聚为 1 个全局 token。
    当前文件里新加的 per-point encoder 不再用它，但保留以向后兼容。
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))  # (1,1,d)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, N, d)
        Returns:
            z: (B, d)
        """
        B = H.size(0)
        Q = self.seed.expand(B, -1, -1)  # (B,1,d)
        Z, _ = self.attn(Q, H, H)        # (B,1,d)
        return self.ln(Z.squeeze(1))     # (B,d)


def _pairwise_edge_geom(
    xyz: torch.Tensor,
    edge_index: torch.Tensor,
    edge_rest_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于 batch 的边几何特征：dist, dist2, delta_d

    Args:
        xyz: (B, N, 3)
        edge_index: (2, E)
        edge_rest_lengths: (E,)

    Returns:
        dist:  (B, E, 1)
        dist2: (B, E, 1)
        delta:(B, E, 1)  # (dist - rest) / rest
    """
    i, j = edge_index  # (E,), (E,)
    diff = xyz[:, i, :] - xyz[:, j, :]              # (B, E, 3)
    dist2 = (diff ** 2).sum(-1, keepdim=True)       # (B, E, 1)
    dist = torch.sqrt(dist2 + 1e-9)                 # (B, E, 1)
    rest = edge_rest_lengths.view(1, -1, 1).to(xyz) # (1, E, 1)
    delta = (dist - rest) / (rest + 1e-9)           # (B, E, 1)
    return dist, dist2, delta


# -----------------------------------------
# Scheme A: EGNN-lite (no coord updates)
# -----------------------------------------
class EdgeStructEmbedding(nn.Module):
    """
    边结构 Embedding（例如：骨架边 / 跨指边 / 掌星形边 ...）

    Args:
        num_edge_types: edge_type 取值范围大小
        d_struct: 边结构 embedding 维度
    """
    def __init__(self, num_edge_types: int, d_struct: int = 8):
        super().__init__()
        self.emb = nn.Embedding(num_edge_types, d_struct)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        # edge_type: (E,)
        return self.emb(edge_type)  # (E, d_struct)


class EGNNLiteLayer(nn.Module):
    """
    极简 EGNN 风格层：仅做特征更新（不更新坐标），
    消息中显式引入 dist^2 / delta_d / 结构 Embedding，并以门控形式作为增益。

    输入输出：
        H: (B, N, d_model) -> (B, N, d_model)
    """
    def __init__(
        self,
        d_model: int,
        d_edge: int = 64,
        d_struct: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1 + 1 + d_struct, d_edge),  # [h_i, h_j, dist2, delta, struct]
            nn.SiLU(),
            nn.Linear(d_edge, d_edge),
            nn.SiLU(),
        )
        self.gate_mlp = nn.Sequential(  # 产生标量门控
            nn.Linear(1 + 1 + d_struct, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(d_model + d_edge, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        H: torch.Tensor,              # (B, N, d)
        xyz: torch.Tensor,            # (B, N, 3)
        edge_index: torch.Tensor,     # (2, E)
        edge_struct: torch.Tensor,    # (E, d_struct)
        edge_rest_lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, N, d = H.shape
        E = edge_index.size(1)
        i, j = edge_index  # (E,), (E,)

        Hi = H[:, i, :]                              # (B, E, d)
        Hj = H[:, j, :]                              # (B, E, d)
        dist, dist2, delta = _pairwise_edge_geom(
            xyz, edge_index, edge_rest_lengths
        )                                            # (B, E, 1)...
        struct = edge_struct.unsqueeze(0).expand(B, E, -1)  # (B, E, d_struct)

        # edge message
        e_in = torch.cat([Hi, Hj, dist2, delta, struct], dim=-1)  # (B,E,2d+1+1+d_struct)
        e_msg = self.edge_mlp(e_in)                               # (B,E,d_edge)

        # gate
        g = self.gate_mlp(torch.cat([dist2, delta, struct], dim=-1))  # (B,E,1)
        e_msg = e_msg * g  # (B,E,d_edge)

        # aggregate to nodes: sum over incoming edges j->i
        agg = torch.zeros(
            B, N, e_msg.size(-1),
            device=H.device,
            dtype=H.dtype,
        )  # (B, N, d_edge)
        idx = i.view(1, E, 1).expand(B, E, e_msg.size(-1))
        agg.scatter_add_(dim=1, index=idx, src=e_msg)

        # node update (residual + LN)
        H_new = self.ln(H + self.node_mlp(torch.cat([H, agg], dim=-1)))
        return H_new


class HandEncoderEGNNLiteGlobal(nn.Module):
    """
    原始方案 A：轻量等变图编码器（不更新坐标）+ PMA(k=1) -> 单个 512 维 hand-token
    （保留以向后兼容；**新方案使用下面的 HandPointTokenEncoderEGNNLite**）

    forward:
        xyz: (B, N, 3)
        finger_ids: (B, N)
        joint_type_ids: (B, N)
        edge_index: (2, E)
        edge_type: (E,)
        edge_rest_lengths: (E,)
      -> (B, 512)
    """
    def __init__(
        self,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,
        n_layers: int = 3,
        d_edge: int = 64,
        d_struct: int = 8,
        num_frequencies: int = 10,
        n_heads_pma: int = 4,
        out_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_embed = HandPointEmbedding(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            d_model=d_model,
            num_frequencies=num_frequencies,
            dropout=dropout,
        )
        self.edge_struct_emb = EdgeStructEmbedding(num_edge_types, d_struct=d_struct)
        self.layers = nn.ModuleList(
            [
                EGNNLiteLayer(
                    d_model=d_model,
                    d_edge=d_edge,
                    d_struct=d_struct,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.pma = PMA1(d_model=d_model, n_heads=n_heads_pma, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    @torch.no_grad()
    def _edge_struct(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.edge_struct_emb(edge_type)  # (E, d_struct)

    def forward(
        self,
        xyz: torch.Tensor,                 # (B, N, 3)
        finger_ids: torch.Tensor,          # (B, N)
        joint_type_ids: torch.Tensor,      # (B, N)
        edge_index: torch.Tensor,          # (2, E)
        edge_type: torch.Tensor,           # (E,)
        edge_rest_lengths: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        H = self.point_embed(xyz, finger_ids, joint_type_ids)     # (B, N, d)
        edge_struct = self._edge_struct(edge_type)                # (E, d_struct)
        for layer in self.layers:
            H = layer(H, xyz, edge_index, edge_struct, edge_rest_lengths)
        z = self.pma(H)                                           # (B, d)
        return self.out(z)                                        # (B, 512)


class HandPointTokenEncoderEGNNLite(nn.Module):
    """
    新方案 A（你要用的这个）：
      轻量等变图编码器（不更新坐标） -> 按点输出 token，用于后续 DiT。

    forward 输入：
        xyz: (B, N, 3)
        finger_ids: (B, N)
        joint_type_ids: (B, N)
        edge_index: (2, E)
        edge_type: (E,)
        edge_rest_lengths: (E,)

    输出：
        tokens: (B, N, out_dim)   # out_dim 默认 512，可以直接作为 DiT 的 d_model
    """
    def __init__(
        self,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,            # 中间维度
        n_layers: int = 3,
        d_edge: int = 64,
        d_struct: int = 8,
        num_frequencies: int = 10,
        out_dim: int = 512,           # 每个点的 token 维度
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_embed = HandPointEmbedding(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            d_model=d_model,
            num_frequencies=num_frequencies,
            dropout=dropout,
        )
        self.edge_struct_emb = EdgeStructEmbedding(num_edge_types, d_struct=d_struct)
        self.layers = nn.ModuleList(
            [
                EGNNLiteLayer(
                    d_model=d_model,
                    d_edge=d_edge,
                    d_struct=d_struct,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # 最后一层将 d_model 映射到 out_dim（默认 512），按点输出
        self.out = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, out_dim),
            nn.LayerNorm(out_dim),
        )

    @torch.no_grad()
    def _edge_struct(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.edge_struct_emb(edge_type)  # (E, d_struct)

    def forward(
        self,
        xyz: torch.Tensor,                 # (B, N, 3)
        finger_ids: torch.Tensor,          # (B, N)
        joint_type_ids: torch.Tensor,      # (B, N)
        edge_index: torch.Tensor,          # (2, E)
        edge_type: torch.Tensor,           # (E,)
        edge_rest_lengths: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """
        Returns:
            tokens: (B, N, out_dim)
        """
        H = self.point_embed(xyz, finger_ids, joint_type_ids)     # (B, N, d_model)
        edge_struct = self._edge_struct(edge_type)                # (E, d_struct)
        for layer in self.layers:
            H = layer(H, xyz, edge_index, edge_struct, edge_rest_lengths)
        tokens = self.out(H)                                      # (B, N, out_dim)
        return tokens


# ------------------------------------------------
# Scheme B: Transformer + structural attention bias
# ------------------------------------------------
class BiasedMHSA(nn.Module):
    """
    自定义多头自注意力，允许加性注意力偏置 attn_bias (B, 1, N, N)。

    注意：这里直接手写 attention（而不是用 scaled_dot_product_attention 的 attn_mask），
    更直观也更稳定。
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d)
            attn_bias: (B, 1, N, N) 或 None，加入到注意力 logits（加性偏置）

        Returns:
            y: (B, N, d)
        """
        B, N, D = x.shape
        h = self.n_heads
        dh = self.d_head

        q = self.q_proj(x).view(B, N, h, dh).transpose(1, 2)  # (B,h,N,dh)
        k = self.k_proj(x).view(B, N, h, dh).transpose(1, 2)  # (B,h,N,dh)
        v = self.v_proj(x).view(B, N, h, dh).transpose(1, 2)  # (B,h,N,dh)

        # (B,h,N,dh) x (B,h,dh,N) -> (B,h,N,N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dh**0.5)

        if attn_bias is not None:
            # attn_bias: (B,1,N,N) -> broadcast to (B,h,N,N)
            scores = scores + attn_bias

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.matmul(attn, v)          # (B,h,N,dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B,N,d)
        return self.o_proj(out)


class TransformerEncoderLayerBias(nn.Module):
    """
    Pre-LN Transformer 编码层，内置 BiasedMHSA。
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        ffn_ratio: int = 2,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BiasedMHSA(d_model, n_heads=n_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        d_ff = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class EdgeBiasBuilder(nn.Module):
    """
    将边几何/结构特征映射为注意力偏置矩阵 (B, 1, N, N)，
    非边位置使用一个可学习的默认偏置。
    """
    def __init__(self, d_struct: int = 8):
        super().__init__()
        self.edge_to_bias = nn.Sequential(
            nn.Linear(1 + 1 + d_struct, 32),  # [dist2, delta, struct_emb]
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # 默认偏置，可以初始化为 0 或负值
        self.non_edge_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        xyz: torch.Tensor,               # (B, N, 3)
        edge_index: torch.Tensor,        # (2, E)
        edge_struct: torch.Tensor,       # (E, d_struct)
        edge_rest_lengths: torch.Tensor, # (E,)
    ) -> torch.Tensor:
        B, N, _ = xyz.shape
        E = edge_index.size(1)
        i, j = edge_index

        dist, dist2, delta = _pairwise_edge_geom(
            xyz, edge_index, edge_rest_lengths
        )                               # (B,E,1)...
        struct = edge_struct.unsqueeze(0).expand(B, E, -1)   # (B,E,d_struct)

        edge_feat = torch.cat([dist2, delta, struct], dim=-1)  # (B,E, 1+1+d_struct)
        edge_bias = self.edge_to_bias(edge_feat).squeeze(-1)   # (B,E)

        # NxN 偏置矩阵，非边位置使用可学习的 non_edge_bias 作为初始值
        bias = torch.ones(
            (B, N, N),
            device=xyz.device,
            dtype=xyz.dtype,
        ) * self.non_edge_bias
        # 写入 i->j 与 j->i（无向）
        for b in range(B):
            bias[b].index_put_((i, j), edge_bias[b])
            bias[b].index_put_((j, i), edge_bias[b])

        # (B,1,N,N) 以便广播到 (B,h,N,N)
        return bias.unsqueeze(1)


class HandEncoderTransformerBiasGlobal(nn.Module):
    """
    原始方案 B：标准 Transformer（带结构/骨长注意力偏置）+ PMA(k=1)
    -> 单个 512 维 hand-token。
    （保留以向后兼容；**新方案使用 HandPointTokenEncoderTransformerBias**）
    """
    def __init__(
        self,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_struct: int = 8,
        num_frequencies: int = 10,
        n_heads_pma: int = 4,
        out_dim: int = 512,
        dropout: float = 0.1,
        ffn_ratio: int = 2,
    ):
        super().__init__()
        self.point_embed = HandPointEmbedding(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            d_model=d_model,
            num_frequencies=num_frequencies,
            dropout=dropout,
        )
        self.edge_struct_emb = EdgeStructEmbedding(num_edge_types, d_struct=d_struct)
        self.bias_builder = EdgeBiasBuilder(d_struct=d_struct)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayerBias(
                    d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_ratio=ffn_ratio,
                )
                for _ in range(n_layers)
            ]
        )

        self.pma = PMA1(d_model=d_model, n_heads=n_heads_pma, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    @torch.no_grad()
    def _edge_struct(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.edge_struct_emb(edge_type)  # (E, d_struct)

    def forward(
        self,
        xyz: torch.Tensor,                 # (B, N, 3)
        finger_ids: torch.Tensor,          # (B, N)
        joint_type_ids: torch.Tensor,      # (B, N)
        edge_index: torch.Tensor,          # (2, E)
        edge_type: torch.Tensor,           # (E,)
        edge_rest_lengths: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        H = self.point_embed(xyz, finger_ids, joint_type_ids)      # (B, N, d)
        edge_struct = self._edge_struct(edge_type)                 # (E, d_struct)
        attn_bias = self.bias_builder(
            xyz, edge_index, edge_struct, edge_rest_lengths
        )                                                           # (B,1,N,N)

        for blk in self.blocks:
            H = blk(H, attn_bias)

        z = self.pma(H)                                             # (B,d)
        return self.out(z)                                          # (B,512)


class HandPointTokenEncoderTransformerBias(nn.Module):
    """
    新方案 B（你要用的这个）：
      标准 Transformer（带结构/骨长注意力偏置） -> 按点输出 token。

    forward 输入：
        xyz: (B, N, 3)
        finger_ids: (B, N)
        joint_type_ids: (B, N)
        edge_index: (2, E)
        edge_type: (E,)
        edge_rest_lengths: (E,)

    输出：
        tokens: (B, N, out_dim)   # out_dim 默认 512，可直接作为 DiT 的输入 token
    """
    def __init__(
        self,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,          # 中间维度
        n_layers: int = 3,
        n_heads: int = 4,
        d_struct: int = 8,
        num_frequencies: int = 10,
        out_dim: int = 512,          # 每个点的 token 维度
        dropout: float = 0.1,
        ffn_ratio: int = 2,
    ):
        super().__init__()
        self.point_embed = HandPointEmbedding(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            d_model=d_model,
            num_frequencies=num_frequencies,
            dropout=dropout,
        )
        self.edge_struct_emb = EdgeStructEmbedding(num_edge_types, d_struct=d_struct)
        self.bias_builder = EdgeBiasBuilder(d_struct=d_struct)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayerBias(
                    d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_ratio=ffn_ratio,
                )
                for _ in range(n_layers)
            ]
        )

        self.out = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, out_dim),
            nn.LayerNorm(out_dim),
        )

    @torch.no_grad()
    def _edge_struct(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.edge_struct_emb(edge_type)  # (E, d_struct)

    def forward(
        self,
        xyz: torch.Tensor,                 # (B, N, 3)
        finger_ids: torch.Tensor,          # (B, N)
        joint_type_ids: torch.Tensor,      # (B, N)
        edge_index: torch.Tensor,          # (2, E)
        edge_type: torch.Tensor,           # (E,)
        edge_rest_lengths: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """
        Returns:
            tokens: (B, N, out_dim)
        """
        H = self.point_embed(xyz, finger_ids, joint_type_ids)      # (B, N, d_model)
        edge_struct = self._edge_struct(edge_type)                 # (E, d_struct)
        attn_bias = self.bias_builder(
            xyz, edge_index, edge_struct, edge_rest_lengths
        )                                                           # (B,1,N,N)

        for blk in self.blocks:
            H = blk(H, attn_bias)

        tokens = self.out(H)                                       # (B, N, out_dim)
        return tokens


# -------------------------
# Minimal quick self-test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 20
    E = 40
    xyz = torch.randn(B, N, 3)
    finger_ids = torch.randint(0, 5, (B, N))
    joint_type_ids = torch.randint(0, 5, (B, N))
    edge_index = torch.randint(0, N, (2, E))
    edge_type = torch.randint(0, 3, (E,))
    edge_rest_lengths = torch.rand(E) * 0.1 + 0.03  # some positive lengths

    # 老的 global token encoder（兼容）
    enc_a_global = HandEncoderEGNNLiteGlobal(
        num_fingers=5,
        num_joint_types=5,
        num_edge_types=3,
    )
    enc_b_global = HandEncoderTransformerBiasGlobal(
        num_fingers=5,
        num_joint_types=5,
        num_edge_types=3,
    )

    # 新的 per-point token encoder（你在 DiT 里主要用这个）
    enc_a_tokens = HandPointTokenEncoderEGNNLite(
        num_fingers=5,
        num_joint_types=5,
        num_edge_types=3,
        out_dim=512,
    )
    enc_b_tokens = HandPointTokenEncoderTransformerBias(
        num_fingers=5,
        num_joint_types=5,
        num_edge_types=3,
        out_dim=512,
    )

    z_a_global = enc_a_global(
        xyz, finger_ids, joint_type_ids, edge_index, edge_type, edge_rest_lengths
    )
    z_b_global = enc_b_global(
        xyz, finger_ids, joint_type_ids, edge_index, edge_type, edge_rest_lengths
    )

    z_a_tokens = enc_a_tokens(
        xyz, finger_ids, joint_type_ids, edge_index, edge_type, edge_rest_lengths
    )
    z_b_tokens = enc_b_tokens(
        xyz, finger_ids, joint_type_ids, edge_index, edge_type, edge_rest_lengths
    )

    print("Global A token:", z_a_global.shape)      # (B,512)
    print("Global B token:", z_b_global.shape)      # (B,512)
    print("Per-point A tokens:", z_a_tokens.shape)  # (B,N,512)
    print("Per-point B tokens:", z_b_tokens.shape)  # (B,N,512)
