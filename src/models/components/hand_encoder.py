"""Hand encoder modules for mapping keypoints to token embeddings.

Provides multiple architectures with strong inductive biases for dexterous
hand modeling, including EGNN-lite and Transformer-based variants.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import inspect

from .embeddings import FourierPositionalEmbedding


logger = logging.getLogger(__name__)


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions for scalar inputs.

    Maps distances to a higher-dimensional feature space using fixed or
    learnable Gaussian kernels. This is useful for giving MLPs a richer
    representation of pairwise distances.
    """

    def __init__(
        self,
        num_kernels: int = 16,
        r_min: float = 0.0,
        r_max: float = 0.2,
        learnable: bool = False,
    ):
        super().__init__()
        if num_kernels < 1:
            raise ValueError("num_kernels must be >= 1")

        centers = torch.linspace(r_min, r_max, num_kernels)
        if num_kernels > 1:
            step = (r_max - r_min) / (num_kernels - 1)
        else:
            # Degenerate case: single kernel sits at midpoint, pick a reasonable span
            step = max(r_max - r_min, 1e-3)
        gammas = 1.0 / (2 * (step ** 2 + 1e-9))
        if learnable:
            self.centers = nn.Parameter(centers)
            self.gammas = nn.Parameter(torch.full_like(centers, gammas))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("gammas", torch.full_like(centers, gammas))

    @property
    def output_dim(self) -> int:
        return int(self.centers.numel())

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian RBF to input distances.

        Args:
            r: Tensor of distances with arbitrary leading shape and a final
               singleton or scalar dimension, e.g. (B, E, 1) or (B, E).

        Returns:
            RBF features of shape (..., num_kernels).
        """
        if r.dim() > 0 and r.size(-1) == 1:
            r = r[..., 0]
        r = r.to(self.centers.dtype)
        diff = r.unsqueeze(-1) - self.centers
        return torch.exp(-self.gammas * diff.pow(2))


class HandPointEmbedding(nn.Module):
    """Embeds hand keypoint features combining spatial and structural information.
    
    Combines Fourier positional encoding of xyz coordinates with learned embeddings
    for finger and joint type identifiers.
    
    Args:
        num_fingers: Number of fingers (vocabulary size for finger_id)
        num_joint_types: Number of joint types (vocabulary size for joint_type_id)
        d_model: Hidden feature dimension
        finger_dim: Finger embedding dimension
        joint_dim: Joint type embedding dimension
        num_frequencies: Number of Fourier frequencies
        dropout: Dropout probability
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
        """Compute point-wise embeddings.
        
        Args:
            xyz: Keypoint coordinates of shape (B, N, 3)
            finger_ids: Finger indices of shape (B, N)
            joint_type_ids: Joint type indices of shape (B, N)

        Returns:
            Point embeddings of shape (B, N, d_model)
        """
        f_xyz = self.fourier(xyz)
        f_finger = self.finger_emb(finger_ids)
        f_joint = self.joint_emb(joint_type_ids)
        features = torch.cat([f_xyz, f_finger, f_joint], dim=-1)
        return self.proj(features)

class PooledMultiheadAttention(nn.Module):
    """Set-Transformer style pooling with multihead attention (k=1 seed).
    
    Aggregates N point-level tokens into a single global token.
    Kept for backward compatibility with older models.
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
        """Pool point tokens into a single global token.
        
        Args:
            H: Point tokens of shape (B, N, d)
            
        Returns:
            Global token of shape (B, d)
        """
        B = H.size(0)
        Q = self.seed.expand(B, -1, -1)
        Z, _ = self.attn(Q, H, H)
        return self.ln(Z.squeeze(1))


def _pairwise_edge_geom(
    xyz: torch.Tensor,
    edge_index: torch.Tensor,
    edge_rest_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch edge geometry helper returning distance, squared distance, and deviation."""
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
    """Embedding table for discrete edge types (skeleton, cross-finger, etc.)."""
    def __init__(self, num_edge_types: int, d_struct: int = 8):
        super().__init__()
        self.emb = nn.Embedding(num_edge_types, d_struct)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        # edge_type: (E,)
        return self.emb(edge_type)  # (E, d_struct)


class EGNNLiteLayer(nn.Module):
    """Feature-only EGNN-style layer with geometric gating on messages."""
    def __init__(
        self,
        d_model: int,
        d_edge: int = 64,
        d_struct: int = 8,
        dropout: float = 0.1,
        use_rbf: bool = False,
        rbf_num_kernels: int = 0,
        rbf_r_min: float = 0.0,
        rbf_r_max: float = 0.2,
        rbf_learnable: bool = False,
    ):
        super().__init__()
        self.use_rbf = use_rbf and rbf_num_kernels > 0
        geom_feat_dim = 1 + 1  # dist2, delta
        gate_geom_dim = 1 + 1
        if self.use_rbf:
            self.rbf = GaussianRBF(
                num_kernels=rbf_num_kernels,
                r_min=rbf_r_min,
                r_max=rbf_r_max,
                learnable=rbf_learnable,
            )
            geom_feat_dim += self.rbf.output_dim
            gate_geom_dim += self.rbf.output_dim
        else:
            self.rbf = None

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model + geom_feat_dim + d_struct, d_edge),
            nn.SiLU(),
            nn.Linear(d_edge, d_edge),
            nn.SiLU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_geom_dim + d_struct, 32),
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
        edge_index: torch.Tensor,     # (2, E)
        geom_feats: Tuple[torch.Tensor, torch.Tensor],  # (dist2, delta)
        edge_struct: torch.Tensor,    # (B, E, d_struct)
        rbf_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = H.shape
        E = edge_index.size(1)
        i, j = edge_index
        Hi = H[:, i, :]
        Hj = H[:, j, :]
        dist2, delta = geom_feats
        geom_cat_parts = [dist2, delta]
        if self.use_rbf:
            if rbf_feat is None:
                raise ValueError("rbf_feat must be provided when use_rbf=True")
            geom_cat_parts.append(rbf_feat)
        geom_cat = torch.cat(geom_cat_parts, dim=-1)
        e_in = torch.cat([Hi, Hj, geom_cat, edge_struct], dim=-1)
        e_msg = self.edge_mlp(e_in)
        g = self.gate_mlp(torch.cat([geom_cat, edge_struct], dim=-1))
        e_msg = e_msg * g
        agg = torch.zeros(B, N, e_msg.size(-1), device=H.device, dtype=H.dtype)
        idx = i.view(1, E, 1).expand(B, E, e_msg.size(-1))
        agg.scatter_add_(dim=1, index=idx, src=e_msg)
        H_new = self.ln(H + self.node_mlp(torch.cat([H, agg], dim=-1)))
        return H_new


class HandEncoderEGNNLiteBase(nn.Module):
    """Shared wiring for EGNN-lite encoders with configurable heads."""

    def __init__(
        self,
        *,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,
        n_layers: int = 3,
        d_edge: int = 64,
        d_struct: int = 8,
        num_frequencies: int = 10,
        out_dim: int = 512,
        dropout: float = 0.1,
        use_rbf: bool = False,
        rbf_num_kernels: int = 0,
        rbf_r_min: float = 0.0,
        rbf_r_max: float = 0.2,
        rbf_learnable: bool = False,
        global_pool: bool = False,
        n_heads_pma: int = 4,
    ):
        super().__init__()
        self.global_pool = global_pool
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
                    use_rbf=use_rbf,
                    rbf_num_kernels=rbf_num_kernels,
                    rbf_r_min=rbf_r_min,
                    rbf_r_max=rbf_r_max,
                    rbf_learnable=rbf_learnable,
                )
                for _ in range(n_layers)
            ]
        )

        if self.global_pool:
            self.pma = PooledMultiheadAttention(
                d_model=d_model,
                n_heads=n_heads_pma,
                dropout=dropout,
            )
            self.out = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.SiLU(),
                nn.Linear(256, out_dim),
            )
        else:
            self.pma = None
            self.out = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, out_dim),
                nn.LayerNorm(out_dim),
            )

    def _edge_struct(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.edge_struct_emb(edge_type)

    def forward(
        self,
        xyz: torch.Tensor,
        finger_ids: torch.Tensor,
        joint_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_rest_lengths: torch.Tensor,
    ) -> torch.Tensor:
        H = self.point_embed(xyz, finger_ids, joint_type_ids)
        B = xyz.size(0)
        edge_struct = self._edge_struct(edge_type).to(H.dtype)
        edge_struct = edge_struct.unsqueeze(0).expand(B, -1, -1)
        dist, dist2, delta = _pairwise_edge_geom(xyz, edge_index, edge_rest_lengths)
        dist2 = dist2.to(H.dtype)
        delta = delta.to(H.dtype)
        geom_feats = (dist2, delta)
        for layer in self.layers:
            rbf_feat = None
            if layer.use_rbf and layer.rbf is not None:
                rbf_feat = layer.rbf(dist.to(H.dtype)).to(H.dtype)
            H = layer(H, edge_index, geom_feats, edge_struct, rbf_feat)
        if self.global_pool:
            H = self.pma(H)
            return self.out(H)
        return self.out(H)


class HandEncoderEGNNLiteGlobal(HandEncoderEGNNLiteBase):
    """Legacy EGNN-lite encoder pooling all points into a single global token."""

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
        use_rbf: bool = False,
        rbf_num_kernels: int = 0,
        rbf_r_min: float = 0.0,
        rbf_r_max: float = 0.2,
        rbf_learnable: bool = False,
    ):
        super().__init__(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            num_edge_types=num_edge_types,
            d_model=d_model,
            n_layers=n_layers,
            d_edge=d_edge,
            d_struct=d_struct,
            num_frequencies=num_frequencies,
            out_dim=out_dim,
            dropout=dropout,
            use_rbf=use_rbf,
            rbf_num_kernels=rbf_num_kernels,
            rbf_r_min=rbf_r_min,
            rbf_r_max=rbf_r_max,
            rbf_learnable=rbf_learnable,
            global_pool=True,
            n_heads_pma=n_heads_pma,
        )


class HandPointTokenEncoderEGNNLite(HandEncoderEGNNLiteBase):
    """Per-point EGNN-lite encoder producing tokens for downstream DiT blocks."""

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
        out_dim: int = 512,
        dropout: float = 0.1,
        use_rbf: bool = False,
        rbf_num_kernels: int = 0,
        rbf_r_min: float = 0.0,
        rbf_r_max: float = 0.2,
        rbf_learnable: bool = False,
    ):
        super().__init__(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            num_edge_types=num_edge_types,
            d_model=d_model,
            n_layers=n_layers,
            d_edge=d_edge,
            d_struct=d_struct,
            num_frequencies=num_frequencies,
            out_dim=out_dim,
            dropout=dropout,
            use_rbf=use_rbf,
            rbf_num_kernels=rbf_num_kernels,
            rbf_r_min=rbf_r_min,
            rbf_r_max=rbf_r_max,
            rbf_learnable=rbf_learnable,
            global_pool=False,
        )

# Transformer with structural attention bias
class BiasedMHSA(nn.Module):
    """Multi-head self-attention with an additive bias tensor."""
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
            attn_bias: (B, 1, N, N) or None, added to attention logits

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
    """Pre-LN Transformer encoder layer with BiasedMHSA."""
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

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class EdgeBiasBuilder(nn.Module):
    """Project edge geometry into an attention bias matrix with learnable fallback."""
    def __init__(self, d_struct: int = 8):
        super().__init__()
        self.edge_to_bias = nn.Sequential(
            nn.Linear(1 + 1 + d_struct, 32),  # [dist2, delta, struct_emb]
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Learnable default bias for non-edge entries
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
        struct = edge_struct.to(dist2.dtype).unsqueeze(0).expand(B, E, -1)
        edge_feat = torch.cat([dist2, delta, struct], dim=-1)
        edge_bias = self.edge_to_bias(edge_feat).squeeze(-1)

        # Ensure bias and edge_bias share the same dtype under mixed precision.
        bias_dtype = edge_bias.dtype
        non_edge = self.non_edge_bias.to(device=xyz.device, dtype=bias_dtype)
        bias = non_edge.expand(B, N, N).clone()
        bias[:, i, j] = edge_bias
        bias[:, j, i] = edge_bias
        return bias.unsqueeze(1)


class HandEncoderTransformerBiasBase(nn.Module):
    """Shared Transformer encoder body supporting pooled or per-point heads."""

    def __init__(
        self,
        *,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_struct: int = 8,
        num_frequencies: int = 10,
        out_dim: int = 512,
        dropout: float = 0.1,
        ffn_ratio: int = 2,
        global_pool: bool = False,
        n_heads_pma: int = 4,
    ):
        super().__init__()
        self.global_pool = global_pool
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

        if self.global_pool:
            self.pma = PooledMultiheadAttention(
                d_model=d_model,
                n_heads=n_heads_pma,
                dropout=dropout,
            )
            self.out = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.SiLU(),
                nn.Linear(256, out_dim),
            )
        else:
            self.pma = None
            self.out = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, out_dim),
                nn.LayerNorm(out_dim),
            )

    def _edge_struct(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.edge_struct_emb(edge_type)

    def forward(
        self,
        xyz: torch.Tensor,
        finger_ids: torch.Tensor,
        joint_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_rest_lengths: torch.Tensor,
    ) -> torch.Tensor:
        H = self.point_embed(xyz, finger_ids, joint_type_ids)
        edge_struct = self._edge_struct(edge_type).to(H.dtype)
        attn_bias = self.bias_builder(xyz, edge_index, edge_struct, edge_rest_lengths)
        for blk in self.blocks:
            H = blk(H, attn_bias)
        if self.global_pool:
            H = self.pma(H)
            return self.out(H)
        return self.out(H)


class HandEncoderTransformerBiasGlobal(HandEncoderTransformerBiasBase):
    """Legacy Transformer encoder with structural bias pooled into one token."""

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
        super().__init__(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            num_edge_types=num_edge_types,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_struct=d_struct,
            num_frequencies=num_frequencies,
            out_dim=out_dim,
            dropout=dropout,
            ffn_ratio=ffn_ratio,
            global_pool=True,
            n_heads_pma=n_heads_pma,
        )


class HandPointTokenEncoderTransformerBias(HandEncoderTransformerBiasBase):
    """Transformer encoder with structural bias that outputs per-point tokens."""

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
        out_dim: int = 512,
        dropout: float = 0.1,
        ffn_ratio: int = 2,
    ):
        super().__init__(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            num_edge_types=num_edge_types,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_struct=d_struct,
            num_frequencies=num_frequencies,
            out_dim=out_dim,
            dropout=dropout,
            ffn_ratio=ffn_ratio,
            global_pool=False,
        )

# Factory function
def build_hand_encoder(
    cfg: Any,
    graph_consts: Dict[str, torch.Tensor],
    out_dim: int,
) -> nn.Module:
    """Build a hand encoder from a Hydra/OmegaConf config.

    Args:
        cfg:    Config node under the `hand_encoder` group.
        graph_consts: Graph constants from the datamodule (see HandFlowMatchingDiT docstring).
        out_dim: Output token dimension (should typically match cfg.model.d_model).
    """
    # Derive vocab sizes from graph constants
    finger_ids = graph_consts["finger_ids"].long()
    joint_type_ids = graph_consts["joint_type_ids"].long()
    edge_type = graph_consts["edge_type"].long()

    num_fingers = int(finger_ids.max().item()) + 1
    num_joint_types = int(joint_type_ids.max().item()) + 1
    num_edge_types = int(edge_type.max().item()) + 1

    # Support Hydra _target_ instantiation with smart argument injection
    if "_target_" in cfg:
        # Candidates for injection (derived from graph_consts and context)
        candidates = {
            "graph_consts": graph_consts,
            "out_dim": out_dim,
            "num_fingers": num_fingers,
            "num_joint_types": num_joint_types,
            "num_edge_types": num_edge_types,
        }

        valid_kwargs: Dict[str, Any] = {}

        # Get the target class to inspect its signature
        try:
            target_cls = hydra.utils.get_class(cfg["_target_"])
            sig = inspect.signature(target_cls)

            # Filter args: include only those present in the __init__ signature
            # or if the init accepts **kwargs
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

            for name, val in candidates.items():
                if accepts_kwargs or name in sig.parameters:
                    valid_kwargs[name] = val

            return hydra.utils.instantiate(cfg, **valid_kwargs)
        except Exception as e:
            target_name = None
            if isinstance(cfg, dict):
                target_name = cfg.get("_target_")
            else:
                target_name = getattr(cfg, "_target_", None)
                if target_name is None and hasattr(cfg, "get"):
                    target_name = cfg.get("_target_", None)
            target_name = target_name or "<unknown>"
            arg_list = ", ".join(sorted(valid_kwargs)) if valid_kwargs else "none"
            logger.exception(
                "Failed to instantiate hand encoder target '%s' with auto-injected "
                "arguments (%s). Falling back to direct hydra.instantiate.",
                target_name,
                arg_list,
            )
        return hydra.utils.instantiate(cfg)

    name = str(getattr(cfg, "name", "transformer_bias")).lower()

    # Utility for reading optional fields from DictConfig / dict
    def _get(key: str, default):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if isinstance(cfg, dict) and key in cfg:
            return cfg[key]
        return default

    if name in ("transformer", "transformer_bias"):
        d_model_inner = _get("d_model", out_dim // 4)
        n_layers = _get("n_layers", 3)
        n_heads = _get("n_heads", 4)
        d_struct = _get("d_struct", 8)
        num_frequencies = _get("num_frequencies", 10)
        dropout = _get("dropout", 0.1)
        ffn_ratio = _get("ffn_ratio", 2)

        return HandPointTokenEncoderTransformerBias(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            num_edge_types=num_edge_types,
            d_model=d_model_inner,
            n_layers=n_layers,
            n_heads=n_heads,
            d_struct=d_struct,
            num_frequencies=num_frequencies,
            out_dim=out_dim,
            dropout=dropout,
            ffn_ratio=ffn_ratio,
        )

    elif name in ("egnn", "egnn_lite"):
        d_model_inner = _get("d_model", out_dim // 4)
        n_layers = _get("n_layers", 3)
        d_edge = _get("d_edge", 64)
        d_struct = _get("d_struct", 8)
        num_frequencies = _get("num_frequencies", 10)
        dropout = _get("dropout", 0.1)
        use_rbf = _get("use_rbf", False)
        rbf_num_kernels = _get("rbf_num_kernels", 0)
        rbf_r_min = _get("rbf_r_min", 0.0)
        rbf_r_max = _get("rbf_r_max", 0.2)
        rbf_learnable = _get("rbf_learnable", False)

        return HandPointTokenEncoderEGNNLite(
            num_fingers=num_fingers,
            num_joint_types=num_joint_types,
            num_edge_types=num_edge_types,
            d_model=d_model_inner,
            n_layers=n_layers,
            d_edge=d_edge,
            d_struct=d_struct,
            num_frequencies=num_frequencies,
            out_dim=out_dim,
            dropout=dropout,
            use_rbf=use_rbf,
            rbf_num_kernels=rbf_num_kernels,
            rbf_r_min=rbf_r_min,
            rbf_r_max=rbf_r_max,
            rbf_learnable=rbf_learnable,
        )

    else:
        raise ValueError(f"Unknown hand_encoder.name: {name}")

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 20
    E = 40
    xyz = torch.randn(B, N, 3)
    finger_ids = torch.randint(0, 5, (B, N))
    joint_type_ids = torch.randint(0, 5, (B, N))
    edge_index = torch.randint(0, N, (2, E))
    edge_type = torch.randint(0, 3, (E,))
    edge_rest_lengths = torch.rand(E) * 0.1 + 0.03  # positive lengths

    # Legacy global-token encoders (kept for compatibility)
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

    # Modern per-point token encoders (used for DiT)
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
