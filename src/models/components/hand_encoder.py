"""Hand encoder modules for mapping keypoints to token embeddings.

Provides multiple architectures with strong inductive biases for dexterous
hand modeling, including EGNN-lite and Transformer-based variants.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import FourierPositionalEmbedding

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
    ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1 + 1 + d_struct, d_edge),
            nn.SiLU(),
            nn.Linear(d_edge, d_edge),
            nn.SiLU(),
        )
        self.gate_mlp = nn.Sequential(
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
        i, j = edge_index
        Hi = H[:, i, :]
        Hj = H[:, j, :]
        dist, dist2, delta = _pairwise_edge_geom(xyz, edge_index, edge_rest_lengths)
        struct = edge_struct.unsqueeze(0).expand(B, E, -1)
        e_in = torch.cat([Hi, Hj, dist2, delta, struct], dim=-1)
        e_msg = self.edge_mlp(e_in)
        g = self.gate_mlp(torch.cat([dist2, delta, struct], dim=-1))
        e_msg = e_msg * g
        agg = torch.zeros(B, N, e_msg.size(-1), device=H.device, dtype=H.dtype)
        idx = i.view(1, E, 1).expand(B, E, e_msg.size(-1))
        agg.scatter_add_(dim=1, index=idx, src=e_msg)
        H_new = self.ln(H + self.node_mlp(torch.cat([H, agg], dim=-1)))
        return H_new


class HandEncoderEGNNLiteGlobal(nn.Module):
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
        z = self.pma(H)
        return self.out(z)

class HandPointTokenEncoderEGNNLite(nn.Module):
    """Per-point EGNN-lite encoder producing tokens for downstream DiT blocks."""
    def __init__(
        self,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,            # hidden width inside EGNN
        n_layers: int = 3,
        d_edge: int = 64,
        d_struct: int = 8,
        num_frequencies: int = 10,
        out_dim: int = 512,           # token dimension per point
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

        # Final projection from d_model to out_dim per point
        self.out = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, out_dim),
            nn.LayerNorm(out_dim),
        )

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
        H = self.point_embed(xyz, finger_ids, joint_type_ids)
        edge_struct = self._edge_struct(edge_type)
        for layer in self.layers:
            H = layer(H, xyz, edge_index, edge_struct, edge_rest_lengths)
        tokens = self.out(H)
        return tokens

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
        struct = edge_struct.unsqueeze(0).expand(B, E, -1)
        edge_feat = torch.cat([dist2, delta, struct], dim=-1)
        edge_bias = self.edge_to_bias(edge_feat).squeeze(-1)
        bias = torch.ones((B, N, N), device=xyz.device, dtype=xyz.dtype) * self.non_edge_bias
        bias[:, i, j] = edge_bias
        bias[:, j, i] = edge_bias
        return bias.unsqueeze(1)


class HandEncoderTransformerBiasGlobal(nn.Module):
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

        z = self.pma(H)
        return self.out(z)

class HandPointTokenEncoderTransformerBias(nn.Module):
    """Transformer encoder with structural bias that outputs per-point tokens."""
    def __init__(
        self,
        num_fingers: int,
        num_joint_types: int,
        num_edge_types: int,
        d_model: int = 128,          # hidden width inside Transformer
        n_layers: int = 3,
        n_heads: int = 4,
        d_struct: int = 8,
        num_frequencies: int = 10,
        out_dim: int = 512,          # token dimension per point
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
        H = self.point_embed(xyz, finger_ids, joint_type_ids)
        edge_struct = self._edge_struct(edge_type)
        attn_bias = self.bias_builder(xyz, edge_index, edge_struct, edge_rest_lengths)
        for blk in self.blocks:
            H = blk(H, attn_bias)
        tokens = self.out(H)
        return tokens

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
