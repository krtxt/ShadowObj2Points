"""Graph-aware DiT (Diffusion Transformer) for hand keypoint generation.

Implements a DiT architecture with graph-structural inductive biases for hand pose
modeling, combining self-attention on hand tokens (with graph bias) and cross-attention
with scene context tokens.

Reference: Inspired by HuggingFace Diffusers' DiT implementations.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hand_encoder import EdgeBiasBuilder

_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

class FeedForward(nn.Module):
    """Position-wise feed-forward network with flexible activation functions.
    
    Supports GEGLU, GELU, SiLU, and ReLU activations. Design follows
    diffusers.models.attention.FeedForward conventions.
    
    Args:
        dim: Input and output dimension
        dropout: Dropout probability
        activation_fn: Activation type ('geglu', 'gelu', 'silu', 'relu')
        final_dropout: Whether to apply dropout after output projection
        inner_dim: Hidden layer dimension (default: dim * 4)
        bias: Whether to use bias in linear layers
    """
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        act = activation_fn.lower()
        self.activation_fn = act
        self.final_dropout = final_dropout
        self.dropout = nn.Dropout(dropout)

        if act == "geglu":
            self.proj_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        else:
            self.proj_in = nn.Linear(dim, inner_dim, bias=bias)

        self.proj_out = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = self.activation_fn
        if act == "geglu":
            x_in, gate = self.proj_in(x).chunk(2, dim=-1)
            x_in = F.gelu(gate) * x_in
        elif act == "gelu":
            x_in = F.gelu(self.proj_in(x))
        elif act == "silu":
            x_in = F.silu(self.proj_in(x))
        elif act == "relu":
            x_in = F.relu(self.proj_in(x))
        else:
            x_in = F.gelu(self.proj_in(x))
        x_in = self.dropout(x_in)
        x_out = self.proj_out(x_in)
        if self.final_dropout:
            x_out = self.dropout(x_out)
        return x_out


class FP32LayerNorm(nn.LayerNorm):
    """Layer normalization computed in float32 for numerical stability.
    
    Casts input to float32, applies normalization, then casts back to original dtype.
    Useful for mixed-precision training stability.
    
    Args:
        normalized_shape: Shape for normalization (int or tuple)
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn affine parameters
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        output = super().forward(x.to(torch.float32))
        return output.to(orig_dtype)

class AdaLayerNormShift(nn.Module):
    """Adaptive layer normalization with timestep-conditioned shift.
    
    Applies layer normalization followed by an additive shift computed from
    timestep embeddings. Adapted from HunyuanDiT.
    
    Args:
        embedding_dim: Embedding dimension (typically matches hidden_size)
        elementwise_affine: Whether to learn affine normalization parameters
        eps: Small constant for numerical stability
    """

    def __init__(self, embedding_dim: int, elementwise_affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = FP32LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization.
        
        Args:
            x: Input features of shape (B, N, D)
            emb: Timestep embeddings of shape (B, D)

        Returns:
            Normalized and shifted features of shape (B, N, D)
        """
        shift = self.linear(self.silu(emb.to(torch.float32))).to(emb.dtype)
        return self.norm(x) + shift.unsqueeze(1)

class GraphSelfAttention(nn.Module):
    """Multi-head self-attention with additive graph-structural attention bias.
    
    Enables incorporation of graph connectivity and geometric features into
    attention weights via an additive bias term.
    
    Args:
        dim: Token dimension
        num_heads: Number of attention heads
        dropout: Attention dropout probability
        qk_norm: Whether to apply LayerNorm to Q and K per head
        eps: LayerNorm epsilon for numerical stability
        use_sdpa: Whether to use scaled_dot_product_attention (if available)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        eps: float = 1e-6,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_sdpa = bool(use_sdpa) and _SDPA_AVAILABLE
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.dropout = dropout

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute self-attention with optional graph-structural bias.
        
        Args:
            x: Input tokens of shape (B, N, D)
            attn_bias: Additive attention bias of shape (B, 1, N, N) or (B, N, N)

        Returns:
            Output tokens of shape (B, N, D)
        """
        B, N, D = x.shape
        h = self.num_heads
        dh = self.head_dim

        q = self.to_q(x).view(B, N, h, dh).transpose(1, 2)
        k = self.to_k(x).view(B, N, h, dh).transpose(1, 2)
        v = self.to_v(x).view(B, N, h, dh).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_sdpa:
            return self._sdpa_attention(q, k, v, attn_bias, B, N, D)
        return self._standard_attention(q, k, v, attn_bias, B, N, D)

    def _format_attn_bias(
        self, attn_bias: torch.Tensor, batch: int, heads: int, seq_len: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert graph bias into SDPA-compatible attn_mask.

        Keeps shape as (B, H, N, N) so it can broadcast to
        query/key of shape (B, H, N, Dh) expected by SDPA.
        """
        if attn_bias.dim() == 3:
            attn_bias = attn_bias.unsqueeze(1)  # (B,1,N,N)
        if attn_bias.size(1) == 1:
            attn_bias = attn_bias.expand(batch, heads, seq_len, seq_len)
        elif attn_bias.size(1) != heads:
            raise ValueError(f"attn_bias second dim must be 1 or num_heads, got {attn_bias.size(1)}")
        return attn_bias.to(dtype=dtype)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        batch: int,
        seq_len: int,
        dim: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        attn_mask = None
        if attn_bias is not None:
            attn_mask = self._format_attn_bias(attn_bias, batch, self.num_heads, seq_len, q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.to_out(out)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        batch: int,
        seq_len: int,
        dim: int,
    ) -> torch.Tensor:
        dh = self.head_dim
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dh**0.5)
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)
            scores = scores + attn_bias
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.to_out(out)

class CrossAttention(nn.Module):
    """Standard cross-attention: Q from target, K/V from context.
    
    Enables hand tokens to attend to scene context tokens.
    
    Args:
        dim: Target (hand token) dimension
        context_dim: Context (scene token) dimension (default: same as dim)
        num_heads: Number of attention heads
        dropout: Attention dropout probability
        qk_norm: Whether to apply LayerNorm to Q and K per head
        eps: LayerNorm epsilon
        use_sdpa: Whether to use scaled_dot_product_attention (if available)
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        qk_norm: bool = False,
        eps: float = 1e-6,
        use_sdpa: bool = True,
    ):
        super().__init__()
        context_dim = context_dim or dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_sdpa = bool(use_sdpa) and _SDPA_AVAILABLE
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(context_dim, dim, bias=True)
        self.to_v = nn.Linear(context_dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.dropout = dropout

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-attention between target and context tokens.
        
        Args:
            x: Target tokens of shape (B, N, D)
            context: Context tokens of shape (B, K, C)
            
        Returns:
            Output tokens of shape (B, N, D)
        """
        B, N, D = x.shape
        h = self.num_heads
        dh = self.head_dim

        q = self.to_q(x).view(B, N, h, dh).transpose(1, 2)
        k = self.to_k(context).view(B, -1, h, dh).transpose(1, 2)
        v = self.to_v(context).view(B, -1, h, dh).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_sdpa:
            return self._sdpa_attention(q, k, v, B, N, D)
        return self._standard_attention(q, k, v, dh, B, N, D)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: int,
        seq_len: int,
        dim: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.to_out(out)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_dim: int,
        batch: int,
        seq_len: int,
        dim: int,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.to_out(out)

def _compute_pairwise_edge_geometry(
    xyz: torch.Tensor,              # (B, N, 3)
    edge_index: torch.Tensor,       # (2, E)
    edge_rest_lengths: torch.Tensor # (E,)
):
    """Compute per-edge distances relative to their rest length."""
    i, j = edge_index   # (E,)
    diff = xyz[:, i, :] - xyz[:, j, :]           # (B,E,3)
    dist2 = (diff ** 2).sum(-1, keepdim=True)   # (B,E,1)
    dist = torch.sqrt(dist2 + 1e-9)             # (B,E,1)
    rest = edge_rest_lengths.view(1, -1, 1).to(xyz)  # (1,E,1)
    delta = (dist - rest) / (rest + 1e-9)       # (B,E,1)
    return dist, dist2, delta


class GraphAttentionBias(nn.Module):
    """Map graph structure and current coordinates to an attention bias tensor.

    Each edge receives geometric and structural features that are scored by a
    small MLP. Non-edges fall back to a learnable default bias. The resulting
    matrix is reshaped to (B, 1, N, N) so it can be injected into
    ``GraphSelfAttention``.
    """

    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        edge_index: torch.Tensor,        # (2, E)
        edge_type: torch.Tensor,         # (E,)
        edge_rest_lengths: torch.Tensor, # (E,)
        d_struct: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.register_buffer("edge_type_idx", edge_type.long(), persistent=False)
        self.register_buffer("edge_rest_lengths", edge_rest_lengths.float(), persistent=False)

        self.edge_type_emb = nn.Embedding(num_edge_types, d_struct)
        # Shared edge-bias builder (geometry + non-edge fallback) reused from
        # hand_encoder to keep behavior and learnable parameters consistent.
        self.edge_bias_builder = EdgeBiasBuilder(d_struct=d_struct)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute attention bias from current keypoint positions.
        
        Args:
            xyz: Keypoint coordinates of shape (B, N, 3)

        Returns:
            Attention bias of shape (B, 1, N, N) for GraphSelfAttention
        """
        B, N, _ = xyz.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {N}")
        edge_struct = self.edge_type_emb(self.edge_type_idx)  # (E, d_struct)
        bias = self.edge_bias_builder(
            xyz=xyz,
            edge_index=self.edge_index,
            edge_struct=edge_struct,
            edge_rest_lengths=self.edge_rest_lengths,
        )
        return bias

# DiT Block: hand + scene + graph
class HandSceneGraphDiTBlock(nn.Module):
    """Graph-aware DiT block with self-attn, cross-attn, and FFN sublayers."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        norm_eps: float = 1e-6,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        qk_norm: bool = False,
        skip: bool = False,
    ):
        super().__init__()
        self.dim = dim

        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=True, eps=norm_eps)
        self.self_attn = GraphSelfAttention(
            dim=dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            eps=norm_eps,
        )

        self.norm2 = FP32LayerNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=cross_attention_dim or dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            eps=norm_eps,
        )

        self.norm3 = FP32LayerNorm(dim, eps=norm_eps, elementwise_affine=True)
        inner_dim = ff_inner_dim or dim * 4
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=inner_dim,
            bias=ff_bias,
        )

        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, eps=norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_norm = None
            self.skip_linear = None

    def forward(
        self,
        hand_tokens: torch.Tensor,            # (B, N, D)
        scene_tokens: Optional[torch.Tensor], # (B, K, D_scene); may be None
        temb: torch.Tensor,                   # (B, D) flow embedding already projected to D
        graph_attn_bias: Optional[torch.Tensor] = None,  # (B,1,N,N) or (B,N,N)
        skip: Optional[torch.Tensor] = None,  # (B,N,D) optional long skip
    ) -> torch.Tensor:
        """
        Returns:
            hand_tokens_out: (B, N, D)
        """
        if self.skip_linear is not None and skip is not None:
            cat = torch.cat([hand_tokens, skip], dim=-1)
            cat = self.skip_norm(cat)
            hand_tokens = self.skip_linear(cat)
        h = self.norm1(hand_tokens, temb)
        h = self.self_attn(h, attn_bias=graph_attn_bias)
        hand_tokens = hand_tokens + h
        if scene_tokens is not None:
            h = self.cross_attn(self.norm2(hand_tokens), scene_tokens)
            hand_tokens = hand_tokens + h
        h = self.ff(self.norm3(hand_tokens))
        hand_tokens = hand_tokens + h
        return hand_tokens

class HandSceneGraphDiT(nn.Module):
    """Stacked DiT blocks that fuse hand tokens, scene context, and graph bias."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        cross_attention_dim: Optional[int] = None,
        graph_bias: Optional[GraphAttentionBias] = None,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        qk_norm: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.graph_bias = graph_bias

        self.blocks = nn.ModuleList(
            [
                HandSceneGraphDiTBlock(
                    dim=dim,
                    num_attention_heads=num_heads,
                    cross_attention_dim=cross_attention_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    ff_inner_dim=ff_inner_dim,
                    ff_bias=ff_bias,
                    qk_norm=qk_norm,
                    skip=skip,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        hand_tokens: torch.Tensor,             # (B, N, D)
        scene_tokens: Optional[torch.Tensor],  # (B, K, D_s) or None
        temb: torch.Tensor,                    # (B, D)
        xyz: Optional[torch.Tensor] = None,    # (B, N, 3) for graph bias; skip if None
    ) -> torch.Tensor:
        """Return updated hand tokens after passing through all DiT blocks."""
        B, N, D = hand_tokens.shape
        assert temb.shape == (B, D), f"temb should be (B,{D}), got {temb.shape}"
        if self.graph_bias is not None and xyz is not None:
            attn_bias = self.graph_bias(xyz)
        else:
            attn_bias = None
        h = hand_tokens
        for block in self.blocks:
            h = block(
                hand_tokens=h,
                scene_tokens=scene_tokens,
                temb=temb,
                graph_attn_bias=attn_bias,
                skip=None,
            )
        return h

# Factory function
def build_dit(
    cfg: Any,
    graph_consts: Dict[str, torch.Tensor],
    dim: int,
) -> HandSceneGraphDiT:
    """Build a HandSceneGraphDiT from a Hydra/OmegaConf config.

    Args:
        cfg:          Config node under the `dit` group.
        graph_consts: Graph constants from the datamodule.
        dim:          Token dimension D (should match hand encoder out_dim).
    """
    finger_ids = graph_consts["finger_ids"].long()
    edge_index = graph_consts["edge_index"].long()
    edge_type = graph_consts["edge_type"].long()
    edge_rest_lengths = graph_consts["edge_rest_lengths"].float()

    num_nodes = int(finger_ids.shape[0])
    num_edge_types = int(edge_type.max().item()) + 1

    # Utility to read optional fields from DictConfig / dict
    def _get(key: str, default):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if isinstance(cfg, dict) and key in cfg:
            return cfg[key]
        return default

    d_struct = _get("d_struct", 8)
    depth = _get("depth", 6)
    num_heads = _get("num_heads", 8)
    cross_attention_dim = _get("cross_attention_dim", None)
    ff_inner_dim = _get("ff_inner_dim", None)
    ff_bias = _get("ff_bias", True)
    skip = _get("skip", False)
    dropout = _get("dropout", 0.0)
    activation_fn = _get("activation_fn", "geglu")
    qk_norm = _get("qk_norm", True)

    def _subcfg_value(node: Any, key: str, default):
        if node is None:
            return default
        if hasattr(node, key):
            return getattr(node, key)
        if isinstance(node, dict) and key in node:
            return node[key]
        return default

    graph_bias_cfg = _get("graph_bias", None)
    graph_bias_enabled = _subcfg_value(graph_bias_cfg, "enabled", True)
    d_struct = _subcfg_value(graph_bias_cfg, "d_struct", d_struct)

    graph_bias_module: Optional[GraphAttentionBias]
    if graph_bias_enabled:
        graph_bias_module = GraphAttentionBias(
            num_nodes=num_nodes,
            num_edge_types=num_edge_types,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_rest_lengths=edge_rest_lengths,
            d_struct=d_struct,
        )
    else:
        graph_bias_module = None

    if cross_attention_dim is None:
        cross_attention_dim = dim

    return HandSceneGraphDiT(
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        cross_attention_dim=cross_attention_dim,
        graph_bias=graph_bias_module,
        dropout=dropout,
        activation_fn=activation_fn,
        qk_norm=qk_norm,
        ff_inner_dim=ff_inner_dim,
        ff_bias=ff_bias,
        skip=skip,
    )

if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, K = 2, 23, 32
    D = 512

    hand_tokens = torch.randn(B, N, D)
    scene_tokens = torch.randn(B, K, D)
    temb = torch.randn(B, D)
    xyz = torch.randn(B, N, 3)

    E = 40
    edge_index = torch.randint(0, N, (2, E))
    edge_type = torch.randint(0, 3, (E,))
    edge_rest_lengths = torch.rand(E) * 0.1 + 0.03

    graph_bias = GraphAttentionBias(
        num_nodes=N,
        num_edge_types=3,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_rest_lengths=edge_rest_lengths,
        d_struct=8,
    )

    model = HandSceneGraphDiT(
        dim=D,
        depth=4,
        num_heads=8,
        cross_attention_dim=D,
        graph_bias=graph_bias,
        dropout=0.0,
        activation_fn="geglu",
        qk_norm=True,
    )

    out = model(hand_tokens, scene_tokens, temb, xyz=xyz)
    print("hand_tokens in:", hand_tokens.shape)  # (B,N,D)
    print("hand_tokens out:", out.shape)         # (B,N,D)
