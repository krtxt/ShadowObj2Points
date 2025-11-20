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
import hydra
import inspect

from .hand_encoder import EdgeBiasBuilder

_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Converges faster and is more computationally efficient than LayerNorm.
    Used in LLaMA, Stable Diffusion 3, etc.
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force float32 for stability
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        x = x.to(orig_dtype)
        
        if self.weight is not None:
            return x * self.weight
        return x


class FP32LayerNorm(nn.LayerNorm):
    """Layer normalization computed in float32 for numerical stability."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        output = super().forward(x.to(torch.float32))
        return output.to(orig_dtype)


@torch.jit.script
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply adaptive modulation: x * (1 + scale) + shift
    Using JIT script for potential fusion optimization.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    """Position-wise feed-forward network (MLP)."""
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
        if self.activation_fn == "geglu":
            x_in, gate = self.proj_in(x).chunk(2, dim=-1)
            x_in = x_in * F.gelu(gate)
        elif self.activation_fn == "gelu":
            x_in = F.gelu(self.proj_in(x))
        elif self.activation_fn == "silu":
            x_in = F.silu(self.proj_in(x))
        else:
            x_in = F.relu(self.proj_in(x))
            
        x_in = self.dropout(x_in)
        x_out = self.proj_out(x_in)
        if self.final_dropout:
            x_out = self.dropout(x_out)
        return x_out


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
        qk_norm_type: str = "layer",
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
            if qk_norm_type == "rms":
                self.q_norm = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
                self.k_norm = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
            else:
                self.q_norm = FP32LayerNorm(self.head_dim, eps=eps, elementwise_affine=True)
                self.k_norm = FP32LayerNorm(self.head_dim, eps=eps, elementwise_affine=True)
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
            
        # Ensure broadcasting capability
        if attn_bias.size(1) == 1:
            # SDPA usually handles (B, 1, N, N) broadcasting automatically,
            # but explicit expand ensures safety.
            attn_bias = attn_bias.expand(batch, heads, seq_len, seq_len)
        
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
        
        # is_causal=False for bidirectional graph attention
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
        # Q @ K^T
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
        qk_norm_type: str = "layer",
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
            if qk_norm_type == "rms":
                self.q_norm = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
                self.k_norm = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
            else:
                self.q_norm = FP32LayerNorm(self.head_dim, eps=eps, elementwise_affine=True)
                self.k_norm = FP32LayerNorm(self.head_dim, eps=eps, elementwise_affine=True)
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
            # For cross attention, attn_mask is usually None unless we need masking
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
            out = out.transpose(1, 2).contiguous().view(B, N, D)
            return self.to_out(out)
            
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dh**0.5)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
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
    """Graph-aware DiT block with self-attn, cross-attn, and FFN sublayers.
    
    Refactored to use AdaLN-Zero (scale, shift, gate) and full modulation.
    """

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
        qk_norm_type: str = "layer",
        skip: bool = False,
    ):
        super().__init__()
        self.dim = dim

        # 1. Attention & FFN Modules
        self.self_attn = GraphSelfAttention(
            dim=dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            eps=norm_eps,
        )

        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=cross_attention_dim or dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            eps=norm_eps,
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=ff_inner_dim or dim * 4,
            bias=ff_bias,
        )

        # 2. Norms (elementwise_affine=False because AdaLN handles parameters)
        # Using RMSNorm is preferred if qk_norm_type is RMS, but standard DiT often uses LayerNorm for the block structure
        # even if QK uses RMS. To be safe/standard, LayerNorm is the default for AdaLN blocks.
        self.norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.norm3 = FP32LayerNorm(dim, elementwise_affine=False, eps=norm_eps)

        # 3. AdaLN-Zero Modulation Projection
        # 9 params: (shift, scale, gate) for SelfAttn, CrossAttn, FFN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True)
        )
        
        # Zero-init: Start as identity block
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # Optional long skip connection
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
        # Optional long skip injection at the beginning
        if self.skip_linear is not None and skip is not None:
            cat = torch.cat([hand_tokens, skip], dim=-1)
            cat = self.skip_norm(cat)
            hand_tokens = self.skip_linear(cat)

        # 1. Generate Modulation Parameters
        # chunk(9, dim=1) -> 9 tensors of shape (B, D)
        shift_msa, scale_msa, gate_msa, \
        shift_ca, scale_ca, gate_ca, \
        shift_ff, scale_ff, gate_ff = self.adaLN_modulation(temb).chunk(9, dim=1)

        # 2. Self-Attention Block
        x = hand_tokens
        h = self.norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        h = self.self_attn(h, attn_bias=graph_attn_bias)
        x = x + gate_msa.unsqueeze(1) * h

        # 3. Cross-Attention Block
        if scene_tokens is not None:
            h = self.norm2(x)
            h = modulate(h, shift_ca, scale_ca)
            h = self.cross_attn(h, scene_tokens)
            x = x + gate_ca.unsqueeze(1) * h
        
        # 4. Feed-Forward Block
        h = self.norm3(x)
        h = modulate(h, shift_ff, scale_ff)
        h = self.ff(h)
        x = x + gate_ff.unsqueeze(1) * h

        return x


class DiTFinalLayer(nn.Module):
    """The final layer of DiT with AdaLN-Zero."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Final norm before output
        self.norm_final = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        
        # AdaLN modulation for final norm
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        
        # The standard DiT logic: The Linear projection (to velocity) is typically
        # handled by the `velocity_head` in the main model class. 
        # BUT, to ensure zero-init behavior, we can perform a projection here 
        # or ensure the external head is zero-initialized.
        # Here we implement the modulation part.
        
        # Zero-init modulation
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


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
        qk_norm_type: str = "layer", # 'layer' or 'rms'
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
        norm_eps: float = 1e-6,
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
                    qk_norm_type=qk_norm_type,
                    skip=skip,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = DiTFinalLayer(dim, eps=norm_eps)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers normally (e.g. Xavier/Kaiming)
        # AdaLN zero-init is handled in the block classes
        
        # Custom initialization logic if needed, otherwise PyTorch defaults are decent.
        # For DiT, standard Linear layers usually work fine with defaults, 
        # but ensure the zero-init blocks are respected.
        pass

    def forward(
        self,
        hand_tokens: torch.Tensor,             # (B, N, D)
        scene_tokens: Optional[torch.Tensor],  # (B, K, D_s) or None
        temb: torch.Tensor,                    # (B, D)
        xyz: Optional[torch.Tensor] = None,    # (B, N, 3) for graph bias; skip if None
    ) -> torch.Tensor:
        """Return updated hand tokens after passing through all DiT blocks."""
        B, N, D = hand_tokens.shape
        
        # Calculate Graph Bias once
        attn_bias = None
        if self.graph_bias is not None and xyz is not None:
            attn_bias = self.graph_bias(xyz)

        h = hand_tokens
        for block in self.blocks:
            h = block(
                hand_tokens=h,
                scene_tokens=scene_tokens,
                temb=temb,
                graph_attn_bias=attn_bias,
                skip=None,
            )
            
        h = self.final_layer(h, temb)
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

    # Support Hydra _target_ instantiation with smart argument injection
    if "_target_" in cfg:
        candidates = {
            "graph_consts": graph_consts,
            "dim": dim,
            "num_nodes": num_nodes,
            "num_edge_types": num_edge_types,
        }
        
        try:
            target_cls = hydra.utils.get_class(cfg["_target_"])
            sig = inspect.signature(target_cls)
            
            valid_kwargs = {}
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            
            for name, val in candidates.items():
                if accepts_kwargs or name in sig.parameters:
                    valid_kwargs[name] = val
            
            return hydra.utils.instantiate(cfg, **valid_kwargs)
        except Exception:
            pass
        return hydra.utils.instantiate(cfg)

    def _get(key: str, default):
        if hasattr(cfg, key): return getattr(cfg, key)
        if isinstance(cfg, dict) and key in cfg: return cfg[key]
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
    qk_norm_type = _get("qk_norm_type", "layer")

    def _subcfg_value(node: Any, key: str, default):
        if node is None: return default
        if hasattr(node, key): return getattr(node, key)
        if isinstance(node, dict) and key in node: return node[key]
        return default

    # Config parsing
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
        qk_norm_type=qk_norm_type,
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
