# graph_dit.py
# ------------------------------------------------------------
# Graph-aware DiT for hand tokens + scene tokens (point cloud)
# Self-attn: operates on hand tokens with graph attention bias
# Cross-attn: injects scene tokens (no mask)
#
# 参考结构：diffusers 中的 HunyuanDiTBlock（self + cross + FFN + AdaLayerNormShift）
# ------------------------------------------------------------
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 你需要安装 diffusers，下面路径如果不对，可以根据自己版本调整：
# 例如：from diffusers.models.attention import FeedForward
from diffusers.attention import FeedForward


# ------------------------------
# 基础：FP32 LayerNorm + AdaLN
# ------------------------------
class FP32LayerNorm(nn.LayerNorm):
    """
    简单版 FP32 LayerNorm：内部用 float32 计算，再 cast 回输入 dtype。

    Args:
        normalized_shape: int or tuple
        eps: float
        elementwise_affine: bool
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        y = super().forward(x.to(torch.float32))
        return y.to(orig_dtype)


class AdaLayerNormShift(nn.Module):
    r"""
    从 HunyuanDiT 改写的 AdaLayerNormShift:
    Norm(x) + shift(temb)

    Parameters:
        embedding_dim (`int`): 嵌入维度（通常等于 hidden_size）
    """

    def __init__(self, embedding_dim: int, elementwise_affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = FP32LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, D)
            emb: (B, D)

        Returns:
            out: (B, N, D)
        """
        shift = self.linear(self.silu(emb.to(torch.float32))).to(emb.dtype)  # (B,D)
        x = self.norm(x) + shift.unsqueeze(1)  # broadcast 到 N 维
        return x


# ------------------------------
# 自定义注意力：支持 graph bias
# ------------------------------
class GraphSelfAttention(nn.Module):
    """
    多头自注意力（MHSA），支持加性 attention bias，用于注入图结构信息。

    Args:
        dim:      token 维度 D
        num_heads:注意力头数
        dropout:  attention drop 概率
        qk_norm:  是否对每个 head 的 Q/K 做 LayerNorm
        eps:      QK LayerNorm 的 eps
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

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
        """
        Args:
            x: (B, N, D)
            attn_bias: (B, 1, N, N) 或 (B, N, N)，会加到注意力 logits 上

        Returns:
            y: (B, N, D)
        """
        B, N, D = x.shape
        h = self.num_heads
        dh = self.head_dim

        # (B,N,D) -> (B,h,N,dh)
        q = self.to_q(x).view(B, N, h, dh).transpose(1, 2)  # (B,h,N,dh)
        k = self.to_k(x).view(B, N, h, dh).transpose(1, 2)
        v = self.to_v(x).view(B, N, h, dh).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 注意力 logits: (B,h,N,N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dh ** 0.5)

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # (B,1,N,N)
            # broadcast 到 (B,h,N,N)
            scores = scores + attn_bias

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)                    # (B,h,N,dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B,N,D)
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    """
    标准 Cross-Attention：Q 来自 hand_tokens，K/V 来自 scene_tokens。

    Args:
        dim:        hand token 维度
        context_dim:scene token 维度（默认为 dim）
        num_heads:  注意力头数
        dropout:    attention dropout
        qk_norm:    是否对每个 head 的 Q/K 做 LayerNorm
        eps:        QK LayerNorm eps
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        context_dim = context_dim or dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

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
        x: torch.Tensor,              # (B, N, D)  hand tokens
        context: torch.Tensor,        # (B, K, C)  scene tokens
    ) -> torch.Tensor:
        B, N, D = x.shape
        h = self.num_heads
        dh = self.head_dim

        # (B,N,D) -> (B,h,N,dh)
        q = self.to_q(x).view(B, N, h, dh).transpose(1, 2)  # (B,h,N,dh)

        # (B,K,C) -> (B,h,K,dh)
        k = self.to_k(context).view(B, -1, h, dh).transpose(1, 2)
        v = self.to_v(context).view(B, -1, h, dh).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (dh ** 0.5)  # (B,h,N,K)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)                    # (B,h,N,dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.to_out(out)
        return out


# ------------------------------
# Graph 结构 → attention bias
# ------------------------------
def _pairwise_edge_geom(
    xyz: torch.Tensor,              # (B, N, 3)
    edge_index: torch.Tensor,       # (2, E)
    edge_rest_lengths: torch.Tensor # (E,)
):
    """
    给定当前坐标 xyz 和 rest_lengths，求：
        dist:  (B,E,1)
        dist2: (B,E,1)
        delta: (B,E,1) = (dist - rest) / rest
    """
    i, j = edge_index   # (E,)
    diff = xyz[:, i, :] - xyz[:, j, :]           # (B,E,3)
    dist2 = (diff ** 2).sum(-1, keepdim=True)   # (B,E,1)
    dist = torch.sqrt(dist2 + 1e-9)             # (B,E,1)
    rest = edge_rest_lengths.view(1, -1, 1).to(xyz)  # (1,E,1)
    delta = (dist - rest) / (rest + 1e-9)       # (B,E,1)
    return dist, dist2, delta


class GraphAttentionBias(nn.Module):
    """
    将图结构 (edge_index, edge_type, edge_rest_lengths) + 当前 xyz
    映射为注意力偏置矩阵 attn_bias: (B,1,N,N)，用于 GraphSelfAttention。

    设计思路：
      - 对每条边 (i,j) 计算特征 [dist^2, delta, struct_emb(edge_type)]
      - 用一个 MLP -> 标量 bias_ij
      - 将这些 bias 写入 NxN 矩阵，对非边用一个可学习的 default bias

    使用方式：
      - 在 DiT 前向中调用：
            attn_bias = graph_bias(xyz)  # xyz: (B,N,3)
      - 传入 Block.self-attn 的 attn_bias 参数。
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

        self.edge_mlp = nn.Sequential(
            nn.Linear(1 + 1 + d_struct, 32),  # [dist2, delta, struct]
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # non-edge bias（标量），可初始化为 0
        self.non_edge_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) 当前关键点坐标（局部坐标或世界坐标都可，但与 rest_lengths 对应即可）

        Returns:
            attn_bias: (B, 1, N, N) 加到 self-attention logits 上
        """
        B, N, _ = xyz.shape
        assert N == self.num_nodes, f"xyz has {N} nodes, but graph_bias was built for {self.num_nodes}"

        edge_index = self.edge_index   # (2,E)
        edge_type_idx = self.edge_type_idx
        edge_rest_lengths = self.edge_rest_lengths

        i, j = edge_index
        dist, dist2, delta = _pairwise_edge_geom(
            xyz, edge_index, edge_rest_lengths
        )                               # (B,E,1)...
        struct = self.edge_type_emb(edge_type_idx)   # (E,d_struct)
        struct = struct.unsqueeze(0).expand(B, -1, -1)  # (B,E,d_struct)

        edge_feat = torch.cat([dist2, delta, struct], dim=-1)  # (B,E,1+1+d_struct)
        edge_bias = self.edge_mlp(edge_feat).squeeze(-1)       # (B,E)

        # 初始化 NxN 为可学习的 non_edge_bias
        bias = torch.ones(
            (B, N, N),
            device=xyz.device,
            dtype=xyz.dtype,
        ) * self.non_edge_bias
        # i->j 和 j->i 都加边 bias
        for b in range(B):
            bias[b].index_put_((i, j), edge_bias[b])
            bias[b].index_put_((j, i), edge_bias[b])

        # (B,1,N,N)
        return bias.unsqueeze(1)


# ------------------------------
# DiT Block: hand + scene + graph
# ------------------------------
class HandSceneGraphDiTBlock(nn.Module):
    """
    Graph-aware DiT Block:

        1) 手部 token 的自注意力（GraphSelfAttention，带 graph_attn_bias）
        2) 对 scene tokens 的跨注意力（CrossAttention）
        3) FeedForward（来自 diffusers.attention.FeedForward）
        4) 所有子模块都是 pre-LN；自注意力用 AdaLayerNormShift(timestep embedding)

    Args:
        dim:               hand token 维度 D（也是 scene token 维度）
        num_attention_heads: 注意力头数
        cross_attention_dim: scene token 维度（如果和 dim 不同）
        dropout:           dropout 概率
        activation_fn:     FeedForward 激活函数
        norm_eps:          LayerNorm eps
        ff_inner_dim:      FFN 隐层维度（默认 dim * 4）
        ff_bias:           FFN 是否带 bias
        qk_norm:           self-attn 和 cross-attn 的 QK 是否做 LayerNorm
        skip:              是否启用 long skip（目前一般设 False）
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
        qk_norm: bool = True,
        skip: bool = False,
    ):
        super().__init__()
        self.dim = dim

        # 1. Self-Attn 部分：AdaLayerNormShift + GraphSelfAttention
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=True, eps=norm_eps)
        self.self_attn = GraphSelfAttention(
            dim=dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            eps=norm_eps,
        )

        # 2. Cross-Attn：标准 LayerNorm + CrossAttention
        self.norm2 = FP32LayerNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=cross_attention_dim or dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            eps=norm_eps,
        )

        # 3. FeedForward：LayerNorm + FeedForward
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

        # 4. 可选 long skip（类似 HunyuanDiT 中 mid-block 之后的 skip）
        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, eps=norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_norm = None
            self.skip_linear = None

    def forward(
        self,
        hand_tokens: torch.Tensor,            # (B, N, D)
        scene_tokens: Optional[torch.Tensor], # (B, K, D_scene)，可以为 None
        temb: torch.Tensor,                   # (B, D) 时间/flow embedding，已映射到 D 维
        graph_attn_bias: Optional[torch.Tensor] = None,  # (B,1,N,N) 或 (B,N,N)
        skip: Optional[torch.Tensor] = None,  # (B,N,D)，long skip，用不到就传 None
    ) -> torch.Tensor:
        """
        Returns:
            hand_tokens_out: (B, N, D)
        """
        # 0. Long Skip （可选）
        if self.skip_linear is not None and skip is not None:
            cat = torch.cat([hand_tokens, skip], dim=-1)     # (B,N,2D)
            cat = self.skip_norm(cat)
            hand_tokens = self.skip_linear(cat)              # (B,N,D)

        # 1. Self-Attn on hand tokens
        h = self.norm1(hand_tokens, temb)                    # (B,N,D)
        h = self.self_attn(h, attn_bias=graph_attn_bias)     # (B,N,D)
        hand_tokens = hand_tokens + h

        # 2. Cross-Attn: hand <- scene
        if scene_tokens is not None:
            h = self.cross_attn(self.norm2(hand_tokens), scene_tokens)  # (B,N,D)
            hand_tokens = hand_tokens + h

        # 3. FFN
        h = self.ff(self.norm3(hand_tokens))                 # (B,N,D)
        hand_tokens = hand_tokens + h

        return hand_tokens


# ------------------------------
# 多层堆叠的 Graph DiT
# ------------------------------
class HandSceneGraphDiT(nn.Module):
    """
    多层堆叠的 Graph-aware DiT。

    典型使用场景：
      - 输入 hand_tokens: (B, N, D)，来自 HandPointTokenEncoder（局部关键点）
      - 输入 scene_tokens: (B, K, D_s)，来自点云 backbone（PCL -> scene tokens）
      - 输入 temb: (B, D)，flow matching / diffusion timestep embedding，已映射为 D 维
      - 输入 xyz: (B, N, 3)，当前关键点坐标（和 rest_lengths 对应），用来构造 graph_attn_bias

    forward:
        hand_out = model(hand_tokens, scene_tokens, temb, xyz)

    Args:
        dim:               手部 token 维度 D（通常等于 hand encoder out_dim）
        depth:             DiT Block 层数
        num_heads:         每层自/交叉注意力头数
        cross_attention_dim: scene token 维度（不填默认 dim）
        graph_bias:        GraphAttentionBias 模块（可选）；
                           如果为 None，则不使用图结构 bias（普通 DiT）
        dropout:           dropout 概率
        activation_fn:     FeedForward 激活
        qk_norm:           是否对 Q/K 归一化
    """

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
                    ff_inner_dim=int(dim * 4),
                    ff_bias=True,
                    qk_norm=qk_norm,
                    skip=False,  # 目前不做 long skip，有需要可以在上层构造
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        hand_tokens: torch.Tensor,             # (B, N, D)
        scene_tokens: Optional[torch.Tensor],  # (B, K, D_s) 或 None
        temb: torch.Tensor,                    # (B, D)
        xyz: Optional[torch.Tensor] = None,    # (B, N, 3) 用于 graph bias；如果 None 则不使用 graph bias
    ) -> torch.Tensor:
        """
        Returns:
            hand_tokens_out: (B, N, D)  # 经过多个 DiTBlock 更新后的手部 token，可接 Flow head 输出速度等
        """
        B, N, D = hand_tokens.shape
        assert temb.shape == (B, D), f"temb should be (B,{D}), got {temb.shape}"

        if self.graph_bias is not None and xyz is not None:
            attn_bias = self.graph_bias(xyz)   # (B,1,N,N)
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


# ------------------------------
# 简单 self-test
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, K = 2, 23, 32
    D = 512

    hand_tokens = torch.randn(B, N, D)
    scene_tokens = torch.randn(B, K, D)
    temb = torch.randn(B, D)
    xyz = torch.randn(B, N, 3)

    # 假设图结构有 E 条边、3 种 edge_type
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
