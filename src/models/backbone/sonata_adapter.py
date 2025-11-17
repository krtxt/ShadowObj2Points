"""
Sonata Adapter for SceneLeapUltra
将 Sonata 编码器输出适配到固定的 Scene Token 格式 (B, K, D)
支持三种压缩策略，推荐使用 Perceiver-style Cross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal
from models.backbone.sonata.model import PointTransformerV3, load as load_sonata
from models.backbone.sonata.structure import Point
from models.backbone.sonata.utils import offset2bincount
import logging
import math

logger = logging.getLogger(__name__)


@torch.jit.script
def _jit_farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    TorchScript 版本的 FPS，避免 Python 循环的 host 同步开销。
    Args:
        points: (N, 3)
        num_samples: 采样点数
    Returns:
        采样索引 (num_samples,)
    """
    N = points.size(0)
    device = points.device
    centroids = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), 1.0e10, device=device)
    farthest = torch.randint(0, N, (1,), device=device, dtype=torch.long)[0]
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest].unsqueeze(0)
        dist = torch.sum((points - centroid) * (points - centroid), dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return centroids


class SonataToTokenAdapter(nn.Module):
    """
    Sonata 编码器到固定 Token 序列的适配器
    
    输入: Sonata 输出的 Point 对象（可变长度点云特征）
    输出: (B, K, D) 固定长度的 Scene Tokens
    
    支持三种压缩策略:
    1. 'perceiver': Perceiver-style Cross-Attention（推荐）
    2. 'detr': DETR-style Learned Queries（与现有 scene_tokenizer 类似）
    3. 'pooling': 简单池化（最快，可能丢失信息）
    """
    
    def __init__(
        self,
        sonata_out_dim: int = 512,           # Sonata 编码器输出维度
        num_tokens: int = 128,                # 目标 token 数量 K
        token_dim: int = 512,                 # Token 特征维度 D
        compression_mode: Literal['perceiver', 'detr', 'pooling'] = 'perceiver',
        
        # Perceiver/DETR 共用参数
        num_cross_attn_layers: int = 2,       # Cross-Attention 层数
        num_heads: int = 8,                   # 注意力头数
        mlp_ratio: float = 4.0,               # FFN 扩展比例
        dropout: float = 0.1,
        
        # 预筛选参数（可选，用于减少计算量）
        use_prefilter: bool = True,           # 是否预筛选候选点
        num_candidates: int = 512,            # 预筛选后的候选点数 M
        prefilter_mode: Literal['fps', 'random', 'surface'] = 'fps',
        
        # 位置编码参数
        use_pos_embed: bool = True,           # 是否使用位置编码
        pos_embed_type: Literal['learnable', 'sinusoidal'] = 'learnable',
    ):
        super().__init__()
        
        self.sonata_out_dim = sonata_out_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.compression_mode = compression_mode
        self.use_prefilter = use_prefilter
        self.num_candidates = num_candidates
        self.prefilter_mode = prefilter_mode
        self.use_pos_embed = use_pos_embed
        
        # 断言与关键信息日志
        if num_heads <= 0 or token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim 必须能被 num_heads 整除，当前 token_dim={token_dim}, num_heads={num_heads}"
            )
        logger.info(
            f"Multi-head 配置: num_heads={num_heads}, head_dim={token_dim // num_heads}"
        )

        logger.info(f"初始化 SonataToTokenAdapter: mode={compression_mode}, "
                    f"tokens={num_tokens}, dim={token_dim}, "
                    f"prefilter={use_prefilter} (M={num_candidates if use_prefilter else 'N'})")
        
        # 特征投影（如果 Sonata 输出维度与 token_dim 不同）
        if sonata_out_dim != token_dim:
            self.feat_proj = nn.Linear(sonata_out_dim, token_dim)
        else:
            self.feat_proj = nn.Identity()
        
        # 预筛选模块（可选）
        if use_prefilter:
            self.prefilter = PointPrefilter(
                in_dim=token_dim,
                num_candidates=num_candidates,
                mode=prefilter_mode,
            )
        else:
            self.prefilter = None
        
        # 压缩模块（核心）
        if compression_mode == 'perceiver':
            self.compressor = PerceiverCompressor(
                feat_dim=token_dim,
                num_tokens=num_tokens,
                num_layers=num_cross_attn_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_pos_embed=use_pos_embed,
                pos_embed_type=pos_embed_type,
            )
        elif compression_mode == 'detr':
            self.compressor = DETRCompressor(
                feat_dim=token_dim,
                num_tokens=num_tokens,
                num_layers=num_cross_attn_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        elif compression_mode == 'pooling':
            self.compressor = PoolingCompressor(
                feat_dim=token_dim,
                num_tokens=num_tokens,
                pool_mode='adaptive_max',  # 可选: adaptive_max, adaptive_avg, learned
            )
        else:
            raise ValueError(f"未知的压缩模式: {compression_mode}")
        
        logger.info(f"✓ Adapter 初始化完成，参数量: {self.count_parameters()/1e6:.2f}M")
    
    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self, 
        point: Point,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            point: Sonata 输出的 Point 对象
                - point.feat: (N_total, C) 所有样本的特征拼接
                - point.coord: (N_total, 3) 所有样本的坐标拼接
                - point.offset: (B,) 累积偏移量，用于分隔不同样本
            return_attention: 是否返回注意力权重（用于可视化）
        
        Returns:
            tokens: (B, K, D) Scene tokens
            coords: (B, K, 3) Token 对应的坐标（可选，用于可视化）
            attention_weights: (B, K, M or N) 注意力权重（可选）
        """
        # 1. 解析批次信息
        feat = point.feat              # (N_total, C)
        coord = point.coord            # (N_total, 3)
        offset = point.offset          # (B,)
        
        B = len(offset)
        sizes = offset2bincount(offset)  # (B,) 每个样本的点数
        
        # 2. 特征投影
        feat = self.feat_proj(feat)    # (N_total, D)
        
        # 3. 按样本分离特征和坐标
        feat_list = feat.split(sizes.tolist(), dim=0)    # List[(N_i, D)]
        coord_list = coord.split(sizes.tolist(), dim=0)  # List[(N_i, 3)]
        
        # 4. 预筛选（可选）
        if self.prefilter is not None:
            feat_list, coord_list = self.prefilter(feat_list, coord_list)
            M = self.num_candidates
        else:
            M = None  # 使用原始点数
        
        # 5. Padding 到统一长度（用于批处理）
        feat_padded, coord_padded, mask = self._pad_and_mask(feat_list, coord_list)
        # feat_padded: (B, M or max_N, D)
        # coord_padded: (B, M or max_N, 3)
        # mask: (B, M or max_N) - True 表示有效点
        
        # 6. 压缩到固定 token 数
        if return_attention and hasattr(self.compressor, 'forward_with_attention'):
            tokens, token_coords, attn = self.compressor.forward_with_attention(
                feat_padded, coord_padded, mask
            )
            return tokens, token_coords, attn
        else:
            tokens, token_coords, _ = self.compressor(feat_padded, coord_padded, mask)
            return tokens, token_coords, None
    
    def _pad_and_mask(
        self, 
        feat_list: list, 
        coord_list: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将可变长度的特征和坐标 Padding 到统一长度
        
        Returns:
            feat_padded: (B, max_N, D)
            coord_padded: (B, max_N, 3)
            mask: (B, max_N) bool，True 表示有效点
        """
        B = len(feat_list)
        max_N = max(f.shape[0] for f in feat_list)
        D = feat_list[0].shape[1]
        device = feat_list[0].device
        
        feat_padded = torch.zeros(B, max_N, D, device=device, dtype=feat_list[0].dtype)
        coord_padded = torch.zeros(B, max_N, 3, device=device, dtype=coord_list[0].dtype)
        mask = torch.zeros(B, max_N, device=device, dtype=torch.bool)
        
        for i, (feat, coord) in enumerate(zip(feat_list, coord_list)):
            N = feat.shape[0]
            feat_padded[i, :N] = feat
            coord_padded[i, :N] = coord
            mask[i, :N] = True
        
        return feat_padded, coord_padded, mask


class PointPrefilter(nn.Module):
    """
    点云预筛选模块
    从 Sonata 的 N 个输出点中选择 M 个候选点（M < N）
    用于减少后续 Cross-Attention 的计算量
    """
    
    def __init__(
        self,
        in_dim: int,
        num_candidates: int,
        mode: Literal['fps', 'random', 'surface'] = 'fps',
    ):
        super().__init__()
        self.num_candidates = num_candidates
        self.mode = mode
        
        if mode == 'surface':
            # Surface-aware: 需要学习一个评分网络
            self.score_mlp = nn.Sequential(
                nn.Linear(in_dim + 3, in_dim),  # feat + coord
                nn.ReLU(),
                nn.Linear(in_dim, 1),
            )
        
        logger.info(f"预筛选模式: {mode}, 候选点数: {num_candidates}")
    
    def forward(
        self, 
        feat_list: list, 
        coord_list: list
    ) -> Tuple[list, list]:
        """
        Args:
            feat_list: List[(N_i, D)]
            coord_list: List[(N_i, 3)]
        
        Returns:
            filtered_feat_list: List[(M, D)]
            filtered_coord_list: List[(M, 3)]
        """
        filtered_feat = []
        filtered_coord = []
        
        for feat, coord in zip(feat_list, coord_list):
            N = feat.shape[0]
            M = min(self.num_candidates, N)
            
            if self.mode == 'fps':
                # Farthest Point Sampling
                idx = self._fps(coord, M)
            elif self.mode == 'random':
                # 随机采样
                idx = torch.randperm(N, device=feat.device)[:M]
            elif self.mode == 'surface':
                # Surface-aware: 基于特征+坐标的重要性评分
                score = self.score_mlp(torch.cat([feat, coord], dim=-1)).squeeze(-1)
                idx = torch.topk(score, M, largest=True).indices
            else:
                raise ValueError(f"未知的预筛选模式: {self.mode}")
            
            filtered_feat.append(feat[idx])
            filtered_coord.append(coord[idx])
        
        return filtered_feat, filtered_coord
    
    def _fps(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Farthest Point Sampling (FPS)
        
        采用 TorchScript 加速版本，避免 Python <-> CUDA 同步开销。
        """
        return _jit_farthest_point_sampling(points, num_samples)


class PerceiverCompressor(nn.Module):
    """
    Perceiver-style 压缩器（推荐）
    
    核心思想：
    - 使用可学习的 Query Embeddings (K, D)
    - 通过 Cross-Attention 从候选点 (N, D) 中提取信息
    - 不断迭代 refine（多层 Cross-Attention）
    
    优势：
    - 参数量少（只有 K 个 query）
    - 计算量可控 O(K × N)
    - 可以处理任意长度的输入 N
    """
    
    def __init__(
        self,
        feat_dim: int,
        num_tokens: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_pos_embed: bool = True,
        pos_embed_type: Literal['learnable', 'sinusoidal'] = 'learnable',
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.use_pos_embed = use_pos_embed
        
        # 可学习的 Query Embeddings
        self.query_embed = nn.Parameter(torch.randn(num_tokens, feat_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        
        # 位置编码（用于候选点）
        if use_pos_embed:
            if pos_embed_type == 'learnable':
                # 学习坐标→位置编码的映射
                self.pos_encoder = nn.Sequential(
                    nn.Linear(3, feat_dim),
                    nn.ReLU(),
                    nn.Linear(feat_dim, feat_dim),
                )
            else:
                # Sinusoidal 位置编码（基于坐标）
                self.pos_encoder = SinusoidalPositionEmbedding(feat_dim)
        else:
            self.pos_encoder = None
        
        # Cross-Attention 层
        self.layers = nn.ModuleList([
            PerceiverBlock(
                query_dim=feat_dim,
                context_dim=feat_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 坐标预测（使用 attention weights 加权平均）
        self.coord_predictor = None  # 在 forward 中动态计算
        
        logger.info(f"PerceiverCompressor: {num_layers} layers, "
                    f"{num_tokens} tokens, {feat_dim} dim")
    
    def forward(
        self, 
        feat: torch.Tensor,       # (B, N, D)
        coord: torch.Tensor,      # (B, N, 3)
        mask: torch.Tensor,       # (B, N) bool
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            tokens: (B, K, D)
            token_coords: (B, K, 3)
            attn_weights (optional): (B, K, N)
        """
        B, N, D = feat.shape
        K = self.num_tokens
        
        # 1. 添加位置编码到候选点特征
        if self.pos_encoder is not None:
            pos_embed = self.pos_encoder(coord)  # (B, N, D)
            context = feat + pos_embed
        else:
            context = feat
        
        # 2. 初始化 queries (每个样本共享相同的初始 query)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        
        # 3. 迭代 Cross-Attention
        last_attn_weights = None
        for layer in self.layers:
            queries, attn_weights = layer(queries, context, mask)
            last_attn_weights = attn_weights  # (B, K, N)
        
        # 4. 预测 token 对应的坐标（使用最后一层的 attention weights）
        # token_coords[b, k] = sum_n attn_weights[b, k, n] * coord[b, n]
        token_coords = torch.einsum('bkn,bnc->bkc', last_attn_weights, coord)
        
        if return_attention:
            return queries, token_coords, last_attn_weights
        return queries, token_coords, None
    
    def forward_with_attention(
        self, 
        feat: torch.Tensor, 
        coord: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """带 attention 权重的前向传播（用于可视化）"""
        tokens, token_coords, attn_weights = self.forward(
            feat, coord, mask, return_attention=True
        )
        assert attn_weights is not None
        return tokens, token_coords, attn_weights


class PerceiverBlock(nn.Module):
    """Perceiver 的单层 Cross-Attention + Self-Attention + FFN"""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Cross-Attention (Query 从 Context 中提取信息)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(query_dim)
        
        # Self-Attention (Query 内部交互)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(query_dim)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, int(query_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(query_dim * mlp_ratio), query_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(query_dim)
    
    def forward(
        self, 
        query: torch.Tensor,        # (B, K, D)
        context: torch.Tensor,      # (B, N, D)
        context_mask: torch.Tensor, # (B, N) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (B, K, D)
            attn_weights: (B, K, N)
        """
        # 1. Cross-Attention
        attn_output, attn_weights = self.cross_attn(
            query=query,
            key=context,
            value=context,
            key_padding_mask=~context_mask,  # True 表示忽略
            need_weights=True,
            average_attn_weights=True,
        )
        query = self.norm1(query + attn_output)
        
        # 2. Self-Attention
        self_attn_output, _ = self.self_attn(
            query=query,
            key=query,
            value=query,
            need_weights=False,
        )
        query = self.norm2(query + self_attn_output)
        
        # 3. FFN
        mlp_output = self.mlp(query)
        query = self.norm3(query + mlp_output)
        
        return query, attn_weights


class DETRCompressor(nn.Module):
    """
    DETR-style 压缩器（与现有 scene_tokenizer 类似）
    使用 Transformer Decoder 结构
    """
    
    def __init__(
        self,
        feat_dim: int,
        num_tokens: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        
        # 可学习的 Query Embeddings
        self.query_embed = nn.Parameter(torch.randn(num_tokens, feat_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=int(feat_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        logger.info(f"DETRCompressor: {num_layers} layers, {num_tokens} tokens")
    
    def forward(
        self, 
        feat: torch.Tensor, 
        coord: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feat.shape[0]
        
        # Query embeddings
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        
        # Decoder
        tokens = self.decoder(
            tgt=queries,
            memory=feat,
            memory_key_padding_mask=~mask,
        )
        
        # 坐标预测（简单版：使用最后一层 attention，这里简化为零坐标）
        token_coords = torch.zeros(B, self.num_tokens, 3, device=feat.device)
        
        return tokens, token_coords


class PoolingCompressor(nn.Module):
    """
    简单池化压缩器（最快但可能丢失信息）
    适合快速原型验证
    """
    
    def __init__(
        self,
        feat_dim: int,
        num_tokens: int,
        pool_mode: str = 'adaptive_max',
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.pool_mode = pool_mode
        
        if pool_mode == 'learned':
            # 学习的池化权重
            self.pool_weights = nn.Parameter(torch.randn(num_tokens, feat_dim))
        
        logger.info(f"PoolingCompressor: mode={pool_mode}, {num_tokens} tokens")
    
    def forward(
        self, 
        feat: torch.Tensor, 
        coord: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = feat.shape
        K = self.num_tokens
        
        if self.pool_mode == 'adaptive_max':
            # 自适应最大池化
            tokens = F.adaptive_max_pool1d(
                feat.transpose(1, 2),  # (B, D, N)
                output_size=K
            ).transpose(1, 2)  # (B, K, D)
        elif self.pool_mode == 'adaptive_avg':
            # 自适应平均池化
            tokens = F.adaptive_avg_pool1d(
                feat.transpose(1, 2),
                output_size=K
            ).transpose(1, 2)
        elif self.pool_mode == 'learned':
            # 学习的池化
            tokens = torch.einsum('bnd,kd->bnk', feat, self.pool_weights)
            tokens = tokens.transpose(1, 2)  # (B, K, D)
        else:
            raise ValueError(f"未知的池化模式: {self.pool_mode}")
        
        # 坐标（简化）
        token_coords = torch.zeros(B, K, 3, device=feat.device)
        
        return tokens, token_coords


class SinusoidalPositionEmbedding(nn.Module):
    """基于坐标的 Sinusoidal 位置编码"""
    
    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) xyz 坐标
        
        Returns:
            pos_embed: (B, N, D)
        """
        B, N, _ = coords.shape
        device = coords.device
        
        # 为每个维度 (x, y, z) 生成位置编码
        div_term = torch.exp(
            torch.arange(0, self.dim // 3, 2, device=device, dtype=torch.float32) *
            -(math.log(self.temperature) / (self.dim // 3))
        )
        
        pos_embed = torch.zeros(B, N, self.dim, device=device)
        
        for i in range(3):  # x, y, z
            start_idx = i * (self.dim // 3)
            end_idx = start_idx + (self.dim // 3)
            
            pos_embed[:, :, start_idx:end_idx:2] = torch.sin(
                coords[:, :, i:i+1] * div_term
            )
            pos_embed[:, :, start_idx+1:end_idx:2] = torch.cos(
                coords[:, :, i:i+1] * div_term
            )
        
        return pos_embed


# 注意：完整的 Sonata Backbone 封装在 models/backbone/sonata_backbone.py 中，避免在本文件重复定义
