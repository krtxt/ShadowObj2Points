"""
PTv3 稀疏 Token 提取器

专门用于从点云中提取稀疏的 scene tokens，适用于：
- Diffusion Transformer (DiT) 的条件输入
- 点云的紧凑表示
- 需要固定数量或可变数量的 scene tokens

关键配置：
- grid_size: 控制体素化精度，越小越精细
- stride: encoder 的下采样步长，控制最终稀疏程度
- 不使用 decoder，直接输出 encoder 的稀疏特征
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
from types import SimpleNamespace

# Add 3rd_party/pointnet2 to sys.path to enable importing the CUDA-optimized version
_repo_root = Path(__file__).resolve().parents[3]
_pn2_path = _repo_root / "3rd_party" / "pointnet2"
if _pn2_path.exists() and str(_pn2_path) not in sys.path:
    sys.path.append(str(_pn2_path))

import torch
import torch.nn as nn

from .ptv3.ptv3 import PointTransformerV3
from .ptv3_backbone import convert_to_ptv3_pc_format

from pointnet2_utils import furthest_point_sample as farthest_point_sample


class PTv3SparseEncoder(nn.Module):
    """
    PTv3 稀疏 Token 提取器
    
    输出稀疏的 scene tokens 而不是密集的点特征
    
    Args:
        cfg: 配置对象
        target_num_tokens: 目标 token 数量（可选，用于后处理）
    """
    
    def __init__(
        self,
        cfg=None,
        target_num_tokens: Optional[int] = None,
        token_strategy: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        if cfg is None:
            cfg = SimpleNamespace(**kwargs)
        
        # 基础配置
        self.grid_size = getattr(cfg, 'grid_size', 0.003)  # 默认0.003以获得更密集的tokens
        self.use_flash_attention = getattr(cfg, 'use_flash_attention', True)
        self.target_num_tokens = target_num_tokens or getattr(cfg, 'target_num_tokens', 128)
        
        # Token 选择策略
        valid_strategies = ['last_layer', 'fps', 'grid', 'learned', 'multiscale', 'perceiver']
        self.token_strategy = token_strategy or getattr(cfg, 'token_strategy', 'last_layer')  # 默认使用last_layer策略
        if self.token_strategy not in valid_strategies:
            raise ValueError(f"token_strategy must be one of {valid_strategies}, got {self.token_strategy}")

        # 输出维度
        self.output_dim = getattr(cfg, 'out_dim', 256)
        
        # Encoder 配置
        encoder_channels = list(getattr(cfg, 'encoder_channels', [32, 64, 128, 256]))
        encoder_depths = list(getattr(cfg, 'encoder_depths', [1, 1, 2, 2]))
        encoder_num_head = tuple(getattr(cfg, 'encoder_num_head', (2, 4, 8, 16)))
        enc_patch_size = tuple(getattr(cfg, 'enc_patch_size', (1024, 1024, 1024, 1024)))
        
        # Stride 配置：控制下采样程度
        # 注意：len(stride) = len(encoder_channels) - 1
        # 例如：4个阶段需要3个stride值，定义相邻阶段间的下采样
        # stride=(2,2,2) 表示每层下采样 2x，总共下采样 8x
        stride = tuple(getattr(cfg, 'stride', (2, 2, 2)))
        
        # 验证 stride 长度
        num_stages = len(encoder_channels)
        assert len(stride) == num_stages - 1, \
            f"stride length ({len(stride)}) must be num_stages - 1 ({num_stages - 1})"
        
        mlp_ratio = getattr(cfg, 'mlp_ratio', 2)
        in_channels = max(1, getattr(cfg, 'input_feature_dim', 1))
        
        # 保存 encoder 配置用于多尺度策略
        self.encoder_channels = encoder_channels
        self.num_stages = num_stages
        # 保存 stride 以便 get_expected_token_count 使用
        self.stride = stride
        # 输出形状控制：是否返回 [B, K, D]
        self.tokens_last = getattr(cfg, 'tokens_last', False)
        
        # Grid 策略的配置
        self.grid_resolution = getattr(cfg, 'grid_resolution', (8, 8, 8))  # 默认 8x8x8 = 512
        
        # 构建只有 encoder 的 PTv3
        # 关键：使用 cls_mode=True 获取稀疏特征（不运行 decoder）
        self.logger.info(
            f"Initializing PTv3SparseEncoder: "
            f"grid_size={self.grid_size}, "
            f"output_dim={self.output_dim}, "
            f"stride={stride}, "
            f"target_tokens={self.target_num_tokens}, "
            f"strategy={self.token_strategy}"
        )
        
        # Decoder 配置（cls_mode=True 时不会使用，但需要提供占位值）
        # 长度必须是 num_stages - 1
        dec_depths = [1] * (num_stages - 1)
        dec_channels = encoder_channels[:-1][::-1]  # 反向，排除最后一个
        dec_num_head = encoder_num_head[:-1][::-1]
        dec_patch_size = enc_patch_size[:-1][::-1]
        
        try:
            self.model = PointTransformerV3(
                in_channels=in_channels,
                order=["z", "z-trans"],
                stride=stride,  # 控制稀疏程度
                enc_depths=encoder_depths,
                enc_channels=encoder_channels,
                enc_num_head=encoder_num_head,
                enc_patch_size=enc_patch_size,
                dec_depths=dec_depths,  # 占位
                dec_channels=dec_channels,  # 占位
                dec_num_head=dec_num_head,  # 占位
                dec_patch_size=dec_patch_size,  # 占位
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                drop_path=0.1,
                pre_norm=True,
                shuffle_orders=True,
                enable_rpe=False,
                enable_flash=self.use_flash_attention,
                upcast_attention=False,
                upcast_softmax=False,
                cls_mode=False,  # ← 使用分割模式，获取点级特征
                pdnorm_bn=False,
                pdnorm_ln=False,
                pdnorm_decouple=True,
                pdnorm_adaptive=False,
                pdnorm_affine=True,
                pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
            )
        except Exception as e:
            self.logger.error(f"Failed to build PTv3 model: {e}")
            raise
        
        # 输出维度投影（如果需要）
        encoder_out_dim = encoder_channels[-1]
        if encoder_out_dim != self.output_dim:
            self.logger.info(
                f"Adding projection: {encoder_out_dim} -> {self.output_dim}"
            )
            self.out_proj = nn.Linear(encoder_out_dim, self.output_dim)
        else:
            self.out_proj = None
        
        # 学习式 Tokenizer（方案④ / Perceiver 风格）
        if self.token_strategy in ('learned', 'perceiver'):
            self.learned_tokenizer = self._build_learned_tokenizer()
            self.logger.info(
                f"Built learned/perceiver tokenizer with {self.target_num_tokens} query tokens"
            )
        else:
            self.learned_tokenizer = None

        # 多尺度投影层（方案⑤）
        if self.token_strategy == 'multiscale':
            self.stage_projections = nn.ModuleDict()
            # 为每个encoder stage创建投影层
            for i, ch in enumerate(encoder_channels[1:]):  # 跳过第0层
                if ch != self.output_dim:
                    self.stage_projections[f'stage{i+1}'] = nn.Linear(ch, self.output_dim)
            self.logger.info(
                f"Built {len(self.stage_projections)} stage projection layers for multiscale"
            )
    
    def forward(self, pos: torch.Tensor, return_full_res: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，提取稀疏 scene tokens
        
        Args:
            pos: (B, N, C) 输入点云，C>=3 (xyz + optional features)
            return_full_res: 是否返回完整分辨率（用于兼容性）
            
        Returns:
            xyz: (B, K, 3) 稀疏点坐标，K << N
            features: (B, D, K) 稀疏点特征
            
        注意：
            - K 的大小取决于 grid_size 和 stride
            - grid_size 越小 → 初始体素越多
            - stride 越大 → 下采样越强 → K 越小
            
        示例 K 值：
            - grid_size=0.02, stride=(2,2,2,2): K ≈ 300-600
            - grid_size=0.01, stride=(2,2,2,2): K ≈ 600-1200
            - grid_size=0.04, stride=(2,2,2,2): K ≈ 150-300
        """
        B, N, C = pos.shape
        device = pos.device
        
        # 分割坐标和特征
        coords = pos[..., :3]  # (B, N, 3)
        
        if C > 3:
            feat = pos[..., 3:]  # (B, N, C-3)
        else:
            feat = torch.ones(B, N, 1, device=device, dtype=pos.dtype)
        
        # 转换为 PTv3 格式
        data_dict = convert_to_ptv3_pc_format(coords, feat, self.grid_size)

        # 先处理多尺度策略，避免重复跑 encoder
        if self.token_strategy == 'multiscale':
            xyz_out, feat_out = self._strategy_multiscale(pos, data_dict)
            # tokens_last 支持
            if self.tokens_last:
                feat_out = feat_out.permute(0, 2, 1)
            # 可选的调试信息
            if return_full_res:
                self.debug_last = {
                    'path': 'multiscale',
                }
            return xyz_out, feat_out
        
        # 手动运行 PTv3 的 encoder 部分
        try:
            # Step 1: Create Point object and serialize
            from .ptv3.ptv3 import Point
            point = Point(data_dict)
            point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
            point.sparsify()
            
            # Step 2: Embedding (初始特征提取)
            point = self.model.embedding(point)
            
            # Step 3: Encoder (多层下采样和特征提取)
            point = self.model.enc(point)
            
            # 现在 point 是 encoder 的稀疏输出
            sparse_coord = point.coord  # (M, 3) M 是总稀疏点数
            sparse_feat = point.feat    # (M, C)
            offset = point.offset        # (B,)
            
            # 日志输出
            M = sparse_coord.shape[0]
            compression_ratio = M / (B * N)
            self.logger.debug(
                f"[PTv3Sparse] Input: {B}x{N}, Output: {M} sparse points "
                f"({M/B:.1f} per sample, compression={compression_ratio:.3f})"
            )
            
        except Exception as e:
            self.logger.error(f"PTv3 forward failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
        # 转换为批次格式
        xyz_sparse, feat_sparse = self._sparsify_by_batch(
            sparse_coord, sparse_feat, offset, B, device
        )
        
        # 特征维度投影
        if self.out_proj is not None:
            # feat_sparse: (B, C_in, K) -> (B, K, C_in) -> (B, K, C_out) -> (B, C_out, K)
            feat_sparse = feat_sparse.permute(0, 2, 1)  # (B, K, C_in)
            feat_sparse = self.out_proj(feat_sparse)     # (B, K, C_out)
            feat_sparse = feat_sparse.permute(0, 2, 1)   # (B, C_out, K)
        
        # 根据策略应用不同的 token 选择方法
        if self.token_strategy == 'last_layer':
            xyz_out, feat_out = self._strategy_last_layer(xyz_sparse, feat_sparse, coords)
        elif self.token_strategy == 'fps':
            xyz_out, feat_out = self._strategy_fps(xyz_sparse, feat_sparse)
        elif self.token_strategy == 'grid':
            xyz_out, feat_out = self._strategy_grid(xyz_sparse, feat_sparse, coords)
        elif self.token_strategy in ('learned', 'perceiver'):
            xyz_out, feat_out = self._strategy_learned(xyz_sparse, feat_sparse)
        else:
            raise ValueError(f"Unknown token strategy: {self.token_strategy}")

        # tokens_last 支持
        if self.tokens_last:
            feat_out = feat_out.permute(0, 2, 1)

        # 返回完整分辨率以便调试
        if return_full_res:
            self.debug_last = {
                'xyz_sparse_full': xyz_sparse,
                'feat_sparse_full': feat_sparse,
            }
            # 若 grid 策略，附带网格索引
            if hasattr(self, '_last_grid_idx'):
                self.debug_last['grid_idx'] = self._last_grid_idx
        
        return xyz_out, feat_out
    
    def _sparsify_by_batch(
        self, 
        coord: torch.Tensor,  # (M, 3)
        feat: torch.Tensor,   # (M, C)
        offset: torch.Tensor, # (B,)
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将稀疏格式转换为批次格式
        
        Returns:
            xyz_sparse: (B, K, 3)
            feat_sparse: (B, C, K)
        """
        C = feat.shape[1]
        
        # Vectorized construction of batch_idx
        # offset contains end indices: [N1, N1+N2, ...]
        # We need counts: [N1, N2, ...]
        counts = torch.diff(offset, prepend=torch.tensor([0], device=device, dtype=torch.long))
        
        # Only use the first batch_size elements of counts if offset has more
        if len(counts) > batch_size:
             counts = counts[:batch_size]
        
        # batch_idx maps each point to its batch index [0, 0, 0, 1, 1, ...]
        batch_idx = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
        
        # 找到最大点数
        if batch_idx.numel() > 0:
            bincount = torch.bincount(batch_idx, minlength=batch_size)
            K = bincount.max().item()
        else:
            K = 0
        
        # 初始化输出张量
        xyz_sparse = torch.zeros(batch_size, K, 3, device=device, dtype=coord.dtype)
        feat_sparse = torch.zeros(batch_size, C, K, device=device, dtype=feat.dtype)
        
        if K > 0:
            # Compute intra-batch indices [0, 1, 2, 0, 1, ...]
            # Global index - start index of corresponding batch
            starts = torch.cat([torch.tensor([0], device=device, dtype=torch.long), offset[:-1]])
            if len(starts) > batch_size:
                starts = starts[:batch_size]
                
            intra_batch_idx = torch.arange(coord.shape[0], device=device) - starts[batch_idx]
            
            # Vectorized assignment
            xyz_sparse[batch_idx, intra_batch_idx] = coord
            
            # feat_sparse (B, C, K) -> (B, K, C) for easy assignment
            # permute returns a view, so modification affects original tensor
            feat_sparse.permute(0, 2, 1)[batch_idx, intra_batch_idx] = feat
        
        return xyz_sparse, feat_sparse
    
    def get_expected_token_count(self, input_points: int) -> int:
        """
        估算给定输入点数下的输出 token 数量
        
        这只是一个粗略估计，实际数量取决于点云的空间分布
        """
        # 粗略估计：每次 stride 下采样会减少点数
        total_stride = 1
        for s in self.stride:
            total_stride *= s
        
        # 体素化 + 下采样
        estimated_tokens = input_points // (total_stride * 4)
        
        if self.target_num_tokens is not None:
            return self.target_num_tokens
        
        return estimated_tokens
    
    # ==================== Token 策略实现方法 ====================
    
    def _strategy_last_layer(
        self, 
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor,  # (B, C, K)
        orig_coords: torch.Tensor  # (B, N, 3) 原始输入坐标
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案①：直接取最后一层特征
        
        如果自然稀疏点数接近target，直接返回；
        否则使用FPS采样到target数量
        """
        B, K, _ = xyz.shape
        
        if K == self.target_num_tokens:
            return xyz, feat
        elif K > self.target_num_tokens:
            # 使用FPS下采样
            return self._fps_sample(xyz, feat, self.target_num_tokens)
        else:
            # 填充到target数量
            self.logger.debug(f"Padding from {K} to {self.target_num_tokens} tokens")
            xyz_padded = torch.zeros(B, self.target_num_tokens, 3, device=xyz.device, dtype=xyz.dtype)
            feat_padded = torch.zeros(B, feat.shape[1], self.target_num_tokens, device=feat.device, dtype=feat.dtype)
            xyz_padded[:, :K] = xyz
            feat_padded[:, :, :K] = feat
            return xyz_padded, feat_padded
    
    def _strategy_fps(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor  # (B, C, K)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案②：FPS采样选K个点
        
        使用Farthest Point Sampling确保空间分布均匀
        """
        return self._fps_sample(xyz, feat, self.target_num_tokens)
    
    def _strategy_grid(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor,  # (B, C, K)
        orig_coords: torch.Tensor  # (B, N, 3) 原始输入坐标
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案③：Grid/Voxel 聚合
        
        将空间划分为规则网格，每个格子聚合特征
        """
        B = xyz.shape[0]
        C = feat.shape[1]
        device = xyz.device
        
        # 网格分辨率
        gx, gy, gz = self.grid_resolution
        grid_total = gx * gy * gz

        # 如果网格总数不等于target，调整为接近 target 的立方体尺寸
        if grid_total != self.target_num_tokens:
            target_per_dim = int(self.target_num_tokens ** (1/3)) + 1
            gx = gy = gz = target_per_dim
            grid_total = gx * gy * gz
            self.logger.debug(f"Adjusted grid to {gx}x{gy}x{gz} = {grid_total}")

        # 计算点云边界（按 batch）
        xyz_min = xyz.min(dim=1, keepdim=True)[0]  # (B, 1, 3)
        xyz_max = xyz.max(dim=1, keepdim=True)[0]  # (B, 1, 3)
        xyz_range = xyz_max - xyz_min + 1e-6

        # 将坐标归一化到[0, 1]
        xyz_norm = (xyz - xyz_min) / xyz_range  # (B, K, 3)

        # 计算网格索引
        grid_idx_x = (xyz_norm[..., 0] * gx).clamp(0, gx - 1).long()
        grid_idx_y = (xyz_norm[..., 1] * gy).clamp(0, gy - 1).long()
        grid_idx_z = (xyz_norm[..., 2] * gz).clamp(0, gz - 1).long()
        grid_idx = grid_idx_x * (gy * gz) + grid_idx_y * gz + grid_idx_z  # (B, K)
        # 存储调试
        self._last_grid_idx = grid_idx

        # 有效点掩码（排除 padding 的 0）
        valid_mask = (xyz.abs().sum(dim=-1) > 0)  # (B, K)

        # 向量化聚合：展平后用 index_add_ 聚合到 (B * grid_total)
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand_as(grid_idx)
        linear_idx = batch_ids * grid_total + grid_idx  # (B, K)
        linear_idx_valid = linear_idx[valid_mask]

        # 聚合坐标
        agg_xyz = torch.zeros(B * grid_total, 3, device=device, dtype=xyz.dtype)
        agg_xyz.index_add_(0, linear_idx_valid, xyz[valid_mask])

        # 聚合特征
        agg_feat = torch.zeros(B * grid_total, C, device=device, dtype=feat.dtype)
        # 将 feat (B, C, K) 转为 (B, K, C) 再按 valid 展平
        feat_bkc = feat.permute(0, 2, 1)  # (B, K, C)
        agg_feat.index_add_(0, linear_idx_valid, feat_bkc[valid_mask])

        # 计数
        agg_cnt = torch.zeros(B * grid_total, 1, device=device, dtype=torch.long)
        ones = torch.ones(linear_idx_valid.shape[0], 1, device=device, dtype=torch.long)
        agg_cnt.index_add_(0, linear_idx_valid, ones)

        # 还原形状
        grid_xyz = agg_xyz.view(B, grid_total, 3)
        grid_feat = agg_feat.view(B, grid_total, C).permute(0, 2, 1)  # (B, C, grid_total)
        grid_count = agg_cnt.view(B, grid_total).clamp_min(1).float()  # 防止除零

        # 平均化
        grid_xyz = grid_xyz / grid_count.unsqueeze(-1)
        grid_feat = grid_feat / grid_count.unsqueeze(1)

        # 为空的单元（真实 count==0）填充网格中心
        true_empty = (agg_cnt.view(B, grid_total) == 0)
        if true_empty.any():
            grid_centers = self._generate_grid_centers(gx, gy, gz, xyz_min, xyz_range, device)  # (B, grid_total, 3)
            grid_xyz[true_empty] = grid_centers[true_empty]

        # 如果 grid 总数不等于 target，使用 FPS 调整
        if grid_total != self.target_num_tokens:
            grid_xyz, grid_feat = self._fps_sample(grid_xyz, grid_feat, self.target_num_tokens)

        return grid_xyz, grid_feat
    
    def _strategy_learned(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor  # (B, C, K)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案④：学习式Tokenizer
        
        使用可学习的cross-attention生成固定数量的tokens
        """
        B = xyz.shape[0]
        device = xyz.device
        
        # feat: (B, C, K) -> (B, K, C)
        feat_transposed = feat.permute(0, 2, 1)  # (B, K, C)
        
        # 使用learned tokenizer进行cross-attention
        # query: (num_tokens, C), key/value: (B, K, C)
        token_feat = self.learned_tokenizer(feat_transposed)  # (B, num_tokens, C)
        
        # 转回 (B, C, num_tokens)
        token_feat = token_feat.permute(0, 2, 1)
        
        # 对于坐标，使用attention权重加权平均原始坐标
        # 简化版：使用FPS采样
        token_xyz = self._fps_sample(xyz, feat, self.target_num_tokens)[0]
        
        return token_xyz, token_feat
    
    def _strategy_multiscale(
        self,
        pos: torch.Tensor,  # (B, N, C) 原始输入
        data_dict: dict  # PTv3格式的输入
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案⑤：多尺度特征融合
        
        从encoder的多个阶段提取特征并拼接
        """
        B, N, C_in = pos.shape
        device = pos.device
        
        # 重新运行encoder并收集中间特征
        from .ptv3.ptv3 import Point
        point = Point(data_dict)
        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
        point.sparsify()
        point = self.model.embedding(point)
        
        # 收集多层特征
        stage_features = []
        stage_coords = []
        
        # 遍历encoder的各个阶段（PointSequential使用_modules存储子模块）
        for stage_idx, (stage_name, stage) in enumerate(self.model.enc._modules.items()):
            point = stage(point)
            
            # 从中间层提取特征（跳过第0层，太密集）
            if stage_idx > 0:
                coord = point.coord  # (M, 3)
                feat = point.feat    # (M, C)
                offset = point.offset  # (B,)
                
                # 转换为batch格式
                xyz_stage, feat_stage = self._sparsify_by_batch(
                    coord, feat, offset, B, device
                )
                
                stage_coords.append(xyz_stage)
                stage_features.append(feat_stage)
        
        # 特征投影到统一维度（使用预先创建的投影层）
        projected_features = []
        for stage_idx, feat in enumerate(stage_features):
            _, C_stage, K_stage = feat.shape
            # 如果这一层的特征维度不等于输出维度，使用对应的投影层
            stage_key = f'stage{stage_idx+1}'  # stage_idx从0开始，但我们跳过了第0层，所以+1
            if stage_key in self.stage_projections:
                feat = feat.permute(0, 2, 1)  # (B, K, C_stage)
                feat = self.stage_projections[stage_key](feat)  # (B, K, C_out)
                feat = feat.permute(0, 2, 1)   # (B, C_out, K)
            projected_features.append(feat)
        stage_features = projected_features
        
        # 从每层采样相同数量的点
        num_stages_used = len(stage_coords)
        tokens_per_stage = self.target_num_tokens // num_stages_used
        
        sampled_coords = []
        sampled_features = []
        
        for xyz, feat in zip(stage_coords, stage_features):
            xyz_s, feat_s = self._fps_sample(xyz, feat, tokens_per_stage)
            sampled_coords.append(xyz_s)
            sampled_features.append(feat_s)
        
        # 拼接所有阶段的tokens
        multi_xyz = torch.cat(sampled_coords, dim=1)  # (B, K_total, 3)
        multi_feat = torch.cat(sampled_features, dim=2)  # (B, C, K_total)
        
        # 如果总数不等于target，调整
        K_total = multi_xyz.shape[1]
        if K_total != self.target_num_tokens:
            multi_xyz, multi_feat = self._fps_sample(multi_xyz, multi_feat, self.target_num_tokens)
        
        return multi_xyz, multi_feat
    
    # ==================== 辅助方法 ====================
    
    def _fps_sample(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor,  # (B, C, K)
        target_K: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用FPS采样到目标数量的点
        """
        # from .pointnet2_utils import farthest_point_sample  # Legacy: Slow Python implementation
        
        # Try importing CUDA-optimized version first
        # try:
        #     from pointnet2_utils import farthest_point_sample
        # except ImportError:
        #     # self.logger.warning("Failed to import CUDA-optimized pointnet2_utils. Falling back to slow Python implementation.")
        #     from .pointnet2_utils import farthest_point_sample
        
        B, K, _ = xyz.shape
        C = feat.shape[1]
        device = xyz.device
        
        if K == target_K:
            return xyz, feat
        
        if K < target_K:
            # 填充
            xyz_padded = torch.zeros(B, target_K, 3, device=device, dtype=xyz.dtype)
            feat_padded = torch.zeros(B, C, target_K, device=device, dtype=feat.dtype)
            xyz_padded[:, :K] = xyz
            feat_padded[:, :, :K] = feat
            return xyz_padded, feat_padded
        
        # FPS下采样
        xyz_sampled = torch.zeros(B, target_K, 3, device=device, dtype=xyz.dtype)
        feat_sampled = torch.zeros(B, C, target_K, device=device, dtype=feat.dtype)
        
        for b in range(B):
            # 找到有效点
            valid_mask = (xyz[b].abs().sum(dim=-1) > 0)
            n_valid = valid_mask.sum().item()
            
            if n_valid <= target_K:
                xyz_sampled[b, :n_valid] = xyz[b, valid_mask]
                feat_sampled[b, :, :n_valid] = feat[b, :, valid_mask]
            else:
                xyz_valid = xyz[b, valid_mask].unsqueeze(0)  # (1, n_valid, 3)
                fps_idx = farthest_point_sample(xyz_valid, target_K)[0]  # (target_K,)
                
                valid_indices = torch.where(valid_mask)[0]
                sampled_indices = valid_indices[fps_idx]
                
                xyz_sampled[b] = xyz[b, sampled_indices]
                feat_sampled[b] = feat[b, :, sampled_indices]
        
        return xyz_sampled, feat_sampled
    
    def _generate_grid_centers(
        self,
        gx: int, gy: int, gz: int,
        xyz_min: torch.Tensor,  # (B, 1, 3)
        xyz_range: torch.Tensor,  # (B, 1, 3)
        device: torch.device
    ) -> torch.Tensor:
        """
        生成规则网格的中心坐标
        
        Returns:
            grid_centers: (B, gx*gy*gz, 3)
        """
        # 在[0,1]空间生成网格中心
        x = torch.linspace(0.5/gx, 1-0.5/gx, gx, device=device)
        y = torch.linspace(0.5/gy, 1-0.5/gy, gy, device=device)
        z = torch.linspace(0.5/gz, 1-0.5/gz, gz, device=device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid_centers_norm = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            grid_z.reshape(-1)
        ], dim=-1)  # (gx*gy*gz, 3)

        # 按 batch 转换回原始空间
        B = xyz_min.shape[0]
        grid_centers = (
            grid_centers_norm.unsqueeze(0) * xyz_range[:, 0].unsqueeze(1) + xyz_min[:, 0].unsqueeze(1)
        )  # (B, gx*gy*gz, 3)

        return grid_centers
    
    def _build_learned_tokenizer(self) -> nn.Module:
        """
        构建学习式Tokenizer模块（TokenLearner风格）
        """
        class LearnedTokenizer(nn.Module):
            def __init__(self, num_tokens, token_dim, num_heads=8):
                super().__init__()
                self.num_tokens = num_tokens
                self.token_dim = token_dim
                
                # 可学习的query tokens
                self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim))
                nn.init.trunc_normal_(self.query_tokens, std=0.02)
                
                # Cross-attention层
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=token_dim,
                    num_heads=num_heads,
                    batch_first=True
                )
                
                # Layer norm
                self.norm = nn.LayerNorm(token_dim)
                
            def forward(self, x):
                """
                Args:
                    x: (B, K, C) 输入特征
                Returns:
                    tokens: (B, num_tokens, C) 输出tokens
                """
                B = x.shape[0]
                
                # 扩展query tokens到batch
                queries = self.query_tokens.expand(B, -1, -1)  # (B, num_tokens, C)
                
                # Cross-attention: queries attend to input features
                attn_out, attn_weights = self.cross_attn(
                    query=queries,
                    key=x,
                    value=x,
                    need_weights=True,
                    average_attn_weights=False,
                )  # (B, num_tokens, C), (B, num_heads, num_tokens, K)
                
                # Residual + norm
                tokens = self.norm(queries + attn_out)

                # 存储注意力权重以便调试
                self.last_attn_weights = attn_weights
                
                return tokens
        
        return LearnedTokenizer(
            num_tokens=self.target_num_tokens,
            token_dim=self.output_dim,
            num_heads=8
        )

