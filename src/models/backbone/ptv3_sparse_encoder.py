"""
PTv3 Sparse Token Encoder

Extract sparse scene tokens from point clouds for:
- DiT conditioning
- Compact point cloud representation  
- Fixed or variable token counts

Key configs:
- grid_size: voxelization precision (smaller = finer)
- stride: encoder downsampling (controls final sparsity)
- token_strategy: how to select output tokens
"""

import logging
import os
from typing import Tuple, Optional
from types import SimpleNamespace

import torch
import torch.nn as nn

from .ptv3.ptv3 import PointTransformerV3, Point
from .ptv3_backbone import convert_to_ptv3_pc_format
from pointnet2.pointnet2_utils import furthest_point_sample


# =============================================================================
# Constants
# =============================================================================

VALID_STRATEGIES = ('last_layer', 'fps', 'grid', 'learned', 'multiscale', 'perceiver')


# =============================================================================
# Learned Tokenizer (Perceiver-style cross-attention)
# =============================================================================

class LearnedTokenizer(nn.Module):
    """Cross-attention based tokenizer with learnable query tokens."""
    
    def __init__(self, num_tokens: int, token_dim: int, num_heads: int = 8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.last_attn_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, C) input features
        Returns:
            (B, num_tokens, C) output tokens
        """
        queries = self.query_tokens.expand(x.size(0), -1, -1)
        attn_out, self.last_attn_weights = self.cross_attn(
            queries, x, x, need_weights=True, average_attn_weights=False
        )
        return self.norm(queries + attn_out)


# =============================================================================
# Main Encoder
# =============================================================================

class PTv3SparseEncoder(nn.Module):
    """
    PTv3-based sparse token extractor.
    
    Outputs sparse scene tokens (B, K, 3) coords and (B, D, K) or (B, K, D) features.
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
        cfg = cfg or SimpleNamespace(**kwargs)
        
        # === Core config ===
        self.grid_size = getattr(cfg, 'grid_size', 0.003)
        self.target_num_tokens = target_num_tokens or getattr(cfg, 'target_num_tokens', 128)
        self.output_dim = getattr(cfg, 'out_dim', 256)
        self.tokens_last = getattr(cfg, 'tokens_last', False)
        self.grid_resolution = tuple(getattr(cfg, 'grid_resolution', (8, 8, 8)))
        
        # === Strategy validation ===
        self.token_strategy = token_strategy or getattr(cfg, 'token_strategy', 'last_layer')
        if self.token_strategy not in VALID_STRATEGIES:
            raise ValueError(f"token_strategy must be one of {VALID_STRATEGIES}")
        self.debug_stats = bool(
            getattr(cfg, 'debug_stats', False) or int(os.environ.get("PTV3_DEBUG_STATS", "0"))
        )
        self.debug_stats_steps = int(getattr(cfg, 'debug_stats_steps', 5))
        self._debug_calls = 0
        self._log_gate = False
        
        # === Encoder architecture ===
        enc_channels = list(getattr(cfg, 'encoder_channels', [32, 64, 128, 256]))
        enc_depths = list(getattr(cfg, 'encoder_depths', [1, 1, 2, 2]))
        enc_num_head = tuple(getattr(cfg, 'encoder_num_head', (2, 4, 8, 16)))
        enc_patch_size = tuple(getattr(cfg, 'enc_patch_size', (1024, 1024, 1024, 1024)))
        stride = tuple(getattr(cfg, 'stride', (2, 2, 2)))
        
        num_stages = len(enc_channels)
        assert len(stride) == num_stages - 1, \
            f"stride length ({len(stride)}) must be num_stages - 1 ({num_stages - 1})"
        
        self.encoder_channels = enc_channels
        self.num_stages = num_stages
        self.stride = stride
        
        # === Build PTv3 backbone ===
        self.model = self._build_ptv3(
            in_channels=max(1, getattr(cfg, 'input_feature_dim', 1)),
            enc_channels=enc_channels,
            enc_depths=enc_depths,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            stride=stride,
            mlp_ratio=getattr(cfg, 'mlp_ratio', 2),
            enable_flash=getattr(cfg, 'use_flash_attention', True),
        )
        
        # === Output projection ===
        self.out_proj = (
            nn.Linear(enc_channels[-1], self.output_dim)
            if enc_channels[-1] != self.output_dim else None
        )
        
        # === Strategy-specific modules ===
        self.learned_tokenizer = None
        self.stage_projections = nn.ModuleDict()
        
        if self.token_strategy in ('learned', 'perceiver'):
            self.learned_tokenizer = LearnedTokenizer(
                self.target_num_tokens, self.output_dim
            )
        elif self.token_strategy == 'multiscale':
            for i, ch in enumerate(enc_channels[1:], start=1):
                if ch != self.output_dim:
                    self.stage_projections[f'stage{i}'] = nn.Linear(ch, self.output_dim)
        
        self._log_init(stride)
    
    def _build_ptv3(
        self, in_channels, enc_channels, enc_depths, enc_num_head, 
        enc_patch_size, stride, mlp_ratio, enable_flash
    ) -> PointTransformerV3:
        """Build PTv3 backbone with placeholder decoder config."""
        n = len(enc_channels)
        return PointTransformerV3(
            in_channels=in_channels,
            order=["z", "z-trans"],
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=[1] * (n - 1),
            dec_channels=enc_channels[:-1][::-1],
            dec_num_head=enc_num_head[:-1][::-1],
            dec_patch_size=enc_patch_size[:-1][::-1],
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.1,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=enable_flash,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        )
    
    def _log_init(self, stride):
        self.logger.info(
            f"PTv3SparseEncoder: grid={self.grid_size}, dim={self.output_dim}, "
            f"stride={stride}, tokens={self.target_num_tokens}, strategy={self.token_strategy}"
        )
    
    # =========================================================================
    # Forward
    # =========================================================================
    
    def forward(
        self, pos: torch.Tensor, return_full_res: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract sparse scene tokens.
        
        Args:
            pos: (B, N, C) point cloud, C >= 3 (xyz + optional features)
            return_full_res: store debug info in self.debug_last
            
        Returns:
            xyz: (B, K, 3) sparse coordinates
            features: (B, D, K) or (B, K, D) if tokens_last=True
        """
        B, N, C = pos.shape
        device = pos.device
        
        coords = pos[..., :3]
        feat = pos[..., 3:] if C > 3 else torch.ones(B, N, 1, device=device, dtype=pos.dtype)
        data_dict = convert_to_ptv3_pc_format(coords, feat, self.grid_size)
        log_debug = self._debug_should_log()
        self._log_gate = log_debug
        if log_debug:
            self._log_dense_input(coords, data_dict)
        
        # Multiscale has its own encoder pass
        if self.token_strategy == 'multiscale':
            xyz_out, feat_out = self._strategy_multiscale(B, device, data_dict)
        else:
            xyz_sparse, feat_sparse = self._run_encoder(data_dict, B, device)
            if log_debug:
                self._log_batched_stats(xyz_sparse, feat_sparse, stage="batched_after_enc")
            xyz_out, feat_out = self._apply_strategy(xyz_sparse, feat_sparse, coords)
        
        if self.tokens_last:
            feat_out = feat_out.permute(0, 2, 1)
        
        if return_full_res:
            self._store_debug(xyz_sparse if self.token_strategy != 'multiscale' else None,
                             feat_sparse if self.token_strategy != 'multiscale' else None)
        
        self._log_gate = False
        return xyz_out, feat_out
    
    def _run_encoder(
        self, data_dict: dict, B: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run PTv3 encoder and convert to batched format."""
        point = Point(data_dict)
        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
        point.sparsify()
        self._log_sparse_state(point, stage="after_sparsify", batch_size=B)
        self._log_spconv_tensor(getattr(point, "sparse_conv_feat", None), stage="after_sparsify")
        point = self.model.embedding(point)
        self._log_spconv_tensor(getattr(point, "sparse_conv_feat", None), stage="after_embedding")
        try:
            point = self.model.enc(point)
        except Exception as e:
            # If spconv fails, dump a bit more detail for debugging.
            prev_gate = self._log_gate
            self._log_gate = True
            self._log_spconv_tensor(getattr(point, "sparse_conv_feat", None), stage="enc_before_fail")
            self._log_sparse_state(point, stage="enc_before_fail", batch_size=B)
            self._log_gate = prev_gate
            raise
        self._log_sparse_state(point, stage="after_encoder", batch_size=B)
        self._log_spconv_tensor(getattr(point, "sparse_conv_feat", None), stage="after_encoder")
        
        xyz_sparse, feat_sparse = self._to_batch_format(
            point.coord, point.feat, point.offset, B, device
        )
        
        if self.out_proj is not None:
            feat_sparse = self._project_features(feat_sparse)
        
        return xyz_sparse, feat_sparse
    
    def _apply_strategy(
        self, xyz: torch.Tensor, feat: torch.Tensor, orig_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply token selection strategy."""
        strategy_map = {
            'last_layer': lambda: self._strategy_last_layer(xyz, feat),
            'fps': lambda: self._fps_sample(xyz, feat, self.target_num_tokens),
            'grid': lambda: self._strategy_grid(xyz, feat),
            'learned': lambda: self._strategy_learned(xyz, feat),
            'perceiver': lambda: self._strategy_learned(xyz, feat),
        }
        return strategy_map[self.token_strategy]()
    
    def _store_debug(self, xyz_sparse, feat_sparse):
        self.debug_last = {'path': self.token_strategy}
        if xyz_sparse is not None:
            self.debug_last.update({
                'xyz_sparse_full': xyz_sparse,
                'feat_sparse_full': feat_sparse,
            })
        if hasattr(self, '_last_grid_idx'):
            self.debug_last['grid_idx'] = self._last_grid_idx
    
    # =========================================================================
    # Token Selection Strategies
    # =========================================================================
    
    def _strategy_last_layer(
        self, xyz: torch.Tensor, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use encoder output directly, adjust to target count."""
        K = xyz.size(1)
        if K == self.target_num_tokens:
            return xyz, feat
        if K > self.target_num_tokens:
            return self._fps_sample(xyz, feat, self.target_num_tokens)
        return self._pad_tokens(xyz, feat, self.target_num_tokens)
    
    def _strategy_grid(
        self, xyz: torch.Tensor, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate features into regular grid cells."""
        B, K, _ = xyz.shape
        C = feat.size(1)
        device = xyz.device
        
        # Compute grid dimensions
        gx, gy, gz = self._compute_grid_dims()
        G = gx * gy * gz
        
        # Normalize coordinates to [0, 1]
        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_range = xyz.max(dim=1, keepdim=True)[0] - xyz_min + 1e-6
        xyz_norm = (xyz - xyz_min) / xyz_range
        
        # Compute grid indices
        grid_idx = self._coords_to_grid_idx(xyz_norm, gx, gy, gz)
        self._last_grid_idx = grid_idx
        
        # Aggregate using scatter
        grid_xyz, grid_feat, grid_count = self._aggregate_to_grid(
            xyz, feat.permute(0, 2, 1), grid_idx, B, G, C, device
        )
        
        # Average and handle empty cells
        grid_count_safe = grid_count.clamp_min(1).float()
        grid_xyz = grid_xyz / grid_count_safe.unsqueeze(-1)
        grid_feat = (grid_feat / grid_count_safe.unsqueeze(-1)).permute(0, 2, 1)
        
        empty_mask = (grid_count == 0)
        if empty_mask.any():
            centers = self._generate_grid_centers(gx, gy, gz, xyz_min, xyz_range, device)
            grid_xyz[empty_mask] = centers[empty_mask]
        
        if G != self.target_num_tokens:
            return self._fps_sample(grid_xyz, grid_feat, self.target_num_tokens)
        return grid_xyz, grid_feat
    
    def _strategy_learned(
        self, xyz: torch.Tensor, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use cross-attention to generate fixed tokens."""
        feat_bkc = feat.permute(0, 2, 1)
        token_feat = self.learned_tokenizer(feat_bkc).permute(0, 2, 1)
        token_xyz = self._fps_sample(xyz, feat, self.target_num_tokens)[0]
        return token_xyz, token_feat
    
    def _strategy_multiscale(
        self, B: int, device: torch.device, data_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse features from multiple encoder stages."""
        point = Point(data_dict)
        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
        point.sparsify()
        point = self.model.embedding(point)
        
        stage_coords, stage_feats = [], []
        for idx, (_, stage) in enumerate(self.model.enc._modules.items()):
            point = stage(point)
            if idx > 0:  # Skip first stage (too dense)
                xyz_s, feat_s = self._to_batch_format(
                    point.coord, point.feat, point.offset, B, device
                )
                # Project to output dim
                key = f'stage{idx}'
                if key in self.stage_projections:
                    feat_s = self._project_with_layer(feat_s, self.stage_projections[key])
                stage_coords.append(xyz_s)
                stage_feats.append(feat_s)
        
        # Sample equally from each stage and concatenate
        tokens_per = self.target_num_tokens // len(stage_coords)
        sampled = [self._fps_sample(x, f, tokens_per) for x, f in zip(stage_coords, stage_feats)]
        
        multi_xyz = torch.cat([s[0] for s in sampled], dim=1)
        multi_feat = torch.cat([s[1] for s in sampled], dim=2)
        
        if multi_xyz.size(1) != self.target_num_tokens:
            return self._fps_sample(multi_xyz, multi_feat, self.target_num_tokens)
        return multi_xyz, multi_feat
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _to_batch_format(
        self, coord: torch.Tensor, feat: torch.Tensor, 
        offset: torch.Tensor, B: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert sparse (M, 3), (M, C) to batched (B, K, 3), (B, C, K)."""
        C = feat.size(1)
        counts = torch.diff(offset, prepend=torch.zeros(1, device=device, dtype=torch.long))
        if len(counts) > B:
            counts = counts[:B]
        
        batch_idx = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
        K = torch.bincount(batch_idx, minlength=B).max().item() if batch_idx.numel() else 0
        
        xyz_out = torch.zeros(B, K, 3, device=device, dtype=coord.dtype)
        feat_out = torch.zeros(B, C, K, device=device, dtype=feat.dtype)
        
        if K > 0:
            starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), offset[:-1]])
            if len(starts) > B:
                starts = starts[:B]
            intra_idx = torch.arange(coord.size(0), device=device) - starts[batch_idx]
            xyz_out[batch_idx, intra_idx] = coord
            feat_out.permute(0, 2, 1)[batch_idx, intra_idx] = feat
        
        return xyz_out, feat_out
    
    def _fps_sample(
        self, xyz: torch.Tensor, feat: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Farthest Point Sampling to target count."""
        B, K, _ = xyz.shape
        C = feat.size(1)
        device = xyz.device
        
        if K == target:
            return xyz, feat
        if K < target:
            return self._pad_tokens(xyz, feat, target)
        
        xyz_out = torch.zeros(B, target, 3, device=device, dtype=xyz.dtype)
        feat_out = torch.zeros(B, C, target, device=device, dtype=feat.dtype)
        
        for b in range(B):
            valid = xyz[b].abs().sum(-1) > 0
            n_valid = valid.sum().item()
            
            if n_valid <= target:
                xyz_out[b, :n_valid] = xyz[b, valid]
                feat_out[b, :, :n_valid] = feat[b, :, valid]
            else:
                fps_idx = furthest_point_sample(xyz[b, valid].unsqueeze(0), target)[0]
                idx = torch.where(valid)[0][fps_idx]
                xyz_out[b] = xyz[b, idx]
                feat_out[b] = feat[b, :, idx]
        
        return xyz_out, feat_out
    
    def _pad_tokens(
        self, xyz: torch.Tensor, feat: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-pad to target token count."""
        B, K, _ = xyz.shape
        device = xyz.device
        
        xyz_pad = torch.zeros(B, target, 3, device=device, dtype=xyz.dtype)
        feat_pad = torch.zeros(B, feat.size(1), target, device=device, dtype=feat.dtype)
        xyz_pad[:, :K] = xyz
        feat_pad[:, :, :K] = feat
        return xyz_pad, feat_pad

    # =========================================================================
    # Debug Utilities
    # =========================================================================
    def _debug_should_log(self) -> bool:
        if not self.debug_stats:
            return False
        if self._debug_calls >= self.debug_stats_steps:
            return False
        self._debug_calls += 1
        return True

    def _log_dense_input(self, coords: torch.Tensor, data_dict: dict):
        """Log dense input stats before sparsification."""
        try:
            B, N, _ = coords.shape
            counts = torch.diff(
                data_dict['offset'],
                prepend=torch.zeros(1, device=coords.device, dtype=data_dict['offset'].dtype),
            )
            coords_min = coords.amin(dim=(0, 1)).tolist() if coords.numel() else []
            coords_max = coords.amax(dim=(0, 1)).tolist() if coords.numel() else []
            self.logger.info(
                "[PTv3SparseEncoder debug][input] "
                f"B={B}, N={N}, dtype={coords.dtype}, "
                f"grid={data_dict.get('grid_size', self.grid_size)}, "
                f"counts(min/mean/max)={counts.min().item()}/{counts.float().mean().item():.1f}/{counts.max().item()}, "
                f"coords_min={coords_min}, coords_max={coords_max}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3SparseEncoder debug] failed to log input stats: {e}")

    def _log_sparse_state(self, point: Point, stage: str, batch_size: int):
        """Log sparse stats at intermediate stages."""
        if not (self.debug_stats and self._log_gate):
            return
        try:
            coord = getattr(point, "coord", None)
            feat = getattr(point, "feat", None)
            offset = getattr(point, "offset", None)
            coord_shape = tuple(coord.shape) if coord is not None else None
            feat_shape = tuple(feat.shape) if feat is not None else None
            if offset is not None:
                counts = torch.diff(
                    offset,
                    prepend=torch.zeros(1, device=offset.device, dtype=offset.dtype),
                )
                count_min, count_max = counts.min().item(), counts.max().item()
                count_mean = counts.float().mean().item()
            else:
                count_min = count_max = count_mean = None
            self.logger.info(
                f"[PTv3SparseEncoder debug][{stage}] "
                f"coord_shape={coord_shape}, feat_shape={feat_shape}, "
                f"counts(min/mean/max)={count_min}/{count_mean}/{count_max}, "
                f"batch_size={batch_size}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3SparseEncoder debug] failed to log sparse stats at {stage}: {e}")

    def _log_batched_stats(
        self, xyz_sparse: torch.Tensor, feat_sparse: torch.Tensor, stage: str
    ):
        """Log batched dense-format stats after encoder."""
        if not (self.debug_stats and self._log_gate):
            return
        try:
            B, K, _ = xyz_sparse.shape
            valid = (xyz_sparse.abs().sum(-1) > 0)
            counts = valid.sum(dim=1)
            self.logger.info(
                f"[PTv3SparseEncoder debug][{stage}] "
                f"xyz_shape={tuple(xyz_sparse.shape)}, feat_shape={tuple(feat_sparse.shape)}, "
                f"valid_tokens(min/mean/max)={counts.min().item()}/{counts.float().mean().item():.1f}/{counts.max().item()}, "
                f"dtype={xyz_sparse.dtype}, device={xyz_sparse.device}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3SparseEncoder debug] failed to log batched stats: {e}")

    def _log_spconv_tensor(self, tensor, stage: str):
        """Log spconv tensor internals (features/indices)."""
        if not (self.debug_stats and self._log_gate):
            return
        try:
            if tensor is None:
                self.logger.info(f"[PTv3SparseEncoder debug][{stage}] sparse_conv_feat=None")
                return
            feat = getattr(tensor, "features", None)
            idx = getattr(tensor, "indices", None)
            spatial = getattr(tensor, "spatial_shape", None)
            bs = getattr(tensor, "batch_size", None)
            feat_shape = tuple(feat.shape) if feat is not None else None
            idx_shape = tuple(idx.shape) if idx is not None else None
            counts = None
            idx_min = idx_max = None
            if idx is not None and idx.numel() > 0:
                counts = torch.bincount(idx[:, 0], minlength=int(bs) if bs is not None else None)
                idx_min = idx[:, 1:].amin(dim=0).tolist()
                idx_max = idx[:, 1:].amax(dim=0).tolist()
            self.logger.info(
                f"[PTv3SparseEncoder debug][{stage}] "
                f"feat_shape={feat_shape}, idx_shape={idx_shape}, "
                f"spatial_shape={tuple(spatial) if spatial is not None else None}, "
                f"batch_size={bs}, "
                f"counts_per_batch={counts.tolist() if counts is not None else None}, "
                f"idx_min={idx_min}, idx_max={idx_max}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3SparseEncoder debug] failed to log spconv tensor at {stage}: {e}")
    
    def _project_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Project features through out_proj: (B, C, K) -> (B, C', K)."""
        return self.out_proj(feat.permute(0, 2, 1)).permute(0, 2, 1)
    
    def _project_with_layer(self, feat: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Project features through given layer: (B, C, K) -> (B, C', K)."""
        return layer(feat.permute(0, 2, 1)).permute(0, 2, 1)
    
    def _compute_grid_dims(self) -> Tuple[int, int, int]:
        """Compute grid dimensions, adjusting to match target if needed."""
        gx, gy, gz = self.grid_resolution
        if gx * gy * gz != self.target_num_tokens:
            dim = int(self.target_num_tokens ** (1/3)) + 1
            return dim, dim, dim
        return gx, gy, gz
    
    def _coords_to_grid_idx(
        self, xyz_norm: torch.Tensor, gx: int, gy: int, gz: int
    ) -> torch.Tensor:
        """Convert normalized coords to flattened grid indices."""
        ix = (xyz_norm[..., 0] * gx).clamp(0, gx - 1).long()
        iy = (xyz_norm[..., 1] * gy).clamp(0, gy - 1).long()
        iz = (xyz_norm[..., 2] * gz).clamp(0, gz - 1).long()
        return ix * (gy * gz) + iy * gz + iz
    
    def _aggregate_to_grid(
        self, xyz: torch.Tensor, feat_bkc: torch.Tensor, 
        grid_idx: torch.Tensor, B: int, G: int, C: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter-add points into grid cells."""
        valid = xyz.abs().sum(-1) > 0
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand_as(grid_idx)
        linear_idx = (batch_ids * G + grid_idx)[valid]
        
        agg_xyz = torch.zeros(B * G, 3, device=device, dtype=xyz.dtype)
        agg_feat = torch.zeros(B * G, C, device=device, dtype=feat_bkc.dtype)
        agg_cnt = torch.zeros(B * G, device=device, dtype=torch.long)
        
        agg_xyz.index_add_(0, linear_idx, xyz[valid])
        agg_feat.index_add_(0, linear_idx, feat_bkc[valid])
        agg_cnt.index_add_(0, linear_idx, torch.ones_like(linear_idx))
        
        return agg_xyz.view(B, G, 3), agg_feat.view(B, G, C), agg_cnt.view(B, G)
    
    def _generate_grid_centers(
        self, gx: int, gy: int, gz: int,
        xyz_min: torch.Tensor, xyz_range: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Generate grid cell center coordinates: (B, G, 3)."""
        x = torch.linspace(0.5/gx, 1-0.5/gx, gx, device=device)
        y = torch.linspace(0.5/gy, 1-0.5/gy, gy, device=device)
        z = torch.linspace(0.5/gz, 1-0.5/gz, gz, device=device)
        
        gx_, gy_, gz_ = torch.meshgrid(x, y, z, indexing='ij')
        centers_norm = torch.stack([gx_.reshape(-1), gy_.reshape(-1), gz_.reshape(-1)], dim=-1)
        
        return centers_norm.unsqueeze(0) * xyz_range[:, 0:1] + xyz_min[:, 0:1]
    
    def get_expected_token_count(self, input_points: int) -> int:
        """Estimate output token count (rough approximation)."""
        if self.target_num_tokens is not None:
            return self.target_num_tokens
        total_stride = 1
        for s in self.stride:
            total_stride *= s
        return input_points // (total_stride * 4)
