"""
PTv3 Sparse Token Encoder

Extract sparse scene tokens from point clouds for DiT conditioning.

Supported token strategies:
- last_layer: Use encoder output directly
- fps: Farthest Point Sampling
- grid: Aggregate into regular grid cells
- learned/perceiver: Cross-attention with learnable queries
- multiscale: Fuse features from multiple encoder stages
- fourier_enhanced: FPS + Fourier positional encoding + surface normals

Key configs:
- grid_size: voxelization precision (smaller = finer)
- stride: encoder downsampling (controls final sparsity)
- token_strategy: how to select output tokens
"""

from __future__ import annotations

import logging
import math
import os
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.pointnet2_utils import furthest_point_sample

from .ptv3.ptv3 import Point, PointTransformerV3
from .ptv3_backbone import convert_to_ptv3_pc_format

# =============================================================================
# Constants
# =============================================================================

VALID_STRATEGIES = (
    "last_layer",
    "fps",
    "grid",
    "learned",
    "multiscale",
    "perceiver",
    "fourier_enhanced",
)


# =============================================================================
# Tokenizer Modules
# =============================================================================


class LearnedTokenizer(nn.Module):
    """Cross-attention based tokenizer with learnable query tokens (Perceiver-style)."""

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


class FourierEnhancedTokenizer(nn.Module):
    """Fourier-enhanced tokenizer combining PTv3 features with explicit geometry.

    Token construction:
        Token = MLP([Feat_PTv3, Fourier(xyz), Normal(nx, ny, nz)])

    Benefits:
    - Fourier(xyz): High-frequency positional encoding for precise localization
    - Normals: Surface orientation for anti-penetration learning in grasping

    Args:
        ptv3_dim: Dimension of PTv3 output features
        output_dim: Output token dimension
        num_frequencies: Number of Fourier frequency bands (default: 10)
        use_normals: Whether to include surface normals (default: True)
        mlp_ratio: MLP hidden layer expansion ratio (default: 2)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        ptv3_dim: int,
        output_dim: int,
        num_frequencies: int = 10,
        use_normals: bool = True,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ptv3_dim = ptv3_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies
        self.use_normals = use_normals

        # Fourier encoding output: xyz + sin/cos for each frequency and dimension
        # = 3 + 3 * 2 * num_frequencies (e.g., 63 for 10 frequencies)
        self.fourier_dim = 3 + 3 * 2 * num_frequencies

        # Log-spaced frequency bands (like NeRF)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

        # Input: PTv3 features + Fourier(xyz) + optionally normals
        self.input_dim = ptv3_dim + self.fourier_dim + (3 if use_normals else 0)

        # MLP: concat -> hidden -> output
        hidden_dim = output_dim * mlp_ratio
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fourier_encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """Apply Fourier positional encoding.

        Args:
            xyz: (B, K, 3) coordinates
        Returns:
            (B, K, fourier_dim) encoded features
        """
        # Cast freq_bands to input dtype for AMP compatibility
        freq = self.freq_bands.to(xyz.dtype).view(1, 1, 1, -1)
        xyz_exp = xyz.unsqueeze(-1)  # (B, K, 3, 1)
        scaled = xyz_exp * freq * math.pi  # (B, K, 3, num_freq)

        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        freq_feat = torch.cat([sin_feat, cos_feat], dim=-1)  # (B, K, 3, 2*num_freq)
        freq_feat = freq_feat.reshape(xyz.shape[0], xyz.shape[1], -1)

        return torch.cat([xyz, freq_feat], dim=-1)

    def forward(
        self,
        ptv3_feat: torch.Tensor,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            ptv3_feat: (B, K, ptv3_dim) PTv3 output features
            xyz: (B, K, 3) point coordinates
            normals: (B, K, 3) optional surface normals
        Returns:
            (B, K, output_dim) enhanced tokens
        """
        fourier_xyz = self._fourier_encode(xyz)
        features = [ptv3_feat, fourier_xyz]

        if self.use_normals:
            if normals is not None:
                normals = F.normalize(normals, dim=-1, eps=1e-6)
            else:
                B, K, _ = xyz.shape
                normals = torch.zeros(B, K, 3, device=xyz.device, dtype=xyz.dtype)
            features.append(normals)

        return self.projection(torch.cat(features, dim=-1))


# =============================================================================
# Main Encoder
# =============================================================================


class PTv3SparseEncoder(nn.Module):
    """PTv3-based sparse token extractor.

    Outputs:
        xyz: (B, K, 3) sparse coordinates
        features: (B, D, K) or (B, K, D) if tokens_last=True
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

        # ---------------------------------------------------------------------
        # Core configuration
        # ---------------------------------------------------------------------
        self.grid_size = getattr(cfg, "grid_size", 0.003)
        self.target_num_tokens = target_num_tokens or getattr(cfg, "target_num_tokens", 128)
        self.output_dim = getattr(cfg, "out_dim", 256)
        self.tokens_last = getattr(cfg, "tokens_last", False)
        self.grid_resolution = tuple(getattr(cfg, "grid_resolution", (8, 8, 8)))

        # Strategy
        self.token_strategy = token_strategy or getattr(cfg, "token_strategy", "last_layer")
        if self.token_strategy not in VALID_STRATEGIES:
            raise ValueError(f"token_strategy must be one of {VALID_STRATEGIES}")

        # Debug
        self.debug_stats = bool(
            getattr(cfg, "debug_stats", False) or int(os.environ.get("PTV3_DEBUG_STATS", "0"))
        )
        self.debug_stats_steps = int(getattr(cfg, "debug_stats_steps", 5))
        self._debug_calls = 0
        self._log_gate = False

        # ---------------------------------------------------------------------
        # Encoder architecture
        # ---------------------------------------------------------------------
        enc_channels = list(getattr(cfg, "encoder_channels", [32, 64, 128, 256]))
        enc_depths = list(getattr(cfg, "encoder_depths", [1, 1, 2, 2]))
        enc_num_head = tuple(getattr(cfg, "encoder_num_head", (2, 4, 8, 16)))
        enc_patch_size = tuple(getattr(cfg, "enc_patch_size", (1024, 1024, 1024, 1024)))
        stride = tuple(getattr(cfg, "stride", (2, 2, 2)))

        num_stages = len(enc_channels)
        assert len(stride) == num_stages - 1, (
            f"stride length ({len(stride)}) must be num_stages - 1 ({num_stages - 1})"
        )

        self.encoder_channels = enc_channels
        self.num_stages = num_stages
        self.stride = stride

        # Scene normals support
        self.use_scene_normals = getattr(cfg, "use_scene_normals", False)

        # Input feature dimension with safety check
        input_feature_dim = max(1, getattr(cfg, "input_feature_dim", 1))
        if self.use_scene_normals and input_feature_dim < 3:
            self.logger.warning(
                f"use_scene_normals=True but input_feature_dim={input_feature_dim} < 3. "
                f"Auto-bumping to 3. Set input_feature_dim=3 in config to suppress this."
            )
            input_feature_dim = 3

        # Build PTv3 backbone
        self.model = self._build_ptv3(
            in_channels=input_feature_dim,
            enc_channels=enc_channels,
            enc_depths=enc_depths,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            stride=stride,
            mlp_ratio=getattr(cfg, "mlp_ratio", 2),
            enable_flash=getattr(cfg, "use_flash_attention", True),
        )

        # Output projection
        self.out_proj = (
            nn.Linear(enc_channels[-1], self.output_dim)
            if enc_channels[-1] != self.output_dim
            else None
        )

        # ---------------------------------------------------------------------
        # Strategy-specific modules
        # ---------------------------------------------------------------------
        self.learned_tokenizer = None
        self.fourier_tokenizer = None
        self.stage_projections = nn.ModuleDict()

        if self.token_strategy in ("learned", "perceiver"):
            self.learned_tokenizer = LearnedTokenizer(self.target_num_tokens, self.output_dim)

        elif self.token_strategy == "multiscale":
            for i, ch in enumerate(enc_channels[1:], start=1):
                if ch != self.output_dim:
                    self.stage_projections[f"stage{i}"] = nn.Linear(ch, self.output_dim)

        elif self.token_strategy == "fourier_enhanced":
            self.fourier_num_frequencies = int(getattr(cfg, "fourier_num_frequencies", 10))
            self.fourier_use_normals = bool(getattr(cfg, "fourier_use_normals", True))
            self.fourier_mlp_ratio = int(getattr(cfg, "fourier_mlp_ratio", 2))
            self.fourier_dropout = float(getattr(cfg, "fourier_dropout", 0.1))
            self.fourier_tokenizer = FourierEnhancedTokenizer(
                ptv3_dim=self.output_dim,
                output_dim=self.output_dim,
                num_frequencies=self.fourier_num_frequencies,
                use_normals=self.fourier_use_normals,
                mlp_ratio=self.fourier_mlp_ratio,
                dropout=self.fourier_dropout,
            )

        self._log_init()

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _build_ptv3(
        self,
        in_channels: int,
        enc_channels: list,
        enc_depths: list,
        enc_num_head: tuple,
        enc_patch_size: tuple,
        stride: tuple,
        mlp_ratio: int,
        enable_flash: bool,
    ) -> PointTransformerV3:
        """Build PTv3 backbone."""
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

    def _log_init(self):
        parts = [
            f"PTv3SparseEncoder: grid={self.grid_size}, dim={self.output_dim}",
            f"stride={self.stride}, tokens={self.target_num_tokens}",
            f"strategy={self.token_strategy}, normals={self.use_scene_normals}",
        ]
        if self.token_strategy == "fourier_enhanced":
            parts.append(f"fourier_freq={self.fourier_num_frequencies}")
            parts.append(f"fourier_normals={self.fourier_use_normals}")
        self.logger.info(", ".join(parts))

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(
        self, pos: torch.Tensor, return_full_res: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract sparse scene tokens.

        Args:
            pos: (B, N, C) point cloud
                 C=3: xyz only
                 C=6: xyz + normals (for fourier_enhanced)
            return_full_res: store debug info in self.debug_last

        Returns:
            xyz: (B, K, 3) sparse coordinates
            features: (B, D, K) or (B, K, D) if tokens_last=True
        """
        B, N, C = pos.shape
        device = pos.device

        # Parse input
        coords = pos[..., :3]
        normals = pos[..., 3:6] if C >= 6 else None
        feat = pos[..., 3:] if C > 3 else torch.ones(B, N, 1, device=device, dtype=pos.dtype)

        # Convert to PTv3 format
        data_dict = convert_to_ptv3_pc_format(coords, feat, self.grid_size)

        # Debug logging
        log_debug = self._debug_should_log()
        self._log_gate = log_debug
        if log_debug:
            self._log_dense_input(coords, data_dict)

        # Run encoder and apply strategy
        if self.token_strategy == "multiscale":
            xyz_out, feat_out = self._strategy_multiscale(B, device, data_dict)
        else:
            xyz_sparse, feat_sparse = self._run_encoder(data_dict, B, device)
            if log_debug:
                self._log_batched_stats(xyz_sparse, feat_sparse, "after_encoder")
            xyz_out, feat_out = self._apply_strategy(xyz_sparse, feat_sparse, coords, normals)

        # Format output
        if self.tokens_last:
            feat_out = feat_out.permute(0, 2, 1)

        if return_full_res and self.token_strategy != "multiscale":
            self._store_debug(xyz_sparse, feat_sparse)

        self._log_gate = False
        return xyz_out, feat_out

    def _run_encoder(
        self, data_dict: dict, B: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run PTv3 encoder and convert to batched format."""
        point = Point(data_dict)
        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
        point.sparsify()

        self._log_sparse_state(point, "after_sparsify", B)
        self._log_spconv_tensor(getattr(point, "sparse_conv_feat", None), "after_sparsify")

        point = self.model.embedding(point)
        self._log_spconv_tensor(getattr(point, "sparse_conv_feat", None), "after_embedding")

        try:
            point = self.model.enc(point)
        except Exception:
            self._log_gate = True
            self._log_spconv_tensor(getattr(point, "sparse_conv_feat", None), "enc_fail")
            self._log_sparse_state(point, "enc_fail", B)
            raise

        self._log_sparse_state(point, "after_encoder", B)

        xyz_sparse, feat_sparse = self._to_batch_format(
            point.coord, point.feat, point.offset, B, device
        )

        if self.out_proj is not None:
            feat_sparse = self._project_features(feat_sparse, self.out_proj)

        return xyz_sparse, feat_sparse

    def _apply_strategy(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        orig_coords: torch.Tensor,
        orig_normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply token selection strategy."""
        dispatch = {
            "last_layer": lambda: self._strategy_last_layer(xyz, feat),
            "fps": lambda: self._fps_sample(xyz, feat, self.target_num_tokens),
            "grid": lambda: self._strategy_grid(xyz, feat),
            "learned": lambda: self._strategy_learned(xyz, feat),
            "perceiver": lambda: self._strategy_learned(xyz, feat),
            "fourier_enhanced": lambda: self._strategy_fourier_enhanced(
                xyz, feat, orig_coords, orig_normals
            ),
        }
        return dispatch[self.token_strategy]()

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

        gx, gy, gz = self._compute_grid_dims()
        G = gx * gy * gz

        # Normalize to [0, 1]
        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_range = xyz.max(dim=1, keepdim=True)[0] - xyz_min + 1e-6
        xyz_norm = (xyz - xyz_min) / xyz_range

        # Grid indices
        grid_idx = self._coords_to_grid_idx(xyz_norm, gx, gy, gz)
        self._last_grid_idx = grid_idx

        # Aggregate
        grid_xyz, grid_feat, grid_count = self._aggregate_to_grid(
            xyz, feat.permute(0, 2, 1), grid_idx, B, G, C, device
        )

        # Average
        grid_count_safe = grid_count.clamp_min(1).float()
        grid_xyz = grid_xyz / grid_count_safe.unsqueeze(-1)
        grid_feat = (grid_feat / grid_count_safe.unsqueeze(-1)).permute(0, 2, 1)

        # Handle empty cells
        empty = grid_count == 0
        if empty.any():
            centers = self._generate_grid_centers(gx, gy, gz, xyz_min, xyz_range, device)
            grid_xyz[empty] = centers[empty]

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
                key = f"stage{idx}"
                if key in self.stage_projections:
                    feat_s = self._project_features(feat_s, self.stage_projections[key])
                stage_coords.append(xyz_s)
                stage_feats.append(feat_s)

        # Sample equally from each stage
        tokens_per = self.target_num_tokens // len(stage_coords)
        sampled = [self._fps_sample(x, f, tokens_per) for x, f in zip(stage_coords, stage_feats)]

        multi_xyz = torch.cat([s[0] for s in sampled], dim=1)
        multi_feat = torch.cat([s[1] for s in sampled], dim=2)

        if multi_xyz.size(1) != self.target_num_tokens:
            return self._fps_sample(multi_xyz, multi_feat, self.target_num_tokens)
        return multi_xyz, multi_feat

    def _strategy_fourier_enhanced(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        orig_coords: torch.Tensor,
        orig_normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fourier-enhanced strategy with explicit geometry encoding.

        Process:
        1. FPS sample from PTv3 output
        2. Find nearest neighbors in original point cloud for normals
        3. Apply Fourier encoding + MLP

        Args:
            xyz: (B, K_enc, 3) sparse xyz from encoder
            feat: (B, C, K_enc) sparse features from encoder
            orig_coords: (B, N, 3) original coordinates
            orig_normals: (B, N, 3) original normals (optional)

        Returns:
            token_xyz: (B, target_num_tokens, 3)
            token_feat: (B, C, target_num_tokens)
        """
        # FPS sample
        token_xyz, token_feat_raw = self._fps_sample(xyz, feat, self.target_num_tokens)
        token_feat_bkc = token_feat_raw.permute(0, 2, 1)  # (B, K, C)

        # Get normals via nearest neighbor lookup
        sampled_normals = None
        if self.fourier_tokenizer.use_normals and orig_normals is not None:
            # cdist: (B, K, N) pairwise distances
            dists = torch.cdist(token_xyz, orig_coords)
            nn_idx = dists.argmin(dim=-1)  # (B, K)
            nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, 3)
            sampled_normals = torch.gather(orig_normals, dim=1, index=nn_idx_exp)

        # Apply Fourier tokenizer
        enhanced_feat = self.fourier_tokenizer(
            ptv3_feat=token_feat_bkc,
            xyz=token_xyz,
            normals=sampled_normals,
        )

        return token_xyz, enhanced_feat.permute(0, 2, 1)

    # =========================================================================
    # Sampling & Padding Utilities
    # =========================================================================

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

    def _to_batch_format(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        offset: torch.Tensor,
        B: int,
        device: torch.device,
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

    @staticmethod
    def _project_features(feat: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Project features: (B, C, K) -> (B, C', K)."""
        return layer(feat.permute(0, 2, 1)).permute(0, 2, 1)

    # =========================================================================
    # Grid Utilities
    # =========================================================================

    def _compute_grid_dims(self) -> Tuple[int, int, int]:
        """Get grid dimensions, adjusting to match target if needed."""
        gx, gy, gz = self.grid_resolution
        if gx * gy * gz != self.target_num_tokens:
            dim = int(self.target_num_tokens ** (1 / 3)) + 1
            return dim, dim, dim
        return gx, gy, gz

    def _coords_to_grid_idx(
        self, xyz_norm: torch.Tensor, gx: int, gy: int, gz: int
    ) -> torch.Tensor:
        """Convert normalized coords [0,1] to flattened grid indices."""
        ix = (xyz_norm[..., 0] * gx).clamp(0, gx - 1).long()
        iy = (xyz_norm[..., 1] * gy).clamp(0, gy - 1).long()
        iz = (xyz_norm[..., 2] * gz).clamp(0, gz - 1).long()
        return ix * (gy * gz) + iy * gz + iz

    def _aggregate_to_grid(
        self,
        xyz: torch.Tensor,
        feat_bkc: torch.Tensor,
        grid_idx: torch.Tensor,
        B: int,
        G: int,
        C: int,
        device: torch.device,
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
        self,
        gx: int,
        gy: int,
        gz: int,
        xyz_min: torch.Tensor,
        xyz_range: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate grid cell center coordinates: (B, G, 3)."""
        x = torch.linspace(0.5 / gx, 1 - 0.5 / gx, gx, device=device)
        y = torch.linspace(0.5 / gy, 1 - 0.5 / gy, gy, device=device)
        z = torch.linspace(0.5 / gz, 1 - 0.5 / gz, gz, device=device)

        gx_, gy_, gz_ = torch.meshgrid(x, y, z, indexing="ij")
        centers_norm = torch.stack(
            [gx_.reshape(-1), gy_.reshape(-1), gz_.reshape(-1)], dim=-1
        )

        return centers_norm.unsqueeze(0) * xyz_range[:, 0:1] + xyz_min[:, 0:1]

    # =========================================================================
    # Debug Utilities
    # =========================================================================

    def _debug_should_log(self) -> bool:
        if not self.debug_stats or self._debug_calls >= self.debug_stats_steps:
            return False
        self._debug_calls += 1
        return True

    def _store_debug(self, xyz_sparse, feat_sparse):
        self.debug_last = {"path": self.token_strategy}
        if xyz_sparse is not None:
            self.debug_last["xyz_sparse_full"] = xyz_sparse
            self.debug_last["feat_sparse_full"] = feat_sparse
        if hasattr(self, "_last_grid_idx"):
            self.debug_last["grid_idx"] = self._last_grid_idx

    def _log_dense_input(self, coords: torch.Tensor, data_dict: dict):
        """Log dense input stats."""
        if not self._log_gate:
            return
        try:
            B, N, _ = coords.shape
            counts = torch.diff(
                data_dict["offset"],
                prepend=torch.zeros(1, device=coords.device, dtype=data_dict["offset"].dtype),
            )
            self.logger.info(
                f"[PTv3 debug][input] B={B}, N={N}, dtype={coords.dtype}, "
                f"grid={data_dict.get('grid_size', self.grid_size)}, "
                f"counts={counts.min().item()}/{counts.float().mean():.1f}/{counts.max().item()}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3 debug] input log failed: {e}")

    def _log_sparse_state(self, point: Point, stage: str, batch_size: int):
        """Log sparse stats at intermediate stages."""
        if not self._log_gate:
            return
        try:
            coord = getattr(point, "coord", None)
            feat = getattr(point, "feat", None)
            offset = getattr(point, "offset", None)

            counts = None
            if offset is not None:
                counts = torch.diff(
                    offset, prepend=torch.zeros(1, device=offset.device, dtype=offset.dtype)
                )

            self.logger.info(
                f"[PTv3 debug][{stage}] "
                f"coord={tuple(coord.shape) if coord is not None else None}, "
                f"feat={tuple(feat.shape) if feat is not None else None}, "
                f"counts={counts.min().item()}/{counts.float().mean():.1f}/{counts.max().item() if counts is not None else None}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3 debug] {stage} log failed: {e}")

    def _log_batched_stats(self, xyz: torch.Tensor, feat: torch.Tensor, stage: str):
        """Log batched format stats."""
        if not self._log_gate:
            return
        try:
            valid = xyz.abs().sum(-1) > 0
            counts = valid.sum(dim=1)
            self.logger.info(
                f"[PTv3 debug][{stage}] xyz={tuple(xyz.shape)}, feat={tuple(feat.shape)}, "
                f"valid={counts.min().item()}/{counts.float().mean():.1f}/{counts.max().item()}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3 debug] {stage} log failed: {e}")

    def _log_spconv_tensor(self, tensor, stage: str):
        """Log spconv tensor internals."""
        if not self._log_gate or tensor is None:
            return
        try:
            feat = getattr(tensor, "features", None)
            idx = getattr(tensor, "indices", None)
            spatial = getattr(tensor, "spatial_shape", None)
            bs = getattr(tensor, "batch_size", None)

            self.logger.info(
                f"[PTv3 debug][{stage}] feat={tuple(feat.shape) if feat is not None else None}, "
                f"idx={tuple(idx.shape) if idx is not None else None}, "
                f"spatial={tuple(spatial) if spatial is not None else None}, batch={bs}"
            )
        except Exception as e:
            self.logger.warning(f"[PTv3 debug] {stage} spconv log failed: {e}")

    # =========================================================================
    # Public Utilities
    # =========================================================================

    def get_expected_token_count(self, input_points: int) -> int:
        """Estimate output token count."""
        if self.target_num_tokens is not None:
            return self.target_num_tokens
        total_stride = 1
        for s in self.stride:
            total_stride *= s
        return input_points // (total_stride * 4)
