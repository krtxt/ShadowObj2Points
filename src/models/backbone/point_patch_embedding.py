"""
Point Cloud Patch Embedding Module.

A decoupled module that transforms point clouds into token sequences for DiT models.
Implements the complete voxelization pipeline: Point Cloud → Voxel Grid → Patches → Tokens.

This module is designed to be weakly coupled with the DiT-3D codebase while providing
a clean interface for point cloud preprocessing.
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


__all__ = ['PointCloudPatchEmbedding', 'PointPatchEmbeddingBackbone']


def _pytorch_avg_voxelize(
    features: Tensor,
    coords: Tensor,
    resolution: int
) -> Tensor:
    """
    Pure PyTorch implementation of average voxelization (fallback when CUDA not available).

    This function aggregates point features into a voxel grid using average pooling.
    Points that fall into the same voxel have their features averaged.

    Args:
        features: Point features of shape [B, C, N] where
            - B: batch size
            - C: number of channels (typically 3 for xyz)
            - N: number of points
        coords: Voxel coordinates (integer) of shape [B, 3, N] where values are in [0, resolution-1]
        resolution: Voxel grid resolution (R). Output will be [B, C, R, R, R]

    Returns:
        Voxelized features of shape [B, C, R, R, R]

    Note:
        - Uses scatter_add for accumulation followed by averaging
        - Empty voxels will have zero values
        - This is slower than the CUDA implementation but provides compatibility
    """
    B, C, N = features.shape
    device = features.device

    # Initialize voxel grid and count grid
    voxel_grid = torch.zeros(
        B, C, resolution, resolution, resolution,
        dtype=features.dtype, device=device
    )
    count_grid = torch.zeros(
        B, resolution, resolution, resolution,
        dtype=torch.float32, device=device
    )

    # Clamp coordinates to valid range
    coords = coords.long()
    coords = torch.clamp(coords, 0, resolution - 1)

    # Process each batch
    for b in range(B):
        # Get coordinates for this batch
        x_coords = coords[b, 0, :]  # [N]
        y_coords = coords[b, 1, :]  # [N]
        z_coords = coords[b, 2, :]  # [N]

        # Compute linear indices for scatter
        linear_idx = (
            x_coords * (resolution * resolution) +
            y_coords * resolution +
            z_coords
        )  # [N]

        # Accumulate features
        for c in range(C):
            feat_flat = features[b, c, :]  # [N]
            voxel_flat = voxel_grid[b, c].view(-1)  # [R^3]
            voxel_flat.scatter_add_(0, linear_idx, feat_flat)

        # Count points per voxel
        count_flat = count_grid[b].view(-1)  # [R^3]
        count_flat.scatter_add_(
            0, linear_idx, torch.ones_like(feat_flat, dtype=torch.float32)
        )

    # Average: divide by count (avoid division by zero)
    count_grid = count_grid.unsqueeze(1)  # [B, 1, R, R, R]
    mask = count_grid > 0
    # Use broadcasting instead of masked indexing for correct shapes
    voxel_grid = torch.where(
        mask,
        voxel_grid / count_grid.clamp(min=1.0),  # Avoid division by zero
        voxel_grid  # Keep zeros where count is 0
    )

    return voxel_grid


class PointCloudPatchEmbedding(nn.Module):
    """
    Point Cloud to Patch Embedding Module.

    Transforms point clouds into token sequences suitable for DiT (Diffusion Transformer) models.
    The pipeline follows: Point Cloud → Voxelization → 3D Conv Patching → Token Sequence.

    Key Features:
        - Single responsibility: (B, C, P) → (B, T, Hidden_Size)
        - Automatic CUDA detection with PyTorch fallback
        - Support for both auto-calculated and fixed token counts
        - Complete type annotations and shape documentation

    Args:
        in_channels: Number of input channels (default: 3 for xyz coordinates)
        hidden_size: Output token dimension (embedding dimension)
        voxel_size: Voxel grid resolution (default: 32)
        patch_size: Size of each 3D patch (default: 4)
        num_tokens: Fixed number of output tokens. If None, automatically calculated
            as (voxel_size // patch_size)^3 (default: None)
        normalize_points: Whether to normalize input point clouds (default: True)
            - True: Apply centering and scaling to [0, 1]
            - False: Assume points are already in [-1, 1] range
        use_cuda_voxelize: Whether to use CUDA voxelization if available (default: True)
            - Automatically falls back to PyTorch if CUDA not available
        bias: Whether to use bias in the Conv3d projection (default: True)

    Input Shape:
        - point_cloud: (B, C, P) where
            - B: batch size
            - C: number of channels (in_channels)
            - P: number of points (can vary)

    Output Shape:
        - tokens: (B, T, Hidden_Size) where
            - T: number of tokens (num_tokens or auto-calculated)
            - Hidden_Size: hidden_size parameter

    Example:
        >>> embed = PointCloudPatchEmbedding(
        ...     in_channels=3,
        ...     hidden_size=384,
        ...     voxel_size=32,
        ...     patch_size=4
        ... )
        >>> point_cloud = torch.randn(16, 3, 2048)  # (B, C, P)
        >>> tokens = embed(point_cloud)  # (B, 512, 384)
        >>> print(tokens.shape)
        torch.Size([16, 512, 384])

    Note:
        - Input point clouds are assumed to be in arbitrary scale; normalization is
          applied internally if normalize_points=True
        - Voxelization strategy: centering + scaling to [0, 1] + discretization
        - Patch extraction uses non-overlapping 3D convolution (stride = kernel_size)
        - voxel_size must be divisible by patch_size

    Warning:
        - CUDA voxelization provides significant speedup but requires compiled operators
        - If CUDA compilation fails, the module automatically falls back to PyTorch
        - The fallback implementation is slower but provides the same functionality
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 384,
        voxel_size: int = 32,
        patch_size: int = 4,
        num_tokens: Optional[int] = None,
        normalize_points: bool = True,
        use_cuda_voxelize: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Validate parameters
        if voxel_size % patch_size != 0:
            raise ValueError(
                f"voxel_size ({voxel_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.normalize_points = normalize_points
        self.use_cuda_voxelize = use_cuda_voxelize

        # Calculate grid size and number of patches
        self.grid_size = voxel_size // patch_size
        auto_num_tokens = self.grid_size ** 3
        self.grid_num_tokens = auto_num_tokens

        if num_tokens is not None and num_tokens != auto_num_tokens:
            warnings.warn(
                f"Specified num_tokens ({num_tokens}) differs from "
                f"auto-calculated value ({auto_num_tokens}). "
                f"This module always produces {auto_num_tokens} tokens; "
                f"please reshape in a wrapper if a different count is required."
            )

        self.num_tokens = auto_num_tokens

        # Try to import CUDA voxelization
        self._cuda_available = False
        if use_cuda_voxelize:
            try:
                from .functional import avg_voxelize
                self._avg_voxelize_cuda = avg_voxelize
                self._cuda_available = True
            except (ImportError, AttributeError) as e:
                warnings.warn(
                    f"CUDA voxelization not available ({e}), "
                    f"falling back to PyTorch implementation. "
                    f"This may be slower."
                )
                self._cuda_available = False

        # 3D Convolution for patch embedding
        # Non-overlapping patches: stride = kernel_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )

        # Store voxelization parameters
        self._voxel_normalize = normalize_points
        self._voxel_eps = 0.0

    def _normalize_coords(self, coords: Tensor) -> Tensor:
        """
        Normalize point cloud coordinates to [0, voxel_size-1] range.

        Strategy:
            1. Center: subtract mean (zero-centering)
            2. Scale: normalize to unit sphere
            3. Shift: move to [0, 1] range
            4. Scale to voxel resolution: multiply by voxel_size
            5. Clamp to valid range

        Args:
            coords: Point coordinates of shape [B, 3, P]

        Returns:
            Normalized coordinates in [0, voxel_size-1], shape [B, 3, P]
        """
        # Detach coordinates (don't need gradients for voxelization grid)
        coords = coords.detach()

        # Step 1: Zero-centering
        norm_coords = coords - coords.mean(dim=2, keepdim=True)

        if self._voxel_normalize:
            # Step 2-3: Normalize to [0, 1]
            # Find max distance from center
            max_dist = (
                norm_coords.norm(dim=1, keepdim=True)
                .max(dim=2, keepdim=True).values
            )
            norm_coords = (
                norm_coords / (max_dist * 2.0 + self._voxel_eps) + 0.5
            )
        else:
            # Alternative: assume coords in [-1, 1], map to [0, 1]
            norm_coords = (norm_coords + 1.0) / 2.0

        # Step 4-5: Scale to voxel resolution and clamp
        norm_coords = torch.clamp(
            norm_coords * self.voxel_size,
            0,
            self.voxel_size - 1
        )

        return norm_coords

    def _voxelize(
        self,
        features: Tensor,
        coords: Tensor
    ) -> Tensor:
        """
        Voxelize point cloud into regular grid.

        Uses CUDA implementation if available, otherwise falls back to PyTorch.

        Args:
            features: Point features, shape [B, C, P]
            coords: Point coordinates, shape [B, 3, P]

        Returns:
            Voxelized features, shape [B, C, R, R, R] where R = voxel_size
        """
        # Normalize coordinates to voxel grid
        norm_coords = self._normalize_coords(coords)

        # Discretize to integer coordinates
        vox_coords = torch.round(norm_coords).to(torch.int32)

        # Perform voxelization
        if self._cuda_available:
            # Use CUDA implementation
            voxel_grid = self._avg_voxelize_cuda(
                features, vox_coords, self.voxel_size
            )
        else:
            # Use PyTorch fallback
            voxel_grid = _pytorch_avg_voxelize(
                features, vox_coords, self.voxel_size
            )

        return voxel_grid

    def forward(self, point_cloud: Tensor) -> Tensor:
        """
        Transform point cloud to token sequence.

        Pipeline:
            1. Voxelization: Point Cloud [B, C, P] → Voxel Grid [B, C, R, R, R]
            2. Patch Embedding: Voxel Grid → Features [B, H, G, G, G]
            3. Flatten & Transpose: Features → Tokens [B, T, H]

        Args:
            point_cloud: Input point cloud of shape [B, C, P] where
                - B: batch size
                - C: number of channels (should equal in_channels)
                - P: number of points (arbitrary)

        Returns:
            Token sequence of shape [B, T, Hidden_Size] where
                - T: num_tokens
                - Hidden_Size: hidden_size

        Raises:
            ValueError: If input shape is invalid
        """
        if point_cloud.dim() != 3:
            raise ValueError(
                f"Expected 3D input (B, C, P), got shape {point_cloud.shape}"
            )

        B, C, P = point_cloud.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {C}"
            )

        # Step 1: Voxelization
        # Both features and coords are the point cloud itself (xyz coordinates)
        features = point_cloud
        coords = point_cloud

        voxel_grid = self._voxelize(features, coords)
        # Shape: [B, C, R, R, R] where R = voxel_size

        # Step 2: 3D Conv Patch Embedding
        voxel_grid = voxel_grid.float()
        patch_features = self.proj(voxel_grid)
        # Shape: [B, hidden_size, G, G, G] where G = grid_size

        # Step 3: Flatten and transpose to token sequence
        # Flatten spatial dimensions: [B, H, G, G, G] → [B, H, G^3]
        B_out, H, G, _, _ = patch_features.shape
        tokens = patch_features.flatten(2)  # [B, H, T]

        # Transpose: [B, H, T] → [B, T, H]
        tokens = tokens.transpose(1, 2)  # [B, T, H]

        return tokens

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f'in_channels={self.in_channels}, '
            f'hidden_size={self.hidden_size}, '
            f'voxel_size={self.voxel_size}, '
            f'patch_size={self.patch_size}, '
            f'num_tokens={self.num_tokens}, '
            f'grid_size={self.grid_size}, '
            f'normalize_points={self.normalize_points}, '
            f'cuda_voxelize={self._cuda_available}'
        )


class PointPatchEmbeddingBackbone(nn.Module):
    """
    轻量点云骨干：使用体素化 + 3D 卷积补丁生成 Token 表征。

    - 输入：`(B, N, C)` 点云（默认 C=3，仅 xyz）
    - 输出：`(xyz_tokens, features)` 其中
        * `xyz_tokens`: `(B, K, 3)`，为每个补丁的归一化空间位置
        * `features`: `(B, D, K)`，为补丁特征（D = hidden_size）

    该骨干封装 `PointCloudPatchEmbedding`，为解码器提供与 PointNet/PointNeXt/PTv3
    相同的接口形式，便于在同一基准测试脚本中比较。
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.in_channels = getattr(cfg, 'in_channels', 3)
        self.hidden_size = getattr(cfg, 'hidden_size', 384)
        self.voxel_size = getattr(cfg, 'voxel_size', 32)
        self.patch_size = getattr(cfg, 'patch_size', 4)
        self.normalize_points = getattr(cfg, 'normalize_points', True)
        self.use_cuda_voxelize = getattr(cfg, 'use_cuda_voxelize', True)
        self.bias = getattr(cfg, 'bias', True)

        self.embedding = PointCloudPatchEmbedding(
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            voxel_size=self.voxel_size,
            patch_size=self.patch_size,
            num_tokens=None,
            normalize_points=self.normalize_points,
            use_cuda_voxelize=self.use_cuda_voxelize,
            bias=self.bias,
        )

        # 对齐输出通道与 token 数
        self.output_dim = getattr(cfg, 'out_dim', self.hidden_size)
        self.base_num_tokens = self.embedding.num_tokens
        self.grid_size = self.embedding.grid_size

        # 目标 token 数，默认等于基础数量，可通过自适应池化压缩
        self.target_num_tokens = getattr(cfg, 'target_num_tokens', self.base_num_tokens)
        if self.target_num_tokens <= 0:
            raise ValueError("target_num_tokens 必须为正整数。")
        if self.target_num_tokens != self.base_num_tokens:
            self.token_pool = nn.AdaptiveAvgPool1d(self.target_num_tokens)
            self.patch_center_pool = nn.AdaptiveAvgPool1d(self.target_num_tokens)
        else:
            self.token_pool = None
            self.patch_center_pool = None
        self.num_tokens = self.target_num_tokens

        patch_centers = self._build_patch_centers()
        # 注册为 buffer 以便随模型移动设备
        self.register_buffer('patch_centers', patch_centers, persistent=False)

        # 可学习位置编码
        self.use_positional_embedding = getattr(cfg, 'learnable_positional_embedding', True)
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(torch.zeros(
                1, self.target_num_tokens, self.hidden_size
            ))
            init.trunc_normal_(self.positional_embedding, std=0.02)
        else:
            self.register_parameter('positional_embedding', None)

        # Transformer 编码层配置
        self.transformer_layers = getattr(cfg, 'transformer_layers', 0)
        self.transformer_heads = getattr(cfg, 'transformer_heads', max(1, self.hidden_size // 64))
        self.transformer_ffn_ratio = getattr(cfg, 'transformer_ffn_ratio', 4.0)
        self.transformer_dropout = getattr(cfg, 'transformer_dropout', 0.0)

        if self.transformer_layers > 0:
            ffn_dim = max(self.hidden_size, int(self.hidden_size * self.transformer_ffn_ratio))
            try:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.transformer_heads,
                    dim_feedforward=ffn_dim,
                    dropout=self.transformer_dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
            except TypeError:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.transformer_heads,
                    dim_feedforward=ffn_dim,
                    dropout=self.transformer_dropout,
                    activation='gelu',
                    batch_first=True,
                )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.transformer_layers,
            )
        else:
            self.transformer = None

        # Token MLP 投影
        mlp_ratio = getattr(cfg, 'mlp_ratio', 4.0)
        proj_hidden = max(self.hidden_size, int(self.hidden_size * mlp_ratio))
        proj_dropout = getattr(cfg, 'mlp_dropout', 0.0)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, proj_hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_hidden, self.output_dim),
        )

    def _build_patch_centers(self) -> Tensor:
        """
        生成补丁中心坐标（归一化到 [-1, 1] 区间）。

        注意：PointCloudPatchEmbedding 会对每个 batch 做居中与尺度归一化，
        因此这里返回的是归一化空间的网格中心，便于保持接口一致性。
        """
        grid_range = torch.arange(self.grid_size, dtype=torch.float32)
        # (K, 3) 网格下标组合
        mesh = torch.stack(torch.meshgrid(
            grid_range, grid_range, grid_range, indexing='ij'
        ), dim=-1).reshape(-1, 3)

        # 体素坐标系 -> [-1, 1] 范围的归一化中心
        scale = float(self.patch_size) / float(self.voxel_size)
        centers = (mesh + 0.5) * scale  # 映射到 [0, 1]
        centers = centers * 2.0 - 1.0   # 映射到 [-1, 1]
        return centers

    def _pooled_patch_centers(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        centers = self.patch_centers.to(device=device, dtype=dtype)  # (K, 3)
        centers = centers.t().unsqueeze(0)  # (1, 3, K)
        if self.patch_center_pool is not None:
            centers = self.patch_center_pool(centers)
        centers = centers.squeeze(0).t().contiguous()  # (target_K, 3)
        return centers

    def forward(self, pointcloud: Tensor) -> Tuple[Tensor, Tensor]:
        if pointcloud.dim() != 3:
            raise ValueError(
                f"PointPatchEmbeddingBackbone 期望输入形状为 (B, N, C)，"
                f"实际得到 {tuple(pointcloud.shape)}"
            )

        if pointcloud.shape[-1] != self.in_channels:
            raise ValueError(
                f"输入通道数与配置不匹配：期望 {self.in_channels}，"
                f"实际 {pointcloud.shape[-1]}"
            )

        # (B, N, C) -> (B, C, N)
        features = pointcloud.transpose(1, 2).contiguous()
        tokens = self.embedding(features)         # (B, K_base, D)

        if self.token_pool is not None:
            tokens = tokens.transpose(1, 2)       # (B, D, K_base)
            tokens = self.token_pool(tokens)      # (B, D, K_target)
            tokens = tokens.transpose(1, 2)       # (B, K_target, D)

        if self.use_positional_embedding and self.positional_embedding is not None:
            tokens = tokens + self.positional_embedding.to(tokens.dtype)

        if self.transformer is not None:
            tokens = self.transformer(tokens)     # (B, K_target, D)

        tokens = self.projector(tokens)           # (B, K_target, out_dim)
        patch_features = tokens.transpose(1, 2)   # (B, out_dim, K_target)

        # 广播补丁中心到 batch 维度
        patch_xyz_single = self._pooled_patch_centers(
            device=pointcloud.device,
            dtype=pointcloud.dtype,
        )
        patch_xyz = patch_xyz_single.unsqueeze(0).expand(pointcloud.shape[0], -1, -1)

        return patch_xyz, patch_features
