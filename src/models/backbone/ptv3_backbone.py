"""
PTv3 Backbone Wrapper for SceneLeapUltra

Wraps the official Point Transformer V3 implementation to provide
a unified interface compatible with PointNet2 and the decoder models.
"""
import logging
import sys
from typing import Tuple

import torch
import torch.nn as nn

# Import PTv3 implementation
from .ptv3.ptv3 import PointTransformerV3


def convert_to_ptv3_pc_format(
    coords: torch.Tensor,
    feat: torch.Tensor = None,
    grid_size: float = 0.03,
    lengths: torch.Tensor = None,
    mask: torch.Tensor = None,
):
    """
    Convert point cloud to PTv3 input format.
    
    Args:
        coords: (B, N, 3) point coordinates
        feat: (B, N, F) point features, if None, uses ones(B, N, 1)
        grid_size: voxel grid size for sparse sampling
        
    Returns:
        data_dict: Dictionary with keys 'coord', 'feat', 'grid_size', 'offset'
    """
    B, N, _ = coords.shape
    device = coords.device
    
    if feat is None:
        feat = torch.ones(B, N, 1, device=device, dtype=coords.dtype)

    # 处理可变长度或掩码（如果提供）
    if lengths is not None or mask is not None:
        if mask is not None:
            lengths = mask.sum(dim=1).to(torch.long)
        else:
            lengths = lengths.to(torch.long)

        coord_list = []
        feat_list = []
        for b in range(B):
            Lb = int(lengths[b].item())
            if mask is not None:
                valid = mask[b]
                coord_list.append(coords[b][valid])
                feat_list.append(feat[b][valid])
            else:
                coord_list.append(coords[b, :Lb])
                feat_list.append(feat[b, :Lb])

        flat_coord = torch.cat(coord_list, dim=0).to(device)
        flat_feat = torch.cat(feat_list, dim=0).to(device)
        offset = lengths.cumsum(dim=0)
    else:
        flat_coord = coords.reshape(-1, 3).to(device)
        flat_feat = feat.reshape(-1, feat.shape[-1]).to(device)
        offset = torch.tensor([N], device=device, dtype=torch.long).repeat(B).cumsum(dim=0)

    return {
        'coord': flat_coord,
        'feat': flat_feat,
        'grid_size': grid_size,
        'offset': offset,
    }


def densify_ptv3_output(point_output, batch_size: int, device: torch.device):
    """
    Convert PTv3 sparse output back to dense format.
    
    Args:
        point_output: PTv3 Point output (sparse)
        batch_size: Original batch size
        device: Device to place tensors on
        
    Returns:
        xyz_dense: (B, K, 3) dense coordinates
        feat_dense: (B, C, K) dense features
    """
    # Extract sparse data
    coord = point_output['coord']  # (M, 3) where M is total sparse points
    feat = point_output['feat']    # (M, C)
    offset = point_output['offset'] # (B,)
    
    # Get dimensions
    C = feat.shape[1]
    
    # Convert offset to batch indices
    batch_idx = torch.zeros(coord.shape[0], dtype=torch.long, device=device)
    start_idx = 0
    for b in range(batch_size):
        end_idx = offset[b].item()
        batch_idx[start_idx:end_idx] = b
        start_idx = end_idx
    
    # Find max points per batch
    bincount = torch.bincount(batch_idx, minlength=batch_size)
    K = bincount.max().item()
    
    # Initialize dense tensors
    xyz_dense = torch.zeros(batch_size, K, 3, device=device, dtype=coord.dtype)
    feat_dense = torch.zeros(batch_size, C, K, device=device, dtype=feat.dtype)
    
    # Fill dense tensors
    for b in range(batch_size):
        mask = batch_idx == b
        pts = coord[mask]
        fts = feat[mask]
        n_pts = pts.shape[0]
        
        if n_pts > 0:
            xyz_dense[b, :n_pts] = pts
            feat_dense[b, :, :n_pts] = fts.t()
    
    return xyz_dense, feat_dense


class PTV3Backbone(nn.Module):
    """
    Point Transformer V3 Backbone wrapper.
    
    Provides a unified interface compatible with PointNet2:
    - Input: (B, N, C) where C = xyz(3) + optional_features
    - Output: (xyz, features) where xyz is (B, K, 3), features is (B, out_dim, K)
    
    Args:
        cfg: Configuration object with PTv3 parameters
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # Grid size for voxelization
        self.grid_size = getattr(cfg, 'grid_size', 0.03)
        
        # Variant: 'light', 'base', 'no_flash'
        self.variant = getattr(cfg, 'variant', 'light')
        
        # Flash attention
        self.use_flash_attention = getattr(cfg, 'use_flash_attention', True)
        
        # Output dimension
        self.output_dim = getattr(cfg, 'out_dim', 256)
        
        # Build PTv3 model based on variant
        self.model = self._build_ptv3_model(cfg)
        
        self.logger.info(
            f"Initialized PTv3Backbone: variant={self.variant}, "
            f"output_dim={self.output_dim}, grid_size={self.grid_size}, "
            f"flash_attn={self.use_flash_attention}"
        )
    
    def _build_ptv3_model(self, cfg):
        """Build PTv3 model based on configuration."""
        # Get configuration parameters
        encoder_channels = list(getattr(cfg, 'encoder_channels', [16, 32, 64, 128, 256]))
        encoder_depths = list(getattr(cfg, 'encoder_depths', [1, 1, 2, 2, 1]))
        mlp_ratio = getattr(cfg, 'mlp_ratio', 2)

        # Newly exposed configurable heads and patch sizes with safe defaults
        encoder_num_head = tuple(getattr(cfg, 'encoder_num_head', (2, 4, 8, 16, 16)))
        enc_patch_size = tuple(getattr(cfg, 'enc_patch_size', (1024, 1024, 1024, 1024, 1024)))
        
        # Calculate input channels from first feature dimension
        # Will be determined dynamically in forward pass
        # PTv3 needs at least 1 input channel (we use ones as fallback if no features)
        in_channels = max(1, getattr(cfg, 'input_feature_dim', 1))
        
        # Decoder channels (for segmentation mode)
        # For backbone, we use cls_mode=False to get per-point features
        # PTv3 decoder will append enc_channels[-1] automatically, so we don't include it
        # decoder_channels should be [C3, C2, C1, C0] (reverse of encoder, excluding last)
        dec_channels = list(getattr(cfg, 'decoder_channels', [256, 128, 64, 32]))
        dec_depths = list(getattr(cfg, 'decoder_depths', [1, 2, 1, 1]))
        dec_patch_size = tuple(getattr(cfg, 'dec_patch_size', (1024, 1024, 1024, 1024)))
        dec_num_head = tuple(getattr(cfg, 'dec_num_head', (4, 4, 8, 16)))
        
        # Note: PTv3 internally does: dec_channels = dec_channels + [enc_channels[-1]]
        # So final output will be enc_channels[-1] = 512
        self.logger.debug(
            f"PTv3 decoder config: dec_channels={dec_channels}, "
            f"final output will be enc_channels[-1]={encoder_channels[-1]}"
        )
        
        # Build model
        try:
            model = PointTransformerV3(
                in_channels=in_channels,
                order=["z", "z-trans"],
                stride=(2, 2, 2, 2),
                enc_depths=encoder_depths,
                enc_channels=encoder_channels,
                enc_num_head=encoder_num_head,
                enc_patch_size=enc_patch_size,
                dec_depths=dec_depths,
                dec_channels=dec_channels,
                dec_num_head=dec_num_head,
                dec_patch_size=dec_patch_size,
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
                cls_mode=False,  # Use segmentation mode to get per-point features
                pdnorm_bn=False,
                pdnorm_ln=False,
                pdnorm_decouple=True,
                pdnorm_adaptive=False,
                pdnorm_affine=True,
                pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
            )
            return model
        except Exception as e:
            self.logger.error(f"Failed to build PTv3 model: {e}")
            raise
    
    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PTv3 backbone.
        
        Args:
            pos: (B, N, C) point cloud tensor
                 C can be 3 (xyz), 4 (xyz+mask), 6 (xyz+rgb), or 7 (xyz+rgb+mask)
        
        Returns:
            xyz: (B, K, 3) sparse point coordinates
            features: (B, output_dim, K) sparse point features
        """
        B, N, C = pos.shape
        device = pos.device
        
        # Split coordinates and features
        coords = pos[..., :3]  # (B, N, 3)
        
        if C > 3:
            # Has additional features (rgb, mask, or both)
            feat = pos[..., 3:]  # (B, N, C-3)
        else:
            # Only xyz, use ones as placeholder features
            feat = torch.ones(B, N, 1, device=device, dtype=pos.dtype)
        
        # Convert to PTv3 format
        data_dict = convert_to_ptv3_pc_format(coords, feat, self.grid_size)
        
        # Forward through PTv3
        try:
            point_output = self.model(data_dict)
        except Exception as e:
            self.logger.error(f"PTv3 forward failed: {e}")
            raise
        
        # --- Logging & assertions for depth and sparse shape ---
        try:
            depth = None
            sparse_shape = None
            if isinstance(point_output, dict):
                depth = point_output.get('serialized_depth', None)
                sparse_shape = point_output.get('sparse_shape', None)
            # Fallback estimation when keys are missing
            if depth is None or sparse_shape is None:
                flat_coords = coords.reshape(-1, 3)
                mins = flat_coords.min(dim=0).values
                grid_coord = torch.div(flat_coords - mins, self.grid_size, rounding_mode="trunc").to(torch.int32)
                depth = int(grid_coord.max().item()).bit_length()
                # pad defaults to 96 in Point.sparsify()
                pad = 96
                grid_max = grid_coord.max(dim=0).values.to(torch.int32)
                sparse_shape = (grid_max + pad).tolist()
            # Log once per forward
            self.logger.info(
                f"[PTv3] grid_size={self.grid_size:.6f}, depth={depth}, sparse_shape={tuple(sparse_shape)} (B={B}, N={N})"
            )
            # Hard guard on depth
            if depth is not None and depth > 16:
                raise RuntimeError(
                    f"PTv3 serialized_depth({depth}) > 16. grid_size={self.grid_size} 可能过小或坐标单位不匹配（m/mm）。"
                )
        except Exception as log_exc:
            # Do not break forward for logging failures; report and continue
            self.logger.warning(f"[PTv3] depth/sparse_shape logging failed: {log_exc}")
        
        # Convert back to dense format
        xyz_dense, feat_dense = densify_ptv3_output(point_output, B, device)
        
        # Ensure output dimension matches expected
        if feat_dense.shape[1] != self.output_dim:
            self.logger.warning(
                f"PTv3 output dim ({feat_dense.shape[1]}) != expected ({self.output_dim}). "
                f"Adding projection layer."
            )
            # Add projection if needed (this shouldn't happen with correct config)
            if not hasattr(self, '_out_proj'):
                self._out_proj = nn.Linear(feat_dense.shape[1], self.output_dim).to(device)
            # Transpose for projection: (B, C, K) -> (B, K, C) -> project -> (B, K, C') -> (B, C', K)
            feat_dense = feat_dense.permute(0, 2, 1)
            feat_dense = self._out_proj(feat_dense)
            feat_dense = feat_dense.permute(0, 2, 1)
        
        return xyz_dense, feat_dense

