"""
Point Transformer Backbone Wrapper for SceneLeapUltra

Wraps the Point Transformer implementation to provide a unified interface
compatible with PointNet2 and PTv3 backbones.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

from .point_transformer import PointTransformerEnc


def convert_to_pxo_format(pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert dense point cloud to Point Transformer's PXO format.
    
    Args:
        pos: (B, N, C) point cloud tensor where C = xyz(3) + optional_features
        
    Returns:
        p: (B*N, 3) - flattened point coordinates
        x: (B*N, C) - flattened point features
        o: (B,) - batch offsets (cumsum of point counts)
    """
    B, N, C = pos.shape
    device = pos.device
    
    # Extract xyz and features
    xyz = pos[..., :3]  # (B, N, 3)
    
    if C > 3:
        # Has additional features (rgb, mask, or both)
        feat = pos[..., 3:]  # (B, N, C-3)
    else:
        # Only xyz, use xyz as features (Point Transformer expects features)
        feat = xyz  # (B, N, 3)
    
    # Flatten batch dimension
    p = xyz.reshape(-1, 3)  # (B*N, 3)
    x = torch.cat([xyz, feat], dim=-1).reshape(-1, 3 + feat.shape[-1])  # (B*N, 3+C-3) = (B*N, C)
    
    # Compute batch offsets (cumulative sum of point counts)
    o = torch.tensor([N * (i + 1) for i in range(B)], device=device, dtype=torch.int32)
    
    return p, x, o


def densify_pxo_output(p: torch.Tensor, x: torch.Tensor, o: torch.Tensor, 
                       batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Point Transformer's sparse PXO output back to dense format.
    
    Args:
        p: (M, 3) - sparse point coordinates
        x: (M, C) - sparse point features
        o: (B,) - batch offsets (cumsum)
        batch_size: Original batch size
        
    Returns:
        xyz_dense: (B, K, 3) - dense coordinates
        feat_dense: (B, C, K) - dense features
    """
    device = p.device
    C = x.shape[1]
    
    # Split by batch
    xyz_list = []
    feat_list = []
    start_idx = 0
    
    for b in range(batch_size):
        end_idx = o[b].item()
        xyz_list.append(p[start_idx:end_idx])  # (K_b, 3)
        feat_list.append(x[start_idx:end_idx])  # (K_b, C)
        start_idx = end_idx
    
    # Find max points per batch for padding
    K = max(xyz.shape[0] for xyz in xyz_list)
    
    # Initialize dense tensors with zeros
    xyz_dense = torch.zeros(batch_size, K, 3, device=device, dtype=p.dtype)
    feat_dense = torch.zeros(batch_size, C, K, device=device, dtype=x.dtype)
    
    # Fill dense tensors
    for b in range(batch_size):
        n_pts = xyz_list[b].shape[0]
        if n_pts > 0:
            xyz_dense[b, :n_pts] = xyz_list[b]
            # Transpose features to (C, K) format
            feat_dense[b, :, :n_pts] = feat_list[b].t()
    
    return xyz_dense, feat_dense


class PointTransformerBackbone(nn.Module):
    """
    Point Transformer Backbone wrapper.
    
    Provides a unified interface compatible with PointNet2 and PTv3:
    - Input: (B, N, C) where C = xyz(3) + optional_features
    - Output: (xyz, features) where xyz is (B, K, 3), features is (B, out_dim, K)
    
    Args:
        cfg: Configuration object with Point Transformer parameters
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # Get configuration parameters
        self.c = getattr(cfg, 'c', 6)  # Input channels (xyz + rgb)
        self.num_points = getattr(cfg, 'num_points', 32768)
        
        # Output dimension (Point Transformer outputs 512 channels)
        self.output_dim = getattr(cfg, 'out_dim', 512)
        
        # Build Point Transformer encoder
        self.model = PointTransformerEnc(
            block=None,  # Will use default PointTransformerBlock
            blocks=[2, 3, 4, 6, 3],  # Default architecture
            c=self.c,
            num_points=self.num_points
        )
        
        self.logger.info(
            f"Initialized PointTransformerBackbone: c={self.c}, "
            f"num_points={self.num_points}, output_dim={self.output_dim}"
        )
    
    def load_pretrained_weight(self, weight_path: str) -> None:
        """
        Load pretrained weights for the Point Transformer encoder.
        
        Args:
            weight_path: Path to pretrained weights file
        """
        self.model.load_pretrained_weight(weight_path)
        self.logger.info(f"Loaded pretrained weights from {weight_path}")
    
    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Point Transformer backbone.
        
        Args:
            pos: (B, N, C) point cloud tensor
                 C can be 3 (xyz), 4 (xyz+mask), 6 (xyz+rgb), or 7 (xyz+rgb+mask)
        
        Returns:
            xyz: (B, K, 3) - sparse point coordinates after downsampling
            features: (B, output_dim, K) - sparse point features
        """
        B, N, C = pos.shape
        device = pos.device
        
        # Convert to PXO format
        p, x, o = convert_to_pxo_format(pos)
        
        # Forward through Point Transformer
        try:
            p5, x5, o5 = self.model((p, x, o))
        except Exception as e:
            self.logger.error(f"Point Transformer forward failed: {e}")
            raise
        
        # Convert back to dense format
        xyz_dense, feat_dense = densify_pxo_output(p5, x5, o5, B)
        
        # Verify output dimension
        if feat_dense.shape[1] != self.output_dim:
            self.logger.warning(
                f"Point Transformer output dim ({feat_dense.shape[1]}) != expected ({self.output_dim}). "
                f"Adding projection layer."
            )
            # Add projection if needed
            if not hasattr(self, '_out_proj'):
                self._out_proj = nn.Linear(feat_dense.shape[1], self.output_dim).to(device)
            # Transpose for projection: (B, C, K) -> (B, K, C) -> project -> (B, K, C') -> (B, C', K)
            feat_dense = feat_dense.permute(0, 2, 1)
            feat_dense = self._out_proj(feat_dense)
            feat_dense = feat_dense.permute(0, 2, 1)
        
        return xyz_dense, feat_dense

