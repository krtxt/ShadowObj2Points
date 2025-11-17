"""
PointNext Backbone Wrapper for SceneLeapUltra

Wraps the official PointNext implementation from OpenPoints to provide
a unified interface compatible with PointNet2, PTv3 and the decoder models.

Model source: from openpoints.models.backbone.pointnext import PointNextEncoder
"""
import logging
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Add third_party path if needed
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
openpoints_path = os.path.join(project_root, "third_party")
if openpoints_path not in sys.path:
    sys.path.insert(0, openpoints_path)

try:
    from openpoints.models.backbone.pointnext import PointNextEncoder
    POINTNEXT_AVAILABLE = True
    logging.info("PointNext imported successfully")
except ImportError as e:
    logging.warning(f"PointNext not available: {e}")
    logging.warning("Please ensure OpenPoints dependencies are installed or use PTv3/PointNet2 instead")
    POINTNEXT_AVAILABLE = False
    PointNextEncoder = None


class PointNextBackbone(nn.Module):
    """
    PointNext Backbone wrapper for SceneLeapUltra.
    
    Provides a unified interface compatible with PointNet2 and PTv3:
    - Input: (B, N, C) where C = xyz(3) + optional_features
    - Output: (xyz, features) where xyz is (B, K, 3), features is (B, out_dim, K)
    
    PointNext is a unified point cloud analysis model that:
    - Uses efficient hierarchical architecture with InvResMLP blocks
    - Supports both classification and segmentation tasks
    - Provides better performance than PointNet++
    
    Args:
        cfg: Configuration object with PointNext parameters
            - num_points: Number of input points (default: 8192)
            - num_tokens: Number of output tokens K (default: 128)
            - out_dim: Output feature dimension (default: 512)
            - width: Base channel width (default: 32)
            - blocks: Number of InvResMLP blocks per stage (list, default: [1,1,1,1,1])
            - strides: Stride for each stage (list, default: [1,4,4,4,4])
            - use_res: Use residual connections (default: True)
            - radius: Ball query radius (default: 0.1)
            - nsample: Number of samples per ball query (default: 32)
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        if not POINTNEXT_AVAILABLE:
            raise ImportError(
                "PointNext not available. Please install OpenPoints: "
                "cd third_party/openpoints && pip install -e ."
            )
        
        self.logger = logging.getLogger(__name__)
        
        # Basic parameters
        self.num_points = getattr(cfg, 'num_points', 8192)
        self.num_tokens = getattr(cfg, 'num_tokens', 128)
        self.out_dim = getattr(cfg, 'out_dim', 512)
        
        # Mode configuration: sparse or dense
        self.mode = getattr(cfg, 'mode', 'sparse')  # sparse | dense
        self.dense_out_dim = getattr(cfg, 'dense_out_dim', 512)
        
        # PointNext architecture parameters
        self.width = getattr(cfg, 'width', 32)
        self.blocks = list(getattr(cfg, 'blocks', [1, 1, 1, 1, 1]))
        self.strides = list(getattr(cfg, 'strides', [1, 4, 4, 4, 4]))
        
        # Local aggregation parameters
        self.use_res = getattr(cfg, 'use_res', True)
        self.radius = getattr(cfg, 'radius', 0.1)
        self.nsample = getattr(cfg, 'nsample', 32)
        
        # Input/output configuration
        self.in_channels = getattr(cfg, 'input_feature_dim', 3)  # xyz only by default
        self.num_classes = self.out_dim  # For segmentation mode
        
        # Compatibility flags
        self.use_xyz = getattr(cfg, 'use_xyz', True)
        self.normalize_xyz = getattr(cfg, 'normalize_xyz', True)
        
        # Build PointNext encoder
        self.encoder = self._build_pointnext_encoder(cfg)
        
        # FPS sampling for token extraction (only used in sparse mode)
        self.use_fps = getattr(cfg, 'use_fps', True)
        
        # Calculate encoder output dimension
        encoder_out_dim = self._get_encoder_output_dim()
        
        # Build decoder for dense mode
        if self.mode == 'dense':
            self._build_dense_decoder()
            self.output_dim = self.dense_out_dim
            self.logger.info(
                f"Built dense decoder with output_dim={self.dense_out_dim}"
            )
        else:
            # Sparse mode: output projection to ensure correct dimension
            if encoder_out_dim != self.out_dim:
                self.output_projection = nn.Sequential(
                    nn.Linear(encoder_out_dim, self.out_dim),
                    nn.LayerNorm(self.out_dim),
                    nn.ReLU(inplace=True)
                )
                self.logger.info(
                    f"Added output projection: {encoder_out_dim} -> {self.out_dim}"
                )
            else:
                self.output_projection = nn.Identity()
            self.output_dim = self.out_dim
        
        self.logger.info(
            f"Initialized PointNextBackbone: "
            f"num_points={self.num_points}, num_tokens={self.num_tokens}, "
            f"out_dim={self.out_dim}, width={self.width}, "
            f"blocks={self.blocks}, strides={self.strides}, mode={self.mode}"
        )
        
        # Debug info for potential errors
        self.logger.debug(
            f"[PointNext-{self.mode}] Input channels: {self.in_channels}, "
            f"Encoder output dim: {encoder_out_dim}, "
            f"Final output dim: {self.output_dim}"
        )
    
    def _get_encoder_output_dim(self):
        """Calculate encoder output dimension based on architecture."""
        # PointNext doubles width after each downsampling (stride != 1)
        # Replicate the logic from PointNextEncoder.__init__
        width = self.width
        for stride in self.strides:
            if stride != 1:
                width *= 2
        return width
    
    def _build_dense_decoder(self):
        """Build FeaturePropogation decoder for dense mode."""
        from openpoints.models.backbone.pointnext import FeaturePropogation
        
        # Get encoder channel list: [width*2^0, width*2^1, ..., width*2^k]
        encoder_channels = self.encoder.channel_list
        
        # Build skip channels list: prepend input channels
        skip_channels = encoder_channels[:-1]
        skip_channels.insert(0, self.in_channels)
        
        # Build decoder stages (upsampling from deep to shallow)
        decoder_stages = []
        in_ch = encoder_channels[-1]  # Start from deepest layer
        
        # Intermediate FP stages (from deepest to shallowest, excluding input layer)
        for i in range(len(encoder_channels) - 1, 0, -1):
            # MLP: [skip_channels + in_ch, intermediate_dim, intermediate_dim]
            intermediate_dim = 256
            mlp = [skip_channels[i] + in_ch, intermediate_dim, intermediate_dim]
            decoder_stages.append(FeaturePropogation(mlp))
            in_ch = intermediate_dim
        
        # Final FP stage: upsample to original points with target output dimension
        mlp = [skip_channels[0] + in_ch, 256, self.dense_out_dim]
        decoder_stages.append(FeaturePropogation(mlp))
        
        self.decoder_stages = nn.ModuleList(decoder_stages)
        
        self.logger.info(
            f"Built {len(decoder_stages)} FeaturePropogation stages for dense mode"
        )
    
    def _build_pointnext_encoder(self, cfg):
        """Build PointNext encoder based on configuration."""
        try:
            # Additional parameters
            sa_layers = getattr(cfg, 'sa_layers', 1)
            sa_use_res = getattr(cfg, 'sa_use_res', False)
            expansion = getattr(cfg, 'expansion', 4)
            reduction = getattr(cfg, 'reduction', 4)
            
            # Aggregation args
            aggr_args = {
                'feature_type': 'dp_fj',
                'reduction': 'max'
            }
            
            # Group args - need to use EasyDict for attribute access
            from easydict import EasyDict
            group_args = EasyDict({'NAME': 'ballquery'})
            
            # Sampler selection
            sampler = getattr(cfg, 'sampler', 'fps')
            
            # Build encoder with parameters
            encoder = PointNextEncoder(
                in_channels=self.in_channels,
                width=self.width,
                blocks=self.blocks,
                strides=self.strides,
                nsample=self.nsample,
                radius=self.radius,
                aggr_args=aggr_args,
                group_args=group_args,
                sa_layers=sa_layers,
                sa_use_res=sa_use_res,
                norm_args={'norm': 'bn'},
                act_args={'act': 'relu'},
                conv_args={},  # Empty dict instead of None
                expansion=expansion,
                sampler=sampler,  # 'fps' or 'random'
            )
            
            self.logger.info(
                f"Built PointNext encoder: in_channels={self.in_channels}, "
                f"width={self.width}, stages={len(self.blocks)}, "
                f"sa_layers={sa_layers}, expansion={expansion}"
            )
            
            return encoder
            
        except Exception as e:
            self.logger.error(f"Failed to build PointNext encoder: {e}")
            self.logger.error(
                f"Config: width={self.width}, blocks={self.blocks}, "
                f"strides={self.strides}, in_channels={self.in_channels}"
            )
            raise
    
    def _fps_sample(self, xyz: torch.Tensor, features: torch.Tensor, 
                    num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Farthest Point Sampling to extract K tokens from N points.
        
        Args:
            xyz: (B, N, 3) coordinates
            features: (B, N, D) features
            num_samples: K, number of tokens to sample
        
        Returns:
            sampled_xyz: (B, K, 3)
            sampled_features: (B, K, D)
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        if num_samples >= N:
            self.logger.warning(
                f"num_samples ({num_samples}) >= N ({N}), returning all points"
            )
            return xyz, features
        
        # Farthest Point Sampling
        try:
            from pointnet2_ops import pointnet2_utils
            # FPS expects (B, N, 3)
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_samples)  # (B, K)
            
            # Gather sampled points and features
            # fps_idx: (B, K) -> (B, K, 1) -> (B, K, 3) for xyz
            fps_idx_expanded = fps_idx.unsqueeze(-1).expand(-1, -1, 3)
            sampled_xyz = torch.gather(xyz, 1, fps_idx_expanded)  # (B, K, 3)
            
            # (B, K, 1) -> (B, K, D) for features
            fps_idx_expanded = fps_idx.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            sampled_features = torch.gather(features, 1, fps_idx_expanded)  # (B, K, D)
            
            return sampled_xyz, sampled_features
            
        except ImportError:
            self.logger.warning(
                "pointnet2_ops not available, using random sampling instead"
            )
            # Fallback to random sampling
            indices = torch.randperm(N, device=device)[:num_samples]
            indices = indices.unsqueeze(0).expand(B, -1)  # (B, K)
            
            indices_xyz = indices.unsqueeze(-1).expand(-1, -1, 3)
            sampled_xyz = torch.gather(xyz, 1, indices_xyz)
            
            indices_feat = indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            sampled_features = torch.gather(features, 1, indices_feat)
            
            return sampled_xyz, sampled_features
    
    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PointNext backbone.
        
        Args:
            pos: (B, N, C) point cloud tensor
                 C can be 3 (xyz), 4 (xyz+mask), 6 (xyz+rgb), or 7 (xyz+rgb+mask)
        
        Returns:
            In sparse mode:
                xyz: (B, K, 3) sampled point coordinates
                features: (B, out_dim, K) sampled point features
            In dense mode:
                xyz: (B, N, 3) original point coordinates
                features: (B, dense_out_dim, N) per-point features
        """
        B, N, C = pos.shape
        device = pos.device
        
        # Split coordinates and features
        xyz = pos[..., :3].contiguous()  # (B, N, 3)
        
        if C > 3:
            # Has additional features (rgb, mask, or both)
            features = pos[..., 3:].contiguous()  # (B, N, C-3)
        else:
            # Only xyz, no additional features
            features = None
        
        # Normalize xyz if required
        if self.normalize_xyz:
            # Center and normalize
            xyz_mean = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            xyz_centered = xyz - xyz_mean
            xyz_std = xyz_centered.std(dim=1, keepdim=True).clamp(min=1e-6)
            xyz_normalized = xyz_centered / xyz_std
        else:
            xyz_normalized = xyz
        
        # Prepare input for PointNext
        # PointNext expects (B, C, N) format for features
        if features is not None:
            # Concatenate xyz with features if use_xyz is True
            if self.use_xyz:
                point_features = torch.cat([xyz_normalized, features], dim=-1)  # (B, N, 3+F)
            else:
                point_features = features  # (B, N, F)
        else:
            # Only xyz as features
            point_features = xyz_normalized  # (B, N, 3)
        
        # Transpose to (B, C, N) for PointNext
        point_features = point_features.transpose(1, 2).contiguous()  # (B, C, N)
        
        # Debug info
        self.logger.debug(
            f"[PointNext-{self.mode}] Input shape: pos={pos.shape}, "
            f"xyz_normalized={xyz_normalized.shape}, "
            f"point_features={point_features.shape}"
        )
        
        # Branch based on mode
        if self.mode == 'sparse':
            return self._forward_sparse(xyz, xyz_normalized, point_features, B, N, device)
        elif self.mode == 'dense':
            return self._forward_dense(xyz, xyz_normalized, point_features, B, N)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'sparse' or 'dense'")
    
    def _forward_sparse(self, xyz: torch.Tensor, xyz_normalized: torch.Tensor, 
                       point_features: torch.Tensor, B: int, N: int, 
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse mode: encode and sample K tokens."""
        # Forward through PointNext encoder
        try:
            # PointNext encoder returns (p_list, f_list)
            p_list, f_list = self.encoder(xyz_normalized, point_features)
            
            # Use the last stage features
            encoded_xyz = p_list[-1]  # (B, Nk, 3)
            encoded_features = f_list[-1]  # (B, Ck, Nk)
            
            self.logger.debug(
                f"[PointNext-sparse] Encoder output shape: xyz={encoded_xyz.shape}, "
                f"features={encoded_features.shape}"
            )
            
        except Exception as e:
            self.logger.error(f"PointNext encoder failed: {e}")
            self.logger.error(
                f"Input shapes - xyz: {xyz_normalized.shape}, "
                f"features: {point_features.shape}"
            )
            import traceback
            traceback.print_exc()
            raise
        
        # Transpose features back to (B, Nk, D)
        encoded_features = encoded_features.transpose(1, 2).contiguous()  # (B, Nk, D)
        
        # Get the number of points from encoder output
        Nk = encoded_xyz.shape[1]
        
        # Sample K tokens using FPS if needed
        if self.use_fps and self.num_tokens < Nk:
            sampled_xyz, sampled_features = self._fps_sample(
                encoded_xyz, encoded_features, self.num_tokens
            )
            K = self.num_tokens
        else:
            # Use all points or simple stride sampling
            if self.num_tokens < Nk:
                stride = Nk // self.num_tokens
                indices = torch.arange(0, Nk, stride, device=device)[:self.num_tokens]
                sampled_xyz = encoded_xyz[:, indices, :]  # (B, K, 3)
                sampled_features = encoded_features[:, indices, :]  # (B, K, D)
                K = sampled_xyz.shape[1]
            else:
                sampled_xyz = encoded_xyz
                sampled_features = encoded_features
                K = Nk
        
        # Apply output projection if needed
        sampled_features = self.output_projection(sampled_features)  # (B, K, out_dim)
        
        # Transpose features to (B, out_dim, K) to match PointNet2 interface
        output_features = sampled_features.transpose(1, 2).contiguous()  # (B, out_dim, K)
        
        self.logger.debug(
            f"[PointNext-sparse] Output shapes: xyz={sampled_xyz.shape}, "
            f"features={output_features.shape}"
        )
        
        # Sanity check
        assert sampled_xyz.shape == (B, K, 3), \
            f"xyz shape mismatch: expected ({B}, {K}, 3), got {sampled_xyz.shape}"
        assert output_features.shape == (B, self.out_dim, K), \
            f"features shape mismatch: expected ({B}, {self.out_dim}, {K}), got {output_features.shape}"
        
        return sampled_xyz, output_features
    
    def _forward_dense(self, xyz: torch.Tensor, xyz_normalized: torch.Tensor,
                      point_features: torch.Tensor, B: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense mode: encode and upsample back to N points."""
        # Forward through PointNext encoder with segmentation features
        try:
            # Use forward_seg_feat to get all layer features
            p_list, f_list = self.encoder.forward_seg_feat(xyz_normalized, point_features)
            # p_list: [p0(B,N,3), p1(B,N1,3), ..., pk(B,Nk,3)]
            # f_list: [f0(B,C0,N), f1(B,C1,N1), ..., fk(B,Ck,Nk)]
            
            self.logger.debug(
                f"[PointNext-dense] Encoder stages: {len(p_list)}, "
                f"last stage xyz={p_list[-1].shape}, features={f_list[-1].shape}"
            )
            
        except Exception as e:
            self.logger.error(f"PointNext encoder (dense) failed: {e}")
            self.logger.error(
                f"Input shapes - xyz: {xyz_normalized.shape}, "
                f"features: {point_features.shape}"
            )
            import traceback
            traceback.print_exc()
            raise
        
        # Upsample through decoder stages
        # Start from the deepest layer and progressively upsample
        f_upsampled = f_list[-1]  # (B, Ck, Nk)
        
        for i, decoder in enumerate(self.decoder_stages):
            # Calculate which layer we're upsampling to
            layer_idx = len(f_list) - 2 - i
            
            if layer_idx >= 0:
                # FeaturePropogation: forward(pf1, pf2)
                # pf1 is target layer (denser), pf2 is source layer (sparser)
                f_upsampled = decoder(
                    (p_list[layer_idx], f_list[layer_idx]),  # Target (denser)
                    (p_list[layer_idx + 1], f_upsampled)     # Source (sparser)
                )
                # f_upsampled: (B, C, N_target)
                
                self.logger.debug(
                    f"[PointNext-dense] Decoder stage {i}: "
                    f"upsampled to layer {layer_idx}, shape={f_upsampled.shape}"
                )
        
        # Final output: f_upsampled should be (B, dense_out_dim, N)
        output_features = f_upsampled
        output_xyz = p_list[0]  # Original points (B, N, 3)
        
        self.logger.debug(
            f"[PointNext-dense] Output shapes: xyz={output_xyz.shape}, "
            f"features={output_features.shape}"
        )
        
        # Sanity check
        assert output_xyz.shape == (B, N, 3), \
            f"xyz shape mismatch: expected ({B}, {N}, 3), got {output_xyz.shape}"
        assert output_features.shape == (B, self.dense_out_dim, N), \
            f"features shape mismatch: expected ({B}, {self.dense_out_dim}, {N}), got {output_features.shape}"
        
        return output_xyz, output_features


# Unit test
if __name__ == "__main__":
    import sys
    sys.path.append("./")
    
    from omegaconf import OmegaConf
    
    # Test configuration
    cfg_dict = {
        'name': 'pointnext',
        'num_points': 8192,
        'num_tokens': 128,
        'out_dim': 512,
        'width': 32,
        'blocks': [1, 1, 1, 1, 1],
        'strides': [1, 4, 4, 4, 4],
        'use_res': True,
        'radius': 0.1,
        'nsample': 32,
        'input_feature_dim': 3,
        'use_xyz': True,
        'normalize_xyz': True,
        'use_fps': True,
    }
    
    cfg = OmegaConf.create(cfg_dict)
    
    print("=" * 80)
    print("Testing PointNext Backbone")
    print("=" * 80)
    
    # Create model
    model = PointNextBackbone(cfg)
    model.eval()
    
    # Test input
    batch_size = 2
    num_points = 8192
    
    # Test case 1: xyz only
    print("\n[Test 1] Input: (B, N, 3) - xyz only")
    pos_xyz = torch.randn(batch_size, num_points, 3)
    xyz_out, feat_out = model(pos_xyz)
    print(f"Input shape: {pos_xyz.shape}")
    print(f"Output xyz shape: {xyz_out.shape}")
    print(f"Output features shape: {feat_out.shape}")
    assert xyz_out.shape == (batch_size, 128, 3)
    assert feat_out.shape == (batch_size, 512, 128)
    print("✓ Test 1 passed!")
    
    # Test case 2: xyz + rgb
    print("\n[Test 2] Input: (B, N, 6) - xyz + rgb")
    pos_xyzrgb = torch.randn(batch_size, num_points, 6)
    xyz_out, feat_out = model(pos_xyzrgb)
    print(f"Input shape: {pos_xyzrgb.shape}")
    print(f"Output xyz shape: {xyz_out.shape}")
    print(f"Output features shape: {feat_out.shape}")
    assert xyz_out.shape == (batch_size, 128, 3)
    assert feat_out.shape == (batch_size, 512, 128)
    print("✓ Test 2 passed!")
    
    # Test case 3: Different output dimensions
    print("\n[Test 3] Custom output dimension: 256")
    cfg_dict['out_dim'] = 256
    cfg_dict['num_tokens'] = 256
    cfg = OmegaConf.create(cfg_dict)
    model2 = PointNextBackbone(cfg)
    model2.eval()
    
    pos_xyz = torch.randn(batch_size, num_points, 3)
    xyz_out, feat_out = model2(pos_xyz)
    print(f"Input shape: {pos_xyz.shape}")
    print(f"Output xyz shape: {xyz_out.shape}")
    print(f"Output features shape: {feat_out.shape}")
    assert xyz_out.shape == (batch_size, 256, 3)
    assert feat_out.shape == (batch_size, 256, 256)
    print("✓ Test 3 passed!")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

