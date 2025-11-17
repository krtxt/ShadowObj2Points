import sys

sys.path.append("./third_party/pointnet2/")

import unittest

import torch
import torch.nn as nn
from pointnet2_modules import PointnetFPModule, PointnetSAModuleVotes
from torch.functional import Tensor


class Pointnet2Backbone(nn.Module):
    """
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            Now expects 3 for RGB features (xyz coordinates are handled separately).

       Note:
            Input point cloud format: (B, N, 6) where 6 = xyz + rgb
            The _break_up_pc method separates xyz (first 3 channels) and
            rgb features (last 3 channels) automatically.
    """

    def __init__(self, cfg):
        super().__init__()

        self.mode = getattr(cfg, "mode", "sparse")
        if self.mode not in ("sparse", "dense"):
            raise ValueError(f"Unsupported PointNet2 mode: {self.mode}")

        self.dense_out_dim = getattr(cfg, "dense_out_dim", 512)

        layer1_mlp = list(cfg.layer1.mlp_list)
        layer2_mlp = list(cfg.layer2.mlp_list)
        layer3_mlp = list(cfg.layer3.mlp_list)
        layer4_mlp = list(cfg.layer4.mlp_list)

        in0_dim = layer1_mlp[0]
        c1 = layer1_mlp[-1]
        c2 = layer2_mlp[-1]
        c3 = layer3_mlp[-1]
        c4 = layer4_mlp[-1]

        self.sa1 = PointnetSAModuleVotes(
            npoint=cfg.layer1.npoint,
            radius=cfg.layer1.radius_list[0],
            nsample=cfg.layer1.nsample_list[0],
            mlp=layer1_mlp.copy(),
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=cfg.layer2.npoint,
            radius=cfg.layer2.radius_list[0],
            nsample=cfg.layer2.nsample_list[0],
            mlp=layer2_mlp.copy(),
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=cfg.layer3.npoint,
            radius=cfg.layer3.radius_list[0],
            nsample=cfg.layer3.nsample_list[0],
            mlp=layer3_mlp.copy(),
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=cfg.layer4.npoint,
            radius=cfg.layer4.radius_list[0],
            nsample=cfg.layer4.nsample_list[0],
            mlp=layer4_mlp.copy(),
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        if self.mode == "dense":
            self.fp4 = PointnetFPModule(mlp=[c3 + c4, 256, 256])
            self.fp3 = PointnetFPModule(mlp=[c2 + 256, 256, 256])
            self.fp2 = PointnetFPModule(mlp=[c1 + 256, 256, 256])
            self.fp1 = PointnetFPModule(mlp=[in0_dim + 256, 256, self.dense_out_dim])
            self.output_dim = self.dense_out_dim
        else:
            self.output_dim = 512

        self.use_pooling = getattr(cfg, "use_pooling", False) and self.mode != "dense"
        if self.use_pooling:
            self.gap = torch.nn.AdaptiveAvgPool1d(1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: Tensor):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(Tensor)
                (B, N, 6) tensor - Point cloud with xyz + rgb
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, r, g, b)

            Returns
            ----------
                xyz: float32 Tensor of shape (B, K, 3)
                features: float32 Tensor of shape (B, D, K)
                inds: int64 Tensor of shape (B, K) values in [0, N-1]
        """
        l0_xyz, l0_features = self._break_up_pc(pointcloud)

        l1_xyz, l1_features, _ = self.sa1(l0_xyz, l0_features)
        l2_xyz, l2_features, _ = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features, _ = self.sa3(l2_xyz, l2_features)
        l4_xyz, l4_features, _ = self.sa4(l3_xyz, l3_features)

        if self.mode == "dense":
            l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features)
            l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
            l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
            l0_features = self.fp1(l0_xyz, l1_xyz, l0_features, l1_features)
            return l0_xyz, l0_features

        features = l4_features
        xyz = l4_xyz
        if self.use_pooling:
            features = self.gap(features).squeeze(-1)  # 移除最后一个维度，从[B, 512, 1]变为[B, 512]
        return xyz, features

