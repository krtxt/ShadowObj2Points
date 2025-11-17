"""
位置编码和特征嵌入模块
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class FourierPositionalEmbedding(nn.Module):
    """
    Fourier 位置编码，将 3D 坐标 (x, y, z) 映射到高维特征空间
    类似 NeRF 中的位置编码方法
    """

    def __init__(
        self,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        """
        Args:
            num_frequencies: 频率数量
            include_input: 是否包含原始输入坐标
            log_sampling: 是否使用对数采样频率（推荐）
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling

        # 计算输出维度
        # 每个频率产生 sin 和 cos，共 2 * num_frequencies
        # 对于 3D 输入，总共是 3 * 2 * num_frequencies
        freq_features = 3 * 2 * num_frequencies
        self.output_dim = freq_features + (3 if include_input else 0)

        # 预计算频率
        if log_sampling:
            # 对数采样：从 2^0 到 2^(num_frequencies-1)
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            # 线性采样：从 1 到 num_frequencies
            freq_bands = torch.linspace(1, num_frequencies, num_frequencies)

        # 注册为 buffer（不参与训练但会随模型移动）
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) 或 (N, 3) 的坐标张量

        Returns:
            embedded: (B, N, output_dim) 或 (N, output_dim) 的嵌入特征
        """
        original_shape = xyz.shape
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)  # (N, 3) -> (1, N, 3)

        B, N, _ = xyz.shape

        # xyz: (B, N, 3)
        # freq_bands: (num_frequencies,)
        # 扩展维度进行广播
        xyz_expanded = xyz.unsqueeze(-1)  # (B, N, 3, 1)
        freq_expanded = self.freq_bands.view(1, 1, 1, -1)  # (1, 1, 1, num_frequencies)

        # 计算 xyz * freq_bands
        scaled = xyz_expanded * freq_expanded  # (B, N, 3, num_frequencies)

        # 计算 sin 和 cos
        sin_features = torch.sin(scaled * math.pi)  # (B, N, 3, num_frequencies)
        cos_features = torch.cos(scaled * math.pi)  # (B, N, 3, num_frequencies)

        # 拼接 sin 和 cos 并展平
        # (B, N, 3, num_frequencies, 2) -> (B, N, 3 * num_frequencies * 2)
        freq_features = torch.stack([sin_features, cos_features], dim=-1)
        freq_features = freq_features.reshape(B, N, -1)

        # 是否包含原始输入
        if self.include_input:
            embedded = torch.cat([xyz, freq_features], dim=-1)
        else:
            embedded = freq_features

        # 恢复原始形状
        if len(original_shape) == 2:
            embedded = embedded.squeeze(0)  # (1, N, D) -> (N, D)

        return embedded


class PointNetEmbedding(nn.Module):
    """
    组合 Fourier 位置编码和 link_id 嵌入，并映射到统一的特征维度
    """

    def __init__(
        self,
        num_links: int,
        embed_dim: int = 256,
        link_embed_dim: int = 64,
        num_frequencies: int = 10,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_links: 连杆数量（link_id 的最大值+1）
            embed_dim: 输出特征维度
            link_embed_dim: link_id 嵌入维度
            num_frequencies: Fourier 编码的频率数量
            dropout: Dropout 比例
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Fourier 位置编码
        self.fourier_encoder = FourierPositionalEmbedding(
            num_frequencies=num_frequencies,
            include_input=True,
            log_sampling=True,
        )

        # Link ID 嵌入
        self.link_embedding = nn.Embedding(num_links, link_embed_dim)

        # 计算拼接后的特征维度
        concat_dim = self.fourier_encoder.output_dim + link_embed_dim

        # MLP 映射到统一维度
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        link_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) 3D 坐标
            link_ids: (N,) 或 (B, N) link 索引

        Returns:
            features: (B, N, embed_dim) 嵌入特征
        """
        # Fourier 编码 xyz
        xyz_features = self.fourier_encoder(xyz)  # (B, N, fourier_dim)

        # Link ID 嵌入
        if link_ids.dim() == 1:
            # (N,) -> (B, N)
            link_ids = link_ids.unsqueeze(0).expand(xyz.shape[0], -1)

        link_features = self.link_embedding(link_ids)  # (B, N, link_embed_dim)

        # 拼接特征
        concat_features = torch.cat([xyz_features, link_features], dim=-1)

        # 映射到统一维度
        embedded = self.projection(concat_features)  # (B, N, embed_dim)

        return embedded


if __name__ == "__main__":
    # 测试代码
    print("Testing FourierPositionalEmbedding...")
    fourier = FourierPositionalEmbedding(num_frequencies=10)
    xyz = torch.randn(2, 50, 3)  # 2个batch，每个50个点
    embedded = fourier(xyz)
    print(f"Input shape: {xyz.shape}")
    print(f"Output shape: {embedded.shape}")
    print(f"Output dim: {fourier.output_dim}")

    print("\nTesting PointNetEmbedding...")
    embedder = PointNetEmbedding(num_links=20, embed_dim=256)
    link_ids = torch.randint(0, 20, (50,))
    features = embedder(xyz, link_ids)
    print(f"Features shape: {features.shape}")

