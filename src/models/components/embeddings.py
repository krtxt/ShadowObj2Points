"""Positional encoding and feature embedding modules."""

import math
from typing import Optional

import torch
import torch.nn as nn

class FourierPositionalEmbedding(nn.Module):
    """Fourier positional encoding for 3D coordinates.
    
    Maps 3D coordinates (x, y, z) to high-dimensional feature space using
    sinusoidal encoding similar to NeRF.
    
    Args:
        num_frequencies: Number of frequency bands
        include_input: Whether to include raw input coordinates
        log_sampling: Whether to use logarithmic frequency sampling (recommended)
    """
    def __init__(
        self,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling
        freq_features = 3 * 2 * num_frequencies
        self.output_dim = freq_features + (3 if include_input else 0)
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            freq_bands = torch.linspace(1, num_frequencies, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute Fourier positional encoding.
        
        Args:
            xyz: Coordinates of shape (B, N, 3) or (N, 3)
            
        Returns:
            Embedded features of shape (B, N, output_dim) or (N, output_dim)
        """
        original_shape = xyz.shape
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)
        B, N, _ = xyz.shape
        xyz_expanded = xyz.unsqueeze(-1)
        freq_expanded = self.freq_bands.view(1, 1, 1, -1)
        scaled = xyz_expanded * freq_expanded
        sin_features = torch.sin(scaled * math.pi)
        cos_features = torch.cos(scaled * math.pi)
        freq_features = torch.stack([sin_features, cos_features], dim=-1)
        freq_features = freq_features.reshape(B, N, -1)
        if self.include_input:
            embedded = torch.cat([xyz, freq_features], dim=-1)
        else:
            embedded = freq_features
        if len(original_shape) == 2:
            embedded = embedded.squeeze(0)
        return embedded


class PointNetEmbedding(nn.Module):
    """Combines Fourier positional encoding and link_id embedding.
    
    Maps coordinates and link IDs to a unified feature dimension.
    
    Args:
        num_links: Number of links (max link_id + 1)
        embed_dim: Output feature dimension
        link_embed_dim: Link ID embedding dimension
        num_frequencies: Number of Fourier frequency bands
        dropout: Dropout probability
    """
    def __init__(
        self,
        num_links: int,
        embed_dim: int = 256,
        link_embed_dim: int = 64,
        num_frequencies: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fourier_encoder = FourierPositionalEmbedding(
            num_frequencies=num_frequencies,
            include_input=True,
            log_sampling=True,
        )
        self.link_embedding = nn.Embedding(num_links, link_embed_dim)
        concat_dim = self.fourier_encoder.output_dim + link_embed_dim
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, xyz: torch.Tensor, link_ids: torch.Tensor) -> torch.Tensor:
        """Compute combined embedding from coordinates and link IDs.
        
        Args:
            xyz: 3D coordinates of shape (B, N, 3)
            link_ids: Link indices of shape (N,) or (B, N)
            
        Returns:
            Embedded features of shape (B, N, embed_dim)
        """
        xyz_features = self.fourier_encoder(xyz)
        if link_ids.dim() == 1:
            link_ids = link_ids.unsqueeze(0).expand(xyz.shape[0], -1)
        link_features = self.link_embedding(link_ids)
        concat_features = torch.cat([xyz_features, link_features], dim=-1)
        embedded = self.projection(concat_features)
        return embedded


if __name__ == "__main__":
    print("Testing FourierPositionalEmbedding...")
    fourier = FourierPositionalEmbedding(num_frequencies=10)
    xyz = torch.randn(2, 50, 3)
    embedded = fourier(xyz)
    print(f"Input shape: {xyz.shape}")
    print(f"Output shape: {embedded.shape}")
    print(f"Output dim: {fourier.output_dim}")
    print("\nTesting PointNetEmbedding...")
    embedder = PointNetEmbedding(num_links=20, embed_dim=256)
    link_ids = torch.randint(0, 20, (50,))
    features = embedder(xyz, link_ids)
    print(f"Features shape: {features.shape}")
