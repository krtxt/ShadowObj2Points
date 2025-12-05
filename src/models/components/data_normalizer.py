"""Data normalization utilities for flow matching models."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class DataNormalizer:
    """Handles coordinate normalization/denormalization for flow matching models.
    
    Manages transformation between original world space and normalized [-1, 1] space.
    Also handles per-point Gaussian noise statistics in both spaces.
    
    Args:
        module: Parent nn.Module for buffer registration
        graph_consts: Dictionary containing normalization bounds and templates
        use_norm_data: Whether normalization is enabled
        use_per_point_gaussian_noise: Whether per-point noise is used
    """
    
    def __init__(
        self,
        module: nn.Module,
        graph_consts: dict,
        use_norm_data: bool = False,
        use_per_point_gaussian_noise: bool = False,
    ):
        self.module = module
        self.use_norm_data = use_norm_data
        self.use_per_point_gaussian_noise = use_per_point_gaussian_noise
        
        # Extract bounds from graph_consts (may be None if not provided)
        self.norm_min: Optional[torch.Tensor] = graph_consts.get("norm_min")
        self.norm_max: Optional[torch.Tensor] = graph_consts.get("norm_max")
        
        if use_norm_data:
            self._validate_bounds()
    
    def _validate_bounds(self) -> None:
        """Ensure normalization bounds are available and correctly shaped."""
        if self.norm_min is None or self.norm_max is None:
            raise RuntimeError(
                "use_norm_data=True requires 'norm_min' and 'norm_max' to be supplied in graph_consts."
            )
        if (self.norm_min.dim() != 3 or self.norm_max.dim() != 3 or 
            self.norm_min.shape[-1] != 3 or self.norm_max.shape[-1] != 3):
            raise ValueError(
                f"Normalization bounds must have shape (1,1,3); "
                f"got {tuple(self.norm_min.shape)} / {tuple(self.norm_max.shape)}"
            )
    
    @torch.no_grad()
    def apply_norm_to_graph_consts(
        self,
        template_xyz: torch.Tensor,
        canonical_xyz: Optional[torch.Tensor],
        approx_std: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        edge_rest_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Recompute template and edge rest lengths in normalized space.
        
        Returns:
            Tuple of (norm_template_xyz, norm_canonical_xyz, norm_approx_std, norm_edge_rest_lengths)
        """
        if not self.use_norm_data or self.norm_min is None or self.norm_max is None:
            return template_xyz, canonical_xyz, approx_std, edge_rest_lengths
        
        scale = 2.0 / torch.clamp(self.norm_max - self.norm_min, min=1e-6)
        bias = -1.0
        
        def _norm_pts(x: torch.Tensor) -> torch.Tensor:
            view_shape = [1] * x.dim()
            view_shape[-1] = 3
            nm = self.norm_min.view(view_shape)
            sc = scale.view(view_shape)
            return (x - nm) * sc + bias
        
        # Normalize template
        tmpl_norm = _norm_pts(template_xyz)
        
        # Normalize canonical xyz if provided
        canon_norm = None
        if canonical_xyz is not None:
            canon_norm = _norm_pts(canonical_xyz)
        
        # Normalize approx_std if provided (scale only, no offset)
        std_norm = None
        if approx_std is not None:
            view_shape = [1] * approx_std.dim()
            view_shape[-1] = 3
            sc = scale.view(view_shape)
            std_norm = approx_std * sc
        
        # Recompute edge rest lengths
        tmpl_for_edge = tmpl_norm
        if tmpl_for_edge.dim() == 3 and tmpl_for_edge.shape[0] == 1:
            tmpl_for_edge = tmpl_for_edge.squeeze(0)
        i, j = edge_index
        diff = tmpl_for_edge[i, :] - tmpl_for_edge[j, :]
        rest_norm = torch.linalg.norm(diff, dim=-1)
        
        return tmpl_norm, canon_norm, std_norm, rest_norm
    
    def denorm_xyz(self, norm_xyz: torch.Tensor) -> torch.Tensor:
        """Map normalized xyz in [-1,1] back to original space.
        
        Supports both 3D (xyz only) and 6D (xyz + normals) inputs.
        For 6D input, only the first 3 channels are denormalized.
        """
        if self.norm_min is None or self.norm_max is None:
            raise RuntimeError("Normalization bounds not loaded.")
        
        # Handle 6D input (xyz + normals)
        if norm_xyz.shape[-1] == 6:
            xyz_part = norm_xyz[..., :3]
            normals_part = norm_xyz[..., 3:]
            xyz_denorm = self.denorm_xyz(xyz_part)
            return torch.cat([xyz_denorm, normals_part], dim=-1)
        
        norm = torch.clamp(norm_xyz, -1.0, 1.0)
        unscaled = (norm + 1.0) * 0.5
        denom = torch.clamp(self.norm_max - self.norm_min, min=1e-6)
        return unscaled * denom + self.norm_min
    
    def get_noise_stats(
        self,
        N: int,
        canonical_xyz: Optional[torch.Tensor],
        approx_std: Optional[torch.Tensor],
        norm_canonical_xyz: Optional[torch.Tensor],
        norm_approx_std: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch per-point Gaussian mean/std depending on normalization setting.
        
        Returns:
            Tuple of (mean, std) tensors of shape (1, N, 3)
        """
        if self.use_norm_data:
            mean = norm_canonical_xyz
            std = norm_approx_std
        else:
            mean = canonical_xyz
            std = approx_std
        
        if mean is None or std is None:
            raise RuntimeError(
                "canonical_xyz/approx_std (or normalized variants) are required for per-point noise."
            )
        
        mean = mean.to(device=device, dtype=dtype).view(1, N, 3)
        std = std.to(device=device, dtype=dtype).view(1, N, 3)
        return mean, std


class SamplingFrequencyManager:
    """Manages sampling frequency for expensive computations.
    
    Used to limit expensive loss/metric computations to a subset of batches.
    Automatically discovers total batch count and computes uniform stride.
    """
    
    def __init__(self, num_sample_batches: Optional[int] = None):
        """
        Args:
            num_sample_batches: Target number of batches to sample per epoch.
                               None or 0 means sample all batches.
        """
        self.num_sample_batches = num_sample_batches
        self._total_batches: int = 0
        self._sample_count: int = 0
    
    def reset(self) -> None:
        """Reset sample counter at epoch start."""
        self._sample_count = 0
    
    def should_sample(self, batch_idx: int) -> bool:
        """Determine if this batch should be sampled.
        
        Args:
            batch_idx: Current batch index
            
        Returns:
            True if this batch should be sampled
        """
        if self.num_sample_batches is None:
            return True
        
        # Track total batches for stride calculation
        self._total_batches = max(self._total_batches, batch_idx + 1)
        
        if self._total_batches > 0:
            stride = max(1, self._total_batches // self.num_sample_batches)
            should = (batch_idx % stride == 0) and (self._sample_count < self.num_sample_batches)
        else:
            # First epoch: sample conservatively
            stride = max(1, 50 // self.num_sample_batches)
            should = (batch_idx % stride == 0) and (self._sample_count < self.num_sample_batches)
        
        if should:
            self._sample_count += 1
        
        return should
    
    @property
    def total_batches(self) -> int:
        return self._total_batches
