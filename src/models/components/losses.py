"""Loss functions and metrics for hand pose generation models.

This module provides:
- Component-based loss system with registry pattern
- Flow matching losses (MSE, tangent regularization)  
- Reconstruction losses (L1, Chamfer, direction, edge length, bone, collision)
- High-level loss managers (FlowMatchingLoss, DeterministicRegressionLoss)
- Validation metric manager (HandValidationMetricManager)
- Legacy loss classes for backward compatibility
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List, Callable, Set, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from kaolin.metrics.trianglemesh import (
    compute_sdf,
    CUSTOM_index_vertices_by_faces as index_vertices_by_faces,
)

from .loss_scheduler import LossScheduler, DEFAULT_WEIGHTS, LEGACY_WEIGHT_ALIASES

# Optional: PyTorch3D for Chamfer distance
try:
    from pytorch3d.loss import chamfer_distance
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: Component Registry & Base Class
# =============================================================================

class LossComponent(ABC):
    """Base class for all loss components.
    
    Each component is a standalone, stateless function that computes a single
    loss term. Components are registered in LOSS_COMPONENT_REGISTRY.
    """
    
    name: str = ""  # Unique identifier
    
    @abstractmethod
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        """Compute the loss value from context dictionary."""
        raise NotImplementedError
    
    @property
    def required_inputs(self) -> Set[str]:
        """Return set of required input keys in ctx."""
        return set()
    
    def is_available(self, ctx: Dict[str, Any]) -> bool:
        """Check if this component can be computed with given context."""
        return all(k in ctx and ctx[k] is not None for k in self.required_inputs)


# Global registry
LOSS_COMPONENT_REGISTRY: Dict[str, LossComponent] = {}


def register_loss_component(component: LossComponent) -> LossComponent:
    """Register a loss component in the global registry."""
    if not component.name:
        raise ValueError(f"Component {component.__class__.__name__} must have a non-empty name")
    LOSS_COMPONENT_REGISTRY[component.name] = component
    return component


def get_loss_component(name: str) -> Optional[LossComponent]:
    """Get a registered loss component by name."""
    return LOSS_COMPONENT_REGISTRY.get(name)


# =============================================================================
# Section 2: Capsule Mesh Utilities (for Collision Loss)
# =============================================================================

# Cached capsule mesh templates
_CAPSULE_TEMPLATE_CACHE: Dict[Tuple[str, torch.dtype, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_REF_CACHE: Dict[Tuple[str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def _build_unit_capsule_template(circle_segments: int = 8, cap_segments: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a coarse unit capsule mesh centered at origin, aligned with +Z."""
    if circle_segments < 3:
        raise ValueError("circle_segments must be >= 3")
    if cap_segments < 1:
        raise ValueError("cap_segments must be >= 1")

    cap_depth = 0.7
    angles = torch.linspace(0.0, 2 * math.pi, steps=circle_segments + 1, dtype=torch.float32)[:-1]
    cos, sin = torch.cos(angles), torch.sin(angles)

    verts: List[torch.Tensor] = [torch.tensor([0.0, 0.0, -0.5 - cap_depth], dtype=torch.float32)]
    ring_starts: List[int] = []

    # Bottom hemisphere
    phi_bottom = torch.linspace(0.2, math.pi / 2.0, steps=cap_segments + 2, dtype=torch.float32)[1:]
    phi_bottom = phi_bottom[phi_bottom < (math.pi / 2.0)]
    for phi in phi_bottom:
        z = -0.5 - cap_depth * torch.cos(phi)
        r = torch.sin(phi)
        ring_starts.append(len(verts))
        ring = torch.stack([r * cos, r * sin, torch.full_like(cos, z)], dim=1)
        verts.extend(ring)

    # Cylinder ring
    ring_starts.append(len(verts))
    ring = torch.stack([cos, sin, torch.full_like(cos, 0.5)], dim=1)
    verts.extend(ring)

    # Top hemisphere
    phi_top = torch.linspace(math.pi / 2.0, 0.2, steps=cap_segments + 2, dtype=torch.float32)[1:]
    for phi in phi_top:
        z = 0.5 + cap_depth * torch.cos(phi)
        r = torch.sin(phi)
        ring_starts.append(len(verts))
        ring = torch.stack([r * cos, r * sin, torch.full_like(cos, z)], dim=1)
        verts.extend(ring)

    top_pole_idx = len(verts)
    verts.append(torch.tensor([0.0, 0.0, 0.5 + cap_depth], dtype=torch.float32))

    # Build faces
    faces: List[List[int]] = []
    K = circle_segments

    if ring_starts:
        first = ring_starts[0]
        for k in range(K):
            k_next = (k + 1) % K
            faces.append([0, first + k_next, first + k])

    for ridx in range(len(ring_starts) - 1):
        a, b = ring_starts[ridx], ring_starts[ridx + 1]
        for k in range(K):
            k_next = (k + 1) % K
            faces.append([a + k, b + k, a + k_next])
            faces.append([a + k_next, b + k, b + k_next])

    if ring_starts:
        last = ring_starts[-1]
        for k in range(K):
            k_next = (k + 1) % K
            faces.append([last + k, last + k_next, top_pole_idx])

    return torch.stack(verts, dim=0), torch.tensor(faces, dtype=torch.long)


def _get_capsule_template(
    device: torch.device, dtype: torch.dtype, circle_segments: int = 8, cap_segments: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cached unit capsule vertices/faces on requested device/dtype."""
    key = (str(device), dtype, circle_segments, cap_segments)
    if key not in _CAPSULE_TEMPLATE_CACHE:
        v, f = _build_unit_capsule_template(circle_segments, cap_segments)
        _CAPSULE_TEMPLATE_CACHE[key] = (v.to(device=device, dtype=dtype), f.to(device=device))
    return _CAPSULE_TEMPLATE_CACHE[key]


def _get_ref_vectors(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get cached reference vectors for capsule orientation."""
    key = (str(device), dtype)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = (
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        )
    return _REF_CACHE[key]


def _capsule_vertices_from_edges(
    template_vertices: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    radius: float,
    cap_scale: float = 1.5,
) -> torch.Tensor:
    """Vectorized capsule generation for multiple edges.
    
    Args:
        template_vertices: (V, 3) unit capsule vertices (z axis aligned)
        starts, ends: (E, 3) edge endpoints
        radius: scalar radius in world units
        cap_scale: scalar factor for cap protrusion
        
    Returns:
        (E, V, 3) transformed vertices
    """
    axis = ends - starts
    length = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    axis_dir = axis / length

    ref_x, ref_y = _get_ref_vectors(starts.device, starts.dtype)
    ref = torch.where(
        torch.abs((axis_dir * ref_x).sum(dim=-1, keepdim=True)) > 0.99,
        ref_y, ref_x,
    )

    basis_u = F.normalize(torch.cross(axis_dir, ref, dim=-1), dim=-1)
    basis_v = torch.cross(axis_dir, basis_u, dim=-1)
    basis = torch.stack([basis_u, basis_v, axis_dir], dim=-2)

    z = template_vertices[:, 2]
    base_xy = template_vertices[:, :2] * radius

    z_cyl = z.clamp(min=-0.5, max=0.5).unsqueeze(0) * length
    cap_offset = (z.abs() - 0.5).clamp(min=0.0) * radius * cap_scale
    z_scaled = z_cyl + torch.sign(z).unsqueeze(0) * cap_offset

    scaled = torch.cat([
        base_xy.unsqueeze(0).expand(length.shape[0], -1, -1),
        z_scaled.unsqueeze(-1),
    ], dim=-1)

    rotated = torch.einsum("evc,ecd->evd", scaled, basis)
    centers = (starts + ends) * 0.5
    return rotated + centers.unsqueeze(1)


# =============================================================================
# Section 3: Reconstruction Loss Components
# =============================================================================

class L1LossComponent(LossComponent):
    """L1 (MAE) position regression loss."""
    
    name = "l1"
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "gt_xyz"}
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz, gt_xyz = ctx["pred_xyz"], ctx["gt_xyz"]
        fixed_point_mask = ctx.get("fixed_point_mask")
        
        if fixed_point_mask is not None and fixed_point_mask.any():
            keep_mask = ~fixed_point_mask.to(pred_xyz.device)
            pred_xyz, gt_xyz = pred_xyz[:, keep_mask, :], gt_xyz[:, keep_mask, :]
        
        if pred_xyz.numel() == 0:
            return torch.tensor(0.0, device=ctx["pred_xyz"].device)
        return F.l1_loss(pred_xyz, gt_xyz)


class ChamferLossComponent(LossComponent):
    """Chamfer distance for global shape consistency."""
    
    name = "chamfer"
    
    def __init__(self, use_pytorch3d: bool = True):
        self.use_pytorch3d = use_pytorch3d
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "gt_xyz"}
    
    def is_available(self, ctx: Dict[str, Any]) -> bool:
        return super().is_available(ctx) and (_PYTORCH3D_AVAILABLE if self.use_pytorch3d else True)
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz, gt_xyz = ctx["pred_xyz"], ctx["gt_xyz"]
        fixed_point_mask = ctx.get("fixed_point_mask")
        
        if fixed_point_mask is not None and fixed_point_mask.any():
            keep_mask = ~fixed_point_mask.to(pred_xyz.device)
            pred_xyz, gt_xyz = pred_xyz[:, keep_mask, :], gt_xyz[:, keep_mask, :]
        
        if pred_xyz.numel() == 0:
            return torch.tensor(0.0, device=ctx["pred_xyz"].device)
        if not _PYTORCH3D_AVAILABLE:
            return torch.tensor(float("nan"), device=ctx["pred_xyz"].device)

        val, _ = chamfer_distance(pred_xyz.float(), gt_xyz.float(), point_reduction="mean", batch_reduction="mean")
        return val


class DirectionLossComponent(LossComponent):
    """Edge direction alignment loss for bone orientations."""
    
    name = "direction"
    
    def __init__(self, mode: str = "cos", eps: float = 1e-6):
        self.mode = mode
        self.eps = eps
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "gt_xyz", "edge_index"}
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz, gt_xyz = ctx["pred_xyz"], ctx["gt_xyz"]
        edge_index = ctx["edge_index"].to(pred_xyz.device)
        
        v_gt = F.normalize(gt_xyz[:, edge_index[1]] - gt_xyz[:, edge_index[0]] + self.eps, dim=-1)
        v_pd = F.normalize(pred_xyz[:, edge_index[1]] - pred_xyz[:, edge_index[0]] + self.eps, dim=-1)
        
        if self.mode == "l2":
            return (v_pd - v_gt).norm(p=2, dim=-1).mean()
        cos_sim = (v_gt * v_pd).sum(dim=-1).clamp(-1, 1)
        return (1.0 - cos_sim).mean()


class EdgeLenLossComponent(LossComponent):
    """Edge length consistency loss for skeleton proportions."""
    
    name = "edge_len"
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "gt_xyz", "edge_index"}
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz, gt_xyz = ctx["pred_xyz"], ctx["gt_xyz"]
        edge_index = ctx["edge_index"].to(pred_xyz.device)
        
        len_gt = (gt_xyz[:, edge_index[0]] - gt_xyz[:, edge_index[1]]).norm(dim=-1)
        len_pd = (pred_xyz[:, edge_index[0]] - pred_xyz[:, edge_index[1]]).norm(dim=-1)
        return ((len_pd - len_gt).abs() / (len_gt + self.eps)).mean()


class BoneLossComponent(LossComponent):
    """Bone length regularization vs template rest lengths."""
    
    name = "bone"
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "edge_index", "edge_rest_lengths"}
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz = ctx["pred_xyz"]
        edge_index = ctx["edge_index"].to(pred_xyz.device)
        edge_rest_lengths = ctx["edge_rest_lengths"]
        active_edge_mask = ctx.get("active_edge_mask")
        
        if edge_rest_lengths is None:
            return torch.tensor(0.0, device=pred_xyz.device)
        
        i, j = edge_index
        lengths = torch.linalg.norm(pred_xyz[:, i] - pred_xyz[:, j], dim=-1)
        target = edge_rest_lengths.view(1, -1).to(pred_xyz.device)
        
        rel_err = (lengths - target) / (target + 1e-6)
        if active_edge_mask is not None:
            rel_err = rel_err[:, active_edge_mask.to(pred_xyz.device)]
        return rel_err.pow(2).mean()


# =============================================================================
# Section 4: Flow Matching Loss Components
# =============================================================================

class FlowMatchingMSEComponent(LossComponent):
    """MSE loss between predicted and target velocity."""
    
    name = "loss_fm"
    
    def __init__(self, loss_clamp: float = 100.0):
        self.loss_clamp = loss_clamp
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"v_hat", "v_star"}
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        loss = F.mse_loss(ctx["v_hat"], ctx["v_star"])
        return torch.clamp(loss, max=self.loss_clamp)


class TangentRegularizationComponent(LossComponent):
    """Tangent regularization for rigid body constraint."""
    
    name = "loss_tangent"
    
    def __init__(self, loss_clamp: float = 100.0):
        self.loss_clamp = loss_clamp
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"v_hat", "y_tau", "edge_index"}
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        v_hat, y_tau = ctx["v_hat"], ctx["y_tau"]
        edge_index = ctx["edge_index"].to(v_hat.device)
        
        i, j = edge_index
        diff_y = y_tau[:, i] - y_tau[:, j]
        diff_v = v_hat[:, i] - v_hat[:, j]
        
        orthogonality = (diff_y * diff_v).sum(dim=-1)
        return torch.clamp((orthogonality ** 2).mean(), max=self.loss_clamp)


# =============================================================================
# Section 5: Collision Loss Component
# =============================================================================

class CollisionLossComponent(LossComponent):
    """Penetration loss against scene geometry using capsule SDF.
    
    Computes in scaled world space (default: centimeters) for numerical stability.
    """
    
    name = "collision"
    
    def __init__(
        self,
        margin: float = 0.2,
        capsule_radius: float = 0.8,
        capsule_circle_segments: int = 8,
        capsule_cap_segments: int = 2,
        max_scene_points: Optional[int] = None,
        scale: float = 100.0,
    ):
        self.margin = margin
        self.capsule_radius = capsule_radius
        self.capsule_circle_segments = capsule_circle_segments
        self.capsule_cap_segments = capsule_cap_segments
        self.max_scene_points = max_scene_points
        self.scale = scale
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "edge_index", "scene_pc"}
    
    def is_available(self, ctx: Dict[str, Any]) -> bool:
        if not super().is_available(ctx):
            return False
        scene_pc = ctx.get("scene_pc")
        return scene_pc is not None and scene_pc.numel() > 0 and scene_pc.shape[1] > 0
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz, scene_pc = ctx["pred_xyz"], ctx["scene_pc"]
        edge_index = ctx["edge_index"].to(pred_xyz.device)
        denorm_fn = ctx.get("denorm_fn")
        active_edge_mask = ctx.get("active_edge_mask")
        
        if scene_pc is None or scene_pc.shape[1] == 0:
            return torch.tensor(0.0, device=pred_xyz.device)
        
        # Transform to world space and scale
        pred_w = denorm_fn(pred_xyz) if denorm_fn else pred_xyz
        scene_w = denorm_fn(scene_pc) if denorm_fn else scene_pc
        if self.scale != 1.0:
            pred_w, scene_w = pred_w * self.scale, scene_w * self.scale
        
        batch_size = pred_w.shape[0]
        i, j = edge_index
        
        # Handle edge mask
        if active_edge_mask is not None:
            mask = active_edge_mask.to(pred_w.device)
            active_edges = torch.nonzero(mask, as_tuple=False).squeeze(-1) if mask.dim() == 1 else None
        else:
            active_edges = torch.arange(i.numel(), device=pred_w.device)
        
        if active_edges is not None and active_edges.numel() == 0:
            return torch.tensor(0.0, device=pred_xyz.device)
        
        # Sample scene points
        num_scene_pts = scene_w.shape[1]
        sample_idx = None
        if self.max_scene_points and num_scene_pts > self.max_scene_points:
            sample_idx = torch.randint(0, num_scene_pts, (batch_size, self.max_scene_points), device=pred_w.device)
        
        # Compute collision (with fp32 fallback)
        result = self._compute_inner(pred_w, scene_w, i, j, active_edges, sample_idx, batch_size, use_fp32=False)
        if not torch.isfinite(result):
            result = self._compute_inner(pred_w, scene_w, i, j, active_edges, sample_idx, batch_size, use_fp32=True)
        return result if torch.isfinite(result) else torch.tensor(0.0, device=pred_xyz.device)
    
    def _compute_inner(
        self, pred_w: torch.Tensor, scene_w: torch.Tensor,
        i: torch.Tensor, j: torch.Tensor, active_edges: torch.Tensor,
        sample_idx: Optional[torch.Tensor], batch_size: int, use_fp32: bool = False,
    ) -> torch.Tensor:
        dtype = torch.float32 if use_fp32 else pred_w.dtype
        template_v, template_f = _get_capsule_template(
            pred_w.device, dtype, self.capsule_circle_segments, self.capsule_cap_segments
        )
        
        if active_edges is None or active_edges.numel() == 0:
            return torch.tensor(float('nan'), device=pred_w.device)
        
        # Build capsule vertices
        starts = pred_w[:, i[active_edges]].to(dtype)
        ends = pred_w[:, j[active_edges]].to(dtype)
        num_edges = active_edges.numel()
        num_verts_per_capsule = template_v.shape[0]
        
        verts = _capsule_vertices_from_edges(
            template_v, starts.reshape(-1, 3), ends.reshape(-1, 3), self.capsule_radius, cap_scale=1.5
        ).reshape(batch_size, num_edges * num_verts_per_capsule, 3)
        
        # Build faces
        faces = template_f.unsqueeze(0) + (
            torch.arange(num_edges, device=pred_w.device).view(-1, 1, 1) * num_verts_per_capsule
        )
        faces = faces.reshape(-1, 3)
        
        # Sample scene points
        if sample_idx is not None:
            scene_sampled = torch.gather(scene_w.to(dtype), 1, sample_idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            scene_sampled = scene_w.to(dtype)
        
        # Per-batch SDF computation
        losses = []
        for b in range(batch_size):
            scene_pts = scene_sampled[b]
            if scene_pts.shape[0] == 0:
                continue
            try:
                vertices = verts[b]
                face_vertices = index_vertices_by_faces(vertices, faces)
                sdf, dist_sign, _, _ = compute_sdf(scene_pts, face_vertices)
                signed_dist = sdf * dist_sign.sign().to(dtype)
                penetration = torch.relu(self.margin - signed_dist)
                if torch.isfinite(penetration).all():
                    losses.append(penetration.mean())
            except Exception:
                continue
        
        if not losses:
            return torch.tensor(float('nan'), device=pred_w.device)
        return torch.stack(losses).mean()


# =============================================================================
# Section 6: Component Registration
# =============================================================================

# Register all components at module load time
register_loss_component(L1LossComponent())
register_loss_component(ChamferLossComponent())
register_loss_component(DirectionLossComponent())
register_loss_component(EdgeLenLossComponent())
register_loss_component(BoneLossComponent())
register_loss_component(FlowMatchingMSEComponent())
register_loss_component(TangentRegularizationComponent())
register_loss_component(CollisionLossComponent())


# =============================================================================
# Section 7: Loss Manager Base Class
# =============================================================================

class LossManager:
    """Base class for loss managers that dispatch to registered components."""
    
    def __init__(self, weights: Dict[str, float], component_configs: Optional[Dict[str, Dict]] = None):
        self.weights = {k: float(v) for k, v in weights.items()}
        self.component_configs = component_configs or {}
        
        # Build active components (weight > 0)
        self._active_components: Dict[str, LossComponent] = {}
        for name, weight in self.weights.items():
            if weight > 0:
                component = self._get_or_create_component(name)
                if component is not None:
                    self._active_components[name] = component
    
    def _get_or_create_component(self, name: str) -> Optional[LossComponent]:
        """Get component from registry or create with custom config."""
        cfg = self.component_configs.get(name, {})
        base = get_loss_component(name)
        if base is None:
            logger.warning(f"Loss component '{name}' not found in registry")
            return None
        if not cfg:
            return base
        try:
            return base.__class__(**cfg)
        except TypeError as e:
            logger.warning(f"Failed to create component '{name}' with config {cfg}: {e}")
            return base
    
    def compute_components(self, ctx: Dict[str, Any], return_weighted: bool = False) -> Dict[str, torch.Tensor]:
        """Compute all active loss components."""
        results = {}
        for name, component in self._active_components.items():
            if not component.is_available(ctx):
                continue
            try:
                value = component.compute(ctx)
                if not torch.isfinite(value):
                    value = self._compute_with_fp32(component, ctx)
                if not torch.isfinite(value):
                    continue
                results[name] = value * self.weights[name] if return_weighted else value
            except Exception as e:
                logger.warning(f"Failed to compute component '{name}': {e}")
        return results
    
    def _compute_with_fp32(self, component: LossComponent, ctx: Dict[str, Any]) -> torch.Tensor:
        """Retry computation with fp32 tensors."""
        ctx_fp32 = {
            k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in ctx.items()
        }
        try:
            return component.compute(ctx_fp32)
        except Exception:
            return torch.tensor(float('nan'))
    
    def compute_total(self, ctx: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted total loss and individual components."""
        components = self.compute_components(ctx, return_weighted=False)
        if not components:
            device = ctx.get("pred_xyz", ctx.get("v_hat", torch.zeros(1))).device
            return torch.tensor(0.0, device=device), {}
        
        device = next(iter(components.values())).device
        total = sum(self.weights[name] * val for name, val in components.items())
        return total, components
    
    @property
    def active_component_names(self) -> List[str]:
        return list(self._active_components.keys())


# =============================================================================
# Section 8: High-Level Loss Classes
# =============================================================================

class FlowMatchingLoss(nn.Module):
    """Flow Matching loss with structure-preserving tangent regularization.
    
    L_total = ||v_pred - v_target||^2 + lambda * ||(v_pred_rel . y_curr_rel)||^2
    """
    
    def __init__(
        self,
        edge_index: torch.Tensor,
        lambda_tangent: float = 1.0,
        loss_clamp: float = 100.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index shape mismatch. Expected (2, E), got {edge_index.shape}")
        
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.loss_clamp = float(loss_clamp)
        
        self._weights = weights if weights else {"loss_fm": 1.0, "loss_tangent": float(lambda_tangent)}
        self.lambda_tangent = self._weights.get("loss_tangent", 0.0)
        
        component_configs = {
            "loss_fm": {"loss_clamp": loss_clamp},
            "loss_tangent": {"loss_clamp": loss_clamp},
        }
        self._manager = LossManager(self._weights, component_configs)

    def forward(self, v_hat: torch.Tensor, v_star: torch.Tensor, y_tau: torch.Tensor) -> Dict[str, torch.Tensor]:
        if v_hat.shape != v_star.shape or v_hat.shape != y_tau.shape:
            raise ValueError(f"Shape mismatch: v_hat={v_hat.shape}, v_star={v_star.shape}, y_tau={y_tau.shape}")
        
        ctx = {"v_hat": v_hat, "v_star": v_star, "y_tau": y_tau, "edge_index": self.edge_index}
        total, components = self._manager.compute_total(ctx)
        return {"loss": total, **components}


class DeterministicRegressionLoss(nn.Module):
    """Loss manager for deterministic hand regression model.
    
    Supports curriculum learning via LossScheduler:
    - Stage 1: Position regression (l1, chamfer)
    - Stage 2: Skeletal constraints (direction, edge_len, bone)
    - Stage 3: Collision avoidance
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        config: Optional[Dict] = None,
        fixed_point_indices: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index mismatch. Expected (2, E), got {edge_index.shape}")

        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        cfg = config or {}

        if fixed_point_indices is not None:
            idx = torch.as_tensor(fixed_point_indices, dtype=torch.long)
            self.register_buffer("fixed_point_indices", idx, persistent=False)
        else:
            self.fixed_point_indices = None

        # Parse component configs
        self._component_configs = self._parse_component_configs(cfg)
        
        # Initialize scheduler
        self.scheduler = self._init_scheduler(cfg)
        self._current_epoch = 0
        self._cached_manager: Optional[LossManager] = None
        self._cached_weights: Optional[Dict[str, float]] = None
        
        logger.info(f"DeterministicRegressionLoss initialized: {self.scheduler}")
    
    def _parse_component_configs(self, cfg: Dict) -> Dict[str, Dict]:
        """Parse component-specific configurations."""
        configs: Dict[str, Dict] = {}
        recon = cfg.get("recon", {})
        
        for name in ["l1", "chamfer", "direction", "edge_len"]:
            sub = recon.get(name, recon.get("smooth_l1", {}) if name == "l1" else {})
            if sub.get("enabled", True):
                configs[name] = {k: v for k, v in sub.items() if k not in ("enabled", "weight")}
        
        # Collision config
        configs["collision"] = {
            "margin": float(cfg.get("collision_margin", 0.2)),
            "capsule_radius": float(cfg.get("collision_capsule_radius", 0.8)),
            "capsule_circle_segments": int(cfg.get("collision_capsule_circle_segments", 8)),
            "capsule_cap_segments": int(cfg.get("collision_capsule_cap_segments", 2)),
            "scale": float(cfg.get("collision_scale", 100.0)),
        }
        max_pts = int(cfg.get("collision_max_scene_points", 0))
        if max_pts > 0:
            configs["collision"]["max_scene_points"] = max_pts
        
        return configs
    
    def _init_scheduler(self, cfg: Dict) -> LossScheduler:
        """Initialize loss scheduler with curriculum config."""
        curriculum = cfg.get("curriculum", {})
        
        # Build default weights
        default_weights = DEFAULT_WEIGHTS.copy()
        if "lambda_pos" in cfg:
            default_weights["l1"] = float(cfg["lambda_pos"]) * default_weights["l1"]
        if "lambda_bone" in cfg:
            default_weights["bone"] = float(cfg["lambda_bone"])
        if "lambda_collision" in cfg:
            default_weights["collision"] = float(cfg["lambda_collision"])
        if "weights" in cfg:
            for k, v in cfg["weights"].items():
                key = LEGACY_WEIGHT_ALIASES.get(k, k)
                if key in default_weights:
                    default_weights[key] = float(v)
        
        return LossScheduler(
            enabled=curriculum.get("enabled", False),
            stages=curriculum.get("stages"),
            warmup_epochs=curriculum.get("warmup_epochs", 5),
            default_weights=default_weights,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for curriculum scheduling."""
        self._current_epoch = epoch
        self.scheduler.set_epoch(epoch)
        self._cached_manager = None
        self._cached_weights = None

    def get_current_weights(self) -> Dict[str, float]:
        return self.scheduler.get_weights()
    
    def _get_manager(self) -> LossManager:
        """Get or create LossManager with current weights."""
        current_weights = self.scheduler.get_weights()
        if self._cached_manager is None or self._cached_weights != current_weights:
            self._cached_weights = current_weights.copy()
            self._cached_manager = LossManager(current_weights, self._component_configs)
        return self._cached_manager
    
    def _build_fixed_point_mask(self, num_points: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.fixed_point_indices is None or self.fixed_point_indices.numel() == 0:
            return None
        mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        idx = self.fixed_point_indices.to(device)
        idx = idx[idx < num_points]
        if idx.numel() > 0:
            mask[idx] = True
        return mask

    def forward(
        self,
        pred_xyz: torch.Tensor,
        gt_xyz: torch.Tensor,
        *,
        edge_rest_lengths: Optional[torch.Tensor] = None,
        active_edge_mask: Optional[torch.Tensor] = None,
        scene_pc: Optional[torch.Tensor] = None,
        denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if epoch is not None:
            self.set_epoch(epoch)

        ctx = {
            "pred_xyz": pred_xyz,
            "gt_xyz": gt_xyz,
            "edge_index": self.edge_index,
            "edge_rest_lengths": edge_rest_lengths,
            "active_edge_mask": active_edge_mask,
            "scene_pc": scene_pc,
            "denorm_fn": denorm_fn,
            "fixed_point_mask": self._build_fixed_point_mask(pred_xyz.shape[1], pred_xyz.device),
        }
        
        return self._get_manager().compute_total(ctx)


class HandValidationMetricManager:
    """Computes validation metrics for flow matching and hand reconstruction."""
    
    def __init__(
        self,
        edge_index: torch.Tensor,
        config: Optional[Dict] = None,
        fixed_point_indices: Optional[torch.Tensor] = None,
        recon_scale: float = 1.0,
    ) -> None:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index mismatch. Expected (2, E), got {edge_index.shape}")
        
        self.edge_index = edge_index.long().detach().cpu()
        self.fixed_point_indices = None
        if fixed_point_indices is not None:
            self.fixed_point_indices = torch.as_tensor(fixed_point_indices, dtype=torch.long).detach().cpu()
        
        cfg = config or {}
        self.flow = cfg.get("flow", {})
        self.recon = cfg.get("recon", {})
        self.flow_enabled = self.flow.get("enabled", True)
        self.recon_enabled = self.recon.get("enabled", True)
        self.recon_scale = float(cfg.get("recon_scale", self.recon.get("scale_factor", recon_scale)))
        self.flow_weights = self.flow.get("weights", {"loss": 1.0})
        
        # Build recon manager
        self._recon_weights, self._recon_configs = self._parse_recon_config()
        self._recon_manager = LossManager(self._recon_weights, self._recon_configs) if self.recon_enabled and self._recon_weights else None
        
        # Parse trajectory metrics config
        self._parse_trajectory_config()
    
    def _parse_recon_config(self) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        weights, configs = {}, {}
        for name in ["l1", "chamfer", "direction", "edge_len"]:
            sub = self.recon.get(name, self.recon.get("smooth_l1", {}) if name == "l1" else {})
            if sub.get("enabled", True):
                weights[name] = float(sub.get("weight", 1.0))
                configs[name] = {k: v for k, v in sub.items() if k not in ("enabled", "weight")}
        
        # Collision config (computed separately, not via LossManager)
        collision_cfg = self.recon.get("collision", {})
        self._recon_collision_enabled = collision_cfg.get("enabled", False)
        self._recon_collision_scale = float(collision_cfg.get("scale", 100.0))  # default: meters -> cm
        self._recon_collision_margin = float(collision_cfg.get("margin", 0.2))  # cm
        self._recon_collision_capsule_radius = float(collision_cfg.get("capsule_radius", 0.8))  # cm
        self._recon_collision_max_scene_points = collision_cfg.get("max_scene_points", 2048)
        self._recon_collision_circle_segments = int(collision_cfg.get("capsule_circle_segments", 8))
        self._recon_collision_cap_segments = int(collision_cfg.get("capsule_cap_segments", 2))
        
        return weights, configs

    def compute_flow_metrics(self, loss_dict: Dict[str, torch.Tensor], flow_edge_len_err: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.flow_enabled:
            return {}
        
        metrics = {}
        for name in ["loss", "loss_fm", "loss_tangent"]:
            w = float(self.flow_weights.get(name, 1.0 if name == "loss" else 0.0))
            if w > 0 and name in loss_dict and loss_dict[name] is not None:
                metrics[name] = loss_dict[name].detach()
        
        if flow_edge_len_err is not None and self.flow_weights.get("edge_len_err", 0.0) > 0:
            metrics["edge_len_err"] = flow_edge_len_err.detach()
        
        # Weighted total
        total = sum(
            self.flow_weights.get(n, 1.0 if n == "loss" else 0.0) * v
            for n, v in metrics.items() if torch.isfinite(v)
        )
        metrics["total"] = total
        return metrics

    def compute_recon_metrics(
        self, 
        pred_xyz: torch.Tensor, 
        gt_xyz: torch.Tensor,
        scene_pc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.recon_enabled:
            return {}
        
        metrics = {}
        
        # Scale to world space (typically cm)
        if self.recon_scale != 1.0:
            pred_xyz_scaled = pred_xyz * self.recon_scale
            gt_xyz_scaled = gt_xyz * self.recon_scale
        else:
            pred_xyz_scaled, gt_xyz_scaled = pred_xyz, gt_xyz
        
        # Standard recon metrics via LossManager
        if self._recon_manager is not None:
            ctx = {
                "pred_xyz": pred_xyz_scaled,
                "gt_xyz": gt_xyz_scaled,
                "edge_index": self.edge_index,
                "fixed_point_mask": self._build_fixed_point_mask(pred_xyz.shape[1], pred_xyz.device),
            }
            metrics.update(self._recon_manager.compute_components(ctx, return_weighted=False))
        
        # Collision metric (computed in cm space, same as reg_collision)
        if self._recon_collision_enabled and scene_pc is not None and scene_pc.numel() > 0:
            collision_val = self._compute_recon_collision(pred_xyz, scene_pc)
            if collision_val is not None and torch.isfinite(collision_val):
                metrics["collision"] = collision_val
        
        return metrics
    
    def _compute_recon_collision(
        self, pred_xyz: torch.Tensor, scene_pc: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Compute collision metric between predicted hand and scene point cloud.
        
        Computed in centimeter space for numerical stability.
        """
        if scene_pc.shape[1] == 0:
            return None
        
        # Scale to cm space
        pred_cm = pred_xyz * self._recon_collision_scale
        scene_cm = scene_pc[..., :3] * self._recon_collision_scale  # Only xyz, ignore normals
        
        # Use CollisionLossComponent logic
        collision_component = CollisionLossComponent(
            margin=self._recon_collision_margin,
            capsule_radius=self._recon_collision_capsule_radius,
            capsule_circle_segments=self._recon_collision_circle_segments,
            capsule_cap_segments=self._recon_collision_cap_segments,
            max_scene_points=self._recon_collision_max_scene_points,
            scale=1.0,  # Already scaled to cm
        )
        
        # Build context for collision computation
        ctx = {
            "pred_xyz": pred_cm,
            "scene_pc": scene_cm,
            "edge_index": self.edge_index,
            "denorm_fn": None,  # Already in world space
            "active_edge_mask": None,
        }
        
        if not collision_component.is_available(ctx):
            return None
        
        try:
            return collision_component.compute(ctx)
        except Exception:
            return None
    
    def _build_fixed_point_mask(self, num_points: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.fixed_point_indices is None or self.fixed_point_indices.numel() == 0:
            return None
        mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        idx = self.fixed_point_indices.to(device)
        idx = idx[idx < num_points]
        if idx.numel() > 0:
            mask[idx] = True
        return mask

    def compute_trajectory_metrics(self, traj_stats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute trajectory quality metrics from sampling statistics.
        
        Args:
            traj_stats: Dict containing:
                - arc_length: (B,) cumulative arc length
                - transport_cost: (B,) cumulative ||v||^2 * dt
                - cos_sim_sum: (B,) sum of cosine similarities
                - cos_sim_count: int, number of similarity measurements
                - x0, x1: (B, N, 3) start and end positions
        
        Returns:
            Dict with plr, transport_cost, velocity_consistency metrics
        """
        if not self._traj_enabled or not traj_stats:
            return {}
        
        metrics = {}
        device = traj_stats.get("x1", torch.zeros(1)).device
        
        # PLR: Path Length Ratio = arc_length / euclidean_distance
        if self._traj_plr_enabled:
            arc_length = traj_stats.get("arc_length")
            x0, x1 = traj_stats.get("x0"), traj_stats.get("x1")
            if arc_length is not None and x0 is not None and x1 is not None:
                euclidean = (x1 - x0).norm(dim=-1).mean(dim=-1)  # (B,)
                plr = arc_length / (euclidean + 1e-6)
                metrics["plr"] = plr.mean()
        
        # Transport Cost: integral of ||v||^2 dt (kinetic energy)
        if self._traj_transport_enabled:
            transport = traj_stats.get("transport_cost")
            if transport is not None:
                metrics["transport_cost"] = transport.mean()
        
        # Velocity Consistency: average cosine similarity between adjacent velocities
        if self._traj_vel_cons_enabled:
            cos_sum = traj_stats.get("cos_sim_sum")
            cos_count = traj_stats.get("cos_sim_count", 0)
            if cos_sum is not None and cos_count > 0:
                metrics["velocity_consistency"] = (cos_sum / cos_count).mean()
        
        return metrics

    def _parse_trajectory_config(self) -> None:
        """Parse trajectory metrics configuration from flow config."""
        self._traj_enabled = self.flow.get("trajectory_metrics", True)
        traj_cfg = self.flow.get("trajectory", {})
        
        self._traj_plr_enabled = traj_cfg.get("plr", {}).get("enabled", True) if isinstance(traj_cfg.get("plr"), dict) else traj_cfg.get("plr", True)
        self._traj_transport_enabled = traj_cfg.get("transport_cost", {}).get("enabled", True) if isinstance(traj_cfg.get("transport_cost"), dict) else traj_cfg.get("transport_cost", True)
        self._traj_vel_cons_enabled = traj_cfg.get("velocity_consistency", {}).get("enabled", True) if isinstance(traj_cfg.get("velocity_consistency"), dict) else traj_cfg.get("velocity_consistency", True)


# =============================================================================
# Section 9: Legacy Loss Classes (Backward Compatibility)
# =============================================================================

class TranslationLoss(nn.Module):
    """SmoothL1 translation error."""
    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred_trans: torch.Tensor, gt_trans: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_trans, gt_trans)


class RotationGeodesicLoss(nn.Module):
    """Geodesic loss on SO(3): theta = arccos((tr(R_diff) - 1) / 2)"""
    def forward(self, pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
        m = torch.bmm(gt_rot.transpose(1, 2), pred_rot)
        trace = m.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(cos_theta).mean()


class JointLoss(nn.Module):
    """SmoothL1 loss for joint angles with limit penalties."""
    def __init__(
        self,
        beta: float = 0.01,
        boundary_weight: float = 1.0,
        joint_lower: Optional[torch.Tensor] = None,
        joint_upper: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)
        self.boundary_weight = boundary_weight
        if joint_lower is not None and joint_upper is not None:
            self.register_buffer("joint_lower", joint_lower)
            self.register_buffer("joint_upper", joint_upper)
        else:
            self.joint_lower = self.joint_upper = None

    def forward(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(pred_joints, gt_joints)
        if self.joint_lower is not None and self.boundary_weight > 0:
            violation = F.relu(self.joint_lower - pred_joints) + F.relu(pred_joints - self.joint_upper)
            loss = loss + self.boundary_weight * violation.mean()
        return loss


class ChamferLoss(nn.Module):
    """PyTorch3D Chamfer Distance wrapper."""
    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        if not _PYTORCH3D_AVAILABLE:
            raise ImportError("pytorch3d required for ChamferLoss")
        self.norm = 1 if loss_type == "l1" else 2

    def forward(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
        loss, _ = chamfer_distance(pred_points.float(), gt_points.float(), point_reduction="mean", batch_reduction="mean", norm=self.norm)
        return loss


class PhysicsLoss(nn.Module):
    """Energy-based physical constraints (requires hand_model)."""
    def __init__(
        self,
        hand_model,
        use_self_penetration: bool = True,
        use_joint_limit: bool = True,
        use_finger_finger: bool = True,
        use_finger_palm: bool = False,
    ):
        super().__init__()
        self.hand_model = hand_model
        self.active_constraints: List[Callable[[], torch.Tensor]] = []
        if use_self_penetration:
            self.active_constraints.append(hand_model.cal_self_penetration_energy)
        if use_joint_limit:
            self.active_constraints.append(hand_model.cal_joint_limit_energy)
        if use_finger_finger:
            self.active_constraints.append(hand_model.cal_finger_finger_distance_energy)
        if use_finger_palm:
            self.active_constraints.append(hand_model.cal_finger_palm_distance_energy)

    def forward(self, hand_pose: torch.Tensor) -> torch.Tensor:
        if not self.active_constraints:
            return torch.tensor(0.0, device=hand_pose.device)
        self.hand_model.set_parameters(hand_pose)
        return sum(fn().mean() for fn in self.active_constraints) / len(self.active_constraints)


class TotalLoss(nn.Module):
    """Aggregates all losses with physics ramp-up scheduling."""
    def __init__(
        self,
        hand_model,
        w_trans: float = 1.0,
        w_rot: float = 1.0,
        w_joint: float = 1.0,
        w_chamfer: float = 1.0,
        w_physics: float = 0.01,
        physics_ramp_epochs: int = 50,
        joint_lower: Optional[torch.Tensor] = None,
        joint_upper: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weights = {"trans": w_trans, "rot": w_rot, "joint": w_joint, "chamfer": w_chamfer, "physics": w_physics}
        self.physics_ramp_epochs = physics_ramp_epochs
        self.current_epoch = 0
        self.hand_model = hand_model

        self.trans_loss = TranslationLoss()
        self.rot_loss = RotationGeodesicLoss()
        self.joint_loss = JointLoss(joint_lower=joint_lower, joint_upper=joint_upper)
        self.chamfer_loss = ChamferLoss() if _PYTORCH3D_AVAILABLE else None
        self.physics_loss = PhysicsLoss(hand_model)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_physics_weight(self) -> float:
        if self.physics_ramp_epochs <= 0:
            return self.weights["physics"]
        return self.weights["physics"] * min(1.0, self.current_epoch / self.physics_ramp_epochs)

    def forward(
        self,
        pred_trans: torch.Tensor, pred_rot: torch.Tensor, pred_joints: torch.Tensor,
        gt_trans: torch.Tensor, gt_rot: torch.Tensor, gt_joints: torch.Tensor,
        pred_hand_pose: Optional[torch.Tensor] = None,
        gt_hand_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = pred_trans.device
        losses = {
            "trans": self.trans_loss(pred_trans, gt_trans) if self.weights["trans"] > 0 else torch.tensor(0.0, device=device),
            "rot": self.rot_loss(pred_rot, gt_rot) if self.weights["rot"] > 0 else torch.tensor(0.0, device=device),
            "joint": self.joint_loss(pred_joints, gt_joints) if self.weights["joint"] > 0 else torch.tensor(0.0, device=device),
            "chamfer": torch.tensor(0.0, device=device),
            "physics": torch.tensor(0.0, device=device),
        }

        if self.weights["chamfer"] > 0 and self.chamfer_loss and pred_hand_pose is not None and gt_hand_pose is not None:
            self.hand_model.set_parameters(pred_hand_pose)
            pred_kps = self.hand_model.get_penetration_keypoints()
            self.hand_model.set_parameters(gt_hand_pose)
            gt_kps = self.hand_model.get_penetration_keypoints()
            losses["chamfer"] = self.chamfer_loss(pred_kps, gt_kps)

        if self._get_physics_weight() > 0 and pred_hand_pose is not None:
            losses["physics"] = self.physics_loss(pred_hand_pose)

        losses["total"] = sum(
            losses[k] * (self._get_physics_weight() if k == "physics" else self.weights[k])
            for k in losses if k in self.weights
        )
        return losses
