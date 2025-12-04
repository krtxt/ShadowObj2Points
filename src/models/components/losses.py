# models/components/losses.py

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

# Optional dependency: PyTorch3D
try:
    from pytorch3d.loss import chamfer_distance
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Loss Component Registry and Base Class
# =============================================================================

class LossComponent(ABC):
    """Base class for all loss components.
    
    Each component is a standalone, stateless function that computes a single
    loss term. Components are registered in LOSS_COMPONENT_REGISTRY and can be
    dynamically selected based on weights configuration.
    """
    
    name: str = ""  # Unique identifier for this component
    
    @abstractmethod
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        """Compute the loss value.
        
        Args:
            ctx: Context dictionary containing all possible inputs:
                - pred_xyz: Predicted keypoints (B, N, 3)
                - gt_xyz: Ground-truth keypoints (B, N, 3)
                - edge_index: Graph edges (2, E)
                - fixed_point_mask: Boolean mask for fixed points (N,)
                - edge_rest_lengths: Rest lengths (E,)
                - active_edge_mask: Boolean mask for active edges (E,)
                - v_hat, v_star, y_tau: For flow matching losses
                - scene_pc: Scene point cloud (B, P, 3)
                - ... other component-specific inputs
                
        Returns:
            Scalar loss tensor (unweighted)
        """
        raise NotImplementedError
    
    @property
    def required_inputs(self) -> Set[str]:
        """Return set of required input keys in ctx."""
        return set()
    
    def is_available(self, ctx: Dict[str, Any]) -> bool:
        """Check if this component can be computed with given context."""
        return all(k in ctx and ctx[k] is not None for k in self.required_inputs)


# Global registry for loss components
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
# Reconstruction Loss Components
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
        pred_xyz = ctx["pred_xyz"]
        gt_xyz = ctx["gt_xyz"]
        fixed_point_mask = ctx.get("fixed_point_mask")
        
        if fixed_point_mask is not None and fixed_point_mask.any():
            # Exclude fixed points
            keep_mask = ~fixed_point_mask.to(pred_xyz.device)
            pred_sel = pred_xyz[:, keep_mask, :]
            gt_sel = gt_xyz[:, keep_mask, :]
        else:
            pred_sel, gt_sel = pred_xyz, gt_xyz
        
        if pred_sel.numel() == 0:
            return torch.tensor(0.0, device=pred_xyz.device)
        
        return F.l1_loss(pred_sel, gt_sel)


class ChamferLossComponent(LossComponent):
    """Chamfer distance for global shape consistency."""
    
    name = "chamfer"
    
    def __init__(self, use_pytorch3d: bool = True):
        self.use_pytorch3d = use_pytorch3d
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "gt_xyz"}
    
    def is_available(self, ctx: Dict[str, Any]) -> bool:
        if not super().is_available(ctx):
            return False
        return _PYTORCH3D_AVAILABLE if self.use_pytorch3d else True
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz = ctx["pred_xyz"]
        gt_xyz = ctx["gt_xyz"]
        fixed_point_mask = ctx.get("fixed_point_mask")
        
        if fixed_point_mask is not None and fixed_point_mask.any():
            keep_mask = ~fixed_point_mask.to(pred_xyz.device)
            pred_sel = pred_xyz[:, keep_mask, :]
            gt_sel = gt_xyz[:, keep_mask, :]
        else:
            pred_sel, gt_sel = pred_xyz, gt_xyz
        
        if pred_sel.numel() == 0:
            return torch.tensor(0.0, device=pred_xyz.device)
        
        if not _PYTORCH3D_AVAILABLE:
            return torch.tensor(float("nan"), device=pred_xyz.device)

        # PyTorch3D KNN kernels expect matching float dtypes (fp32 recommended).
        pred_sel = pred_sel.float()
        gt_sel = gt_sel.float()

        val, _ = chamfer_distance(
            pred_sel, gt_sel, 
            point_reduction="mean", 
            batch_reduction="mean"
        )
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
        pred_xyz = ctx["pred_xyz"]
        gt_xyz = ctx["gt_xyz"]
        edge_index = ctx["edge_index"].to(pred_xyz.device)
        
        v_gt = F.normalize(
            gt_xyz[:, edge_index[1]] - gt_xyz[:, edge_index[0]] + self.eps, 
            dim=-1
        )
        v_pd = F.normalize(
            pred_xyz[:, edge_index[1]] - pred_xyz[:, edge_index[0]] + self.eps, 
            dim=-1
        )
        
        if self.mode == "l2":
            return (v_pd - v_gt).norm(p=2, dim=-1).mean()
        else:  # cos mode
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
        pred_xyz = ctx["pred_xyz"]
        gt_xyz = ctx["gt_xyz"]
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
# Flow Matching Loss Components
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
        v_hat = ctx["v_hat"]
        v_star = ctx["v_star"]
        loss = F.mse_loss(v_hat, v_star)
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
        v_hat = ctx["v_hat"]
        y_tau = ctx["y_tau"]
        edge_index = ctx["edge_index"].to(v_hat.device)
        
        i, j = edge_index
        diff_y = y_tau[:, i] - y_tau[:, j]
        diff_v = v_hat[:, i] - v_hat[:, j]
        
        orthogonality = (diff_y * diff_v).sum(dim=-1)
        loss = (orthogonality ** 2).mean()
        return torch.clamp(loss, max=self.loss_clamp)


# =============================================================================
# Collision Loss Component (requires kaolin)
# =============================================================================

class CollisionLossComponent(LossComponent):
    """Penetration loss against scene geometry using capsule SDF."""
    
    name = "collision"
    
    def __init__(
        self,
        margin: float = 0.0,
        capsule_radius: float = 0.008,
        capsule_circle_segments: int = 8,
        capsule_cap_segments: int = 2,
        max_scene_points: Optional[int] = None,
    ):
        self.margin = margin
        self.capsule_radius = capsule_radius
        self.capsule_circle_segments = capsule_circle_segments
        self.capsule_cap_segments = capsule_cap_segments
        self.max_scene_points = max_scene_points
    
    @property
    def required_inputs(self) -> Set[str]:
        return {"pred_xyz", "edge_index", "scene_pc"}
    
    def is_available(self, ctx: Dict[str, Any]) -> bool:
        if not super().is_available(ctx):
            return False
        scene_pc = ctx.get("scene_pc")
        return scene_pc is not None and scene_pc.numel() > 0 and scene_pc.shape[1] > 0
    
    def compute(self, ctx: Dict[str, Any]) -> torch.Tensor:
        pred_xyz = ctx["pred_xyz"]
        scene_pc = ctx["scene_pc"]
        edge_index = ctx["edge_index"].to(pred_xyz.device)
        denorm_fn = ctx.get("denorm_fn")
        active_edge_mask = ctx.get("active_edge_mask")
        
        if scene_pc is None or scene_pc.shape[1] == 0:
            return torch.tensor(0.0, device=pred_xyz.device)
        
        # Work in world space if denorm_fn provided
        pred_w = denorm_fn(pred_xyz) if denorm_fn is not None else pred_xyz
        scene_w = denorm_fn(scene_pc) if denorm_fn is not None else scene_pc
        
        template_v, template_f = _get_capsule_template(
            device=pred_w.device,
            dtype=pred_w.dtype,
            circle_segments=self.capsule_circle_segments,
            cap_segments=self.capsule_cap_segments,
        )
        
        batch_size = pred_w.shape[0]
        i, j = edge_index
        
        # Handle edge mask
        if active_edge_mask is not None:
            mask = active_edge_mask.to(pred_w.device)
            if mask.dim() == 1:
                active_edges = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            else:
                active_edges = None  # Per-batch mask not supported in simple form
        else:
            active_edges = torch.arange(i.numel(), device=pred_w.device)
        
        if active_edges is not None and active_edges.numel() == 0:
            return torch.tensor(0.0, device=pred_xyz.device)
        
        # Sample scene points if needed
        num_scene_pts = scene_w.shape[1]
        if self.max_scene_points and num_scene_pts > self.max_scene_points:
            sample_idx = torch.randint(0, num_scene_pts, (batch_size, self.max_scene_points), device=pred_w.device)
        else:
            sample_idx = None
        
        losses = []
        if active_edges is not None:
            starts = pred_w[:, i[active_edges]]
            ends = pred_w[:, j[active_edges]]
            verts = _capsule_vertices_from_edges(
                template_vertices=template_v,
                starts=starts.reshape(-1, 3),
                ends=ends.reshape(-1, 3),
                radius=self.capsule_radius,
                cap_scale=1.5,
            ).reshape(batch_size, -1, template_v.shape[0], 3)
            
            num_edges = active_edges.numel()
            faces = template_f.unsqueeze(0) + (
                torch.arange(num_edges, device=pred_w.device).view(-1, 1, 1) * template_v.shape[0]
            )
            faces = faces.reshape(-1, 3)
            
            for b in range(batch_size):
                scene_pts = scene_w[b]
                if sample_idx is not None:
                    scene_pts = scene_pts[sample_idx[b]]
                if scene_pts.shape[0] == 0:
                    continue
                
                vertices = verts[b].reshape(-1, 3)
                face_vertices = index_vertices_by_faces(vertices, faces)
                sdf, dist_sign, _, _ = compute_sdf(scene_pts, face_vertices)
                signed_dist = sdf * dist_sign.sign().to(sdf.dtype)
                penetration = torch.relu(self.margin - signed_dist)
                losses.append(penetration.mean())
        
        if not losses:
            return torch.tensor(0.0, device=pred_xyz.device)
        return torch.stack(losses).mean()


# =============================================================================
# Register Default Components
# =============================================================================

# Reconstruction components
register_loss_component(L1LossComponent())
register_loss_component(ChamferLossComponent())
register_loss_component(DirectionLossComponent())
register_loss_component(EdgeLenLossComponent())
register_loss_component(BoneLossComponent())
register_loss_component(CollisionLossComponent())

# Flow matching components
register_loss_component(FlowMatchingMSEComponent())
register_loss_component(TangentRegularizationComponent())


# =============================================================================
# Loss Manager Base Class
# =============================================================================

class LossManager:
    """Base class for loss managers that dispatch to registered components."""
    
    def __init__(
        self,
        weights: Dict[str, float],
        component_configs: Optional[Dict[str, Dict]] = None,
    ):
        """
        Args:
            weights: Dict mapping component name to weight. Components with weight=0 are skipped.
            component_configs: Optional per-component configuration overrides.
        """
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
        
        # Try to get from registry first
        base_component = get_loss_component(name)
        if base_component is None:
            logger.warning(f"Loss component '{name}' not found in registry, skipping")
            return None
        
        # If no custom config, use the registered instance
        if not cfg:
            return base_component
        
        # Create new instance with custom config
        component_class = base_component.__class__
        try:
            return component_class(**cfg)
        except TypeError as e:
            logger.warning(f"Failed to create component '{name}' with config {cfg}: {e}")
            return base_component
    
    def compute_components(
        self, 
        ctx: Dict[str, Any],
        return_weighted: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute all active loss components.
        
        Args:
            ctx: Context dictionary with all inputs
            return_weighted: If True, return weighted values; otherwise raw values
            
        Returns:
            Dict of component name -> loss value (only for active, computable components)
        """
        results = {}
        for name, component in self._active_components.items():
            if not component.is_available(ctx):
                continue
            try:
                value = component.compute(ctx)
                if return_weighted:
                    value = value * self.weights[name]
                results[name] = value
            except Exception as e:
                logger.warning(f"Failed to compute component '{name}': {e}")
        return results
    
    def compute_total(self, ctx: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted total loss and individual components.
        
        Returns:
            (total_loss, {name: unweighted_value})
        """
        components = self.compute_components(ctx, return_weighted=False)
        
        if not components:
            device = ctx.get("pred_xyz", ctx.get("v_hat", torch.zeros(1))).device
            return torch.tensor(0.0, device=device), {}
        
        device = next(iter(components.values())).device
        total = torch.tensor(0.0, device=device)
        for name, value in components.items():
            total = total + self.weights[name] * value
        
        return total, components
    
    @property
    def active_component_names(self) -> List[str]:
        """Return list of active component names."""
        return list(self._active_components.keys())

# Cached unit capsule mesh (z-axis aligned, length=1 between cap centers, radius=1)
_CAPSULE_TEMPLATE_CACHE: Dict[
    Tuple[str, torch.dtype, int, int], Tuple[torch.Tensor, torch.Tensor]
] = {}
_REF_CACHE: Dict[Tuple[str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def _build_unit_capsule_template(
    circle_segments: int = 8, cap_segments: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a coarse unit capsule mesh centered at origin, aligned with +Z.

    The template uses cap centers at z=-0.5 and z=0.5 with softened poles
    (reduced cap depth) to avoid sharp tips. Faces are oriented outward.
    """
    if circle_segments < 3:
        raise ValueError("circle_segments must be >= 3 for a valid capsule mesh.")
    if cap_segments < 1:
        raise ValueError("cap_segments must be >= 1 for a valid capsule mesh.")

    cap_depth = 0.7  # slightly deeper caps to restore roundness

    angles = torch.linspace(0.0, 2 * math.pi, steps=circle_segments + 1, dtype=torch.float32)[:-1]
    cos, sin = torch.cos(angles), torch.sin(angles)

    verts: List[torch.Tensor] = [torch.tensor([0.0, 0.0, -0.5 - cap_depth], dtype=torch.float32)]  # bottom pole
    ring_starts: List[int] = []

    # Bottom hemisphere (exclude the pole to avoid degeneracy)
    phi_bottom = torch.linspace(0.2, math.pi / 2.0, steps=cap_segments + 2, dtype=torch.float32)[1:]
    phi_bottom = phi_bottom[phi_bottom < (math.pi / 2.0)]
    for phi in phi_bottom:
        z = -0.5 - cap_depth * torch.cos(phi)
        r = torch.sin(phi)
        ring_starts.append(len(verts))
        ring = torch.stack([r * cos, r * sin, torch.full_like(cos, z)], dim=1)
        verts.extend(ring)

    # Cylinder end at z = 0.5 (start is already added by the last bottom ring)
    ring_starts.append(len(verts))
    ring = torch.stack([cos, sin, torch.full_like(cos, 0.5)], dim=1)
    verts.extend(ring)

    # Top hemisphere (exclude the top pole to connect separately)
    phi_top = torch.linspace(math.pi / 2.0, 0.2, steps=cap_segments + 2, dtype=torch.float32)[1:]
    for phi in phi_top:
        z = 0.5 + cap_depth * torch.cos(phi)
        r = torch.sin(phi)
        ring_starts.append(len(verts))
        ring = torch.stack([r * cos, r * sin, torch.full_like(cos, z)], dim=1)
        verts.extend(ring)

    top_pole_idx = len(verts)
    verts.append(torch.tensor([0.0, 0.0, 0.5 + cap_depth], dtype=torch.float32))

    faces: List[List[int]] = []
    K = circle_segments

    # Connect bottom pole to first ring
    if ring_starts:
        first = ring_starts[0]
        for k in range(K):
            k_next = (k + 1) % K
            faces.append([0, first + k_next, first + k])

    # Connect intermediate rings (including cylinder)
    for ridx in range(len(ring_starts) - 1):
        a = ring_starts[ridx]
        b = ring_starts[ridx + 1]
        for k in range(K):
            k_next = (k + 1) % K
            faces.append([a + k, b + k, a + k_next])
            faces.append([a + k_next, b + k, b + k_next])

    # Connect last ring to top pole
    if ring_starts:
        last = ring_starts[-1]
        for k in range(K):
            k_next = (k + 1) % K
            faces.append([last + k, last + k_next, top_pole_idx])

    return torch.stack(verts, dim=0), torch.tensor(faces, dtype=torch.long)


def _get_capsule_template(
    device: torch.device, dtype: torch.dtype, circle_segments: int = 8, cap_segments: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cached unit capsule vertices/faces on the requested device/dtype."""
    key = (str(device), dtype, circle_segments, cap_segments)
    if key not in _CAPSULE_TEMPLATE_CACHE:
        v, f = _build_unit_capsule_template(circle_segments, cap_segments)
        _CAPSULE_TEMPLATE_CACHE[key] = (v.to(device=device, dtype=dtype), f.to(device=device))
    return _CAPSULE_TEMPLATE_CACHE[key]


def _get_ref_vectors(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), dtype)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = (
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        )
    return _REF_CACHE[key]


def _capsule_vertices_from_edge(
    template_vertices: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Transform a unit capsule template to match an edge segment."""
    verts = _capsule_vertices_from_edges(
        template_vertices,
        start.unsqueeze(0),
        end.unsqueeze(0),
        radius,
    )
    return verts[0]


def _capsule_vertices_from_edges(
    template_vertices: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    radius: float,
    cap_scale: float = 1.5,
) -> torch.Tensor:
    """Vectorized capsule generation for multiple edges.

    Args:
        template_vertices: (V, 3) unit capsule vertices (z axis aligned).
        starts, ends: (E, 3) edge endpoints.
        radius: scalar radius in world units.
        cap_scale: scalar factor for cap protrusion (relative to radius).
    Returns:
        (E, V, 3) transformed vertices.
    """
    axis = ends - starts  # (E, 3)
    length = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    axis_dir = axis / length

    ref_x, ref_y = _get_ref_vectors(starts.device, starts.dtype)
    ref = torch.where(
        torch.abs((axis_dir * ref_x).sum(dim=-1, keepdim=True)) > 0.99,
        ref_y,
        ref_x,
    )

    basis_u = F.normalize(torch.cross(axis_dir, ref, dim=-1), dim=-1)
    basis_v = torch.cross(axis_dir, basis_u, dim=-1)
    basis = torch.stack([basis_u, basis_v, axis_dir], dim=-2)  # (E, 3, 3)

    z = template_vertices[:, 2]
    base_xy = template_vertices[:, :2] * radius  # (V, 2)

    z_cyl = z.clamp(min=-0.5, max=0.5).unsqueeze(0) * length  # (E, V)
    cap_offset = (z.abs() - 0.5).clamp(min=0.0) * radius * cap_scale
    z_scaled = z_cyl + torch.sign(z).unsqueeze(0) * cap_offset  # (E, V)

    scaled = torch.cat(
        [
            base_xy.unsqueeze(0).expand(length.shape[0], -1, -1),
            z_scaled.unsqueeze(-1),
        ],
        dim=-1,
    )  # (E, V, 3)

    rotated = torch.einsum("evc,ecd->evd", scaled, basis)
    centers = (starts + ends) * 0.5
    return rotated + centers.unsqueeze(1)


class FlowMatchingLoss(nn.Module):
    """
    Computes Flow Matching loss with structure-preserving tangent regularization.
    
    Uses component registry for modular loss computation. Only computes and returns
    components with weight > 0.
    
    Formula:
        L_total = ||v_pred - v_target||^2 + λ * ||(v_pred_rel · y_curr_rel)||^2
    
    Args:
        edge_index: Graph edge indices of shape (2, E).
        lambda_tangent: Weight for tangent regularization loss (0 to disable).
        loss_clamp: Maximum value for individual loss terms to prevent gradient explosion.
        weights: Optional dict of component weights (overrides lambda_tangent if provided).
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
        
        # Build weights dict
        if weights is not None:
            self._weights = {k: float(v) for k, v in weights.items()}
        else:
            self._weights = {
                "loss_fm": 1.0,
                "loss_tangent": float(lambda_tangent),
            }
        self.lambda_tangent = self._weights.get("loss_tangent", 0.0)
        
        # Build loss manager with component configs
        component_configs = {
            "loss_fm": {"loss_clamp": loss_clamp},
            "loss_tangent": {"loss_clamp": loss_clamp},
        }
        self._manager = LossManager(self._weights, component_configs)

    def forward(
        self, 
        v_hat: torch.Tensor, 
        v_star: torch.Tensor, 
        y_tau: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if v_hat.shape != v_star.shape or v_hat.shape != y_tau.shape:
            raise ValueError(f"Shape mismatch: v_hat={v_hat.shape}, v_star={v_star.shape}, y_tau={y_tau.shape}")

        # Build context for components
        ctx = {
            "v_hat": v_hat,
            "v_star": v_star,
            "y_tau": y_tau,
            "edge_index": self.edge_index,
        }
        
        # Compute active components
        total, components = self._manager.compute_total(ctx)
        
        # Return with "loss" as the total
        result = {"loss": total}
        result.update(components)
        return result


class TranslationLoss(nn.Module):
    """Simple wrapper for SmoothL1 translation error."""
    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred_trans: torch.Tensor, gt_trans: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_trans, gt_trans)


class RotationGeodesicLoss(nn.Module):
    """
    Geodesic loss on SO(3).
    Computes the angle θ needed to rotate pred to gt: θ = arccos((tr(R_diff) - 1) / 2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
        # R_diff = R_gt^T * R_pred
        m = torch.bmm(gt_rot.transpose(1, 2), pred_rot)
        
        # Trace: sum of diagonal elements
        trace = m.diagonal(dim1=-2, dim2=-1).sum(-1)
        
        # Clamp for numerical stability before acos
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        
        return torch.acos(cos_theta).mean()


class JointLoss(nn.Module):
    """SmoothL1 loss for joint angles with physical limit penalties."""
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
        
        # Only register if limits are provided
        if joint_lower is not None and joint_upper is not None:
            self.register_buffer("joint_lower", joint_lower)
            self.register_buffer("joint_upper", joint_upper)
        else:
            self.joint_lower = None
            self.joint_upper = None

    def forward(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(pred_joints, gt_joints)
        
        if self.joint_lower is not None and self.boundary_weight > 0:
            violation_low = F.relu(self.joint_lower - pred_joints)
            violation_high = F.relu(pred_joints - self.joint_upper)
            boundary_loss = (violation_low + violation_high).mean()
            loss = loss + self.boundary_weight * boundary_loss
            
        return loss


class ChamferLoss(nn.Module):
    """Interface for PyTorch3D Chamfer Distance."""
    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        if not _PYTORCH3D_AVAILABLE:
            # We raise at runtime init rather than import time to allow flexible usage
            raise ImportError("pytoch3d is required for ChamferLoss but not found.")
        self.norm = 1 if loss_type == "l1" else 2

    def forward(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
        # Force fp32 to avoid mixed-precision dtype mismatches in KNN kernel.
        pred = pred_points.float()
        gt = gt_points.float()

        loss, _ = chamfer_distance(
            pred, 
            gt, 
            point_reduction="mean", 
            batch_reduction="mean", 
            norm=self.norm
        )
        return loss


class PhysicsLoss(nn.Module):
    """
    Computes energy-based physical constraints.
    Note: Requires a stateful `hand_model` that can update parameters.
    """
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
        
        # Register active energy functions
        self.active_constraints: List[Callable[[], torch.Tensor]] = []
        if use_self_penetration:
            self.active_constraints.append(self.hand_model.cal_self_penetration_energy)
        if use_joint_limit:
            self.active_constraints.append(self.hand_model.cal_joint_limit_energy)
        if use_finger_finger:
            self.active_constraints.append(self.hand_model.cal_finger_finger_distance_energy)
        if use_finger_palm:
            self.active_constraints.append(self.hand_model.cal_finger_palm_distance_energy)

    def forward(self, hand_pose: torch.Tensor) -> torch.Tensor:
        if not self.active_constraints:
            return torch.tensor(0.0, device=hand_pose.device)

        # Update model state (side-effect)
        self.hand_model.set_parameters(hand_pose)
        
        # Accumulate energies
        total_energy = sum(fn().mean() for fn in self.active_constraints)
        return total_energy / len(self.active_constraints)


class TotalLoss(nn.Module):
    """
    Orchestrator class that aggregates all specific losses.
    Handles physics loss ramp-up scheduling.
    """
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
        self.weights = {
            "trans": w_trans,
            "rot": w_rot,
            "joint": w_joint,
            "chamfer": w_chamfer,
            "physics": w_physics
        }
        self.physics_ramp_epochs = physics_ramp_epochs
        self.current_epoch = 0
        self.hand_model = hand_model

        # Initialize sub-modules
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
        factor = min(1.0, self.current_epoch / self.physics_ramp_epochs)
        return self.weights["physics"] * factor

    def _compute_component(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Helper to handle zero-weighted losses efficiently."""
        w = self._get_physics_weight() if name == "physics" else self.weights.get(name, 0.0)
        if w > 0.0:
            return w * value
        # Return detached zero to avoid graph computation if weight is 0
        return torch.zeros_like(value).detach() if value.numel() > 1 else torch.tensor(0.0, device=value.device)

    def forward(
        self,
        pred_trans: torch.Tensor, pred_rot: torch.Tensor, pred_joints: torch.Tensor,
        gt_trans: torch.Tensor, gt_rot: torch.Tensor, gt_joints: torch.Tensor,
        pred_hand_pose: Optional[torch.Tensor] = None,
        gt_hand_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        device = pred_trans.device
        losses = {}

        # 1. Basic Regression Losses
        losses["trans"] = self.trans_loss(pred_trans, gt_trans) if self.weights["trans"] > 0 else torch.tensor(0.0, device=device)
        losses["rot"] = self.rot_loss(pred_rot, gt_rot) if self.weights["rot"] > 0 else torch.tensor(0.0, device=device)
        losses["joint"] = self.joint_loss(pred_joints, gt_joints) if self.weights["joint"] > 0 else torch.tensor(0.0, device=device)

        # 2. Geometric / Physics Losses (Conditional)
        losses["chamfer"] = torch.tensor(0.0, device=device)
        if self.weights["chamfer"] > 0 and self.chamfer_loss and pred_hand_pose is not None and gt_hand_pose is not None:
            # We must use hand_model to get keypoints. Note: side-effects on hand_model state.
            self.hand_model.set_parameters(pred_hand_pose)
            pred_kps = self.hand_model.get_penetration_keypoints()
            self.hand_model.set_parameters(gt_hand_pose)
            gt_kps = self.hand_model.get_penetration_keypoints()
            losses["chamfer"] = self.chamfer_loss(pred_kps, gt_kps)

        losses["physics"] = torch.tensor(0.0, device=device)
        if self._get_physics_weight() > 0 and pred_hand_pose is not None:
            losses["physics"] = self.physics_loss(pred_hand_pose)

        # 3. Aggregate
        total = sum(
            (losses[k] * (self._get_physics_weight() if k == "physics" else self.weights[k]))
            for k in losses if k in self.weights
        )
        losses["total"] = total
        
        return losses


class HandValidationMetricManager:
    """
    Computes validation metrics for Flow Matching and Hand Reconstruction.
    
    Uses component registry for modular metric computation. Only computes and returns
    components with weight > 0 in config.
    """
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
        self.fixed_point_mask = None
        if fixed_point_indices is not None:
            idx = torch.as_tensor(fixed_point_indices, dtype=torch.long)
            self.fixed_point_indices = idx.detach().cpu()
        cfg = config or {}

        # --- Parse Config Safely ---
        self.flow = cfg.get("flow", {})
        self.recon = cfg.get("recon", {})
        
        self.flow_enabled = self.flow.get("enabled", True)
        self.recon_enabled = self.recon.get("enabled", True)
        self.recon_scale = float(cfg.get("recon_scale", self.recon.get("scale_factor", recon_scale)))

        # Flow weights
        self.flow_weights = self.flow.get("weights", {"loss": 1.0})
        
        # Build recon weights from config (enabled=True means weight=1.0 by default)
        self._recon_weights = {}
        self._recon_component_configs = {}
        
        for name in ["l1", "chamfer", "direction", "edge_len"]:
            sub_cfg = self.recon.get(name, self.recon.get("smooth_l1", {}) if name == "l1" else {})
            if sub_cfg.get("enabled", True):
                self._recon_weights[name] = float(sub_cfg.get("weight", 1.0))
                # Extract component-specific params
                self._recon_component_configs[name] = {
                    k: v for k, v in sub_cfg.items() 
                    if k not in ("enabled", "weight")
                }
        
        # Build recon manager
        if self.recon_enabled and self._recon_weights:
            self._recon_manager = LossManager(self._recon_weights, self._recon_component_configs)
        else:
            self._recon_manager = None

    def compute_flow_metrics(
        self, 
        loss_dict: Dict[str, torch.Tensor], 
        flow_edge_len_err: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        if not self.flow_enabled:
            return {}

        # Only include metrics with weight > 0
        metrics = {}
        for name in ["loss", "loss_fm", "loss_tangent"]:
            w = float(self.flow_weights.get(name, 1.0 if name == "loss" else 0.0))
            if w > 0 and name in loss_dict and loss_dict[name] is not None:
                metrics[name] = loss_dict[name].detach()
        
        # edge_len_err is always useful
        if flow_edge_len_err is not None:
            w = float(self.flow_weights.get("edge_len_err", 0.0))
            if w > 0:
                metrics["edge_len_err"] = flow_edge_len_err.detach()
        
        # Compute Weighted Total
        total = torch.tensor(0.0, device=flow_edge_len_err.device)
        for name, value in metrics.items():
            w = float(self.flow_weights.get(name, 1.0 if name == "loss" else 0.0))
            if torch.isfinite(value):
                total += w * value
                
        metrics["total"] = total
        return metrics

    def compute_recon_metrics(
        self, 
        pred_xyz: torch.Tensor, 
        gt_xyz: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if not self.recon_enabled or self._recon_manager is None:
            return {}
        
        if self.recon_scale != 1.0:
            pred_xyz = pred_xyz * self.recon_scale
            gt_xyz = gt_xyz * self.recon_scale
        
        # Build context
        ctx = {
            "pred_xyz": pred_xyz,
            "gt_xyz": gt_xyz,
            "edge_index": self.edge_index,
            "fixed_point_mask": self._build_fixed_point_mask(pred_xyz.shape[1], pred_xyz.device),
        }
        
        # Compute using manager (returns unweighted values)
        return self._recon_manager.compute_components(ctx, return_weighted=False)
    
    def _build_fixed_point_mask(self, num_points: int, device: torch.device) -> Optional[torch.Tensor]:
        """Build fixed point mask tensor."""
        if self.fixed_point_indices is None or self.fixed_point_indices.numel() == 0:
            return None
        mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        idx = self.fixed_point_indices.to(device)
        idx = idx[idx < num_points]
        if idx.numel() > 0:
            mask[idx] = True
        return mask


class DeterministicRegressionLoss(nn.Module):
    """Loss manager for deterministic hand regression model.

    Uses component registry for modular loss computation. Only computes and returns
    components with weight > 0.

    Supports curriculum learning via LossScheduler for multi-stage training:
    - Stage 1 (Reconstruction): Focus on position regression (l1, chamfer)
    - Stage 2 (Structure): Add skeletal constraints (direction, edge_len, bone)
    - Stage 3 (Physics): Add collision avoidance

    Loss Terms (registered components):
    - l1: Position regression (MAE)
    - chamfer: Bidirectional point cloud distance for global shape consistency
    - direction: Edge direction alignment for correct bone orientations
    - edge_len: Edge length consistency for skeleton proportions
    - bone: Bone length regularization vs template
    - collision: Penetration penalty against scene geometry
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
            self.fixed_point_indices = idx
        else:
            self.fixed_point_indices = None

        # Parse recon config for component-specific settings
        recon = cfg.get("recon", {})
        self.recon_enabled = recon.get("enabled", True)
        
        # Build component configs from recon sub-configs
        self._component_configs: Dict[str, Dict] = {}
        for name in ["l1", "chamfer", "direction", "edge_len"]:
            sub_cfg = recon.get(name, recon.get("smooth_l1", {}) if name == "l1" else {})
            if sub_cfg.get("enabled", True):
                self._component_configs[name] = {
                    k: v for k, v in sub_cfg.items() 
                    if k not in ("enabled", "weight")
                }
        
        # Collision config
        collision_cfg = {
            "margin": float(cfg.get("collision_margin", 0.0)),
            "capsule_radius": float(cfg.get("collision_capsule_radius", 0.008)),
            "capsule_circle_segments": int(cfg.get("collision_capsule_circle_segments", 8)),
            "capsule_cap_segments": int(cfg.get("collision_capsule_cap_segments", 2)),
        }
        max_pts = int(cfg.get("collision_max_scene_points", 0))
        if max_pts > 0:
            collision_cfg["max_scene_points"] = max_pts
        self._component_configs["collision"] = collision_cfg

        # Initialize LossScheduler for curriculum learning
        curriculum_cfg = cfg.get("curriculum", {})
        curriculum_enabled = curriculum_cfg.get("enabled", False)

        # Build default weights from config
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

        self.scheduler = LossScheduler(
            enabled=curriculum_enabled,
            stages=curriculum_cfg.get("stages", None),
            warmup_epochs=curriculum_cfg.get("warmup_epochs", 5),
            default_weights=default_weights,
        )

        self._current_epoch = 0
        self._cached_manager: Optional[LossManager] = None
        self._cached_weights: Optional[Dict[str, float]] = None
        logger.info(f"DeterministicRegressionLoss initialized: {self.scheduler}")

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for curriculum scheduling."""
        self._current_epoch = epoch
        self.scheduler.set_epoch(epoch)
        # Invalidate cached manager when weights might change
        self._cached_manager = None
        self._cached_weights = None

    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights based on training progress."""
        return self.scheduler.get_weights()
    
    def _get_manager(self) -> LossManager:
        """Get or create LossManager with current weights."""
        current_weights = self.scheduler.get_weights()
        # Rebuild manager if weights changed
        if self._cached_manager is None or self._cached_weights != current_weights:
            self._cached_weights = current_weights.copy()
            self._cached_manager = LossManager(current_weights, self._component_configs)
        return self._cached_manager
    
    def _build_fixed_point_mask(self, num_points: int, device: torch.device) -> Optional[torch.Tensor]:
        """Build fixed point mask tensor."""
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
        """Compute total deterministic regression loss and its components.

        Uses component registry for modular loss computation. Only computes and
        returns components with weight > 0.

        Args:
            pred_xyz: Predicted hand keypoints (B, N, 3), typically in normalized space.
            gt_xyz: Ground-truth keypoints (B, N, 3), same space as pred_xyz.
            edge_rest_lengths: Rest bone lengths for regularization (E,).
            active_edge_mask: Optional mask to select active edges for bone and collision losses.
            scene_pc: Scene point cloud (B, P, 3) in same space as pred_xyz.
            denorm_fn: Optional function mapping normalized coords -> world coords
                for collision loss when working in normalized space.
            epoch: Current training epoch for curriculum scheduling.

        Returns:
            total: Weighted sum of all loss terms.
            components: Dictionary of unweighted individual loss values (only active components).
        """
        # Update epoch if provided
        if epoch is not None:
            self.set_epoch(epoch)

        # Build context for components
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
        
        # Use manager to compute active components
        manager = self._get_manager()
        total, components = manager.compute_total(ctx)
        
        return total, components
