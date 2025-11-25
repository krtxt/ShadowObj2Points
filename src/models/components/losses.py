# models/components/losses.py

import logging
import math
from typing import Dict, Optional, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from kaolin.metrics.trianglemesh import (
    compute_sdf,
    CUSTOM_index_vertices_by_faces as index_vertices_by_faces,
)

# Optional dependency: PyTorch3D
try:
    from pytorch3d.loss import chamfer_distance
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False

logger = logging.getLogger(__name__)

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
    
    Formula:
        L_total = ||v_pred - v_target||^2 + λ * ||(v_pred_rel · y_curr_rel)||^2
    """
    def __init__(self, edge_index: torch.Tensor, lambda_tangent: float = 1.0) -> None:
        super().__init__()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index shape mismatch. Expected (2, E), got {edge_index.shape}")
            
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.lambda_tangent = float(lambda_tangent)

    def forward(
        self, 
        v_hat: torch.Tensor, 
        v_star: torch.Tensor, 
        y_tau: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if v_hat.shape != v_star.shape or v_hat.shape != y_tau.shape:
            raise ValueError(f"Shape mismatch: v_hat={v_hat.shape}, v_star={v_star.shape}, y_tau={y_tau.shape}")

        # 1. Flow Matching MSE
        loss_fm = F.mse_loss(v_hat, v_star)

        # 2. Tangent Regularization (vectorized)
        loss_tan = torch.tensor(0.0, device=v_hat.device)
        if self.lambda_tangent > 0.0:
            i, j = self.edge_index
            # Relative positions and velocities along edges
            diff_y = y_tau[:, i] - y_tau[:, j]
            diff_v = v_hat[:, i] - v_hat[:, j]
            
            # Penalize velocity components parallel to the bone vector (rigid body constraint)
            # Dot product along the last dim
            orthogonality = (diff_y * diff_v).sum(dim=-1)
            loss_tan = (orthogonality ** 2).mean()

        total_loss = loss_fm + self.lambda_tangent * loss_tan
        
        return {
            "loss": total_loss,
            "loss_fm": loss_fm,
            "loss_tangent": loss_tan
        }


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
        loss, _ = chamfer_distance(
            pred_points, 
            gt_points, 
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
    Configurable via nested dictionary.
    """
    def __init__(self, edge_index: torch.Tensor, config: Optional[Dict] = None) -> None:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index mismatch. Expected (2, E), got {edge_index.shape}")
        
        self.edge_index = edge_index.long().detach().cpu()
        cfg = config or {}

        # --- Parse Config Safely ---
        self.flow = cfg.get("flow", {})
        self.recon = cfg.get("recon", {})
        
        self.flow_enabled = self.flow.get("enabled", True)
        self.recon_enabled = self.recon.get("enabled", True)

        # Pre-fetch weights to avoid dict lookups in loop
        self.flow_weights = self.flow.get("weights", {})
        
        # Recon Sub-configs
        self.cfg_smooth = self.recon.get("smooth_l1", {})
        self.cfg_chamfer = self.recon.get("chamfer", {})
        self.cfg_dir = self.recon.get("direction", {})
        self.cfg_edge = self.recon.get("edge_len", {})

    def compute_flow_metrics(
        self, 
        loss_dict: Dict[str, torch.Tensor], 
        flow_edge_len_err: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        if not self.flow_enabled:
            return {}

        # Collect raw values
        metrics = {
            "loss": loss_dict.get("loss"),
            "loss_fm": loss_dict.get("loss_fm"),
            "loss_tangent": loss_dict.get("loss_tangent"),
            "edge_len_err": flow_edge_len_err
        }
        
        # Filter None values and detach
        metrics = {k: v.detach() for k, v in metrics.items() if v is not None}
        
        # Compute Weighted Total
        total = torch.tensor(0.0, device=flow_edge_len_err.device)
        for name, value in metrics.items():
            w = float(self.flow_weights.get(name, 0.0 if name != "loss" else 1.0))
            if w > 0 and torch.isfinite(value):
                total += w * value
                
        metrics["total"] = total
        return metrics

    def compute_recon_metrics(
        self, 
        pred_xyz: torch.Tensor, 
        gt_xyz: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return _compute_recon_components(
            edge_index=self.edge_index,
            pred_xyz=pred_xyz,
            gt_xyz=gt_xyz,
            cfg_smooth=self.cfg_smooth,
            cfg_chamfer=self.cfg_chamfer,
            cfg_dir=self.cfg_dir,
            cfg_edge=self.cfg_edge,
            recon_enabled=self.recon_enabled,
        )


def _compute_recon_components(
    edge_index: torch.Tensor,
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    cfg_smooth: Dict,
    cfg_chamfer: Dict,
    cfg_dir: Dict,
    cfg_edge: Dict,
    recon_enabled: bool = True,
) -> Dict[str, torch.Tensor]:
    """Shared implementation for reconstruction-style losses/metrics.

    Used both by HandValidationMetricManager and deterministic regression loss
    to avoid code duplication.
    """

    if not recon_enabled:
        return {}
    if pred_xyz.shape != gt_xyz.shape:
        raise ValueError("Shape mismatch in reconstruction metrics.")

    device = pred_xyz.device
    metrics: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=device)

    # 1. Smooth L1
    if cfg_smooth.get("enabled", True):
        val = F.smooth_l1_loss(pred_xyz, gt_xyz, beta=cfg_smooth.get("beta", 0.01))
        metrics["smooth_l1"] = val
        total += val * cfg_smooth.get("weight", 1.0)

    # 2. Chamfer
    if cfg_chamfer.get("enabled", True):
        w = cfg_chamfer.get("weight", 1.0)
        if _PYTORCH3D_AVAILABLE and cfg_chamfer.get("use_pytorch3d", True):
            val, _ = chamfer_distance(pred_xyz, gt_xyz, point_reduction="mean", batch_reduction="mean")
            metrics["chamfer"] = val
            total += val * w
        else:
            metrics["chamfer"] = torch.tensor(float("nan"), device=device)

    # 3. Edge Direction
    if cfg_dir.get("enabled", True):
        w = cfg_dir.get("weight", 1.0)
        eps = cfg_dir.get("eps", 1e-6)
        edges = edge_index.to(device)
        
        v_gt = F.normalize(gt_xyz[:, edges[1]] - gt_xyz[:, edges[0]] + eps, dim=-1)
        v_pd = F.normalize(pred_xyz[:, edges[1]] - pred_xyz[:, edges[0]] + eps, dim=-1)
        
        if cfg_dir.get("mode", "cos") == "l2":
            val = (v_pd - v_gt).norm(p=2, dim=-1).mean()
        else:
            cos_sim = (v_gt * v_pd).sum(dim=-1).clamp(-1, 1)
            val = (1.0 - cos_sim).mean()
        
        metrics["direction"] = val
        total += val * w

    # 4. Edge Length
    if cfg_edge.get("enabled", True):
        w = cfg_edge.get("weight", 1.0)
        eps = cfg_edge.get("eps", 1e-6)
        edges = edge_index.to(device)
        
        len_gt = (gt_xyz[:, edges[0]] - gt_xyz[:, edges[1]]).norm(dim=-1)
        len_pd = (pred_xyz[:, edges[0]] - pred_xyz[:, edges[1]]).norm(dim=-1)
        
        val = ((len_pd - len_gt).abs() / (len_gt + eps)).mean()
        metrics["edge_len_err"] = val
        total += val * w

    metrics["total"] = total
    return metrics


class DeterministicRegressionLoss(nn.Module):
    """Loss manager for deterministic hand regression model.

    Combines reconstruction-style terms (Smooth L1, Chamfer, edge direction,
    and edge length) with optional bone-length regularization and collision
    penalty against the scene point cloud.
    """

    def __init__(self, edge_index: torch.Tensor, config: Optional[Dict] = None) -> None:
        super().__init__()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index mismatch. Expected (2, E), got {edge_index.shape}")

        self.register_buffer("edge_index", edge_index.long(), persistent=False)

        cfg = config or {}

        # Recon-style loss configuration (mirrors HandValidationMetricManager)
        self.recon = cfg.get("recon", {})
        self.recon_enabled = self.recon.get("enabled", True)

        self.cfg_smooth = dict(self.recon.get("smooth_l1", {}))
        self.cfg_chamfer = dict(self.recon.get("chamfer", {}))
        self.cfg_dir = dict(self.recon.get("direction", {}))
        self.cfg_edge = dict(self.recon.get("edge_len", {}))

        # Backwards-compat: allow a top-level lambda_pos to scale SmoothL1
        lambda_pos = float(cfg.get("lambda_pos", 1.0))
        if lambda_pos != 1.0:
            base_w = float(self.cfg_smooth.get("weight", 1.0))
            self.cfg_smooth["weight"] = base_w * lambda_pos

        # Structural / collision terms
        self.lambda_bone = float(cfg.get("lambda_bone", 0.0))
        self.lambda_collision = float(cfg.get("lambda_collision", 0.0))
        self.collision_margin = float(cfg.get("collision_margin", 0.0))
        self.collision_capsule_radius = float(cfg.get("collision_capsule_radius", 0.008))
        self.collision_capsule_circle_segments = int(cfg.get("collision_capsule_circle_segments", 8))
        self.collision_capsule_cap_segments = int(cfg.get("collision_capsule_cap_segments", 2))

        max_pts = int(cfg.get("collision_max_scene_points", 0))
        self.collision_max_scene_points = max_pts if max_pts > 0 else None

    def _compute_bone_loss(
        self,
        pred_xyz: torch.Tensor,
        edge_rest_lengths: Optional[torch.Tensor],
        active_edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.lambda_bone <= 0.0 or edge_rest_lengths is None:
            return torch.tensor(0.0, device=pred_xyz.device)

        i, j = self.edge_index
        lengths = torch.linalg.norm(pred_xyz[:, i] - pred_xyz[:, j], dim=-1)
        target = edge_rest_lengths.view(1, -1)

        rel_err = (lengths - target) / (target + 1e-6)
        if active_edge_mask is not None:
            rel_err = rel_err[:, active_edge_mask]

        return rel_err.pow(2).mean()

    def _compute_collision_loss(
        self,
        pred_xyz: torch.Tensor,
        scene_pc: Optional[torch.Tensor],
        denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        active_edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            self.lambda_collision <= 0.0
            or scene_pc is None
            or scene_pc.shape[1] == 0
        ):
            return torch.tensor(0.0, device=pred_xyz.device)

        # Work in world space for physical margin if denorm_fn is provided
        pred_w = denorm_fn(pred_xyz) if denorm_fn is not None else pred_xyz
        scene_w = denorm_fn(scene_pc) if denorm_fn is not None else scene_pc

        template_v, template_f = _get_capsule_template(
            device=pred_w.device,
            dtype=pred_w.dtype,
            circle_segments=self.collision_capsule_circle_segments,
            cap_segments=self.collision_capsule_cap_segments,
        )

        batch_size = pred_w.shape[0]
        edge_index = self.edge_index.to(pred_w.device)
        i, j = edge_index

        # Normalize active edge mask shape: allow (E,) or (B, E)
        edge_mask_global = None
        if active_edge_mask is not None:
            edge_mask_global = active_edge_mask.to(device=pred_w.device)

        static_active_edges = None
        if edge_mask_global is not None and edge_mask_global.dim() == 1:
            static_active_edges = torch.nonzero(edge_mask_global, as_tuple=False).squeeze(-1)

        faces_cache: Dict[Tuple[str, torch.dtype, int], torch.Tensor] = {}

        losses: List[torch.Tensor] = []
        if static_active_edges is not None:
            active_edges = static_active_edges

            starts_all = pred_w[:, i[active_edges]]  # (B, E, 3)
            ends_all = pred_w[:, j[active_edges]]    # (B, E, 3)
            verts_all = _capsule_vertices_from_edges(
                template_vertices=template_v,
                starts=starts_all.reshape(-1, 3),
                ends=ends_all.reshape(-1, 3),
                radius=self.collision_capsule_radius,
                cap_scale=1.5,
            ).reshape(batch_size, -1, template_v.shape[0], 3)

            num_edges = active_edges.numel()
            num_vertices = template_v.shape[0]
            num_faces = template_f.shape[0]

            cache_key = (str(pred_w.device), pred_w.dtype, num_edges)
            if cache_key not in faces_cache:
                base_faces = template_f.unsqueeze(0) + (
                    torch.arange(num_edges, device=pred_w.device).view(-1, 1, 1) * num_vertices
                )
                faces_cache[cache_key] = base_faces.reshape(-1, 3)
            base_faces = faces_cache[cache_key]

            for b in range(batch_size):
                scene_pts = scene_w[b]
                if scene_pts.shape[0] == 0:
                    continue

                if (
                    self.collision_max_scene_points
                    and scene_pts.shape[0] > self.collision_max_scene_points
                ):
                    perm = torch.randperm(scene_pts.shape[0], device=scene_pts.device)
                    scene_pts = scene_pts[perm[: self.collision_max_scene_points]]

                vertices = verts_all[b].reshape(-1, 3)
                faces = base_faces + (b * num_edges * num_vertices)
                face_vertices = index_vertices_by_faces(vertices, faces)

                sdf, dist_sign, _, _ = compute_sdf(scene_pts, face_vertices)
                signed_dist = sdf * dist_sign.sign().to(sdf.dtype)

                penetration = torch.relu(self.collision_margin - signed_dist)
                losses.append(penetration.mean())
        else:
            for b in range(batch_size):
                edge_mask = edge_mask_global[b] if edge_mask_global is not None else None
                if edge_mask is not None:
                    active_edges = torch.nonzero(edge_mask, as_tuple=False).squeeze(-1)
                else:
                    active_edges = torch.arange(i.numel(), device=pred_w.device)

                if active_edges.numel() == 0:
                    continue

                scene_pts = scene_w[b]
                if scene_pts.shape[0] == 0:
                    continue

                if (
                    self.collision_max_scene_points
                    and scene_pts.shape[0] > self.collision_max_scene_points
                ):
                    perm = torch.randperm(scene_pts.shape[0], device=scene_pts.device)
                    scene_pts = scene_pts[perm[: self.collision_max_scene_points]]

                starts = pred_w[b, i[active_edges]]
                ends = pred_w[b, j[active_edges]]

                verts = _capsule_vertices_from_edges(
                    template_vertices=template_v,
                    starts=starts,
                    ends=ends,
                    radius=self.collision_capsule_radius,
                    cap_scale=1.5,
                )  # (E, V, 3)

                num_edges = verts.shape[0]
                if num_edges == 0:
                    continue

                vertices = verts.reshape(-1, 3)
                faces = template_f.unsqueeze(0) + (
                    torch.arange(num_edges, device=pred_w.device).view(-1, 1, 1) * template_v.shape[0]
                )
                faces = faces.reshape(-1, 3)
                face_vertices = index_vertices_by_faces(vertices, faces)

                sdf, dist_sign, _, _ = compute_sdf(scene_pts, face_vertices)
                signed_dist = sdf * dist_sign.sign().to(sdf.dtype)

                penetration = torch.relu(self.collision_margin - signed_dist)
                losses.append(penetration.mean())

        if not losses:
            return torch.tensor(0.0, device=pred_xyz.device)

        return torch.stack(losses).mean()

    def forward(
        self,
        pred_xyz: torch.Tensor,
        gt_xyz: torch.Tensor,
        *,
        edge_rest_lengths: Optional[torch.Tensor] = None,
        active_edge_mask: Optional[torch.Tensor] = None,
        scene_pc: Optional[torch.Tensor] = None,
        denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total deterministic regression loss and its components.

        Args:
            pred_xyz: Predicted hand keypoints (B, N, 3), typically in normalized space.
            gt_xyz: Ground-truth keypoints (B, N, 3), same space as pred_xyz.
            edge_rest_lengths: Rest bone lengths for regularization (E,).
            active_edge_mask: Optional mask to select active edges for bone and collision losses.
            scene_pc: Scene point cloud (B, P, 3) in same space as pred_xyz.
            denorm_fn: Optional function mapping normalized coords -> world coords
                for collision loss when working in normalized space.
        """

        recon_metrics = _compute_recon_components(
            edge_index=self.edge_index,
            pred_xyz=pred_xyz,
            gt_xyz=gt_xyz,
            cfg_smooth=self.cfg_smooth,
            cfg_chamfer=self.cfg_chamfer,
            cfg_dir=self.cfg_dir,
            cfg_edge=self.cfg_edge,
            recon_enabled=self.recon_enabled,
        )

        device = pred_xyz.device
        zero = torch.tensor(0.0, device=device)

        smooth = recon_metrics.get("smooth_l1", zero)
        chamfer = recon_metrics.get("chamfer", zero)
        direction = recon_metrics.get("direction", zero)
        edge_len_err = recon_metrics.get("edge_len_err", zero)
        recon_total = recon_metrics.get("total", zero)

        bone = self._compute_bone_loss(pred_xyz, edge_rest_lengths, active_edge_mask)
        collision = self._compute_collision_loss(
            pred_xyz,
            scene_pc,
            denorm_fn,
            active_edge_mask=active_edge_mask,
        )

        total = recon_total + self.lambda_bone * bone + self.lambda_collision * collision

        components: Dict[str, torch.Tensor] = {
            "smooth_l1": smooth,
            "chamfer": chamfer,
            "direction": direction,
            "edge_len_err": edge_len_err,
            "bone": bone,
            "collision": collision,
            "recon_total": recon_total,
        }

        return total, components
