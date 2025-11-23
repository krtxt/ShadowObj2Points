# models/components/losses.py

import logging
from typing import Dict, Optional, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependency: PyTorch3D
try:
    from pytorch3d.loss import chamfer_distance
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        
        if not self.recon_enabled:
            return {}
        if pred_xyz.shape != gt_xyz.shape:
            raise ValueError("Shape mismatch in reconstruction metrics.")

        device = pred_xyz.device
        metrics = {}
        total = torch.tensor(0.0, device=device)

        # 1. Smooth L1
        if self.cfg_smooth.get("enabled", True):
            val = F.smooth_l1_loss(pred_xyz, gt_xyz, beta=self.cfg_smooth.get("beta", 0.01))
            metrics["smooth_l1"] = val
            total += val * self.cfg_smooth.get("weight", 1.0)

        # 2. Chamfer
        if self.cfg_chamfer.get("enabled", True):
            w = self.cfg_chamfer.get("weight", 1.0)
            if _PYTORCH3D_AVAILABLE and self.cfg_chamfer.get("use_pytorch3d", True):
                val, _ = chamfer_distance(pred_xyz, gt_xyz, point_reduction="mean", batch_reduction="mean")
                metrics["chamfer"] = val
                total += val * w
            else:
                metrics["chamfer"] = torch.tensor(float("nan"), device=device)

        # 3. Edge Direction
        if self.cfg_dir.get("enabled", True):
            w = self.cfg_dir.get("weight", 1.0)
            eps = self.cfg_dir.get("eps", 1e-6)
            edges = self.edge_index.to(device)
            
            v_gt = F.normalize(gt_xyz[:, edges[1]] - gt_xyz[:, edges[0]] + eps, dim=-1)
            v_pd = F.normalize(pred_xyz[:, edges[1]] - pred_xyz[:, edges[0]] + eps, dim=-1)
            
            if self.cfg_dir.get("mode", "cos") == "l2":
                val = (v_pd - v_gt).norm(p=2, dim=-1).mean()
            else:
                cos_sim = (v_gt * v_pd).sum(dim=-1).clamp(-1, 1)
                val = (1.0 - cos_sim).mean()
            
            metrics["direction"] = val
            total += val * w

        # 4. Edge Length
        if self.cfg_edge.get("enabled", True):
            w = self.cfg_edge.get("weight", 1.0)
            eps = self.cfg_edge.get("eps", 1e-6)
            edges = self.edge_index.to(device)
            
            len_gt = (gt_xyz[:, edges[0]] - gt_xyz[:, edges[1]]).norm(dim=-1)
            len_pd = (pred_xyz[:, edges[0]] - pred_xyz[:, edges[1]]).norm(dim=-1)
            
            val = ((len_pd - len_gt).abs() / (len_gt + eps)).mean()
            metrics["edge_len_err"] = val
            total += val * w

        metrics["total"] = total
        return metrics