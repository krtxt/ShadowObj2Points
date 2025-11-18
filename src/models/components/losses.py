"""Loss function modules for hand pose estimation and flow matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

try:
    from pytorch3d.loss import chamfer_distance
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available. Chamfer distance will not work.")

class FlowMatchingLoss(nn.Module):
    """Flow matching loss with optional tangent-space regularization.

    For HandFlowMatchingDiT:
      - Flow matching term: MSE(v_hat, v_star)
      - Tangent constraint term: For each edge (i,j), penalizes ((y_i - y_j) dot (v_i - v_j))^2

    Args:
        edge_index: Edge connectivity of shape (2, E)
        lambda_tangent: Weight of the tangent-space regularization term
    """
    def __init__(self, edge_index: torch.Tensor, lambda_tangent: float = 1.0) -> None:
        super().__init__()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
            )
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.lambda_tangent = float(lambda_tangent)

    def forward(
        self,
        v_hat: torch.Tensor,
        v_star: torch.Tensor,
        y_tau: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow-matching loss and tangent regularization.
        
        Args:
            v_hat: Predicted velocity of shape (B, N, 3)
            v_star: Target velocity of shape (B, N, 3)
            y_tau: Current keypoints of shape (B, N, 3)
            
        Returns:
            Dictionary with keys: loss, loss_fm, loss_tangent
        """
        if v_hat.shape != v_star.shape:
            raise ValueError(f"v_hat and v_star must have the same shape, got {v_hat.shape} vs {v_star.shape}")
        if v_hat.shape != y_tau.shape:
            raise ValueError(f"v_hat and y_tau must have the same shape, got {v_hat.shape} vs {y_tau.shape}")
        loss_fm = F.mse_loss(v_hat, v_star)
        if self.lambda_tangent > 0.0:
            i, j = self.edge_index
            diff_y = y_tau[:, i, :] - y_tau[:, j, :]
            diff_v = v_hat[:, i, :] - v_hat[:, j, :]
            residual = (diff_y * diff_v).sum(-1)
            loss_tan = (residual ** 2).mean()
        else:
            loss_tan = torch.zeros((), device=v_hat.device, dtype=v_hat.dtype)
        loss = loss_fm + self.lambda_tangent * loss_tan
        return {"loss": loss, "loss_fm": loss_fm, "loss_tangent": loss_tan}


class TranslationLoss(nn.Module):
    """Translation loss using SmoothL1.
    
    Args:
        beta: Smoothing parameter for SmoothL1Loss
    """
    def __init__(self, beta: float = 0.01):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred_trans: torch.Tensor, gt_trans: torch.Tensor) -> torch.Tensor:
        """Compute translation loss.
        
        Args:
            pred_trans: Predicted translation of shape (B, 3)
            gt_trans: Ground truth translation of shape (B, 3)
            
        Returns:
            Translation loss (scalar)
        """
        return self.loss_fn(pred_trans, gt_trans)

class RotationGeodesicLoss(nn.Module):
    """Geodesic loss for rotation matrices."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance between rotation matrices.
        
        Args:
            pred_rot: Predicted rotation matrices of shape (B, 3, 3)
            gt_rot: Ground truth rotation matrices of shape (B, 3, 3)
            
        Returns:
            Geodesic distance loss in radians (scalar)
        """
        M = torch.bmm(gt_rot.transpose(1, 2), pred_rot)
        trace = M.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1.0) / 2.0
        eps = 1e-5
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        geodesic_dist = torch.acos(cos_theta)
        return geodesic_dist.mean()


class JointLoss(nn.Module):
    """Joint angle loss with SmoothL1 and boundary penalty.
    
    Args:
        beta: Smoothing parameter for SmoothL1Loss
        boundary_weight: Weight for boundary violation penalty
        joint_lower: Lower joint limits of shape (num_joints,)
        joint_upper: Upper joint limits of shape (num_joints,)
    """
    def __init__(
        self,
        beta: float = 0.01,
        boundary_weight: float = 1.0,
        joint_lower: Optional[torch.Tensor] = None,
        joint_upper: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)
        self.boundary_weight = boundary_weight
        if joint_lower is not None and joint_upper is not None:
            self.register_buffer("joint_lower", joint_lower)
            self.register_buffer("joint_upper", joint_upper)
        else:
            self.joint_lower = None
            self.joint_upper = None

    def forward(self, pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
        """Compute joint angle loss with boundary penalties.
        
        Args:
            pred_joints: Predicted joint angles of shape (B, num_joints)
            gt_joints: Ground truth joint angles of shape (B, num_joints)
            
        Returns:
            Joint loss (scalar)
        """
        base_loss = self.loss_fn(pred_joints, gt_joints)
        if self.joint_lower is not None and self.joint_upper is not None:
            lower_violation = F.relu(self.joint_lower - pred_joints)
            upper_violation = F.relu(pred_joints - self.joint_upper)
            boundary_loss = (lower_violation + upper_violation).mean()
            return base_loss + self.boundary_weight * boundary_loss
        else:
            return base_loss


class ChamferLoss(nn.Module):
    """Chamfer distance loss for reconstruction quality assessment.
    
    Args:
        loss_type: Distance norm ('l1' or 'l2')
    """
    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        if not _PYTORCH3D_AVAILABLE:
            raise ImportError("pytorch3d is required for ChamferLoss")
        self.loss_type = loss_type

    def forward(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
        """Compute Chamfer distance between point clouds.
        
        Args:
            pred_points: Predicted point cloud of shape (B, N1, 3)
            gt_points: Ground truth point cloud of shape (B, N2, 3)
            
        Returns:
            Chamfer distance loss (scalar)
        """
        if self.loss_type == "l1":
            loss, _ = chamfer_distance(
                pred_points, gt_points,
                point_reduction="mean", batch_reduction="mean", norm=1,
            )
        else:
            loss, _ = chamfer_distance(
                pred_points, gt_points,
                point_reduction="mean", batch_reduction="mean", norm=2,
            )
        return loss


class PhysicsLoss(nn.Module):
    """Physics-based loss for self-collision, joint limits, and constraints.
    
    Args:
        hand_model: HandModel instance
        use_self_penetration: Whether to use self-collision loss
        use_joint_limit: Whether to use joint limit loss
        use_finger_finger: Whether to use finger-finger distance loss
        use_finger_palm: Whether to use finger-palm distance loss
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
        self.use_self_penetration = use_self_penetration
        self.use_joint_limit = use_joint_limit
        self.use_finger_finger = use_finger_finger
        self.use_finger_palm = use_finger_palm

    def forward(self, hand_pose: torch.Tensor) -> torch.Tensor:
        """Compute physics-based energy terms.
        
        Args:
            hand_pose: Hand pose parameters of shape (B, pose_dim)
            
        Returns:
            Physics loss (scalar)
        """
        self.hand_model.set_parameters(hand_pose)
        total_loss = 0.0
        count = 0
        if self.use_self_penetration:
            total_loss = total_loss + self.hand_model.cal_self_penetration_energy().mean()
            count += 1
        if self.use_joint_limit:
            total_loss = total_loss + self.hand_model.cal_joint_limit_energy().mean()
            count += 1
        if self.use_finger_finger:
            total_loss = total_loss + self.hand_model.cal_finger_finger_distance_energy().mean()
            count += 1
        if self.use_finger_palm:
            total_loss = total_loss + self.hand_model.cal_finger_palm_distance_energy().mean()
            count += 1
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=hand_pose.device)


class TotalLoss(nn.Module):
    """Unified loss function combining all loss components.
    
    Args:
        hand_model: HandModel instance
        w_trans: Translation loss weight
        w_rot: Rotation loss weight
        w_joint: Joint loss weight
        w_chamfer: Chamfer loss weight
        w_physics: Physics loss weight (final value)
        physics_ramp_epochs: Number of epochs to ramp up physics loss weight
        joint_lower: Lower joint limits
        joint_upper: Upper joint limits
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
        self.hand_model = hand_model
        self.w_trans = w_trans
        self.w_rot = w_rot
        self.w_joint = w_joint
        self.w_chamfer = w_chamfer
        self.w_physics = w_physics
        self.physics_ramp_epochs = physics_ramp_epochs
        self.translation_loss = TranslationLoss()
        self.rotation_loss = RotationGeodesicLoss()
        self.joint_loss = JointLoss(joint_lower=joint_lower, joint_upper=joint_upper)
        self.chamfer_loss = ChamferLoss()
        self.physics_loss = PhysicsLoss(hand_model)
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set current epoch for physics loss ramp-up."""
        self.current_epoch = epoch

    def get_physics_weight(self) -> float:
        """Compute current physics loss weight with ramp-up schedule."""
        if self.physics_ramp_epochs <= 0:
            return self.w_physics
        ramp_progress = min(1.0, self.current_epoch / self.physics_ramp_epochs)
        return self.w_physics * ramp_progress

    def forward(
        self,
        pred_trans: torch.Tensor,
        pred_rot: torch.Tensor,
        pred_joints: torch.Tensor,
        gt_trans: torch.Tensor,
        gt_rot: torch.Tensor,
        gt_joints: torch.Tensor,
        pred_hand_pose: Optional[torch.Tensor] = None,
        gt_hand_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components and return as dictionary.
        
        Args:
            pred_trans: Predicted translation of shape (B, 3)
            pred_rot: Predicted rotation matrices of shape (B, 3, 3)
            pred_joints: Predicted joint angles of shape (B, num_joints)
            gt_trans: Ground truth translation of shape (B, 3)
            gt_rot: Ground truth rotation matrices of shape (B, 3, 3)
            gt_joints: Ground truth joint angles of shape (B, num_joints)
            pred_hand_pose: Predicted full hand pose of shape (B, pose_dim)
            gt_hand_pose: Ground truth full hand pose of shape (B, pose_dim)
            
        Returns:
            Dictionary containing all loss components and total loss
        """
        losses = {}
        if self.w_trans > 0:
            losses["trans"] = self.translation_loss(pred_trans, gt_trans)
        else:
            losses["trans"] = torch.tensor(0.0, device=pred_trans.device)
        if self.w_rot > 0:
            losses["rot"] = self.rotation_loss(pred_rot, gt_rot)
        else:
            losses["rot"] = torch.tensor(0.0, device=pred_trans.device)
        if self.w_joint > 0:
            losses["joint"] = self.joint_loss(pred_joints, gt_joints)
        else:
            losses["joint"] = torch.tensor(0.0, device=pred_trans.device)
        if self.w_chamfer > 0 and pred_hand_pose is not None and gt_hand_pose is not None:
            self.hand_model.set_parameters(pred_hand_pose)
            pred_keypoints = self.hand_model.get_penetration_keypoints()
            self.hand_model.set_parameters(gt_hand_pose)
            gt_keypoints = self.hand_model.get_penetration_keypoints()
            losses["chamfer"] = self.chamfer_loss(pred_keypoints, gt_keypoints)
        else:
            losses["chamfer"] = torch.tensor(0.0, device=pred_trans.device)
        if pred_hand_pose is not None:
            physics_weight = self.get_physics_weight()
            if physics_weight > 0:
                losses["physics"] = self.physics_loss(pred_hand_pose)
            else:
                losses["physics"] = torch.tensor(0.0, device=pred_trans.device)
        else:
            losses["physics"] = torch.tensor(0.0, device=pred_trans.device)
        physics_weight = self.get_physics_weight()
        losses["total"] = (
            self.w_trans * losses["trans"]
            + self.w_rot * losses["rot"]
            + self.w_joint * losses["joint"]
            + self.w_chamfer * losses["chamfer"]
            + physics_weight * losses["physics"]
        )
        return losses


if __name__ == "__main__":
    print("Testing loss functions...")
    trans_loss = TranslationLoss()
    pred_trans = torch.randn(4, 3)
    gt_trans = torch.randn(4, 3)
    loss = trans_loss(pred_trans, gt_trans)
    print(f"Translation loss: {loss.item():.6f}")
    rot_loss = RotationGeodesicLoss()
    pred_rot = torch.randn(4, 3, 3)
    gt_rot = torch.randn(4, 3, 3)
    loss = rot_loss(pred_rot, gt_rot)
    print(f"Rotation loss: {loss.item():.6f} rad ({loss.item() * 180 / 3.14159:.2f} deg)")
    joint_loss = JointLoss()
    pred_joints = torch.randn(4, 16)
    gt_joints = torch.randn(4, 16)
    loss = joint_loss(pred_joints, gt_joints)
    print(f"Joint loss: {loss.item():.6f}")
