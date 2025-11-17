"""
损失函数模块
包含参数损失、重建损失和物理先验损失
"""
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


class TranslationLoss(nn.Module):
    """平移损失（SmoothL1）"""
    
    def __init__(self, beta: float = 0.01):
        """
        Args:
            beta: SmoothL1Loss 的平滑参数
        """
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)
    
    def forward(
        self,
        pred_trans: torch.Tensor,
        gt_trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_trans: (B, 3) 预测平移
            gt_trans: (B, 3) GT 平移
        
        Returns:
            loss: 标量损失
        """
        return self.loss_fn(pred_trans, gt_trans)


class RotationGeodesicLoss(nn.Module):
    """旋转测地线损失"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred_rot: torch.Tensor,
        gt_rot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_rot: (B, 3, 3) 预测旋转矩阵
            gt_rot: (B, 3, 3) GT 旋转矩阵
        
        Returns:
            loss: 标量损失（弧度）
        """
        # 计算 R_gt^T @ R_pred
        M = torch.bmm(gt_rot.transpose(1, 2), pred_rot)  # (B, 3, 3)
        
        # 计算 trace
        trace = M.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  # (B,)
        
        # 计算 cos(theta) = (trace(M) - 1) / 2
        cos_theta = (trace - 1.0) / 2.0

        # 稳健裁剪到 (-1+eps, 1-eps) 避免在 ±1 处 acos 梯度为无穷
        eps = 1e-5
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        
        # 计算测地线距离（弧度）
        geodesic_dist = torch.acos(cos_theta)  # (B,)
        
        return geodesic_dist.mean()


class JointLoss(nn.Module):
    """关节角损失（SmoothL1 + 越界惩罚）"""
    
    def __init__(
        self,
        beta: float = 0.01,
        boundary_weight: float = 1.0,
        joint_lower: Optional[torch.Tensor] = None,
        joint_upper: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            beta: SmoothL1Loss 的平滑参数
            boundary_weight: 越界惩罚权重
            joint_lower: (num_joints,) 关节下限
            joint_upper: (num_joints,) 关节上限
        """
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)
        self.boundary_weight = boundary_weight
        
        if joint_lower is not None and joint_upper is not None:
            self.register_buffer("joint_lower", joint_lower)
            self.register_buffer("joint_upper", joint_upper)
        else:
            self.joint_lower = None
            self.joint_upper = None
    
    def forward(
        self,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_joints: (B, num_joints) 预测关节角
            gt_joints: (B, num_joints) GT 关节角
        
        Returns:
            loss: 标量损失
        """
        # 基础损失
        base_loss = self.loss_fn(pred_joints, gt_joints)
        
        # 越界惩罚
        if self.joint_lower is not None and self.joint_upper is not None:
            lower_violation = F.relu(self.joint_lower - pred_joints)
            upper_violation = F.relu(pred_joints - self.joint_upper)
            boundary_loss = (lower_violation + upper_violation).mean()
            
            return base_loss + self.boundary_weight * boundary_loss
        else:
            return base_loss


class ChamferLoss(nn.Module):
    """Chamfer 距离损失（用于重建质量评估）"""
    
    def __init__(self, loss_type: str = "l2"):
        """
        Args:
            loss_type: 'l1' 或 'l2'
        """
        super().__init__()
        if not _PYTORCH3D_AVAILABLE:
            raise ImportError("pytorch3d is required for ChamferLoss")
        self.loss_type = loss_type
    
    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_points: (B, N1, 3) 预测点云
            gt_points: (B, N2, 3) GT 点云
        
        Returns:
            loss: 标量损失
        """
        # 使用 pytorch3d 的 chamfer_distance
        # 返回 (loss, loss_normals)，我们只用第一个
        if self.loss_type == "l1":
            loss, _ = chamfer_distance(
                pred_points,
                gt_points,
                point_reduction="mean",
                batch_reduction="mean",
                norm=1,
            )
        else:  # l2
            loss, _ = chamfer_distance(
                pred_points,
                gt_points,
                point_reduction="mean",
                batch_reduction="mean",
                norm=2,
            )
        
        return loss


class PhysicsLoss(nn.Module):
    """物理先验损失（自碰撞、关节限位等）"""
    
    def __init__(
        self,
        hand_model,
        use_self_penetration: bool = True,
        use_joint_limit: bool = True,
        use_finger_finger: bool = True,
        use_finger_palm: bool = False,
    ):
        """
        Args:
            hand_model: HandModel 实例
            use_self_penetration: 是否使用自碰撞损失
            use_joint_limit: 是否使用关节限位损失
            use_finger_finger: 是否使用手指间距离损失
            use_finger_palm: 是否使用手指-手掌距离损失
        """
        super().__init__()
        self.hand_model = hand_model
        self.use_self_penetration = use_self_penetration
        self.use_joint_limit = use_joint_limit
        self.use_finger_finger = use_finger_finger
        self.use_finger_palm = use_finger_palm
    
    def forward(self, hand_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand_pose: (B, pose_dim) 手部位姿
        
        Returns:
            loss: 标量损失
        """
        # 设置手部模型状态
        self.hand_model.set_parameters(hand_pose)
        
        total_loss = 0.0
        count = 0
        
        # 自碰撞能量
        if self.use_self_penetration:
            total_loss = total_loss + self.hand_model.cal_self_penetration_energy().mean()
            count += 1
        
        # 关节限位能量
        if self.use_joint_limit:
            total_loss = total_loss + self.hand_model.cal_joint_limit_energy().mean()
            count += 1
        
        # 手指间距离能量
        if self.use_finger_finger:
            total_loss = total_loss + self.hand_model.cal_finger_finger_distance_energy().mean()
            count += 1
        
        # 手指-手掌距离能量
        if self.use_finger_palm:
            total_loss = total_loss + self.hand_model.cal_finger_palm_distance_energy().mean()
            count += 1
        
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=hand_pose.device)


class TotalLoss(nn.Module):
    """
    总损失函数
    整合所有损失组件
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
        """
        Args:
            hand_model: HandModel 实例
            w_trans: 平移损失权重
            w_rot: 旋转损失权重
            w_joint: 关节损失权重
            w_chamfer: Chamfer 损失权重
            w_physics: 物理先验损失权重（最终值）
            physics_ramp_epochs: 物理损失权重 ramp up 的 epoch 数
            joint_lower: 关节下限
            joint_upper: 关节上限
        """
        super().__init__()
        
        self.hand_model = hand_model
        self.w_trans = w_trans
        self.w_rot = w_rot
        self.w_joint = w_joint
        self.w_chamfer = w_chamfer
        self.w_physics = w_physics
        self.physics_ramp_epochs = physics_ramp_epochs
        
        # 损失组件
        self.translation_loss = TranslationLoss()
        self.rotation_loss = RotationGeodesicLoss()
        self.joint_loss = JointLoss(
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )
        self.chamfer_loss = ChamferLoss()
        self.physics_loss = PhysicsLoss(hand_model)
        
        # 当前 epoch（用于 ramp）
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch（用于物理损失 ramp up）"""
        self.current_epoch = epoch
    
    def get_physics_weight(self) -> float:
        """计算当前的物理损失权重（ramp up）"""
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
        """
        Args:
            pred_trans: (B, 3) 预测平移
            pred_rot: (B, 3, 3) 预测旋转矩阵
            pred_joints: (B, num_joints) 预测关节角
            gt_trans: (B, 3) GT 平移
            gt_rot: (B, 3, 3) GT 旋转矩阵
            gt_joints: (B, num_joints) GT 关节角
            pred_hand_pose: (B, pose_dim) 预测完整 hand_pose（用于物理损失和重建）
            gt_hand_pose: (B, pose_dim) GT 完整 hand_pose（用于重建）
        
        Returns:
            losses: 包含各项损失的字典
        """
        losses = {}
        
        # 参数损失（当对应权重为 0 时跳过计算以节省算力）
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
        
        # Chamfer 损失（当权重为 0 或缺少 hand_pose 时跳过计算）
        if self.w_chamfer > 0 and pred_hand_pose is not None and gt_hand_pose is not None:
            self.hand_model.set_parameters(pred_hand_pose)
            pred_keypoints = self.hand_model.get_penetration_keypoints()

            self.hand_model.set_parameters(gt_hand_pose)
            gt_keypoints = self.hand_model.get_penetration_keypoints()

            losses["chamfer"] = self.chamfer_loss(pred_keypoints, gt_keypoints)
        else:
            losses["chamfer"] = torch.tensor(0.0, device=pred_trans.device)
        
        # 物理先验损失
        if pred_hand_pose is not None:
            physics_weight = self.get_physics_weight()
            if physics_weight > 0:
                losses["physics"] = self.physics_loss(pred_hand_pose)
            else:
                losses["physics"] = torch.tensor(0.0, device=pred_trans.device)
        else:
            losses["physics"] = torch.tensor(0.0, device=pred_trans.device)
        
        # 总损失
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
    # 测试代码
    print("Testing loss functions...")
    
    # 测试平移损失
    trans_loss = TranslationLoss()
    pred_trans = torch.randn(4, 3)
    gt_trans = torch.randn(4, 3)
    loss = trans_loss(pred_trans, gt_trans)
    print(f"Translation loss: {loss.item():.6f}")
    
    # 测试旋转损失
    rot_loss = RotationGeodesicLoss()
    pred_rot = torch.randn(4, 3, 3)
    gt_rot = torch.randn(4, 3, 3)
    loss = rot_loss(pred_rot, gt_rot)
    print(f"Rotation loss: {loss.item():.6f} rad ({loss.item() * 180 / 3.14159:.2f} deg)")
    
    # 测试关节损失
    joint_loss = JointLoss()
    pred_joints = torch.randn(4, 16)
    gt_joints = torch.randn(4, 16)
    loss = joint_loss(pred_joints, gt_joints)
    print(f"Joint loss: {loss.item():.6f}")

