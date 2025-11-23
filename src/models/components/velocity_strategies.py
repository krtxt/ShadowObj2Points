import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any, Union, Callable

# Optional Import: PyTorch3D
# Handle missing dependency gracefully to ensure portability across environments.
try:
    from pytorch3d.transforms import rotation_6d_to_matrix
except ImportError:
    rotation_6d_to_matrix = None


# =============================================================================
# Section 1: Geometric & Graph Primitives
# Core mathematical transformations and topological utilities.
# =============================================================================

def r6d_to_rotation(r6d: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D continuous rotation representation to a 3x3 orthogonal rotation matrix.
    Based on 'On the Continuity of Rotation Representations in Neural Networks' (Zhou et al.).
    
    Args:
        r6d: (..., 6) Input tensor containing the first two columns of the rotation matrix.
    
    Returns:
        R: (..., 3, 3) Orthonormal rotation matrices.
    """
    if rotation_6d_to_matrix is not None:
        return rotation_6d_to_matrix(r6d)
    
    # Gram-Schmidt orthogonalization fallback
    x_raw = r6d[..., 0:3]
    y_raw = r6d[..., 3:6]
    
    x = F.normalize(x_raw, dim=-1, eps=1e-8)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps=1e-8)
    y = torch.cross(z, x, dim=-1)
    
    return torch.stack([x, y, z], dim=-1)


def build_rigid_groups(edge_index: torch.Tensor, num_points: int) -> List[torch.Tensor]:
    """
    Decomposes a graph into rigid sub-structures by identifying connected components.
    
    Args:
        edge_index: (2, E) tensor of source and target node indices.
        num_points: Total number of nodes in the graph.
        
    Returns:
        List[torch.Tensor]: A list where each tensor contains indices of a rigid cluster.
    """
    # Convert edge list to adjacency list for efficient traversal
    adj: List[List[int]] = [[] for _ in range(num_points)]
    src, dst = edge_index.cpu().tolist()
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)
        
    # Iterative DFS to find connected components
    visited = [False] * num_points
    groups = []
    
    for i in range(num_points):
        if visited[i]:
            continue
            
        stack = [i]
        component = []
        visited[i] = True
        
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    
        # Sort indices for deterministic behavior across runs
        groups.append(torch.tensor(sorted(component), dtype=torch.long))
        
    return groups


# =============================================================================
# Section 2: Differentiable Constraints & Physics
# Modules handling tangent space projection, rigid alignment, and PBD.
# =============================================================================

class TangentProjector(nn.Module):
    """
    Projects velocity fields onto the tangent space of distance manifolds.
    Mathematically solves J*v = 0 via the KKT system, minimizing kinetic energy change
    while satisfying holonomic distance constraints.
    
    Features:
        - Pre-computed topological sign matrices for O(1) Jacobian construction.
        - Robust linear solver with Cholesky -> Least Squares -> Zero fallback.
        - Mixed-precision safe (internal logic runs in FP32).
    """
    def __init__(self, edge_index: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        # Ensure topology is stored as long integers
        edge_index = edge_index.long()
        self.register_buffer("edge_index", edge_index.clone())
        
        # Pre-compute the topological structure of the Jacobian Gram matrix (J @ J.T)
        # This block constructs a sparse interaction pattern between edges.
        src, dst = edge_index
        E = edge_index.shape[1]
        
        # Broadcasting to compare all edges against all edges
        s1, s2 = src.unsqueeze(1), src.unsqueeze(0)
        d1, d2 = dst.unsqueeze(1), dst.unsqueeze(0)
        
        # sign[a, b] determines if edges 'a' and 'b' share a node and the directionality
        # +1 if flow aligns, -1 if opposing, 0 if disjoint.
        sign = (s1 == s2).float() - (s1 == d2).float() - (d1 == s2).float() + (d1 == d2).float()
        
        self.register_buffer("sign", sign)
        self.register_buffer("eye", torch.eye(E, dtype=torch.float32))
        
        # Pre-allocate index buffers for scatter operations to avoid runtime creation
        self.register_buffer("idx_src_base", src.view(1, -1, 1))
        self.register_buffer("idx_dst_base", dst.view(1, -1, 1))

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) Current positions.
            v: (B, N, 3) Proposed velocities.
        Returns:
            v_projected: (B, N, 3) Velocity satisfying distance constraints.
        """
        B, N, _ = x.shape
        src, dst = self.edge_index
        orig_dtype = v.dtype

        # Lift to FP32 for numerical stability in linear algebra
        x_f32 = x.float()
        v_f32 = v.float()
        
        # Edge vectors and velocity differences
        dx = x_f32[:, src] - x_f32[:, dst]  # (B, E, 3)
        dv = v_f32[:, src] - v_f32[:, dst]  # (B, E, 3)
        
        # 1. Construct RHS: -J * v (constraint violation magnitude)
        # Note: Factor 2.0 comes from derivative of x^2
        rhs = 2.0 * (dx * dv).sum(dim=-1, keepdim=True) # (B, E, 1)
        
        # 2. Construct System Matrix: A = J * M^{-1} * J^T
        # Dot product between all pairs of edge vectors weighted by topological sign
        # (B, E, 3) @ (B, 3, E) -> (B, E, E)
        edge_dot = torch.matmul(dx, dx.transpose(1, 2))
        A = 4.0 * edge_dot * self.sign + self.eps * self.eye
        
        # 3. Solve for Lagrange Multipliers: A * lambda = rhs
        # Strategy: Cholesky (Fast) -> LU/Lstsq (Robust) -> Zero (Safe Fallback)
        try:
            L, info = torch.linalg.cholesky_ex(A)
            lam = torch.zeros_like(rhs)
            
            # Mask for successful decompositions
            is_spd = (info == 0)
            if is_spd.any():
                lam[is_spd] = torch.cholesky_solve(rhs[is_spd], L[is_spd])
            
            # Fallback for ill-conditioned matrices (e.g., collinear points)
            if (~is_spd).any():
                # Use lstsq for non-SPD cases (more expensive but robust)
                mask_bad = ~is_spd
                sol = torch.linalg.lstsq(A[mask_bad], rhs[mask_bad]).solution
                lam[mask_bad] = sol
                
        except RuntimeError:
            # Catastrophic failure (e.g., NaN in input), return original v to prevent crash
            return v
            
        # 4. Apply Correction: v_new = v - J^T * lambda
        # Gradient of constraint w.r.t position is parallel to edge vector
        coeff = 2.0 * lam      # (B, E, 1)
        correction_force = coeff * dx  # (B, E, 3)
        
        v_correction = torch.zeros_like(v_f32)
        
        # Scatter add forces back to particles
        # Expand indices to (B, E, 3)
        idx_src = self.idx_src_base.expand(B, -1, 3)
        idx_dst = self.idx_dst_base.expand(B, -1, 3)
        
        v_correction.scatter_add_(1, idx_src,  correction_force)
        v_correction.scatter_add_(1, idx_dst, -correction_force)
        
        return (v_f32 - v_correction).to(orig_dtype)


class RigidGroupProjector(nn.Module):
    """
    Enforces rigid body constraints by optimally aligning input points to a template
    using the Kabsch algorithm (SVD). Handles multiple disjoint rigid groups.
    """
    def __init__(self, template_xyz: torch.Tensor, groups: List[torch.Tensor]):
        super().__init__()
        self.register_buffer("template_xyz", template_xyz.clone())
        # Cloning groups ensures we own the tensor data
        self.groups = [g.clone().detach() for g in groups]

    @staticmethod
    def _kabsch_align(source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finds (R, t) minimizing || (source @ R^T + t) - target ||^2.
        Args:
            source: (B, N, 3) Template points.
            target: (B, N, 3) Current deformed points.
        Returns:
            R: (B, 3, 3) Rotation matrix.
            t: (B, 3) Translation vector.
        """
        mu_s = source.mean(dim=1, keepdim=True)
        mu_t = target.mean(dim=1, keepdim=True)
        
        # Center the point clouds
        src_c = source - mu_s
        tgt_c = target - mu_t
        
        # Covariance matrix H = src^T * tgt
        H = src_c.transpose(1, 2) @ tgt_c
        
        # SVD: H = U S V^T
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-2, -1)
        
        # Compute Rotation R = V U^T
        R = V @ U.transpose(-2, -1)
        
        # Correction for reflection (ensure determinant is +1)
        det = torch.det(R)
        mask_reflection = (det < 0).view(-1, 1, 1)
        
        if mask_reflection.any():
            V_fixed = V.clone()
            # Negate the last column of V where reflection occurred
            V_fixed[mask_reflection.squeeze(), :, -1] *= -1
            R = torch.where(mask_reflection, V_fixed @ U.transpose(-2, -1), R)
            
        # Compute Translation
        # t = centroid_target - (R @ centroid_source.T).T
        t = mu_t.squeeze(1) - (R @ mu_s.transpose(1, 2)).squeeze(2)
        
        return R, t

    def forward(self, x_proto: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_proto: (B, N, 3) Input configuration (potentially distorted).
        Returns:
            out: (B, N, 3) Configuration projected onto the rigid manifold.
        """
        B, N, _ = x_proto.shape
        device = x_proto.device
        template = self.template_xyz.to(device)
        
        out = torch.zeros_like(x_proto)
        weights = torch.zeros(B, N, 1, device=device, dtype=x_proto.dtype)
        
        for group_idx in self.groups:
            group_idx = group_idx.to(device)
            
            # Template (Source) -> Current Prediction (Target)
            src = template[group_idx].unsqueeze(0).expand(B, -1, -1)
            tgt = x_proto[:, group_idx, :]
            
            if src.shape[1] < 3:
                # Degenerate case: Translation only
                mu_s = src.mean(dim=1, keepdim=True)
                mu_t = tgt.mean(dim=1, keepdim=True)
                projected = src - mu_s + mu_t
            else:
                # Full Rigid Alignment
                R, t = self._kabsch_align(src, tgt)
                # Apply transform: (R @ src.T).T + t
                projected = (R @ src.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
            
            # Accumulate results (handles overlapping groups if any)
            out[:, group_idx, :] += projected
            weights[:, group_idx, :] += 1.0
            
        # Normalize by overlap count
        return out / weights.clamp(min=1.0)


class XPBDProjector(nn.Module):
    """
    Extended Position Based Dynamics (XPBD) constraint solver.
    Iteratively satisfies edge length constraints.
    """
    def __init__(
        self,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        iters: int = 6,
        inv_mass: Optional[torch.Tensor] = None,
        compliance: float = 0.0,
        max_corr: float = 0.2,
    ):
        super().__init__()
        self.register_buffer("edge_index", edge_index.clone())
        self.register_buffer("rest_lengths", rest_lengths.clone())
        
        if inv_mass is not None:
            self.register_buffer("inv_mass", inv_mass.clone())
        else:
            self.inv_mass = None
            
        self.iters = iters
        self.compliance = compliance
        self.max_corr = max_corr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        src, dst = self.edge_index
        L0 = self.rest_lengths.view(1, -1, 1)
        
        # Initialize inverse mass
        if self.inv_mass is not None:
            w = self.inv_mass.view(-1, 1, 1)
        else:
            w = torch.ones(N, 1, 1, device=x.device, dtype=x.dtype)
            
        w_i = w[src].transpose(1, 2) # (1, 1, E) -> View adjustment needed based on dim
        w_i = w[src].view(1, -1, 1)
        w_j = w[dst].view(1, -1, 1)
        
        x_curr = x.clone()
        alpha = self.compliance / (float(self.iters) + 1e-8)
        
        # Pre-expand indices for scatter operations
        idx_src = src.view(1, -1, 1).expand(B, -1, 3)
        idx_dst = dst.view(1, -1, 1).expand(B, -1, 3)
        
        for _ in range(self.iters):
            # 1. Calculate Constraint Violation
            # x_i - x_j
            diff = x_curr[:, src] - x_curr[:, dst]
            dist = torch.norm(diff, dim=-1, keepdim=True)
            
            # C(x) = |x_ij| - L0
            C = dist - L0
            # Gradient direction n = (x_i - x_j) / |x_ij|
            n = diff / (dist + 1e-9)
            
            # 2. Calculate Lagrange Multiplier (Lambda)
            # lambda = -C / (w_i + w_j + alpha)
            denom = w_i + w_j + alpha
            d_lambda = -C / (denom + 1e-9)
            
            # 3. Calculate Correction Vectors
            correction = d_lambda * n
            
            # Stability: Limit maximum correction per step
            correction = torch.clamp(correction, -self.max_corr, self.max_corr)
            
            dx_i = w_i * correction
            dx_j = -w_j * correction # Newton's 3rd law
            
            # 4. Apply Deltas
            # Use scatter_add to handle particles connected to multiple edges simultaneously
            delta_accum = torch.zeros_like(x_curr)
            delta_accum.scatter_add_(1, idx_src, dx_i)
            delta_accum.scatter_add_(1, idx_dst, dx_j)
            
            # Note: In strict PBD, we update immediately. 
            # With scatter_add (Jacobi-style update), we average corrections or just sum.
            # Summing is standard for parallel GPU implementation.
            x_curr = x_curr + delta_accum
            
        return x_curr


# =============================================================================
# Section 3: Velocity Strategies
# Logic to determine particle velocities from latent embeddings.
# =============================================================================

class VelocityStrategyBase(nn.Module):
    """Abstract base class for all velocity prediction strategies."""
    
    def __init__(self, tau_min: float = 1e-3):
        super().__init__()
        self.tau_min = tau_min
        self.tangent_projector: Optional[TangentProjector] = None

    def predict(self, model, keypoints: torch.Tensor, timesteps: torch.Tensor, hand_tokens: torch.Tensor) -> torch.Tensor:
        """
        Main interface for velocity prediction.
        Args:
            model: The calling parent model (context).
            keypoints: (B, N, 3) Current positions.
            timesteps: (B,) or (B, 1) Time embedding/value.
            hand_tokens: (B, N, D) Latent features from the transformer.
        """
        raise NotImplementedError

    def _get_tau(self, t: torch.Tensor) -> torch.Tensor:
        """Computes time scaling factor. t goes from 0 -> 1."""
        if t.dim() == 1:
            t = t.view(-1, 1, 1)
        # Prevent division by zero near t=1
        return torch.clamp(1.0 - t, min=self.tau_min)

    def _apply_constraints(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.tangent_projector is not None:
            return self.tangent_projector(x, v)
        return v


class DirectVelocityStrategy(VelocityStrategyBase):
    """Directly projects latent features to 3D velocity vectors."""
    
    def __init__(self, d_model: int, edge_index: torch.Tensor, use_tangent: bool):
        super().__init__()
        self.head = nn.Linear(d_model, 3)
        if use_tangent:
            self.tangent_projector = TangentProjector(edge_index)

    def predict(self, model, keypoints, timesteps, hand_tokens) -> torch.Tensor:
        v = self.head(hand_tokens)
        return self._apply_constraints(v, keypoints)


class PhysGuidedVelocityStrategy(VelocityStrategyBase):
    """Adds a gradient descent term on edge-length energy to the predicted velocity."""
    
    def __init__(self, d_model: int, edge_index: torch.Tensor, rest_lengths: torch.Tensor, 
                 eta: float = 0.5, use_tangent: bool = True):
        super().__init__()
        self.head = nn.Linear(d_model, 3)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("rest_lengths", rest_lengths)
        self.eta = eta
        if use_tangent:
            self.tangent_projector = TangentProjector(edge_index)

    def _compute_energy_grad(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient of E = 0.5 * sum( (|x_i - x_j| - L0)^2 )
        src, dst = self.edge_index
        diff = x[:, src] - x[:, dst]
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-9
        L0 = self.rest_lengths.view(1, -1, 1)
        
        # grad_E w.r.t x_i = (|x_ij| - L0) * (x_ij / |x_ij|)
        grad_mag = (dist - L0) / dist
        grad_vec = grad_mag * diff
        
        grad = torch.zeros_like(x)
        idx_src = src.view(1, -1, 1).expand(x.shape[0], -1, 3)
        idx_dst = dst.view(1, -1, 1).expand(x.shape[0], -1, 3)
        
        grad.scatter_add_(1, idx_src, grad_vec)
        grad.scatter_add_(1, idx_dst, -grad_vec)
        return grad

    def predict(self, model, keypoints, timesteps, hand_tokens) -> torch.Tensor:
        v_pred = self.head(hand_tokens)
        v_phys = -self.eta * self._compute_energy_grad(keypoints)
        return self._apply_constraints(v_pred + v_phys, keypoints)


class GoalKabschVelocityStrategy(VelocityStrategyBase):
    """
    Predicts a deformation, projects it to a valid rigid goal, and steers towards it.
    v = (Rigid(x + delta) - x) / (1 - t)
    """
    def __init__(self, d_model: int, template_xyz: torch.Tensor, groups: List[torch.Tensor],
                 edge_index: torch.Tensor, tau_min: float = 1e-3, use_tangent: bool = True):
        super().__init__(tau_min)
        self.decode = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(), nn.Linear(256, 3)
        )
        self.projector = RigidGroupProjector(template_xyz, groups)
        if use_tangent:
            self.tangent_projector = TangentProjector(edge_index)

    def predict(self, model, keypoints, timesteps, hand_tokens) -> torch.Tensor:
        delta = self.decode(hand_tokens)
        
        # Access template from model if available (handles device/type sync), else fallback
        ref_template = getattr(model, 'template_xyz', self.projector.template_xyz)
        
        x_proto = ref_template.unsqueeze(0) + delta
        x_goal = self.projector(x_proto)
        
        tau = self._get_tau(timesteps)
        v = (x_goal - keypoints) / tau
        return self._apply_constraints(v, keypoints)


class GroupRigidParamVelocityStrategy(VelocityStrategyBase):
    """
    Predicts explicit 6D Rotation and Translation per rigid group.
    """
    def __init__(self, d_model: int, template_xyz: torch.Tensor, groups: List[torch.Tensor],
                 edge_index: torch.Tensor, use_tangent: bool = True, tau_min: float = 1e-3):
        super().__init__(tau_min)
        self.groups = [g.clone() for g in groups]
        self.register_buffer("template_xyz", template_xyz.clone())
        
        # Head outputs 9 params per group: 6 (Rot6D) + 3 (Trans)
        self.group_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(), nn.Linear(256, 9)
        )
        if use_tangent:
            self.tangent_projector = TangentProjector(edge_index)

    def predict(self, model, keypoints, timesteps, hand_tokens) -> torch.Tensor:
        B, N, D = hand_tokens.shape
        device = hand_tokens.device
        
        # 1. Pooling: Aggregate tokens per group
        group_embeddings = []
        for g in self.groups:
            # Mean pool tokens belonging to group g
            g_tokens = hand_tokens[:, g.to(device), :].mean(dim=1)
            group_embeddings.append(g_tokens)
            
        # (B, NumGroups, D)
        group_in = torch.stack(group_embeddings, dim=1)
        
        # 2. Predict SE(3) params
        params = self.group_head(group_in) # (B, G, 9)
        r6d = params[..., :6]
        t = params[..., 6:]
        
        # (B, G, 3, 3)
        R = r6d_to_rotation(r6d)
        
        # 3. Apply Transforms to Template
        template = self.template_xyz.to(device)
        x_goal_accum = torch.zeros(B, N, 3, device=device, dtype=hand_tokens.dtype)
        weights = torch.zeros(B, N, 1, device=device, dtype=hand_tokens.dtype)
        
        for i, g in enumerate(self.groups):
            idx = g.to(device)
            src = template[idx].unsqueeze(0) # (1, M, 3)
            
            # X_new = (R @ X_old.T).T + t
            # Using matmul with broadcasting: R is (B, 3, 3), src is (1, M, 3)
            # We need (B, M, 3)
            src_centered = src.expand(B, -1, -1)
            
            # Rotation: (B, 3, 3) x (B, 3, M) -> (B, 3, M) -> (B, M, 3)
            rotated = torch.matmul(R[:, i], src_centered.transpose(1, 2)).transpose(1, 2)
            transformed = rotated + t[:, i].unsqueeze(1)
            
            x_goal_accum[:, idx, :] += transformed
            weights[:, idx, :] += 1.0
            
        x_goal = x_goal_accum / weights.clamp(min=1.0)
        
        # 4. Compute Velocity
        tau = self._get_tau(timesteps)
        v = (x_goal - keypoints) / tau
        return self._apply_constraints(v, keypoints)


class PBDCorrectedVelocityStrategy(VelocityStrategyBase):
    """
    Predicts velocity, simulates a step, corrects with XPBD, and computes effective velocity.
    This ensures the velocity field inherently respects the physics manifold.
    """
    def __init__(self, d_model: int, edge_index: torch.Tensor, rest_lengths: torch.Tensor,
                 xpbd_iters: int = 4, xpbd_compliance: float = 0.0, xpbd_max_corr: float = 0.15,
                 use_tangent: bool = False, tau_min: float = 1e-3):
        super().__init__(tau_min)
        self.head = nn.Linear(d_model, 3)
        self.projector = XPBDProjector(
            edge_index, rest_lengths, iters=xpbd_iters, 
            compliance=xpbd_compliance, max_corr=xpbd_max_corr
        )
        if use_tangent:
            self.tangent_projector = TangentProjector(edge_index)

    def predict(self, model, keypoints, timesteps, hand_tokens) -> torch.Tensor:
        tau = self._get_tau(timesteps)
        v_raw = self.head(hand_tokens)
        
        # Tentative step
        x_pred = keypoints + tau * v_raw
        # Manifold projection
        x_corrected = self.projector(x_pred)
        
        # Effective velocity
        v_eff = (x_corrected - keypoints) / tau
        return self._apply_constraints(v_eff, keypoints)


# =============================================================================
# Section 4: Factory & Registry
# Centralized instantiation logic.
# =============================================================================

class ComponentFactory:
    """Helper class to manage component creation and parameter validation."""
    
    @staticmethod
    def create_velocity_strategy(
        mode: str,
        d_model: int,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        template_xyz: Optional[torch.Tensor],
        groups: List[torch.Tensor],
        kwargs: Dict[str, Any]
    ) -> VelocityStrategyBase:
        
        mode = mode.lower()
        kwargs = kwargs or {}
        
        # Registry mapping: name -> (class, list of allowed kwarg keys)
        registry = {
            "direct": (DirectVelocityStrategy, ["use_tangent"]),
            "direct_tangent": (
                lambda **k: DirectVelocityStrategy(**k, use_tangent=True), ["use_tangent"]
            ),
            "phys_guided": (PhysGuidedVelocityStrategy, ["eta", "use_tangent"]),
            "goal_kabsch": (GoalKabschVelocityStrategy, ["tau_min", "use_tangent"]),
            "group_rigid": (GroupRigidParamVelocityStrategy, ["tau_min", "use_tangent"]),
            "pbd_corrected": (PBDCorrectedVelocityStrategy, ["iters", "compliance", "max_corr", "use_tangent", "tau_min"]),
        }

        if mode not in registry:
            raise ValueError(f"Unknown velocity mode: {mode}. Options: {list(registry.keys())}")

        cls_or_func, allowed_keys = registry[mode]
        
        # Filter kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        if len(filtered_kwargs) != len(kwargs):
            unknown = set(kwargs.keys()) - set(allowed_keys)
            raise ValueError(f"Unknown kwargs for mode {mode}: {unknown}")

        # Common arguments injection based on Strategy signature
        # We construct a dynamic arg dictionary to satisfy different constructors
        build_args = {"d_model": d_model, "edge_index": edge_index}
        
        if mode in ["phys_guided", "pbd_corrected"]:
            build_args["rest_lengths"] = rest_lengths
        
        if mode in ["goal_kabsch", "group_rigid"]:
            if template_xyz is None:
                raise ValueError(f"Mode {mode} requires template_xyz.")
            build_args["template_xyz"] = template_xyz
            build_args["groups"] = groups

        # Merge specific kwargs
        build_args.update(filtered_kwargs)
        
        return cls_or_func(**build_args)

    @staticmethod
    def create_state_projector(
        mode: str,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        template_xyz: Optional[torch.Tensor],
        groups: List[torch.Tensor],
        kwargs: Dict[str, Any]
    ):
        mode = mode.lower()
        kwargs = kwargs or {}
        
        if mode == "none":
            return None
        
        elif mode == "pbd":
            return XPBDProjector(
                edge_index, rest_lengths,
                iters=int(kwargs.get("iters", 2)),
                compliance=float(kwargs.get("compliance", 0.0)),
                max_corr=float(kwargs.get("max_corr", 0.2))
            )
            
        elif mode == "rigid":
            if template_xyz is None: raise ValueError("Rigid projection requires template_xyz")
            return RigidGroupProjector(template_xyz, groups)
            
        elif mode == "hybrid":
            if template_xyz is None: raise ValueError("Hybrid projection requires template_xyz")
            rigid = RigidGroupProjector(template_xyz, groups)
            pbd = XPBDProjector(
                edge_index, rest_lengths,
                iters=int(kwargs.get("hybrid_pbd_iters", 2)),
                compliance=float(kwargs.get("hybrid_compliance", 0.0)),
                max_corr=float(kwargs.get("hybrid_max_corr", 0.2))
            )
            return ("hybrid", rigid, pbd)
            
        else:
            raise ValueError(f"Unknown state projector: {mode}")


# Public API Wrappers
def build_velocity_strategy(mode, d_model, edge_index, rest_lengths, template_xyz, groups, kwargs=None):
    return ComponentFactory.create_velocity_strategy(
        mode, d_model, edge_index, rest_lengths, template_xyz, groups, kwargs or {}
    )

def build_state_projector(mode, edge_index, rest_lengths, template_xyz, groups, kwargs=None):
    return ComponentFactory.create_state_projector(
        mode, edge_index, rest_lengths, template_xyz, groups, kwargs or {}
    )