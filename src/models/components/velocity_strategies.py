import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Try importing pytorch3d
try:
    from pytorch3d.transforms import rotation_6d_to_matrix
except ImportError:
    rotation_6d_to_matrix = None


# Geometric & Graph Utilities

def r6d_to_rotation(r6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D continuous rotation representation to 3x3 rotation matrix.
    Uses pytorch3d if available, otherwise falls back to local implementation of Zhou et al.
    """
    if rotation_6d_to_matrix is not None:
        return rotation_6d_to_matrix(r6d)
    
    # Local implementation fallback
    a1 = r6d[..., 0:3]
    a2 = r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-9)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-9)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)
    return R

def build_rigid_groups(edge_index: torch.Tensor, num_points: int) -> List[torch.Tensor]:
    """
    Identify undirected connected components as rigid groups.
    
    Args:
        edge_index: (2, E) edge list
        num_points: Total number of points
        
    Returns:
        List of tensors, each containing indices of a connected component.
    """
    i, j = edge_index
    adj = [[] for _ in range(num_points)]
    for u, v in zip(i.tolist(), j.tolist()):
        adj[u].append(v)
        adj[v].append(u)
        
    seen = [False] * num_points
    groups = []
    for s in range(num_points):
        if seen[s]:
            continue
        stack = [s]
        comp = []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        groups.append(torch.tensor(sorted(set(comp)), dtype=torch.long))
    return groups

class TangentProjector(nn.Module):
    """
    Orthogonally project velocity to the tangent space of distance constraints (rigid edges).
    Solves for minimal correction to satisfy J*v = 0 where J is constraint Jacobian.
    
    Optimized with pre-computed topological structure and Cholesky decomposition.
    """
    def __init__(self, edge_index: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        # Ensure indices are in long dtype for safe tensor indexing
        edge_index = edge_index.long()
        self.register_buffer("edge_index", edge_index.clone().detach())
        
        # Pre-compute sign matrix for A = J J^T construction
        # This depends only on graph topology
        i, j = self.edge_index
        E = edge_index.shape[1]
        
        i1 = i.view(-1, 1)
        i2 = i.view(1, -1)
        j1 = j.view(-1, 1)
        j2 = j.view(1, -1)

        # 1.0 if indices match
        eq_i_i = (i1 == i2).float()
        eq_i_j = (i1 == j2).float()
        eq_j_i = (j1 == i2).float()
        eq_j_j = (j1 == j2).float()

        # sign[e1, e2] selects the correct sign for the dot product
        sign = eq_i_i - eq_i_j - eq_j_i + eq_j_j
        self.register_buffer("sign", sign)

        # We always build A in float32 for numerical stability, even if x is float16/64.
        # The result is cast back to the original v.dtype in forward().
        self.register_buffer("eye", torch.eye(E, dtype=torch.float32))

        # Base edge indices for scatter_add in forward; expanded per batch on the fly
        self.register_buffer("edge_index_i_base", i.view(1, -1, 1))
        self.register_buffer("edge_index_j_base", j.view(1, -1, 1))

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        i, j = self.edge_index
        orig_dtype = v.dtype

        # Work in float32 for stable linear algebra; results are cast back to orig_dtype.
        diff_x = (x[:, i, :] - x[:, j, :]).to(torch.float32)  # (B, E, 3)
        v32 = v.to(torch.float32)

        # 1. RHS = J * v
        # (J v)_e = 2 * (x_i - x_j) * (v_i - v_j)
        v_diff = v32[:, i, :] - v32[:, j, :]
        rhs = 2.0 * (diff_x * v_diff).sum(dim=-1, keepdim=True)  # (B, E, 1)

        # 2. A = J J^T
        # dot[b, e1, e2] = diff_x[b, e1] . diff_x[b, e2]
        # Using matmul: (B, E, 3) @ (B, 3, E) -> (B, E, E)
        dot = torch.matmul(diff_x, diff_x.transpose(1, 2))
        A = 4.0 * dot * self.sign + self.eps * self.eye

        # 3. Solve A * lambda = rhs using Cholesky (A is SPD) with robust fallback.
        #    If all solvers fail for some batch elements, we fall back to zero correction for them.
        try:
            # cholesky_ex gives per-batch status; we fall back to lstsq only for failing batches
            L, info = torch.linalg.cholesky_ex(A)
            lam = torch.zeros_like(rhs)
            good = info == 0
            if good.any():
                lam[good] = torch.cholesky_solve(rhs[good], L[good])
            bad = ~good
            if bad.any():
                # For ill-conditioned / non-SPD batches, fall back to least-squares per bad batch
                try:
                    lam_bad = torch.linalg.lstsq(A[bad], rhs[bad]).solution
                    lam[bad] = lam_bad
                except Exception:
                    # If least-squares still fails, keep zero correction for these batches
                    pass
        except Exception:
            # Global fallback: if Cholesky or lstsq fail on the whole batch, skip projection
            lam = torch.zeros_like(rhs)

        # 4. Correction = J^T * lambda
        # For each edge e, coeff is 2 * lambda_e
        coeff = 2.0 * lam.squeeze(-1).unsqueeze(-1)  # (B, E, 1)
        corr_edge = coeff * diff_x  # (B, E, 3)

        corr = torch.zeros(B, N, 3, device=x.device, dtype=torch.float32)
        idx_i = self.edge_index_i_base.expand(B, -1, 3)
        idx_j = self.edge_index_j_base.expand(B, -1, 3)
        
        corr.scatter_add_(1, idx_i, corr_edge)
        corr.scatter_add_(1, idx_j, -corr_edge)

        return (v32 - corr).to(orig_dtype)


# Projectors

class RigidGroupProjector(nn.Module):
    """
    Per-group Kabsch/Umeyama rigid projection with overlap-friendly averaging.
    Projects points to the nearest configuration consistent with rigid groups derived from template.
    """
    def __init__(self, template_xyz: torch.Tensor, groups: List[torch.Tensor]):
        super().__init__()
        self.register_buffer("template_xyz", template_xyz.clone().detach())  # (N,3)
        self.groups = [g.clone().detach() for g in groups]

    @staticmethod
    def _kabsch_batch(Y: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find rigid transform (R, t) that aligns X to Y (X -> Y).
        Y, X: (B, N, 3)
        Returns: R (B,3,3), t (B,3) such that X @ R.T + t ~ Y
        Note: The implementation below returns R, t such that (R @ Y.T).T + t matches X? 
        Wait, let's check the math in the original code.
        Original: R, t = _kabsch_batch(Yg, Xg)
                  Xg_proj = (R @ Yg.transpose(1,2)).transpose(1,2) + t
        This implies we are aligning Y (template) TO X (current state).
        So we find R, t s.t. R*Y + t ~ X.
        """
        muY = Y.mean(dim=1, keepdim=True)
        muX = X.mean(dim=1, keepdim=True)
        Yc = Y - muY
        Xc = X - muX
        
        # Covariance matrix H
        H = Yc.transpose(1,2) @ Xc  # (B, 3, 3)
        U, S, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-2, -1)
        
        # R = V @ U.T
        R = V @ U.transpose(-2, -1)
        
        # Reflection correction
        det = torch.det(R)
        neg = (det < 0).view(-1, 1, 1)
        if neg.any():
            V_adj = V.clone()
            V_adj[neg.squeeze(-1).squeeze(-1), :, -1] *= -1
            R = V_adj @ U.transpose(-2, -1)
            
        # t = muX - R @ muY
        t = (muX.squeeze(1) - (R @ muY.transpose(1,2)).squeeze(2))
        return R, t

    def forward(self, x_proto: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_proto.shape
        device = x_proto.device
        template = self.template_xyz.to(device)
        out = torch.zeros_like(x_proto)
        counts = torch.zeros(B, N, 1, device=device, dtype=x_proto.dtype)
        
        for g in self.groups:
            idx = g.to(device)
            # Yg: template points (target shape)
            Yg = template[idx].unsqueeze(0).expand(B, -1, -1)
            # Xg: input points (source shape)
            Xg = x_proto[:, idx, :]
            
            if Yg.shape[1] < 3:
                # Not enough points for rotation, just translation
                muY = Yg.mean(dim=1, keepdim=True)
                muX = Xg.mean(dim=1, keepdim=True)
                Xg_proj = Yg - muY + muX
            else:
                # Rigid align template (Y) to input (X)
                R, t = self._kabsch_batch(Yg, Xg)
                Xg_proj = (R @ Yg.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)
                
            out[:, idx, :] += Xg_proj
            counts[:, idx, :] += 1.0
            
        counts = counts.clamp_min(1.0)
        return out / counts


class XPBDProjector(nn.Module):
    """
    XPBD/Sequential Gauss–Seidel projection on edge-length constraints.
    Enforces rest lengths between connected particles.
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
        self.register_buffer("edge_index", edge_index.clone().detach())
        self.register_buffer("rest_lengths", rest_lengths.clone().detach())
        self.iters = int(iters)
        self.compliance = float(compliance)
        self.max_corr = float(max_corr)
        if inv_mass is not None:
            self.register_buffer("inv_mass", inv_mass.clone().detach())
        else:
            self.inv_mass = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        # Ensure edge indices live on the same device as the input
        edge_index = self.edge_index.to(x.device)
        i, j = edge_index
        rl = self.rest_lengths.view(1, -1, 1).to(x)
        
        if self.inv_mass is not None:
            im = self.inv_mass.to(x)
        else:
            im = torch.ones(N, device=x.device, dtype=x.dtype)
            
        im_i = im[i].view(1, -1, 1)
        im_j = im[j].view(1, -1, 1)
        
        x_proj = x.clone()
        alpha = self.compliance
        
        for _ in range(self.iters):
            diff = x_proj[:, i, :] - x_proj[:, j, :]
            dist = (diff.pow(2).sum(-1, keepdim=True) + 1e-9).sqrt()
            
            C = dist - rl
            n = diff / (dist + 1e-9)
            
            w = im_i + im_j + alpha
            lam = -C / (w + 1e-9)
            corr = lam * n
            
            # Clamp per-edge correction magnitude
            corr = corr.clamp(min=-self.max_corr, max=self.max_corr)
            
            delta_i = -im_i * corr
            delta_j =  im_j * corr
            
            zeros = torch.zeros_like(x_proj)
            idx_i = i.view(1, -1, 1).expand(B, -1, 3)
            idx_j = j.view(1, -1, 1).expand(B, -1, 3)
            
            zeros.scatter_add_(1, idx_i, delta_i)
            zeros.scatter_add_(1, idx_j, delta_j)
            
            x_proj = x_proj + zeros
            
        return x_proj


# Velocity Strategies

class VelocityStrategyBase(nn.Module):
    """Abstract base class for velocity strategies."""
    
    def __init__(self, tau_min: float = 1e-3):
        super().__init__()
        self.tau_min = float(tau_min)

    def predict(self, model, keypoints, timesteps, hand_tokens_out) -> torch.Tensor:
        raise NotImplementedError

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time scaling factor tau from timestep t (0 to 1)."""
        if t.dim() == 1:
            t = t.view(-1, 1, 1)
        return torch.clamp(1.0 - t, min=self.tau_min)

    def apply_tangent_projection(self, v: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Apply tangent projection if enabled."""
        if hasattr(self, 'tangent_projector') and self.tangent_projector is not None:
            return self.tangent_projector(keypoints, v)
        return v


class DirectVelocityStrategy(VelocityStrategyBase):
    """
    Predicts velocity directly from tokens.
    v = head(tokens)
    """
    def __init__(self, d_model: int, use_tangent: bool, edge_index: torch.Tensor):
        super().__init__()
        self.head = nn.Linear(d_model, 3)
        self.register_buffer("edge_index", edge_index.clone().detach())
        self.tangent_projector = TangentProjector(edge_index) if use_tangent else None

    def predict(self, model, keypoints, timesteps, hand_tokens_out) -> torch.Tensor:
        v = self.head(hand_tokens_out)
        v = self.apply_tangent_projection(v, keypoints)
        return v


class PhysGuidedVelocityStrategy(VelocityStrategyBase):
    """
    Direct velocity + gradient correction towards satisfying lengths.
    v = v_pred - eta * grad(ConstraintEnergy)
    """
    def __init__(
        self,
        d_model: int,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        eta: float = 0.5,
        use_tangent: bool = True,
    ):
        super().__init__()
        self.head = nn.Linear(d_model, 3)
        self.register_buffer("edge_index", edge_index.clone().detach())
        self.register_buffer("rest_lengths", rest_lengths.clone().detach())
        self.eta = float(eta)
        self.tangent_projector = TangentProjector(edge_index) if use_tangent else None

    def _length_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate gradient of edge length energy potential."""
        B, N, _ = x.shape
        i, j = self.edge_index
        rl = self.rest_lengths.view(1, -1, 1).to(x)
        
        diff = x[:, i, :] - x[:, j, :]
        dist = (diff.pow(2).sum(-1, keepdim=True) + 1e-9).sqrt()
        
        # Energy = 0.5 * (dist - rl)^2
        # Grad = (dist - rl) * grad(dist)
        # grad(dist) = diff / dist
        coeff = (1.0 - rl / dist)  # (B, E, 1)
        
        grad_i =  coeff * diff
        grad_j = -coeff * diff
        
        out = torch.zeros_like(x)
        idx_i = i.view(1, -1, 1).expand(B, -1, 3)
        idx_j = j.view(1, -1, 1).expand(B, -1, 3)
        
        out.scatter_add_(1, idx_i, grad_i)
        out.scatter_add_(1, idx_j, grad_j)
        return out

    def predict(self, model, keypoints, timesteps, hand_tokens_out) -> torch.Tensor:
        v_raw = self.head(hand_tokens_out)
        grad = self._length_grad(keypoints)
        v = v_raw - self.eta * grad
        v = self.apply_tangent_projection(v, keypoints)
        return v


class GoalKabschVelocityStrategy(VelocityStrategyBase):
    """
    Predicts a deformation delta, projects to a valid rigid assembly, then computes velocity to reach that goal.
    x_proto = template + delta(tokens)
    x_goal = RigidProject(x_proto)
    v = (x_goal - x_curr) / tau
    """
    def __init__(
        self,
        d_model: int,
        template_xyz: torch.Tensor,
        groups: List[torch.Tensor],
        edge_index: torch.Tensor,
        tau_min: float = 1e-3,
        use_tangent: bool = True,
    ):
        super().__init__(tau_min=tau_min)
        self.decode = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(),
            nn.Linear(256, 3)
        )
        self.projector = RigidGroupProjector(template_xyz, groups)
        self.register_buffer("edge_index", edge_index.clone().detach())
        self.tangent_projector = TangentProjector(edge_index) if use_tangent else None

    def predict(self, model, keypoints, timesteps, hand_tokens_out) -> torch.Tensor:
        delta = self.decode(hand_tokens_out)  # (B, N, 3)
        # Note: We use model.template_xyz to ensure we use the same device/dtype as the main model
        # if self.projector.template_xyz is not enough. 
        # Assuming model has template_xyz.
        ref_template = getattr(model, 'template_xyz', None)
        if ref_template is None:
            ref_template = self.projector.template_xyz.to(keypoints.device)
            
        x_proto = ref_template.unsqueeze(0) + delta
        x_goal = self.projector(x_proto)
        
        tau = self._tau(timesteps)
        v = (x_goal - keypoints) / tau
        v = self.apply_tangent_projection(v, keypoints)
        return v


class GroupRigidParamVelocityStrategy(VelocityStrategyBase):
    """
    Predicts Rigid Transforms (R, t) per group from pooled tokens, applies to template to get goal.
    v = (x_goal - x_curr) / tau
    """
    def __init__(
        self,
        d_model: int,
        template_xyz: torch.Tensor,
        groups: List[torch.Tensor],
        edge_index: torch.Tensor,
        use_tangent: bool = True,
        tau_min: float = 1e-3,
    ):
        super().__init__(tau_min=tau_min)
        self.groups = [g.clone().detach() for g in groups]
        self.template = template_xyz.clone().detach()
        self.register_buffer("edge_index", edge_index.clone().detach())
        self.tangent_projector = TangentProjector(edge_index) if use_tangent else None
        
        # group head: 6D rot + 3 t per group
        self.group_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(),
            nn.Linear(256, 9)
        )

    def predict(self, model, keypoints, timesteps, hand_tokens_out) -> torch.Tensor:
        B, N, D = hand_tokens_out.shape
        device = hand_tokens_out.device
        
        # per-group pooling -> (B, G, d_model)
        g_tokens_list = []
        for g in self.groups:
            # Gather tokens for this group and mean pool
            g_tokens_list.append(hand_tokens_out[:, g.to(device), :].mean(dim=1))
            
        g_tokens = torch.stack(g_tokens_list, dim=1)  # (B, G, D)
        params = self.group_head(g_tokens)            # (B, G, 9)
        
        r6d = params[..., :6]
        t = params[..., 6:]      # (B, G, 3)
        R = r6d_to_rotation(r6d) # (B, G, 3, 3)

        template = self.template.to(device)
        out = torch.zeros(B, N, 3, device=device, dtype=hand_tokens_out.dtype)
        counts = torch.zeros(B, N, 1, device=device, dtype=hand_tokens_out.dtype)
        
        for gi, g in enumerate(self.groups):
            idx = g.to(device)
            Yg = template[idx].unsqueeze(0).expand(B, -1, -1)
            
            # Apply predicted transform: R * Y + t
            # Yg is (B, M, 3), R is (B, 3, 3) (slice for this group)
            # (R @ Yg.T).T + t
            Xg = (R[:, gi] @ Yg.transpose(1,2)).transpose(1,2) + t[:, gi].unsqueeze(1)
            
            out[:, idx, :] += Xg
            counts[:, idx, :] += 1.0
            
        x_goal = out / counts.clamp_min(1.0)
        
        tau = self._tau(timesteps)
        v = (x_goal - keypoints) / tau
        v = self.apply_tangent_projection(v, keypoints)
        return v


class PBDCorrectedVelocityStrategy(VelocityStrategyBase):
    """
    Predicts velocity, takes a step, then corrects position with XPBD, then re-computes velocity.
    v = (XPBD(x + tau*v_pred) - x) / tau
    """
    def __init__(
        self,
        d_model: int,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        xpbd_iters: int = 4,
        xpbd_compliance: float = 0.0,
        xpbd_max_corr: float = 0.15,
        use_tangent: bool = False,
        tau_min: float = 1e-3,
    ):
        super().__init__(tau_min=tau_min)
        self.head = nn.Linear(d_model, 3)
        self.projector = XPBDProjector(
            edge_index=edge_index,
            rest_lengths=rest_lengths,
            iters=xpbd_iters,
            compliance=xpbd_compliance,
            max_corr=xpbd_max_corr,
        )
        self.register_buffer("edge_index", edge_index.clone().detach())
        self.tangent_projector = TangentProjector(edge_index) if use_tangent else None

    def predict(self, model, keypoints, timesteps, hand_tokens_out) -> torch.Tensor:
        tau = self._tau(timesteps)
        v_pred = self.head(hand_tokens_out)
        
        x_raw = keypoints + tau * v_pred
        x_corr = self.projector(x_raw)
        
        v = (x_corr - keypoints) / tau
        v = self.apply_tangent_projection(v, keypoints)
        return v


# Velocity Strategy Registry

def _build_direct(d_model, edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for DirectVelocityStrategy"""
    use_tangent = bool(kwargs.get("use_tangent", False))
    return DirectVelocityStrategy(d_model=d_model, use_tangent=use_tangent, edge_index=edge_index)

def _build_direct_tangent(d_model, edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for DirectVelocityStrategy with tangent=True"""
    use_tangent = bool(kwargs.get("use_tangent", True))
    return DirectVelocityStrategy(d_model=d_model, use_tangent=use_tangent, edge_index=edge_index)

def _build_goal_kabsch(d_model, edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for GoalKabschVelocityStrategy"""
    if template_xyz is None:
        raise ValueError("template_xyz required for goal_kabsch strategy")
    tau_min = float(kwargs.get("tau_min", 1e-3))
    use_tangent = bool(kwargs.get("use_tangent", True))
    return GoalKabschVelocityStrategy(
        d_model=d_model,
        template_xyz=template_xyz,
        groups=groups,
        edge_index=edge_index,
        tau_min=tau_min,
        use_tangent=use_tangent,
    )

def _build_group_rigid(d_model, edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for GroupRigidParamVelocityStrategy"""
    if template_xyz is None:
        raise ValueError("template_xyz required for group_rigid strategy")
    tau_min = float(kwargs.get("tau_min", 1e-3))
    use_tangent = bool(kwargs.get("use_tangent", True))
    return GroupRigidParamVelocityStrategy(
        d_model=d_model,
        template_xyz=template_xyz,
        groups=groups,
        edge_index=edge_index,
        use_tangent=use_tangent,
        tau_min=tau_min,
    )

def _build_pbd_corrected(d_model, edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for PBDCorrectedVelocityStrategy"""
    return PBDCorrectedVelocityStrategy(
        d_model=d_model,
        edge_index=edge_index,
        rest_lengths=rest_lengths,
        xpbd_iters=int(kwargs.get("iters", 4)),
        xpbd_compliance=float(kwargs.get("compliance", 0.0)),
        xpbd_max_corr=float(kwargs.get("max_corr", 0.15)),
        use_tangent=bool(kwargs.get("use_tangent", False)),
        tau_min=float(kwargs.get("tau_min", 1e-3)),
    )

def _build_phys_guided(d_model, edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for PhysGuidedVelocityStrategy"""
    return PhysGuidedVelocityStrategy(
        d_model=d_model,
        edge_index=edge_index,
        rest_lengths=rest_lengths,
        eta=float(kwargs.get("eta", 0.5)),
        use_tangent=bool(kwargs.get("use_tangent", True)),
    )

VELOCITY_STRATEGY_REGISTRY = {
    "direct": {
        "builder": _build_direct,
        "allowed_kwargs": {"use_tangent"},
    },
    "direct_tangent": {
        "builder": _build_direct_tangent,
        "allowed_kwargs": {"use_tangent"},
    },
    "goal_kabsch": {
        "builder": _build_goal_kabsch,
        "allowed_kwargs": {"tau_min", "use_tangent"},
    },
    "group_rigid": {
        "builder": _build_group_rigid,
        "allowed_kwargs": {"tau_min", "use_tangent"},
    },
    "pbd_corrected": {
        "builder": _build_pbd_corrected,
        "allowed_kwargs": {"iters", "compliance", "max_corr", "use_tangent", "tau_min"},
    },
    "phys_guided": {
        "builder": _build_phys_guided,
        "allowed_kwargs": {"eta", "use_tangent"},
    },
}

def build_velocity_strategy(
    mode: str,
    d_model: int,
    edge_index: torch.Tensor,
    rest_lengths: torch.Tensor,
    template_xyz: Optional[torch.Tensor],
    groups: List[torch.Tensor],
    kwargs: Optional[dict] = None,
) -> VelocityStrategyBase:
    """
    Factory function to build a velocity strategy from registry.
    
    Args:
        mode: Strategy mode name (e.g., 'direct', 'goal_kabsch', etc.)
        d_model: Model dimension
        edge_index: Graph edge indices
        rest_lengths: Rest lengths for edges
        template_xyz: Template keypoints (optional, required for some strategies)
        groups: Rigid groups (required for some strategies)
        kwargs: Additional strategy-specific keyword arguments
    
    Returns:
        VelocityStrategyBase instance
    
    Raises:
        ValueError: If mode is unknown or if unsupported kwargs are provided
    """
    mode_lower = str(mode).lower()
    kwargs = dict(kwargs or {})
    
    if mode_lower not in VELOCITY_STRATEGY_REGISTRY:
        available = sorted(VELOCITY_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown velocity_mode: '{mode}'. Available modes: {available}")
    
    entry = VELOCITY_STRATEGY_REGISTRY[mode_lower]
    allowed = entry["allowed_kwargs"]
    
    # Validate kwargs
    extra = sorted(set(kwargs.keys()) - allowed)
    if extra:
        raise ValueError(
            f"Unsupported velocity_kwargs for mode '{mode}': {extra}. "
            f"Allowed: {sorted(allowed)}"
        )
    
    builder = entry["builder"]
    return builder(d_model, edge_index, rest_lengths, template_xyz, groups, kwargs)


# State Projector Registry

def _build_projector_none(edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for identity/no-op projector"""
    return None

def _build_projector_pbd(edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for XPBDProjector"""
    return XPBDProjector(
        edge_index=edge_index,
        rest_lengths=rest_lengths,
        iters=int(kwargs.get("iters", 2)),
        compliance=float(kwargs.get("compliance", 0.0)),
        max_corr=float(kwargs.get("max_corr", 0.2)),
    )

def _build_projector_rigid(edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for RigidGroupProjector"""
    if template_xyz is None:
        raise ValueError("template_xyz required for 'rigid' state projection mode")
    return RigidGroupProjector(template_xyz=template_xyz, groups=groups)

def _build_projector_hybrid(edge_index, rest_lengths, template_xyz, groups, kwargs):
    """Builder for hybrid projector (rigid + pbd)"""
    if template_xyz is None:
        raise ValueError("template_xyz required for 'hybrid' state projection mode")
    
    rigid_proj = RigidGroupProjector(template_xyz=template_xyz, groups=groups)
    pbd_proj = XPBDProjector(
        edge_index=edge_index,
        rest_lengths=rest_lengths,
        iters=int(kwargs.get("hybrid_pbd_iters", 2)),
        compliance=float(kwargs.get("hybrid_compliance", 0.0)),
        max_corr=float(kwargs.get("hybrid_max_corr", 0.2)),
    )
    
    # Return a tuple for hybrid mode
    return ("hybrid", rigid_proj, pbd_proj)

STATE_PROJECTOR_REGISTRY = {
    "none": {
        "builder": _build_projector_none,
        "allowed_kwargs": set(),
    },
    "pbd": {
        "builder": _build_projector_pbd,
        "allowed_kwargs": {"iters", "compliance", "max_corr"},
    },
    "rigid": {
        "builder": _build_projector_rigid,
        "allowed_kwargs": set(),
    },
    "hybrid": {
        "builder": _build_projector_hybrid,
        "allowed_kwargs": {"hybrid_pbd_iters", "hybrid_compliance", "hybrid_max_corr"},
    },
}

def build_state_projector(
    mode: str,
    edge_index: torch.Tensor,
    rest_lengths: torch.Tensor,
    template_xyz: Optional[torch.Tensor],
    groups: List[torch.Tensor],
    kwargs: Optional[dict] = None,
):
    """
    Factory function to build a state projector from registry.
    
    Args:
        mode: Projector mode ('none', 'pbd', 'rigid', 'hybrid')
        edge_index: Graph edge indices
        rest_lengths: Rest lengths for edges
        template_xyz: Template keypoints (optional, required for rigid/hybrid)
        groups: Rigid groups (required for rigid/hybrid)
        kwargs: Additional projector-specific keyword arguments
    
    Returns:
        Projector module or None for 'none' mode.
        For 'hybrid' mode, returns tuple ('hybrid', rigid_proj, pbd_proj)
    
    Raises:
        ValueError: If mode is unknown or if unsupported kwargs are provided
    """
    mode_lower = str(mode).lower()
    kwargs = dict(kwargs or {})
    
    if mode_lower not in STATE_PROJECTOR_REGISTRY:
        available = sorted(STATE_PROJECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown state_projection_mode: '{mode}'. Available modes: {available}")
    
    entry = STATE_PROJECTOR_REGISTRY[mode_lower]
    allowed = entry["allowed_kwargs"]
    
    # Validate kwargs
    extra = sorted(set(kwargs.keys()) - allowed)
    if extra:
        raise ValueError(
            f"Unsupported state_projection_kwargs for mode '{mode}': {extra}. "
            f"Allowed: {sorted(allowed)}"
        )
    
    builder = entry["builder"]
    return builder(edge_index, rest_lengths, template_xyz, groups, kwargs)
