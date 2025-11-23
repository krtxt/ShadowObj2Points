"""Sampling configuration encapsulation."""

from dataclasses import dataclass
from typing import Optional, Any, Dict
import torch.nn as nn

from .solvers import build_solver, ODESolverBase
from .time_schedules import build_time_schedule, TimeScheduleBase
from .velocity_strategies import build_state_projector


@dataclass
class SamplingConfig:
    """Encapsulates sampling-related configuration.
    
    Attributes:
        num_steps: Number of ODE integration steps
        solver: ODE solver instance (Euler, Heun, RK4)
        time_schedule: Time schedule instance for timestep distribution
        state_projector: State projection module (optional)
        rigid_projector: Rigid group projector for hybrid mode (optional)
        pbd_projector: PBD projector for hybrid mode (optional)
    """
    
    num_steps: int
    solver: ODESolverBase
    time_schedule: TimeScheduleBase
    state_projector: Optional[nn.Module] = None
    rigid_projector: Optional[nn.Module] = None
    pbd_projector: Optional[nn.Module] = None
    
    @classmethod
    def from_params(
        cls,
        num_steps: int,
        solver_name: str,
        schedule_name: str,
        schedule_shift: float,
        projector_mode: str,
        projector_kwargs: Dict[str, Any],
        edge_index: Any,
        rest_lengths: Any,
        template_xyz: Any,
        rigid_groups: Any,
    ) -> "SamplingConfig":
        """Build SamplingConfig from individual parameters.
        
        Args:
            num_steps: Number of ODE steps
            solver_name: Solver type ('euler', 'heun', 'rk4')
            schedule_name: Time schedule ('linear', 'cosine', 'shift')
            schedule_shift: Shift factor for shift schedule
            projector_mode: State projection mode ('none', 'pbd', 'rigid', 'hybrid')
            projector_kwargs: Additional projector configuration
            edge_index: Graph edge indices
            rest_lengths: Edge rest lengths
            template_xyz: Template keypoints (for rigid/hybrid)
            rigid_groups: Rigid groups (for rigid/hybrid)
        
        Returns:
            SamplingConfig instance
        """
        solver = build_solver(solver_name)
        time_schedule = build_time_schedule(schedule_name, shift=schedule_shift)
        
        projector_result = build_state_projector(
            mode=projector_mode,
            edge_index=edge_index,
            rest_lengths=rest_lengths,
            template_xyz=template_xyz,
            groups=rigid_groups,
            kwargs=projector_kwargs,
        )
        
        if isinstance(projector_result, tuple) and projector_result[0] == "hybrid":
            _, rigid_proj, pbd_proj = projector_result
            return cls(
                num_steps=num_steps,
                solver=solver,
                time_schedule=time_schedule,
                state_projector=None,
                rigid_projector=rigid_proj,
                pbd_projector=pbd_proj,
            )
        else:
            return cls(
                num_steps=num_steps,
                solver=solver,
                time_schedule=time_schedule,
                state_projector=projector_result,
                rigid_projector=None,
                pbd_projector=None,
            )
    
    def is_hybrid(self) -> bool:
        """Check if using hybrid projection mode."""
        return self.rigid_projector is not None and self.pbd_projector is not None
