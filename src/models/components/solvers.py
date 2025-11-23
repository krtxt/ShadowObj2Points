"""ODE Solver strategies for flow matching sampling."""

import torch
from typing import Callable, Optional


class ODESolverBase:
    """Base class for ODE solvers used in flow matching sampling."""
    
    def __init__(self, name: str):
        self.name = name
    
    def step(
        self,
        keypoints: torch.Tensor,
        velocity_fn: Callable,
        t_curr: float,
        t_next: float,
        scene_pc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take one ODE integration step.
        
        Args:
            keypoints: Current keypoints (B, N, 3)
            velocity_fn: Function that computes velocity given (keypoints, scene_pc, timesteps)
            t_curr: Current timestep scalar
            t_next: Next timestep scalar
            scene_pc: Scene point cloud (B, P, 3)
        
        Returns:
            Updated keypoints (B, N, 3)
        """
        raise NotImplementedError


class EulerSolver(ODESolverBase):
    """Explicit Euler method (first-order)."""
    
    def __init__(self):
        super().__init__("euler")
    
    def step(
        self,
        keypoints: torch.Tensor,
        velocity_fn: Callable,
        t_curr: float,
        t_next: float,
        scene_pc: torch.Tensor,
    ) -> torch.Tensor:
        B = keypoints.shape[0]
        device = keypoints.device
        dt = t_next - t_curr
        
        t_batch = torch.full((B,), t_curr, device=device, dtype=keypoints.dtype)
        velocity = velocity_fn(keypoints, scene_pc, t_batch)
        
        return keypoints + dt * velocity


class HeunSolver(ODESolverBase):
    """Heun's method / RK2 (second-order predictor-corrector)."""
    
    def __init__(self):
        super().__init__("heun")
    
    def step(
        self,
        keypoints: torch.Tensor,
        velocity_fn: Callable,
        t_curr: float,
        t_next: float,
        scene_pc: torch.Tensor,
    ) -> torch.Tensor:
        B = keypoints.shape[0]
        device = keypoints.device
        dt = t_next - t_curr
        
        # Predictor: evaluate at t_curr
        t_batch_curr = torch.full((B,), t_curr, device=device, dtype=keypoints.dtype)
        v1 = velocity_fn(keypoints, scene_pc, t_batch_curr)
        
        # Predictor step
        keypoints_pred = keypoints + dt * v1
        
        # Corrector: evaluate at t_next
        t_batch_next = torch.full((B,), t_next, device=device, dtype=keypoints.dtype)
        v2 = velocity_fn(keypoints_pred, scene_pc, t_batch_next)
        
        # Corrector step (average of two slopes)
        return keypoints + 0.5 * dt * (v1 + v2)


class RK4Solver(ODESolverBase):
    """Classic 4th-order Runge-Kutta method."""
    
    def __init__(self):
        super().__init__("rk4")
    
    def step(
        self,
        keypoints: torch.Tensor,
        velocity_fn: Callable,
        t_curr: float,
        t_next: float,
        scene_pc: torch.Tensor,
    ) -> torch.Tensor:
        B = keypoints.shape[0]
        device = keypoints.device
        dt = t_next - t_curr
        
        # k1: slope at beginning
        t_batch_curr = torch.full((B,), t_curr, device=device, dtype=keypoints.dtype)
        v1 = velocity_fn(keypoints, scene_pc, t_batch_curr)
        
        # k2, k3: slopes at midpoint
        half_step_time = t_curr + 0.5 * dt
        t_batch_half = torch.full((B,), half_step_time, device=device, dtype=keypoints.dtype)
        
        v2 = velocity_fn(keypoints + 0.5 * dt * v1, scene_pc, t_batch_half)
        v3 = velocity_fn(keypoints + 0.5 * dt * v2, scene_pc, t_batch_half)
        
        # k4: slope at end
        t_batch_next = torch.full((B,), t_next, device=device, dtype=keypoints.dtype)
        v4 = velocity_fn(keypoints + dt * v3, scene_pc, t_batch_next)
        
        # Weighted combination
        return keypoints + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)


# Solver Registry

SOLVER_REGISTRY = {
    "euler": EulerSolver,
    "heun": HeunSolver,
    "rk2": HeunSolver,  # Alias
    "rk4": RK4Solver,
    "runge_kutta4": RK4Solver,  # Alias
}


def build_solver(solver_name: str) -> ODESolverBase:
    """
    Factory function to build an ODE solver from registry.
    
    Args:
        solver_name: Solver name ('euler', 'heun'/'rk2', 'rk4'/'runge_kutta4')
    
    Returns:
        ODESolverBase instance
    
    Raises:
        ValueError: If solver_name is unknown
    """
    solver_lower = str(solver_name).lower()
    
    if solver_lower not in SOLVER_REGISTRY:
        available = sorted(set(SOLVER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown solver: '{solver_name}'. Available solvers: {available}"
        )
    
    solver_class = SOLVER_REGISTRY[solver_lower]
    return solver_class()
