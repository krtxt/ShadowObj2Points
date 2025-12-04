"""Time scheduling strategies for flow matching sampling."""

import torch
from typing import Callable, Optional, Dict, Any


class TimeScheduleBase:
    """Base class for time scheduling strategies.
    
    Time schedules control how timesteps are distributed in [0, 1]
    during ODE integration for sampling.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def generate(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Generate timesteps from 0 to 1.
        
        Args:
            num_steps: Number of integration steps
            device: Device to create tensor on
            
        Returns:
            Timesteps tensor of shape (num_steps + 1,)
        """
        raise NotImplementedError


class LinearSchedule(TimeScheduleBase):
    """Uniform linear time schedule: t = [0, 1/N, 2/N, ..., 1]."""
    
    def __init__(self):
        super().__init__("linear")
    
    def generate(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device)


class CosineSchedule(TimeScheduleBase):
    """Cosine time schedule: t' = 1 - cos(t * pi/2).
    
    Used in diffusion models to allocate more steps near t=0 (noise).
    """
    
    def __init__(self):
        super().__init__("cosine")
    
    def generate(self, num_steps: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        return 1.0 - torch.cos(t * (torch.pi / 2))


class ShiftSchedule(TimeScheduleBase):
    """Time shift schedule: t' = (s * t) / (1 + (s - 1) * t).
    
    From Stable Diffusion 3 / Flux. Shift factor s > 1 pushes steps
    toward t=1 (data), prioritizing low-noise refinement.
    
    Args:
        shift: Shift factor (default: 1.0 = linear, > 1.0 = more steps near data)
    """
    
    def __init__(self, shift: float = 1.0):
        super().__init__("shift")
        self.shift = float(shift)
    
    def generate(self, num_steps: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        s = self.shift
        if abs(s - 1.0) < 1e-6:
            return t
        return (s * t) / (1 + (s - 1) * t)


TIME_SCHEDULE_REGISTRY = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
    "shift": ShiftSchedule,
}


def build_time_schedule(
    schedule_name: str,
    shift: Optional[float] = None,
) -> TimeScheduleBase:
    """Factory function to build a time schedule from registry.
    
    Args:
        schedule_name: Schedule type ('linear', 'cosine', 'shift')
        shift: Shift factor for shift schedule (ignored for others)
    
    Returns:
        TimeScheduleBase instance
    
    Raises:
        ValueError: If schedule_name is unknown
    """
    schedule_lower = str(schedule_name).lower()
    
    if schedule_lower not in TIME_SCHEDULE_REGISTRY:
        available = sorted(TIME_SCHEDULE_REGISTRY.keys())
        raise ValueError(
            f"Unknown time schedule: '{schedule_name}'. Available: {available}"
        )
    
    schedule_class = TIME_SCHEDULE_REGISTRY[schedule_lower]
    
    if schedule_lower == "shift" and shift is not None:
        return schedule_class(shift=shift)
    
    return schedule_class()


# --------------------------------------------------------------------------
# Training time samplers (for stochastic timestep draws during training)
# --------------------------------------------------------------------------

def build_training_time_sampler(
    name: str = "uniform",
    params: Optional[Dict[str, Any]] = None,
) -> Callable[[int, torch.device, torch.dtype], torch.Tensor]:
    """
    Build a callable that samples timesteps for training.
    
    Args:
        name: Sampler name ('uniform' or 'logit_normal')
        params: Optional dict of sampler-specific params.
            - uniform: t_min (float), t_max (float)
            - logit_normal: mu (float), sigma (float), t_min (float), t_max (float)
    
    Returns:
        Callable: f(batch_size, device, dtype) -> (B,) timesteps in [0, 1]
    """
    params = dict(params or {})
    sampler_name = str(name).lower()
    
    if sampler_name == "uniform":
        t_min = float(params.get("t_min", 0.0))
        t_max = float(params.get("t_max", 1.0))
        t_min = max(0.0, min(t_min, 1.0))
        t_max = max(t_min, min(t_max, 1.0))

        def _sample(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            t = torch.rand(batch_size, device=device, dtype=dtype)
            if t_min != 0.0 or t_max != 1.0:
                t = t_min + (t_max - t_min) * t
            return t.clamp(0.0, 1.0)

        return _sample

    if sampler_name == "logit_normal":
        mu = float(params.get("mu", -0.8))
        sigma = float(params.get("sigma", 1.6))
        t_min = float(params.get("t_min", 1e-4))
        t_max = float(params.get("t_max", 1.0 - 1e-4))
        t_min = max(0.0, min(t_min, 1.0))
        t_max = max(t_min, min(t_max, 1.0))

        def _sample(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            eps = torch.randn(batch_size, device=device, dtype=dtype)
            logits = mu + sigma * eps
            t = torch.sigmoid(logits)
            return t.clamp(t_min, t_max)

        return _sample

    raise ValueError(
        f"Unknown training time sampler '{name}'. "
        "Supported: 'uniform', 'logit_normal'."
    )
