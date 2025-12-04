# models/components/loss_scheduler.py
"""
Curriculum Loss Scheduler for multi-stage training.

Supports smooth weight transitions between training stages, allowing
gradual introduction of physics and structural constraints.
"""

import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# Default weights when curriculum is disabled.
# These keys correspond to registered LossComponent names in LOSS_COMPONENT_REGISTRY.
# Components with weight=0 are not computed by LossManager (saves compute).
DEFAULT_WEIGHTS = {
    "l1": 2.0,           # Primary position regression (MAE)
    "chamfer": 1.0,      # Bidirectional point cloud distance for global shape consistency
    "direction": 1.0,    # Edge direction alignment for correct bone orientations
    "edge_len": 1.0,     # Edge length consistency for skeleton proportions
    "bone": 0.0,         # Bone length regularization vs template (weight=0 skips computation)
    "collision": 0.25,   # Penetration penalty against scene geometry
}

LEGACY_WEIGHT_ALIASES = {
    "smooth_l1": "l1",
}

# Default curriculum stages with rationale:
# Stage 1 (Reconstruction): Focus on learning basic point positions
# Stage 2 (Structure): Introduce skeletal constraints for valid hand topology
# Stage 3 (Physics): Add collision avoidance for physically plausible grasps
DEFAULT_STAGES = [
    {
        "name": "reconstruction",
        "epochs": [0, 30],
        "weights": {
            "l1": 2.0,
            "chamfer": 1.0,
            "direction": 0.5,
            "edge_len": 0.5,
            "bone": 0.0,
            "collision": 0.0,
        },
    },
    {
        "name": "structure",
        "epochs": [30, 60],
        "weights": {
            "l1": 2.0,
            "chamfer": 1.0,
            "direction": 1.0,
            "edge_len": 1.0,
            "bone": 0.5,
            "collision": 0.0,
        },
    },
    {
        "name": "physics",
        "epochs": [60, None],
        "weights": {
            "l1": 2.0,
            "chamfer": 1.0,
            "direction": 1.0,
            "edge_len": 1.0,
            "bone": 1.0,
            "collision": 0.25,
        },
    },
]


class LossScheduler:
    """
    Curriculum loss weight scheduler with smooth stage transitions.

    Loss Terms Explained:
    - l1: Position regression loss (MAE). Core reconstruction signal.
    - chamfer: Chamfer distance ensuring global point distribution matches GT.
    - direction: Cosine similarity of edge directions. Ensures correct bone orientations.
    - edge_len: Relative edge length error. Maintains skeleton proportions.
    - bone: Bone length deviation from template. Structural regularization.
    - collision: Penetration depth penalty. Prevents hand-object intersection.

    Training Strategy:
    1. Reconstruction phase: Learn approximate positions with l1 + chamfer.
    2. Structure phase: Refine with direction + edge_len + bone for valid topology.
    3. Physics phase: Add collision for physically plausible results.
    """

    def __init__(
        self,
        enabled: bool = False,
        stages: Optional[List[Dict]] = None,
        warmup_epochs: int = 5,
        default_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            enabled: Whether curriculum scheduling is active.
            stages: List of stage configs, each with 'name', 'epochs', 'weights'.
            warmup_epochs: Number of epochs for smooth transition between stages.
            default_weights: Weights to use when curriculum is disabled.
        """
        self.enabled = enabled
        self.warmup_epochs = max(0, warmup_epochs)
        self.default_weights = self._normalize_weight_keys(default_weights or DEFAULT_WEIGHTS.copy())

        if enabled and stages:
            self.stages = self._validate_stages(stages)
        else:
            self.stages = []

        self._current_epoch = 0
        self._current_stage_name = None
        self._logged_stage = None

    def _normalize_weight_keys(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize legacy weight keys to current names."""
        normalized: Dict[str, float] = {}
        for k, v in weights.items():
            key = LEGACY_WEIGHT_ALIASES.get(k, k)
            if key in DEFAULT_WEIGHTS:
                normalized[key] = float(v)
            else:
                logger.warning(f"Unknown loss key '{k}', ignored.")
        # Ensure all default keys exist
        for k, v in DEFAULT_WEIGHTS.items():
            normalized.setdefault(k, float(v))
        return normalized

    def _validate_stages(self, stages: List[Dict]) -> List[Dict]:
        """Validate and normalize stage configurations."""
        validated = []
        all_keys = set(DEFAULT_WEIGHTS.keys())

        for i, stage in enumerate(stages):
            name = stage.get("name", f"stage_{i}")
            epochs = stage.get("epochs", [0, None])
            weights = self._normalize_weight_keys(stage.get("weights", {}))

            # Ensure all weight keys are present
            full_weights = DEFAULT_WEIGHTS.copy()
            for k, v in weights.items():
                if k in all_keys:
                    full_weights[k] = float(v)
                else:
                    logger.warning(f"Unknown loss key '{k}' in stage '{name}', ignored.")

            # Parse epoch range
            start = epochs[0] if epochs[0] is not None else 0
            end = epochs[1]  # None means infinity

            validated.append({
                "name": name,
                "start": start,
                "end": end,
                "weights": full_weights,
            })

        # Sort by start epoch
        validated.sort(key=lambda x: x["start"])
        return validated

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for weight computation."""
        self._current_epoch = epoch

    def get_weights(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Get loss weights for the given epoch.

        Args:
            epoch: Training epoch (uses internal state if None).

        Returns:
            Dictionary of loss term weights.
        """
        if epoch is not None:
            self._current_epoch = epoch

        if not self.enabled or not self.stages:
            return self.default_weights.copy()

        epoch = self._current_epoch
        return self._compute_weights(epoch)

    def _compute_weights(self, epoch: int) -> Dict[str, float]:
        """Compute interpolated weights based on current epoch."""
        # Find current and next stage
        current_stage = None
        next_stage = None

        for i, stage in enumerate(self.stages):
            start, end = stage["start"], stage["end"]
            if start <= epoch and (end is None or epoch < end):
                current_stage = stage
                if i + 1 < len(self.stages):
                    next_stage = self.stages[i + 1]
                break

        if current_stage is None:
            # Before first stage or after all stages
            if epoch < self.stages[0]["start"]:
                current_stage = self.stages[0]
            else:
                current_stage = self.stages[-1]

        # Log stage transitions
        if current_stage["name"] != self._logged_stage:
            logger.info(f"[LossScheduler] Entering stage: {current_stage['name']} at epoch {epoch}")
            self._logged_stage = current_stage["name"]
        self._current_stage_name = current_stage["name"]

        # Check if we need interpolation (warmup into next stage)
        if next_stage is not None and self.warmup_epochs > 0:
            transition_start = next_stage["start"] - self.warmup_epochs
            if epoch >= transition_start:
                # Interpolate between current and next stage
                progress = (epoch - transition_start) / self.warmup_epochs
                progress = min(1.0, max(0.0, progress))
                return self._interpolate_weights(
                    current_stage["weights"],
                    next_stage["weights"],
                    progress,
                )

        return current_stage["weights"].copy()

    def _interpolate_weights(
        self,
        weights_a: Dict[str, float],
        weights_b: Dict[str, float],
        t: float,
    ) -> Dict[str, float]:
        """Linear interpolation between two weight dicts."""
        result = {}
        all_keys = set(weights_a.keys()) | set(weights_b.keys())
        for k in all_keys:
            wa = weights_a.get(k, 0.0)
            wb = weights_b.get(k, 0.0)
            result[k] = wa + t * (wb - wa)
        return result

    def get_current_stage_name(self) -> Optional[str]:
        """Return name of the current training stage."""
        return self._current_stage_name

    def __repr__(self) -> str:
        if not self.enabled:
            return f"LossScheduler(enabled=False, weights={self.default_weights})"
        stage_names = [s["name"] for s in self.stages]
        return f"LossScheduler(enabled=True, stages={stage_names}, warmup={self.warmup_epochs})"
