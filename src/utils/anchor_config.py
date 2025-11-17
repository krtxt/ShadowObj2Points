"""
Shared anchor-related configuration utilities.

This module centralizes the palm_center margin so that both datasets and
hand_helper read the same value. It supports:
- Programmatic override via set_palm_center_margin(…)
- Environment variable override via HAND_ANCHOR_PALM_CENTER_MARGIN (or PALM_CENTER_MARGIN)
- Sensible default of 0.25
"""
from __future__ import annotations

import os
from typing import Optional

# In-memory override (takes precedence over env/default)
_PALM_CENTER_MARGIN_OVERRIDE: Optional[float] = None

# Default margin if no override is provided
_DEFAULT_PALM_CENTER_MARGIN: float = 0.25

# Env var keys checked in order
_ENV_KEYS = (
    "HAND_ANCHOR_PALM_CENTER_MARGIN",
    "PALM_CENTER_MARGIN",
)


def set_palm_center_margin(value: float) -> None:
    """Set a global override for the palm_center margin.

    This override has the highest priority (above env vars).
    """
    global _PALM_CENTER_MARGIN_OVERRIDE
    try:
        _PALM_CENTER_MARGIN_OVERRIDE = float(value)
    except Exception:
        # Silently ignore invalid values to avoid raising during config parsing
        pass


def get_palm_center_margin() -> float:
    """Get the palm_center margin to use across the codebase.

    Priority:
    1) In-memory override set via set_palm_center_margin
    2) Environment variable HAND_ANCHOR_PALM_CENTER_MARGIN (or PALM_CENTER_MARGIN)
    3) Default value (0.25)
    """
    if _PALM_CENTER_MARGIN_OVERRIDE is not None:
        return _PALM_CENTER_MARGIN_OVERRIDE

    for k in _ENV_KEYS:
        val = os.environ.get(k)
        if val is not None:
            try:
                return float(val)
            except Exception:
                # If env var exists but invalid, fall back to next option
                continue

    return _DEFAULT_PALM_CENTER_MARGIN

