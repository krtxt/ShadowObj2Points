"""Visualization callbacks."""

from __future__ import annotations

from typing import Optional

from lightning.pytorch.callbacks import Callback

from .image_logger import ImageLogger


class HandVisCallback(ImageLogger):
    """Backward-compatible alias for the legacy hand visualization callback.

    Older configs referenced ``callbacks.visualization.HandVisCallback`` but the
    implementation was renamed to :class:`ImageLogger`.  This thin wrapper keeps
    the existing configs working while still exposing the functionality through
    :class:`ImageLogger`.
    """

    def __init__(
        self,
        num_samples: int = 4,
        log_key: str = "val/hand_samples",
        sample_num_steps: Optional[int] = None,
        save_to_logger: bool = False,
        save_to_disk: bool = True,
        output_dir: Optional[str] = None,
        html_subdir: str = "val_visualizations",
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            log_key=log_key,
            sample_steps=sample_num_steps,
            save_to_logger=save_to_logger,
            save_to_disk=save_to_disk,
            output_dir=output_dir,
            html_subdir=html_subdir,
        )
