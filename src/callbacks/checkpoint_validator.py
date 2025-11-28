"""Checkpoint validation callback for verifying checkpoint integrity."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

log = logging.getLogger(__name__)


class CheckpointValidator(Callback):
    """Validate checkpoint integrity before and after saving.
    
    This callback provides:
    - Pre-save validation of model state
    - Post-save verification of checkpoint file
    - Checkpoint loading test (optional)
    - Corruption detection
    
    Args:
        validate_on_save: Validate checkpoint immediately after saving.
        validate_state_dict: Check state_dict keys and tensor shapes.
        validate_loadable: Actually try to load the checkpoint (slower but thorough).
        log_checkpoint_size: Log the size of saved checkpoints.
        required_keys: List of keys that must exist in the checkpoint.
    """

    def __init__(
        self,
        validate_on_save: bool = True,
        validate_state_dict: bool = True,
        validate_loadable: bool = False,
        log_checkpoint_size: bool = True,
        required_keys: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.validate_on_save = validate_on_save
        self.validate_state_dict = validate_state_dict
        self.validate_loadable = validate_loadable
        self.log_checkpoint_size = log_checkpoint_size
        self.required_keys = required_keys or ["state_dict", "epoch", "global_step"]
        
        self._last_checkpoint_path: Optional[str] = None
        self._validation_errors: list = []

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def _validate_state_dict(self, state_dict: Dict[str, Any]) -> list:
        """Validate state_dict for common issues.
        
        Returns:
            List of validation error messages.
        """
        errors = []
        
        if not state_dict:
            errors.append("State dict is empty")
            return errors
        
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Check for NaN values
                if torch.isnan(value).any():
                    errors.append(f"NaN values found in {key}")
                # Check for Inf values
                if torch.isinf(value).any():
                    errors.append(f"Inf values found in {key}")
                # Check for zero-sized tensors (potential corruption)
                if value.numel() == 0:
                    errors.append(f"Zero-sized tensor in {key}")
        
        return errors

    def _validate_checkpoint_file(self, checkpoint_path: str) -> tuple[bool, list]:
        """Validate a checkpoint file.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        path = Path(checkpoint_path)
        
        # Check file exists
        if not path.exists():
            return False, ["Checkpoint file does not exist"]
        
        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            return False, ["Checkpoint file is empty (0 bytes)"]
        
        if self.log_checkpoint_size:
            log.info(f"Checkpoint size: {self._format_size(file_size)}")
        
        try:
            # Try to load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Check required keys
            for key in self.required_keys:
                if key not in checkpoint:
                    errors.append(f"Missing required key: {key}")
            
            # Validate state_dict if requested
            if self.validate_state_dict and "state_dict" in checkpoint:
                state_errors = self._validate_state_dict(checkpoint["state_dict"])
                errors.extend(state_errors)
            
            # Log checkpoint info
            if "epoch" in checkpoint:
                log.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if "global_step" in checkpoint:
                log.info(f"Checkpoint global_step: {checkpoint['global_step']}")
                
        except Exception as e:
            return False, [f"Failed to load checkpoint: {str(e)}"]
        
        return len(errors) == 0, errors

    @rank_zero_only
    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Validate checkpoint before saving."""
        if not self.validate_state_dict:
            return
        
        if "state_dict" in checkpoint:
            errors = self._validate_state_dict(checkpoint["state_dict"])
            if errors:
                log.warning(f"Pre-save checkpoint validation warnings: {errors}")
                self._validation_errors.extend(errors)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Check for new checkpoints and validate them."""
        if not self.validate_on_save:
            return
        
        # Get checkpoint callback
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return
        
        # Check if a new checkpoint was saved
        current_best = getattr(ckpt_callback, "best_model_path", None)
        current_last = getattr(ckpt_callback, "last_model_path", None)
        
        paths_to_check = []
        if current_best and current_best != self._last_checkpoint_path:
            paths_to_check.append(("best", current_best))
        if current_last and current_last not in [p[1] for p in paths_to_check]:
            paths_to_check.append(("last", current_last))
        
        for ckpt_type, ckpt_path in paths_to_check:
            if ckpt_path and Path(ckpt_path).exists():
                is_valid, errors = self._validate_checkpoint_file(ckpt_path)
                if is_valid:
                    log.info(f"Checkpoint ({ckpt_type}) validated successfully: {ckpt_path}")
                else:
                    log.error(f"Checkpoint ({ckpt_type}) validation failed: {errors}")
                    self._validation_errors.extend(errors)
        
        if current_best:
            self._last_checkpoint_path = current_best

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Final validation summary."""
        if self._validation_errors:
            log.warning(
                f"Training completed with {len(self._validation_errors)} checkpoint validation warnings"
            )
        else:
            log.info("All checkpoint validations passed")


def validate_checkpoint_before_resume(
    checkpoint_path: str,
    required_keys: Optional[list] = None,
    log_instance: Optional[logging.Logger] = None,
) -> tuple[bool, Dict[str, Any], list]:
    """Validate a checkpoint file before resuming training.
    
    This is a standalone function that can be called before trainer.fit()
    to ensure the checkpoint is valid.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        required_keys: List of keys that must exist in the checkpoint.
        log_instance: Logger instance for output.
    
    Returns:
        Tuple of (is_valid, checkpoint_dict, list_of_errors)
    """
    log_instance = log_instance or log
    required_keys = required_keys or ["state_dict", "epoch", "global_step"]
    errors = []
    
    path = Path(checkpoint_path)
    if not path.exists():
        return False, {}, ["Checkpoint file does not exist"]
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Check required keys
        for key in required_keys:
            if key not in checkpoint:
                errors.append(f"Missing required key: {key}")
        
        # Check state_dict for issues
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        errors.append(f"NaN values in state_dict[{key}]")
                    if torch.isinf(value).any():
                        errors.append(f"Inf values in state_dict[{key}]")
        
        if errors:
            log_instance.warning(f"Checkpoint validation warnings: {errors}")
        else:
            log_instance.info(f"Checkpoint validated: epoch={checkpoint.get('epoch')}, "
                            f"global_step={checkpoint.get('global_step')}")
        
        return len(errors) == 0, checkpoint, errors
        
    except Exception as e:
        return False, {}, [f"Failed to load checkpoint: {str(e)}"]
