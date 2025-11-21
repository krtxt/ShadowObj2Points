import copy
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

class ModelEma(Callback):
    """
    Maintains an Exponential Moving Average (EMA) of the model parameters.
    Swaps weights during validation and testing to evaluate the EMA model.
    """
    def __init__(
        self,
        decay: float = 0.999,
        ema_device: Optional[str] = None,
        update_every_n_steps: int = 4,
    ):
        super().__init__()
        self.decay = decay
        self.ema_device = ema_device
        self.update_every_n_steps = update_every_n_steps
        
        self.ema_model: Optional[nn.Module] = None
        self._backup_params: Optional[Dict[str, torch.Tensor]] = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Initialize EMA model copy."""
        # Only initialize if not already initialized (e.g. from checkpoint)
        if self.ema_model is None:
            self._init_ema_model(pl_module)

    def _init_ema_model(self, pl_module: L.LightningModule) -> None:
        """Create a copy of the model for EMA tracking."""
        with torch.no_grad():
            # Deepcopy the module
            self.ema_model = copy.deepcopy(pl_module)
            
            # Ensure EMA model is non-trainable and (optionally) on separate device
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            
            if self.ema_device:
                self.ema_model.to(self.ema_device)
            else:
                # If no specific device, keep it on same device (or cpu if preferred?)
                # Usually keeping on same device avoids transfer overhead, 
                # but might consume VRAM. Config default was CPU or None.
                pass

    @torch.no_grad()
    def on_train_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Update EMA weights."""
        if self.ema_model is None:
            return

        if (trainer.global_step % self.update_every_n_steps) == 0:
            self._update_weights(pl_module)

    @torch.no_grad()
    def _update_weights(self, pl_module: L.LightningModule) -> None:
        """Perform the EMA update: ema = decay * ema + (1 - decay) * new."""
        # Handle device mismatch if ema_device is set
        target_device = next(self.ema_model.parameters()).device
        
        msd = pl_module.state_dict()
        esd = self.ema_model.state_dict()
        
        for k, ema_v in esd.items():
            if k not in msd:
                continue
            
            model_v = msd[k].detach()
            
            if self.ema_device and model_v.device != target_device:
                model_v = model_v.to(target_device)
                
            if torch.is_floating_point(ema_v):
                ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)
            else:
                # For non-float (buffers like LongTensor), just copy
                ema_v.copy_(model_v)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Swap model weights with EMA weights for validation."""
        if self.ema_model is not None:
            self._swap_weights(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Restore original model weights after validation."""
        if self.ema_model is not None:
            self._restore_weights(pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Swap model weights with EMA weights for testing."""
        if self.ema_model is not None:
            self._swap_weights(pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Restore original model weights after testing."""
        if self.ema_model is not None:
            self._restore_weights(pl_module)

    @torch.no_grad()
    def _swap_weights(self, pl_module: L.LightningModule) -> None:
        """Store current weights and load EMA weights into pl_module."""
        self._backup_params = {
            k: v.detach().clone() for k, v in pl_module.state_dict().items()
        }
        
        # Load EMA state into pl_module
        # Note: strict=False to allow for slight mismatches if any (though shouldn't happen with deepcopy)
        pl_module.load_state_dict(self.ema_model.state_dict(), strict=False)

    @torch.no_grad()
    def _restore_weights(self, pl_module: L.LightningModule) -> None:
        """Restore backed-up weights into pl_module."""
        if self._backup_params is None:
            return
        pl_module.load_state_dict(self._backup_params, strict=False)
        self._backup_params = None

    def state_dict(self) -> Dict[str, Any]:
        """Save EMA state to checkpoint."""
        return {
            "ema_model_state_dict": self.ema_model.state_dict() if self.ema_model else None,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict.get("decay", self.decay)
        if state_dict.get("ema_model_state_dict"):
            # We need to reconstruct the structure first if not present
            # But usually load_state_dict is called after model init, so we can just load?
            # Actually, we can't instantiate self.ema_model here because we don't have pl_module.
            # So we temporarily store the state dict and load it in on_fit_start or on_load_checkpoint?
            # Wait, Lightning calls on_load_checkpoint(trainer, pl_module, checkpoint).
            # The Callback.load_state_dict is for the callback's state.
            
            # Let's store it and load it when we have the model.
            self._saved_ema_state_dict = state_dict["ema_model_state_dict"]

    def on_load_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Called when resuming from checkpoint."""
        # The callback state is loaded via load_state_dict automatically by Lightning if 
        # the callback is in the list.
        # However, if we rely on `self._saved_ema_state_dict`, we can use it here.
        if hasattr(self, "_saved_ema_state_dict") and self._saved_ema_state_dict is not None:
            if self.ema_model is None:
                self._init_ema_model(pl_module)
            self.ema_model.load_state_dict(self._saved_ema_state_dict)
            del self._saved_ema_state_dict
