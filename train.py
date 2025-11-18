#!/usr/bin/env python
"""Training script for Flow Matching Hand DiT model.

Uses Hydra + PyTorch Lightning to train src.models.flow_matching_hand_dit.HandFlowMatchingDiT.
Config entry point: configs/config.yaml

Example usage:
  python train.py model=flow_matching_hand_dit backbone=ptv3_sparse datamodule=handencoder_dm_dex
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
from hydra.utils import instantiate as hydra_instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
import torch
import torch.nn as nn
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Path & precision setup
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datamodules.HandEncoderDataModule import HandEncoderDataModule
from models.flow_matching_hand_dit import HandFlowMatchingDiT
from models.backbone import build_backbone
from models.components.hand_encoder import build_hand_encoder
from models.components.graph_dit import build_dit

torch.set_float32_matmul_precision("high")

# Logging utilities
def setup_rich_logging(level: int = logging.INFO) -> None:
    """Configure rich logging for cleaner console output."""
    log_format = (
        "[%(asctime)s] {%(filename)s:%(lineno)d} "
        "[%(funcName)s] <%(levelname)s> - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    install_rich_traceback(show_locals=False)
    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=False,
        show_level=False,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for noisy in ["matplotlib", "numba", "PIL", "urllib3", "asyncio", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_callbacks(cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from Hydra config."""
    callbacks: List[Callback] = []
    if "callbacks" not in cfg:
        return callbacks
    callbacks_cfg = cfg.callbacks
    if "model_checkpoint" in callbacks_cfg and callbacks_cfg.model_checkpoint is not None:
        callbacks.append(hydra_instantiate(callbacks_cfg.model_checkpoint))
    if "early_stopping" in callbacks_cfg and callbacks_cfg.early_stopping is not None:
        callbacks.append(hydra_instantiate(callbacks_cfg.early_stopping))
    if "lr_monitor" in callbacks_cfg and callbacks_cfg.lr_monitor is not None:
        callbacks.append(hydra_instantiate(callbacks_cfg.lr_monitor))
    if "model_summary" in callbacks_cfg and callbacks_cfg.model_summary is not None:
        callbacks.append(hydra_instantiate(callbacks_cfg.model_summary))
    if "rich_progress_bar" in callbacks_cfg and callbacks_cfg.rich_progress_bar is not None:
        callbacks.append(hydra_instantiate(callbacks_cfg.rich_progress_bar))
    return callbacks


def setup_logger(cfg: DictConfig) -> Optional[Logger]:
    """Instantiate logger from Hydra config (if configured)."""
    if "logger" not in cfg or cfg.logger is None:
        return None
    return hydra_instantiate(cfg.logger)


def validate_core_config(cfg: DictConfig) -> None:
    required_groups = ("datamodule", "model", "backbone", "hand_encoder", "dit", "trainer")
    missing = [group for group in required_groups if OmegaConf.select(cfg, group) is None]
    if missing:
        raise ValueError(f"Missing required config group(s): {', '.join(missing)}")


def apply_seed(cfg: DictConfig, log: logging.Logger) -> None:
    seed = cfg.get("seed", None)
    if seed is None:
        return
    seed_everything(int(seed), workers=True)
    log.info(f"[bold #33FFAA]Seed set to[/]: {seed}")

# Hydra instantiation helpers
def instantiate_datamodule(cfg: DictConfig, log: logging.Logger) -> pl.LightningDataModule:
    """Instantiate and set up the configured datamodule."""
    if not hasattr(cfg, "datamodule"):
        raise ValueError("Config must define a 'datamodule' group")
    datamodule: Optional[pl.LightningDataModule] = None
    if "_target_" in cfg.datamodule:
        datamodule = hydra_instantiate(cfg.datamodule, _recursive_=False)
    else:
        datamodule = HandEncoderDataModule(data_cfg=cfg.datamodule)
    log.info(f"Datamodule: [bold]{datamodule.__class__.__name__}[/]")
    datamodule.setup(stage="fit")
    if not hasattr(datamodule, "get_graph_constants"):
        raise RuntimeError("Datamodule must implement get_graph_constants() for flow-matching training.")
    return datamodule


def extract_graph_constants(datamodule: pl.LightningDataModule, log: logging.Logger) -> Dict[str, torch.Tensor]:
    """Fetch graph constants from datamodule and print a short summary."""
    graph_consts = datamodule.get_graph_constants()
    N = int(graph_consts["finger_ids"].shape[0])
    E = int(graph_consts["edge_index"].shape[1])
    log.info(f"Graph constants: [bold #55FF55]N={N}[/], [bold #55FF55]E={E}[/]")
    return graph_consts


def build_backbone_module(backbone_cfg: DictConfig) -> nn.Module:
    if "_target_" in backbone_cfg:
        return hydra_instantiate(backbone_cfg, _recursive_=False)
    return build_backbone(backbone_cfg)


def build_hand_encoder_module(
    hand_encoder_cfg: DictConfig,
    graph_consts: Dict[str, torch.Tensor],
    out_dim: int,
) -> nn.Module:
    if "_target_" in hand_encoder_cfg:
        return hydra_instantiate(
            hand_encoder_cfg,
            graph_consts=graph_consts,
            out_dim=out_dim,
            _recursive_=False,
        )
    return build_hand_encoder(hand_encoder_cfg, graph_consts=graph_consts, out_dim=out_dim)


def build_dit_module(
    dit_cfg: DictConfig,
    graph_consts: Dict[str, torch.Tensor],
    dim: int,
) -> nn.Module:
    if "_target_" in dit_cfg:
        return hydra_instantiate(
            dit_cfg,
            graph_consts=graph_consts,
            dim=dim,
            _recursive_=False,
        )
    return build_dit(dit_cfg, graph_consts=graph_consts, dim=dim)


def build_model_modules(
    cfg: DictConfig, graph_consts: Dict[str, torch.Tensor]
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Build backbone / hand encoder / DiT modules from their respective config groups."""
    d_model = int(cfg.model.d_model)
    backbone = build_backbone_module(cfg.backbone)
    hand_encoder = build_hand_encoder_module(cfg.hand_encoder, graph_consts=graph_consts, out_dim=d_model)
    dit = build_dit_module(cfg.dit, graph_consts=graph_consts, dim=d_model)
    return backbone, hand_encoder, dit


def instantiate_model(
    cfg: DictConfig,
    graph_consts: Dict[str, torch.Tensor],
    backbone: nn.Module,
    hand_encoder: nn.Module,
    dit: nn.Module,
) -> nn.Module:
    """Instantiate the LightningModule, injecting sub-modules from builders."""
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "_target_"):
        raise ValueError("cfg.model must specify a Hydra '_target_'")
    if hasattr(cfg, "loss") and hasattr(cfg.loss, "lambda_tangent"):
        cfg.model.lambda_tangent = cfg.loss.lambda_tangent
    loss_cfg = cfg.loss if hasattr(cfg, "loss") else None
    optimizer_cfg = cfg.optimizer if hasattr(cfg, "optimizer") else None
    return hydra_instantiate(
        cfg.model,
        graph_consts=graph_consts,
        backbone=backbone,
        hand_encoder=hand_encoder,
        dit=dit,
        loss_cfg=loss_cfg,
        optimizer_cfg=optimizer_cfg,
        _recursive_=False,
    )


def ensure_output_dir(cfg: DictConfig, log: logging.Logger) -> None:
    if "paths" in cfg and "output_dir" in cfg.paths:
        output_dir = Path(to_absolute_path(str(cfg.paths.output_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[bold #33AAFF]Output directory[/]: {output_dir}")


def maybe_save_final_config(cfg: DictConfig, log: logging.Logger) -> None:
    if "paths" not in cfg or "output_dir" not in cfg.paths:
        return
    config_path = Path(cfg.paths.output_dir) / "config.yaml"
    OmegaConf.save(config=cfg, f=str(config_path))
    log.info(f"Configuration saved to: [bold]{config_path}[/]")


def log_trainer_summary(trainer: Trainer, cfg: DictConfig, log: logging.Logger) -> None:
    log.info("Trainer configuration:")
    log.info(f"  - Accelerator: [bold]{trainer.accelerator}[/]")
    log.info(f"  - Devices: [bold]{cfg.trainer.devices}[/]")
    log.info(f"  - Precision: [bold]{cfg.trainer.precision}[/]")
    log.info(f"  - Max epochs: [bold]{cfg.trainer.max_epochs}[/]")


def log_best_checkpoint(trainer: Trainer, log: logging.Logger) -> None:
    if getattr(trainer, "checkpoint_callback", None) is None:
        return
    if not hasattr(trainer.checkpoint_callback, "best_model_path"):
        return
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        log.info(f"Best model checkpoint: [bold]{best_path}[/]")

# Main training entry
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the Flow Matching Hand DiT model using Hydra configuration."""
    setup_rich_logging()
    log = logging.getLogger("train_flow_matching")
    validate_core_config(cfg)
    log.info("=" * 80)
    log.info("[bold italic #FF33AA]Training Configuration:[/]")
    log.info("=" * 80)
    log.info(OmegaConf.to_yaml(cfg))
    log.info("=" * 80)
    apply_seed(cfg, log)
    ensure_output_dir(cfg, log)
    
    # Initialize DataModule
    log.info("=" * 80)
    log.info("[bold italic #FFD700]Initializing DataModule...[/]")
    log.info("=" * 80)
    datamodule = instantiate_datamodule(cfg, log)
    graph_consts = extract_graph_constants(datamodule, log)
    
    # Initialize Flow Matching model and its sub-modules
    log.info("=" * 80)
    log.info("[bold italic #FFD700]Initializing Flow Matching Hand DiT model...[/]")
    log.info("=" * 80)
    backbone, hand_encoder, dit = build_model_modules(cfg, graph_consts)
    model = instantiate_model(cfg, graph_consts, backbone, hand_encoder, dit)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: [bold #33AAFF]{total_params:,}[/]")
    log.info(f"Trainable parameters: [bold #33AAFF]{trainable_params:,}[/]")
    
    # Callbacks & Logger
    log.info("=" * 80)
    log.info("[bold italic #FFD700]Setting up Callbacks & Logger...[/]")
    log.info("=" * 80)
    callbacks = setup_callbacks(cfg)
    for cb in callbacks:
        log.info(f"  - [bold]{cb.__class__.__name__}[/]")
    logger = setup_logger(cfg)
    if logger is not None:
        log.info(f"Logger: [bold]{logger.__class__.__name__}[/]")
    else:
        log.warning("No logger configured")
    
    # Trainer
    log.info("=" * 80)
    log.info("[bold italic #FFD700]Initializing Trainer...[/]")
    log.info("=" * 80)
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    log_trainer_summary(trainer, cfg, log)
    
    # Start training
    log.info("=" * 80)
    log.info("[bold italic #FF33AA]Starting Training...[/]")
    log.info("=" * 80)
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and str(ckpt_path).lower() == "none":
        ckpt_path = None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info("=" * 80)
    log.info("[bold italic #33FFAA]Training Completed![/]")
    log.info("=" * 80)
    log_best_checkpoint(trainer, log)
    maybe_save_final_config(cfg, log)


if __name__ == "__main__":
    main()
