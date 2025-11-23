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
from typing import Dict, List, Optional

import hydra
import lightning.pytorch as L
import torch
from hydra.utils import instantiate as hydra_instantiate, to_absolute_path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from utils.logging_utils import setup_rich_logging
from itertools import compress

# Ensure src is in path for module resolution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

torch.set_float32_matmul_precision("high")


def setup_callbacks(cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from Hydra config."""
    callbacks: List[Callback] = []
    if cfg.get("callbacks"):
        for _, cb_conf in cfg.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                callbacks.append(hydra_instantiate(cb_conf))
    return callbacks


def setup_logger(cfg: DictConfig) -> Optional[Logger]:
    """Instantiate logger from Hydra config (if configured)."""
    return hydra_instantiate(cfg.logger) if cfg.get("logger") else None


def validate_core_config(cfg: DictConfig) -> None:
    """Ensure required config groups are present."""
    required_groups = ("datamodule", "model", "trainer")
    missing = [group for group in required_groups if OmegaConf.select(cfg, group) is None]
    if missing:
        raise ValueError(f"Missing required config group(s): {', '.join(missing)}")


def apply_seed(cfg: DictConfig, log: logging.Logger) -> None:
    if seed := cfg.get("seed"):
        seed_everything(int(seed), workers=True)
        log.info(f"[bold #33FFAA]Seed set to[/]: {seed}")


def instantiate_datamodule(cfg: DictConfig, log: logging.Logger) -> L.LightningDataModule:
    """Instantiate and set up the configured datamodule."""
    if not hasattr(cfg, "datamodule"):
        raise ValueError("Config must define a 'datamodule' group")

    # Use Hydra instantiation directly
    # This requires the config to have a _target_ field and correct parameters
    datamodule = hydra_instantiate(cfg.datamodule, _recursive_=False)

    log.info(f"Datamodule: [bold]{datamodule.__class__.__name__}[/]")

    datamodule.setup(stage="fit")
    
    if not hasattr(datamodule, "get_graph_constants"):
        raise RuntimeError("Datamodule must implement get_graph_constants() for training.")
    return datamodule


def extract_graph_constants(datamodule: L.LightningDataModule, log: logging.Logger) -> Dict[str, torch.Tensor]:
    """Fetch graph constants from datamodule and print a short summary."""
    graph_consts = datamodule.get_graph_constants()
    log.info(f"Graph constants: [bold #55FF55]N={int(graph_consts['finger_ids'].shape[0])}[/], "
             f"[bold #55FF55]E={int(graph_consts['edge_index'].shape[1])}[/]")
    return graph_consts


def ensure_output_dir(cfg: DictConfig, log: logging.Logger) -> None:
    if cfg.get("paths", {}).get("output_dir"):
        output_dir = Path(to_absolute_path(str(cfg.paths.output_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[bold #33AAFF]Output directory[/]: {output_dir}")


def maybe_save_final_config(cfg: DictConfig, log: logging.Logger) -> None:
    if cfg.get("paths", {}).get("output_dir"):
        config_path = Path(cfg.paths.output_dir) / "config.yaml"
        OmegaConf.save(config=cfg, f=str(config_path))
        log.info(f"Configuration saved to: [bold]{config_path}[/]")


def log_trainer_summary(trainer: Trainer, cfg: DictConfig, log: logging.Logger) -> None:
    log.info("Trainer configuration:")
    log.info(f" Accelerator: [bold]{trainer.accelerator}[/]")
    log.info(f" Devices: [bold]{cfg.trainer.devices}[/]")
    log.info(f" Precision: [bold]{cfg.trainer.precision}[/]")
    log.info(f" Max epochs: [bold]{cfg.trainer.max_epochs}[/]")


def log_best_checkpoint(trainer: Trainer, log: logging.Logger) -> None:
    if (cb := getattr(trainer, "checkpoint_callback", None)) and hasattr(cb, "best_model_path"):
        if best_path := cb.best_model_path:
            log.info(f"Best model checkpoint: [bold]{best_path}[/]")


def find_last_checkpoint(cfg: DictConfig, log: logging.Logger) -> Optional[str]:
    """Attempt to find the last checkpoint in the experiment directory to resume training."""
    if not cfg.get("auto_resume", False):
        return None
        
    current_output_dir = Path(to_absolute_path(str(cfg.paths.output_dir)))
    experiment_dir = current_output_dir.parent
    
    if not experiment_dir.exists():
        log.warning(f"Auto-resume: Experiment directory {experiment_dir} does not exist.")
        return None

    # Find latest run (excluding current output dir)
    runs = sorted(
        [d for d in experiment_dir.iterdir() if d.is_dir() and d != current_output_dir], 
        key=lambda x: x.name
    )
    
    if runs:
        last_ckpt = runs[-1] / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            log.info(f"[bold #55FF55]Auto-resume: Found last checkpoint at {last_ckpt}[/]")
            return str(last_ckpt)
        log.info(f"Auto-resume: No 'last.ckpt' found in latest run {runs[-1]}")
    else:
        log.info("Auto-resume: No previous runs found.")
    
    return None


def run_auto_tuner(trainer: Trainer, model: L.LightningModule, datamodule: L.LightningDataModule, cfg: DictConfig, log: logging.Logger) -> None:
    """Run auto-scaling for batch size and learning rate if configured."""
    if not (cfg.get("auto_scale_batch_size", False) or cfg.get("auto_lr_find", False)):
        return

    log.info("=" * 80)
    log.info("[bold italic #FFAA33]Running Tuner...[/]")
    log.info("=" * 80)
    tuner = Tuner(trainer)
    
    if cfg.get("auto_scale_batch_size", False):
        log.info("Tuning: Finding max batch size...")
        tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
        tuned_bs = int(getattr(datamodule, "batch_size", 0))
        if "datamodule" in cfg:
            try:
                cfg.datamodule.batch_size = tuned_bs
            except Exception:
                pass
        log.info(f"Tuning: Batch size set to [bold]{tuned_bs}[/]")

    if cfg.get("auto_lr_find", False):
        log.info("Tuning: Finding optimal learning rate...")
        lr_finder = tuner.lr_find(model, datamodule=datamodule, update_attr=False)
        if lr_finder is not None:
            new_lr = lr_finder.suggestion()
            cfg.optimizer.lr = new_lr 
            if hasattr(model, "optimizer_cfg"):
                model.optimizer_cfg["lr"] = new_lr
            log.info(f"Tuning: Learning rate set to [bold]{new_lr}[/]")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the Flow Matching Hand DiT model using Hydra configuration."""
    setup_rich_logging()
    console = Console()
    log = logging.getLogger("train_flow_matching")
    validate_core_config(cfg)
    
    # log.info(f"{'=' * 80}\n[bold italic #FF33AA]Training Configuration:[/]\n{'=' * 80}")
    console.print("")
    console.rule("🌑 [bold italic #FF33AA] Training Configuration:[/]")
    console.print("")
    log.info(OmegaConf.to_yaml(cfg))
    
    apply_seed(cfg, log)
    ensure_output_dir(cfg, log)
    
    # Initialize DataModule
    console.print("")
    console.rule("😈 [bold italic #33FF00] Initializing DataModule[/]")
    console.print("")
    datamodule = instantiate_datamodule(cfg, log)
    graph_consts = extract_graph_constants(datamodule, log)
    
    # Initialize Model
    console.print("")
    console.rule("🔮 [bold italic #FF33CC] Initializing Model[/]")
    console.print("")
    if not hasattr(cfg.get("model", {}), "_target_"):
        raise ValueError("cfg.model must specify a Hydra '_target_'")

    model = hydra_instantiate(cfg.model, graph_consts=graph_consts, _recursive_=False)
    # run_meta = {}
    # if cfg.get("experiment_name"):
    #     run_meta["experiment_name"] = cfg.experiment_name
    # backbone_cfg = OmegaConf.select(cfg, "backbone")
    # if backbone_cfg is not None:
    #     name = getattr(backbone_cfg, "name", None)
    #     if name is not None:
    #         run_meta["backbone_name"] = name
    # if hasattr(model, "velocity_mode"):
    #     run_meta["velocity_mode"] = model.velocity_mode
    # model.run_meta = run_meta

    model.run_meta = {
        k: v for k, v in {
            "experiment_name": cfg.get("experiment_name"),
            "backbone_name": getattr(OmegaConf.select(cfg, "backbone"), "name", None),
            "velocity_mode": getattr(model, "velocity_mode", None)
        }.items() if v is not None
    }
    # Compile model if configured
    if cfg.get("compile", False):
        if hasattr(torch, "compile"):
            log.info("[bold italic #33FFAA]Compiling model.dit with torch.compile(dynamic=True)...[/]")
            try:
                model.dit = torch.compile(model.dit, dynamic=True)
            except TypeError:
                model.dit = torch.compile(model.dit)
        else:
            log.warning("cfg.compile=True but torch.compile not supported. Skipping compilation.")
            
    # Log Model Stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: [bold #33AAFF]{total_params:,}[/]")
    log.info(f"Trainable parameters: [bold #33AAFF]{trainable_params:,}[/]")
    
    # Callbacks & Logger
    console.print("")
    console.rule("🕸️ [bold italic #9900CC] Setting up Callbacks & Logger[/]")
    # console.print("")
    callbacks = setup_callbacks(cfg)
    for cb in callbacks:
        log.info(f" [bold]{cb.__class__.__name__}[/]")
        
    logger = setup_logger(cfg)
    if logger:
        log.info(f" [bold]{logger.__class__.__name__}[/]")
    else:
        log.warning("No logger configured")
    
    # Trainer
    console.print("")
    console.rule("🦄 [bold italic #3300FF] Initializing Trainer[/]")
    console.print("")
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

    # Configure Profiler
    if trainer_kwargs.get("profiler") == "advanced":
        from lightning.pytorch.profilers import AdvancedProfiler
        log.info(f"[bold #FFAA33]Enabling AdvancedProfiler with file output...[/]")
        trainer_kwargs["profiler"] = AdvancedProfiler(
            dirpath=to_absolute_path(str(cfg.paths.output_dir)), 
            filename="profile_report"
        )

    # Manual GradNormLogger attachment removed. 
    # Use 'callbacks: grad_norm' in config if needed.

    trainer = Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)
    log_trainer_summary(trainer, cfg, log)
    
    # Auto-Tuning
    run_auto_tuner(trainer, model, datamodule, cfg, log)

    # Start Training
    ckpt_path = cfg.get("ckpt_path")
    if str(ckpt_path).lower() == "none": 
        ckpt_path = None
    
    if ckpt_path is None:
        ckpt_path = find_last_checkpoint(cfg, log)
    console.print("")
    console.rule("🖤 [bold italic #CC9900] Starting Training[/]")
    console.print("")
    
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    console.print("")
    console.rule("🗡️ [bold italic #EEDD00] Training Completed![/]")
    console.print("")
    log_best_checkpoint(trainer, log)
    maybe_save_final_config(cfg, log)


if __name__ == "__main__":
    main()
