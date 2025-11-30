#!/usr/bin/env python
"""Training script for Flow Matching Hand DiT model.

Uses Hydra + PyTorch Lightning for training.
Config entry point: configs/config.yaml

Usage:
    python train.py model=flow_matching_hand_dit backbone=ptv3_sparse datamodule=handencoder_dm_dex
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if (src := str(PROJECT_ROOT / "src")) not in sys.path:
    sys.path.insert(0, src)

import atexit
import logging
import os
import signal
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import lightning.pytorch as L
import torch
import torch.distributed as dist
from hydra.utils import instantiate as hydra_instantiate, to_absolute_path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from utils.logging_utils import setup_rich_logging

torch.set_float32_matmul_precision("high")


class RankZeroLogger:
    """Logger wrapper that only emits on rank zero."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @rank_zero_only
    def info(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)

    @rank_zero_only
    def warning(self, *args, **kwargs):
        return self._logger.warning(*args, **kwargs)

    @rank_zero_only
    def error(self, *args, **kwargs):
        return self._logger.error(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._logger, name)


class RankZeroConsole:
    """Console wrapper that only prints on rank zero."""

    def __init__(self, console: Console) -> None:
        self._console = console

    @rank_zero_only
    def print(self, *args, **kwargs):
        return self._console.print(*args, **kwargs)

    @rank_zero_only
    def rule(self, *args, **kwargs):
        return self._console.rule(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._console, name)


class GracefulExitManager:
    """Handles graceful shutdown on SIGINT/SIGTERM with emergency checkpoint saving."""

    _instance: Optional["GracefulExitManager"] = None

    def __init__(self) -> None:
        self._trainer: Optional[Trainer] = None
        self._cfg: Optional[DictConfig] = None
        self._log: Optional[logging.Logger] = None
        self._console: Optional[RankZeroConsole] = None
        self._interrupted = False
        self._original_handlers: Dict[int, Any] = {}

    @classmethod
    def instance(cls) -> "GracefulExitManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        trainer: Trainer,
        cfg: DictConfig,
        log: logging.Logger,
        console: RankZeroConsole,
    ) -> None:
        self._trainer = trainer
        self._cfg = cfg
        self._log = log
        self._console = console

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._handle)
        atexit.register(self._restore_handlers)

    def _handle(self, signum: int, frame: Any) -> None:
        if self._interrupted:
            self._log and self._log.warning("Force exit requested.")
            self._cleanup_distributed()
            sys.exit(1)

        self._interrupted = True
        sig_name = signal.Signals(signum).name

        if self._console:
            self._console.print("")
            self._console.rule(f"[bold red]{sig_name} received - graceful shutdown[/]")

        self._log and self._log.warning(f"Interrupted by {sig_name}, saving checkpoint...")
        self._trainer and setattr(self._trainer, "should_stop", True)
        self._save_emergency()
        self._cleanup_distributed()

    def _cleanup_distributed(self) -> None:
        """Clean up distributed training resources to prevent zombie processes."""
        try:
            if dist.is_initialized():
                self._log and self._log.info("Destroying distributed process group...")
                dist.destroy_process_group()
        except Exception as e:
            self._log and self._log.debug(f"Error cleaning up distributed: {e}")

    def _save_emergency(self, suffix: str = "interrupt") -> None:
        if not (self._trainer and self._cfg):
            return
        try:
            if output_dir := self._cfg.get("paths", {}).get("output_dir"):
                ckpt_dir = Path(to_absolute_path(str(output_dir))) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                path = ckpt_dir / f"emergency_{suffix}.ckpt"
                self._trainer.save_checkpoint(str(path))
                self._log and self._log.info(f"Emergency checkpoint: {path}")
        except Exception as e:
            self._log and self._log.error(f"Failed to save emergency checkpoint: {e}")

    def _restore_handlers(self) -> None:
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def handle_exception(self) -> None:
        self._save_emergency(suffix="exception")


def instantiate_callbacks(
    cfg: Union[DictConfig, Dict, List, None],
    source: str = "callbacks",
    log: Optional[logging.Logger] = None,
) -> List[Callback]:
    """Instantiate callbacks from config, filtering invalid entries."""
    if cfg is None:
        return []

    log = log or logging.getLogger(__name__)
    items = cfg.items() if isinstance(cfg, (DictConfig, dict)) else enumerate(cfg)
    callbacks = []

    for name, conf in items:
        if conf is None or not isinstance(conf, (DictConfig, dict)) or "_target_" not in conf:
            continue
        try:
            cb = hydra_instantiate(conf)
            if isinstance(cb, Callback):
                callbacks.append(cb)
            else:
                log.warning(f"Non-Callback in {source}.{name}: {type(cb).__name__}")
        except Exception as e:
            log.warning(f"Failed to instantiate {source}.{name}: {e}")

    return callbacks


def validate_config(cfg: DictConfig) -> None:
    """Ensure required config groups exist."""
    missing = [g for g in ("datamodule", "model", "trainer") if not OmegaConf.select(cfg, g)]
    if missing:
        raise ValueError(f"Missing required config: {', '.join(missing)}")


def find_last_checkpoint(cfg: DictConfig, log: logging.Logger) -> Optional[str]:
    """Find the latest checkpoint for auto-resume."""
    if not cfg.get("auto_resume"):
        return None

    output_dir = Path(to_absolute_path(str(cfg.paths.output_dir)))
    exp_dir = output_dir.parent

    if not exp_dir.exists():
        log.warning(f"Experiment directory not found: {exp_dir}")
        return None

    runs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d != output_dir], key=lambda x: x.name)
    if runs and (ckpt := runs[-1] / "checkpoints" / "last.ckpt").exists():
        log.info(f"[bold #55FF55]Auto-resume: {ckpt}[/]")
        return str(ckpt)

    log.info("Auto-resume: no previous checkpoint found")
    return None


def validate_checkpoint(path: str, log: logging.Logger, load: bool = True) -> Tuple[bool, Optional[int]]:
    """Validate checkpoint integrity before resuming."""
    if not path:
        return True, None

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {path}")
        return False, None

    size = ckpt_path.stat().st_size
    if size == 0:
        log.error(f"Checkpoint is empty: {path}")
        return False, None

    log.info(f"Checkpoint size: {size / 1024 / 1024:.2f} MB")

    if not load:
        return True, None

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", 0)
        step = ckpt.get("global_step", 0)
        log.info(f"Checkpoint valid: epoch={epoch}, step={step}")

        # Sample check for corruption
        if state := ckpt.get("state_dict"):
            for key in list(state.keys())[:10]:
                if isinstance(t := state[key], torch.Tensor):
                    if torch.isnan(t).any():
                        log.warning(f"NaN in checkpoint: {key}")
                    if torch.isinf(t).any():
                        log.warning(f"Inf in checkpoint: {key}")

        return True, epoch
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        return False, None


def resolve_resume_run_dir(cfg: DictConfig, log: logging.Logger) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve resume_run_dir to (run_dir, ckpt_path). Supports absolute/relative paths."""
    if not (resume_dir := cfg.get("resume_run_dir")):
        return None, None

    run_dir = Path(resume_dir) if Path(resume_dir).is_absolute() else PROJECT_ROOT / resume_dir
    if not run_dir.is_dir():
        log.warning(f"resume_run_dir invalid: {resume_dir}")
        return None, None

    ckpt_path = next(
        (str(run_dir / "checkpoints" / n) for n in ("last.ckpt", "last-v1.ckpt", "last-v2.ckpt")
         if (run_dir / "checkpoints" / n).exists()),
        None,
    )
    if not ckpt_path:
        log.warning(f"No checkpoint in {run_dir / 'checkpoints'}")

    return run_dir, ckpt_path


def resolve_resume_config(cfg: DictConfig, log: logging.Logger) -> DictConfig:
    """Merge previous run config. Priority: CLI > resume config > defaults."""
    from hydra.core.hydra_config import HydraConfig

    run_dir, auto_ckpt = resolve_resume_run_dir(cfg, log)
    config_path = None

    # Find config: resume_run_dir > resume_config > auto_resume_config
    if run_dir and (cand := run_dir / ".hydra" / "config.yaml").exists():
        config_path = cand
        log.info(f"[bold #55FF55]Resume run dir[/]: {run_dir}")
    elif (p := cfg.get("resume_config")) and (cand := Path(to_absolute_path(str(p)))).is_file():
        config_path = cand
    elif cfg.get("auto_resume_config"):
        fallback = None
        if (ckpt := cfg.get("ckpt_path")) and (p := Path(to_absolute_path(str(ckpt)))).exists():
            fallback = p.parent.parent
        elif last := find_last_checkpoint(cfg, log):
            fallback = Path(last).parent.parent
        elif (out := cfg.get("paths", {}).get("output_dir")):
            exp_dir = Path(to_absolute_path(str(out))).parent
            if exp_dir.exists():
                runs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
                fallback = runs[-1] if runs else None
        if fallback and (cand := fallback / ".hydra" / "config.yaml").exists():
            config_path = cand

    # Merge previous config
    if config_path:
        try:
            prev = OmegaConf.load(str(config_path))
        except Exception as e:
            log.warning(f"Failed to load resume config: {e}")
            return cfg

        cli_keys = set()
        try:
            for ov in HydraConfig.get().overrides.task:
                if "=" in ov and (k := ov.split("=", 1)[0]) != "resume_run_dir":
                    cli_keys.add(k)
            log.info(f"CLI overrides: {list(cli_keys)}")
        except Exception:
            pass

        merged = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        for key, value in OmegaConf.to_container(prev, resolve=False).items():
            if key in ("paths", "hydra") or key in cli_keys:
                continue
            nested = [k for k in cli_keys if k.startswith(f"{key}.")]
            OmegaConf.update(merged, key, value, merge=True)
            for nk in nested:
                if (nv := OmegaConf.select(cfg, nk)) is not None:
                    OmegaConf.update(merged, nk, nv, merge=False)

        log.info(f"Applied resume config: [bold]{config_path}[/]")
        cfg = merged

    # Auto-set ckpt_path and handle resume_in_place
    if auto_ckpt and not cfg.get("ckpt_path"):
        cfg = OmegaConf.merge(cfg, {"ckpt_path": auto_ckpt})
        log.info(f"[bold #55FF55]Auto checkpoint[/]: {auto_ckpt}")

    if run_dir and cfg.get("resume_in_place", False):
        OmegaConf.update(cfg, "paths.output_dir", str(run_dir), merge=False)
        log.info(f"[bold #FFAA33]Resume in place[/]: {run_dir}")

    return cfg


def build_trainer(cfg: DictConfig, callbacks: List[Callback], logger: Optional[Logger], log: logging.Logger) -> Trainer:
    """Build Trainer with resolved config."""
    kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

    # Handle trainer.callbacks
    trainer_cbs = instantiate_callbacks(kwargs.pop("callbacks", None), "trainer.callbacks", log)
    for cb in trainer_cbs:
        log.info(f" [bold]{cb.__class__.__name__}[/]")

    # Handle strategy
    if isinstance(strategy := kwargs.get("strategy"), (DictConfig, dict)) and "_target_" in strategy:
        kwargs["strategy"] = hydra_instantiate(strategy)

    # Handle profiler
    if isinstance(prof := kwargs.get("profiler"), (DictConfig, dict)) and "_target_" in prof:
        kwargs["profiler"] = hydra_instantiate(prof)
    elif prof == "advanced":
        from lightning.pytorch.profilers import AdvancedProfiler
        log.info("[bold #FFAA33]AdvancedProfiler enabled[/]")
        kwargs["profiler"] = AdvancedProfiler(
            dirpath=to_absolute_path(str(cfg.paths.output_dir)),
            filename="profile_report",
        )

    return Trainer(**kwargs, callbacks=callbacks + trainer_cbs, logger=logger)


def run_tuner(trainer: Trainer, model: L.LightningModule, dm: L.LightningDataModule, cfg: DictConfig, log: logging.Logger) -> None:
    """Run auto batch size / LR tuning if configured."""
    if not (cfg.get("auto_scale_batch_size") or cfg.get("auto_lr_find")):
        return

    log.info("[bold italic #FFAA33]Running Tuner...[/]")
    tuner = Tuner(trainer)

    if cfg.get("auto_scale_batch_size"):
        tuner.scale_batch_size(model, datamodule=dm, mode="power")
        log.info(f"Batch size: [bold]{getattr(dm, 'batch_size', '?')}[/]")

    if cfg.get("auto_lr_find"):
        if result := tuner.lr_find(model, datamodule=dm, update_attr=False):
            lr = result.suggestion()
            cfg.optimizer.lr = lr
            if hasattr(model, "optimizer_cfg"):
                model.optimizer_cfg["lr"] = lr
            log.info(f"Learning rate: [bold]{lr}[/]")


def section(console: RankZeroConsole, title: str) -> None:
    """Print a section header."""
    console.print("")
    console.rule(title)
    console.print("")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint."""
    setup_rich_logging()
    console = RankZeroConsole(Console())
    log = RankZeroLogger(logging.getLogger("train"))

    cfg = resolve_resume_config(cfg, log)
    validate_config(cfg)

    section(console, "🌑 [bold italic #FF33AA] Training Configuration[/]")
    log.info(OmegaConf.to_yaml(cfg))

    if seed := cfg.get("seed"):
        seed_everything(int(seed), workers=True)
        log.info(f"[bold #33FFAA]Seed[/]: {seed}")

    if output_dir := cfg.get("paths", {}).get("output_dir"):
        Path(to_absolute_path(str(output_dir))).mkdir(parents=True, exist_ok=True)
        log.info(f"[bold #33AAFF]Output[/]: {output_dir}")

    # DataModule
    section(console, "😈 [bold italic #33FF00] Initializing DataModule[/]")
    dm = hydra_instantiate(cfg.datamodule, _recursive_=False)
    log.info(f"[bold]{dm.__class__.__name__}[/]")
    dm.setup(stage="fit")

    if not hasattr(dm, "get_graph_constants"):
        raise RuntimeError("DataModule must implement get_graph_constants()")

    graph = dm.get_graph_constants()
    log.info(f"Graph: [bold #55FF55]N={graph['finger_ids'].shape[0]}, E={graph['edge_index'].shape[1]}[/]")

    # Model
    section(console, "🔮 [bold italic #FF33CC] Initializing Model[/]")
    if "_target_" not in cfg.get("model", {}):
        raise ValueError("cfg.model must have '_target_'")

    model = hydra_instantiate(cfg.model, graph_consts=graph, _recursive_=False)
    model.run_meta = {
        k: v for k, v in {
            "experiment_name": cfg.get("experiment_name"),
            "backbone_name": getattr(OmegaConf.select(cfg, "backbone"), "name", None),
            "velocity_mode": getattr(model, "velocity_mode", None),
        }.items() if v is not None
    }

    if cfg.get("compile") and hasattr(torch, "compile"):
        log.info("[bold italic #33FFAA]Compiling model.dit...[/]")
        try:
            model.dit = torch.compile(model.dit, dynamic=True)
        except TypeError:
            model.dit = torch.compile(model.dit)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: [bold #33AAFF]{total:,}[/] (trainable: {trainable:,})")

    # Callbacks & Logger
    section(console, "🕸️ [bold italic #9900CC] Setting up Callbacks & Logger[/]")
    callbacks = instantiate_callbacks(cfg.get("callbacks"), "callbacks", log)
    for cb in callbacks:
        log.info(f" [bold]{cb.__class__.__name__}[/]")

    logger = hydra_instantiate(cfg.logger) if cfg.get("logger") else None
    if logger:
        log.info(f" [bold]{logger.__class__.__name__}[/]")

    # Trainer
    section(console, "🦄 [bold italic #3300FF] Initializing Trainer[/]")
    trainer = build_trainer(cfg, callbacks, logger, log)
    log.info(f"Accelerator: [bold]{trainer.accelerator}[/], Devices: [bold]{cfg.trainer.devices}[/]")
    log.info(f"Precision: [bold]{cfg.trainer.precision}[/], Max epochs: [bold]{cfg.trainer.max_epochs}[/]")

    exit_mgr = GracefulExitManager.instance()
    exit_mgr.register(trainer, cfg, log, console)

    run_tuner(trainer, model, dm, cfg, log)

    # Checkpoint
    ckpt_path = cfg.get("ckpt_path")
    if str(ckpt_path).lower() == "none":
        ckpt_path = None
    ckpt_path = ckpt_path or find_last_checkpoint(cfg, log)

    if ckpt_path:
        section(console, "🔍 [bold italic #00AAFF] Checkpoint Validation[/]")
        valid, epoch = validate_checkpoint(ckpt_path, log, cfg.get("validate_checkpoint", True))
        if not valid:
            raise RuntimeError(f"Invalid checkpoint: {ckpt_path}")
        if epoch is not None:
            model.resume_epoch = epoch
            log.info(f"Resume epoch: {epoch}")

    # Train
    section(console, "🖤 [bold italic #CC9900] Starting Training[/]")
    try:
        trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)

        console.print("")
        if exit_mgr.interrupted:
            console.rule("[bold yellow]Interrupted - Checkpoint Saved[/]")
        else:
            console.rule("🗡️ [bold italic #EEDD00]Training Completed![/]")
        console.print("")

    except Exception as e:
        console.print("")
        console.rule("[bold red]Training Failed[/]")
        log.error(f"Error: {e}")
        exit_mgr.handle_exception()
        raise

    if cb := getattr(trainer, "checkpoint_callback", None):
        if best := getattr(cb, "best_model_path", None):
            log.info(f"Best checkpoint: [bold]{best}[/]")

    if output_dir := cfg.get("paths", {}).get("output_dir"):
        config_path = Path(output_dir) / "config.yaml"
        OmegaConf.save(cfg, str(config_path))
        log.info(f"Config saved: [bold]{config_path}[/]")


if __name__ == "__main__":
    main()
