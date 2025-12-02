import logging
import math
from pathlib import Path
from typing import Dict, Optional

from rich import box
from rich.align import Align
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback
from lightning.pytorch.utilities import rank_zero_only


class ColoredLevelFormatter(logging.Formatter):
    """Custom formatter that adds Rich color markup to log level names."""

    LEVEL_COLORS = {
        logging.DEBUG: "dim",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "bold red",
    }

    def format(self, record):
        original_levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelno, "white")
        record.levelname = f"[{color}]{original_levelname}[/{color}]"
        result = super().format(record)
        record.levelname = original_levelname
        return result


def setup_rich_logging(level: int = logging.INFO) -> None:
    """Configure rich logging for cleaner console output."""
    # Suppress logging on non-zero ranks by default
    logging.getLogger().setLevel(logging.ERROR)

    @rank_zero_only
    def _setup_on_rank_zero():
        log_format = (
            "[%(asctime)s] {%(filename)s:%(lineno)d} "
            "[%(funcName)s] %(levelname)s - %(message)s"
        )
        date_format = "%m/%d/%y %H:%M:%S"

        install_rich_traceback(show_locals=False)
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=False,
            show_level=False,
            show_path=False,
        )
        handler.setFormatter(ColoredLevelFormatter(log_format, datefmt=date_format))
        
        # Get current root logger handlers (typically includes Hydra's FileHandler)
        root_logger = logging.getLogger()
        existing_handlers = root_logger.handlers
        
        # Filter out existing stream handlers to prevent duplicates/conflict with RichHandler
        # but PRESERVE FileHandlers (which write to train.log)
        handlers_to_keep = [
            h for h in existing_handlers 
            if not isinstance(h, logging.StreamHandler) or isinstance(h, logging.FileHandler)
        ]
        
        # Configure logging with both the new RichHandler and preserved handlers
        logging.basicConfig(
            level=level, 
            handlers=[handler] + handlers_to_keep, 
            force=True
        )
        
        for noisy in ["matplotlib", "numba", "PIL", "urllib3", "asyncio", "botocore"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)
    
    _setup_on_rank_zero()


# Create Rich console with custom theme for validation / training summaries
_custom_theme = Theme({
    "metric": "bold cyan",
    "value": "bold grey50",
    "good": "bold green",
    "warn": "bold yellow",
    "err": "bold red",
    "breakdown": "bold magenta",
    "header": "bold blue",
})


class DualConsole:
    """Console wrapper: rank_zero_only + dual output (terminal + file)."""

    def __init__(self, theme: Theme) -> None:
        self._terminal = Console(theme=theme)
        self._file: Optional[Console] = None
        self._file_handle = None

    @rank_zero_only
    def setup_file(self, path: Path) -> None:
        """Setup file output. Call once at startup (rank 0 only)."""
        if self._file_handle:
            self._file_handle.close()
        self._file_handle = open(path, "w", encoding="utf-8")
        self._file = Console(
            file=self._file_handle, force_terminal=False, width=120, theme=_custom_theme
        )

    @rank_zero_only
    def print(self, *args, **kwargs) -> None:
        self._terminal.print(*args, **kwargs)
        if self._file:
            self._file.print(*args, **kwargs)

    @rank_zero_only
    def rule(self, *args, **kwargs) -> None:
        self._terminal.rule(*args, **kwargs)
        if self._file:
            self._file.rule(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._terminal, name)


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


console = DualConsole(theme=_custom_theme)


def get_logger(name: str) -> RankZeroLogger:
    """Get a RankZeroLogger for any module. Use for strict rank 0 only logging."""
    return RankZeroLogger(logging.getLogger(name))


def setup_logging(output_dir: Optional[Path] = None, logger_name: str = "train") -> RankZeroLogger:
    """One-time initialization for logging and console.

    Call this once in main() after output_dir is known:
        log = setup_logging(output_dir)

    This function:
    - Configures Rich logging (RichHandler for terminal, preserves Hydra's FileHandler for train.log)
    - Sets up console file output to output_dir/console.log (if output_dir provided)
    - Returns a RankZeroLogger instance

    Args:
        output_dir: Output directory path. If provided, console output is also saved to console.log.
        logger_name: Name for the logger (default: "train").

    Returns:
        RankZeroLogger instance for logging.
    """
    setup_rich_logging()
    if output_dir:
        console.setup_file(output_dir / "console.log")
    return RankZeroLogger(logging.getLogger(logger_name))


@rank_zero_only
def log_validation_summary(epoch: int,
                           num_batches: int,
                           avg_loss: float,
                           loss_std: float,
                           loss_min: float,
                           loss_max: float,
                           val_detailed_loss: Dict[str, float],
                           val_set_metrics: Dict[str, float] = None,
                           val_quality_metrics: Dict[str, float] = None,
                           val_diversity_metrics: Dict[str, float] = None,
                           val_flow_metrics: Dict[str, float] = None) -> None:
    """Pretty-print validation summary results with Rich console output only."""

    # Header (console pretty print)
    # console.print("") 
    # console.print("")  # blank line
    console.rule(f"[header]🚩 Epoch {epoch} - Validation Results[/header]", style="cyan")

    # --- Main Metrics Table ---
    main_table = Table(
        show_header=True,
        header_style="bold blue",
        box=box.ROUNDED,
        pad_edge=True,
        expand=False,
        border_style="cyan",
        title="Main Metrics",
    )
    main_table.add_column("📊 Metric", style="metric", no_wrap=True)
    main_table.add_column("✨ Value", style="value", justify="right")

    main_table.add_row("Total validation batches", f"[good]{num_batches}[/good]")
    main_table.add_row("Average Loss", f"[good]{avg_loss:.4f}[/good]")
    main_table.add_row("Loss Std", f"[value]{loss_std:.4f}[/value]")
    main_table.add_row("Loss Min", f"[value]{loss_min:.4f}[/value]")
    main_table.add_row("Loss Max", f"[warn]{loss_max:.4f}[/warn]")

    console.print(Align.center(main_table))

    # --- Detailed Loss Breakdown ---
    if val_detailed_loss:
        console.print("[breakdown]📉 Detailed Loss Breakdown[/breakdown]")

        breakdown_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.HORIZONTALS,
            pad_edge=True,
            expand=False,
            border_style="magenta",
            title="Detailed Loss",
        )
        breakdown_table.add_column("🔹 Component", style="breakdown", no_wrap=True)
        breakdown_table.add_column("Value", style="value", justify="right")

        for k, v in val_detailed_loss.items():
            try:
                key_norm = k.replace(" ", "_").lower()
                # 合并展示：Rotation_Geodesic (rad / deg)
                if key_norm == "rotation_geodesic":
                    rad_val = float(v)
                    deg_val = rad_val * 180.0 / math.pi if math.isfinite(rad_val) else float("nan")
                    comp = "Rotation_Geodesic (rad / deg)"
                    val_str = (
                        f"{rad_val:.4f} / {deg_val:.2f}°" if math.isfinite(rad_val) else f"{rad_val:.4f} / -"
                    )
                    breakdown_table.add_row(comp, f"[value]{val_str}[/value]")
                else:
                    breakdown_table.add_row(f"{k.title()}", f"[value]{float(v):.4f}[/value]")
            except Exception:
                # 兜底：按原格式输出
                breakdown_table.add_row(f"{k.title()}", f"[value]{v}[/value]")

        console.print(Align.center(breakdown_table))

    # --- Set-level Metrics ---
    if isinstance(val_set_metrics, dict) and len(val_set_metrics) > 0:
        console.print("")  # blank line
        console.print("[metric]📊 Set Metrics[/metric]")

        set_metrics_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.MINIMAL_DOUBLE_HEAD,
            pad_edge=True,
            expand=False,
            border_style="blue",
            title="Set Metrics",
        )
        set_metrics_table.add_column("Metric", style="metric", no_wrap=True)
        set_metrics_table.add_column("Value", style="value", justify="right")

        for k, v in val_set_metrics.items():
            set_metrics_table.add_row(f"{k}", f"[value]{v:.4f}[/value]")

        console.print(Align.center(set_metrics_table))

    # --- Quality / Reconstruction Metrics ---
    if isinstance(val_quality_metrics, dict) and len(val_quality_metrics) > 0:
        console.print("")  # blank line
        console.print("[metric]⛲ Quality Metrics[/metric]")

        quality_table = Table(
            show_header=True,
            header_style="bold green",
            box=box.MINIMAL,
            pad_edge=True,
            expand=False,
            border_style="green",
            title="Quality",
        )
        quality_table.add_column("Metric", style="good", no_wrap=True)
        quality_table.add_column("Value", style="value", justify="right")

        for k, v in val_quality_metrics.items():
            metric_name = k.replace("_", " ").title()
            quality_table.add_row(f"{metric_name}", f"[value]{v:.4f}[/value]")

        console.print(Align.center(quality_table))

    # --- Flow Matching Metrics ---
    if isinstance(val_flow_metrics, dict) and len(val_flow_metrics) > 0:
        console.print("")  # blank line
        console.print("[metric]🫗 Flow Matching Metrics[/metric]")

        flow_table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.MINIMAL,
            pad_edge=True,
            expand=False,
            border_style="blue",
            title="Flow Metrics",
        )
        flow_table.add_column("Metric", style="metric", no_wrap=True)
        flow_table.add_column("Value", style="value", justify="right")

        for k, v in val_flow_metrics.items():
            metric_name = k.replace("_", " ").title()
            flow_table.add_row(f"{metric_name}", f"[value]{v:.4f}[/value]")

        console.print(Align.center(flow_table))

    # --- Diversity Metrics ---
    if isinstance(val_diversity_metrics, dict) and len(val_diversity_metrics) > 0:
        console.print("")  # blank line
        console.print("[metric]🎲 Diversity Metrics[/metric]")

        diversity_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.MINIMAL,
            pad_edge=True,
            expand=False,
            border_style="cyan",
            title="Diversity",
        )
        diversity_table.add_column("Metric", style="metric", no_wrap=True)
        diversity_table.add_column("Value", style="value", justify="right")

        for k, v in val_diversity_metrics.items():
            metric_name = k.replace("_", " ").title()
            diversity_table.add_row(f"{metric_name}", f"[value]{v:.4f}[/value]")

        console.print(Align.center(diversity_table))

    # --- Footer Rule ---
    console.rule("[good]✔️  Validation Summary Complete [/good]", style="green")

    # --- Debug Info ---
    if not isinstance(val_detailed_loss, dict):
        console.print("[warn]Warning: val_detailed_loss is not of type dict as expected.[/warn]")
    if avg_loss is None or avg_loss < 0:
        console.print("[warn]Warning: average_loss has an abnormal value.[/warn]")
