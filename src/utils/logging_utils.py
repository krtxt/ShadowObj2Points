import logging
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from lightning.pytorch.utilities import rank_zero_only

def setup_rich_logging(level: int = logging.INFO) -> None:
    """Configure rich logging for cleaner console output."""
    # Suppress logging on non-zero ranks by default
    logging.getLogger().setLevel(logging.ERROR)

    @rank_zero_only
    def _setup_on_rank_zero():
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
