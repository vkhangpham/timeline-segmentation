"""Centralized logging utility for the timeline segmentation project.

Provides consistent logging with timestamp formatting, verbosity control,
emoji prefixes for warnings/errors, and single file logging per run.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
import datetime

_global_log_file = None
_logging_configured = False


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emojis only to WARNING and ERROR messages."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record):
        formatted = super().format(record)

        if record.levelno >= logging.ERROR:
            formatted = f"❌ {formatted}"
        elif record.levelno >= logging.WARNING:
            formatted = f"⚠️ {formatted}"

        return formatted


class FileFormatter(logging.Formatter):
    """Plain formatter for file logging (no emojis)."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )


def ensure_log_directory() -> Path:
    """Ensure the results/logs directory exists.
    
    Returns:
        Path to the logs directory
    """
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def create_log_filename(domain_name: str = None) -> str:
    """Create a timestamped log filename.
    
    Args:
        domain_name: Optional domain name to include in filename
        
    Returns:
        Log filename with timestamp
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if domain_name:
        return f"timeline_analysis_{domain_name}_{timestamp}.log"
    else:
        return f"timeline_analysis_{timestamp}.log"


def setup_logging(
    verbose: bool = False, module_name: Optional[str] = None, domain_name: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration for a module.

    Args:
        verbose: If True, set to DEBUG level. If False, set to INFO level.
        module_name: Name of the module (defaults to calling module)
        domain_name: Optional domain name (ignored, kept for compatibility)

    Returns:
        Configured logger instance
    """
    global _global_log_file, _logging_configured
    
    if module_name is None:
        import inspect

        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get("__name__", "timeline")

    logger = logging.getLogger(module_name)

    if logger.handlers:
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    emoji_formatter = EmojiFormatter()
    console_handler.setFormatter(emoji_formatter)

    logger.addHandler(console_handler)

    if _logging_configured and _global_log_file:
        file_handler = logging.FileHandler(_global_log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)

        file_formatter = FileFormatter()
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def get_logger(module_name: str = None, verbose: bool = False, domain_name: str = None) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        module_name: Name of the module
        verbose: Verbosity flag
        domain_name: Optional domain name (ignored, kept for compatibility)

    Returns:
        Logger instance
    """
    if module_name is None:
        import inspect

        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get("__name__", "timeline")

    return setup_logging(verbose=verbose, module_name=module_name)


def configure_global_logging(verbose: bool = False, domain_name: str = None):
    """Configure global logging settings for the entire application.

    Args:
        verbose: If True, enable DEBUG level logging globally
        domain_name: Optional domain name for log file naming
    """
    global _global_log_file, _logging_configured
    
    if _logging_configured:
        return
    
    root_logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    emoji_formatter = EmojiFormatter()
    console_handler.setFormatter(emoji_formatter)

    root_logger.addHandler(console_handler)

    logs_dir = ensure_log_directory()
    log_filename = create_log_filename(domain_name)
    _global_log_file = logs_dir / log_filename

    file_handler = logging.FileHandler(_global_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)

    file_formatter = FileFormatter()
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(file_handler)

    _logging_configured = True

    root_logger.info(f"=== NEW ANALYSIS SESSION STARTED ===")
    if domain_name:
        root_logger.info(f"Domain: {domain_name}")
    root_logger.info(f"Log file: {_global_log_file}")
    root_logger.info(f"Verbose mode: {verbose}")
    root_logger.info(f"="*50)
