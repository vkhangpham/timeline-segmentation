"""
Centralized logging utility for the timeline segmentation project.

Provides consistent logging with:
- Timestamp, log level, and message formatting
- Verbosity control
- Emoji prefixes only for warnings and errors
- Module-specific loggers
"""

import logging
import sys
from typing import Optional


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emojis only to WARNING and ERROR messages."""
    
    def __init__(self):
        # Format: 2025-01-04 10:30:45 | INFO | message
        super().__init__(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        # Get the basic formatted message
        formatted = super().format(record)
        
        # Add emoji prefix only for WARNING and ERROR
        if record.levelno >= logging.ERROR:
            formatted = f"❌ {formatted}"
        elif record.levelno >= logging.WARNING:
            formatted = f"⚠️ {formatted}"
        
        return formatted


def setup_logging(verbose: bool = False, module_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for a module.
    
    Args:
        verbose: If True, set to DEBUG level. If False, set to INFO level.
        module_name: Name of the module (defaults to calling module)
        
    Returns:
        Configured logger instance
    """
    # Get logger name - use module_name or calling module
    if module_name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get('__name__', 'timeline')
    
    logger = logging.getLogger(module_name)
    
    # Avoid adding multiple handlers if already configured
    if logger.handlers:
        return logger
    
    # Set logging level based on verbosity
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Set custom formatter
    formatter = EmojiFormatter()
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(module_name: str = None, verbose: bool = False) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        verbose: Verbosity flag
        
    Returns:
        Logger instance
    """
    if module_name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get('__name__', 'timeline')
    
    return setup_logging(verbose=verbose, module_name=module_name)


def configure_global_logging(verbose: bool = False):
    """
    Configure global logging settings for the entire application.
    
    Args:
        verbose: If True, enable DEBUG level logging globally
    """
    # Set root logger level
    root_logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler for root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Set custom formatter
    formatter = EmojiFormatter()
    handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(handler)


# Convenience functions for common logging patterns
def log_info(message: str, module_name: str = None):
    """Log an info message."""
    logger = get_logger(module_name)
    logger.info(message)


def log_warning(message: str, module_name: str = None):
    """Log a warning message."""
    logger = get_logger(module_name)
    logger.warning(message)


def log_error(message: str, module_name: str = None):
    """Log an error message."""
    logger = get_logger(module_name)
    logger.error(message)


def log_debug(message: str, module_name: str = None):
    """Log a debug message."""
    logger = get_logger(module_name)
    logger.debug(message) 