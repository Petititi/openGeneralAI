"""
Logging Configuration Module - Centralized logging setup.

This module provides a consistent logging configuration for the application.
All modules should use this logger to maintain consistent logging behavior.

Responsibility: Logging configuration and logger instances.
"""

import logging
import sys
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    handler: Optional[logging.Handler] = None
) -> logging.Logger:
    """
    Set up a logger with consistent configuration.
    
    Args:
        name: The name of the logger (typically __name__)
        level: The logging level (default: INFO)
        log_format: The format string for log messages
        date_format: The format string for timestamps
        handler: Optional custom handler (default: StreamHandler to stderr)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers if logger already has them
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
    
    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with standard configuration.
    
    This is a convenience function that returns a logger with standard settings.
    Use this for most logging needs in the application.
    
    Args:
        name: The name of the logger (typically __name__)
        level: The logging level (default: INFO)
        
    Returns:
        Logger instance
    """
    return setup_logger(name, level)


# Pre-configured loggers for common modules
# These can be imported and used directly
APP_LOGGER = "openGeneralAI"
storage_logger = lambda: get_logger("openGeneralAI.storage")
agents_logger = lambda: get_logger("openGeneralAI.agents")
tools_logger = lambda: get_logger("openGeneralAI.tools")
