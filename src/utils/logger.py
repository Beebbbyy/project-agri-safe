"""
Logging utilities for Project Agri-Safe
"""

import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "agrisafe.log",
    rotation: str = "10 MB",
    retention: str = "30 days",
    serialize: bool = False
):
    """
    Set up logging configuration for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files (defaults to logs/)
        log_file: Name of the log file
        rotation: When to rotate the log file (e.g., "10 MB", "1 day")
        retention: How long to keep old log files
        serialize: Whether to serialize logs as JSON
    """
    # Remove default logger
    logger.remove()

    # Get log level from environment or use provided
    level = os.getenv("LOG_LEVEL", log_level).upper()

    # Console logging with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # File logging
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")

    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(log_dir, log_file)

    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        serialize=serialize
    )

    logger.info(f"Logging initialized - Level: {level}, Log file: {log_path}")

    return logger


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance

    Args:
        name: Optional logger name (for context)

    Returns:
        Loguru logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logging on import
setup_logging()
