"""
Waste Classifier - Logging utility
Provided a centralized, configurable logger for the entire project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_loggers: dict = {}


def get_logger(
    name: str = "WasteClassifier",
    log_file: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Get or create a named logger with file and console handlers.

    Args:
        name: Logger name (typically module name).
        log_file: Path to log file. If none, logs only to console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to also log to console.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    Formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )    

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(Formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(Formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
