"""Basic logging setup for NEER."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "neer", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a consistent format across the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
