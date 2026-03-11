"""Centralised logging setup for the backend application.

Format: YYYY-MM-DD HH:MM:SS | LEVEL | message
Outputs to both the console (stdout) and a rotating log file.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """Configure root logger with console + rotating-file handlers.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        log_file:  Path to the output log file (parent dirs are created as needed).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers when called more than once (e.g. during tests)
    if root.handlers:
        root.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # --- Rotating file handler ---
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet overly verbose third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "qdrant_client", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
