"""Logging utility for Explainable AI Plant Disease Detection.

Configures file and stream loggers with tqdm progress bar compatibility.
"""

import logging
import sys
from pathlib import Path


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that redirects logs via tqdm.write to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            import tqdm
            tqdm.tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(log_dir: Path, name: str = "xai_pds") -> logging.Logger:
    """Sets up a logger instance writing to logs/training.log and console.

    Args:
        log_dir: Directory where log files are stored.
        name: Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Tqdm Console Handler
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
