"""
Centralized logging setup.

Console handler: INFO level, concise summaries.
File handler: DEBUG level, timestamps, full details. Written to logs/<run_id>.log.
"""
import logging
import os
import time
from datetime import datetime

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_initialized = False


def setup_logging(run_id: str | None = None) -> logging.Logger:
    """
    Configures and returns the project-wide logger.

    @param run_id: Unique identifier for this run (used for log filename).
                   Defaults to a timestamp.
    @return: Configured logger instance.
    """
    global _initialized

    logger = logging.getLogger("saxe")

    if _initialized:
        return logger

    logger.setLevel(logging.DEBUG)

    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Console handler: concise, INFO-level
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

    # File handler: verbose, DEBUG-level
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_path = os.path.join(_LOG_DIR, f"{run_id}.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(file_handler)

    _initialized = True
    logger.info(f"Logging initialized. Debug log: {log_path}")
    return logger


class Timer:
    """Simple context-manager stopwatch that logs elapsed time."""

    def __init__(self, description: str, level: int = logging.INFO):
        self.description = description
        self.level = level
        self.logger = logging.getLogger("saxe")

    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.debug(f"Started: {self.description}")
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self.start
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            mins, secs = divmod(elapsed, 60)
            time_str = f"{int(mins)}m {secs:.1f}s"
        self.logger.log(self.level, f"{self.description} — {time_str}")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start
