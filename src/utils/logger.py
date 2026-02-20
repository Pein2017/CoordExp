"""
Unified logging system for Qwen3-VL with distributed GPU support.

This module provides rank-aware logging that integrates with ms-swift's logger.
In distributed training environments, only rank 0 logs by default to reduce noise.
Set QWEN3VL_VERBOSE=1 to enable logging from all ranks.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

# Try to import ms-swift's logger, fall back to standard logging
try:
    from swift.utils import get_logger as swift_get_logger

    _SWIFT_AVAILABLE = True
except ImportError:
    _SWIFT_AVAILABLE = False


def get_rank() -> int:
    """
    Get the current process rank in distributed training.

    Returns:
        Process rank (0 for single GPU or main process)
    """
    # Check various distributed training environment variables
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)

    # Not in distributed mode
    return 0


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).

    Returns:
        True if main process or not in distributed mode
    """
    return get_rank() == 0


def should_log() -> bool:
    """
    Determine if current process should log messages.

    Returns:
        True if rank 0 or verbose mode enabled
    """
    # Check if verbose mode is enabled
    verbose = os.environ.get("QWEN3VL_VERBOSE", "0").strip().lower()
    if verbose in ("1", "true", "yes"):
        return True

    # Only log from rank 0
    return is_main_process()


class RankFilter(logging.Filter):
    """
    Logging filter that blocks messages from non-main processes.

    In distributed training, this ensures only rank 0 logs by default.
    Can be overridden with QWEN3VL_VERBOSE=1 environment variable.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records based on process rank.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged
        """
        return should_log()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a rank-aware logger that integrates with ms-swift.

    This logger automatically filters messages in distributed training
    so only rank 0 logs by default. Set QWEN3VL_VERBOSE=1 to enable
    logging from all ranks.

    Args:
        name: Logger name. If None, uses the calling module's name.
              Can be a simple name like "sft" or hierarchical like "datasets.builder".

    Returns:
        Configured logger instance

    Examples:
        >>> from src.utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")  # Only logs from rank 0

        >>> # Enable verbose mode to see logs from all ranks
        >>> import os
        >>> os.environ['QWEN3VL_VERBOSE'] = '1'
        >>> logger.info("This will log from all ranks")
    """
    # Use ms-swift's logger if available, otherwise standard logging
    if _SWIFT_AVAILABLE:
        # ms-swift's get_logger returns a logger from their infrastructure
        logger = swift_get_logger()

        # If a specific name is requested, get or create that logger
        if name:
            # Get logger with the specified name under swift's hierarchy
            logger = logging.getLogger(f"swift.custom.{name}")
    else:
        # Fall back to standard Python logging
        if name is None:
            name = __name__
        logger = logging.getLogger(name)

        # Configure basic logging if no handlers exist
        if not logger.handlers and not logging.getLogger().handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    # Add rank filter to prevent duplicate logs in distributed training
    # Check if filter already exists to avoid adding multiple times
    has_rank_filter = any(isinstance(f, RankFilter) for f in logger.filters)
    if not has_rank_filter:
        logger.addFilter(RankFilter())

    return logger


def set_log_level(level: int) -> None:
    """
    Set the log level for all Qwen3-VL loggers.

    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)

    Examples:
        >>> import logging
        >>> from src.utils.logger import set_log_level
        >>> set_log_level(logging.DEBUG)  # Show debug messages
    """
    # Set level on root logger
    logging.getLogger().setLevel(level)

    # Set level on swift loggers if available
    if _SWIFT_AVAILABLE:
        swift_logger = logging.getLogger("swift")
        swift_logger.setLevel(level)


def enable_verbose_logging() -> None:
    """
    Enable logging from all ranks in distributed training.

    This sets the QWEN3VL_VERBOSE environment variable to '1',
    allowing all processes to emit log messages.

    Examples:
        >>> from src.utils.logger import enable_verbose_logging
        >>> enable_verbose_logging()
        >>> # Now all ranks will log
    """
    os.environ["QWEN3VL_VERBOSE"] = "1"


def disable_verbose_logging() -> None:
    """
    Disable verbose logging (back to rank 0 only).

    Examples:
        >>> from src.utils.logger import disable_verbose_logging
        >>> disable_verbose_logging()
        >>> # Only rank 0 will log
    """
    os.environ["QWEN3VL_VERBOSE"] = "0"


# ---------------------------------------------------------------------------
# File logging helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FileLoggingConfig:
    enabled: bool = True
    filename: str = "train.log"
    capture_stdout: bool = False
    capture_stderr: bool = False


class _TeeStream:
    def __init__(self, stream, file_obj, *, name: str):
        self._stream = stream
        self._file = file_obj
        self._name = str(name)
        self._warned = False

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        try:
            sys.__stderr__.write(str(message) + "\n")
            sys.__stderr__.flush()
        except (OSError, ValueError):
            # Best-effort: do not let logging diagnostics crash the run.
            return

    def write(self, data):
        self._stream.write(data)
        if self._file is None:
            return
        try:
            self._file.write(data)
        except (OSError, ValueError) as exc:
            self._warn_once(
                f"Log tee ({self._name}) disabled: {exc.__class__.__name__}: {exc}"
            )
            self._file = None

    def flush(self):
        self._stream.flush()
        if self._file is None:
            return
        try:
            self._file.flush()
        except (OSError, ValueError) as exc:
            self._warn_once(
                f"Log tee ({self._name}) disabled: {exc.__class__.__name__}: {exc}"
            )
            self._file = None

    def isatty(self):
        try:
            return bool(self._stream.isatty())
        except (AttributeError, OSError, ValueError):
            return False


def enable_output_dir_file_logging(
    output_dir: str, cfg: Optional[FileLoggingConfig] = None
) -> str | None:
    """Mirror logs (and optionally stdout/stderr) into output_dir.

    - Only activates on the main process (rank 0).
    - Attaches a FileHandler to the `swift` logger so ms-swift logs are captured.
    - Optionally tees stdout/stderr to the same file for quick review.
    """

    if not output_dir or not is_main_process():
        return None

    if cfg is None:
        cfg = FileLoggingConfig()
    if not cfg.enabled:
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "File logging disabled: failed to create output_dir %s (%s)",
            output_dir,
            exc,
        )
        return None

    filename = cfg.filename.strip() if isinstance(cfg.filename, str) else "train.log"
    if not filename:
        filename = "train.log"
    log_path = os.path.join(output_dir, filename)

    swift_logger = logging.getLogger("swift")
    for h in list(getattr(swift_logger, "handlers", []) or []):
        base = getattr(h, "baseFilename", None)
        if isinstance(base, str) and os.path.abspath(base) == os.path.abspath(log_path):
            return log_path

    try:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "File logging disabled: failed to open log file %s (%s)",
            log_path,
            exc,
        )
        return None

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    swift_logger.addHandler(handler)
    swift_logger.setLevel(logging.DEBUG)

    try:
        file_obj = open(log_path, "a", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "File logging tee disabled: failed to open %s (%s)",
            log_path,
            exc,
        )
        file_obj = None

    if file_obj is not None:
        if cfg.capture_stdout:
            try:
                sys.stdout = _TeeStream(sys.stdout, file_obj, name="stdout")
            except (AttributeError, TypeError) as exc:
                logger.warning("Failed to tee stdout to %s (%s)", log_path, exc)
        if cfg.capture_stderr:
            try:
                sys.stderr = _TeeStream(sys.stderr, file_obj, name="stderr")
            except (AttributeError, TypeError) as exc:
                logger.warning("Failed to tee stderr to %s (%s)", log_path, exc)

    return log_path


# Convenience logger for this module
logger = get_logger(__name__)
