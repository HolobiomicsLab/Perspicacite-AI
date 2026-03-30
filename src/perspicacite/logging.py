"""Structured logging setup."""

import logging
import sys
from typing import Any


class DummyLogger:
    """Dummy logger for when structlog is not available."""

    def debug(self, msg: str, **kwargs: Any) -> None:
        pass

    def info(self, msg: str, **kwargs: Any) -> None:
        print(f"INFO: {msg}", file=sys.stderr)

    def warning(self, msg: str, **kwargs: Any) -> None:
        print(f"WARNING: {msg}", file=sys.stderr)

    def error(self, msg: str, **kwargs: Any) -> None:
        print(f"ERROR: {msg}", file=sys.stderr)


try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


from perspicacite.config.schema import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure structured logging.

    Args:
        config: Logging configuration
    """
    if not STRUCTLOG_AVAILABLE:
        # Fall back to standard logging
        logging.basicConfig(
            level=_get_log_level(config.level),
            format="%(levelname)s: %(message)s",
            stream=sys.stdout,
        )
        return

    # Configure structlog (avoid stdlib.filter_by_level: it requires a stdlib
    # Logger; PrintLoggerFactory yields PrintLogger without .disabled).
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if config.format == "json":
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ])
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            _get_log_level(config.level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=_get_log_level(config.level),
    )


def _get_log_level(level: str) -> int:
    """Convert string level to logging constant."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    return levels.get(level, logging.INFO)


def get_logger(name: str) -> Any:
    """Get a logger instance."""
    if not STRUCTLOG_AVAILABLE:
        return DummyLogger()
    return structlog.get_logger(name)


def mask_secret(value: str) -> str:
    """Mask a secret value for logging."""
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
