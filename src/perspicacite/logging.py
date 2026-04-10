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

    Routes structlog through stdlib logging so that any FileHandler
    attached to the root logger (e.g. by web_app_full.py) captures output.

    Args:
        config: Logging configuration
    """
    log_level = _get_log_level(config.level)

    if not STRUCTLOG_AVAILABLE:
        # Fall back to standard logging
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s: %(message)s",
            stream=sys.stdout,
        )
        return

    # Choose a renderer for the final output
    if config.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    # Shared processors that run for *all* structlog loggers
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Convert to stdlib logging record
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib root logger so structlog records flow to
    # whatever handlers are attached (StreamHandler, FileHandler, etc.)
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    # Only add our handler if the root logger has none yet (web_app sets its own)
    if not root.handlers:
        root.addHandler(handler)


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
