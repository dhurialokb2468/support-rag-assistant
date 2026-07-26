import logging
import sys

from app.config import settings

_LOGGERS = {}


def setup_logging(level: str | None = None) -> None:
    """Configures root logger with standard formatting respecting LOG_LEVEL settings."""
    log_level_str = (level or settings.log_level).upper()
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("support_rag")
    root_logger.setLevel(numeric_level)

    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers[0] = handler

    root_logger.info(f"InsightFlow RAG Application Logging initialized (LOG_LEVEL={log_level_str}).")


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger under the 'support_rag' namespace."""
    logger_name = f"support_rag.{name}"
    if logger_name not in _LOGGERS:
        root_logger = logging.getLogger("support_rag")
        if not root_logger.handlers:
            setup_logging()
        logger = logging.getLogger(logger_name)
        _LOGGERS[logger_name] = logger
    return _LOGGERS[logger_name]
