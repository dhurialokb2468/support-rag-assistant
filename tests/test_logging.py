import logging

from app.config import settings
from app.logger import get_logger, setup_logging


def test_logger_initialization():
    setup_logging(level="DEBUG")
    logger = get_logger("test_module")

    assert logger.name == "support_rag.test_module"
    assert logger.getEffectiveLevel() == logging.DEBUG


def test_logger_formatting(caplog):
    setup_logging(level="INFO")
    logger = get_logger("test_formatter")

    with caplog.at_level(logging.INFO):
        logger.info("Test log message for InsightFlow RAG")

    assert "Test log message for InsightFlow RAG" in caplog.text


def test_config_log_level():
    assert settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
