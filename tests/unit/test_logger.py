import json
import logging
import os
from unittest import mock

from mlops_utils.logger import get_logger


def test_get_logger_standard(capsys):
    # Ensure environment is clean
    if "MLOPS_JSON_LOGS" in os.environ:
        del os.environ["MLOPS_JSON_LOGS"]

    logger = get_logger("test_standard")
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert logger.propagate is False

    logger.info("Test message")
    captured = capsys.readouterr()
    assert "Test message" in captured.out
    assert "INFO" in captured.out
    assert "test_standard" in captured.out

def test_get_logger_json(capsys):
    with mock.patch.dict(os.environ, {"MLOPS_JSON_LOGS": "1"}):
        logger = get_logger("test_json")
        logger.info("Test json message")

        captured = capsys.readouterr()

        # Output should be valid JSON
        log_dict = json.loads(captured.out)
        assert log_dict["message"] == "Test json message"
        assert log_dict["level"] == "INFO"
        assert log_dict["name"] == "test_json"
        assert "timestamp" in log_dict

def test_get_logger_singleton_handlers():
    logger1 = get_logger("test_singleton")
    assert len(logger1.handlers) == 1

    # Calling it again shouldn't add another handler
    logger2 = get_logger("test_singleton")
    assert len(logger2.handlers) == 1
    assert logger1 is logger2
