import json
import logging
import os
import sys
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include exception traceback if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a configured logger with the specified name.
    
    If the environment variable MLOPS_JSON_LOGS is set to '1' or 'true',
    logs will be output as JSON. Otherwise, standard readable text is used.
    
    Ensures that stream handlers are not duplicated across multiple calls.
    """
    logger = logging.getLogger(name)

    # Only configure if the logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Output to stdout instead of stderr (which is standard logging default)
        # Databricks captures stdout and stderr cleanly, but stdout is often preferred for app logs
        handler = logging.StreamHandler(sys.stdout)

        use_json = os.environ.get("MLOPS_JSON_LOGS", "").lower() in ("1", "true")

        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent propagation to the root logger to avoid double logging
        logger.propagate = False

    return logger
