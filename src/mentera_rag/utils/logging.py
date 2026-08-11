"""
Structured JSON Logger & Contextual Logging Middleware — Mentera RAG Pipeline.

Configures application-wide JSON logging and provides a ContextVar-backed
request ID logger for tracking log messages across async/sync bounds.
"""

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime

# Context variable to hold the request ID for the current request context
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class StructuredJSONFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs log records as single-line JSON objects.
    Automatically injects the active request_id from ContextVars.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Resolve exception traceback if present
        exc_info = ""
        if record.exc_info:
            exc_info = "".join(traceback.format_exception(*record.exc_info))

        log_payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "filename": record.filename,
            "line_number": record.lineno,
        }

        if exc_info:
            log_payload["exception"] = exc_info

        # Add any extra attributes supplied via extra={} in logging call
        for key, val in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                log_payload[key] = val

        return json.dumps(log_payload)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    Configures root logger to output JSON to stdout.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Output JSON to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJSONFormatter())
    root_logger.addHandler(handler)

    # Minimize verbose noise from library logs
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
