# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import logging
import sys
from typing import Any


PACKAGE_LOGGER_NAME = "GBOpt"
_RESERVED_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


def _ensure_package_logger() -> logging.Logger:
    """Keep package loggers quiet until callers opt in with a handler."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


class EventFormatter(logging.Formatter):
    """Render a readable message plus stable key-value fields."""

    def __init__(self, include_fields: bool = True):
        super().__init__("%(levelname)s %(name)s %(message)s")
        self.include_fields = include_fields

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not self.include_fields:
            return base

        fields = []
        event = getattr(record, "event", None)
        if event is not None:
            fields.append(("event", event))

        for key, value in sorted(record.__dict__.items()):
            if key in _RESERVED_RECORD_FIELDS or key == "event":
                continue
            if value is None or callable(value):
                continue
            fields.append((key, value))

        if not fields:
            return base

        rendered = " ".join(f"{key}={value}" for key, value in fields)
        return f"{base} | {rendered}"


class RunLoggerAdapter(logging.LoggerAdapter):
    """Bind shared context while allowing per-call overrides."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(self.extra)
        if "extra" in kwargs and kwargs["extra"] is not None:
            extra.update(kwargs["extra"])
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    _ensure_package_logger()
    return logging.getLogger(name)


def configure_logging(
    level: str | int = "INFO",
    stream=None,
    include_fields: bool = True,
) -> logging.Logger:
    """Attach a console handler to the GBOpt package logger."""
    logger = _ensure_package_logger()
    handler_stream = sys.stderr if stream is None else stream
    resolved_level = logging._nameToLevel.get(level.upper(), logging.INFO) if isinstance(level, str) else level

    logger.handlers = [
        handler for handler in logger.handlers
        if isinstance(handler, logging.NullHandler)
    ]

    handler = logging.StreamHandler(handler_stream)
    handler.setFormatter(EventFormatter(include_fields=include_fields))
    logger.addHandler(handler)
    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger


def make_run_adapter(logger: logging.Logger, **context: Any) -> RunLoggerAdapter:
    return RunLoggerAdapter(logger, context)
