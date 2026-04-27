# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import io
import logging
import unittest
from contextlib import redirect_stderr

from GBOpt.Utils.logging_utils import EventFormatter, configure_logging, get_logger, make_run_adapter


class TestLoggingUtils(unittest.TestCase):

    def setUp(self):
        self.package_logger = logging.getLogger("GBOpt")
        self.original_handlers = list(self.package_logger.handlers)
        self.original_level = self.package_logger.level
        self.original_propagate = self.package_logger.propagate

    def tearDown(self):
        self.package_logger.handlers = self.original_handlers
        self.package_logger.setLevel(self.original_level)
        self.package_logger.propagate = self.original_propagate

    def test_event_formatter_handles_missing_optional_fields(self):
        formatter = EventFormatter()
        logger = logging.getLogger("GBOpt.test.formatter")
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            __file__,
            1,
            "Hello world",
            (),
            None,
            extra={"event": "hello_world"},
        )

        formatted = formatter.format(record)

        self.assertIn("INFO GBOpt.test.formatter Hello world", formatted)
        self.assertIn("event=hello_world", formatted)

    def test_configure_logging_emits_structured_fields(self):
        stream = io.StringIO()
        configure_logging(level="INFO", stream=stream)
        logger = make_run_adapter(
            get_logger("GBOpt.test.configure"),
            component="test_component",
            unique_id="run-1",
        )

        logger.info("Configured logger", extra={"event": "configured", "step": 3})

        output = stream.getvalue()
        self.assertIn("INFO GBOpt.test.configure Configured logger", output)
        self.assertIn("event=configured", output)
        self.assertIn("component=test_component", output)
        self.assertIn("step=3", output)
        self.assertIn("unique_id=run-1", output)

    def test_package_logger_is_silent_without_configured_handlers(self):
        self.package_logger.handlers = []
        logger = get_logger("GBOpt.test.silent")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            logger.warning("No console output expected", extra={"event": "silent_warning"})

        self.assertEqual(stderr.getvalue(), "")
