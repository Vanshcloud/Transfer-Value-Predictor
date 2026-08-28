"""Logging is configured once, and configuring twice does not duplicate output."""

from __future__ import annotations

import logging

from src.utils.logging import configure_logging, get_logger, reset_logging


def teardown_function() -> None:
    reset_logging()


def test_configure_installs_exactly_one_handler() -> None:
    reset_logging()
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == 1


def test_repeated_configuration_does_not_add_handlers() -> None:
    """The bug this guards: every call appending a handler, so each log line
    prints once per call."""
    reset_logging()
    configure_logging("INFO")
    configure_logging("INFO")
    configure_logging("DEBUG")
    assert len(logging.getLogger().handlers) == 1


def test_force_reconfigures_without_duplicating() -> None:
    reset_logging()
    configure_logging("INFO")
    configure_logging("DEBUG", force=True)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG


def test_level_is_applied() -> None:
    reset_logging()
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_get_logger_returns_named_logger() -> None:
    assert get_logger("src.demo").name == "src.demo"
