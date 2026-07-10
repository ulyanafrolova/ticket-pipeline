"""Unit tests for src/logger.py — JSON structured logging."""

import json
import uuid

from src.logger import get_logger


def _fresh_name() -> str:
    """Unique logger name per test: loggers are cached process-wide by name."""
    return f"test-logger-{uuid.uuid4().hex[:8]}"


def _log_lines(capsys) -> list[dict]:
    out = capsys.readouterr().out.strip().splitlines()
    return [json.loads(line) for line in out if line.strip()]


def test_output_is_valid_json(capsys):
    logger = get_logger(_fresh_name())
    logger.info("hello")
    logger.error("world")
    lines = _log_lines(capsys)  # json.loads raises if any line is not valid JSON
    assert len(lines) == 2


def test_required_fields_present(capsys):
    name = _fresh_name()
    logger = get_logger(name)
    logger.info("checking fields")
    entry = _log_lines(capsys)[-1]
    for field in ("timestamp", "level", "component", "message"):
        assert field in entry, f"missing required field: {field}"
    assert entry["component"] == name
    assert entry["message"] == "checking fields"


def test_extra_fields_merged(capsys):
    logger = get_logger(_fresh_name())
    logger.info("step done", extra={"step": "Transform", "records": 500})
    entry = _log_lines(capsys)[-1]
    assert entry["step"] == "Transform"
    assert entry["records"] == 500


def test_run_id_included(capsys):
    run_id = "2026-07-07T12:00:00+00:00"
    logger = get_logger(_fresh_name(), run_id=run_id)
    logger.info("with run id")
    entry = _log_lines(capsys)[-1]
    assert entry["run_id"] == run_id


def test_no_duplicate_handlers():
    name = _fresh_name()
    logger_first = get_logger(name)
    logger_second = get_logger(name)
    assert logger_first is logger_second
    assert len(logger_second.handlers) == 1


def test_warning_level(capsys):
    logger = get_logger(_fresh_name())
    logger.warning("careful")
    entry = _log_lines(capsys)[-1]
    assert entry["level"] == "WARNING"
