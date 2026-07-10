"""
Pytest suite for the orchestration runner (src/pipeline.py).

Every test mocks all seven step functions (ingest / transform / validate /
classify / enrich / detect / run_agent) at their import site in
``src.pipeline`` using ``unittest.mock.patch`` so no real ingestion, LLM
calls, or cloud/data access occurs. All PATHS entries are redirected into
pytest's ``tmp_path`` fixture via ``patch.dict`` so no files are ever written
under the real ``data/`` directory. Tests that exercise the success path also
mock ``os.path.exists`` because the pipeline checks that prerequisite
artifacts exist before running each step, and those files are never
physically created by the mocked step functions.
"""

import json
import os
from unittest.mock import Mock, patch

import pytest

from src import pipeline


def _ok_returns(mock_ingest, mock_transform, mock_validate,
                mock_classify, mock_enrich, mock_detect, mock_agent):
    """Give the mocked steps the return shapes _record_count expects."""
    mock_ingest.return_value = 500
    mock_transform.return_value = 500
    mock_validate.return_value = {"total_records": 500}
    mock_classify.return_value = {"total": 500, "classified": 500}
    mock_enrich.return_value = {"total": 500, "enriched": 500}
    mock_detect.return_value = {"total_tickets": 500, "anomalies_found": 3}
    mock_agent.return_value = {"anomalies_processed": 3, "total_tool_calls": 5}


def _all_paths(tmp_path):
    """patch.dict payload redirecting every PATHS entry to tmp_path subtrees."""
    return {
        "raw":            str(tmp_path / "raw" / "tickets.parquet"),
        "normalized":     str(tmp_path / "processed" / "tickets_normalized.parquet"),
        "validated":      str(tmp_path / "processed" / "tickets_validated.parquet"),
        "quality_report": str(tmp_path / "quality" / "quality_report.json"),
        "pipeline_state": str(tmp_path / "pipeline_state.json"),
        "classified":     str(tmp_path / "enriched" / "tickets_classified.parquet"),
        "enriched":       str(tmp_path / "enriched" / "tickets_enriched.parquet"),
        "anomalies":      str(tmp_path / "anomalies" / "anomalies.parquet"),
        "anomaly_report": str(tmp_path / "anomalies" / "anomaly_report.json"),
    }


# 1 --------------------------------------------------------------------------
@patch("src.pipeline.os.path.exists", return_value=True)
@patch("src.pipeline.run_agent")
@patch("src.pipeline.detect")
@patch("src.pipeline.enrich")
@patch("src.pipeline.classify")
@patch("src.pipeline.validate")
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_all_steps_called_in_order(mock_ingest, mock_transform, mock_validate,
                                   mock_classify, mock_enrich, mock_detect,
                                   mock_agent, _mock_exists, tmp_path):
    _ok_returns(mock_ingest, mock_transform, mock_validate,
                mock_classify, mock_enrich, mock_detect, mock_agent)
    manager = Mock()
    manager.attach_mock(mock_ingest, "ingest")
    manager.attach_mock(mock_transform, "transform")
    manager.attach_mock(mock_validate, "validate")
    manager.attach_mock(mock_classify, "classify")
    manager.attach_mock(mock_enrich, "enrich")
    manager.attach_mock(mock_detect, "detect")
    manager.attach_mock(mock_agent, "run_agent")

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        pipeline.run_pipeline()

    called = [name for name, _args, _kwargs in manager.mock_calls]
    assert called == ["ingest", "transform", "validate", "classify",
                      "enrich", "detect", "run_agent"]


# 2 --------------------------------------------------------------------------
@patch("src.pipeline.os.path.exists", return_value=True)
@patch("src.pipeline.run_agent")
@patch("src.pipeline.detect")
@patch("src.pipeline.enrich")
@patch("src.pipeline.classify")
@patch("src.pipeline.validate")
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_returns_ok_status(mock_ingest, mock_transform, mock_validate,
                           mock_classify, mock_enrich, mock_detect,
                           mock_agent, _mock_exists, tmp_path):
    _ok_returns(mock_ingest, mock_transform, mock_validate,
                mock_classify, mock_enrich, mock_detect, mock_agent)

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        result = pipeline.run_pipeline()

    assert result["pipeline_status"] == "ok"


# 3 --------------------------------------------------------------------------
@patch("src.pipeline.os.path.exists", return_value=True)
@patch("src.pipeline.run_agent")
@patch("src.pipeline.detect")
@patch("src.pipeline.enrich")
@patch("src.pipeline.classify")
@patch("src.pipeline.validate")
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_duration_ms_present(mock_ingest, mock_transform, mock_validate,
                             mock_classify, mock_enrich, mock_detect,
                             mock_agent, _mock_exists, tmp_path):
    _ok_returns(mock_ingest, mock_transform, mock_validate,
                mock_classify, mock_enrich, mock_detect, mock_agent)

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        result = pipeline.run_pipeline()

    assert len(result["steps"]) == 7  # all seven steps present
    for step in result["steps"].values():
        assert "duration_ms" in step


# 4 --------------------------------------------------------------------------
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_ingest_failure_exits_1(mock_ingest, mock_transform, tmp_path):
    mock_ingest.side_effect = RuntimeError("ingest boom")

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        with pytest.raises(SystemExit) as exc_info:
            pipeline.run_pipeline()

    assert exc_info.value.code == 1
    mock_transform.assert_not_called()


# 5 --------------------------------------------------------------------------
@patch("src.pipeline.os.path.exists", return_value=True)
@patch("src.pipeline.validate")
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_transform_failure_exits_1(mock_ingest, mock_transform, mock_validate, _mock_exists, tmp_path):
    mock_ingest.return_value = 500
    mock_transform.side_effect = RuntimeError("transform boom")

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        with pytest.raises(SystemExit) as exc_info:
            pipeline.run_pipeline()

    assert exc_info.value.code == 1
    mock_validate.assert_not_called()


# 6 --------------------------------------------------------------------------
@patch("src.pipeline.os.path.exists", return_value=True)
@patch("src.pipeline.run_agent")
@patch("src.pipeline.detect")
@patch("src.pipeline.enrich")
@patch("src.pipeline.classify")
@patch("src.pipeline.validate")
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_state_file_written_on_success(mock_ingest, mock_transform, mock_validate,
                                       mock_classify, mock_enrich, mock_detect,
                                       mock_agent, _mock_exists, tmp_path):
    _ok_returns(mock_ingest, mock_transform, mock_validate,
                mock_classify, mock_enrich, mock_detect, mock_agent)
    state_path = tmp_path / "pipeline_state.json"

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        pipeline.run_pipeline()

    assert state_path.exists()


# 7 --------------------------------------------------------------------------
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_state_file_written_on_failure(mock_ingest, mock_transform, tmp_path):
    mock_ingest.side_effect = RuntimeError("ingest boom")
    state_path = tmp_path / "pipeline_state.json"

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        with pytest.raises(SystemExit):
            pipeline.run_pipeline()

    assert state_path.exists()


# 8 --------------------------------------------------------------------------
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_state_file_status_failed(mock_ingest, mock_transform, tmp_path):
    mock_ingest.side_effect = RuntimeError("ingest boom")
    state_path = tmp_path / "pipeline_state.json"

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        with pytest.raises(SystemExit):
            pipeline.run_pipeline()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert "error" in state


# 9 --------------------------------------------------------------------------
@patch("src.pipeline.validate")
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_dry_run_no_steps_executed(mock_ingest, mock_transform, mock_validate, tmp_path):
    state_path = tmp_path / "pipeline_state.json"

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        pipeline.run_pipeline(dry_run=True)

    mock_ingest.assert_not_called()
    mock_transform.assert_not_called()
    mock_validate.assert_not_called()
    assert not state_path.exists()  # dry-run modifies no files


# 10 -------------------------------------------------------------------------
@patch("src.pipeline.os.path.exists", return_value=True)
@patch("src.pipeline.validate")
@patch("src.pipeline.transform")
@patch("src.pipeline.ingest")
def test_step_flag_runs_only_transform(
    mock_ingest, mock_transform, mock_validate, _mock_exists, tmp_path
):
    mock_transform.return_value = 500

    with patch.dict(pipeline.PATHS, _all_paths(tmp_path)):
        pipeline.run_pipeline(step="transform")

    mock_transform.assert_called_once()
    mock_ingest.assert_not_called()
    mock_validate.assert_not_called()
