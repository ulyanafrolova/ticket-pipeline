"""Pytest suite for the advanced validation layer.

Tests build normalized-schema DataFrames in memory, write them to a temporary
Parquet file, and run the real validate() against tmp_path. The dead-letter
queue, HTML report, and previous-run report are derived from report_path's
directory, so every artifact lands under tmp_path — no real data is touched.
"""

import json
import os
import uuid

import pandas as pd
import pytest

from src.validate import FLAG_NAMES, validate


def _uuid():
    return str(uuid.uuid4())


def _base_row(**overrides):
    """A fully valid normalized ticket row; override individual fields per test."""
    row = {
        "ticket_id": _uuid(),
        "created_at": "2024-01-15T10:30:00+00:00",
        "customer_id": "cust-1",
        "channel": "email",
        "subject": "Need help with my billing question",
        "body": "Please assist me with my invoice problem.",
        "priority": "high",
        "category": "billing",
        "status": "open",
        "agent_id": "agent-1",
    }
    row.update(overrides)
    return row


@pytest.fixture
def make_input(tmp_path):
    """Factory: write rows to a normalized Parquet and return its path."""

    def _make(rows, name="normalized.parquet"):
        df = pd.DataFrame(rows)
        path = tmp_path / name
        df.to_parquet(path, index=False)
        return str(path)

    return _make


@pytest.fixture
def paths(tmp_path):
    """Canonical output paths, all under tmp_path/quality and tmp_path/processed."""
    quality = tmp_path / "quality"
    processed = tmp_path / "processed"
    return {
        "output": str(processed / "tickets_validated.parquet"),
        "report": str(quality / "quality_report.json"),
        "html": str(quality / "quality_report.html"),
        "rejected": str(quality / "rejected.parquet"),
        "previous": str(quality / "quality_report_previous.json"),
    }


@pytest.fixture
def run_validate(make_input, paths):
    """Run validate() over `rows`; return (metrics, validated_df)."""

    def _run(rows):
        input_path = make_input(rows)
        metrics = validate(input_path, paths["output"], paths["report"])
        out = pd.read_parquet(paths["output"])
        return metrics, out

    return _run


# 1
def test_completeness_subject_passes(run_validate):
    rows = [_base_row() for _ in range(50)]
    rows[0]["subject"] = None  # 49/50 = 0.98 >= 0.90
    metrics, _ = run_validate(rows)
    assert metrics["checks"]["completeness_subject"]["passed"] is True
    assert metrics["checks"]["completeness_subject"]["value"] == 0.98


# 2
def test_completeness_subject_fails_warn(run_validate, caplog):
    rows = [_base_row() for _ in range(10)]
    for r in rows[:2]:
        r["subject"] = None  # 8/10 = 0.80 < 0.90
    # Must warn but NOT exit.
    metrics, _ = run_validate(rows)
    assert metrics["checks"]["completeness_subject"]["passed"] is False
    assert metrics["checks"]["completeness_subject"]["value"] == 0.80


# 3
def test_completeness_priority_passes(run_validate):
    rows = [_base_row() for _ in range(20)]
    rows[0]["priority"] = None  # 19/20 = 0.95 >= 0.80
    metrics, _ = run_validate(rows)
    assert metrics["checks"]["completeness_priority"]["passed"] is True


# 4
def test_uniqueness_hard_fail(run_validate):
    rows = [_base_row() for _ in range(10)]
    shared = _uuid()
    for r in rows[:3]:  # 3 rows share one id -> dup_count = 2 > threshold (1)
        r["ticket_id"] = shared
    with pytest.raises(SystemExit) as exc:
        run_validate(rows)
    assert exc.value.code == 1


# 5
def test_uniqueness_passes(run_validate):
    rows = [_base_row() for _ in range(10)]  # all unique ids
    metrics, _ = run_validate(rows)
    assert metrics["checks"]["uniqueness_ticket_id"]["passed"] is True
    assert metrics["checks"]["uniqueness_ticket_id"]["value"] == 0


# 6
def test_quality_flags_empty_string_for_clean(run_validate):
    _, out = run_validate([_base_row()])
    assert out["quality_flags"].iloc[0] == ""


# 7
def test_quality_flags_missing_subject(run_validate):
    _, out = run_validate([_base_row(subject=None)])
    assert "missing_subject" in out["quality_flags"].iloc[0]


# 8
def test_quality_flags_multiple(run_validate):
    _, out = run_validate([_base_row(subject=None, priority=None)])
    flags = out["quality_flags"].iloc[0]
    assert "missing_subject" in flags
    assert "missing_priority" in flags
    assert "|" in flags  # pipe-separated when multiple


# 9
def test_short_body_flagged(run_validate):
    _, out = run_validate([_base_row(body="hello")])  # 5 chars < 10
    assert "short_body" in out["quality_flags"].iloc[0]


# 10
def test_future_created_at_flagged(run_validate):
    _, out = run_validate([_base_row(created_at="2099-01-01T00:00:00+00:00")])
    assert "future_created_at" in out["quality_flags"].iloc[0]


# 11
def test_closed_without_agent_flagged(run_validate):
    _, out = run_validate([_base_row(status="closed", agent_id=None)])
    assert "closed_without_agent" in out["quality_flags"].iloc[0]


# 12
def test_rejected_parquet_written(run_validate, paths):
    run_validate([_base_row(subject=None), _base_row()])
    assert os.path.exists(paths["rejected"])


# 13
def test_rejected_parquet_subset(run_validate, paths):
    rows = [
        _base_row(),  # clean
        _base_row(subject=None),  # flagged
        _base_row(),  # clean
        _base_row(body="tiny"),  # flagged (short_body)
    ]
    run_validate(rows)
    rejected = pd.read_parquet(paths["rejected"])
    assert len(rejected) == 2
    assert (rejected["quality_flags"] != "").all()


# 14
def test_report_json_structure(run_validate, paths):
    run_validate([_base_row()])
    with open(paths["report"], encoding="utf-8") as fh:
        report = json.load(fh)
    expected = {
        "completeness_subject",
        "completeness_priority",
        "validity_channel",
        "validity_status",
        "uniqueness_ticket_id",
        "validity_created_at",
        "short_body",
        "future_created_at",
        "closed_without_agent",
    }
    assert set(report["checks"].keys()) == expected
    assert len(report["checks"]) == 9


# 15
def test_trend_added_when_previous_exists(run_validate, paths):
    rows = [_base_row() for _ in range(5)]
    run_validate(rows)  # first run writes quality_report_previous.json
    assert os.path.exists(paths["previous"])
    run_validate(rows)  # second run sees the previous report
    with open(paths["report"], encoding="utf-8") as fh:
        report = json.load(fh)
    for name, check in report["checks"].items():
        assert "trend" in check, f"missing trend for {name}"


# 16
def test_idempotency(run_validate, paths):
    rows = [
        _base_row(subject=None),
        _base_row(body="x"),
        _base_row(status="closed", agent_id=None),
        _base_row(),
    ]
    _, first = run_validate(rows)
    first_flags = first["quality_flags"].tolist()
    first_rejected = pd.read_parquet(paths["rejected"]).reset_index(drop=True)

    _, second = run_validate(rows)
    second_flags = second["quality_flags"].tolist()
    second_rejected = pd.read_parquet(paths["rejected"]).reset_index(drop=True)

    assert first_flags == second_flags
    pd.testing.assert_frame_equal(first_rejected, second_rejected)


# --- Additional coverage ---------------------------------------------------


# 17
def test_validate_importable():
    from src.validate import validate as v  # noqa: F401

    assert callable(v)


# 18
def test_html_report_written_and_colored(run_validate, paths):
    # One clean row (green) plus a hard-fail-free flagged row keeps it simple;
    # force a failing check via a null subject so a red row appears too.
    rows = [_base_row()] + [_base_row(subject=None) for _ in range(2)]
    run_validate(rows)
    assert os.path.exists(paths["html"])
    html = open(paths["html"], encoding="utf-8").read()
    assert "<table" in html
    assert "Check Name" in html
    assert "background:#ffe0e0" in html  # at least one failed (red) row
    assert "background:#e0ffe0" in html  # at least one passed (green) row
    assert "Report generated:" in html
    assert "Total records: 3" in html


# 19
def test_flags_only_contain_known_names(run_validate):
    rows = [
        _base_row(subject=None, priority=None),
        _base_row(body="hi", status="closed", agent_id=None),
        _base_row(created_at="2099-06-01T00:00:00+00:00"),
        _base_row(),
    ]
    _, out = run_validate(rows)
    known = set(FLAG_NAMES)
    for value in out["quality_flags"]:
        if value == "":
            continue
        for token in value.split("|"):
            assert token in known


# 20
def test_html_trend_grey_dash_without_previous(run_validate, paths):
    # No previous report on first run -> trends render as grey "-".
    run_validate([_base_row()])
    html = open(paths["html"], encoding="utf-8").read()
    assert "color:grey" in html
    assert ">-<" in html


# 21
def test_returns_metrics_dict(run_validate):
    metrics, _ = run_validate([_base_row()])
    assert metrics["total_records"] == 1
    assert "generated_at" in metrics
    assert isinstance(metrics["checks"], dict)


# 22
def test_short_body_check_count(run_validate):
    rows = [_base_row(body="short") for _ in range(3)] + [_base_row() for _ in range(7)]
    metrics, _ = run_validate(rows)
    assert metrics["checks"]["short_body"]["value"] == 3
