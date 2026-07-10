"""
Integration tests: full pipeline (ingest → transform → validate → classify →
enrich → detect → agent) via run_pipeline() on 10 tickets from data/tickets.csv.

The pipeline runs exactly once (module-scoped fixture) inside an isolated
working directory; the six tests below each assert one contract of that run.
Must complete in under 60 seconds.

Requires a real ANTHROPIC_API_KEY (classify/enrich/detect/agent make live LLM
calls); the tests are skipped when the key is not configured (e.g. in CI).
"""

import json
import os
import pathlib
import time
from unittest.mock import patch

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
TICKETS_CSV = PROJECT_ROOT / "data" / "tickets.csv"

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; integration test needs live LLM access",
)

ARTIFACTS = [
    "data/raw/tickets.parquet",
    "data/processed/tickets_normalized.parquet",
    "data/processed/tickets_validated.parquet",
    "data/enriched/tickets_classified.parquet",
    "data/enriched/tickets_enriched.parquet",
    "data/anomalies/anomalies.parquet",
    "data/agent/actions.jsonl",
]


def _count_lines(path) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run the full pipeline once in an isolated cwd; share the result."""
    if not TICKETS_CSV.exists():
        pytest.skip("data/tickets.csv not found")

    tmp_dir = tmp_path_factory.mktemp("pipeline")

    # First 10 rows of the seed CSV. Duplicate one body across two rows: the
    # duplicate_body check in detect is deterministic (no LLM), so this
    # guarantees at least one anomaly and therefore at least one agent action.
    df_sample = pd.read_csv(TICKETS_CSV).head(10)
    df_sample.iloc[1, df_sample.columns.get_loc("body")] = df_sample.iloc[0]["body"]

    def fake_ingest(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sample.to_parquet(output_path, index=False)
        return len(df_sample)

    from src.pipeline import run_pipeline

    old_cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        started = time.monotonic()
        # Ingest is mocked (no cloud I/O) and platform detection is forced to
        # a non-Azure answer so FabricUpload never makes network calls here.
        with patch("src.pipeline.ingest", side_effect=fake_ingest), \
             patch("src.pipeline._detect_platform", return_value="aws"):
            summary = run_pipeline()
        elapsed = time.monotonic() - started
    finally:
        os.chdir(old_cwd)

    return {"dir": tmp_dir, "summary": summary, "elapsed": elapsed}


def test_full_pipeline_10_rows(pipeline_run):
    summary = pipeline_run["summary"]
    assert summary["pipeline_status"] == "ok"
    assert summary["steps"]["Ingest"]["records"] == 10
    expected_steps = {"Ingest", "Transform", "Validate", "Classify", "Enrich", "Detect", "Agent"}
    assert expected_steps.issubset(summary["steps"].keys())
    assert pipeline_run["elapsed"] < 60, (
        f"pipeline took {pipeline_run['elapsed']:.1f}s; must complete in under 60 seconds"
    )


def test_each_artifact_exists(pipeline_run):
    base = pipeline_run["dir"]
    for artifact in ARTIFACTS:
        assert (base / artifact).exists(), f"missing artifact: {artifact}"
    assert _count_lines(base / "data/agent/actions.jsonl") >= 1, (
        "actions.jsonl must have at least 1 line"
    )


def test_transform_row_count(pipeline_run):
    df = pd.read_parquet(pipeline_run["dir"] / "data/processed/tickets_normalized.parquet")
    assert len(df) == 10, "10 rows in must produce 10 rows out of transform"


def test_quality_report_valid_json(pipeline_run):
    report_path = pipeline_run["dir"] / "data/quality/quality_report.json"
    with open(report_path) as f:
        report = json.load(f)
    assert isinstance(report, dict)
    assert "checks" in report


def test_healthcheck_ok_after_run(pipeline_run, monkeypatch):
    from src.healthcheck import run_healthcheck

    monkeypatch.chdir(pipeline_run["dir"])
    monkeypatch.delenv("FABRIC_WORKSPACE_ID", raising=False)
    result = run_healthcheck()
    assert result["status"] == "ok", f"unhealthy checks: {result['checks']}"


def test_metrics_recorded(pipeline_run):
    history_path = pipeline_run["dir"] / "data/metrics/run_history.jsonl"
    assert history_path.exists(), "run_history.jsonl must be written by run_pipeline"
    with open(history_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    assert len(entries) == 1, "exactly one new entry per pipeline run"
    entry = entries[0]
    assert entry["run_id"] == pipeline_run["summary"]["run_id"]
    assert entry["pipeline_status"] == "ok"
    assert entry["records_ingested"] == 10
    assert entry["records_validated"] == 10
