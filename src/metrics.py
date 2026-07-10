"""
Per-run metrics history: append one JSON line per pipeline run to
data/metrics/run_history.jsonl.
"""

import json
import os
from datetime import datetime, timezone

METRICS_DIR = "data/metrics"
HISTORY_PATH = os.path.join(METRICS_DIR, "run_history.jsonl")

_QUALITY_REPORT_PATH = "data/quality/quality_report.json"
_ANOMALY_REPORT_PATH = "data/anomalies/anomaly_report.json"
_AGENT_SUMMARY_PATH = "data/agent/agent_summary.json"


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_run_metrics(pipeline_result: dict) -> None:
    """
    Append metrics from a pipeline run to data/metrics/run_history.jsonl.
    Creates the directory and file if they don't exist.
    """
    steps = pipeline_result.get("steps") or {}

    quality_checks = _read_json(_QUALITY_REPORT_PATH).get("checks") or {}
    quality_checks_failed = sum(
        1 for check in quality_checks.values() if not check.get("passed", True)
    )

    entry = {
        "run_id": pipeline_result.get("run_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_status": pipeline_result.get("pipeline_status"),
        "total_duration_ms": pipeline_result.get("total_duration_ms"),
        "records_ingested": steps.get("Ingest", {}).get("records", 0),
        "records_validated": steps.get("Validate", {}).get("records", 0),
        "quality_checks_failed": quality_checks_failed,
        "anomalies_found": _read_json(_ANOMALY_REPORT_PATH).get("anomalies_found", 0),
        "agent_actions_taken": _read_json(_AGENT_SUMMARY_PATH).get("actions_taken", 0),
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_run_summary(last_n: int = 10) -> list[dict]:
    """Return the last N run records from run_history.jsonl."""
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records[-last_n:]
