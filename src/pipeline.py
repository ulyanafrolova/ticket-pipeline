"""
Orchestration runner: wire ingest -> transform -> validate into one pipeline.

Each step is timed with ``time.perf_counter`` and logged as JSON via get_logger.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.agent import run_agent
from src.classify import classify
from src.config import PATHS, PROFILE
from src.detect_anomalies import detect
from src.enrich import enrich
from src.ingestion import _detect_platform, ingest
from src.logger import get_logger
from src.metrics import record_run_metrics
from src.transform import transform
from src.validate import validate

logger = get_logger("Pipeline")

# Valid values for the --step flag, in execution order.
STEP_CHOICES = ("ingest", "transform", "validate", "classify", "enrich", "detect", "agent")


def _now_iso() -> str:
    """Current UTC time as 'YYYY-MM-DDTHH:MM:SSZ'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_count(result) -> int:
    """Steps return either an int record count or a metrics dict with varying keys."""
    if isinstance(result, dict):
        for key in ("total_records", "total", "total_tickets", "anomalies_processed"):
            if key in result:
                return int(result[key])
        return 0
    return int(result)


def _build_steps():
    """
    Return the ordered step table as a list of
    ``(name, required_input_path, callable)`` tuples. ``required_input_path``
    is None when the step needs no pre-existing input artifact (ingest).

    All paths are read from PATHS so nothing here is hardcoded.
    """
    return [
        ("Ingest", None, lambda: ingest(PATHS["raw"])),
        (
            "Transform",
            PATHS["raw"],
            lambda: transform(PATHS["raw"], PATHS["normalized"]),
        ),
        (
            "Validate",
            PATHS["normalized"],
            lambda: validate(
                PATHS["normalized"], PATHS["validated"], PATHS["quality_report"]
            ),
        ),
        (
            "Classify",
            PATHS["validated"],
            lambda: classify(PATHS["validated"], PATHS["classified"]),
        ),
        (
            "Enrich",
            PATHS["classified"],
            lambda: enrich(PATHS["classified"], PATHS["enriched"]),
        ),
        (
            "Detect",
            PATHS["enriched"],
            lambda: detect(PATHS["enriched"], PATHS["anomalies"], PATHS["anomaly_report"]),
        ),
        (
            "Agent",
            PATHS["anomalies"],
            lambda: run_agent(PATHS["anomalies"]),
        ),
    ]


def _select_steps(step):
    """Return the full step table, or a one-element slice when ``step`` is set."""
    steps = _build_steps()
    if step is None:
        return steps
    index = STEP_CHOICES.index(step)  # ValueError-proof: argparse restricts choices
    return [steps[index]]


def _fabric_upload_all() -> int:
    """Upload pipeline artifacts to Fabric Lakehouse. Returns count of files uploaded."""
    from src.load_to_fabric import load_to_fabric

    files = [
        ("data/enriched/tickets_enriched.parquet", "Files/processed/tickets_enriched.parquet"),
        ("data/quality/quality_report.json", "Files/quality/quality_report.json"),
        ("data/anomalies/anomaly_report.json", "Files/anomalies/anomaly_report.json"),
    ]
    uploaded = 0
    for local_path, blob_name in files:
        load_to_fabric(local_path, blob_name=blob_name)
        uploaded += 1
    return uploaded


def _storage_platform() -> str:
    """_detect_platform, but 'none' instead of raising when unconfigured."""
    try:
        return _detect_platform()
    except EnvironmentError:
        return "none"


def _run_step(name: str, func) -> dict:
    """
    Run one pipeline step: log start, time it with perf_counter, log completion
    with duration and record count, and return a per-step result dict. Any
    exception propagates to the caller (run_pipeline handles state + exit).
    """
    logger.info("step_start", extra={"step": name})
    start = time.perf_counter()
    result = func()
    duration_ms = int(round((time.perf_counter() - start) * 1000))
    records = _record_count(result)
    logger.info("step_complete", extra={"step": name, "duration_ms": duration_ms, "records": records})
    return {"status": "ok", "records": records, "duration_ms": duration_ms}


def _write_state(state: dict) -> None:
    """Write the pipeline state file to PATHS["pipeline_state"]."""
    path = PATHS["pipeline_state"]
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    logger.info("pipeline_state_written", extra={"path": path})


def _record_metrics_safe(pipeline_result: dict) -> None:
    """Append run metrics; a metrics failure must never break the pipeline."""
    try:
        record_run_metrics(pipeline_result)
    except Exception as exc:
        logger.warning("metrics_record_failed", extra={"error": str(exc)})


def _print_dry_run() -> None:
    """Log what each step would do, executing nothing and writing nothing."""
    logger.info("dry_run_plan", extra={
        "step1": f"Ingest -> {PATHS['raw']}",
        "step2": f"Transform -> {PATHS['normalized']}",
        "step3": f"Validate -> {PATHS['validated']}",
        "step4": f"Classify -> {PATHS['classified']}",
        "step5": f"Enrich -> {PATHS['enriched']}",
        "step6": f"Detect -> {PATHS['anomalies']}",
        "step7": f"Agent -> data/agent/",
        "note": "No steps executed.",
    })


def run_pipeline(dry_run: bool = False, step: str = None) -> dict:
    """
    Run the ticket pipeline: ingest -> transform -> validate -> classify -> enrich -> detect -> agent.

    dry_run: print the plan and return without executing or writing anything.
    step:    if set, run only that named step; its required input artifact must already exist.

    Writes PATHS["pipeline_state"] on every real run (status 'ok' on success).
    On any step failure, writes the state with status 'failed' plus an 'error'
    message and raises SystemExit(1).

    Returns a summary dict with per-step results and ``pipeline_status``.
    """
    if dry_run:
        _print_dry_run()
        return {"steps": {}, "pipeline_status": "dry-run"}

    selected = _select_steps(step)
    run_id = datetime.now(timezone.utc).isoformat()
    started_at = _now_iso()
    get_logger("Pipeline", run_id=run_id)
    total_start = time.perf_counter()
    steps: dict = {}
    completed_steps: list = []
    name = None

    try:
        for name, required_input, func in selected:
            if required_input is not None and not os.path.exists(required_input):
                raise FileNotFoundError(
                    f"Required input for step '{name}' not found: {required_input}"
                )
            steps[name] = _run_step(name, func)
            completed_steps.append(name)

        # Azure only: push key artifacts to the Fabric Lakehouse as the final
        # step of a full pipeline run.
        if step is None and _storage_platform() == "azure":
            steps["FabricUpload"] = _run_step("FabricUpload", _fabric_upload_all)
            completed_steps.append("FabricUpload")
    except (Exception, SystemExit) as exc:
        total_duration_ms = int(round((time.perf_counter() - total_start) * 1000))
        logger.error("step_failed", extra={"step": name or "unknown", "error": str(exc)})
        _write_state(
            {
                "run_id": run_id,
                "profile": PROFILE,
                "status": "failed",
                "started_at": started_at,
                "finished_at": _now_iso(),
                "total_duration_ms": total_duration_ms,
                "steps": steps,
                "error": str(exc),
            }
        )
        _record_metrics_safe(
            {
                "run_id": run_id,
                "steps": steps,
                "total_duration_ms": total_duration_ms,
                "pipeline_status": "failed",
            }
        )
        sys.exit(1)

    total_duration_ms = int(round((time.perf_counter() - total_start) * 1000))
    _write_state(
        {
            "run_id": run_id,
            "profile": PROFILE,
            "status": "ok",
            "started_at": started_at,
            "finished_at": _now_iso(),
            "total_duration_ms": total_duration_ms,
            "steps": steps,
        }
    )

    logger.info("pipeline_complete", extra={
        "total_duration_ms": total_duration_ms,
        "steps_completed": len(completed_steps),
        "pipeline_status": "ok",
    })

    summary = {
        "run_id": run_id,
        "steps": steps,
        "total_duration_ms": total_duration_ms,
        "pipeline_status": "ok",
    }
    _record_metrics_safe(summary)
    return summary


def main(argv=None) -> dict:
    """Parse CLI flags and run the pipeline accordingly."""
    parser = argparse.ArgumentParser(
        description="Ticket pipeline orchestration runner"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what each step would do without executing it.",
    )
    parser.add_argument(
        "--step",
        choices=STEP_CHOICES,
        default=None,
        help="Run only the named step (its input artifact must exist).",
    )
    args = parser.parse_args(argv)

    result = run_pipeline(dry_run=args.dry_run, step=args.step)
    return result


if __name__ == "__main__":
    main()
