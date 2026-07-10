"""
Advanced validation layer: read normalized ticket Parquet, run nine quality
checks (six standard + three additional), annotate every record with a
``quality_flags`` column, write a structured JSON quality report and an HTML
quality report, route flagged records to a dead-letter queue, compare against
the previous report (historical trend), and hard-fail the pipeline if duplicate
ticket_ids exceed their threshold.

Every check is vectorized over the full DataFrame. The Pydantic ``Ticket``
model documents the per-record contract of the normalized schema; it is NOT
used for the aggregate checks (per the task requirements).

Path convention: the dead-letter queue, the HTML report, and the previous-run
report all live alongside ``report_path`` (same directory). With the default
``report_path`` of ``data/quality/quality_report.json`` this yields exactly the
spec paths ``data/quality/rejected.parquet``,
``data/quality/quality_report.html`` and
``data/quality/quality_report_previous.json``.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from src.schema import VALID_CHANNELS, VALID_STATUSES
from src.config import STRICT_MODE

# The normalized data maps invalid enum values to "unknown", so the validation
# sets here extend the central schema sets with "unknown" (an expected value
# in normalized data, not a quality failure).
VALID_CHANNEL_VALUES = VALID_CHANNELS | {"unknown"}
VALID_STATUS_VALUES = VALID_STATUSES | {"unknown"}

# All flag names that may appear in the quality_flags column, in emission order.
FLAG_NAMES = (
    "missing_subject",
    "missing_priority",
    "invalid_channel",
    "invalid_status",
    "null_created_at",
    "short_body",
    "future_created_at",
    "closed_without_agent",
)

_HTML_TEMPLATE = (
    "<html><head><title>Quality Report</title></head><body>\n"
    "<p>{summary}</p>\n"
    '<table border="1" cellpadding="4" cellspacing="0">\n'
    "<tr><th>Check Name</th><th>Passed</th><th>Value</th>"
    "<th>Threshold</th><th>Trend</th></tr>\n"
    "{rows}\n"
    "</table>\n"
    "</body></html>\n"
)

_HTML_ROW = (
    '<tr style="background:{bg}">'
    "<td>{name}</td><td>{passed}</td><td>{value}</td><td>{threshold}</td>"
    '<td style="color:{trend_color}">{trend}</td></tr>'
)


class Ticket(BaseModel):
    """Per-record contract for a normalized ticket."""

    ticket_id: str
    created_at: Optional[datetime] = None
    customer_id: str
    channel: str
    subject: Optional[str] = None
    body: Optional[str] = None
    priority: Optional[str] = None
    category: Optional[str] = None
    status: str
    agent_id: Optional[str] = None

    @field_validator("channel")
    @classmethod
    def _validate_channel(cls, v: str) -> str:
        if v not in VALID_CHANNEL_VALUES:
            raise ValueError(
                f"invalid channel {v!r} (allowed: {sorted(VALID_CHANNEL_VALUES)})"
            )
        return v

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in VALID_STATUS_VALUES:
            raise ValueError(
                f"invalid status {v!r} (allowed: {sorted(VALID_STATUS_VALUES)})"
            )
        return v


def _row_condition_masks(df: pd.DataFrame, now: pd.Timestamp):
    """Ordered list of (flag_name, boolean Series) for every per-row check.

    ``df["created_at"]`` is expected to already be a UTC-aware datetime column
    (the caller coerces it). All masks are fully vectorized.
    """
    body_len = df["body"].str.len()
    return [
        ("missing_subject", df["subject"].isna()),
        ("missing_priority", df["priority"].isna()),
        ("invalid_channel", ~df["channel"].isin(VALID_CHANNEL_VALUES)),
        ("invalid_status", ~df["status"].isin(VALID_STATUS_VALUES)),
        ("null_created_at", df["created_at"].isna()),
        ("short_body", df["body"].notna() & (body_len < 10)),
        ("future_created_at", df["created_at"].notna() & (df["created_at"] > now)),
        ("closed_without_agent", (df["status"] == "closed") & df["agent_id"].isna()),
    ]


def _compute_quality_flags(mask_list, index: pd.Index) -> pd.Series:
    """Build the pipe-separated per-row flag Series (vectorized).

    Initialized strictly with "" and dtype object; clean rows keep "" (never
    null). Flags are concatenated with "|" using numpy element-wise ops — no
    per-row Python loop.
    """
    flags = pd.Series("", index=index, dtype="object")
    for name, mask in mask_list:
        m = mask.fillna(False).to_numpy(dtype=bool)
        current = flags.to_numpy()
        sep = np.where((current != "") & m, "|", "").astype(object)
        add = np.where(m, name, "").astype(object)
        flags = pd.Series(current + sep + add, index=index, dtype="object")
    return flags


def _rate_check(name: str, mask: pd.Series, threshold: float, total: int) -> dict:
    """A WARN-gated completeness/validity check over a fraction of rows."""
    value = float(mask.mean()) if total else 1.0
    passed = value >= threshold
    if not passed:
        logging.warning(
            f"Quality check {name} failed: {value} (threshold {threshold})"
        )
    return {"passed": bool(passed), "value": round(value, 4), "threshold": threshold}


def _count_check(name: str, value: int, threshold: int) -> dict:
    """A WARN-gated count check that fails when value EXCEEDS the threshold."""
    passed = value <= threshold
    if not passed:
        logging.warning(
            f"Quality check {name} failed: {value} (threshold {threshold})"
        )
    return {"passed": bool(passed), "value": int(value), "threshold": int(threshold)}


def _format_trend(trend) -> str:
    """Render a trend delta with an explicit sign (ints as ints, floats x.4)."""
    if isinstance(trend, bool):  # guard: bool is an int subclass
        trend = int(trend)
    if isinstance(trend, float):
        return "{:+.4f}".format(trend)
    return "{:+d}".format(int(trend))


def _write_html_report(report: dict, html_path: str) -> None:
    """Write an HTML quality report using only str.format() (no templating lib)."""
    checks = report["checks"]
    failed = sum(1 for c in checks.values() if not c["passed"])
    summary = (
        "Report generated: {timestamp} | Total records: {n} | "
        "Checks failed: {k}"
    ).format(timestamp=report["generated_at"], n=report["total_records"], k=failed)

    rows = []
    for name, c in checks.items():
        bg = "#e0ffe0" if c["passed"] else "#ffe0e0"
        trend = c.get("trend", None)
        if trend is None:
            trend_str, trend_color = "-", "grey"
        elif trend > 0:
            trend_str, trend_color = _format_trend(trend), "green"
        elif trend < 0:
            trend_str, trend_color = _format_trend(trend), "red"
        else:
            trend_str, trend_color = _format_trend(trend), "grey"
        rows.append(
            _HTML_ROW.format(
                bg=bg,
                name=name,
                passed=c["passed"],
                value=c["value"],
                threshold=c["threshold"],
                trend=trend_str,
                trend_color=trend_color,
            )
        )

    html = _HTML_TEMPLATE.format(summary=summary, rows="\n".join(rows))
    html_dir = os.path.dirname(html_path)
    if html_dir:
        os.makedirs(html_dir, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def _apply_trends(checks: dict, previous_path: str) -> None:
    """If a previous report exists, add a numeric ``trend`` to each check."""
    if not os.path.exists(previous_path):
        return
    try:
        with open(previous_path, "r", encoding="utf-8") as fh:
            previous = json.load(fh)
    except (json.JSONDecodeError, OSError):
        logging.warning("Could not read previous report at %s; skipping trends", previous_path)
        return
    prev_checks = previous.get("checks", {})
    for name, entry in checks.items():
        prev_entry = prev_checks.get(name)
        if prev_entry is not None and "value" in prev_entry:
            delta = entry["value"] - prev_entry["value"]
            if isinstance(delta, float):
                delta = round(delta, 4)
            entry["trend"] = delta


def validate(input_path: str, output_path: str, report_path: str) -> dict:
    """
    Read normalized ticket Parquet, run nine quality checks, annotate records
    with quality_flags, write the validated Parquet, JSON + HTML reports, and a
    dead-letter queue of flagged records. Compares against the previous report
    (historical trend) when available.

    Returns the quality metrics dict (same structure as the JSON report).
    Calls sys.exit(1) — at the very end, after all Parquet files are written —
    if the duplicate ticket_id check exceeds its threshold.
    """
    df = pd.read_parquet(input_path)
    total = len(df)

    # created_at is parsed/normalized upstream; re-coerce to UTC-aware so the
    # created_at-based checks are robust regardless of input representation.
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    now = pd.Timestamp(datetime.now(timezone.utc))
    mask_list = _row_condition_masks(df, now)
    masks = dict(mask_list)

    checks = {}

    # Check 1 — completeness_subject (WARN if < 0.90).
    checks["completeness_subject"] = _rate_check(
        "completeness_subject", df["subject"].notna(), 0.90, total
    )

    # Check 2 — completeness_priority (WARN if < 0.80).
    checks["completeness_priority"] = _rate_check(
        "completeness_priority", df["priority"].notna(), 0.80, total
    )

    # Check 3 — validity_channel (WARN if < 0.95).
    checks["validity_channel"] = _rate_check(
        "validity_channel", df["channel"].isin(VALID_CHANNEL_VALUES), 0.95, total
    )

    # Check 4 — validity_status (WARN if < 0.95).
    checks["validity_status"] = _rate_check(
        "validity_status", df["status"].isin(VALID_STATUS_VALUES), 0.95, total
    )

    # Check 5 — uniqueness_ticket_id (HARD FAIL if value > threshold).
    distinct = int(df["ticket_id"].nunique(dropna=False))
    dup_count = total - distinct
    dup_threshold = max(1, int(len(df) * 0.05))
    checks["uniqueness_ticket_id"] = {
        "passed": bool(dup_count <= dup_threshold),
        "value": int(dup_count),
        "threshold": int(dup_threshold),
    }

    # Check 6 — validity_created_at (WARN if value > threshold).
    null_created = int(df["created_at"].isna().sum())
    created_threshold = max(1, int(len(df) * 0.10))
    checks["validity_created_at"] = _count_check(
        "validity_created_at", null_created, created_threshold
    )

    # Check 7 — short_body (WARN if value > 1% of records).
    short_body_count = int(masks["short_body"].sum())
    checks["short_body"] = _count_check(
        "short_body", short_body_count, max(1, int(len(df) * 0.01))
    )

    # Check 8 — future_created_at (WARN if any future timestamp).
    future_count = int(masks["future_created_at"].sum())
    checks["future_created_at"] = _count_check(
        "future_created_at", future_count, 0
    )

    # Check 9 — closed_without_agent (WARN if any closed ticket lacks an agent).
    closed_no_agent_count = int(masks["closed_without_agent"].sum())
    checks["closed_without_agent"] = _count_check(
        "closed_without_agent", closed_no_agent_count, int(len(df)*0.05)
    )

    metrics = {
        "total_records": int(total),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }

    # Annotate every row with its failure flags (object dtype, never null).
    df["quality_flags"] = _compute_quality_flags(mask_list, df.index)

    # Historical trend comparison (mutates `checks`/`metrics` in place).
    report_dir = os.path.dirname(report_path) or "."
    previous_path = os.path.join(report_dir, "quality_report_previous.json")
    _apply_trends(checks, previous_path)

    os.makedirs(report_dir, exist_ok=True)

    # Write the JSON quality report.
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Write the HTML quality report alongside it.
    html_path = os.path.splitext(report_path)[0] + ".html"
    _write_html_report(metrics, html_path)

    # Write the validated output Parquet (all rows, including flagged).
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Dead-letter queue: every row carrying at least one quality flag.
    rejected = df[df["quality_flags"] != ""]
    rejected_path = os.path.join(report_dir, "rejected.parquet")
    rejected.to_parquet(rejected_path, index=False)
    logging.info(
        "{n} records written to dead-letter queue at {path}".format(
            n=len(rejected), path=rejected_path
        )
    )

    # Copy the current report to the previous-run report for the next run.
    with open(previous_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Hard gate — at the very end, after all Parquet files are written, so the
    # SystemExit is never swallowed and outputs are always produced first.
    if dup_count > dup_threshold:
        logging.error(
            f"HARD FAIL: {dup_count} duplicate ticket_ids exceed threshold "
            f"{dup_threshold}"
        )
        sys.exit(1)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate(
        "data/processed/tickets_normalized.parquet",
        "data/processed/tickets_validated.parquet",
        "data/quality/quality_report.json",
    )
