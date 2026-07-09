"""
Transform layer: read raw ticket Parquet, normalize, write normalized Parquet.
"""

import json
import os
import shutil
from datetime import datetime, timezone

import pandas as pd

from src.logger import get_logger
from src.schema import (
    EXPECTED_COLUMNS,
    NORMALIZED_SCHEMA_VERSION,
    UUID_V4_PATTERN,
    VALID_CATEGORIES,
    VALID_CHANNELS,
    VALID_PRIORITIES,
    VALID_STATUSES,
)

logger = get_logger("Transform")

# HTML tag pattern for Rule 8.
_HTML_TAG_PATTERN = r"<[^>]+>"


def _normalize_enum(series: pd.Series, valid: set, column: str, strict: bool):
    """Strip + lowercase an enum column.

    Returns (normalized_series, valid_mask). In strict mode, raises ValueError
    if any non-null value is outside the valid set. Null values are treated as
    missing (never an "invalid enum value") so they do not trigger strict mode.
    """
    normalized = series.str.strip().str.lower()
    valid_mask = normalized.isin(valid)
    invalid_mask = normalized.notna() & ~valid_mask
    if strict and bool(invalid_mask.any()):
        bad_values = sorted(normalized[invalid_mask].unique().tolist())
        raise ValueError(
            f"strict mode: invalid {column} value(s) {bad_values} "
            f"(allowed: {sorted(valid)})"
        )
    return normalized, valid_mask


def _apply_rules(df: pd.DataFrame, processed_at: str, strict: bool) -> pd.DataFrame:
    """
    Apply normalization Rules 1-9 to a DataFrame slice and append metadata.
    Operates on a copy and returns a fully normalized frame with all 16 columns.
    The same row count goes in and comes out (no rows dropped).
    """
    df = df.copy()

    # Rule 1 — Parse created_at to UTC-aware datetime; flag unparseable rows.
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True, format="mixed")
    df["parse_error"] = df["created_at"].isna()

    # Rule 2 — Normalize channel (invalid -> 'unknown').
    channel, channel_valid = _normalize_enum(
        df["channel"], VALID_CHANNELS, "channel", strict
    )
    df["channel"] = channel.where(channel_valid, "unknown")

    # Rule 3 — Normalize priority (invalid/empty -> null); flag whether kept.
    priority, priority_valid = _normalize_enum(
        df["priority"], VALID_PRIORITIES, "priority", strict
    )
    df["priority"] = priority.where(priority_valid, None)
    df["priority_normalized"] = priority_valid

    # Rule 4 — Normalize status (invalid -> 'unknown').
    status, status_valid = _normalize_enum(
        df["status"], VALID_STATUSES, "status", strict
    )
    df["status"] = status.where(status_valid, "unknown")

    # Rule 5 — Normalize category (invalid -> 'unknown').
    category, category_valid = _normalize_enum(
        df["category"], VALID_CATEGORIES, "category", strict
    )
    df["category"] = category.where(category_valid, "unknown")

    # Rule 6 — Clean subject and body: strip whitespace, empty string -> null.
    for col in ("subject", "body"):
        cleaned = df[col].str.strip()
        df[col] = cleaned.where(cleaned != "", None)

    # Rule 7 — Clean id columns: strip whitespace only (values preserved).
    for col in ("ticket_id", "customer_id", "agent_id"):
        df[col] = df[col].str.strip()

    # Rule 8 — Strip HTML tags from body (vectorized); flag where a tag removed.
    body = df["body"]
    body_not_null = body.notna()
    stripped_body = body.str.replace(_HTML_TAG_PATTERN, "", regex=True)
    df["html_stripped"] = (
        (body_not_null & (stripped_body != body)).fillna(False).astype(bool)
    )
    df["body"] = stripped_body  # str.replace preserves NaN for null bodies.

    # Rule 9 — Validate ticket_id UUID v4 format; flag only, never modify value.
    matches = df["ticket_id"].str.fullmatch(UUID_V4_PATTERN, case=False, na=False)
    df["invalid_ticket_id"] = ~matches.astype(bool)

    # Metadata columns.
    df["_schema_version"] = NORMALIZED_SCHEMA_VERSION
    df["_processed_at"] = processed_at

    return df


def _write_partitioned(df: pd.DataFrame, partition_dir: str) -> int:
    """Write a channel-partitioned copy of the output. Returns partition count."""
    if os.path.isdir(partition_dir):
        shutil.rmtree(partition_dir)
    os.makedirs(partition_dir, exist_ok=True)
    df.to_parquet(partition_dir, partition_cols=["channel"], index=False)
    return int(df["channel"].nunique())


def _write_stats(df: pd.DataFrame, stats_path: str, processed_at: str) -> None:
    """Write transform_stats.json with all required aggregate keys."""
    stats = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "processed_at": processed_at,
        "total_records": int(len(df)),
        "parse_errors": int(df["parse_error"].sum()),
        "html_stripped": int(df["html_stripped"].sum()),
        "invalid_ticket_ids": int(df["invalid_ticket_id"].sum()),
        "priority_nulled": int((~df["priority_normalized"]).sum()),
        "channel_distribution": {
            str(k): int(v) for k, v in df["channel"].value_counts().items()
        },
        "status_distribution": {
            str(k): int(v) for k, v in df["status"].value_counts().items()
        },
        "null_counts": {
            "subject": int(df["subject"].isna().sum()),
            "body": int(df["body"].isna().sum()),
            "priority": int(df["priority"].isna().sum()),
            "agent_id": int(df["agent_id"].isna().sum()),
        },
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)


def transform(
    input_path: str,
    output_path: str,
    chunk_size: int = None,
    strict: bool = False,
) -> int:
    """
    Read raw Parquet, apply all normalization rules, write normalized Parquet.
    chunk_size: if set, process in chunks of this many rows (memory-efficient mode).
    strict: if True, raise ValueError on any invalid enum value instead of replacing with 'unknown'.
    Returns the number of records in the output.
    """
    start_time = datetime.now(timezone.utc)
    processed_at = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("transform_start", extra={"source": input_path, "start": start_time.isoformat()})

    df = pd.read_parquet(input_path)
    records_read = len(df)
    logger.info("transform_records_read", extra={"records_read": records_read})

    # Schema Enforcement — verify all 10 expected columns are present.
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing expected column(s): {missing}. "
            f"Expected columns: {EXPECTED_COLUMNS}"
        )

    if chunk_size:
        # Parquet has no native row streaming, so the frame is read whole but
        # processed in independent slices to bound peak per-slice work.
        total_chunks = max(1, (records_read + chunk_size - 1) // chunk_size)
        processed_chunks = []
        for chunk_index, start in enumerate(range(0, records_read, chunk_size), 1):
            end = min(start + chunk_size, records_read)
            rows = end - start
            logger.info("transform_chunk", extra={"chunk": chunk_index, "total_chunks": total_chunks, "rows": rows})
            slice_df = df.iloc[start:end]
            processed_chunks.append(_apply_rules(slice_df, processed_at, strict))
        result = (
            pd.concat(processed_chunks, ignore_index=True)
            if processed_chunks
            else _apply_rules(df, processed_at, strict)
        )
    else:
        result = _apply_rules(df, processed_at, strict).reset_index(drop=True)

    # Main normalized output.
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_parquet(output_path, index=False)

    # Partitioned output (data/processed/partitioned/channel=<value>/...).
    base_dir = out_dir if out_dir else "."
    partition_dir = os.path.join(base_dir, "partitioned")
    partition_count = _write_partitioned(result, partition_dir)
    logger.info("transform_partitions_written", extra={"partition_count": partition_count, "partition_dir": partition_dir})

    # Transform statistics.
    stats_path = os.path.join(base_dir, "transform_stats.json")
    _write_stats(result, stats_path, processed_at)
    logger.info("transform_stats_written", extra={"stats_path": stats_path})

    records_written = len(result)
    end_time = datetime.now(timezone.utc)
    logger.info("transform_complete", extra={
        "records_written": records_written,
        "parse_error_rows": int(result["parse_error"].sum()),
        "end": end_time.isoformat(),
    })

    return records_written


if __name__ == "__main__":
    inp = os.environ.get("TRANSFORM_INPUT", "data/raw/tickets.parquet")
    out = os.environ.get(
        "TRANSFORM_OUTPUT", "data/processed/tickets_normalized.parquet"
    )
    count = transform(inp, out)
    print(f"Transform complete: {count} records -> {out}")
