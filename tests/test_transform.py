"""Pytest suite for the production transform layer.

Tests build raw DataFrames in memory, write them to a temporary Parquet file,
and run the real transform() against tmp_path — no cloud source is touched.
"""

import json
import uuid

import pandas as pd
import pytest

from src.schema import (
    EXPECTED_COLUMNS,
    NORMALIZED_SCHEMA_VERSION,
    VALID_CHANNELS,
)
from src.transform import transform

# A real, well-formed UUID v4 used wherever a valid ticket_id is needed.
VALID_UUID = "9f8b1c2d-3e4f-4a5b-8c6d-7e8f9a0b1c2d"

OUTPUT_COLUMNS = EXPECTED_COLUMNS + [
    "parse_error",
    "priority_normalized",
    "html_stripped",
    "invalid_ticket_id",
    "_schema_version",
    "_processed_at",
]


def _base_row(**overrides):
    """A fully valid ticket row; override individual fields per test."""
    row = {
        "ticket_id": VALID_UUID,
        "created_at": "2024-01-15T10:30:00",
        "customer_id": str(uuid.uuid4()),
        "channel": "email",
        "subject": "Need help with billing",
        "body": "Please assist me with my invoice.",
        "priority": "high",
        "category": "billing",
        "status": "open",
        "agent_id": str(uuid.uuid4()),
    }
    row.update(overrides)
    return row


@pytest.fixture
def make_input(tmp_path):
    """Factory: write rows to a raw Parquet and return its path."""

    def _make(rows, name="raw.parquet", columns=None):
        df = pd.DataFrame(rows, columns=columns)
        path = tmp_path / name
        df.to_parquet(path, index=False)
        return str(path)

    return _make


@pytest.fixture
def out_path(tmp_path):
    return str(tmp_path / "processed" / "tickets_normalized.parquet")


def _run(make_input, out_path, rows, **kwargs):
    """Transform `rows` and return the resulting output DataFrame."""
    columns = kwargs.pop("columns", None)
    input_path = make_input(rows, columns=columns)
    transform(input_path, out_path, **kwargs)
    return pd.read_parquet(out_path)


# 1
def test_schema_enforcement_missing_column(make_input, out_path):
    rows = [_base_row()]
    columns = [c for c in EXPECTED_COLUMNS if c != "priority"]
    input_path = make_input(
        [{k: v for k, v in rows[0].items() if k != "priority"}],
        columns=columns,
    )
    with pytest.raises(ValueError):
        transform(input_path, out_path)


# 2
def test_created_at_valid_parsed(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(created_at="2024-01-15T10:30:00")])
    assert pd.api.types.is_datetime64_any_dtype(out["created_at"])
    assert out["parse_error"].iloc[0] == False  # noqa: E712


# 3
def test_created_at_malformed_becomes_nat(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(created_at="not-a-date")])
    assert pd.isna(out["created_at"].iloc[0])
    assert out["parse_error"].iloc[0] == True  # noqa: E712


# 4
def test_created_at_utc_normalized(make_input, out_path):
    out = _run(
        make_input,
        out_path,
        [
            _base_row(created_at="2024-01-15T10:30:00"),
            _base_row(created_at="2024-02-20T08:00:00+05:00"),
        ],
    )
    assert str(out["created_at"].dt.tz) == "UTC"


# 5
def test_channel_lowercased(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(channel="Email")])
    assert out["channel"].iloc[0] == "email"


# 6
def test_channel_invalid_becomes_unknown(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(channel="fax")])
    assert out["channel"].iloc[0] == "unknown"


# 7
def test_priority_null_preserved(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(priority=None)])
    assert pd.isna(out["priority"].iloc[0])
    assert out["priority_normalized"].iloc[0] == False  # noqa: E712


# 8
def test_subject_empty_string_becomes_null(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(subject="")])
    assert pd.isna(out["subject"].iloc[0])


# 9
def test_whitespace_stripped(make_input, out_path):
    out = _run(
        make_input,
        out_path,
        [_base_row(subject="  hello  ", customer_id="  cust-1  ")],
    )
    assert out["subject"].iloc[0] == "hello"
    assert out["customer_id"].iloc[0] == "cust-1"


# 10
def test_html_stripped_from_body(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(body="<b>Help</b>")])
    assert out["body"].iloc[0] == "Help"
    assert out["html_stripped"].iloc[0] == True  # noqa: E712


# 11
def test_no_html_flag_false(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(body="Plain text body")])
    assert out["body"].iloc[0] == "Plain text body"
    assert out["html_stripped"].iloc[0] == False  # noqa: E712


# 12
def test_valid_uuid_not_flagged(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(ticket_id=VALID_UUID)])
    assert out["invalid_ticket_id"].iloc[0] == False  # noqa: E712
    assert out["ticket_id"].iloc[0] == VALID_UUID  # value never modified


# 13
def test_invalid_uuid_flagged(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(ticket_id="not-a-uuid")])
    assert out["invalid_ticket_id"].iloc[0] == True  # noqa: E712
    assert out["ticket_id"].iloc[0] == "not-a-uuid"  # value preserved


# 14
def test_output_has_16_columns(make_input, out_path):
    out = _run(make_input, out_path, [_base_row()])
    assert out.shape[1] == 16
    assert set(out.columns) == set(OUTPUT_COLUMNS)


# 15
def test_schema_version_column(make_input, out_path):
    out = _run(make_input, out_path, [_base_row(), _base_row()])
    assert "_schema_version" in out.columns
    assert (out["_schema_version"] == NORMALIZED_SCHEMA_VERSION).all()


# 16
def test_processed_at_column(make_input, out_path):
    out = _run(make_input, out_path, [_base_row()])
    value = out["_processed_at"].iloc[0]
    # Must parse as a valid ISO-8601 timestamp.
    parsed = pd.to_datetime(value)
    assert not pd.isna(parsed)


# 17
def test_idempotency(make_input, out_path):
    rows = [
        _base_row(channel="Email", body="<i>hi</i>", priority="bogus"),
        _base_row(ticket_id="bad-id", created_at="nope"),
    ]
    input_path = make_input(rows)
    transform(input_path, out_path)
    first = pd.read_parquet(out_path)
    transform(input_path, out_path)
    second = pd.read_parquet(out_path)
    # _processed_at is wall-clock and intentionally differs between runs.
    drop = ["_processed_at"]
    pd.testing.assert_frame_equal(first.drop(columns=drop), second.drop(columns=drop))


# 18
def test_chunk_size_same_result(make_input, out_path):
    rows = [
        _base_row(
            ticket_id=VALID_UUID if i % 2 == 0 else "bad-id",
            channel=["email", "Chat", "fax", "phone"][i % 4],
            body="<b>x</b>" if i % 3 == 0 else "plain",
            priority=[None, "high", "bogus", "low"][i % 4],
            created_at="2024-01-15T10:30:00" if i % 5 else "broken",
        )
        for i in range(25)
    ]
    input_path = make_input(rows)

    transform(input_path, out_path)
    no_chunk = pd.read_parquet(out_path)

    transform(input_path, out_path, chunk_size=10)
    chunked = pd.read_parquet(out_path)

    drop = ["_processed_at"]
    pd.testing.assert_frame_equal(
        no_chunk.drop(columns=drop), chunked.drop(columns=drop)
    )


# --- Additional production-standard coverage -------------------------------


# 19
def test_strict_mode_raises_on_invalid_channel(make_input, out_path):
    input_path = make_input([_base_row(channel="fax")])
    with pytest.raises(ValueError):
        transform(input_path, out_path, strict=True)


# 20
def test_no_rows_dropped(make_input, out_path):
    rows = [_base_row(ticket_id="bad", created_at="x", channel="fax") for _ in range(7)]
    returned = transform(make_input(rows), out_path)
    out = pd.read_parquet(out_path)
    assert returned == 7
    assert len(out) == 7


# 21
def test_partitioned_output_created(make_input, out_path, tmp_path):
    rows = [_base_row(channel="email"), _base_row(channel="chat")]
    transform(make_input(rows), out_path)
    partition_root = tmp_path / "processed" / "partitioned"
    assert (partition_root / "channel=email").is_dir()
    assert (partition_root / "channel=chat").is_dir()


# 22
def test_transform_stats_written(make_input, out_path, tmp_path):
    rows = [
        _base_row(),
        _base_row(priority=None),
        _base_row(created_at="broken"),
        _base_row(body="<b>x</b>"),
    ]
    transform(make_input(rows), out_path)
    stats_path = tmp_path / "processed" / "transform_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    for key in (
        "schema_version",
        "processed_at",
        "total_records",
        "parse_errors",
        "html_stripped",
        "invalid_ticket_ids",
        "priority_nulled",
        "channel_distribution",
        "status_distribution",
        "null_counts",
    ):
        assert key in stats
    assert stats["schema_version"] == NORMALIZED_SCHEMA_VERSION
    assert stats["total_records"] == 4
    assert stats["parse_errors"] == 1
    assert stats["priority_nulled"] == 1


# 23
def test_valid_channels_imported_from_schema():
    # Guards that the central schema module is the source of truth.
    assert VALID_CHANNELS == {"email", "chat", "phone", "web"}
