"""Tests for src/healthcheck.py — local artifact checks, CLI exit code, and
Fabric Lakehouse connectivity checks (mocked)."""

import json
import os
import pathlib
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from src.healthcheck import EXPECTED_FILES, run_healthcheck

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _create_all_artifacts(base: pathlib.Path) -> None:
    for rel_path, _ in EXPECTED_FILES:
        path = base / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")


@pytest.fixture
def fresh_data(tmp_path, monkeypatch):
    """All 8 expected artifacts freshly created, cwd moved to tmp_path,
    Fabric checks disabled."""
    _create_all_artifacts(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FABRIC_WORKSPACE_ID", raising=False)
    return tmp_path


@pytest.fixture
def fabric_env(fresh_data, monkeypatch):
    """fresh_data plus the environment the Fabric check requires."""
    monkeypatch.setenv("FABRIC_WORKSPACE_ID", "ws-test")
    monkeypatch.setenv("FABRIC_LAKEHOUSE_ID", "lh-test")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-test")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-test")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-test")
    return fresh_data


def test_ok_when_all_files_fresh(fresh_data):
    result = run_healthcheck()
    assert result["status"] == "ok"
    assert all(c["status"] == "ok" for c in result["checks"])


def test_failed_when_critical_missing(fresh_data):
    os.remove(fresh_data / "data" / "raw" / "tickets.parquet")
    result = run_healthcheck()
    assert result["status"] == "failed"


def test_degraded_when_optional_missing(fresh_data):
    os.remove(fresh_data / "data" / "agent" / "actions.jsonl")
    result = run_healthcheck()
    assert result["status"] == "degraded"


def test_degraded_when_stale(fresh_data):
    stale_time = time.time() - 25 * 3600  # older than the 24h default
    stale_path = fresh_data / "data" / "raw" / "tickets.parquet"
    os.utime(stale_path, (stale_time, stale_time))
    result = run_healthcheck()
    assert result["status"] == "degraded"
    stale_check = next(c for c in result["checks"] if "raw" in c["name"])
    assert stale_check["status"] == "degraded"


def test_checks_list_length(fresh_data):
    assert len(run_healthcheck()["checks"]) == 8
    # Still 8 checks when files are missing
    os.remove(fresh_data / "data" / "raw" / "tickets.parquet")
    assert len(run_healthcheck()["checks"]) == 8


def test_cli_exit_code(tmp_path):
    # No artifacts at all → critical files missing → "failed" → exit code 1
    env = {k: v for k, v in os.environ.items() if k != "FABRIC_WORKSPACE_ID"}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "src.healthcheck"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["status"] == "failed"


def test_fabric_check_ok(fabric_env):
    props = MagicMock()
    props.size = 12345
    service = MagicMock()
    service.return_value.get_blob_client.return_value.get_blob_properties.return_value = props

    with patch("azure.identity.ClientSecretCredential", MagicMock()), \
         patch("azure.storage.blob.BlobServiceClient", service):
        result = run_healthcheck()

    fabric_checks = [c for c in result["checks"] if c["name"].startswith("fabric:")]
    assert len(fabric_checks) == 3
    assert all(c["status"] == "ok" for c in fabric_checks)
    assert result["status"] == "ok"


def test_fabric_check_failed_when_unreachable(fabric_env):
    service = MagicMock()
    service.return_value.get_blob_client.side_effect = Exception("OneLake unreachable")

    with patch("azure.identity.ClientSecretCredential", MagicMock()), \
         patch("azure.storage.blob.BlobServiceClient", service):
        result = run_healthcheck()

    fabric_checks = [c for c in result["checks"] if c["name"].startswith("fabric:")]
    assert all(c["status"] == "failed" for c in fabric_checks)
    assert all("unreachable" in c["detail"] for c in fabric_checks)
    assert result["status"] == "failed"
