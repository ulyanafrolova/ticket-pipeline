import io
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.ingestion import _detect_platform, _ingest_aws, _ingest_azure


def _csv_bytes(n: int = 3) -> bytes:
    df = pd.DataFrame({"ticket_id": range(n), "subject": [f"t{i}" for i in range(n)]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# --- _detect_platform ---

def test_detect_platform_aws(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT", raising=False)
    assert _detect_platform() == "aws"


def test_detect_platform_azure(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "myaccount")
    assert _detect_platform() == "azure"


def test_detect_platform_neither_raises(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT", raising=False)
    with pytest.raises(EnvironmentError):
        _detect_platform()


def test_detect_platform_both_prefers_aws(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "myaccount")
    assert _detect_platform() == "aws"


# --- _ingest_aws ---

def test_ingest_aws_reads_csvs(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setenv("S3_PREFIX", "")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    mock_client = MagicMock()
    mock_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "a.csv"}, {"Key": "b.csv"}]}
    ]
    mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: _csv_bytes(3))}

    with patch("src.ingestion.boto3.client", return_value=mock_client):
        count = _ingest_aws(str(tmp_path / "out.parquet"))

    assert count == 6  # 3 rows × 2 files
    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 6


def test_ingest_aws_skips_non_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setenv("S3_PREFIX", "")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    mock_client = MagicMock()
    mock_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "a.csv"}, {"Key": "b.parquet"}]}
    ]
    mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: _csv_bytes(3))}

    with patch("src.ingestion.boto3.client", return_value=mock_client):
        count = _ingest_aws(str(tmp_path / "out.parquet"))

    assert count == 3  # only 1 CSV file processed


def test_ingest_aws_no_files_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setenv("S3_PREFIX", "")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    mock_client = MagicMock()
    mock_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": []}
    ]

    with patch("src.ingestion.boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError):
            _ingest_aws(str(tmp_path / "out.parquet"))


# --- _ingest_azure ---

def test_ingest_azure_reads_csvs(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "myaccount")
    monkeypatch.setenv("AZURE_STORAGE_KEY", "mykey")
    monkeypatch.setenv("AZURE_CONTAINER_NAME", "mycontainer")
    monkeypatch.setenv("AZURE_BLOB_PREFIX", "")

    blob_a = MagicMock()
    blob_a.name = "a.csv"
    blob_b = MagicMock()
    blob_b.name = "b.csv"

    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value.readall.return_value = _csv_bytes(3)

    mock_container_client = MagicMock()
    mock_container_client.list_blobs.return_value = [blob_a, blob_b]
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container_client

    with patch("azure.storage.blob.BlobServiceClient", return_value=mock_service):
        count = _ingest_azure(str(tmp_path / "out.parquet"))

    assert count == 6  # 3 rows × 2 files
    df = pd.read_parquet(tmp_path / "out.parquet")
    assert len(df) == 6


def test_ingest_azure_skips_non_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "myaccount")
    monkeypatch.setenv("AZURE_STORAGE_KEY", "mykey")
    monkeypatch.setenv("AZURE_CONTAINER_NAME", "mycontainer")
    monkeypatch.setenv("AZURE_BLOB_PREFIX", "")

    blob_a = MagicMock()
    blob_a.name = "a.csv"
    blob_b = MagicMock()
    blob_b.name = "b.json"

    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value.readall.return_value = _csv_bytes(3)

    mock_container_client = MagicMock()
    mock_container_client.list_blobs.return_value = [blob_a, blob_b]
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container_client

    with patch("azure.storage.blob.BlobServiceClient", return_value=mock_service):
        count = _ingest_azure(str(tmp_path / "out.parquet"))

    assert count == 3  # only 1 CSV blob processed


def test_ingest_azure_no_files_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "myaccount")
    monkeypatch.setenv("AZURE_STORAGE_KEY", "mykey")
    monkeypatch.setenv("AZURE_CONTAINER_NAME", "mycontainer")
    monkeypatch.setenv("AZURE_BLOB_PREFIX", "")

    mock_container_client = MagicMock()
    mock_container_client.list_blobs.return_value = []

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container_client

    with patch("azure.storage.blob.BlobServiceClient", return_value=mock_service):
        with pytest.raises(RuntimeError):
            _ingest_azure(str(tmp_path / "out.parquet"))
