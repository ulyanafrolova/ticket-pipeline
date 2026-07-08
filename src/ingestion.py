import io
import logging
import os
from datetime import datetime, timezone

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


def _detect_platform() -> str:
    """Return 'aws' or 'azure'. Raise EnvironmentError if neither is configured."""
    has_aws = bool(os.environ.get("S3_BUCKET"))
    has_azure = bool(os.environ.get("AZURE_STORAGE_ACCOUNT"))

    if has_aws and has_azure:
        logging.warning(
            "Both S3_BUCKET and AZURE_STORAGE_ACCOUNT are set - using AWS."
            "Unset S3_BUCKET to switch to Azure."
        )
        return "aws"
    if has_aws:
        return "aws"
    if has_azure:
        return "azure"
    raise EnvironmentError(
        "Storage platform not configured."
        "Set S3_BUCKET (AWS) or AZURE_STORAGE_ACCOUNT (Azure)."
    )


def _write_parquet(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)


def _ingest_aws(output_path: str) -> int:
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    logger.info("[ingest/aws] source=s3://%s/%s", bucket, prefix)
    start = datetime.now(timezone.utc)
    logger.info("[ingest/aws] start=%s", start.isoformat())

    client = boto3.client("s3", region_name=region)
    dfs = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".csv"):
                continue
            body = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            dfs.append(pd.read_csv(io.BytesIO(body), encoding="utf-8"))

    if not dfs:
        raise RuntimeError(f"No .csv objects found under s3://{bucket}/{prefix}")

    df = pd.concat(dfs, ignore_index=True)
    _write_parquet(df, output_path)

    end = datetime.now(timezone.utc)
    logger.info("[ingest/aws] end=%s  records=%d", end.isoformat(), len(df))
    return len(df)


def _ingest_azure(output_path: str) -> int:
    from azure.storage.blob import BlobServiceClient

    account = os.environ["AZURE_STORAGE_ACCOUNT"]
    key = os.environ["AZURE_STORAGE_KEY"]
    container = os.environ["AZURE_CONTAINER_NAME"]
    prefix = os.environ.get("AZURE_BLOB_PREFIX", "")

    logger.info("[ingest/azure] source=azure://%s/%s/%s", account, container, prefix)
    start = datetime.now(timezone.utc)
    logger.info("[ingest/azure] start=%s", start.isoformat())

    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=key,
    )
    container_client = service.get_container_client(container)

    dfs = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if not blob.name.endswith(".csv"):
            continue
        data = container_client.get_blob_client(blob.name).download_blob().readall()
        dfs.append(pd.read_csv(io.BytesIO(data), encoding="utf-8"))

    if not dfs:
        raise RuntimeError(
            f"No .csv blobs found in {account}/{container}/{prefix}"
        )

    df = pd.concat(dfs, ignore_index=True)
    _write_parquet(df, output_path)

    end = datetime.now(timezone.utc)
    logger.info("[ingest/azure] end=%s  records=%d", end.isoformat(), len(df))
    return len(df)


def ingest(output_path: str) -> int:
    """
    Read all ticket records from the cloud source and write to output_path as Parquet.
    Platform is auto-detected from environment variables.
    Returns the number of records ingested.
    """
    platform = _detect_platform()
    logger.info("[ingest] platform=%s", platform)
    if platform == "aws":
        return _ingest_aws(output_path)
    return _ingest_azure(output_path)


if __name__ == "__main__":
    output = os.environ.get("INGEST_OUTPUT", "data/raw/tickets.parquet")
    count = ingest(output)
    print(f"Ingested {count} records → {output}")