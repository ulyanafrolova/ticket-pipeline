"""
Pipeline health check: verify expected output artifacts exist and are recent.

Usage:
    python -m src.healthcheck
Prints a JSON summary; exits 0 if status is ok/degraded, 1 if failed.
"""

import json
import os
import sys
import time

# (path, status when the file is missing). A stale file is always "degraded".
EXPECTED_FILES = [
    ("data/raw/tickets.parquet", "failed"),
    ("data/processed/tickets_normalized.parquet", "failed"),
    ("data/processed/tickets_validated.parquet", "failed"),
    ("data/quality/quality_report.json", "failed"),
    ("data/enriched/tickets_classified.parquet", "degraded"),
    ("data/enriched/tickets_enriched.parquet", "degraded"),
    ("data/anomalies/anomalies.parquet", "degraded"),
    ("data/agent/actions.jsonl", "degraded"),
]

# OneLake blobs uploaded by the FabricUpload pipeline step (checked only when
# FABRIC_WORKSPACE_ID is set).
FABRIC_BLOBS = [
    "Files/processed/tickets_enriched.parquet",
    "Files/quality/quality_report.json",
    "Files/anomalies/anomaly_report.json",
]


def _check_file(path: str, missing_status: str, max_age_hours: int) -> dict:
    """Check one local artifact: missing → missing_status, stale → degraded."""
    if not os.path.exists(path):
        return {"name": path, "status": missing_status, "detail": "file missing"}
    age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
    if age_hours > max_age_hours:
        return {
            "name": path,
            "status": "degraded",
            "detail": f"stale: {age_hours:.1f}h old (max {max_age_hours}h)",
        }
    return {"name": path, "status": "ok", "detail": f"age={age_hours:.1f}h"}


def _check_fabric_file(workspace_id: str, lakehouse_id: str, blob_name: str) -> dict:
    """
    Verify a file exists in the Fabric Lakehouse Files section via OneLake.
    Returns {"name": ..., "status": "ok"|"failed", "detail": ...}
    """
    try:
        from azure.identity import ClientSecretCredential
        from azure.storage.blob import BlobServiceClient
        credential = ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
        )
        client = BlobServiceClient(
            account_url="https://onelake.dfs.fabric.microsoft.com",
            credential=credential,
        )
        container = f"{workspace_id}/{lakehouse_id}"
        props = client.get_blob_client(container=container, blob=blob_name).get_blob_properties()
        return {"name": f"fabric:{blob_name}", "status": "ok", "detail": f"size={props.size}"}
    except Exception as exc:
        return {"name": f"fabric:{blob_name}", "status": "failed", "detail": str(exc)}


def run_healthcheck(max_age_hours: int = 24) -> dict:
    """
    Check that all expected pipeline output files exist and were modified
    within the last max_age_hours.
    Returns: {"status": "ok"|"degraded"|"failed", "checks": [...]}
    """
    checks = [
        _check_file(path, missing_status, max_age_hours)
        for path, missing_status in EXPECTED_FILES
    ]

    workspace_id = os.environ.get("FABRIC_WORKSPACE_ID")
    if workspace_id:
        lakehouse_id = os.environ.get("FABRIC_LAKEHOUSE_ID", "")
        for blob_name in FABRIC_BLOBS:
            checks.append(_check_fabric_file(workspace_id, lakehouse_id, blob_name))

    statuses = {c["status"] for c in checks}
    if "failed" in statuses:
        overall = "failed"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    return {"status": overall, "checks": checks}


def main() -> int:
    result = run_healthcheck()
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
