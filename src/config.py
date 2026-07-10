import os

PROFILE = os.environ.get("PIPELINE_PROFILE", "dev")  # dev | staging | prod

PATHS = {
    "raw":            "data/raw/tickets.parquet",
    "normalized":     "data/processed/tickets_normalized.parquet",
    "validated":      "data/processed/tickets_validated.parquet",
    "quality_report": "data/quality/quality_report.json",
    "pipeline_state": "data/pipeline_state.json",
    "classified":     "data/enriched/tickets_classified.parquet",
    "enriched":       "data/enriched/tickets_enriched.parquet",
    "anomalies":      "data/anomalies/anomalies.parquet",
    "anomaly_report": "data/anomalies/anomaly_report.json",
}

LOG_LEVEL = {
    "dev":     "DEBUG",
    "staging": "INFO",
    "prod":    "WARNING",
}.get(PROFILE, "INFO")

STRICT_MODE = PROFILE == "prod" 