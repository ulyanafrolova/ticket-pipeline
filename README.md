
# Ticket Pipeline

An LLM-augmented, cloud-agnostic data pipeline for customer-support tickets. It ingests raw CSVs from AWS S3 or Azure Blob Storage, normalizes and quality-checks them into Parquet, classifies and enriches every ticket with Claude models under a hard cost ceiling, detects anomalies with a hybrid statistical + LLM approach, and routes each anomaly through an autonomous tool-using agent with human-in-the-loop gating for irreversible actions.

This README is the operational overview: what the system does, how it is built, and how to run it.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Pipeline Stages](#pipeline-stages)
- [Data Contract](#data-contract)
- [Data Quality Checks](#data-quality-checks)
- [LLM Strategy and Cost Control](#llm-strategy-and-cost-control)
- [Cloud Integrations](#cloud-integrations)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Observability and Operations](#observability-and-operations)
- [Reliability and Failure Handling](#reliability-and-failure-handling)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Project Layout](#project-layout)
- [Design Principles](#design-principles)

---

## Key Capabilities

- **Multi-cloud ingestion** - pulls CSV objects from AWS S3 (`boto3`, paginated `list_objects_v2`) or Azure Blob Storage (`BlobServiceClient`), auto-detected from environment variables at runtime. No backend-specific code leaks into the orchestration layer.
- **Deterministic normalization** - nine vectorized transformation rules (datetime coercion, enum normalization, HTML stripping, UUID validation) with full provenance via append-only metadata columns. Row counts are preserved end to end; nothing is silently dropped.
- **Data-quality gate with dead-letter queue** - nine dataset-relative quality checks, per-row `quality_flags` annotation, JSON + self-contained HTML reports with run-over-run trend deltas, and a `rejected.parquet` dead-letter artifact.
- **Two-tier LLM classification** - Claude Haiku classifies 100% of volume; only low-confidence tickets (< 0.70) escalate to Claude Sonnet. A `max_cost_usd` budget guard halts processing when the running token cost reaches the ceiling.
- **Concurrent LLM enrichment** - sentiment, entity extraction, one-sentence summaries, and urgency scoring via a 5-worker `ThreadPoolExecutor`, with a simplified-prompt fallback and a `--resume` mode that reprocesses only failed rows.
- **Hybrid anomaly detection** - statistical pre-filtering (3σ body-length spikes, duplicate bodies, repeat frustrated contacts, unclassified tickets) so the LLM reasons only over genuine candidates, plus a threshold sensitivity analysis artifact.
- **Agentic routing with human-in-the-loop** - an autonomous Claude agent selects among six tools (escalate, alert, create task, auto-respond, fetch history, update status) per anomaly. High-severity escalations are never executed autonomously; they are queued to `pending_approval.jsonl` for explicit human confirmation.
- **Microsoft Fabric loading** - on Azure-configured runs, key artifacts are pushed to a Fabric Lakehouse via the OneLake blob endpoint as the pipeline's final step.
- **Full observability** - structured JSON logging with `run_id` correlation, per-run state files, an append-only metrics history, and a standalone health check that verifies artifact existence, freshness, and (optionally) Fabric uploads.

---

## Architecture

The system is a linear seven-stage pipeline in which **Parquet is the inter-stage contract**: every stage reads the previous stage's Parquet artifact and writes its own, giving schema preservation, 3–5× compression over CSV, and column-projection reads without any database dependency.

```
        Cloud Storage (AWS S3  /  Azure Blob Storage)
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ [1] Ingest        boto3 / azure-storage-blob                │──► data/raw/tickets.parquet
 ├─────────────────────────────────────────────────────────────┤
 │ [2] Transform     9 normalization rules, chunkable          │──► data/processed/tickets_normalized.parquet
 │                                                             │    data/processed/partitioned/channel=*/
 │                                                             │    data/processed/transform_stats.json
 ├─────────────────────────────────────────────────────────────┤
 │ [3] Validate      9 quality checks, DLQ, trend diffs        │──► data/processed/tickets_validated.parquet
 │                   HARD FAIL on excessive duplicate IDs      │    data/quality/{quality_report.json,.html,
 │                                                             │                  rejected.parquet}
 ├─────────────────────────────────────────────────────────────┤
 │ [4] Classify      Haiku → Sonnet escalation, cost guard     │──► data/enriched/tickets_classified.parquet
 ├─────────────────────────────────────────────────────────────┤
 │ [5] Enrich        sentiment / entities / summary / urgency  │──► data/enriched/tickets_enriched.parquet
 │                   ThreadPoolExecutor(5), resume mode        │
 ├─────────────────────────────────────────────────────────────┤
 │ [6] Detect        statistical flags → LLM classification    │──► data/anomalies/{anomalies.parquet,
 │                   3σ threshold + sensitivity analysis       │        anomaly_report.json, threshold_analysis.json,
 │                                                             │        high_severity_alerts.json}
 ├─────────────────────────────────────────────────────────────┤
 │ [7] Agent         tool-use loop, HITL approval queue        │──► data/agent/{actions.jsonl, reasoning.jsonl,
 │                                                             │        pending_approval.jsonl, agent_summary.json}
 ├─────────────────────────────────────────────────────────────┤
 │ [+] FabricUpload  (Azure runs only)                         │──► OneLake / Fabric Lakehouse Files
 └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        data/pipeline_state.json   +   data/metrics/run_history.jsonl
```

The orchestrator (`src/pipeline.py`) executes all seven stages in order, verifies each stage's required input artifact before running it, times every step with `time.perf_counter`, and writes `data/pipeline_state.json` on both success and failure. When the storage platform resolves to Azure, a `FabricUpload` step is appended to full runs.

### Architectural highlights

- **Append-only metadata columns** — Transform and Validate add columns (`parse_error`, `html_stripped`, `invalid_ticket_id`, `quality_flags`, …) rather than mutating values, so raw provenance is always recoverable.
- **Vectorized hot paths** — quality-flag computation uses NumPy element-wise string operations with no per-row Python loop; ~500K rows validate in sub-second time.
- **Hard-fail *after* writes** — the duplicate-ID hard failure in Validate triggers `sys.exit(1)` only after all Parquet and report artifacts are written, so operators always have diagnostics for the failed run.
- **Statistical pre-filtering before LLM reasoning** — the anomaly detector never sends the full dataset to a model; only statistically flagged candidates receive LLM classification.
- **Bounded agent loop** — the tool-use loop caps at 10 iterations per anomaly and truncates conversation history when the estimated context exceeds 40,000 tokens (retaining the initial message and the last 4 turns), preventing runaway loops and context-length API errors.

---

## Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Runtime | Python 3.11 | Target for all CI and Docker deployments; modern `list[str]` / `X \| Y` annotations throughout |
| Columnar storage | Apache Parquet via pandas + PyArrow | Efficient columnar I/O, nullable types, hive-style channel partitioning |
| Data manipulation | pandas 2.x, NumPy | Fully vectorized transforms and validation — no per-row loops in hot paths |
| Schema / contracts | Pydantic v2 | `@field_validator`-based record contract; strict typing of all LLM JSON outputs |
| LLM inference | Anthropic SDK (`anthropic`) | Claude Haiku / Sonnet tiers for classification, enrichment, detection, and the agent loop |
| Azure LLM backend | `azure-ai-projects` + `azure-identity` | Drop-in replacement agent backend using Azure AI Agents `FunctionTool` |
| Cloud storage | `boto3` (AWS), `azure-storage-blob` (Azure) | Selected at runtime by environment inspection |
| Fabric loading | `azure-storage-blob` against the OneLake endpoint | Uploads artifacts into a Fabric Lakehouse `Files/` section |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` | I/O-bound LLM calls in the enrichment stage (5 workers) |
| Synthetic data | Faker | Seeded, reproducible test datasets with deliberately injected quality defects |
| Testing | pytest + `unittest.mock` | Full mocking of cloud and LLM I/O; no network or `data/` writes in tests |
| Packaging / runtime env | Docker (`python:3.11-slim`), docker-compose, Make | Reproducible local and CI execution |
| CI | GitHub Actions | Docker build + containerized pytest on every push/PR to `main` |

---

## Pipeline Stages

### Stage 1 — Ingest (`src/ingestion.py`)

Detects the storage platform from the environment: `S3_BUCKET` selects AWS, `AZURE_STORAGE_ACCOUNT` selects Azure. If both are set, AWS wins with a logged warning; if neither is set, `EnvironmentError` is raised immediately. All `.csv` objects under the configured bucket/container prefix are read, concatenated, and materialized as a single Parquet file at `data/raw/tickets.parquet`. Zero matching CSVs is a `RuntimeError`. The raw artifact is **write-once** — no downstream stage may modify it.

### Stage 2 — Transform (`src/transform.py`)

Enforces the 10-column canonical schema (a missing column raises `ValueError` listing every absent column), then applies nine deterministic normalization rules to a *copy* of the data:

1. `created_at` → UTC-aware datetime; unparseable values become null and set `parse_error`.
2–5. `channel`, `priority`, `status`, `category` → strip + lowercase; out-of-enum values become `"unknown"` (or null for `priority`). In **strict mode** (`PIPELINE_PROFILE=prod`) invalid enums raise instead.
6. `subject` / `body` → whitespace-stripped; empty strings become null.
7. ID fields → whitespace-stripped only, never modified.
8. `body` → vectorized regex HTML-tag stripping, flagged via `html_stripped`.
9. `ticket_id` → validated against the UUID v4 pattern; flagged via `invalid_ticket_id`, value untouched.

Outputs: the normalized Parquet (plus `_schema_version` / `_processed_at` metadata), a PyArrow **channel-partitioned copy** (`data/processed/partitioned/channel=*/`), and `transform_stats.json` with distribution and defect counts. An optional `chunk_size` processes the frame in independent slices to bound peak memory without changing output semantics.

### Stage 3 — Validate (`src/validate.py`)

Runs nine quality checks (see [table below](#data-quality-checks)), annotates every row with a pipe-delimited `quality_flags` column, writes flagged rows to the dead-letter queue (`data/quality/rejected.parquet`), and emits structured JSON and self-contained HTML quality reports. Each check carries a run-over-run **trend delta** computed against the previous run's report. Duplicate `ticket_id` counts exceeding `max(1, floor(N × 0.01))` are the single **hard failure**: `sys.exit(1)`, but only after all artifacts are written.

### Stage 4 — Classify (`src/classify.py`)

Assigns `llm_category`, `llm_priority`, and `llm_confidence` to every ticket via a two-tier escalation strategy:

- **Tier 1** — `claude-haiku-4-5-20251001` (128 max tokens, temperature 0.0) classifies all tickets. Confidence < 0.70 sets `needs_review = True`.
- **Tier 2** — flagged tickets are re-classified by `claude-sonnet-4-6` (256 max tokens). Sonnet results with confidence ≥ 0.85 replace the Haiku result and clear the review flag; otherwise the Haiku classification is retained and the flag stands.

A running cost estimate (per-model input/output token pricing) is recomputed every 10-ticket batch; reaching `max_cost_usd` halts processing early, leaving unprocessed rows as `llm_category = "unknown"`. `classification_report.json` records counts, per-model cost breakdown, category/priority distributions, and a confidence histogram.

### Stage 5 — Enrich (`src/enrich.py`)

Annotates each classified ticket with `sentiment` (+confidence), extracted entities (`product_names`, `error_codes`, `account_ids`), a one-sentence past-tense `summary`, and an `urgency_score` with reasoning — via `claude-sonnet-4-6` (512 max tokens) on a 5-worker thread pool. A failed full-enrichment attempt falls back to a simplified sentiment + summary prompt with documented urgency defaults; a second failure marks the row `enrichment_failed = True`. **Resume mode** (`--resume`) reprocesses only failed rows in an existing output. After enrichment, customers appearing in ≥ 3 records are flagged `repeat_contact = True` for the anomaly detector.

### Stage 6 — Detect (`src/detect_anomalies.py`)

Hybrid detection in two phases:

1. **Statistical** — five vectorized flags per row: `body_length_spike` (length > mean + 3σ), `frustrated_sentiment`, `unclassified`, `repeat_frustrated`, and `duplicate_body` (normalized body text appearing ≥ 2 times with length > 20). The 3σ threshold was selected from a 2.0σ / 2.5σ / 3.0σ sensitivity analysis preserved in `threshold_analysis.json`.
2. **LLM classification** — each flagged row with a non-null body is classified by Claude Haiku (200 max tokens) into an anomaly type, `low`/`medium`/`high` severity, a one-sentence reason, and a recommended action. Null-body rows are recorded as statistical-only detections.

High-severity anomalies are routed by recommended action (`escalate` → slack, `create_task` → email, otherwise monitor) and serialized to `high_severity_alerts.json`. The anomaly report carries trend deltas versus the previous run.

### Stage 7 — Agent (`src/agent.py`)

For each anomaly record, an autonomous Claude agent (`claude-sonnet-5` on the Anthropic path) runs a standard tool-use loop with six tools:

| Tool | Purpose |
|---|---|
| `escalate_ticket` | Escalate to a senior support agent (low / medium / high severity) |
| `send_alert` | Notify the support team via slack / email / pagerduty |
| `create_task` | Create a follow-up task for tier1 / tier2 / billing_team / engineering |
| `auto_respond` | Send a templated customer acknowledgement |
| `get_ticket_history` | Retrieve customer support history for context |
| `update_ticket_status` | Set ticket status to open / pending / closed |

**Human-in-the-loop gate:** `escalate_ticket` with `severity == "high"` is *never executed* — it is serialized to `pending_approval.jsonl` with its full payload and timestamp, and the agent is told the escalation was queued. Every tool call, result, and iteration stop-reason is journaled to `actions.jsonl` / `reasoning.jsonl`; `agent_summary.json` aggregates the run.

An alternative **Azure AI Agents backend** (`--backend azure`) registers the same tool set as an Azure `FunctionTool` via `azure-ai-projects`, creates the agent per run (`claude-sonnet-4-6`), and guarantees deletion in a `finally` block. The HITL gate applies identically.

### Final step — FabricUpload (Azure runs only)

When a full pipeline run detects Azure as the storage platform, `src/load_to_fabric.py` uploads `tickets_enriched.parquet`, `quality_report.json`, and `anomaly_report.json` to the configured Fabric Lakehouse `Files/` section through the OneLake blob endpoint, authenticated via an Entra ID service principal (`ClientSecretCredential`).

---

## Data Contract

Every ingested record must conform to this 10-column schema, defined once in `src/schema.py` (the single source of truth — inline enum redefinition is forbidden by design):

| Column | Type | Constraints |
|---|---|---|
| `ticket_id` | string | UUID v4 format; flagged (never modified) if invalid |
| `created_at` | string | ISO 8601; malformed values coerced to null |
| `customer_id` | string | Non-null; whitespace-stripped only |
| `channel` | string | `email` \| `chat` \| `phone` \| `web` |
| `subject` | string | Free text ≤ 80 chars; empty treated as missing |
| `body` | string | Free text; HTML stripped; empty treated as missing |
| `priority` | string | `low` \| `medium` \| `high` \| `critical` |
| `category` | string | `billing` \| `technical` \| `general` \| `other` |
| `status` | string | `open` \| `closed` \| `pending` |
| `agent_id` | string | UUID; nullable; whitespace-stripped only |

A missing column raises `ValueError` at Transform. The pipeline never infers or synthesizes missing columns.

---

## Data Quality Checks

All thresholds are **dataset-relative** so small datasets cannot vacuously pass fixed-count checks.

| # | Check | Type | Threshold | On failure |
|---|---|---|---|---|
| 1 | `completeness_subject` | rate | ≥ 0.90 | WARN |
| 2 | `completeness_priority` | rate | ≥ 0.80 | WARN |
| 3 | `validity_channel` | rate | ≥ 0.95 | WARN |
| 4 | `validity_status` | rate | ≥ 0.95 | WARN |
| 5 | `uniqueness_ticket_id` | count | ≤ max(1, ⌊N × 0.01⌋) | **HARD FAIL** (`exit 1`, after artifacts are written) |
| 6 | `validity_created_at` | count | ≤ max(1, ⌊N × 0.10⌋) | WARN |
| 7 | `short_body` | count | ≤ max(1, ⌊N × 0.01⌋) | WARN |
| 8 | `future_created_at` | count | = 0 | WARN |
| 9 | `closed_without_agent` | count | = 0 | WARN |

Per-row failures are recorded in the `quality_flags` column (pipe-delimited, empty string for clean rows) and flagged rows are materialized to the dead-letter queue at `data/quality/rejected.parquet`. Flagged rows still flow through all downstream stages — **no silent data loss**.

---

## LLM Strategy and Cost Control

The pipeline treats LLM spend as a first-class constraint:

- **Tiered routing.** Haiku handles 100% of classification volume; Sonnet sees only the ~5–15% of tickets that Haiku is unsure about. Anomaly classification also runs on Haiku; only enrichment and the agent loop use Sonnet-class models by default.
- **Hard budget ceiling.** `classify(..., max_cost_usd=X)` (or `--max-cost X` on the CLI) recomputes the running cost before each batch of 10 and halts when the ceiling is reached. Costs are computed from per-model input/output token pricing and surfaced in every report.
- **Bounded output tokens.** Every call sets an explicit `max_tokens` (128–512 depending on task) and temperature 0.0 for deterministic structured output.
- **Structured-output validation.** All model responses are parsed as JSON (markdown fences stripped) and validated with Pydantic; parse/validation failures retry up to 3 attempts before falling back to documented defaults.
- **Context guard.** The agent loop truncates conversations that exceed ~40,000 estimated tokens, and truncation counts are reported in `agent_summary.json`.

---

## Cloud Integrations

The pipeline is **cloud-agnostic by configuration** (spec principle 1.7): the storage backend and LLM/agent backend are selected at runtime purely from environment variables, and the orchestration layer contains no backend-specific code.

### AWS

- **Ingestion source:** S3. Paginated `list_objects_v2` over `s3://$S3_BUCKET/$S3_PREFIX`; only `.csv` keys are read.
- **Managed orchestration:** [`infra/aws/state_machine.json`](infra/aws/state_machine.json) defines an AWS Step Functions state machine (Ingest → Transform → Validate → Load) where each stage is a Lambda task with exponential-backoff retries (30s base, 3 attempts, 2× backoff) and per-stage `Catch` routing to explicit failure states. Account-specific ARNs are placeholders to be substituted at deploy time.

### Azure

- **Ingestion source:** Blob Storage. Blobs are listed via `BlobServiceClient` under `$AZURE_CONTAINER_NAME/$AZURE_BLOB_PREFIX`, `.csv` only, using account-key authentication.
- **Microsoft Fabric / OneLake:** on Azure-configured full runs, enriched data and reports are uploaded to a Fabric Lakehouse (`Files/` section) through `https://onelake.blob.fabric.microsoft.com`, addressed as `{workspace_id}/{lakehouse_id}` and authenticated with an Entra ID service principal. The health check can verify these uploads via the OneLake DFS endpoint.
- **Managed orchestration:** [`infra/azure/fabric_pipeline.json`](infra/azure/fabric_pipeline.json) defines a Fabric Data Pipeline: a Copy activity lands raw tickets from ADLS into a Lakehouse table, Transform and Validate run as Fabric notebooks (`TridentNotebook` activities), and a final Copy activity publishes the validated table. Notebook and workspace IDs are placeholders.
- **Azure AI Agents:** the Stage 7 agent can run entirely on the `azure-ai-projects` SDK (`--backend azure`), with the same tool set and the same human-in-the-loop gate.

**Platform precedence:** if both `S3_BUCKET` and `AZURE_STORAGE_ACCOUNT` are set, AWS takes precedence and a warning is logged. If neither is set, ingestion fails fast with `EnvironmentError`.

---

## Configuration

All configuration is sourced **exclusively from the process environment** — no module reads a `.env` file directly. A template is provided in [`.env.example`](.env.example); docker-compose maps these variables into the container.

| Variable | Required by | Description |
|---|---|---|
| `PIPELINE_PROFILE` | all stages | `dev` \| `staging` \| `prod` (default `dev`) |
| `S3_BUCKET` | Ingest (AWS) | Source bucket name |
| `S3_PREFIX` | Ingest (AWS) | Key prefix filter (default `""`) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | Ingest (AWS) | Standard AWS credentials |
| `AWS_DEFAULT_REGION` | Ingest (AWS) | Region (default `us-east-1`) |
| `AZURE_STORAGE_ACCOUNT` | Ingest (Azure) | Storage account name |
| `AZURE_STORAGE_KEY` | Ingest (Azure) | Storage account key |
| `AZURE_CONTAINER_NAME` | Ingest (Azure) | Blob container |
| `AZURE_BLOB_PREFIX` | Ingest (Azure) | Prefix filter (default `""`) |
| `ANTHROPIC_API_KEY` | Classify, Enrich, Detect, Agent | Anthropic API key |
| `AZURE_PROJECT_CONNECTION_STRING` | Agent (Azure backend) | Azure AI Projects connection string |
| `FABRIC_WORKSPACE_ID` / `FABRIC_LAKEHOUSE_ID` | FabricUpload, Healthcheck | Target Fabric Lakehouse |
| `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` | FabricUpload, Healthcheck | Entra ID service-principal credentials for OneLake |

### Environment profiles

| Profile | `LOG_LEVEL` | Strict mode | Behavior |
|---|---|---|---|
| `dev` (default) | `DEBUG` | off | Invalid enum values become `"unknown"`; no exception |
| `staging` | `INFO` | off | Same as dev with reduced log verbosity |
| `prod` | `WARNING` | **on** | Invalid enum values raise `ValueError` immediately — the production safety gate |

---

## Getting Started

### Prerequisites

- Python **3.11** (the CI/Docker target; the codebase uses 3.10+ annotation syntax throughout)
- Docker + docker-compose (optional, for containerized runs)
- Cloud credentials for at least one storage backend, and an `ANTHROPIC_API_KEY` for stages 4–7

### Install

```bash
git clone <repo-url> && cd ticket-pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Unix: source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env      # fill in credentials, then export into your shell/session
```

> `.env` is git-ignored and is **not** read by the application; export the variables into the process environment (docker-compose does this automatically from `.env`).

### Generate a test dataset (optional)

```bash
python -m src.generate_dataset --count 500 --seed 42 --output data/tickets.csv
```

Produces a seeded, reproducible CSV with realistic Faker-generated fields and deliberately injected defects (5% null subjects, 8% null priorities, 3% duplicate ticket IDs, 2% malformed dates, 1% short bodies) — upload it to your bucket/container to exercise the full pipeline.

---

## Running the Pipeline

### Full run

```bash
python src/pipeline.py            # or: make pipeline
```

Runs Ingest → Transform → Validate → Classify → Enrich → Detect → Agent (plus FabricUpload on Azure), writing `data/pipeline_state.json` and appending to `data/metrics/run_history.jsonl`.

### Dry run

```bash
python src/pipeline.py --dry-run  # or: make pipeline-dry
```

Prints the per-step plan (inputs → outputs) and exits without executing or writing anything.

### Single step

```bash
python src/pipeline.py --step ingest      # any of: ingest | transform | validate |
python src/pipeline.py --step classify    #         classify | enrich | detect | agent
```

The step's required input artifact must already exist (`FileNotFoundError` otherwise).

### Stage CLIs (direct invocation with options)

```bash
python -m src.ingestion                                   # INGEST_OUTPUT overrides target path
python -m src.transform                                   # TRANSFORM_INPUT / TRANSFORM_OUTPUT overrides
python -m src.validate
python -m src.classify --sample 50 --max-cost 1.50        # cap volume and spend
python -m src.enrich --resume                             # reprocess only failed rows
python -m src.detect_anomalies
python -m src.agent --backend python                      # or --backend azure
python -m src.load_to_fabric                              # manual Fabric upload
python -m src.healthcheck                                 # artifact existence/freshness check
```

### Make targets

| Target | Action |
|---|---|
| `make pipeline` / `make pipeline-dry` | Full run / dry run |
| `make ingest` / `make transform` / `make validate` | Individual early stages |
| `make test` | `pytest tests/ -v` |
| `make clean` | Remove all generated `data/` artifacts |
| `make docker-up` | Build and run the test suite in Docker |

### Docker

```bash
docker-compose run --rm app python src/pipeline.py   # full pipeline in-container
docker-compose run --rm app pytest                    # test suite (default command)
```

The image is `python:3.11-slim`; the project directory is volume-mounted and all cloud/LLM variables are passed through from the host environment (or `.env` via compose interpolation).

---

## Observability and Operations

- **Structured JSON logging** (`src/logger.py`) — every log line is a single-line JSON object on stdout with UTC timestamp, level, component, message, and arbitrary structured extras. A logging filter injects the pipeline `run_id` into every record, so a whole run is correlatable with one grep.
- **Run state** (`data/pipeline_state.json`) — written on *every* real run: `run_id` (UTC ISO-8601), profile, start/finish timestamps, total duration, and per-step record counts and durations. Failed runs carry `status: "failed"` plus the error message.
- **Metrics history** (`src/metrics.py` → `data/metrics/run_history.jsonl`) — one JSON line appended per run: status, duration, records ingested/validated, failed quality checks, anomalies found, and agent actions taken. `get_run_summary(n)` returns the last N runs. A metrics failure never breaks the pipeline.
- **Health check** (`python -m src.healthcheck`) — verifies that all expected artifacts exist and are fresher than 24 hours. Core artifacts (raw/normalized/validated Parquet, quality report) missing → `failed` (exit 1); downstream artifacts missing or anything stale → `degraded` (exit 0). When `FABRIC_WORKSPACE_ID` is set, it additionally verifies the OneLake uploads. Output is a JSON summary suitable for cron/monitoring integration.
- **Trend reporting** — quality and anomaly reports each diff against their previous-run counterpart, with signed deltas rendered green/red in the HTML quality report.

---

## Reliability and Failure Handling

- **Fail-fast on hard errors, warn on soft errors.** Missing prerequisite artifacts and excessive duplicate IDs terminate the run with exit code 1; malformed enums and unparseable dates become data-quality flags, never silent drops.
- **Generic retry decorator** (`src/retry.py`) — configurable max attempts, exponential backoff with cap, and jitter, restricted to a declared tuple of retryable exception types.
- **LLM call resilience** — up to 3 attempts per call; `RateLimitError` sleeps 30s and retries; `AuthenticationError` / `BadRequestError` propagate immediately as non-retryable; JSON/validation errors retry then fall back to documented defaults (`llm_category="unknown"`, enrichment fallback prompt, `severity="low"` on detection failure).
- **Resume-ability** — enrichment `--resume` reprocesses only rows with `enrichment_failed == True`, enabling partial recovery from API outages without re-spending on successful rows.
- **Managed-orchestration retries** — the Step Functions definition mirrors the same policy at the infrastructure level (30s interval, 3 attempts, 2× backoff, per-stage failure states).
- **Human-in-the-loop for irreversible actions** — high-severity escalations always require explicit human approval via `pending_approval.jsonl`.
- **State on failure** — `pipeline_state.json` and the metrics history are written even when a run fails, preserving the diagnostic trail.

---

## Testing

```bash
make test          # pytest tests/ -v
# or in Docker:
make docker-up
```

Twelve test modules cover every source module plus an end-to-end integration suite. The strategy (spec §3.5):

- **Unit isolation** — step functions are mocked at their import site in `src.pipeline`; no real cloud I/O, LLM calls, or writes to `data/` occur in any test.
- **Filesystem hygiene** — file-writing tests redirect `PATHS` to `tmp_path` / `TemporaryDirectory`.
- **LLM mocking** — `anthropic.Anthropic` and classification entry points are patched; API keys are faked via `patch.dict(os.environ, ...)`.
- **Behavioral contracts** — step ordering, state-file contents on success *and* failure, dry-run no-op, single-step isolation, prerequisite checks, threshold boundary conditions, and hard-fail exit codes (`pytest.raises(SystemExit)` asserting code 1).

---

## CI/CD

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request to `main`:

1. Build the Docker image (`python:3.11-slim` base).
2. Run the full pytest suite inside the container.

Because tests mock all external I/O, CI needs no cloud credentials or API keys.

---

## Project Layout

```
ticket-pipeline/
├── src/
│   ├── pipeline.py            # Orchestration runner (CLI: --dry-run, --step)
│   ├── config.py              # Profile, PATHS, log level, strict mode
│   ├── schema.py              # Canonical schema + enum sets (single source of truth)
│   ├── ingestion.py           # Stage 1 — S3 / Azure Blob → raw Parquet
│   ├── transform.py           # Stage 2 — normalization rules, partitioning, stats
│   ├── validate.py            # Stage 3 — quality checks, DLQ, HTML/JSON reports
│   ├── classify.py            # Stage 4 — two-tier Haiku/Sonnet classification
│   ├── enrich.py              # Stage 5 — sentiment/entities/summary/urgency
│   ├── detect_anomalies.py    # Stage 6 — statistical + LLM anomaly detection
│   ├── agent.py               # Stage 7 — tool-use agent, HITL queue, Azure backend
│   ├── load_to_fabric.py      # OneLake / Fabric Lakehouse uploads
│   ├── generate_dataset.py    # Seeded synthetic CSV generator with injected defects
│   ├── healthcheck.py         # Artifact existence/freshness + Fabric verification
│   ├── metrics.py             # Per-run metrics history (JSONL)
│   ├── retry.py               # Exponential-backoff retry decorator
│   └── logger.py              # Structured JSON logger with run_id injection
├── tests/                     # pytest suite (unit + integration, fully mocked I/O)
├── infra/
│   ├── aws/state_machine.json      # Step Functions definition (Lambda per stage)
│   └── azure/fabric_pipeline.json  # Fabric Data Pipeline definition
├── data/                      # All pipeline artifacts (git-ignored; see spec §3.3)
├── Dockerfile                 # python:3.11-slim runtime image
├── docker-compose.yml         # Containerized runs with env passthrough
├── Makefile                   # ingest / transform / validate / pipeline / test / clean
├── .github/workflows/ci.yml   # Docker build + containerized pytest
├── .env.example               # Environment variable template
├── requirements.txt
└── spec.md                    # Full technical specification
```

The complete artifact tree produced under `data/` (raw, processed, quality, enriched, anomalies, agent, metrics).

---

## Design Principles

The system is governed by eight non-negotiable principles; every design decision traces back to one of them:

1. **Immutability of raw data** — the ingested Parquet is write-once; all transforms produce new artifacts.
2. **Schema as a single source of truth** — all field names, enums, and format constraints live in `src/schema.py`.
3. **Fail-fast on hard errors, warn on soft errors.**
4. **No silent data loss** — rows are never deleted; input count equals output count at every stage, with flagged rows additionally dead-lettered.
5. **Cost-bounded LLM usage** — every invocation is token- and cost-tracked under a hard ceiling.
6. **Reproducibility and observability** — every run has a UTC `run_id`, a state artifact on success *and* failure, and perf-counter-timed structured logs.
7. **Cloud-agnostic by configuration** — storage and LLM backends are runtime environment choices, never code branches in the orchestration layer.
8. **Human-in-the-loop for irreversible actions** — high-severity escalations always await explicit human approval.
