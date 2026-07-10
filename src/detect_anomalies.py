import datetime
import json
import logging
import os
import shutil

import anthropic
import pandas as pd
from anthropic import APIStatusError, RateLimitError
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal

from src.retry import retry

# Retry transient API errors and malformed model output; never retry
# AuthenticationError / BadRequestError (they are not in this tuple).
RETRYABLE_EXCEPTIONS = (RateLimitError, APIStatusError, json.JSONDecodeError, ValidationError)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

_ANOMALY_SYSTEM_PROMPT = """You are an anomaly classifier for a customer support system. Analyze the ticket data and statistical flags provided, then return a JSON classification.

Output ONLY valid JSON with this exact schema:
{
  "anomaly_type": "<body_length_spike|frustrated_sentiment|unclassified|potential_incident|repeat_frustrated|duplicate_body>",
  "severity": "<low|medium|high>",
  "reason": "<one sentence specific to this ticket>",
  "recommended_action": "<escalate|auto_respond|create_task|monitor>"
}

No explanation. No markdown. Only valid JSON."""


class AnomalyResult(BaseModel):
    anomaly_type: Literal[
        "body_length_spike",
        "frustrated_sentiment",
        "unclassified",
        "potential_incident",
        "repeat_frustrated",
        "duplicate_body",
    ]
    severity: Literal["low", "medium", "high"]
    reason: str
    recommended_action: Literal["escalate", "auto_respond", "create_task", "monitor"]

    @field_validator("reason")
    @classmethod
    def reason_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("reason must not be empty")
        return v.strip()


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _classify_anomaly(
    client: anthropic.Anthropic,
    ticket_id: str,
    subject: str,
    body: str,
    sentiment: str,
    llm_category: str,
    flags: list[str],
) -> AnomalyResult | None:
    body_preview = (body[:500] if body else "")
    user_message = (
        f"ticket_id: {ticket_id}\n"
        f"subject: {subject}\n"
        f"body (first 500 chars): {body_preview}\n"
        f"sentiment: {sentiment}\n"
        f"llm_category: {llm_category}\n"
        f"statistical_flags: {', '.join(flags)}"
    )

    @retry(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        jitter=True,
        retryable_exceptions=RETRYABLE_EXCEPTIONS,
    )
    def _call_llm():
        return client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0.0,
            system=_ANOMALY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

    try:
        response = _call_llm()
        if not response.content:
            raise ValueError("Empty response content from API")
        raw = _strip_markdown_fences(response.content[0].text)
        data = json.loads(raw)
        return AnomalyResult(**data)
    except Exception as e:
        logger.warning("Failed to classify anomaly for ticket %s: %s", ticket_id, e)
        return None


def _analyze_thresholds(series: pd.Series, thresholds: list[float]) -> dict:
    """Return flagged counts for body_length_spike at each z-score threshold."""
    body_lengths = series.dropna()
    total = len(series)
    if len(body_lengths) < 2:
        return {str(t): {"flagged": 0, "pct_of_total": 0.0} for t in thresholds}
    mean_len = body_lengths.mean()
    std_len = body_lengths.std()
    result = {}
    for t in thresholds:
        spike_threshold = mean_len + t * std_len
        flagged_count = int((body_lengths > spike_threshold).sum())
        result[str(t)] = {
            "flagged": flagged_count,
            "pct_of_total": round(flagged_count / total, 3) if total > 0 else 0.0,
        }
    return result


def _alert_channel(recommended_action: str) -> str:
    if recommended_action == "escalate":
        return "slack"
    if recommended_action == "create_task":
        return "email"
    return "monitor"


def detect(input_path: str, output_path: str, report_path: str) -> dict:
    """
    Read enriched tickets, run statistical and LLM-based anomaly detection,
    write anomaly records and summary report.
    Returns: {total_tickets, anomalies_found, llm_calls_made}
    """
    df = pd.read_parquet(input_path)
    total_tickets = len(df)

    # --- Statistical detection ---
    body_lengths = df["body"].str.len().dropna()
    if len(body_lengths) >= 2:
        mean_len = body_lengths.mean()
        std_len = body_lengths.std()
        spike_threshold = mean_len + 3 * std_len
    else:
        spike_threshold = float("inf")

    # Check 5: duplicate_body mask computed over the whole dataset
    cleaned = df["body"].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    duplicate_mask = cleaned.duplicated(keep=False) & cleaned.notna() & (cleaned.str.len() > 20)

    flagged: dict[str, list[str]] = {}

    for idx, row in df.iterrows():
        flags = []

        body_val = row.get("body")
        if pd.notna(body_val):
            if len(body_val) > spike_threshold:
                flags.append("body_length_spike")

        sentiment_val = "" if pd.isna(row.get("sentiment")) else str(row.get("sentiment", ""))

        if sentiment_val == "frustrated":
            flags.append("frustrated_sentiment")

        if row.get("llm_category") == "unknown":
            flags.append("unclassified")

        # Check 4: repeat_frustrated — skip gracefully if column absent
        if "repeat_contact" in df.columns:
            repeat_contact = row.get("repeat_contact")
            if pd.notna(repeat_contact) and bool(repeat_contact):
                if sentiment_val in {"frustrated", "negative"}:
                    flags.append("repeat_frustrated")

        # Check 5: duplicate_body
        if duplicate_mask.loc[idx]:
            flags.append("duplicate_body")

        if flags:
            flagged[str(row["ticket_id"])] = flags

    # --- LLM classification ---
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    anomaly_rows = []
    llm_calls_made = 0
    seen_bodies = set()

    for idx, row in df.iterrows():
        ticket_id = str(row["ticket_id"])
        if ticket_id not in flagged:
            continue

        flags = flagged[ticket_id]
        body_val = row.get("body")
        has_body = pd.notna(body_val)
        if has_body and "duplicate_body" in flags:
            body_str = str(body_val).strip().lower()
            if body_str in seen_bodies:
                continue
            seen_bodies.add(body_str)

        statistical_flags_str = "|".join(flags)

        if not has_body:
            anomaly_rows.append({
                "ticket_id": ticket_id,
                "statistical_flags": statistical_flags_str,
                "anomaly_type": flags[0],
                "severity": "low",
                "reason": "Ticket body is null; statistical flag only",
                "recommended_action": "manual_review",
                "detection_method": "statistical",
            })
            continue

        subject = row.get("subject", "") or ""
        sentiment = str(row.get("sentiment") or "")
        llm_category = str(row.get("llm_category") or "")
        body_str = str(body_val)

        result = _classify_anomaly(
            client, ticket_id, subject, body_str, sentiment, llm_category, flags
        )
        llm_calls_made += 1

        if result is not None:
            anomaly_rows.append({
                "ticket_id": ticket_id,
                "statistical_flags": statistical_flags_str,
                "anomaly_type": result.anomaly_type,
                "severity": result.severity,
                "reason": result.reason,
                "recommended_action": result.recommended_action,
                "detection_method": "hybrid",
            })
        else:
            anomaly_rows.append({
                "ticket_id": ticket_id,
                "statistical_flags": statistical_flags_str,
                "anomaly_type": flags[0],
                "severity": "low",
                "reason": "Classification failed",
                "recommended_action": "monitor",
                "detection_method": "hybrid",
            })

    anomalies_found = len(anomaly_rows)

    # --- Write anomalies parquet ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    _COLUMNS = [
        "ticket_id", "statistical_flags", "anomaly_type", "severity",
        "reason", "recommended_action", "detection_method",
    ]
    if anomaly_rows:
        anomalies_df = pd.DataFrame(anomaly_rows, columns=_COLUMNS)
    else:
        anomalies_df = pd.DataFrame(columns=_COLUMNS)

    anomalies_df = anomalies_df.astype(str)
    anomalies_df.to_parquet(output_path, index=False)

    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for row_data in anomaly_rows:
        t = row_data["anomaly_type"]
        by_type[t] = by_type.get(t, 0) + 1
        s = row_data["severity"]
        by_severity[s] = by_severity.get(s, 0) + 1

    # --- Part 2: Threshold Sensitivity Analysis ---
    report_dir = os.path.dirname(report_path)
    if not report_dir:
        report_dir = "."
    os.makedirs(report_dir, exist_ok=True)

    threshold_results = _analyze_thresholds(df["body"].str.len(), [2.0, 2.5, 3.0])
    threshold_analysis = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "check": "body_length_spike",
        "total_tickets": total_tickets,
        "thresholds": threshold_results,
        "selected_threshold": 3.0,
        "rationale": "3.0σ minimizes false positives while catching genuine spikes",
    }
    with open(os.path.join(report_dir, "threshold_analysis.json"), "w") as f:
        json.dump(threshold_analysis, f, indent=2)

    # --- Part 3: Alert Routing ---
    high_severity_alerts = []
    for row_data in anomaly_rows:
        if row_data["severity"] == "high":
            channel = _alert_channel(row_data["recommended_action"])
            msg = (
                f"HIGH anomaly: ticket {row_data['ticket_id']} — {row_data['anomaly_type']}. "
                f"Action: {row_data['recommended_action']}. Reason: {row_data['reason']}"
            )
            high_severity_alerts.append({
                "ticket_id": row_data["ticket_id"],
                "anomaly_type": row_data["anomaly_type"],
                "severity": row_data["severity"],
                "reason": row_data["reason"],
                "recommended_action": row_data["recommended_action"],
                "alert_channel": channel,
                "message": msg,
            })

    with open(os.path.join(report_dir, "high_severity_alerts.json"), "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "alert_count": len(high_severity_alerts),
            "alerts": high_severity_alerts,
        }, f, indent=2)

    # --- Part 4: Trend Comparison ---
    previous_report_path = os.path.join(report_dir, "anomaly_report_previous.json")

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_tickets": total_tickets,
        "anomalies_found": anomalies_found,
        "llm_calls_made": llm_calls_made,
        "by_type": by_type,
        "by_severity": by_severity,
    }

    if os.path.exists(previous_report_path):
        with open(previous_report_path) as f:
            prev = json.load(f)
        prev_anomalies = prev.get("anomalies_found", 0)
        prev_by_type = prev.get("by_type", {})
        all_types = set(by_type.keys()) | set(prev_by_type.keys())
        report["trends"] = {
            "anomalies_found_delta": anomalies_found - prev_anomalies,
            "by_type_delta": {
                t: by_type.get(t, 0) - prev_by_type.get(t, 0)
                for t in all_types
                if (by_type.get(t, 0) - prev_by_type.get(t, 0)) != 0
            },
        }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    shutil.copy2(report_path, previous_report_path)

    logger.info(
        "Total: %d | Anomalies: %d | LLM calls: %d",
        total_tickets, anomalies_found, llm_calls_made,
    )

    return {
        "total_tickets": total_tickets,
        "anomalies_found": anomalies_found,
        "llm_calls_made": llm_calls_made,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = detect(
        "data/enriched/tickets_enriched.parquet",
        "data/anomalies/anomalies.parquet",
        "data/anomalies/anomaly_report.json",
    )
    print(result)
