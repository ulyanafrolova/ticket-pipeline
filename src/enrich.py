import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import anthropic
import pandas as pd
from anthropic import APIStatusError, RateLimitError
from pydantic import BaseModel, field_validator, ValidationError

from src.retry import retry

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Retry transient API errors and malformed model output; never retry
# AuthenticationError / BadRequestError (they are not in this tuple).
RETRYABLE_EXCEPTIONS = (RateLimitError, APIStatusError, json.JSONDecodeError, ValidationError)

HAIKU_INPUT_COST_PER_MILLION = 0.80
HAIKU_OUTPUT_COST_PER_MILLION = 4.00
SONNET_INPUT_COST_PER_MILLION = 3.00
SONNET_OUTPUT_COST_PER_MILLION = 15.00

ENRICHMENT_SYSTEM_PROMPT = """You are a ticket enrichment function. Your job is to analyze support ticket text and return sentiment analysis, entity extraction, a summary, and urgency assessment in a single JSON response.

SENTIMENT DEFINITIONS:
- positive: customer is happy, satisfied, or expressing gratitude
- neutral: transactional or informational tone, no strong emotion
- negative: customer is dissatisfied or frustrated but calm
- frustrated: customer is upset, mentions repeated issues, or has an urgent/escalated tone

ENTITY EXTRACTION:
Extract ONLY entities explicitly mentioned in the text. Do not infer or guess entities that are not directly stated.
- product_names: product or service names mentioned
- error_codes: error codes, HTTP status codes, or numeric error identifiers
- account_ids: account numbers, user IDs, or customer identifiers

SUMMARY:
Write exactly one sentence in past tense describing the customer's specific issue.

URGENCY SCORE (0.0 to 1.0):
- 1.0: data loss, security incident, complete outage
- 0.7-0.9: major feature broken, repeated failures
- 0.4-0.6: partial functionality, workaround exists
- 0.0-0.3: informational, cosmetic, minor inconvenience

URGENCY REASON:
One sentence explaining why this ticket is urgent or not.

OUTPUT SCHEMA:
{
  "sentiment": "<positive|neutral|negative|frustrated>",
  "sentiment_confidence": <0.0-1.0>,
  "entities": {
    "product_names": [],
    "error_codes": [],
    "account_ids": []
  },
  "summary": "<one sentence>",
  "urgency_score": <0.0-1.0>,
  "urgency_reason": "<one sentence>"
}

Output ONLY valid JSON. No explanation. No markdown."""

_SIMPLIFIED_PROMPT = """Analyze this support ticket and return ONLY valid JSON with these two fields:
{"sentiment": "<positive|neutral|negative|frustrated>", "summary": "<one sentence in past tense describing the issue>"}

Output ONLY valid JSON. No explanation. No markdown."""

_URGENCY_FALLBACK_SCORE = 0.5
_URGENCY_FALLBACK_REASON = "Classification failed; manual review required"


class EntitySet(BaseModel):
    product_names: list[str] = []
    error_codes: list[str] = []
    account_ids: list[str] = []


class EnrichmentResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "frustrated"]
    sentiment_confidence: float
    entities: EntitySet
    summary: str
    urgency_score: float
    urgency_reason: str

    @field_validator("sentiment_confidence")
    @classmethod
    def confidence_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{v} out of range")
        return round(v, 3)

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("summary must not be empty")
        return v.strip()

    @field_validator("urgency_score")
    @classmethod
    def urgency_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{v} out of range")
        return round(v, 2)

    @field_validator("urgency_reason")
    @classmethod
    def urgency_reason_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("urgency_reason must not be empty")
        return v.strip()


def _first_text_block(response) -> "str | None":
    if response.stop_reason == "refusal":
        logger.warning("Enrichment request refused (stop_reason=refusal)")
        return None
    return next((b.text for b in response.content if b.type == "text"), None)


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

@retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True,
    retryable_exceptions=RETRYABLE_EXCEPTIONS,
)
def _enrich_ticket(
    client: anthropic.Anthropic, subject: str, body: str
) -> "tuple[EnrichmentResult | None, dict | None]":
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""

    if not subject.strip() and not body.strip():
        return None, None

    user_content = f"Subject: {subject}\n\nBody: {body}"

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            temperature=0.0,
            system=[{
                "type": "text",
                "text": ENRICHMENT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_content}],
        )
        text = _first_text_block(response)
        if text is None:
            return None, None
        raw = _strip_markdown_fences(text)
        data = json.loads(raw)
        result = EnrichmentResult(**data)
        usage = {
            "model": "haiku",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        }
        return result, usage

    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("Attempt 1 full enrichment failed: %s. Trying simplified prompt...", e)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=_SIMPLIFIED_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = _first_text_block(response)
        if text is None:
            return None, None
        raw = _strip_markdown_fences(text)
        data = json.loads(raw)
        result = EnrichmentResult(
            sentiment=data["sentiment"],
            sentiment_confidence=0.5,
            entities=EntitySet(),
            summary=data["summary"],
            urgency_score=_URGENCY_FALLBACK_SCORE,
            urgency_reason=_URGENCY_FALLBACK_REASON,
        )
        usage = {
            "model": "sonnet",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        return result, usage

def _flag_repeat_contacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add repeat_contact (bool) column.
    True if the customer_id appears in 3 or more rows in this dataset.
    """
    counts = df["customer_id"].value_counts()
    repeat_ids = set(counts[counts >= 3].index)
    df = df.copy()
    df["repeat_contact"] = df["customer_id"].isin(repeat_ids)
    n_tickets = int(df["repeat_contact"].sum())
    k_customers = len(repeat_ids)
    logger.info(
        "Flagged %d tickets as repeat contacts (%d unique customers with 3+ tickets)",
        n_tickets, k_customers,
    )
    return df


def enrich(input_path: str, output_path: str, resume: bool = False) -> dict:
    """
    Read classified ticket Parquet, enrich each ticket with sentiment, entities, summary, and urgency.
    Uses ThreadPoolExecutor for concurrent API calls.
    Returns: {total, enriched, failed, tokens_used, cost_usd, repeat_contacts}
    """
    empty_entities_json = json.dumps({"product_names": [], "error_codes": [], "account_ids": []})

    if resume and os.path.exists(output_path):
        df = pd.read_parquet(output_path)
        failed_mask = df["enrichment_failed"].astype(bool)
        indices_to_process = [i for i, v in enumerate(failed_mask) if v]
        logger.info("Resume mode: re-processing %d previously failed rows", len(indices_to_process))
        sentiments = df["sentiment"].tolist()
        sentiment_confidences = df["sentiment_confidence"].tolist()
        entities_jsons = df["entities_json"].tolist()
        summaries = df["summary"].tolist()
        urgency_scores = df["urgency_score"].tolist()
        urgency_reasons = df["urgency_reason"].tolist()
        enrichment_faileds = df["enrichment_failed"].tolist()
    else:
        df = pd.read_parquet(input_path)
        indices_to_process = list(range(len(df)))
        total = len(df)
        sentiments = [None] * total
        sentiment_confidences = [None] * total
        entities_jsons = [empty_entities_json] * total
        summaries = [None] * total
        urgency_scores = [_URGENCY_FALLBACK_SCORE] * total
        urgency_reasons = [_URGENCY_FALLBACK_REASON] * total
        enrichment_faileds = [True] * total

    total = len(df)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = [(None, None)] * total
    future_to_index = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        for i in indices_to_process:
            row = df.iloc[i]
            subject = row.get("subject", "")
            body = row.get("body", "")
            future = executor.submit(_enrich_ticket, client, subject, body)
            future_to_index[future] = i

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as e:
                logger.error("Unexpected error for row %d: %s", i, e)
                results[i] = (None, None)

    haiku_input_tokens = 0
    haiku_output_tokens = 0
    sonnet_input_tokens = 0
    sonnet_output_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0

    for i in indices_to_process:
        result, usage = results[i]
        if result is not None:
            sentiments[i] = result.sentiment
            sentiment_confidences[i] = result.sentiment_confidence
            entities_jsons[i] = json.dumps(result.entities.model_dump())
            summaries[i] = result.summary
            urgency_scores[i] = result.urgency_score
            urgency_reasons[i] = result.urgency_reason
            enrichment_faileds[i] = False
        else:
            sentiments[i] = None
            sentiment_confidences[i] = None
            entities_jsons[i] = empty_entities_json
            summaries[i] = None
            urgency_scores[i] = _URGENCY_FALLBACK_SCORE
            urgency_reasons[i] = _URGENCY_FALLBACK_REASON
            enrichment_faileds[i] = True

        if usage is not None:
            if usage.get("model") == "sonnet":
                sonnet_input_tokens += usage["input_tokens"]
                sonnet_output_tokens += usage["output_tokens"]
            else:
                haiku_input_tokens += usage["input_tokens"]
                haiku_output_tokens += usage["output_tokens"]
            cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
            cache_creation_tokens += usage.get("cache_creation_input_tokens", 0) or 0

    df = df.copy()
    df["sentiment"] = sentiments
    df["sentiment_confidence"] = sentiment_confidences
    df["entities_json"] = entities_jsons
    df["summary"] = summaries
    df["urgency_score"] = urgency_scores
    df["urgency_reason"] = urgency_reasons
    df["enrichment_failed"] = enrichment_faileds
    df["enrichment_failed"] = df["enrichment_failed"].astype(bool)

    df = _flag_repeat_contacts(df)
    df["repeat_contact"] = df["repeat_contact"].astype(bool)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)

    total_tokens = (
        haiku_input_tokens + haiku_output_tokens
        + sonnet_input_tokens + sonnet_output_tokens
    )
    cost = (
        haiku_input_tokens / 1_000_000 * HAIKU_INPUT_COST_PER_MILLION
        + haiku_output_tokens / 1_000_000 * HAIKU_OUTPUT_COST_PER_MILLION
        + sonnet_input_tokens / 1_000_000 * SONNET_INPUT_COST_PER_MILLION
        + sonnet_output_tokens / 1_000_000 * SONNET_OUTPUT_COST_PER_MILLION
    )

    # Cache hit rate over the cached (haiku) call path: cache-read tokens as a
    # share of all prompt tokens that could have been served from cache.
    total_input_with_cache = haiku_input_tokens + cache_read_tokens
    cache_hit_rate = (
        round(cache_read_tokens / total_input_with_cache, 4)
        if total_input_with_cache > 0
        else 0.0
    )

    enriched_count = int((~df["enrichment_failed"]).sum())
    failed_count = int(df["enrichment_failed"].sum())
    repeat_contacts_count = int(df["repeat_contact"].sum())

    sentiment_dist = {"positive": 0, "neutral": 0, "negative": 0, "frustrated": 0}
    sent_counts = df.loc[df["sentiment"].notna(), "sentiment"].value_counts().to_dict()
    sentiment_dist.update(sent_counts)

    urgency_dist = {"low (0.0-0.3)": 0, "medium (0.4-0.6)": 0, "high (0.7-1.0)": 0}
    for score in df.loc[df["urgency_score"].notna(), "urgency_score"]:
        if score <= 0.3:
            urgency_dist["low (0.0-0.3)"] += 1
        elif score <= 0.6:
            urgency_dist["medium (0.4-0.6)"] += 1
        else:
            urgency_dist["high (0.7-1.0)"] += 1

    enrichment_failed_pct = round(failed_count / total, 4) if total > 0 else 0.0

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total": total,
        "enriched": enriched_count,
        "failed": failed_count,
        "repeat_contacts": repeat_contacts_count,
        "tokens_used": total_tokens,
        "cost_usd": round(cost, 4),
        "cache_hit_rate": cache_hit_rate,
        "sentiment_distribution": sentiment_dist,
        "urgency_distribution": urgency_dist,
        "enrichment_failed_pct": enrichment_failed_pct,
    }

    report_dir = output_dir if output_dir else "."
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "enrichment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Total: %d | Enriched: %d | Failed: %d | Repeat Contacts: %d | Tokens: %d | "
        "Cost: $%.4f | Cache Hit Rate: %.4f (cache_read=%d / total_input=%d)",
        total, enriched_count, failed_count, repeat_contacts_count, total_tokens,
        round(cost, 4), cache_hit_rate, cache_read_tokens, total_input_with_cache,
    )

    return {
        "total": total,
        "enriched": enriched_count,
        "failed": failed_count,
        "tokens_used": total_tokens,
        "cost_usd": round(cost, 4),
        "cache_hit_rate": cache_hit_rate,
        "repeat_contacts": repeat_contacts_count,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/enriched/tickets_classified.parquet")
    parser.add_argument("--output", default="data/enriched/tickets_enriched.parquet")
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    result = enrich(args.input, args.output, resume=args.resume)
    print(result)
