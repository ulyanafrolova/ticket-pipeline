import datetime
import json
import logging
import os
import time

import anthropic
import pandas as pd
from anthropic import APIStatusError, RateLimitError
from pydantic import BaseModel, field_validator, ValidationError
from typing import Literal

from src.retry import retry

# Retry transient API errors and malformed model output; never retry
# AuthenticationError / BadRequestError (they are not in this tuple).
RETRYABLE_EXCEPTIONS = (RateLimitError, APIStatusError, json.JSONDecodeError, ValidationError)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

HAIKU_INPUT_COST_PER_MILLION = 0.80
HAIKU_OUTPUT_COST_PER_MILLION = 4.00
SONNET_INPUT_COST_PER_MILLION = 3.00
SONNET_OUTPUT_COST_PER_MILLION = 15.00

CLASSIFICATION_SYSTEM_PROMPT = """You are a ticket classification function. Your sole job is to classify support tickets into a category and priority, and return a confidence score.

CATEGORIES:
- billing: payment issues, subscription questions, invoice requests, refunds
- technical: bugs, errors, API issues, feature questions, integration problems
- general: account questions, feedback, greetings, unclear requests, general inquiries
- other: anything that does not fit the above three categories

PRIORITIES:
- critical: service down, data loss, security breach, complete inability to use product
- high: major feature broken, repeated failures, significant business impact
- medium: feature partially working, workaround exists, moderate impact
- low: minor inconvenience, cosmetic issue, informational question

OUTPUT SCHEMA:
{"category": "...", "priority": "...", "confidence": 0.0-1.0}

Output ONLY valid JSON. No explanation. No commentary. No markdown.

## Examples

Input: {"subject": "Invoice shows wrong amount", "body": "My last invoice was for $299 but I'm on the $99 plan."}
Output: {"category": "billing", "priority": "high", "confidence": 0.97}

Input: {"subject": "API 500 error on /data endpoint", "body": "Every POST to /api/v1/data returns 500 since yesterday."}
Output: {"category": "technical", "priority": "high", "confidence": 0.96}

Input: {"subject": "When does my billing cycle reset?", "body": "Just checking when I will be charged next month. No urgent issue."}
Output: {"category": "billing", "priority": "low", "confidence": 0.95}

Input: {"subject": "Entire platform down since maintenance window", "body": "None of our 50 employees can log in or use any feature. We are completely blocked and cannot run our business."}
Output: {"category": "technical", "priority": "critical", "confidence": 0.98}

Input: {"subject": "How do I update my account email address?", "body": "I recently changed my work email and need to update it on my account. No urgency, just housekeeping."}
Output: {"category": "general", "priority": "low", "confidence": 0.94}

Input: {"subject": "Feedback on the new dashboard layout", "body": "The redesigned dashboard is harder to navigate. Our team has to take extra steps every day to generate reports. Please consider reverting the export button location."}
Output: {"category": "general", "priority": "medium", "confidence": 0.88}

Input: {"subject": "Do you offer nonprofit pricing?", "body": "We are a registered 501(c)(3) nonprofit organization and are looking for discounted plans if available."}
Output: {"category": "other", "priority": "low", "confidence": 0.93}

Input: {"subject": "Partnership and co-marketing proposal", "body": "Our company is interested in exploring a co-marketing partnership. Could you connect us with your business development team at your earliest convenience?"}
Output: {"category": "other", "priority": "medium", "confidence": 0.87}"""

class ClassificationResult(BaseModel):
    category: Literal["billing", "technical", "general", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence {v} not in [0, 1]")
        return round(v, 3)


def _first_text_block(response) -> "str | None":
    if response.stop_reason == "refusal":
        logger.warning("Classification request refused (stop_reason=refusal)")
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
def _classify_ticket(
    client: anthropic.Anthropic, subject: str, body: str
) -> "tuple[ClassificationResult | None, dict | None]":
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""

    if not subject.strip() and not body.strip():
        return None, None

    user_content = f"Subject: {subject}\n\nBody: {body}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        temperature=0.0,
        system=[
            {
                "type": "text",
                "text": CLASSIFICATION_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    text = _first_text_block(response)
    if text is None:
        return None, None

    raw = _strip_markdown_fences(text)
    data = json.loads(raw)
    result = ClassificationResult(**data)

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    }
    return result, usage


@retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True,
    retryable_exceptions=RETRYABLE_EXCEPTIONS,
)
def _classify_ticket_sonnet(
    client: anthropic.Anthropic, subject: str, body: str
) -> "tuple[ClassificationResult | None, dict | None]":
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""

    if not subject.strip() and not body.strip():
        return None, None

    user_content = f"Subject: {subject}\n\nBody: {body}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=CLASSIFICATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    text = _first_text_block(response)
    if text is None:
        return None, None

    raw = _strip_markdown_fences(text)
    data = json.loads(raw)
    result = ClassificationResult(**data)

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return result, usage

def classify(
    input_path: str,
    output_path: str,
    sample: int = None,
    max_cost_usd: float = None,
) -> dict:
    """
    Classify tickets. Stops early if max_cost_usd is exceeded.
    Returns: {total, classified, failed, needs_review, escalated_to_sonnet, cost_estimate_usd}
    """
    df = pd.read_parquet(input_path)

    if sample is not None:
        df = df.head(sample)

    total = len(df)
    classified = 0
    failed = 0

    haiku_input_tokens = 0
    haiku_output_tokens = 0
    haiku_cache_read_tokens = 0
    haiku_cache_creation_tokens = 0
    sonnet_input_tokens = 0
    sonnet_output_tokens = 0

    llm_categories = ["unknown"] * total
    llm_priorities = ["unknown"] * total
    llm_confidences = [0.0] * total
    needs_review = [False] * total
    escalated_to_sonnet = [False] * total

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Initial Haiku classification pass in batches of 10
    processed_so_far = 0

    for batch_start in range(0, total, 10):
        cost = (
            haiku_input_tokens / 1_000_000 * HAIKU_INPUT_COST_PER_MILLION
            + haiku_output_tokens / 1_000_000 * HAIKU_OUTPUT_COST_PER_MILLION
            + sonnet_input_tokens / 1_000_000 * SONNET_INPUT_COST_PER_MILLION
            + sonnet_output_tokens / 1_000_000 * SONNET_OUTPUT_COST_PER_MILLION
        )
        if max_cost_usd is not None and cost >= max_cost_usd:
            logger.info(
                "Cost cap reached at %d tickets (estimated $%.4f >= limit $%.4f). Stopping.",
                processed_so_far,
                cost,
                max_cost_usd,
            )
            break

        batch_end = min(batch_start + 10, total)
        for i in range(batch_start, batch_end):
            row = df.iloc[i]
            subject = row.get("subject", "")
            body = row.get("body", "")

            try:
                result, usage = _classify_ticket(client, subject, body)
            except Exception as e:
                logger.error("Classification failed after all retries: %s", e)
                result, usage = None, None

            if result is not None:
                llm_categories[i] = result.category
                llm_priorities[i] = result.priority
                llm_confidences[i] = result.confidence
                classified += 1
                haiku_input_tokens += usage["input_tokens"]
                haiku_output_tokens += usage["output_tokens"]
                haiku_cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
                haiku_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0) or 0
                if result.confidence < 0.70:
                    needs_review[i] = True
            else:
                failed += 1

        processed_so_far = batch_end

    logger.info("Flagged %d tickets for review (confidence < 0.70)", sum(needs_review))

    # Sonnet escalation for low-confidence tickets
    for i in range(total):
        if not needs_review[i]:
            continue

        cost = (
            haiku_input_tokens / 1_000_000 * HAIKU_INPUT_COST_PER_MILLION
            + haiku_output_tokens / 1_000_000 * HAIKU_OUTPUT_COST_PER_MILLION
            + sonnet_input_tokens / 1_000_000 * SONNET_INPUT_COST_PER_MILLION
            + sonnet_output_tokens / 1_000_000 * SONNET_OUTPUT_COST_PER_MILLION
        )
        if max_cost_usd is not None and cost >= max_cost_usd:
            break

        row = df.iloc[i]
        subject = row.get("subject", "")
        body = row.get("body", "")

        try:
            sonnet_result, sonnet_usage = _classify_ticket_sonnet(client, subject, body)
        except Exception as e:
            logger.error("Sonnet escalation failed after all retries: %s", e)
            sonnet_result, sonnet_usage = None, None
        escalated_to_sonnet[i] = True

        if sonnet_usage is not None:
            sonnet_input_tokens += sonnet_usage["input_tokens"]
            sonnet_output_tokens += sonnet_usage["output_tokens"]

        if sonnet_result is not None and sonnet_result.confidence >= 0.85:
            llm_categories[i] = sonnet_result.category
            llm_priorities[i] = sonnet_result.priority
            llm_confidences[i] = sonnet_result.confidence
            needs_review[i] = False

    df = df.copy()
    df["llm_category"] = llm_categories
    df["llm_priority"] = llm_priorities
    df["llm_confidence"] = llm_confidences
    df["needs_review"] = needs_review
    df["escalated_to_sonnet"] = escalated_to_sonnet

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)

    haiku_cost = round(
        haiku_input_tokens / 1_000_000 * HAIKU_INPUT_COST_PER_MILLION
        + haiku_output_tokens / 1_000_000 * HAIKU_OUTPUT_COST_PER_MILLION,
        6,
    )
    sonnet_cost = round(
        sonnet_input_tokens / 1_000_000 * SONNET_INPUT_COST_PER_MILLION
        + sonnet_output_tokens / 1_000_000 * SONNET_OUTPUT_COST_PER_MILLION,
        6,
    )
    cost_estimate_usd = round(haiku_cost + sonnet_cost, 4)

    needs_review_final = int(sum(df["needs_review"]))
    escalated_final = int(sum(df["escalated_to_sonnet"]))

    # Category and priority distributions (classified tickets only)
    category_dist = {"billing": 0, "technical": 0, "general": 0, "other": 0}
    priority_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    cat_counts = df.loc[df["llm_category"] != "unknown", "llm_category"].value_counts().to_dict()
    pri_counts = df.loc[df["llm_priority"] != "unknown", "llm_priority"].value_counts().to_dict()

    category_dist.update(cat_counts)
    priority_dist.update(pri_counts)

    conf_buckets = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    for conf in df["llm_confidence"]:
        if conf < 0.5:
            conf_buckets["0.0-0.5"] += 1
        elif conf < 0.7:
            conf_buckets["0.5-0.7"] += 1
        elif conf < 0.9:
            conf_buckets["0.7-0.9"] += 1
        else:
            conf_buckets["0.9-1.0"] += 1

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_tickets": total,
        "classified": classified,
        "failed": failed,
        "needs_review": needs_review_final,
        "escalated_to_sonnet": escalated_final,
        "haiku_cost_usd": haiku_cost,
        "sonnet_cost_usd": sonnet_cost,
        "total_cost_usd": cost_estimate_usd,
        "category_distribution": category_dist,
        "priority_distribution": priority_dist,
        "confidence_buckets": conf_buckets,
    }

    report_dir = output_dir if output_dir else "."
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    total_input_with_cache = haiku_input_tokens + haiku_cache_read_tokens
    cache_hit_rate = round(haiku_cache_read_tokens / total_input_with_cache, 4) if total_input_with_cache > 0 else 0.0

    logger.info(
        "Total: %d | Classified: %d | Failed: %d | Needs Review: %d | Escalated: %d | "
        "Haiku Cost: $%.4f | Sonnet Cost: $%.4f | Total Cost: $%.4f | "
        "Cache Hit Rate: %.4f (cache_read=%d / total_input=%d)",
        total, classified, failed, needs_review_final, escalated_final,
        haiku_cost, sonnet_cost, cost_estimate_usd,
        cache_hit_rate, haiku_cache_read_tokens, total_input_with_cache,
    )
    return {
        "total": total,
        "classified": classified,
        "failed": failed,
        "needs_review": needs_review_final,
        "escalated_to_sonnet": escalated_final,
        "cost_estimate_usd": cost_estimate_usd,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-cost", type=float, default=None, dest="max_cost")
    parser.add_argument("--input", default="data/processed/tickets_validated.parquet")
    parser.add_argument("--output", default="data/enriched/tickets_classified.parquet")
    args = parser.parse_args()

    result = classify(args.input, args.output, sample=args.sample, max_cost_usd=args.max_cost)
    print(result)
