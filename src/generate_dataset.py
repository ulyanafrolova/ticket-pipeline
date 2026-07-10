import argparse
import os
import random
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from faker import Faker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic support ticket dataset."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of synthetic tickets to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tickets.csv",
        help="Output CSV file path.",
    )
    return parser.parse_args()


def build_ticket(fake: Faker) -> dict:
    return {
        "ticket_id": str(uuid.uuid4()),
        "created_at": fake.date_time_between(start_date='-1y', end_date='now').isoformat(timespec='seconds'),
        "customer_id": str(uuid.uuid4()),
        "channel": random.choice(["email", "chat", "phone", "web"]),
        "subject": fake.sentence(nb_words=random.randint(3, 10))[:80],
        "body": fake.paragraph(nb_sentences=random.randint(2, 6)),
        "priority": random.choice(["low", "medium", "high", "critical"]),
        "category": random.choice(["billing", "technical", "general", "other"]),
        "status": random.choice(["open", "closed", "pending"]),
        "agent_id": str(uuid.uuid4()),
    }


def inject_quality_issues(rows: list[dict], fake: Faker) -> None:
    count = len(rows)
    subject_issue_count = max(1, round(count * 0.05))
    priority_issue_count = max(1, round(count * 0.08))
    duplicate_id_count = max(1, round(count * 0.03))
    malformed_created_count = max(1, round(count * 0.02))
    short_body_count = max(1, round(count * 0.01))

    indices = list(range(count))
    random.shuffle(indices)

    for idx in indices[:subject_issue_count]:
        rows[idx]["subject"] = None if random.random() < 0.5 else ""

    for idx in indices[subject_issue_count : subject_issue_count + priority_issue_count]:
        rows[idx]["priority"] = None if random.random() < 0.5 else ""

    for dup_idx, target_idx in zip(
        indices[subject_issue_count + priority_issue_count : subject_issue_count + priority_issue_count + duplicate_id_count],
        indices[:duplicate_id_count],
    ):
        rows[dup_idx]["ticket_id"] = rows[target_idx]["ticket_id"]

    malformed_values = ["yesterday", "N/A", "unknown", "31/02/2024", "2024-13-01T00:00:00"]
    for idx in indices[
        subject_issue_count + priority_issue_count + duplicate_id_count :
        subject_issue_count + priority_issue_count + duplicate_id_count + malformed_created_count
    ]:
        rows[idx]["created_at"] = random.choice(malformed_values)

    for idx in indices[
        subject_issue_count + priority_issue_count + duplicate_id_count + malformed_created_count :
        subject_issue_count + priority_issue_count + duplicate_id_count + malformed_created_count + short_body_count
    ]:
        rows[idx]["body"] = fake.word()[:9]


def ensure_output_dir(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    fake = Faker(seed=args.seed)

    output_path = Path(args.output)
    ensure_output_dir(output_path)

    rows: list[dict] = []
    for i in range(1, args.count + 1):
        rows.append(build_ticket(fake))
        print(f"Generated {i}/{args.count} tickets...")

    inject_quality_issues(rows, fake)

    df = pd.DataFrame(rows, columns=[
        "ticket_id",
        "created_at",
        "customer_id",
        "channel",
        "subject",
        "body",
        "priority",
        "category",
        "status",
        "agent_id",
    ])
    df.to_csv(output_path, index=False)

    print(f"Wrote {output_path.resolve()}.")


if __name__ == "__main__":
    main()
