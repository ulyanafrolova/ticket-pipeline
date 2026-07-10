"""
Central schema definitions for the ticket pipeline.
Every other module (transform, validate, downstream consumers) must import
its schema constants from here so there is a single source of truth.
"""

EXPECTED_COLUMNS = [
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
]

VALID_CHANNELS = {"email", "chat", "phone", "web"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_STATUSES = {"open", "closed", "pending"}
VALID_CATEGORIES = {"billing", "technical", "general", "other"}

NORMALIZED_SCHEMA_VERSION = "1.0"

# UUID v4 pattern used by Rule 9 to validate ticket_id format.
UUID_V4_PATTERN = (
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
