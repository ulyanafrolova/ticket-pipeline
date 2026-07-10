import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import anthropic
import pandas as pd
from anthropic import APIStatusError, RateLimitError
from pydantic import ValidationError

from src.retry import retry

# Retry transient API errors and malformed model output; never retry
# AuthenticationError / BadRequestError (they are not in this tuple).
RETRYABLE_EXCEPTIONS = (RateLimitError, APIStatusError, json.JSONDecodeError, ValidationError)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

ACTIONS_LOG = "data/agent/actions.jsonl"
REASONING_LOG = "data/agent/reasoning.jsonl"
PENDING_APPROVAL_LOG = "data/agent/pending_approval.jsonl"
AGENT_SUMMARY = "data/agent/agent_summary.json"
MAX_ITERATIONS = 10


def _detect_platform() -> str:
    """Return 'aws' or 'azure'. Raise EnvironmentError if neither is configured."""
    has_aws = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_azure = bool(os.environ.get("AZURE_PROJECT_CONNECTION_STRING"))

    if has_aws and has_azure:
        logging.warning(
            "Both ANTHROPIC_API_KEY and AZURE_PROJECT_CONNECTION_STRING are set — using AWS. "
            "Unset ANTHROPIC_API_KEY to switch to Azure."
        )
        return "aws"
    if has_aws:
        return "aws"
    if has_azure:
        return "azure"
    raise EnvironmentError(
        "Agent platform not configured. "
        "Set ANTHROPIC_API_KEY (AWS) or AZURE_PROJECT_CONNECTION_STRING (Azure)."
    )


def _estimate_tokens(messages: list) -> int:
    """Rough estimate: count characters in all message content, divide by 4."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                total_chars += len(str(item))
    return total_chars // 4


# Tool definitions (Anthropic format for AWS path)

TOOLS = [
    {
        "name": "escalate_ticket",
        "description": "Escalate a ticket to a senior support agent for immediate handling.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to escalate",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this ticket needs immediate escalation",
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Urgency level",
                },
            },
            "required": ["ticket_id", "reason", "severity"],
        },
    },
    {
        "name": "send_alert",
        "description": "Send an alert notification to the support team about an anomalous ticket pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The alert message to send",
                },
                "channel": {
                    "type": "string",
                    "enum": ["slack", "email", "pagerduty"],
                    "description": "Notification channel",
                },
            },
            "required": ["message", "channel"],
        },
    },
    {
        "name": "create_task",
        "description": "Create a follow-up task in the task management system for a ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Task title",
                },
                "description": {
                    "type": "string",
                    "description": "What needs to be done",
                },
                "assignee": {
                    "type": "string",
                    "enum": ["tier1", "tier2", "billing_team", "engineering"],
                    "description": "Who should handle it",
                },
            },
            "required": ["title", "description", "assignee"],
        },
    },
    {
        "name": "auto_respond",
        "description": "Send an automated response to the customer acknowledging their ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to respond to",
                },
                "response_template": {
                    "type": "string",
                    "enum": [
                        "acknowledge_delay",
                        "sla_commitment",
                        "escalation_notice",
                        "general_thanks",
                    ],
                    "description": "Which response template to use",
                },
            },
            "required": ["ticket_id", "response_template"],
        },
    },
    {
        "name": "get_ticket_history",
        "description": "Retrieve the support history for a customer to provide context for routing decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer whose history to retrieve",
                },
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "update_ticket_status",
        "description": "Update the status of a ticket in the support system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to update",
                },
                "new_status": {
                    "type": "string",
                    "enum": ["open", "pending", "closed"],
                    "description": "The new status",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the status is being changed",
                },
            },
            "required": ["ticket_id", "new_status", "reason"],
        },
    },
]


# Logging helpers

def _log_action(step: int, tool: str, input_data: dict, result: str):
    os.makedirs(os.path.dirname(ACTIONS_LOG), exist_ok=True)
    with open(ACTIONS_LOG, "a") as f:
        f.write(json.dumps({
            "step": step,
            "tool": tool,
            "input": input_data,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


def _log_reasoning(entry: dict):
    os.makedirs(os.path.dirname(REASONING_LOG), exist_ok=True)
    with open(REASONING_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _queue_pending_approval(ticket_id: str, tool: str, input_data: dict, anomaly_type: str):
    os.makedirs(os.path.dirname(PENDING_APPROVAL_LOG), exist_ok=True)
    with open(PENDING_APPROVAL_LOG, "a") as f:
        f.write(json.dumps({
            "ticket_id": ticket_id,
            "tool": tool,
            "input": input_data,
            "anomaly_type": anomaly_type,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


# Tool handlers

def handle_escalate_ticket(ticket_id: str, reason: str, severity: str) -> str:
    return f"Escalated {ticket_id} to escalation queue. Severity: {severity}."


def handle_send_alert(message: str, channel: str) -> str:
    return f"Alert sent to {channel}: {message[:100]}"


def handle_create_task(title: str, description: str, assignee: str) -> str:
    return f"Task created: '{title}' assigned to {assignee}"


def handle_auto_respond(ticket_id: str, response_template: str) -> str:
    return f"Auto-response '{response_template}' queued for {ticket_id}"


def handle_get_ticket_history(customer_id: str) -> str:
    templates = [
        "Customer has 3 prior tickets: 2 closed, 1 pending. Last contact: 14 days ago.",
        "Customer has 1 prior ticket: 1 closed. Last contact: 30 days ago.",
        "Customer has 7 prior tickets: 5 closed, 2 pending. Last contact: 3 days ago.",
    ]
    return templates[hash(customer_id) % len(templates)]


def handle_update_ticket_status(ticket_id: str, new_status: str, reason: str) -> str:
    return f"Ticket {ticket_id} status updated to {new_status}."


TOOL_HANDLERS = {
    "escalate_ticket": handle_escalate_ticket,
    "send_alert": handle_send_alert,
    "create_task": handle_create_task,
    "auto_respond": handle_auto_respond,
    "get_ticket_history": handle_get_ticket_history,
    "update_ticket_status": handle_update_ticket_status,
}


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _write_agent_summary(anomalies_processed: int, total_tool_calls: int, context_truncations: int):
    actions_taken = _count_lines(ACTIONS_LOG)
    pending_approval = _count_lines(PENDING_APPROVAL_LOG)

    tool_distribution = {
        "escalate_ticket": 0,
        "send_alert": 0,
        "create_task": 0,
        "auto_respond": 0,
        "get_ticket_history": 0,
        "update_ticket_status": 0,
    }
    if os.path.exists(ACTIONS_LOG):
        with open(ACTIONS_LOG) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    tool_name = entry.get("tool", "")
                    if tool_name in tool_distribution:
                        tool_distribution[tool_name] += 1

    avg = round(total_tool_calls / anomalies_processed, 2) if anomalies_processed > 0 else 0.0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "anomalies_processed": anomalies_processed,
        "total_tool_calls": total_tool_calls,
        "actions_taken": actions_taken,
        "pending_approval": pending_approval,
        "tool_distribution": tool_distribution,
        "avg_tool_calls_per_anomaly": avg,
        "context_truncations": context_truncations,
    }

    os.makedirs(os.path.dirname(AGENT_SUMMARY), exist_ok=True)
    with open(AGENT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)


# AWS path

def _run_agent_for_anomaly(client: anthropic.Anthropic, anomaly: dict):
    """
    Run the tool-use loop for one anomaly record.
    Returns: (tool_calls_made, context_truncations)
    """
    user_message = (
        f"Analyze this support ticket anomaly and take the appropriate actions.\n\n"
        f"ticket_id: {anomaly.get('ticket_id')}\n"
        f"anomaly_type: {anomaly.get('anomaly_type')}\n"
        f"severity: {anomaly.get('severity')}\n"
        f"reason: {anomaly.get('reason')}\n"
        f"recommended_action: {anomaly.get('recommended_action')}"
    )
    messages = [{"role": "user", "content": user_message}]

    tool_calls_made = 0
    context_truncations = 0
    iterations = 0

    while True:
        if iterations >= MAX_ITERATIONS:
            logger.warning(
                "Max iterations (%d) reached for ticket %s",
                MAX_ITERATIONS,
                anomaly.get("ticket_id"),
            )
            break

        # Context length guard
        n = _estimate_tokens(messages)
        if n > 40_000:
            first = messages[:1]
            last_4 = messages[-4:] if len(messages) > 5 else messages[1:]
            summary_msg = {"role": "user", "content": "[Earlier turns summarized for context length management]"}
            messages = first + [summary_msg] + last_4
            context_truncations += 1
            logger.info("Context truncated: estimated %d tokens exceeded 40k limit", n)

        @retry(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            jitter=True,
            retryable_exceptions=RETRYABLE_EXCEPTIONS,
        )
        def _call_llm():
            return client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            )

        response = _call_llm()

        iterations += 1

        messages.append({"role": "assistant", "content": response.content})

        tool_called = None
        tool_use_id = None
        for block in response.content:
            if block.type == "tool_use":
                tool_called = block.name
                tool_use_id = block.id
                break

        _log_reasoning({
            "step": iterations,
            "stop_reason": response.stop_reason,
            "tool_called": tool_called,
            "tool_use_id": tool_use_id,
        })

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # Human-in-the-loop: queue high-severity escalations instead of executing
            if block.name == "escalate_ticket" and block.input.get("severity") == "high":
                ticket_id = block.input.get("ticket_id", "unknown")
                _queue_pending_approval(
                    ticket_id,
                    block.name,
                    block.input,
                    anomaly.get("anomaly_type", ""),
                )
                logger.info("HIGH severity escalation queued for approval: ticket %s", ticket_id)
                tool_calls_made += 1
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Escalation for ticket {ticket_id} queued for human approval.",
                })
                continue

            handler = TOOL_HANDLERS.get(block.name)
            if handler is not None:
                result = handler(**block.input)
            else:
                result = f"Error: tool '{block.name}' not found"
            _log_action(iterations, block.name, block.input, result)
            tool_calls_made += 1
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return tool_calls_made, context_truncations


# Azure path

def _run_agent_azure_for_anomaly(project_client, agent_id: str, anomaly: dict):
    """
    Run the Azure AI Agents loop for one anomaly record.
    Returns: (tool_calls_made, context_truncations)
    """
    import time

    user_message = (
        f"Analyze this support ticket anomaly and take the appropriate actions.\n\n"
        f"ticket_id: {anomaly.get('ticket_id')}\n"
        f"anomaly_type: {anomaly.get('anomaly_type')}\n"
        f"severity: {anomaly.get('severity')}\n"
        f"reason: {anomaly.get('reason')}\n"
        f"recommended_action: {anomaly.get('recommended_action')}"
    )

    thread = project_client.agents.threads.create()
    project_client.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = project_client.agents.runs.create(
        thread_id=thread.id,
        agent_id=agent_id,
    )

    tool_calls_made = 0
    step = 0

    while run.status not in ("completed", "failed"):
        time.sleep(0.5)
        run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action":
            step += 1
            tool_outputs = []

            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                name = tool_call.function.name
                try:
                    input_data = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    input_data = {}

                # Human-in-the-loop: queue high-severity escalations
                if name == "escalate_ticket" and input_data.get("severity") == "high":
                    ticket_id = input_data.get("ticket_id", "unknown")
                    _queue_pending_approval(
                        ticket_id,
                        name,
                        input_data,
                        anomaly.get("anomaly_type", ""),
                    )
                    logger.info("HIGH severity escalation queued for approval: ticket %s", ticket_id)
                    tool_calls_made += 1
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Escalation for ticket {ticket_id} queued for human approval.",
                    })
                    _log_reasoning({
                        "step": step,
                        "stop_reason": "requires_action",
                        "tool_called": name,
                        "tool_use_id": tool_call.id,
                    })
                    continue

                handler = TOOL_HANDLERS.get(name)
                if handler is not None:
                    result = handler(**input_data)
                else:
                    result = f"Error: tool '{name}' not found"

                _log_action(step, name, input_data, result)
                tool_calls_made += 1

                _log_reasoning({
                    "step": step,
                    "stop_reason": "requires_action",
                    "tool_called": name,
                    "tool_use_id": tool_call.id,
                })

                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": result,
                })

            run = project_client.agents.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )

    _log_reasoning({
        "step": step + 1,
        "stop_reason": run.status,
        "tool_called": None,
        "tool_use_id": None,
    })

    project_client.agents.threads.delete(thread_id=thread.id)

    return tool_calls_made, 0


def run_agent(anomalies_path: str = "data/anomalies/anomalies.parquet", backend=None) -> dict:
    """
    Read anomaly records, run the Claude agent to route each anomaly via tool use.
    Returns: {anomalies_processed, actions_taken, total_tool_calls}
    """
    df = pd.read_parquet(anomalies_path)

    # Clear log files from previous runs (idempotency)
    for path in [ACTIONS_LOG, REASONING_LOG, PENDING_APPROVAL_LOG]:
        if os.path.exists(path):
            os.remove(path)

    if backend is None:
        platform = _detect_platform()
    elif backend == "python":
        platform = "aws"
    else:
        platform = "azure"

    logger.info("[agent] platform=%s", platform)

    total_tool_calls = 0
    context_truncations = 0

    if platform == "azure":
        from azure.ai.projects import AIProjectClient
        from azure.ai.projects.models import FunctionTool, ToolSet
        from azure.identity import DefaultAzureCredential

        def escalate_ticket(
            ticket_id: str,
            reason: str,
            severity: Literal["low", "medium", "high"],
        ) -> str:
            """Escalate a ticket to a senior support agent for immediate handling."""
            return handle_escalate_ticket(ticket_id, reason, severity)

        def send_alert(
            message: str,
            channel: Literal["slack", "email", "pagerduty"],
        ) -> str:
            """Send an alert notification to the support team about an anomalous ticket pattern."""
            return handle_send_alert(message, channel)

        def create_task(
            title: str,
            description: str,
            assignee: Literal["tier1", "tier2", "billing_team", "engineering"],
        ) -> str:
            """Create a follow-up task in the task management system for a ticket."""
            return handle_create_task(title, description, assignee)

        def auto_respond(
            ticket_id: str,
            response_template: Literal[
                "acknowledge_delay", "sla_commitment", "escalation_notice", "general_thanks"
            ],
        ) -> str:
            """Send an automated response to the customer acknowledging their ticket."""
            return handle_auto_respond(ticket_id, response_template)

        def get_ticket_history(customer_id: str) -> str:
            """Retrieve the support history for a customer to provide context for routing decisions."""
            return handle_get_ticket_history(customer_id)

        def update_ticket_status(
            ticket_id: str,
            new_status: Literal["open", "pending", "closed"],
            reason: str,
        ) -> str:
            """Update the status of a ticket in the support system."""
            return handle_update_ticket_status(ticket_id, new_status, reason)

        functions = FunctionTool(
            functions={escalate_ticket, send_alert, create_task, auto_respond,
                       get_ticket_history, update_ticket_status}
        )
        toolset = ToolSet()
        toolset.add(functions)

        project_client = AIProjectClient.from_connection_string(
            conn_str=os.environ["AZURE_PROJECT_CONNECTION_STRING"],
            credential=DefaultAzureCredential(),
        )

        az_agent = project_client.agents.create_agent(
            model="claude-sonnet-4-6",
            name="ticket-anomaly-router",
            instructions=(
                "You are a ticket anomaly routing agent for a customer support system. "
                "For each anomaly, analyze the details and use the available tools to take "
                "appropriate action: escalate high-severity tickets, send alerts for anomalous "
                "patterns, create follow-up tasks, or send automated responses."
            ),
            toolset=toolset,
        )

        try:
            for _, row in df.iterrows():
                tool_calls, trunc = _run_agent_azure_for_anomaly(
                    project_client, az_agent.id, row.to_dict()
                )
                total_tool_calls += tool_calls
                context_truncations += trunc
        finally:
            project_client.agents.delete_agent(az_agent.id)

    else:
        client = anthropic.Anthropic()
        for _, row in df.iterrows():
            tool_calls, trunc = _run_agent_for_anomaly(client, row.to_dict())
            total_tool_calls += tool_calls
            context_truncations += trunc

    _write_agent_summary(len(df), total_tool_calls, context_truncations)

    return {
        "anomalies_processed": len(df),
        "actions_taken": _count_lines(ACTIONS_LOG),
        "total_tool_calls": total_tool_calls,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ticket anomaly routing agent")
    parser.add_argument(
        "--backend",
        choices=["python", "azure"],
        default=None,
        help="Backend to use: 'python' for Anthropic API, 'azure' for Azure AI Agents SDK",
    )
    args = parser.parse_args()
    result = run_agent(backend=args.backend)
    print(json.dumps(result, indent=2))
