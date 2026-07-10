import json
import logging
import os
from unittest.mock import MagicMock, patch

import anthropic
import pandas as pd
import pytest

from src import agent


# ─────────────────────────── mock response helpers ──────────────────────────

def _text_block():
    b = MagicMock()
    b.type = "text"
    b.text = "Done."
    return b


def _tool_block(name, tool_id, input_data):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.id = tool_id
    b.input = input_data
    return b


def _end_turn():
    r = MagicMock()
    r.content = [_text_block()]
    r.stop_reason = "end_turn"
    return r


def _tool_use(name, tool_id, input_data):
    r = MagicMock()
    r.content = [_tool_block(name, tool_id, input_data)]
    r.stop_reason = "tool_use"
    return r


def _make_client(*responses):
    c = MagicMock(spec=anthropic.Anthropic)
    c.messages.create.side_effect = list(responses)
    return c


def _count_nonempty_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


# ─────────────────────────── shared fixtures ────────────────────────────────

ANOMALY = {
    "ticket_id": "T001",
    "anomaly_type": "spike",
    "severity": "medium",
    "reason": "Unusual pattern detected",
    "recommended_action": "escalate",
}


@pytest.fixture(autouse=True)
def _tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ACTIONS_LOG", str(tmp_path / "actions.jsonl"))
    monkeypatch.setattr(agent, "REASONING_LOG", str(tmp_path / "reasoning.jsonl"))
    monkeypatch.setattr(agent, "PENDING_APPROVAL_LOG", str(tmp_path / "pending.jsonl"))
    monkeypatch.setattr(agent, "AGENT_SUMMARY", str(tmp_path / "summary.json"))


# ──────────────────────────────── tests ─────────────────────────────────────

def test_tool_dispatch_escalate():
    """escalate_ticket tool → handle_escalate_ticket called."""
    inp = {"ticket_id": "T001", "reason": "urgent", "severity": "medium"}
    c = _make_client(_tool_use("escalate_ticket", "id1", inp), _end_turn())
    mock_handler = MagicMock(return_value="Escalated T001.")

    with patch.dict(agent.TOOL_HANDLERS, {"escalate_ticket": mock_handler}):
        agent._run_agent_for_anomaly(c, ANOMALY)

    mock_handler.assert_called_once_with(**inp)


def test_tool_dispatch_send_alert():
    """send_alert tool → handle_send_alert called."""
    inp = {"message": "alert!", "channel": "slack"}
    c = _make_client(_tool_use("send_alert", "id2", inp), _end_turn())
    mock_handler = MagicMock(return_value="Alert sent.")

    with patch.dict(agent.TOOL_HANDLERS, {"send_alert": mock_handler}):
        agent._run_agent_for_anomaly(c, ANOMALY)

    mock_handler.assert_called_once_with(**inp)


def test_tool_dispatch_get_history():
    """get_ticket_history → handler called, result appended to actions log."""
    inp = {"customer_id": "C123"}
    c = _make_client(_tool_use("get_ticket_history", "id3", inp), _end_turn())
    mock_handler = MagicMock(return_value="Customer has 3 prior tickets.")

    with patch.dict(agent.TOOL_HANDLERS, {"get_ticket_history": mock_handler}):
        agent._run_agent_for_anomaly(c, ANOMALY)

    mock_handler.assert_called_once_with(**inp)
    entries = [json.loads(l) for l in open(agent.ACTIONS_LOG)]
    assert any(e["tool"] == "get_ticket_history" for e in entries)


def test_tool_dispatch_update_status():
    """update_ticket_status → handler called."""
    inp = {"ticket_id": "T001", "new_status": "closed", "reason": "resolved"}
    c = _make_client(_tool_use("update_ticket_status", "id4", inp), _end_turn())
    mock_handler = MagicMock(return_value="Ticket T001 status updated to closed.")

    with patch.dict(agent.TOOL_HANDLERS, {"update_ticket_status": mock_handler}):
        agent._run_agent_for_anomaly(c, ANOMALY)

    mock_handler.assert_called_once_with(**inp)


def test_unknown_tool_returns_error():
    """Hallucinated tool name → error string returned and logged."""
    inp = {}
    c = _make_client(_tool_use("hallucinated_tool", "id5", inp), _end_turn())

    agent._run_agent_for_anomaly(c, ANOMALY)

    entries = [json.loads(l) for l in open(agent.ACTIONS_LOG)]
    assert any("not found" in e["result"] for e in entries)


def test_loop_exits_on_end_turn():
    """stop_reason='end_turn' → loop exits after exactly one API call."""
    c = _make_client(_end_turn())

    agent._run_agent_for_anomaly(c, ANOMALY)

    assert c.messages.create.call_count == 1


def test_max_iterations_cap_enforced():
    """Loop stops at MAX_ITERATIONS even when model keeps returning tool_use."""
    inp = {"message": "alert!", "channel": "slack"}
    c = MagicMock(spec=anthropic.Anthropic)
    c.messages.create.return_value = _tool_use("send_alert", "idX", inp)

    agent._run_agent_for_anomaly(c, ANOMALY)

    assert c.messages.create.call_count == agent.MAX_ITERATIONS


def test_high_severity_escalation_queued():
    """High severity escalate_ticket → written to pending_approval.jsonl, NOT actions.jsonl."""
    inp = {"ticket_id": "T001", "reason": "critical", "severity": "high"}
    c = _make_client(_tool_use("escalate_ticket", "id8", inp), _end_turn())

    agent._run_agent_for_anomaly(c, {**ANOMALY, "severity": "high"})

    assert os.path.exists(agent.PENDING_APPROVAL_LOG)
    pending = [json.loads(l) for l in open(agent.PENDING_APPROVAL_LOG)]
    assert len(pending) == 1
    assert pending[0]["ticket_id"] == "T001"
    assert pending[0]["tool"] == "escalate_ticket"

    if os.path.exists(agent.ACTIONS_LOG):
        actions = [json.loads(l) for l in open(agent.ACTIONS_LOG)]
        assert not any(a["tool"] == "escalate_ticket" for a in actions)


def test_normal_severity_escalation_executed():
    """Medium severity escalate_ticket → written to actions.jsonl, NOT pending."""
    inp = {"ticket_id": "T002", "reason": "minor", "severity": "medium"}
    c = _make_client(_tool_use("escalate_ticket", "id9", inp), _end_turn())

    agent._run_agent_for_anomaly(c, {**ANOMALY, "severity": "medium"})

    actions = [json.loads(l) for l in open(agent.ACTIONS_LOG)]
    assert any(a["tool"] == "escalate_ticket" for a in actions)
    assert not os.path.exists(agent.PENDING_APPROVAL_LOG)


def test_actions_jsonl_written():
    """Actions log file exists and is non-empty after a tool call."""
    inp = {"message": "test alert", "channel": "email"}
    c = _make_client(_tool_use("send_alert", "id10", inp), _end_turn())

    agent._run_agent_for_anomaly(c, ANOMALY)

    assert os.path.exists(agent.ACTIONS_LOG)
    assert _count_nonempty_lines(agent.ACTIONS_LOG) > 0


def test_reasoning_jsonl_written():
    """Reasoning log file exists and is non-empty after any run."""
    c = _make_client(_end_turn())

    agent._run_agent_for_anomaly(c, ANOMALY)

    assert os.path.exists(agent.REASONING_LOG)
    assert _count_nonempty_lines(agent.REASONING_LOG) > 0


def test_idempotency_clears_logs():
    """Running run_agent twice rewrites log files rather than appending."""
    inp = {"message": "alert", "channel": "slack"}
    anomaly_df = pd.DataFrame([ANOMALY])

    def fresh_client():
        c = MagicMock(spec=anthropic.Anthropic)
        c.messages.create.side_effect = [
            _tool_use("send_alert", "idY", inp),
            _end_turn(),
        ]
        return c

    with patch("pandas.read_parquet", return_value=anomaly_df):
        with patch("anthropic.Anthropic", side_effect=[fresh_client(), fresh_client()]):
            agent.run_agent(backend="python")
            lines_first = _count_nonempty_lines(agent.ACTIONS_LOG)

            agent.run_agent(backend="python")
            lines_second = _count_nonempty_lines(agent.ACTIONS_LOG)

    assert lines_first > 0
    assert lines_first == lines_second


def test_agent_summary_written():
    """agent_summary.json exists with all required keys after run_agent completes."""
    anomaly_df = pd.DataFrame([ANOMALY])
    c = MagicMock(spec=anthropic.Anthropic)
    c.messages.create.return_value = _end_turn()

    with patch("pandas.read_parquet", return_value=anomaly_df):
        with patch("anthropic.Anthropic", return_value=c):
            agent.run_agent(backend="python")

    assert os.path.exists(agent.AGENT_SUMMARY)
    with open(agent.AGENT_SUMMARY) as f:
        summary = json.load(f)

    required_keys = {
        "generated_at", "anomalies_processed", "total_tool_calls",
        "actions_taken", "pending_approval", "tool_distribution",
        "avg_tool_calls_per_anomaly", "context_truncations",
    }
    assert required_keys.issubset(summary.keys())
    assert set(summary["tool_distribution"].keys()) == {
        "escalate_ticket", "send_alert", "create_task", "auto_respond",
        "get_ticket_history", "update_ticket_status",
    }


def test_context_truncation_triggered(monkeypatch, caplog):
    """Simulated long context triggers truncation log and increments counter."""
    monkeypatch.setattr(agent, "_estimate_tokens", lambda msgs: 50_000)
    c = _make_client(_end_turn())

    with caplog.at_level(logging.INFO, logger="src.agent"):
        tool_calls, truncations = agent._run_agent_for_anomaly(c, ANOMALY)

    assert any("Context truncated" in r.message for r in caplog.records)
    assert truncations >= 1
