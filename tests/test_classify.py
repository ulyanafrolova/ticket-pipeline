import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import anthropic
from src.classify import (
    ClassificationResult,
    _classify_ticket,
    classify,
)

HIGH_CONF = ClassificationResult(category="billing", priority="low", confidence=0.9)
LOW_CONF = ClassificationResult(category="general", priority="medium", confidence=0.5)
HIGH_CONF_SONNET = ClassificationResult(category="billing", priority="low", confidence=0.92)
GOOD_USAGE = {"input_tokens": 100, "output_tokens": 20}
SONNET_USAGE = {"input_tokens": 200, "output_tokens": 30}


def _make_df(n=2, subject="Test subject", body="Test body"):
    return pd.DataFrame({"subject": [subject] * n, "body": [body] * n})


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmpdir.name, "input.parquet")
        self.output_path = os.path.join(self.tmpdir.name, "output.parquet")
        _make_df(2).to_parquet(self.input_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_classify(self, mock_haiku=None, mock_sonnet=None, **kwargs):
        if mock_haiku is None:
            mock_haiku = MagicMock(return_value=(HIGH_CONF, GOOD_USAGE))
        if mock_sonnet is None:
            mock_sonnet = MagicMock(return_value=(None, None))
        with patch("src.classify._classify_ticket", mock_haiku), \
             patch("src.classify._classify_ticket_sonnet", mock_sonnet), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            return classify(self.input_path, self.output_path, **kwargs)

    # --- 1. Return dict keys ---
    def test_classify_returns_correct_keys(self):
        result = self._run_classify()
        self.assertEqual(
            set(result.keys()),
            {"total", "classified", "failed", "needs_review", "escalated_to_sonnet", "cost_estimate_usd"},
        )

    # --- 2. Haiku called once per ticket ---
    def test_haiku_called_for_each_ticket(self):
        mock_haiku = MagicMock(return_value=(HIGH_CONF, GOOD_USAGE))
        self._run_classify(mock_haiku=mock_haiku, sample=2)
        self.assertEqual(mock_haiku.call_count, 2)

    # --- 3. JSONDecodeError → "unknown" label, no crash ---
    def test_fallback_on_json_error(self):
        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.content = [MagicMock(type="text", text="not valid json {{{")]
        bad_resp.usage.input_tokens = 50
        bad_resp.usage.output_tokens = 5
        mock_client.messages.create.return_value = bad_resp

        with patch("src.classify.anthropic.Anthropic", return_value=mock_client), \
             patch("src.classify._classify_ticket_sonnet", return_value=(None, None)), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            classify(self.input_path, self.output_path)

        out_df = pd.read_parquet(self.output_path)
        self.assertTrue((out_df["llm_category"] == "unknown").all())

    # --- 4. RateLimitError → retries after sleep ---
    def test_retry_on_rate_limit(self):
        mock_client = MagicMock()
        good_resp = MagicMock()
        good_resp.content = [MagicMock(type="text", text='{"category": "billing", "priority": "low", "confidence": 0.9}')]
        good_resp.usage.input_tokens = 100
        good_resp.usage.output_tokens = 20

        import httpx
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(429, request=req)
        rate_err = anthropic.RateLimitError(message="Rate limited", response=resp, body={})
        mock_client.messages.create.side_effect = [rate_err, good_resp]

        with patch("src.classify.time.sleep"):
            result, usage = _classify_ticket(mock_client, "Subject", "Body")

        self.assertIsNotNone(result)
        self.assertEqual(mock_client.messages.create.call_count, 2)

    # --- 5. AuthenticationError → propagates immediately ---
    def test_auth_error_raises(self):
        mock_client = MagicMock()

        import httpx
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(401, request=req)
        auth_err = anthropic.AuthenticationError(message="Invalid API key", response=resp, body={})
        mock_client.messages.create.side_effect = auth_err

        with self.assertRaises(anthropic.AuthenticationError):
            _classify_ticket(mock_client, "Subject", "Body")

    # --- 6. Both null → no API call, "unknown" result ---
    def test_null_subject_body_skipped(self):
        mock_client = MagicMock()
        result, usage = _classify_ticket(mock_client, None, None)
        self.assertIsNone(result)
        self.assertIsNone(usage)
        mock_client.messages.create.assert_not_called()

    # --- 7. confidence < 0.70 → needs_review=True ---
    def test_needs_review_low_confidence(self):
        mock_haiku = MagicMock(return_value=(LOW_CONF, GOOD_USAGE))
        mock_sonnet = MagicMock(return_value=(None, None))
        self._run_classify(mock_haiku=mock_haiku, mock_sonnet=mock_sonnet, sample=2)
        out_df = pd.read_parquet(self.output_path)
        self.assertTrue(out_df["needs_review"].all())

    # --- 8. confidence >= 0.70 → needs_review=False ---
    def test_no_review_high_confidence(self):
        mock_haiku = MagicMock(return_value=(HIGH_CONF, GOOD_USAGE))
        self._run_classify(mock_haiku=mock_haiku, sample=2)
        out_df = pd.read_parquet(self.output_path)
        self.assertFalse(out_df["needs_review"].any())

    # --- 9. needs_review=True ticket → Sonnet called ---
    def test_sonnet_called_for_low_confidence(self):
        mock_haiku = MagicMock(return_value=(LOW_CONF, GOOD_USAGE))
        mock_sonnet = MagicMock(return_value=(HIGH_CONF_SONNET, SONNET_USAGE))
        self._run_classify(mock_haiku=mock_haiku, mock_sonnet=mock_sonnet, sample=2)
        self.assertGreater(mock_sonnet.call_count, 0)

    # --- 10. High confidence → Sonnet NOT called ---
    def test_sonnet_not_called_for_high_confidence(self):
        mock_haiku = MagicMock(return_value=(HIGH_CONF, GOOD_USAGE))
        mock_sonnet = MagicMock(return_value=(None, None))
        self._run_classify(mock_haiku=mock_haiku, mock_sonnet=mock_sonnet, sample=2)
        mock_sonnet.assert_not_called()

    # --- 11. Output has escalated_to_sonnet bool column ---
    def test_escalated_column_present(self):
        mock_haiku = MagicMock(return_value=(LOW_CONF, GOOD_USAGE))
        mock_sonnet = MagicMock(return_value=(HIGH_CONF_SONNET, SONNET_USAGE))
        self._run_classify(mock_haiku=mock_haiku, mock_sonnet=mock_sonnet, sample=2)
        out_df = pd.read_parquet(self.output_path)
        self.assertIn("escalated_to_sonnet", out_df.columns)
        self.assertTrue(out_df["escalated_to_sonnet"].any())
        self.assertTrue(pd.api.types.is_bool_dtype(out_df["escalated_to_sonnet"]))

    # --- 12. max_cost_usd=0.0 → 0 tickets processed ---
    def test_cost_cap_stops_early(self):
        mock_haiku = MagicMock(return_value=(HIGH_CONF, GOOD_USAGE))
        self._run_classify(mock_haiku=mock_haiku, sample=5, max_cost_usd=0.0)
        mock_haiku.assert_not_called()

    # --- 13. No null values in any of the 5 new columns ---
    def test_output_no_nulls(self):
        mock_haiku = MagicMock(return_value=(HIGH_CONF, GOOD_USAGE))
        self._run_classify(mock_haiku=mock_haiku, sample=2)
        out_df = pd.read_parquet(self.output_path)
        for col in ["llm_category", "llm_priority", "llm_confidence", "needs_review", "escalated_to_sonnet"]:
            self.assertFalse(out_df[col].isnull().any(), f"Column '{col}' has null values")

    # --- 14. classification_report.json exists with all required keys ---
    def test_classification_report_written(self):
        mock_haiku = MagicMock(return_value=(HIGH_CONF, GOOD_USAGE))
        self._run_classify(mock_haiku=mock_haiku, sample=2)
        report_path = os.path.join(self.tmpdir.name, "classification_report.json")
        self.assertTrue(os.path.exists(report_path), "classification_report.json not found")
        with open(report_path) as f:
            report = json.load(f)
        required_keys = {
            "generated_at", "total_tickets", "classified", "failed",
            "needs_review", "escalated_to_sonnet", "haiku_cost_usd",
            "sonnet_cost_usd", "total_cost_usd", "category_distribution",
            "priority_distribution", "confidence_buckets",
        }
        self.assertEqual(set(report.keys()), required_keys)


if __name__ == "__main__":
    unittest.main()
