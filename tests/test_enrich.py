import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.enrich import (
    EntitySet,
    EnrichmentResult,
    _flag_repeat_contacts,
    enrich,
)

GOOD_RESULT = EnrichmentResult(
    sentiment="neutral",
    sentiment_confidence=0.9,
    entities=EntitySet(),
    summary="Customer reported a minor issue with the product.",
    urgency_score=0.2,
    urgency_reason="Informational query with no urgency.",
)
GOOD_USAGE = {"input_tokens": 100, "output_tokens": 50}


def _make_df(n=3, subjects=None, bodies=None, customer_ids=None):
    if subjects is None:
        subjects = [f"Subject {i}" for i in range(n)]
    if bodies is None:
        bodies = [f"Body {i}" for i in range(n)]
    if customer_ids is None:
        customer_ids = [f"cust-{i:03d}" for i in range(n)]
    return pd.DataFrame({
        "subject": subjects,
        "body": bodies,
        "customer_id": customer_ids,
    })


def _make_enriched_df(n=3, failed=None, customer_ids=None):
    """Create a pre-enriched dataframe with all output columns."""
    if failed is None:
        failed = [False] * n
    if customer_ids is None:
        customer_ids = [f"cust-{i:03d}" for i in range(n)]

    empty_ej = json.dumps({"product_names": [], "error_codes": [], "account_ids": []})

    return pd.DataFrame({
        "subject": [f"Subject {i}" for i in range(n)],
        "body": [f"Body {i}" for i in range(n)],
        "customer_id": customer_ids,
        "sentiment": [None if f else "neutral" for f in failed],
        "sentiment_confidence": [None if f else 0.9 for f in failed],
        "entities_json": [empty_ej] * n,
        "summary": [None if f else "Customer had an issue." for f in failed],
        "urgency_score": [0.5 if f else 0.2 for f in failed],
        "urgency_reason": [
            "Classification failed; manual review required" if f else "Minor issue."
            for f in failed
        ],
        "enrichment_failed": [bool(f) for f in failed],
        "repeat_contact": [False] * n,
    })


class TestEnrich(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmpdir.name, "input.parquet")
        self.output_path = os.path.join(self.tmpdir.name, "output.parquet")
        _make_df(3).to_parquet(self.input_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_enrich(self, mock_enrich_fn=None, **kwargs):
        if mock_enrich_fn is None:
            mock_enrich_fn = MagicMock(return_value=(GOOD_RESULT, GOOD_USAGE))
        with patch("src.enrich._enrich_ticket", mock_enrich_fn), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            return enrich(self.input_path, self.output_path, **kwargs)

    # --- 1. Return dict has all 7 required keys ---
    def test_enrich_returns_correct_keys(self):
        result = self._run_enrich()
        self.assertEqual(
            set(result.keys()),
            {"total", "enriched", "failed", "tokens_used", "cost_usd",
             "cache_hit_rate", "repeat_contacts"},
        )

    # --- 2. Output has sentiment column ---
    def test_sentiment_column_present(self):
        self._run_enrich()
        out_df = pd.read_parquet(self.output_path)
        self.assertIn("sentiment", out_df.columns)

    # --- 3. Output has urgency_score float column ---
    def test_urgency_score_column_present(self):
        self._run_enrich()
        out_df = pd.read_parquet(self.output_path)
        self.assertIn("urgency_score", out_df.columns)
        self.assertTrue(pd.api.types.is_float_dtype(out_df["urgency_score"]))
        self.assertTrue(out_df["urgency_score"].notna().all())

    # --- 4. Output has urgency_reason string column ---
    def test_urgency_reason_column_present(self):
        self._run_enrich()
        out_df = pd.read_parquet(self.output_path)
        self.assertIn("urgency_reason", out_df.columns)
        self.assertTrue(out_df["urgency_reason"].notna().all())
        self.assertIsInstance(out_df["urgency_reason"].iloc[0], str)

    # --- 5. Exception → enrichment_failed=True, no crash ---
    def test_enrichment_failed_on_api_error(self):
        mock_enrich = MagicMock(side_effect=RuntimeError("Unexpected API error"))
        result = self._run_enrich(mock_enrich_fn=mock_enrich)
        self.assertIsNotNone(result)
        out_df = pd.read_parquet(self.output_path)
        self.assertTrue(out_df["enrichment_failed"].all())

    # --- 6. Both null → skipped, enrichment_failed=True ---
    def test_null_subject_body_skipped(self):
        df = pd.DataFrame({
            "subject": [None],
            "body": [None],
            "customer_id": ["cust-001"],
        })
        df.to_parquet(self.input_path)

        mock_client = MagicMock()
        with patch("src.enrich.anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            enrich(self.input_path, self.output_path)

        mock_client.messages.create.assert_not_called()
        out_df = pd.read_parquet(self.output_path)
        self.assertTrue(out_df["enrichment_failed"].all())

    # --- 7. Failed row → entities_json is valid JSON string ---
    def test_entities_json_valid_for_failed(self):
        mock_enrich = MagicMock(return_value=(None, None))
        self._run_enrich(mock_enrich_fn=mock_enrich)
        out_df = pd.read_parquet(self.output_path)
        for ej in out_df["entities_json"]:
            parsed = json.loads(ej)
            self.assertIsInstance(parsed, dict)
            self.assertIn("product_names", parsed)
            self.assertIn("error_codes", parsed)
            self.assertIn("account_ids", parsed)

    # --- 8. Customer with 3 tickets → repeat_contact=True ---
    def test_repeat_contact_flag_3_tickets(self):
        df = _make_df(5, customer_ids=["cust-A", "cust-A", "cust-A", "cust-B", "cust-C"])
        df.to_parquet(self.input_path)

        mock_enrich = MagicMock(return_value=(GOOD_RESULT, GOOD_USAGE))
        with patch("src.enrich._enrich_ticket", mock_enrich), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            enrich(self.input_path, self.output_path)

        out_df = pd.read_parquet(self.output_path)
        cust_a_rows = out_df[out_df["customer_id"] == "cust-A"]
        self.assertTrue(cust_a_rows["repeat_contact"].all())
        cust_b_rows = out_df[out_df["customer_id"] == "cust-B"]
        self.assertFalse(cust_b_rows["repeat_contact"].any())

    # --- 9. Customer with 2 tickets → repeat_contact=False ---
    def test_no_repeat_contact_2_tickets(self):
        df = _make_df(2, customer_ids=["cust-A", "cust-A"])
        df.to_parquet(self.input_path)

        mock_enrich = MagicMock(return_value=(GOOD_RESULT, GOOD_USAGE))
        with patch("src.enrich._enrich_ticket", mock_enrich), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            enrich(self.input_path, self.output_path)

        out_df = pd.read_parquet(self.output_path)
        self.assertFalse(out_df["repeat_contact"].any())

    # --- 10. Output rows in same order as input (not completion order) ---
    def test_concurrent_results_in_order(self):
        subjects = [f"subj-{i}" for i in range(3)]
        df = _make_df(3, subjects=subjects)
        df.to_parquet(self.input_path)

        def mock_enrich(client, subject, body):
            result = EnrichmentResult(
                sentiment="neutral",
                sentiment_confidence=0.9,
                entities=EntitySet(),
                summary=f"Summary for {subject}",
                urgency_score=0.1,
                urgency_reason="Low urgency.",
            )
            return result, {"input_tokens": 10, "output_tokens": 5}

        with patch("src.enrich._enrich_ticket", side_effect=mock_enrich), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            enrich(self.input_path, self.output_path)

        out_df = pd.read_parquet(self.output_path)
        for i, subject in enumerate(subjects):
            self.assertEqual(out_df.iloc[i]["summary"], f"Summary for {subject}")

    # --- 11. Resume mode → API not called for already-enriched rows ---
    def test_resume_skips_successful_rows(self):
        pre_existing = _make_enriched_df(3, failed=[False, False, True])
        pre_existing.to_parquet(self.output_path)

        mock_enrich = MagicMock(return_value=(GOOD_RESULT, GOOD_USAGE))
        with patch("src.enrich._enrich_ticket", mock_enrich), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            enrich(self.input_path, self.output_path, resume=True)

        self.assertEqual(mock_enrich.call_count, 1)

    # --- 12. Resume mode → API called for enrichment_failed=True rows ---
    def test_resume_reprocesses_failed_rows(self):
        pre_existing = _make_enriched_df(3, failed=[False, True, True])
        pre_existing.to_parquet(self.output_path)

        mock_enrich = MagicMock(return_value=(GOOD_RESULT, GOOD_USAGE))
        with patch("src.enrich._enrich_ticket", mock_enrich), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            enrich(self.input_path, self.output_path, resume=True)

        self.assertEqual(mock_enrich.call_count, 2)
        out_df = pd.read_parquet(self.output_path)
        self.assertFalse(out_df["enrichment_failed"].any())

    # --- 13. enrichment_report.json exists with required keys ---
    def test_enrichment_report_written(self):
        self._run_enrich()
        report_path = os.path.join(self.tmpdir.name, "enrichment_report.json")
        self.assertTrue(os.path.exists(report_path), "enrichment_report.json not found")
        with open(report_path) as f:
            report = json.load(f)
        required_keys = {
            "generated_at", "total", "enriched", "failed", "repeat_contacts",
            "tokens_used", "cost_usd", "cache_hit_rate", "sentiment_distribution",
            "urgency_distribution", "enrichment_failed_pct",
        }
        self.assertEqual(set(report.keys()), required_keys)

    # --- 14. All urgency_score values between 0.0 and 1.0 ---
    def test_urgency_score_in_range(self):
        self._run_enrich()
        out_df = pd.read_parquet(self.output_path)
        scores = out_df["urgency_score"]
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
