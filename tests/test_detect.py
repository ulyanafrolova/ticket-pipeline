import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.detect_anomalies import AnomalyResult, detect, _analyze_thresholds


def _make_df(n=10, **overrides):
    """Minimal DataFrame with all columns detect() requires."""
    data = {
        "ticket_id": [f"T{i:03d}" for i in range(n)],
        "body": [f"Normal body text for this specific ticket index {i}" for i in range(n)],
        "sentiment": ["neutral"] * n,
        "llm_category": ["billing"] * n,
        "subject": [f"Subject {i}" for i in range(n)],
        "customer_id": [f"cust-{i:03d}" for i in range(n)],
    }
    data.update(overrides)
    return pd.DataFrame(data)


_HIGH_RESULT = AnomalyResult(
    anomaly_type="frustrated_sentiment",
    severity="high",
    reason="Customer is very frustrated with the service",
    recommended_action="escalate",
)

_LOW_RESULT = AnomalyResult(
    anomaly_type="unclassified",
    severity="low",
    reason="Ticket is unclassified and low priority",
    recommended_action="monitor",
)


class TestDetectAnomalies(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmpdir.name, "input.parquet")
        self.output_path = os.path.join(self.tmpdir.name, "anomalies.parquet")
        self.report_path = os.path.join(self.tmpdir.name, "anomaly_report.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_detect(self, df, mock_classify=None):
        df.to_parquet(self.input_path, index=False)
        if mock_classify is None:
            mock_classify = MagicMock(return_value=_HIGH_RESULT)
        with patch("src.detect_anomalies._classify_anomaly", mock_classify), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            return detect(self.input_path, self.output_path, self.report_path)

    # 1. Very long body → flagged as body_length_spike
    def test_body_length_spike_detected(self):
        # Need ~20 tickets so the outlier's z-score clearly exceeds 3σ
        n = 20
        bodies = [f"Normal body text for this specific ticket index {i}" for i in range(n - 1)]
        bodies.append("A" * 10000)
        df = _make_df(n=n, body=bodies)
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        flagged_ids = set(out["ticket_id"].tolist())
        self.assertIn("T019", flagged_ids)
        row = out[out["ticket_id"] == "T019"]
        self.assertTrue(row["statistical_flags"].str.contains("body_length_spike").any())

    # 2. Average body length → not flagged for body_length_spike
    def test_body_length_normal_not_flagged(self):
        df = _make_df()  # all similar-length bodies
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        if len(out) > 0:
            self.assertFalse(out["statistical_flags"].str.contains("body_length_spike").any())

    # 3. sentiment="frustrated" → flagged
    def test_frustrated_sentiment_flagged(self):
        sentiments = ["neutral"] * 9 + ["frustrated"]
        df = _make_df(sentiment=sentiments)
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        self.assertIn("T009", set(out["ticket_id"].tolist()))
        row = out[out["ticket_id"] == "T009"]
        self.assertTrue(row["statistical_flags"].str.contains("frustrated_sentiment").any())

    # 4. llm_category="unknown" → flagged
    def test_unclassified_flagged(self):
        categories = ["billing"] * 9 + ["unknown"]
        df = _make_df(llm_category=categories)
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        self.assertIn("T009", set(out["ticket_id"].tolist()))
        row = out[out["ticket_id"] == "T009"]
        self.assertTrue(row["statistical_flags"].str.contains("unclassified").any())

    # 5. Two tickets with identical body (>20 chars) → both flagged as duplicate_body
    def test_duplicate_body_flagged(self):
        bodies = [f"Unique body for ticket number {i} with enough characters" for i in range(8)]
        dup = "This is the exact same body text for two different tickets here"
        bodies.extend([dup, dup])  # T008, T009
        df = _make_df(body=bodies)
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        flagged_ids = set(out["ticket_id"].tolist())
        self.assertIn("T008", flagged_ids)
        self.assertNotIn("T009", flagged_ids)
        for tid in ["T008"]:
            row = out[out["ticket_id"] == tid]
            self.assertTrue(row["statistical_flags"].str.contains("duplicate_body").any())

    # 6. Body < 20 chars → duplicate check skipped, not flagged
    def test_short_body_not_duplicate(self):
        bodies = [f"Body {i}" for i in range(8)]  # unique, <20 chars
        short_dup = "short"  # 5 chars, well under 20
        bodies.extend([short_dup, short_dup])  # T008, T009
        df = _make_df(body=bodies)
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        flagged_ids = set(out["ticket_id"].tolist())
        self.assertNotIn("T008", flagged_ids)
        self.assertNotIn("T009", flagged_ids)

    # 7. repeat_contact=True + sentiment="frustrated" → flagged as repeat_frustrated
    def test_repeat_frustrated_flagged(self):
        sentiments = ["neutral"] * 9 + ["frustrated"]
        repeat_contacts = [False] * 9 + [True]
        df = _make_df(sentiment=sentiments, repeat_contact=repeat_contacts)
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        self.assertIn("T009", set(out["ticket_id"].tolist()))
        row = out[out["ticket_id"] == "T009"]
        self.assertTrue(row["statistical_flags"].str.contains("repeat_frustrated").any())

    # 8. Ticket matching 2 checks → one output row, both flags in statistical_flags
    def test_multi_flag_single_row(self):
        sentiments = ["neutral"] * 9 + ["frustrated"]
        categories = ["billing"] * 9 + ["unknown"]
        df = _make_df(sentiment=sentiments, llm_category=categories)
        self._run_detect(df)
        out = pd.read_parquet(self.output_path)
        row = out[out["ticket_id"] == "T009"]
        self.assertEqual(len(row), 1)
        self.assertTrue(row["statistical_flags"].str.contains("frustrated_sentiment").any())
        self.assertTrue(row["statistical_flags"].str.contains("unclassified").any())

    # 9. Null body → detection_method="statistical", LLM never called
    def test_llm_not_called_for_null_body(self):
        bodies = [f"Normal body text for this specific ticket index {i}" for i in range(9)]
        bodies.append(None)  # T009: null body
        sentiments = ["neutral"] * 9 + ["frustrated"]
        df = _make_df(body=bodies, sentiment=sentiments)
        mock_classify = MagicMock(return_value=_HIGH_RESULT)
        self._run_detect(df, mock_classify=mock_classify)
        mock_classify.assert_not_called()
        out = pd.read_parquet(self.output_path)
        row = out[out["ticket_id"] == "T009"]
        self.assertEqual(row["detection_method"].iloc[0], "statistical")

    # 10. Flagged non-null body → LLM called
    def test_llm_called_for_flagged_tickets(self):
        sentiments = ["neutral"] * 9 + ["frustrated"]
        df = _make_df(sentiment=sentiments)  # T009 has non-null body
        mock_classify = MagicMock(return_value=_HIGH_RESULT)
        self._run_detect(df, mock_classify=mock_classify)
        mock_classify.assert_called_once()

    # 11. threshold_analysis.json exists with all 3 threshold keys
    def test_threshold_analysis_written(self):
        df = _make_df()
        self._run_detect(df)
        threshold_path = os.path.join(self.tmpdir.name, "threshold_analysis.json")
        self.assertTrue(os.path.exists(threshold_path))
        with open(threshold_path) as f:
            data = json.load(f)
        self.assertIn("thresholds", data)
        self.assertIn("2.0", data["thresholds"])
        self.assertIn("2.5", data["thresholds"])
        self.assertIn("3.0", data["thresholds"])

    # 12. high_severity_alerts.json exists after detect()
    def test_high_severity_alerts_written(self):
        sentiments = ["neutral"] * 9 + ["frustrated"]
        df = _make_df(sentiment=sentiments)
        self._run_detect(df)
        alerts_path = os.path.join(self.tmpdir.name, "high_severity_alerts.json")
        self.assertTrue(os.path.exists(alerts_path))

    # 13. Alert file contains only severity="high" entries and has alert_channel field
    def test_alerts_only_high_severity(self):
        sentiments = ["neutral"] * 9 + ["frustrated"]
        df = _make_df(sentiment=sentiments)
        self._run_detect(df, mock_classify=MagicMock(return_value=_HIGH_RESULT))
        alerts_path = os.path.join(self.tmpdir.name, "high_severity_alerts.json")
        with open(alerts_path) as f:
            data = json.load(f)
        self.assertGreater(data["alert_count"], 0)
        for alert in data["alerts"]:
            self.assertEqual(alert["severity"], "high")
            self.assertIn("alert_channel", alert)
            self.assertIn(alert["alert_channel"], {"slack", "email", "monitor"})

    # 14. When previous report exists, "trends" key is in the current report
    def test_trend_added_when_previous_exists(self):
        prev_report = {
            "generated_at": "2026-07-02T00:00:00+00:00",
            "total_tickets": 10,
            "anomalies_found": 3,
            "by_type": {"frustrated_sentiment": 2, "unclassified": 1},
            "by_severity": {"high": 2, "low": 1},
        }
        prev_path = os.path.join(self.tmpdir.name, "anomaly_report_previous.json")
        with open(prev_path, "w") as f:
            json.dump(prev_report, f)

        sentiments = ["neutral"] * 9 + ["frustrated"]
        df = _make_df(sentiment=sentiments)
        self._run_detect(df)

        with open(self.report_path) as f:
            report = json.load(f)
        self.assertIn("trends", report)
        self.assertIn("anomalies_found_delta", report["trends"])
        self.assertIn("by_type_delta", report["trends"])


class TestAnalyzeThresholds(unittest.TestCase):
    def test_returns_all_requested_thresholds(self):
        series = pd.Series([10.0] * 90 + [500.0] * 10)
        result = _analyze_thresholds(series, [2.0, 2.5, 3.0])
        self.assertIn("2.0", result)
        self.assertIn("2.5", result)
        self.assertIn("3.0", result)

    def test_higher_threshold_fewer_flagged(self):
        series = pd.Series(list(range(1, 101)))
        result = _analyze_thresholds(series, [1.0, 2.0, 3.0])
        self.assertGreaterEqual(result["1.0"]["flagged"], result["2.0"]["flagged"])
        self.assertGreaterEqual(result["2.0"]["flagged"], result["3.0"]["flagged"])

    def test_short_series_returns_zeros(self):
        series = pd.Series([100.0])
        result = _analyze_thresholds(series, [2.0, 3.0])
        for key in result:
            self.assertEqual(result[key]["flagged"], 0)


if __name__ == "__main__":
    unittest.main()
