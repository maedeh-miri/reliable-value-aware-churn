"""Tests for the paired Complains ablation experiment."""

import unittest

import pandas as pd

from experiments.complains_ablation_cv import (
    build_reduced_pipeline,
    evaluate_ablation,
)


class TestComplainsAblation(unittest.TestCase):
    """Verify reduced preprocessing and paired CV results."""

    def setUp(self):
        rows = list(range(100))

        self.X = pd.DataFrame({
            "Call  Failure": [i % 21 for i in rows],
            "Subscription  Length": [3 + i % 30 for i in rows],
            "Charge  Amount": [i % 11 for i in rows],
            "Seconds of Use": [100 + 20 * i for i in rows],
            "Frequency of use": [i % 50 for i in rows],
            "Frequency of SMS": [i % 40 for i in rows],
            "Distinct Called Numbers": [i % 25 for i in rows],
            "Complains": [i % 2 for i in rows],
            "Age Group": [1 + i % 5 for i in rows],
            "Tariff Plan": [1 + i % 2 for i in rows],
        })

        self.y = pd.Series([0, 1] * 50)

        reduced_features = [
            column
            for column in self.X.columns
            if column != "Complains"
        ]

        self.groups = self.X.groupby(
            reduced_features,
            dropna=False,
        ).ngroup()

    def test_reduced_pipeline_excludes_complains(self):
        model = build_reduced_pipeline()

        preprocessor = model.named_steps["preprocessing"]

        used_columns = []

        for _, _, columns in preprocessor.transformers:
            used_columns.extend(columns)

        self.assertNotIn("Complains", used_columns)

    def test_ablation_returns_five_folds(self):
        results = evaluate_ablation(
            self.X,
            self.y,
            self.groups,
        )

        self.assertEqual(len(results), 5)
        self.assertEqual(results["fold"].tolist(), [0, 1, 2, 3, 4])

    def test_deltas_match_model_differences(self):
        results = evaluate_ablation(
            self.X,
            self.y,
            self.groups,
        )

        metric_names = [
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
        ]

        for metric_name in metric_names:
            full = results[f"full_{metric_name}"]
            reduced = results[f"without_{metric_name}"]
            delta = results[f"delta_{metric_name}"]

            expected_delta = reduced - full

            for actual, expected in zip(delta, expected_delta):
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
