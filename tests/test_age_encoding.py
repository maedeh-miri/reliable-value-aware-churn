"""Tests for Age Group encoding sensitivity."""

import unittest

import pandas as pd

from experiments.age_encoding_sensitivity import (
    build_ordinal_age_pipeline,
    evaluate_age_encoding,
)


class TestAgeEncoding(unittest.TestCase):
    """Verify ordinal encoding and paired CV results."""

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

        self.groups = self.X.groupby(
            list(self.X.columns),
            dropna=False,
        ).ngroup()

    def test_age_group_is_numeric(self):
        model = build_ordinal_age_pipeline()

        preprocessor = model.named_steps["preprocessing"]

        transformer_columns = {
            name: columns
            for name, _, columns in preprocessor.transformers
        }

        self.assertIn(
            "Age Group",
            transformer_columns["numeric"],
        )

        self.assertNotIn(
            "Age Group",
            transformer_columns["categorical"],
        )

    def test_age_scaler_uses_fit_data(self):
        X_fit = self.X.iloc[:30]
        y_fit = self.y.iloc[:30]

        model = build_ordinal_age_pipeline()
        model.fit(X_fit, y_fit)

        scaler = model.named_steps[
            "preprocessing"
        ].named_transformers_["numeric"]

        expected_mean = X_fit["Age Group"].mean()

        actual_mean = scaler.mean_[-1]

        self.assertAlmostEqual(
            actual_mean,
            expected_mean,
        )

    def test_paired_deltas_are_correct(self):
        results = evaluate_age_encoding(
            self.X,
            self.y,
            self.groups,
        )

        self.assertEqual(len(results), 5)

        metric_names = [
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
        ]

        for metric_name in metric_names:
            onehot = results[f"onehot_{metric_name}"]
            ordinal = results[f"ordinal_{metric_name}"]
            delta = results[f"delta_{metric_name}"]

            expected_delta = ordinal - onehot

            for actual, expected in zip(delta, expected_delta):
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
