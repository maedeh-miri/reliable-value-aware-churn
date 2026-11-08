"""Tests for Charge Amount encoding sensitivity."""

import unittest

import pandas as pd

from experiments.charge_encoding_sensitivity import (
    build_onehot_charge_pipeline,
    evaluate_charge_encoding,
)


class TestChargeEncoding(unittest.TestCase):
    """Verify encoding configuration and paired CV outputs."""

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

    def test_charge_is_categorical(self):
        model = build_onehot_charge_pipeline()

        preprocessor = model.named_steps["preprocessing"]

        transformer_columns = {
            name: columns
            for name, _, columns in preprocessor.transformers
        }

        self.assertIn(
            "Charge  Amount",
            transformer_columns["categorical"],
        )

        self.assertNotIn(
            "Charge  Amount",
            transformer_columns["numeric"],
        )

    def test_unseen_charge_is_handled(self):
        model = build_onehot_charge_pipeline()

        X_fit = self.X.iloc[:30]
        y_fit = self.y.iloc[:30]

        model.fit(X_fit, y_fit)

        X_validation = self.X.iloc[[30]].copy()

        X_validation.loc[30, "Charge  Amount"] = 999

        probabilities = model.predict_proba(X_validation)

        self.assertEqual(probabilities.shape, (1, 2))
        self.assertAlmostEqual(probabilities[0].sum(), 1.0)

    def test_paired_deltas_are_correct(self):
        results = evaluate_charge_encoding(
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
            numeric = results[f"numeric_{metric_name}"]
            onehot = results[f"onehot_{metric_name}"]
            delta = results[f"delta_{metric_name}"]

            expected_delta = onehot - numeric

            for actual, expected in zip(delta, expected_delta):
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
