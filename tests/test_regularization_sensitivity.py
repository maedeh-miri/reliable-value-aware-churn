"""Tests for Logistic Regression regularization sensitivity."""

import math
import unittest

import pandas as pd

from experiments.regularization_sensitivity import (
    C_VALUES,
    evaluate_regularization,
)


class TestRegularizationSensitivity(unittest.TestCase):
    """Verify the predefined grid and evaluation outputs."""

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

    def test_predefined_c_values(self):
        self.assertEqual(
            C_VALUES,
            [0.1, 1.0, 10.0],
        )

    def test_expected_number_of_results(self):
        results = evaluate_regularization(
            self.X,
            self.y,
            self.groups,
        )

        self.assertEqual(len(results), 15)

        for c_value in C_VALUES:
            subset = results[results["C"] == c_value]

            self.assertEqual(len(subset), 5)
            self.assertEqual(
                subset["fold"].tolist(),
                [0, 1, 2, 3, 4],
            )

    def test_metrics_and_iterations_are_valid(self):
        results = evaluate_regularization(
            self.X,
            self.y,
            self.groups,
        )

        metric_columns = [
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
        ]

        for column in metric_columns:
            for value in results[column]:
                self.assertTrue(math.isfinite(value))

        for column in [
            "roc_auc",
            "average_precision",
            "brier",
        ]:
            self.assertTrue(
                results[column].between(0, 1).all()
            )

        self.assertTrue((results["log_loss"] >= 0).all())

        self.assertTrue(
            results["iterations"].between(1, 1000).all()
        )


if __name__ == "__main__":
    unittest.main()
