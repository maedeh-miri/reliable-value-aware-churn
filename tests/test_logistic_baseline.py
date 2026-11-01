"""Automated tests for the logistic churn baseline."""

import math
import unittest

import pandas as pd

from experiments.logistic_baseline_cv import (
    build_logistic_pipeline,
    evaluate_logistic_cv,
)


class TestLogisticBaseline(unittest.TestCase):
    """Test the pipeline and group-aware evaluation."""

    def setUp(self):
        row_numbers = list(range(100))

        self.X = pd.DataFrame({
            "Call  Failure": [i % 21 for i in row_numbers],
            "Subscription  Length": [3 + i % 30 for i in row_numbers],
            "Charge  Amount": [i % 11 for i in row_numbers],
            "Seconds of Use": [100 + 20 * i for i in row_numbers],
            "Frequency of use": [i % 50 for i in row_numbers],
            "Frequency of SMS": [i % 40 for i in row_numbers],
            "Distinct Called Numbers": [i % 25 for i in row_numbers],
            "Complains": [i % 2 for i in row_numbers],
            "Age Group": [1 + i % 5 for i in row_numbers],
            "Tariff Plan": [1 + i % 2 for i in row_numbers],
        })

        self.y = pd.Series(
            [0, 1] * 50,
            name="Churn",
        )

        self.groups = pd.Series(
            row_numbers,
            name="group",
        )

    def test_pipeline_has_expected_steps(self):
        model = build_logistic_pipeline()

        self.assertEqual(
            list(model.named_steps),
            ["preprocessing", "classifier"],
        )

    def test_preprocessing_uses_only_fit_data(self):
        X_fit = self.X.iloc[:20]
        y_fit = self.y.iloc[:20]

        model = build_logistic_pipeline()
        model.fit(X_fit, y_fit)

        scaler = model.named_steps[
            "preprocessing"
        ].named_transformers_["numeric"]

        mean_before = scaler.mean_[0]
        expected_mean = X_fit['Call  Failure'].mean()

        self.assertAlmostEqual(mean_before, expected_mean)

        X_validation = self.X.iloc[[20]].copy()

        X_validation.loc[20, "Call  Failure"] = 9999
        X_validation.loc[20, "Age Group"] = 99
        X_validation.loc[20, "Tariff Plan"] = 3

        probabilities = model.predict_proba(X_validation)

        mean_after = scaler.mean_[0]

        self.assertAlmostEqual(mean_before, mean_after)
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertAlmostEqual(probabilities[0].sum(), 1.0)

    def test_cv_returns_valid_metrics(self):
        results = evaluate_logistic_cv(
            self.X,
            self.y,
            self.groups,
        )

        self.assertEqual(len(results), 5)
        self.assertEqual(results["fold"].tolist(), [0, 1, 2, 3, 4])

        metric_columns = [
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
        ]

        for column in metric_columns:
            for value in results[column]:
                self.assertTrue(math.isfinite(value))

        for column in [
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
            "brier",
        ]:
            self.assertTrue(
                results[column].between(0, 1).all()
            )

        self.assertTrue((results["log_loss"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
