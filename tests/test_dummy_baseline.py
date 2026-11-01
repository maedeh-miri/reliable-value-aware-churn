"""Unit tests for the dummy churn baseline."""

import unittest

import pandas as pd

from experiments.dummy_baseline_cv import evaluate_dummy_cv


class TestDummyBaseline(unittest.TestCase):
    """Verify the structure and expected behavior of dummy CV."""

    def setUp(self):
        self.X = pd.DataFrame({
            "feature": list(range(100)),
        })

        self.y = pd.Series(
            [0, 1] * 50,
            name="Churn",
        )

        self.groups = pd.Series(
            list(range(100)),
            name="group",
        )

    def test_returns_five_folds(self):
        results = evaluate_dummy_cv(
            self.X,
            self.y,
            self.groups,
        )

        self.assertEqual(len(results), 5)
        self.assertEqual(results["fold"].tolist(), [0, 1, 2, 3, 4])

    def test_expected_metric_columns_exist(self):
        results = evaluate_dummy_cv(
            self.X,
            self.y,
            self.groups,
        )

        expected_columns = {
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
        }

        self.assertTrue(
            expected_columns.issubset(results.columns)
        )

    def test_dummy_has_no_ranking_power(self):
        results = evaluate_dummy_cv(
            self.X,
            self.y,
            self.groups,
        )

        for auc_value in results["roc_auc"]:
            self.assertAlmostEqual(auc_value, 0.5)

    def test_dummy_predicts_no_positive_labels(self):
        results = evaluate_dummy_cv(
            self.X,
            self.y,
            self.groups,
        )

        self.assertTrue((results["precision"] == 0).all())
        self.assertTrue((results["recall"] == 0).all())


if __name__ == "__main__":
    unittest.main()
