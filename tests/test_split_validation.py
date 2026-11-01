"""Automated tests for split validation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.split_validation import validate_split


class TestSplitValidation(unittest.TestCase):
    """Test valid and deliberately corrupted split inputs."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.directory = Path(self.temporary_directory.name)

        self.data_path = self.directory / "data.csv"
        self.manifest_path = self.directory / "manifest.csv"
        self.metadata_path = self.directory / "metadata.json"

        data = pd.DataFrame({
            "Feature": [10, 10, 20, 30, 40, 50],
            "Churn": [0, 0, 1, 0, 1, 0],
        })

        data.to_csv(self.data_path, index=False)

        manifest = pd.DataFrame({
            "row_index": [0, 1, 2, 3, 4, 5],
            "fold": [0, 0, 3, 3, 4, 4],
            "split": [
                "train",
                "train",
                "calibration",
                "calibration",
                "test",
                "test",
            ],
        })

        manifest.to_csv(self.manifest_path, index=False)

        dataset_hash = hashlib.sha256(
            self.data_path.read_bytes()
        ).hexdigest()

        metadata = {
            "dataset_sha256": dataset_hash,
            "row_count": 6,
            "grouping_columns": ["Feature"],
            "fold_to_split": {
                "0": "train",
                "3": "calibration",
                "4": "test",
            },
        }

        self.metadata_path.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

    def test_valid_manifest_passes(self):
        result = validate_split(
            self.data_path,
            self.manifest_path,
            self.metadata_path,
            expected_counts={
                "train": 2,
                "calibration": 2,
                "test": 2,
            },
        )

        self.assertEqual(result["row_count"], 6)
        self.assertEqual(result["group_count"], 5)

    def test_changed_dataset_hash_fails(self):
        data = pd.read_csv(self.data_path)

        data.loc[0, "Feature"] = 999
        data.to_csv(self.data_path, index=False)

        with self.assertRaisesRegex(
            ValueError,
            "Dataset SHA-256 mismatch",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
            )


    def test_missing_manifest_row_fails(self):
        manifest = pd.read_csv(self.manifest_path)
        manifest = manifest.iloc[:-1]
        manifest.to_csv(self.manifest_path, index=False)

        with self.assertRaisesRegex(
            ValueError,
            "Manifest row count mismatch",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
            )

    def test_invalid_split_label_fails(self):
        manifest = pd.read_csv(self.manifest_path)
        manifest.loc[2, "split"] = "train"
        manifest.to_csv(self.manifest_path, index=False)

        with self.assertRaisesRegex(
            ValueError,
            "Invalid split labels or fold mapping",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
            )

    def test_group_overlap_across_folds_fails(self):
        manifest = pd.read_csv(self.manifest_path)

        manifest.loc[1, "fold"] = 3
        manifest.loc[1, "split"] = "calibration"

        manifest.to_csv(self.manifest_path, index=False)

        with self.assertRaisesRegex(
            ValueError,
            "A group appears in multiple folds",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
            )

    def test_duplicate_row_index_fails(self):
        manifest = pd.read_csv(self.manifest_path)

        manifest.loc[3, "row_index"] = 2
        manifest.to_csv(self.manifest_path, index=False)

        with self.assertRaisesRegex(
            ValueError,
            "Invalid or incomplete row assignments",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
            )

    def test_invalid_fold_number_fails(self):
        manifest = pd.read_csv(self.manifest_path)

        manifest.loc[2, "fold"] = 9
        manifest.to_csv(self.manifest_path, index=False)

        with self.assertRaisesRegex(
            ValueError,
            "Invalid fold assignment",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
            )

    def test_missing_manifest_column_fails(self):
        manifest = pd.read_csv(self.manifest_path)

        manifest = manifest.drop(columns=["split"])
        manifest.to_csv(self.manifest_path, index=False)

        with self.assertRaisesRegex(
            ValueError,
            "Missing required manifest columns",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
            )

    def test_unexpected_split_sizes_fail(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unexpected split sizes",
        ):
            validate_split(
                self.data_path,
                self.manifest_path,
                self.metadata_path,
                expected_counts={
                    "train": 3,
                    "calibration": 1,
                    "test": 2,
                },
            )

if __name__ == "__main__":
    unittest.main()
