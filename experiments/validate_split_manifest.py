"""Validate the frozen v1 split against its original contract."""

import json
from pathlib import Path

from experiments.split_validation import validate_split


DATA_PATH = Path("data/raw/iranian_churn.csv")
MANIFEST_PATH = Path("data/splits/v1_manifest.csv")
METADATA_PATH = Path("data/splits/v1_metadata.json")

EXPECTED_SHA256 = (
    "395d04854ce6daa1cdee8ab820df5a90691de9277dbba72096b8b01a60a655d1"
)

EXPECTED_ROW_COUNT = 3150

EXPECTED_COUNTS = {
    "train": 1894,
    "calibration": 628,
    "test": 628,
}

EXPECTED_FOLD_MAPPING = {
    "0": "train",
    "1": "train",
    "2": "train",
    "3": "calibration",
    "4": "test",
}

EXPECTED_GROUPING_COLUMNS = [
    "Call  Failure",
    "Complains",
    "Subscription  Length",
    "Charge  Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
]


def main():
    """Validate the original v1 split and its frozen metadata."""

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if metadata["dataset_sha256"] != EXPECTED_SHA256:
        raise ValueError("Frozen dataset hash changed.")

    if metadata["row_count"] != EXPECTED_ROW_COUNT:
        raise ValueError("Frozen row count changed.")

    if metadata["fold_to_split"] != EXPECTED_FOLD_MAPPING:
        raise ValueError("Frozen fold mapping changed.")

    if metadata["grouping_columns"] != EXPECTED_GROUPING_COLUMNS:
        raise ValueError("Frozen grouping columns changed.")

    if metadata["n_splits"] != 5:
        raise ValueError("Frozen number of folds changed.")

    if metadata["random_state"] != 42:
        raise ValueError("Frozen random state changed.")

    result = validate_split(
        data_path=DATA_PATH,
        manifest_path=MANIFEST_PATH,
        metadata_path=METADATA_PATH,
        expected_counts=EXPECTED_COUNTS,
    )

    print("Dataset SHA-256: PASSED")
    print("Row assignment integrity: PASSED")
    print("Fold mapping: PASSED")
    print("Split sizes: PASSED")
    print("Group isolation: PASSED")

    print("\nSplit validation summary:")
    print(result)

    print("\nAll split validation checks passed.")


if __name__ == "__main__":
    main()
