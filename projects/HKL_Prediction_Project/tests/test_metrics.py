import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hkl_prediction.pipeline import metric_row, time_aware_split


class MetricRowTests(unittest.TestCase):
    def test_records_expected_confusion_matrix(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.7, 0.2])

        row = metric_row(
            "example",
            probabilities,
            y_true,
            threshold=0.5,
            threshold_f1=0.5,
        )

        self.assertEqual(row["true_negative"], 1)
        self.assertEqual(row["false_positive"], 1)
        self.assertEqual(row["false_negative"], 1)
        self.assertEqual(row["true_positive"], 1)
        self.assertEqual(row["accuracy"], 0.5)
        self.assertEqual(row["balanced_accuracy"], 0.5)

    def test_time_aware_split_keeps_each_race_together(self) -> None:
        data = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2025-01-01"] * 3 + ["2025-01-02"] * 3 + ["2025-01-03"] * 3
                ),
                "Race": [1] * 3 + [2] * 3 + [3] * 3,
                "HitTop4": [0, 1, 0] * 3,
            }
        )

        train, validation = time_aware_split(data, validation_fraction=0.2)

        train_races = set(zip(train["Date"], train["Race"]))
        validation_races = set(zip(validation["Date"], validation["Race"]))
        self.assertTrue(train_races.isdisjoint(validation_races))
        self.assertEqual(validation["Race"].unique().tolist(), [3])
