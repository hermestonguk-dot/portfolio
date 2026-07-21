import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hkl_prediction.reporting import generate_markdown_report


class ReportingTests(unittest.TestCase):
    def test_generate_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_and_check_report(root)

    def _write_and_check_report(self, root: Path) -> None:
        metrics_path = root / "metrics.csv"
        meta_path = root / "meta.json"
        report_path = root / "RESULTS.md"

        pd.DataFrame(
            [
                {
                    "model": "ensemble_soft",
                    "threshold": 0.4,
                    "threshold_f1": 0.61,
                    "roc_auc": 0.77,
                    "accuracy": 0.69,
                    "balanced_accuracy": 0.71,
                    "precision": 0.53,
                    "recall": 0.72,
                    "specificity": 0.70,
                    "predicted_positive_rate": 0.45,
                    "true_negative": 70,
                    "false_positive": 30,
                    "false_negative": 14,
                    "true_positive": 36,
                    "validation_rows": 150,
                }
            ]
        ).to_csv(metrics_path, index=False)
        meta_path.write_text(
            json.dumps(
                {
                    "target_definition": "FinishPosition <= 4",
                    "validation_strategy": "Chronological holdout of the final 20% of races",
                    "historical_rows": 750,
                    "train_rows": 600,
                    "validation_rows": 150,
                    "training_date_start": "2020-01-01",
                    "training_date_end": "2023-12-31",
                    "validation_date_start": "2024-01-01",
                    "validation_date_end": "2024-12-31",
                    "training_positive_rate": 0.34,
                    "validation_positive_rate": 0.33,
                    "majority_baseline_accuracy": 0.67,
                    "generated_at": "2026-07-21T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )

        generate_markdown_report(metrics_path, meta_path, report_path)

        report = report_path.read_text(encoding="utf-8")
        self.assertIn("# Model Evaluation Results", report)
        self.assertIn("Soft-Voting Ensemble", report)
        self.assertIn("69.0%", report)
        self.assertIn("Majority-class baseline accuracy", report)
        self.assertIn("Confusion Matrix Counts", report)
