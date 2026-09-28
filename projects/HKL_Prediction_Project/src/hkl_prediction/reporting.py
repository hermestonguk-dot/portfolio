from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DISPLAY_NAMES = {
    "log_reg": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "ensemble_soft": "Soft-Voting Ensemble",
}


def generate_markdown_report(
    metrics_path: Path,
    meta_path: Path,
    report_path: Path,
) -> Path:
    """Generate a GitHub-readable evaluation report from pipeline artefacts."""

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}. Run training first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}. Run training first.")

    metrics = pd.read_csv(metrics_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(metrics, meta), encoding="utf-8")
    return report_path


def render_markdown(metrics: pd.DataFrame, meta: dict) -> str:
    required_columns = {
        "model",
        "threshold",
        "threshold_f1",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }
    missing = required_columns.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics file is missing columns: {', '.join(sorted(missing))}")

    best_f1 = metrics.loc[metrics["threshold_f1"].idxmax()]
    best_auc = metrics.loc[metrics["roc_auc"].idxmax()]
    best_balanced = metrics.loc[metrics["balanced_accuracy"].idxmax()]
    baseline_accuracy = float(meta["majority_baseline_accuracy"])
    accuracy_change = float(best_balanced["accuracy"]) - baseline_accuracy
    balanced_accuracy_change = float(best_balanced["balanced_accuracy"]) - 0.5

    lines = [
        "# Model Evaluation Results",
        "",
        (
            "This report is generated from the pipeline's chronological holdout evaluation. "
            "It can be reproduced with the commands in the Reproduction section."
        ),
        "",
        "## Headline Results",
        "",
        (
            f"The **{display_name(best_balanced['model'])}** produced the highest balanced "
            f"accuracy at **{pct(best_balanced['balanced_accuracy'])}**, "
            f"{percentage_points(balanced_accuracy_change)} percentage points above the "
            "majority-class baseline of **50.0%**."
        ),
        "",
        (
            f"Its standard accuracy was **{pct(best_balanced['accuracy'])}**, "
            f"{relative_difference(accuracy_change)} the majority-class accuracy of "
            f"**{pct(baseline_accuracy)}**. Unlike that baseline, the model identifies both "
            "top-4 and non-top-4 outcomes; this is why balanced accuracy is more informative."
        ),
        "",
        (
            f"The highest ROC-AUC was **{decimal(best_auc['roc_auc'])}** from the "
            f"**{display_name(best_auc['model'])}**. ROC-AUC is the main threshold-independent "
            "measure used here because the target classes are imbalanced."
        ),
        "",
        (
            f"The **{display_name(best_f1['model'])}** achieved the highest F1 score at "
            f"**{pct(best_f1['threshold_f1'])}**, with **{pct(best_f1['recall'])}** recall. "
            "This configuration identifies more top-4 finishers at the cost of more false positives."
        ),
        "",
        "## Dataset And Evaluation",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Target | `{meta.get('target_definition', 'HitTop4')}` |",
        f"| Validation strategy | {meta.get('validation_strategy', 'Holdout')} |",
        f"| Historical rows | {integer(meta.get('historical_rows'))} |",
        f"| Training rows | {integer(meta.get('train_rows'))} |",
        f"| Validation rows | {integer(meta.get('validation_rows'))} |",
        (
            f"| Training period | {meta.get('training_date_start', 'Unknown')} to "
            f"{meta.get('training_date_end', 'Unknown')} |"
        ),
        (
            f"| Validation period | {meta.get('validation_date_start', 'Unknown')} to "
            f"{meta.get('validation_date_end', 'Unknown')} |"
        ),
        f"| Training top-4 rate | {pct(meta.get('training_positive_rate'))} |",
        f"| Validation top-4 rate | {pct(meta.get('validation_positive_rate'))} |",
        f"| Majority-class baseline accuracy | {pct(baseline_accuracy)} |",
        "| Majority-class baseline balanced accuracy | 50.0% |",
        "",
        "## Validation Metrics",
        "",
        "| Model | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {display_name(row.model)} | {row.threshold:.2f} | {pct(row.accuracy)} | "
            f"{pct(row.balanced_accuracy)} | {pct(row.precision)} | {pct(row.recall)} | "
            f"{pct(row.threshold_f1)} | {decimal(row.roc_auc)} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix Counts",
            "",
            "Positive means the horse finished in the top four.",
            "",
            "| Model | True negative | False positive | False negative | True positive | Specificity |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {display_name(row.model)} | {int(row.true_negative):,} | "
            f"{int(row.false_positive):,} | {int(row.false_negative):,} | "
            f"{int(row.true_positive):,} | {pct(row.specificity)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Accuracy** is the proportion of all validation rows classified correctly.",
            "- **Balanced accuracy** gives equal weight to top-4 and non-top-4 recall.",
            "- **Precision** is the proportion of predicted top-4 finishes that were correct.",
            "- **Recall** is the proportion of actual top-4 finishes identified by the model.",
            "- **F1** balances precision and recall at the tuned decision threshold.",
            "- **ROC-AUC** measures ranking quality across all possible thresholds; 0.5 is random.",
            "",
            "## Methodology And Limitations",
            "",
            (
                "Races were ordered by date and race number, with the final 20% of whole races held "
                "out for validation. This prevents runners from one race appearing on both sides of "
                "the split and is more representative of future prediction than a random split. "
                "Decision thresholds were selected on this same holdout set, so threshold-dependent "
                "metrics should be treated as development estimates rather than a final independent test."
            ),
            "",
            (
                "The results demonstrate an offline machine-learning workflow; they do not establish "
                "future profitability, and changing race conditions or data quality may reduce performance."
            ),
            "",
            "## Automated Checks",
            "",
            "| Test | What it verifies |",
            "|---|---|",
            "| Metric accounting | Accuracy and confusion-matrix counts are calculated correctly |",
            "| Split integrity | Every race stays entirely within training or validation data |",
            "| Report generation | Saved metrics and metadata produce the expected Markdown report |",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m hkl_prediction run",
            "python -m hkl_prediction report --report-path RESULTS.md",
            "python -m unittest discover -s tests",
            "```",
            "",
            f"Report generated from pipeline artefacts at `{meta.get('generated_at', 'unknown')}`.",
            "",
        ]
    )
    return "\n".join(lines)


def display_name(model: object) -> str:
    key = str(model)
    return DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def pct(value: object) -> str:
    return f"{float(value):.1%}"


def decimal(value: object) -> str:
    return f"{float(value):.3f}"


def integer(value: object) -> str:
    return f"{int(value):,}"


def percentage_points(value: float) -> str:
    return f"{abs(value) * 100:.1f}"


def relative_difference(value: float) -> str:
    direction = "above" if value >= 0 else "below"
    return f"{abs(value) * 100:.1f} percentage points {direction}"
