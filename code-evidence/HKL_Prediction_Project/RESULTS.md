# Model Evaluation Results

This report is generated from the pipeline's chronological holdout evaluation. It can be reproduced with the commands in the Reproduction section.

## Headline Results

The **Random Forest** produced the highest balanced accuracy at **69.4%**, 19.4 percentage points above the majority-class baseline of **50.0%**.

Its standard accuracy was **66.2%**, 0.6 percentage points below the majority-class accuracy of **66.8%**. Unlike that baseline, the model identifies both top-4 and non-top-4 outcomes; this is why balanced accuracy is more informative.

The highest ROC-AUC was **0.773** from the **Soft-Voting Ensemble**. ROC-AUC is the main threshold-independent measure used here because the target classes are imbalanced.

The **Random Forest** achieved the highest F1 score at **60.8%**, with **79.1%** recall. This configuration identifies more top-4 finishers at the cost of more false positives.

## Dataset And Evaluation

| Statistic | Value |
|---|---:|
| Target | `FinishPosition <= 4` |
| Validation strategy | Chronological holdout of the final 20% of races |
| Historical rows | 107,033 |
| Training rows | 85,870 |
| Validation rows | 21,163 |
| Training period | 2015-01-01 to 2023-08-11 |
| Validation period | 2023-08-11 to 2025-12-03 |
| Training top-4 rate | 32.8% |
| Validation top-4 rate | 33.2% |
| Majority-class baseline accuracy | 66.8% |
| Majority-class baseline balanced accuracy | 50.0% |

## Validation Metrics

| Model | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.45 | 66.4% | 69.3% | 49.6% | 78.1% | 60.7% | 0.769 |
| Random Forest | 0.45 | 66.2% | 69.4% | 49.4% | 79.1% | 60.8% | 0.772 |
| Histogram Gradient Boosting | 0.30 | 66.3% | 69.3% | 49.5% | 78.4% | 60.7% | 0.771 |
| Soft-Voting Ensemble | 0.40 | 66.3% | 69.4% | 49.5% | 78.7% | 60.8% | 0.773 |

## Confusion Matrix Counts

Positive means the horse finished in the top four.

| Model | True negative | False positive | False negative | True positive | Specificity |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 8,551 | 5,579 | 1,539 | 5,494 | 60.5% |
| Random Forest | 8,442 | 5,688 | 1,471 | 5,562 | 59.7% |
| Histogram Gradient Boosting | 8,515 | 5,615 | 1,521 | 5,512 | 60.3% |
| Soft-Voting Ensemble | 8,493 | 5,637 | 1,501 | 5,532 | 60.1% |

## Interpretation

- **Accuracy** is the proportion of all validation rows classified correctly.
- **Balanced accuracy** gives equal weight to top-4 and non-top-4 recall.
- **Precision** is the proportion of predicted top-4 finishes that were correct.
- **Recall** is the proportion of actual top-4 finishes identified by the model.
- **F1** balances precision and recall at the tuned decision threshold.
- **ROC-AUC** measures ranking quality across all possible thresholds; 0.5 is random.

## Methodology And Limitations

Races were ordered by date and race number, with the final 20% of whole races held out for validation. This prevents runners from one race appearing on both sides of the split and is more representative of future prediction than a random split. Decision thresholds were selected on this same holdout set, so threshold-dependent metrics should be treated as development estimates rather than a final independent test.

The results demonstrate an offline machine-learning workflow; they do not establish future profitability, and changing race conditions or data quality may reduce performance.

## Automated Checks

| Test | What it verifies |
|---|---|
| Metric accounting | Accuracy and confusion-matrix counts are calculated correctly |
| Split integrity | Every race stays entirely within training or validation data |
| Report generation | Saved metrics and metadata produce the expected Markdown report |

## Reproduction

```bash
python -m hkl_prediction run
python -m hkl_prediction report --report-path RESULTS.md
python -m unittest discover -s tests
```

Report generated from pipeline artefacts at `2026-07-21T13:09:18Z`.
