# Experiment Evidence

This folder contains compact tables and figures exported from the completed
experiments. It is included so the reported comparisons can be inspected
without downloading the raw dataset or large model checkpoints.

## Contents

- `derived_data/`: CSV tables used by the figures.
- `figures/`: PNG plots for depth, embedding size, final model comparison, and
  reranking behaviour.

The full checkpoints and processed interaction files remain outside this
portfolio repository because they are large. The runnable scripts in the
parent project can reproduce new experiments when those local inputs are
available.

To recreate the compact visual summaries from the included CSV tables:

```powershell
python .\scripts\plot_report_evidence.py
```
