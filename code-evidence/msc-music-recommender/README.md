# Music Recommendation With LightGCN

This MSc project implements a graph collaborative-filtering recommender for
the KGRec-music dataset. It compares LightGCN configurations, evaluates
held-out songs against the full eligible catalogue, and studies whether
popularity-aware reranking can increase catalogue breadth while retaining
ranking quality.

## What is included

- `src/kgrec_lightgcn/`: data preparation, graph construction, LightGCN,
  ranking metrics, and popularity-aware reranking.
- `scripts/`: dataset preparation, training, repeated sweeps, evaluation,
  reranking comparisons, and a small presentation demo.
- `tests/`: focused checks for graph normalisation, BPR sampling, ranking
  metrics, and reranking behaviour.
- `results/`: compact dataset and experiment summaries used in the report.
- `report_evidence/`: derived tables and figures for the included results.

## Results snapshot

The frozen three-seed comparison used 5,199 users, 8,640 items, and 751,531
implicit interactions. The selected one-layer, 128-dimensional model reached
mean test NDCG@20 of **0.1294** (SD **0.0008**). The zero-layer 128-dimensional
model reached **0.1135** (SD **0.0020**) under the same evaluation protocol.

The selected Smooth xQuAD reranker reduced test NDCG@20 to **0.1288** while
increasing catalogue coverage from **0.6439** to **0.6855** and long-tail
coverage from **0.4417** to **0.5073**. These measurements describe popularity
bias in item exposure; they are not a claim about demographic fairness.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The raw KGRec-music dataset is not included. Place it at:

```text
KGRec-dataset/KGRec-music/
```

The directory must contain `implicit_lf_dataset.csv` and the `tags/` and
`descriptions/` directories where available. The interaction file is read as
tab-separated data despite its `.csv` extension.

## Run the pipeline

Prepare the mapped interactions and per-user split:

```powershell
python .\scripts\prepare_kgrec.py
```

Run one small training check:

```powershell
python .\scripts\train_lightgcn.py --epochs 1 --samples-per-epoch 2048 --batch-size 2048 --embedding-dim 8 --num-layers 1 --top-k 10
```

Run the focused tests:

```powershell
python -m unittest discover -s tests -v
```

The full experiment scripts require the processed dataset and checkpoints.
They write generated files under `artifacts/`, which is excluded from the
portfolio because the trained checkpoints are large and the results are
already represented by the compact files in `results/` and `report_evidence/`.

## Evaluation design

Users are split into training, validation, and test interactions. Training
items are masked during validation, while training and validation items are
masked during test evaluation. Each held-out item is ranked against all
eligible catalogue items. The code reports Recall@K, Precision@K, NDCG@K, and
MRR, together with catalogue coverage, long-tail coverage, exposure Gini, and
average popularity rank for the reranking experiments.

The project uses validation data to select model settings and the Smooth xQuAD
weight. The test results are then kept separate for the final comparison.
