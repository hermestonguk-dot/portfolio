"""Recreate simple plots from the compact result tables shipped with the repo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "report_evidence"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_depth(output_dir: Path) -> None:
    rows = read_rows(EVIDENCE / "derived_data" / "figure_2_layer_sweep.csv")
    layers = [int(row["layers"]) for row in rows]
    values = [float(row["validation_ndcg20_mean"]) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(layers, values, marker="o")
    plt.xlabel("Propagation layers")
    plt.ylabel("Validation NDCG@20")
    plt.title("Validation performance by LightGCN depth")
    plt.tight_layout()
    plt.savefig(output_dir / "depth_summary.png", dpi=160)
    plt.close()


def plot_model_comparison(output_dir: Path) -> None:
    rows = read_rows(EVIDENCE / "derived_data" / "figure_4_final_model_comparison.csv")
    labels = [row["model"] + " " + row["split"] for row in rows]
    values = [float(row["ndcg20_mean"]) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color=["#2d6a6a", "#d9895b", "#2d6a6a", "#d9895b"])
    plt.ylabel("NDCG@20")
    plt.title("Matched model comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_summary.png", dpi=160)
    plt.close()


def plot_reranking(output_dir: Path) -> None:
    rows = read_rows(EVIDENCE / "derived_data" / "figure_6_reranking_test.csv")
    labels = [row["policy"] for row in rows]
    coverage = [float(row["catalogue_coverage20_mean"]) for row in rows]
    ndcg = [float(row["test_ndcg20_mean"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, coverage, color="#2d6a6a")
    axes[0].set_title("Catalogue coverage")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, ndcg, color="#d9895b")
    axes[1].set_title("Test NDCG@20")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_dir / "reranking_summary.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "plots")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_depth(args.output_dir)
    plot_model_comparison(args.output_dir)
    plot_reranking(args.output_dir)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
