"""Evaluate post-training popularity-aware reranking for LightGCN."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kgrec_lightgcn.data import build_normalized_adjacency, load_kgrec_data  # noqa: E402
from kgrec_lightgcn.fairness import (  # noqa: E402
    evaluate_recommendation_lists,
    exposure_metrics,
    rerank_with_popularity_penalty,
    train_item_popularity,
)
from kgrec_lightgcn.model import LightGCN  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_checkpoint(path_or_dir: Path, device: torch.device) -> dict[str, object]:
    checkpoint_path = path_or_dir / "best_model.pt" if path_or_dir.is_dir() else path_or_dir
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    device = resolve_device(args.device)
    data = load_kgrec_data(args.data_dir)
    adjacency = build_normalized_adjacency(data, device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    model = LightGCN(
        num_users=int(checkpoint["num_users"]),
        num_items=int(checkpoint["num_items"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(adjacency)

    if args.split == "validation":
        targets = data.validation_targets
        exclusions = data.train_seen
    else:
        targets = data.test_targets
        exclusions = data.test_exclusions()

    users = sorted(targets)
    popularity = train_item_popularity(data)
    rows: list[dict[str, object]] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for weight in args.fairness_weights:
        recommendations = rerank_with_popularity_penalty(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            users=users,
            exclusions=exclusions,
            popularity=popularity,
            top_k=max(args.top_k),
            batch_size=args.eval_batch_size,
            fairness_weight=weight,
            candidate_pool_size=args.candidate_pool_size,
        )
        accuracy = evaluate_recommendation_lists(recommendations, targets, args.top_k)
        exposure = exposure_metrics(
            recommendations,
            num_items=data.num_items,
            popularity=popularity,
            top_ks=args.top_k,
            head_fraction=args.head_fraction,
        )
        row: dict[str, object] = {
            "split": args.split,
            "fairness_weight": weight,
            "candidate_pool_size": args.candidate_pool_size,
            "checkpoint_epoch": checkpoint["epoch"],
            "selection_metric": checkpoint["selection_metric"],
            "selection_score": checkpoint["selection_score"],
        }
        row.update(accuracy)
        row.update(exposure)
        rows.append(row)

        print(
            f"weight={weight:g} | "
            f"recall@20={row.get('recall@20', 0.0):.5f} | "
            f"ndcg@20={row.get('ndcg@20', 0.0):.5f} | "
            f"coverage@20={row.get('catalogue_coverage@20', 0.0):.5f} | "
            f"gini@20={row.get('exposure_gini@20', 0.0):.5f} | "
            f"tail/head@20={row.get('tail_to_head_exposure_ratio@20', 0.0):.5f}"
        )

    (args.output_dir / "fair_rerank_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "fair_rerank_results.csv", rows)
    print(f"Saved fairness evaluation to {args.output_dir}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LightGCN with post-training popularity-aware reranking."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed" / "kgrec_music",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts") / "lightgcn_500",
        help="Checkpoint directory containing best_model.pt, or a checkpoint file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "fair_rerank",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Use validation during design; reserve test for final reporting.",
    )
    parser.add_argument(
        "--fairness-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4],
        help="Strength of the log-popularity penalty. 0.0 is the non-fair baseline.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--head-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
