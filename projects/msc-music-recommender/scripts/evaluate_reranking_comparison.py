"""Compare base ranking, Global Popularity Discount, and Smooth xQuAD.

Every method receives the same relevance-ranked candidate pool from the same
frozen checkpoint. Validation can sweep Smooth xQuAD lambda and select the
largest value that satisfies a predeclared NDCG-loss tolerance. The selected
lambda should then be passed explicitly to a separate test invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kgrec_lightgcn.data import build_normalized_adjacency, load_kgrec_data  # noqa: E402
from kgrec_lightgcn.fairness import (  # noqa: E402
    build_candidate_pool,
    evaluate_recommendation_lists,
    exposure_metrics,
    popularity_head_mask_by_interaction_share,
    recommendations_from_candidate_pool,
    rerank_candidate_pool_with_popularity_penalty,
    rerank_candidate_pool_with_smooth_xquad,
    train_item_popularity,
    user_head_tail_preferences,
)
from kgrec_lightgcn.model import LightGCN  # noqa: E402


CONFIG_FIELDS = {
    "split",
    "method",
    "parameter_name",
    "parameter_value",
    "checkpoint",
    "seed",
    "checkpoint_epoch",
    "selection_metric",
    "selection_score",
    "embedding_dim",
    "num_layers",
    "candidate_pool_size",
    "head_interaction_share_target",
    "head_interaction_share_actual",
    "head_popularity_cutoff",
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_checkpoint(path_or_dir: Path, device: torch.device) -> tuple[Path, dict[str, object]]:
    checkpoint_path = path_or_dir / "best_model.pt" if path_or_dir.is_dir() else path_or_dir
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    return checkpoint_path, torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )


def checkpoint_seed(path: Path) -> str:
    match = re.search(r"seed(\d+)", str(path), flags=re.IGNORECASE)
    return match.group(1) if match else path.parent.name


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_row(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    seed: str,
    method: str,
    parameter_name: str,
    parameter_value: float,
    recommendations: dict[int, list[int]],
    targets: dict[int, int],
    popularity: torch.Tensor,
    head_mask: torch.Tensor,
    head_interaction_share_actual: float,
    head_popularity_cutoff: float,
) -> dict[str, object]:
    accuracy = evaluate_recommendation_lists(
        recommendations, targets=targets, top_ks=args.top_k
    )
    exposure = exposure_metrics(
        recommendations,
        num_items=popularity.numel(),
        popularity=popularity,
        top_ks=args.top_k,
        head_fraction=float(head_mask.float().mean().item()),
        head_mask=head_mask,
    )
    row: dict[str, object] = {
        "split": args.split,
        "method": method,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_metric": str(checkpoint["selection_metric"]),
        "selection_score": float(checkpoint["selection_score"]),
        "embedding_dim": int(checkpoint["embedding_dim"]),
        "num_layers": int(checkpoint["num_layers"]),
        "candidate_pool_size": args.candidate_pool_size,
        "head_interaction_share_target": args.head_interaction_share,
        "head_interaction_share_actual": head_interaction_share_actual,
        "head_popularity_cutoff": head_popularity_cutoff,
    }
    row.update(accuracy)
    row.update(exposure)
    return row


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["split"]),
            str(row["method"]),
            str(row["parameter_name"]),
            float(row["parameter_value"]),
        )
        grouped[key].append(row)

    aggregate: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        split, method, parameter_name, parameter_value = key
        summary: dict[str, object] = {
            "split": split,
            "method": method,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "run_count": len(group),
        }
        metric_keys = sorted(set(group[0]) - CONFIG_FIELDS)
        for metric in metric_keys:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregate.append(summary)
    return aggregate


def select_smooth_lambda(
    aggregate: list[dict[str, object]],
    selection_metric: str,
    maximum_relative_loss: float,
) -> dict[str, object]:
    metric_name = f"{selection_metric}_mean"
    base_rows = [row for row in aggregate if row["method"] == "base_relevance"]
    if len(base_rows) != 1:
        raise ValueError("Expected exactly one aggregate base-relevance row")
    base_value = float(base_rows[0][metric_name])
    threshold = base_value * (1.0 - maximum_relative_loss)

    smooth_rows = [
        row
        for row in aggregate
        if row["method"] == "smooth_xquad"
        and float(row[metric_name]) >= threshold
    ]
    if smooth_rows:
        selected = max(smooth_rows, key=lambda row: float(row["parameter_value"]))
        selected_lambda = float(selected["parameter_value"])
        selected_value = float(selected[metric_name])
    else:
        selected_lambda = 0.0
        selected_value = base_value

    return {
        "selection_split": "validation",
        "selection_metric": selection_metric,
        "maximum_relative_loss": maximum_relative_loss,
        "base_value": base_value,
        "minimum_acceptable_value": threshold,
        "selected_smooth_xquad_lambda": selected_lambda,
        "selected_value": selected_value,
        "selected_relative_loss": (
            (base_value - selected_value) / base_value if base_value else 0.0
        ),
        "rule": (
            "Largest tested Smooth xQuAD lambda whose aggregate validation "
            f"{selection_metric} loss does not exceed "
            f"{maximum_relative_loss:.1%} relative to base ranking."
        ),
    }


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if args.candidate_pool_size < max(args.top_k):
        raise ValueError("candidate_pool_size must be at least max(top_k)")
    if not args.checkpoints:
        raise ValueError("At least one checkpoint is required")

    device = resolve_device(args.device)
    data = load_kgrec_data(args.data_dir)
    adjacency = build_normalized_adjacency(data, device)
    popularity = train_item_popularity(data)
    head_mask = popularity_head_mask_by_interaction_share(
        popularity, args.head_interaction_share
    )
    user_preferences = user_head_tail_preferences(data.train_seen, head_mask)
    head_interaction_share_actual = float(
        popularity[head_mask].sum().item() / popularity.sum().item()
    )
    head_popularity_cutoff = float(popularity[head_mask].min().item())

    if args.split == "validation":
        targets = data.validation_targets
        exclusions = data.train_seen
    else:
        targets = data.test_targets
        exclusions = data.test_exclusions()
    users = sorted(targets)
    max_k = max(args.top_k)
    rows: list[dict[str, object]] = []

    print(
        f"device={device} | split={args.split} | users={len(users)} | "
        f"items={data.num_items} | head_items={int(head_mask.sum())} | "
        f"head_event_share={head_interaction_share_actual:.5f}"
    )

    for checkpoint_input in args.checkpoints:
        checkpoint_path, checkpoint = load_checkpoint(checkpoint_input, device)
        if int(checkpoint["num_users"]) != data.num_users:
            raise ValueError(f"User count mismatch in {checkpoint_path}")
        if int(checkpoint["num_items"]) != data.num_items:
            raise ValueError(f"Item count mismatch in {checkpoint_path}")

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

        candidate_pool = build_candidate_pool(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            users=users,
            exclusions=exclusions,
            batch_size=args.eval_batch_size,
            candidate_pool_size=args.candidate_pool_size,
        )
        seed = checkpoint_seed(checkpoint_path)

        method_outputs: list[tuple[str, str, float, dict[int, list[int]]]] = [
            (
                "base_relevance",
                "none",
                0.0,
                recommendations_from_candidate_pool(candidate_pool, max_k),
            )
        ]
        for alpha in args.popularity_weights:
            method_outputs.append(
                (
                    "global_popularity_discount",
                    "alpha",
                    alpha,
                    rerank_candidate_pool_with_popularity_penalty(
                        candidate_pool=candidate_pool,
                        popularity=popularity,
                        top_k=max_k,
                        fairness_weight=alpha,
                    ),
                )
            )
        for lambda_weight in args.smooth_xquad_lambdas:
            method_outputs.append(
                (
                    "smooth_xquad",
                    "lambda",
                    lambda_weight,
                    rerank_candidate_pool_with_smooth_xquad(
                        candidate_pool=candidate_pool,
                        head_mask=head_mask,
                        user_preferences=user_preferences,
                        top_k=max_k,
                        lambda_weight=lambda_weight,
                    ),
                )
            )

        for method, parameter_name, parameter_value, recommendations in method_outputs:
            row = method_row(
                args=args,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                seed=seed,
                method=method,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                recommendations=recommendations,
                targets=targets,
                popularity=popularity,
                head_mask=head_mask,
                head_interaction_share_actual=head_interaction_share_actual,
                head_popularity_cutoff=head_popularity_cutoff,
            )
            rows.append(row)
            print(
                f"seed={seed} | {method} {parameter_name}={parameter_value:g} | "
                f"recall@{max_k}={row[f'recall@{max_k}']:.5f} | "
                f"ndcg@{max_k}={row[f'ndcg@{max_k}']:.5f} | "
                f"coverage@{max_k}={row[f'catalogue_coverage@{max_k}']:.5f} | "
                f"tail-coverage@{max_k}="
                f"{row[f'long_tail_catalogue_coverage@{max_k}']:.5f} | "
                f"ARP@{max_k}={row[f'arp@{max_k}']:.2f}"
            )

        del model, user_embeddings, item_embeddings, candidate_pool
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "comparison_results.csv", rows)
    (args.output_dir / "comparison_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "comparison_aggregate.csv", aggregate)

    run_config = {
        "split": args.split,
        "checkpoints": [str(path) for path in args.checkpoints],
        "data_dir": str(args.data_dir),
        "candidate_pool_size": args.candidate_pool_size,
        "top_k": args.top_k,
        "popularity_weights": args.popularity_weights,
        "smooth_xquad_lambdas": args.smooth_xquad_lambdas,
        "head_interaction_share_target": args.head_interaction_share,
        "head_interaction_share_actual": head_interaction_share_actual,
        "head_item_count": int(head_mask.sum().item()),
        "tail_item_count": int((~head_mask).sum().item()),
        "head_popularity_cutoff": head_popularity_cutoff,
        "profile_source": "training interactions only",
        "device": str(device),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    if args.split == "validation" and args.smooth_xquad_lambdas:
        selection = select_smooth_lambda(
            aggregate,
            selection_metric=args.selection_metric,
            maximum_relative_loss=args.maximum_ndcg_loss,
        )
        (args.output_dir / "smooth_xquad_selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        print(
            "Selected Smooth xQuAD lambda="
            f"{selection['selected_smooth_xquad_lambda']:g} using "
            f"{args.selection_metric} with a "
            f"{args.maximum_ndcg_loss:.1%} loss tolerance."
        )

    print(f"Saved matched reranking comparison to {args.output_dir}")
    return rows, aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare unchanged base ranking, Global Popularity Discount, and "
            "paper-backed Smooth xQuAD on identical candidate pools."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed" / "kgrec_music",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Checkpoint files or directories containing best_model.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "reranking_comparison",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Tune lambda on validation; pass only the selected lambda to test.",
    )
    parser.add_argument(
        "--popularity-weights",
        type=float,
        nargs="*",
        default=[0.2],
        help="Global Popularity Discount alpha values to compare.",
    )
    parser.add_argument(
        "--smooth-xquad-lambdas",
        type=float,
        nargs="*",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Smooth xQuAD lambda values. The source paper evaluates 0 to 1.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--head-interaction-share",
        type=float,
        default=0.8,
        help="Training-event share defining the paper's short-head category.",
    )
    parser.add_argument(
        "--selection-metric",
        default="ndcg@20",
        help="Aggregate validation metric used for Smooth xQuAD selection.",
    )
    parser.add_argument(
        "--maximum-ndcg-loss",
        type=float,
        default=0.01,
        help="Maximum relative validation utility loss for lambda selection.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
