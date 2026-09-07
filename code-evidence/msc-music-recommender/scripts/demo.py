"""Presentation-only illustration of the existing reranking pipeline.

This script does not retrain or modify the recommender. It loads one saved
checkpoint, builds the same masked top-100 candidate pool used by the matched
reranking evaluation, and shows one test user's ranking under:

1. unchanged base relevance,
2. Global Popularity Discount (GPD), and
3. Smooth xQuAD.

The output is intended as an illustrative video/demo artefact. Aggregate claims
should continue to come from the full multi-seed evaluation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kgrec_lightgcn.data import build_normalized_adjacency, load_kgrec_data  # noqa: E402
from kgrec_lightgcn.fairness import (  # noqa: E402
    build_candidate_pool,
    popularity_head_mask_by_interaction_share,
    recommendations_from_candidate_pool,
    rerank_candidate_pool_with_popularity_penalty,
    rerank_candidate_pool_with_smooth_xquad,
    train_item_popularity,
    user_head_tail_preferences,
)
from kgrec_lightgcn.model import LightGCN  # noqa: E402


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return checkpoint_path, checkpoint


def policy_rows(
    *,
    policy: str,
    user: int,
    recommendations: list[int],
    popularity: torch.Tensor,
    head_mask: torch.Tensor,
    base_candidate_ranks: dict[int, int],
    test_target: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, item in enumerate(recommendations, start=1):
        base_rank = base_candidate_ranks[item]
        rows.append(
            {
                "policy": policy,
                "user_index": user,
                "rank": rank,
                "item_id": item,
                "training_popularity": int(popularity[item].item()),
                "popularity_group": "SHORT-HEAD" if bool(head_mask[item].item()) else "LONG-TAIL",
                "base_candidate_rank": base_rank,
                "rank_movement": base_rank - rank,
                "is_test_target": item == test_target,
            }
        )
    return rows


def print_policy(
    title: str,
    rows: list[dict[str, object]],
    *,
    top_k: int,
    display_k: int,
) -> None:
    tail_count = sum(row["popularity_group"] == "LONG-TAIL" for row in rows[:top_k])
    print("\n" + title)
    print("-" * len(title))
    print(f"{'Rank':>4}  {'Item':>7}  {'Popularity':>10}  {'Group':>10}  {'Base rank':>9}  {'Move':>6}  Target")
    for row in rows[:display_k]:
        movement = int(row["rank_movement"])
        movement_text = "-" if movement == 0 else (f"+{movement}" if movement > 0 else str(movement))
        target = "YES" if row["is_test_target"] else ""
        print(
            f"{int(row['rank']):>4}  "
            f"{int(row['item_id']):>7}  "
            f"{int(row['training_popularity']):>10}  "
            f"{str(row['popularity_group']):>10}  "
            f"{int(row['base_candidate_rank']):>9}  "
            f"{movement_text:>6}  "
            f"{target}"
        )
    print(f"Long-tail items in top {top_k}: {tail_count}/{top_k}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "user_index",
        "rank",
        "item_id",
        "training_popularity",
        "popularity_group",
        "base_candidate_rank",
        "rank_movement",
        "is_test_target",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.candidate_pool_size < args.top_k:
        raise ValueError("candidate-pool-size must be at least top-k")
    if not 0.0 <= args.lambda_weight <= 1.0:
        raise ValueError("lambda-weight must be between 0 and 1")
    if args.alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    device = resolve_device(args.device)
    data = load_kgrec_data(args.data_dir)

    user = args.user_index
    if user not in data.test_targets:
        raise ValueError(
            f"User {user} does not have a test target. Choose one of: "
            f"{min(data.test_targets)}..{max(data.test_targets)}"
        )

    checkpoint_path, checkpoint = load_checkpoint(args.checkpoint, device)
    if int(checkpoint["num_users"]) != data.num_users:
        raise ValueError("Checkpoint user count does not match the processed dataset")
    if int(checkpoint["num_items"]) != data.num_items:
        raise ValueError("Checkpoint item count does not match the processed dataset")

    print(f"checkpoint: {checkpoint_path}")
    print(
        f"model: {int(checkpoint['num_layers'])} propagation layer(s), "
        f"{int(checkpoint['embedding_dim'])}D | epoch={int(checkpoint['epoch'])}"
    )
    print(f"device: {device}")
    print(f"test seed: {user}")

    adjacency = build_normalized_adjacency(data, device)
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

    popularity = train_item_popularity(data)
    head_mask = popularity_head_mask_by_interaction_share(
        popularity, args.head_interaction_share
    )
    user_preferences = user_head_tail_preferences(data.train_seen, head_mask)

    # Test evaluation masks the user's training history plus validation item.
    # The held-out test target remains eligible, matching the real evaluation.
    exclusions = data.test_exclusions()
    candidate_pool = build_candidate_pool(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        users=[user],
        exclusions=exclusions,
        batch_size=args.eval_batch_size,
        candidate_pool_size=args.candidate_pool_size,
    )

    base = recommendations_from_candidate_pool(candidate_pool, args.top_k)[user]
    gpd = rerank_candidate_pool_with_popularity_penalty(
        candidate_pool=candidate_pool,
        popularity=popularity,
        top_k=args.top_k,
        fairness_weight=args.alpha,
    )[user]
    smooth = rerank_candidate_pool_with_smooth_xquad(
        candidate_pool=candidate_pool,
        head_mask=head_mask,
        user_preferences=user_preferences,
        top_k=args.top_k,
        lambda_weight=args.lambda_weight,
    )[user]

    candidate_ids = candidate_pool.item_ids[0].tolist()
    base_candidate_ranks = {item: rank for rank, item in enumerate(candidate_ids, start=1)}
    test_target = data.test_targets[user]

    profile = user_preferences[user]
    print(
        f"training profile: {len(data.train_seen[user])} songs | "
        f"tail preference={float(profile[0]):.1%} | "
        f"head preference={float(profile[1]):.1%}"
    )
    print(
        f"shared candidate pool: {args.candidate_pool_size} | final list: {args.top_k} | "
        f"GPD alpha={args.alpha:g} | Smooth xQuAD lambda={args.lambda_weight:g}"
    )
    print(f"held-out test target item: {test_target}")

    base_rows = policy_rows(
        policy="Base",
        user=user,
        recommendations=base,
        popularity=popularity,
        head_mask=head_mask,
        base_candidate_ranks=base_candidate_ranks,
        test_target=test_target,
    )
    gpd_rows = policy_rows(
        policy=f"GPD alpha={args.alpha:g}",
        user=user,
        recommendations=gpd,
        popularity=popularity,
        head_mask=head_mask,
        base_candidate_ranks=base_candidate_ranks,
        test_target=test_target,
    )
    smooth_rows = policy_rows(
        policy=f"Smooth xQuAD lambda={args.lambda_weight:g}",
        user=user,
        recommendations=smooth,
        popularity=popularity,
        head_mask=head_mask,
        base_candidate_ranks=base_candidate_ranks,
        test_target=test_target,
    )

    display_k = min(args.display_k, args.top_k)
    print_policy("BASE RANKING", base_rows, top_k=args.top_k, display_k=display_k)
    print_policy(
        f"GLOBAL POPULARITY DISCOUNT (alpha={args.alpha:g})",
        gpd_rows,
        top_k=args.top_k,
        display_k=display_k,
    )
    print_policy(
        f"SMOOTH XQUAD (lambda={args.lambda_weight:g})",
        smooth_rows,
        top_k=args.top_k,
        display_k=display_k,
    )

    all_rows = base_rows + gpd_rows + smooth_rows
    write_csv(args.output, all_rows)
    print(f"\nSaved to: {args.output}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show one real test user's base, GPD, and Smooth xQuAD rankings."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file or directory containing best_model.pt.",
    )
    parser.add_argument("--user-index", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed" / "kgrec_music",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--display-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lambda-weight", type=float, default=0.4)
    parser.add_argument("--head-interaction-share", type=float, default=0.8)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("presentation") / "user_rankings.csv",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
