"""Training orchestration for the KGRec-music LightGCN model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .data import BPRSampler, build_normalized_adjacency, load_kgrec_data
from .evaluation import evaluate_topk
from .model import LightGCN


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(
        f"{name}={value:.5f}"
        for name, value in metrics.items()
        if name != "evaluated_users"
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0 or args.samples_per_epoch <= 0:
        raise ValueError("batch_size and samples_per_epoch must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval_every must be positive")

    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    data = load_kgrec_data(args.data_dir)
    adjacency = build_normalized_adjacency(data, device)
    sampler = BPRSampler(data, args.seed)

    model = LightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    history: list[dict[str, object]] = []
    best_score = float("-inf")
    stale_evaluations = 0
    selection_metric = f"ndcg@{max(args.top_k)}"

    print(
        f"Device={device} | users={data.num_users} | items={data.num_items} | "
        f"training interactions={len(data.train_users)}"
    )
    print(
        f"LightGCN: dimension={args.embedding_dim}, layers={args.num_layers}, "
        f"BPR samples/epoch={args.samples_per_epoch}, batch size={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        users, positives, negatives = sampler.sample(args.samples_per_epoch)
        total_loss = 0.0
        total_ranking_loss = 0.0
        total_regularization = 0.0
        processed_samples = 0

        for start in range(0, len(users), args.batch_size):
            end = min(start + args.batch_size, len(users))
            batch_users = users[start:end].to(device)
            batch_positives = positives[start:end].to(device)
            batch_negatives = negatives[start:end].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, ranking_loss, regularization = model.bpr_loss(
                batch_users,
                batch_positives,
                batch_negatives,
                adjacency,
                args.reg_weight,
            )
            loss.backward()
            optimizer.step()

            batch_count = end - start
            processed_samples += batch_count
            total_loss += float(loss.item()) * batch_count
            total_ranking_loss += float(ranking_loss.item()) * batch_count
            total_regularization += float(regularization.item()) * batch_count

        mean_loss = total_loss / processed_samples
        mean_ranking_loss = total_ranking_loss / processed_samples
        mean_regularization = total_regularization / processed_samples

        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "loss": mean_loss,
            "ranking_loss": mean_ranking_loss,
            "regularization": mean_regularization,
            "seconds": time.perf_counter() - started,
        }
        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            model.eval()
            user_embeddings, item_embeddings = model.propagate(adjacency)
            validation_metrics = evaluate_topk(
                user_embeddings,
                item_embeddings,
                data.validation_targets,
                data.train_seen,
                args.top_k,
                args.eval_batch_size,
            )
            epoch_record["validation"] = validation_metrics
            score = validation_metrics[selection_metric]
            print(
                f"Epoch {epoch:03d} | loss={mean_loss:.5f} | "
                f"{_format_metrics(validation_metrics)}"
            )

            if score > best_score:
                best_score = score
                stale_evaluations = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_users": data.num_users,
                        "num_items": data.num_items,
                        "embedding_dim": args.embedding_dim,
                        "num_layers": args.num_layers,
                        "epoch": epoch,
                        "selection_metric": selection_metric,
                        "selection_score": score,
                    },
                    checkpoint_path,
                )
            else:
                stale_evaluations += 1

            if args.patience > 0 and stale_evaluations >= args.patience:
                history.append(epoch_record)
                print(f"Early stopping after {stale_evaluations} evaluations without improvement.")
                break
        else:
            print(f"Epoch {epoch:03d} | loss={mean_loss:.5f}")
        history.append(epoch_record)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_metrics: dict[str, float] | None = None
    if not args.skip_test:
        user_embeddings, item_embeddings = model.propagate(adjacency)
        test_metrics = evaluate_topk(
            user_embeddings,
            item_embeddings,
            data.test_targets,
            data.test_exclusions(),
            args.top_k,
            args.eval_batch_size,
        )

    results: dict[str, object] = {
        "configuration": {
            "data_dir": str(args.data_dir),
            "device": str(device),
            "seed": args.seed,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "epochs_requested": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "reg_weight": args.reg_weight,
            "top_k": args.top_k,
            "eval_every": args.eval_every,
            "patience": args.patience,
        },
        "best_epoch": checkpoint["epoch"],
        "selection_metric": checkpoint["selection_metric"],
        "best_validation_score": checkpoint["selection_score"],
        "test_metrics": test_metrics,
        "history": history,
    }
    (args.output_dir / "training_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Best epoch: {checkpoint['epoch']}")
    if test_metrics is None:
        print("Test evaluation skipped; use the validation history for model selection.")
    else:
        print(f"Test | {_format_metrics(test_metrics)}")
    print(f"Saved checkpoint and results to {args.output_dir}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LightGCN on KGRec-music.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed" / "kgrec_music",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "lightgcn",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-epoch", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many evaluations without improvement; 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not evaluate the held-out test split; useful for validation-only tuning.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
