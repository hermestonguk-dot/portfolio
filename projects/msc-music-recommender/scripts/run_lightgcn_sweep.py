"""Run repeated LightGCN experiments for tuning and variance checks."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kgrec_lightgcn.training import train  # noqa: E402


def safe_label(value: object) -> str:
    return str(value).replace("\\", "-").replace("/", "-").replace(".", "p")


def validation_at_best(results: dict[str, object]) -> dict[str, float]:
    best_epoch = results["best_epoch"]
    for record in results["history"]:
        if record["epoch"] == best_epoch and "validation" in record:
            return record["validation"]
    return {}


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_aggregate(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    group_keys = [
        "embedding_dim",
        "num_layers",
        "learning_rate",
        "reg_weight",
        "epochs_requested",
    ]
    metric_keys = sorted(
        key
        for key in {key for row in rows for key in row}
        if key == "best_validation_score" or key.startswith("val_")
    )
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for key, group_rows in grouped.items():
        aggregate: dict[str, object] = dict(zip(group_keys, key))
        aggregate["run_count"] = len(group_rows)
        for metric in metric_keys:
            values = [
                float(row[metric])
                for row in group_rows
                if metric in row and row[metric] != ""
            ]
            if not values:
                continue
            aggregate[f"{metric}_mean"] = mean(values)
            aggregate[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            aggregate[f"{metric}_min"] = min(values)
            aggregate[f"{metric}_max"] = max(values)
        aggregate_rows.append(aggregate)

    fieldnames = sorted({key for row in aggregate_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)


def run_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.output_root / args.run_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "sweep_summary.csv"
    aggregate_path = sweep_dir / "sweep_aggregate.csv"

    rows: list[dict[str, object]] = []
    combinations = itertools.product(
        args.data_dirs,
        args.embedding_dims,
        args.num_layers,
        args.learning_rates,
        args.reg_weights,
        args.seeds,
    )

    for data_dir, embedding_dim, num_layers, learning_rate, reg_weight, seed in combinations:
        run_id = (
            f"data-{safe_label(data_dir.name)}__dim{embedding_dim}"
            f"__layers{num_layers}__lr{safe_label(learning_rate)}"
            f"__reg{safe_label(reg_weight)}__seed{seed}"
        )
        output_dir = sweep_dir / run_id
        print(f"\n=== Running {run_id} ===")

        train_args = argparse.Namespace(
            data_dir=data_dir,
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            reg_weight=reg_weight,
            top_k=args.top_k,
            eval_batch_size=args.eval_batch_size,
            eval_every=args.eval_every,
            patience=args.patience,
            seed=seed,
            device=args.device,
            skip_test=not args.include_test,
        )
        results = train(train_args)
        best_validation = validation_at_best(results)

        row: dict[str, object] = {
            "run_id": run_id,
            "data_dir": str(data_dir),
            "seed": seed,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "reg_weight": reg_weight,
            "epochs_requested": args.epochs,
            "best_epoch": results["best_epoch"],
            "selection_metric": results["selection_metric"],
            "best_validation_score": results["best_validation_score"],
            "output_dir": str(output_dir),
        }
        row.update({f"val_{key}": value for key, value in best_validation.items()})
        test_metrics = results.get("test_metrics")
        if isinstance(test_metrics, dict):
            row.update({f"test_{key}": value for key, value in test_metrics.items()})
        rows.append(row)
        write_summary(summary_path, rows)
        write_aggregate(aggregate_path, rows)

    print(f"\nSaved sweep summary to {summary_path}")
    print(f"Saved sweep aggregate to {aggregate_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LightGCN grids and repeated seeds for validation tuning."
    )
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        default=[Path("data") / "processed" / "kgrec_music"],
        help="Processed split directories. Use multiple dirs to test hold-out split variance.",
    )
    parser.add_argument("--embedding-dims", type=int, nargs="+", default=[64])
    parser.add_argument("--num-layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--reg-weights", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples-per-epoch", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also evaluate test. Leave off during tuning to avoid test-set leakage.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "lightgcn_sweeps",
    )
    parser.add_argument("--run-name", default=None)
    return parser


def main() -> None:
    run_sweep(build_parser().parse_args())


if __name__ == "__main__":
    main()
