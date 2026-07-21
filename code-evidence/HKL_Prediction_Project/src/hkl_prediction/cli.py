from __future__ import annotations

import argparse
from pathlib import Path

from hkl_prediction.pipeline import (
    DEFAULT_HISTORICAL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_UPCOMING,
    PipelinePaths,
    predict_upcoming,
    run_pipeline,
    train_models,
)
from hkl_prediction.reporting import generate_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hkl-predict",
        description="Train and run the HKL horse racing prediction pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in [
        ("train", "Train models from historical race data."),
        ("predict", "Generate predictions with saved models."),
        ("run", "Train models and generate predictions."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        add_common_arguments(subparser)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate a Markdown evaluation report from saved metrics.",
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing metrics and models. Default: {DEFAULT_OUTPUT_DIR}",
    )
    report_parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("RESULTS.md"),
        help="Markdown report destination. Default: RESULTS.md",
    )

    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--historical",
        type=Path,
        default=DEFAULT_HISTORICAL,
        help=f"Historical training CSV. Default: {DEFAULT_HISTORICAL}",
    )
    parser.add_argument(
        "--upcoming",
        type=Path,
        default=DEFAULT_UPCOMING,
        help=f"Upcoming-races CSV. Default: {DEFAULT_UPCOMING}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional recent-row sample for quick smoke tests.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training.",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        paths = PipelinePaths(output_dir=args.output_dir)
        generate_markdown_report(
            metrics_path=paths.metrics_path,
            meta_path=paths.meta_path,
            report_path=args.report_path,
        )
        return

    paths = PipelinePaths(
        historical=args.historical,
        upcoming=args.upcoming,
        output_dir=args.output_dir,
    )

    if args.command == "train":
        train_models(paths, sample_size=args.sample_size, random_state=args.random_state)
    elif args.command == "predict":
        predict_upcoming(paths)
    elif args.command == "run":
        run_pipeline(paths, sample_size=args.sample_size, random_state=args.random_state)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
