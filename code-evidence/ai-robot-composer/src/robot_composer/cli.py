from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from robot_composer.analysis import analyze_midi
from robot_composer.generation import GenerationConfig, generate_events, write_midi
from robot_composer.model import ModelConfig, load_model, parameter_count
from robot_composer.reporting import generate_report
from robot_composer.vocabulary import Vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "V7.1_final_paper_VAD.pth"
DEFAULT_VOCABULARY = PROJECT_ROOT / "artifacts" / "vocabulary.json"
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "samples"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-composer",
        description="Generate and inspect VAD-conditioned MIDI with the final-year model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect model and vocabulary metadata.")
    add_asset_arguments(inspect_parser)

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Load the checkpoint and run a small forward pass.",
    )
    add_asset_arguments(smoke_parser)

    generate_parser = subparsers.add_parser("generate", help="Generate a conditioned MIDI file.")
    add_asset_arguments(generate_parser)
    generate_parser.add_argument("--valence", type=float, required=True)
    generate_parser.add_argument("--arousal", type=float, required=True)
    generate_parser.add_argument("--dominance", type=float, required=True)
    generate_parser.add_argument("--length", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.6)
    generate_parser.add_argument("--top-k", type=int, default=5)
    generate_parser.add_argument("--tempo", type=int, default=120)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--c-major-only", action="store_true")
    generate_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    generate_parser.add_argument("--output", type=Path, default=Path("outputs/composition.mid"))

    analyze_parser = subparsers.add_parser("analyze", help="Display MIDI summary statistics.")
    analyze_parser.add_argument("midi_file", type=Path)

    report_parser = subparsers.add_parser("report", help="Regenerate sample analysis Markdown.")
    report_parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLE_DIR)
    report_parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    report_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    report_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "RESULTS.md")
    return parser


def add_asset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        print(json.dumps(analyze_midi(args.midi_file).to_dict(), indent=2))
        return
    if args.command == "report":
        path = generate_report(args.samples, args.vocabulary, args.checkpoint, args.output)
        print(f"Report written to {path}")
        return

    vocabulary = Vocabulary.load(args.vocabulary)
    model = load_model(args.checkpoint)
    if len(vocabulary.note_to_int) != model.config.vocab_size:
        raise ValueError("Vocabulary and checkpoint have different note dimensions.")

    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "parameters": parameter_count(model),
                    "vocabulary_tokens": len(vocabulary.note_to_int),
                    "duration_classes": len(vocabulary.duration_bins),
                    "sequence_length": vocabulary.sequence_length,
                    "training_sequences": vocabulary.training_sequences,
                },
                indent=2,
            )
        )
        return
    if args.command == "smoke-test":
        run_smoke_test(model, vocabulary)
        return
    if args.command == "generate":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        model.to(device)
        config = GenerationConfig(
            valence=args.valence,
            arousal=args.arousal,
            dominance=args.dominance,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            tempo_bpm=args.tempo,
            seed=args.seed,
            c_major_only=args.c_major_only,
        )
        events = generate_events(model, vocabulary, config, device=device)
        written = write_midi(events, args.output, tempo_bpm=config.tempo_bpm)
        print(json.dumps({"output": str(args.output), "events": written, "vad": config.vad}, indent=2))
        return
    raise ValueError(f"Unknown command: {args.command}")


def run_smoke_test(model, vocabulary: Vocabulary) -> None:
    sequence_length = 8
    notes = torch.zeros((1, sequence_length), dtype=torch.long)
    durations = torch.zeros((1, sequence_length), dtype=torch.long)
    vad = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(notes, durations, vad)
    payload = {
        "status": "ok",
        "note_logits": list(outputs[0].shape),
        "duration_logits": list(outputs[1].shape),
        "velocities": list(outputs[2].shape),
        "expected_vocabulary": len(vocabulary.note_to_int),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

