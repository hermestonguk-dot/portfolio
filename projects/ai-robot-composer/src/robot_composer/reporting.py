from __future__ import annotations

import re
from pathlib import Path
from statistics import mean

import torch

from robot_composer.analysis import MidiStatistics, analyze_midi
from robot_composer.vocabulary import Vocabulary


VAD_PATTERN = re.compile(r"_V(?P<v>[0-9.]+)_A(?P<a>[0-9.]+)_D(?P<d>[0-9.]+)\.mid$")


def generate_report(
    sample_dir: Path,
    vocabulary_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    samples = [analyze_midi(path) for path in sorted(sample_dir.glob("*.mid"))]
    if not samples:
        raise FileNotFoundError(f"No MIDI samples found in {sample_dir}.")
    vocabulary = Vocabulary.load(vocabulary_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    parameters = sum(tensor.numel() for tensor in state_dict.values())
    checkpoint_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    output_path.write_text(
        render_report(samples, vocabulary, parameters, checkpoint_mb),
        encoding="utf-8",
    )
    return output_path


def render_report(
    samples: list[MidiStatistics],
    vocabulary: Vocabulary,
    parameters: int,
    checkpoint_mb: float,
) -> str:
    lines = [
        "# Robot Composer Results",
        "",
        "This report combines the evaluation recorded in the final notebook with a fresh, "
        "reproducible analysis of the representative generated MIDI files shipped here.",
        "",
        "## Model Artefact",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Trainable parameters | {parameters:,} |",
        f"| Checkpoint size | {checkpoint_mb:.1f} MB |",
        f"| Note/chord vocabulary | {len(vocabulary.note_to_int):,} tokens |",
        f"| Duration vocabulary | {len(vocabulary.duration_bins)} classes |",
        f"| Training sequence length | {vocabulary.sequence_length} events |",
        f"| Cached training sequences | {vocabulary.training_sequences:,} |",
        "",
        "## Original Notebook Evaluation",
        "",
        "| Metric | Recorded value | Interpretation |",
        "|---|---:|---|",
        "| Note perplexity | 1.7574 | Lower values indicate more confident next-token prediction |",
        "| Short-term KL divergence | 11.1348 | Local latent-space regularization |",
        "| Long-term KL divergence | 2.2843 | Phrase-level latent-space regularization |",
        "| Example average pitch | 59.50 | MIDI pitch number |",
        "| Example average duration | 0.667 | Quarter lengths |",
        "| Example average velocity | 84.48 | MIDI velocity |",
        "",
        "The associated CV reports an approximately 18% improvement in perceived dominance "
        "expression over baseline outputs. The available export does not contain the raw human-study "
        "responses, so that result has not been independently recalculated in this repository.",
        "",
        "## Included Sample Analysis",
        "",
        "| Sample V/A/D | Events | Chords | Avg pitch | Pitch spread | Avg duration | Avg velocity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in samples:
        lines.append(sample_row(sample))

    lines.extend(
        [
            "",
            "Across the included samples:",
            "",
            f"- Mean event count: **{mean(sample.musical_events for sample in samples):.1f}**",
            f"- Mean pitch: **{mean(sample.average_pitch for sample in samples):.2f}**",
            f"- Mean duration: **{mean(sample.average_duration for sample in samples):.3f}**",
            f"- Mean velocity: **{mean(sample.average_velocity for sample in samples):.2f}**",
            "",
            "These descriptive statistics confirm that each checked-in artefact is parseable MIDI "
            "and make its pitch, rhythm, and dynamics inspectable. They are not evidence that the "
            "conditioning dimensions were perceived correctly; that requires a controlled listener study.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "robot-composer smoke-test",
            "robot-composer report --output RESULTS.md",
            "python -m unittest discover -s tests",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def sample_row(sample: MidiStatistics) -> str:
    match = VAD_PATTERN.search(sample.file)
    label = sample.file
    if match:
        label = f"{match.group('v')} / {match.group('a')} / {match.group('d')}"
    return (
        f"| [{label}](samples/{sample.file}) | {sample.musical_events} | "
        f"{sample.chord_events} | {sample.average_pitch:.2f} | {sample.pitch_std:.2f} | "
        f"{sample.average_duration:.3f} | {sample.average_velocity:.2f} |"
    )

