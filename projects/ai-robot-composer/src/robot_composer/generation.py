from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from music21 import chord, instrument, note, pitch, stream, tempo

from robot_composer.model import HierarchicalVAE
from robot_composer.vocabulary import Vocabulary


@dataclass(frozen=True)
class GenerationConfig:
    valence: float
    arousal: float
    dominance: float
    length: int = 128
    temperature: float = 0.6
    top_k: int = 5
    tempo_bpm: int = 120
    seed: int = 42
    c_major_only: bool = False

    @property
    def vad(self) -> tuple[float, float, float]:
        return self.valence, self.arousal, self.dominance

    def validate(self) -> None:
        if any(value < 0.0 or value > 1.0 for value in self.vad):
            raise ValueError("Valence, arousal, and dominance must each be between 0 and 1.")
        if self.length < 1:
            raise ValueError("Generation length must be positive.")
        if self.temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")
        if self.top_k < 1:
            raise ValueError("top_k must be positive.")
        if self.tempo_bpm < 20 or self.tempo_bpm > 300:
            raise ValueError("Tempo must be between 20 and 300 BPM.")


@dataclass(frozen=True)
class GeneratedEvent:
    token: str
    duration: float
    velocity: int


def generate_events(
    model: HierarchicalVAE,
    vocabulary: Vocabulary,
    config: GenerationConfig,
    device: torch.device | str = "cpu",
) -> list[GeneratedEvent]:
    config.validate()
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=str(device)).manual_seed(config.seed)
    int_to_note = vocabulary.int_to_note

    valid_indexes = [
        index
        for index, token in int_to_note.items()
        if not config.c_major_only or is_c_major_token(token)
    ]
    if not valid_indexes:
        raise ValueError("No vocabulary entries satisfy the requested pitch constraint.")

    seed_indexes = torch.tensor(valid_indexes, dtype=torch.long)
    chosen = torch.randint(
        low=0,
        high=len(seed_indexes),
        size=(1, vocabulary.sequence_length),
        generator=generator,
    )
    notes = seed_indexes[chosen].to(device)
    durations = torch.zeros_like(notes, dtype=torch.long, device=device)
    vad = torch.tensor([config.vad], dtype=torch.float32, device=device)
    allowed_mask = make_allowed_mask(valid_indexes, len(vocabulary.note_to_int), device)
    events: list[GeneratedEvent] = []

    model.eval()
    for _ in range(config.length):
        with torch.no_grad():
            note_logits, duration_logits, velocities, *_ = model(notes, durations, vad)

        next_note = sample_index(
            note_logits[:, -1, :],
            temperature=config.temperature,
            top_k=config.top_k,
            generator=generator,
            allowed_mask=allowed_mask,
        )
        next_duration = sample_index(
            duration_logits[:, -1, :],
            temperature=config.temperature,
            top_k=len(vocabulary.duration_bins),
            generator=generator,
        )
        velocity = int(round(float(velocities[0, -1, 0]) * 87 + 40))
        velocity = max(1, min(127, velocity))
        events.append(
            GeneratedEvent(
                token=int_to_note[next_note],
                duration=vocabulary.duration_bins[next_duration],
                velocity=velocity,
            )
        )

        notes = torch.cat(
            [notes[:, 1:], torch.tensor([[next_note]], dtype=torch.long, device=device)],
            dim=1,
        )
        durations = torch.cat(
            [
                durations[:, 1:],
                torch.tensor([[next_duration]], dtype=torch.long, device=device),
            ],
            dim=1,
        )

    return events


def sample_index(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
    allowed_mask: torch.Tensor | None = None,
) -> int:
    scaled = logits / temperature
    if allowed_mask is not None:
        scaled = scaled.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
    k = min(top_k, scaled.shape[-1])
    top_values, top_indexes = torch.topk(scaled, k=k, dim=-1)
    probabilities = torch.softmax(top_values, dim=-1)
    sampled_position = torch.multinomial(probabilities[0], 1, generator=generator).item()
    return int(top_indexes[0, sampled_position].item())


def make_allowed_mask(indexes: list[int], size: int, device: torch.device | str) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[indexes] = True
    return mask


def is_c_major_token(token: str) -> bool:
    try:
        return all(pitch.Pitch(part).pitchClass in {0, 2, 4, 5, 7, 9, 11} for part in token.split("."))
    except Exception:
        return False


def write_midi(events: list[GeneratedEvent], output_path: Path, tempo_bpm: int) -> int:
    composition = stream.Stream()
    composition.append(tempo.MetronomeMark(number=tempo_bpm))
    composition.append(instrument.Piano())
    written = 0

    for event in events:
        try:
            if "." in event.token:
                musical_event = chord.Chord(event.token.split("."))
            else:
                musical_event = note.Note(event.token)
        except Exception:
            continue

        musical_event.quarterLength = event.duration
        musical_event.volume.velocity = event.velocity
        composition.append(musical_event)
        written += 1

    if written == 0:
        raise ValueError("The generated sequence did not contain any writable MIDI events.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composition.write("midi", fp=str(output_path))
    return written

