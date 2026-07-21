from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Vocabulary:
    note_to_int: dict[str, int]
    duration_bins: list[float]
    sequence_length: int
    max_velocity: float
    training_sequences: int

    @property
    def int_to_note(self) -> dict[int, str]:
        return {index: token for token, index in self.note_to_int.items()}

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocabulary = cls(
            note_to_int={str(key): int(value) for key, value in payload["note_to_int"].items()},
            duration_bins=[float(value) for value in payload["duration_bins"]],
            sequence_length=int(payload["sequence_length"]),
            max_velocity=float(payload["max_velocity"]),
            training_sequences=int(payload["training_sequences"]),
        )
        vocabulary.validate()
        return vocabulary

    def validate(self) -> None:
        indexes = sorted(self.note_to_int.values())
        if indexes != list(range(len(indexes))):
            raise ValueError("Vocabulary indexes must be contiguous and begin at zero.")
        if not self.duration_bins:
            raise ValueError("At least one duration class is required.")

