from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

from music21 import chord, converter, note


@dataclass(frozen=True)
class MidiStatistics:
    file: str
    musical_events: int
    pitches: int
    chord_events: int
    average_pitch: float
    pitch_std: float
    pitch_min: int
    pitch_max: int
    average_duration: float
    average_velocity: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_midi(path: Path) -> MidiStatistics:
    score = converter.parse(str(path))
    pitches: list[int] = []
    durations: list[float] = []
    velocities: list[int] = []
    chord_events = 0

    for element in score.flatten().notes:
        durations.append(float(element.quarterLength))
        velocities.append(int(element.volume.velocity or 64))
        if isinstance(element, note.Note):
            pitches.append(int(element.pitch.midi))
        elif isinstance(element, chord.Chord):
            chord_events += 1
            pitches.extend(int(value.midi) for value in element.pitches)

    if not pitches or not durations:
        raise ValueError(f"No note or chord events found in {path}.")

    return MidiStatistics(
        file=path.name,
        musical_events=len(durations),
        pitches=len(pitches),
        chord_events=chord_events,
        average_pitch=round(mean(pitches), 3),
        pitch_std=round(pstdev(pitches), 3),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        average_duration=round(mean(durations), 3),
        average_velocity=round(mean(velocities), 3),
    )

