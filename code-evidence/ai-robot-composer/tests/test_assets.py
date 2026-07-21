import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_composer.analysis import analyze_midi
from robot_composer.vocabulary import Vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AssetTests(unittest.TestCase):
    def test_extracted_vocabulary_matches_checkpoint_dimensions(self) -> None:
        vocabulary = Vocabulary.load(PROJECT_ROOT / "artifacts" / "vocabulary.json")
        self.assertEqual(len(vocabulary.note_to_int), 4343)
        self.assertEqual(len(vocabulary.duration_bins), 11)
        self.assertEqual(vocabulary.sequence_length, 64)

    def test_included_midi_sample_is_parseable(self) -> None:
        statistics = analyze_midi(PROJECT_ROOT / "samples" / "generated_V1.0_A1.0_D1.0.mid")
        self.assertGreater(statistics.musical_events, 0)
        self.assertGreater(statistics.pitches, 0)

