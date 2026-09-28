import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_composer.generation import GenerationConfig, is_c_major_token


class GenerationTests(unittest.TestCase):
    def test_vad_validation_rejects_out_of_range_value(self) -> None:
        config = GenerationConfig(valence=1.2, arousal=0.5, dominance=0.5)
        with self.assertRaises(ValueError):
            config.validate()

    def test_c_major_filter_supports_notes_and_chords(self) -> None:
        self.assertTrue(is_c_major_token("C4"))
        self.assertTrue(is_c_major_token("C4.E4.G4"))
        self.assertFalse(is_c_major_token("F#4"))

