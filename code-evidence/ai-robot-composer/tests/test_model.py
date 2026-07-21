import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_composer.model import HierarchicalVAE, ModelConfig, align_durations


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        config = ModelConfig(
            vocab_size=17,
            duration_bins=5,
            embed_dim=8,
            st_hidden_dim=12,
            lt_hidden_dim=16,
            latent_dim=6,
            num_layers=2,
            lt_num_layers=2,
        )
        model = HierarchicalVAE(config)
        notes = torch.randint(0, config.vocab_size, (2, 7))
        durations = torch.randint(0, config.duration_bins, (2, 7))
        vad = torch.rand((2, 3))

        outputs = model(notes, durations, vad)

        self.assertEqual(outputs[0].shape, (2, 7, config.vocab_size))
        self.assertEqual(outputs[1].shape, (2, 7, config.duration_bins))
        self.assertEqual(outputs[2].shape, (2, 7, 1))
        self.assertEqual(outputs[3].shape, (2, config.latent_dim))
        self.assertEqual(outputs[5].shape, (2, config.latent_dim))

    def test_duration_alignment(self) -> None:
        durations = torch.tensor([[1, 2]])
        aligned = align_durations(durations, sequence_length=4)
        self.assertEqual(aligned.tolist(), [[1, 2, 2, 2]])

