from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4343
    duration_bins: int = 11
    vad_dim: int = 3
    embed_dim: int = 128
    st_hidden_dim: int = 256
    lt_hidden_dim: int = 512
    latent_dim: int = 128
    num_layers: int = 2
    lt_num_layers: int = 3
    dropout: float = 0.3


class HierarchicalVAE(nn.Module):
    """Hierarchical VAE used by the final-year emotion-conditioned composer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.duration_embedding = nn.Embedding(config.duration_bins, config.embed_dim)
        self.st_rnn = nn.LSTM(
            config.embed_dim * 2,
            config.st_hidden_dim,
            config.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lt_rnn = nn.LSTM(
            config.st_hidden_dim * 2,
            config.lt_hidden_dim,
            num_layers=config.lt_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lt_num_layers > 1 else 0.0,
        )

        self.vad_fc = nn.Linear(config.vad_dim, config.lt_hidden_dim)
        self.fc_mu_st = nn.Linear(config.st_hidden_dim * 2, config.latent_dim)
        self.fc_logvar_st = nn.Linear(config.st_hidden_dim * 2, config.latent_dim)
        self.fc_mu_lt = nn.Linear(config.lt_hidden_dim, config.latent_dim)
        self.fc_logvar_lt = nn.Linear(config.lt_hidden_dim, config.latent_dim)

        self.fc_decode_st = nn.Linear(config.latent_dim, config.st_hidden_dim)
        self.fc_decode_lt = nn.Linear(config.latent_dim, config.lt_hidden_dim)
        self.decoder_lstm = nn.LSTM(
            config.embed_dim * 2 + config.lt_hidden_dim,
            config.st_hidden_dim,
            config.num_layers,
            batch_first=True,
        )

        self.output_fc_notes = nn.Linear(config.st_hidden_dim, config.vocab_size)
        self.output_fc_durations = nn.Linear(config.st_hidden_dim, config.duration_bins)
        self.output_fc_velocities = nn.Linear(config.st_hidden_dim, 1)

    def encode(
        self,
        notes: torch.Tensor,
        durations: torch.Tensor,
        vad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        durations = align_durations(durations, notes.shape[1]).long()
        note_embeddings = self.embedding(notes)
        duration_embeddings = self.duration_embedding(durations)
        combined = torch.cat([note_embeddings, duration_embeddings], dim=-1)

        short_output, (short_hidden, _) = self.st_rnn(combined)
        short_hidden = torch.cat([short_hidden[-2], short_hidden[-1]], dim=-1)
        mu_st = self.fc_mu_st(short_hidden)
        logvar_st = self.fc_logvar_st(short_hidden)

        _, (long_hidden, _) = self.lt_rnn(short_output)
        conditioned_long_hidden = long_hidden[-1] + self.vad_fc(vad)
        mu_lt = self.fc_mu_lt(conditioned_long_hidden)
        logvar_lt = self.fc_logvar_lt(conditioned_long_hidden)
        return mu_st, logvar_st, mu_lt, logvar_lt

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(
        self,
        z_st: torch.Tensor,
        z_lt: torch.Tensor,
        notes: torch.Tensor,
        durations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        durations = align_durations(durations, notes.shape[1]).long()
        short_initial = self.fc_decode_st(z_st).unsqueeze(0).repeat(
            self.config.num_layers,
            1,
            1,
        )
        long_context = self.fc_decode_lt(z_lt).unsqueeze(1).expand(
            -1,
            notes.shape[1],
            -1,
        )

        combined = torch.cat(
            [self.embedding(notes), self.duration_embedding(durations), long_context],
            dim=-1,
        )
        cell_initial = torch.zeros_like(short_initial)
        decoded, _ = self.decoder_lstm(combined, (short_initial, cell_initial))

        note_logits = self.output_fc_notes(decoded)
        duration_logits = self.output_fc_durations(decoded)
        velocities = torch.sigmoid(self.output_fc_velocities(decoded))
        return note_logits, duration_logits, velocities

    def forward(
        self,
        notes: torch.Tensor,
        durations: torch.Tensor,
        vad: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mu_st, logvar_st, mu_lt, logvar_lt = self.encode(notes, durations, vad)
        z_st = self.reparameterize(mu_st, logvar_st)
        z_lt = self.reparameterize(mu_lt, logvar_lt)
        outputs = self.decode(z_st, z_lt, notes, durations)
        return (*outputs, mu_st, logvar_st, mu_lt, logvar_lt)


def align_durations(durations: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if durations.ndim == 1:
        return durations.unsqueeze(1).repeat(1, sequence_length)
    if durations.shape[1] < sequence_length:
        padding = durations[:, -1:].repeat(1, sequence_length - durations.shape[1])
        return torch.cat([durations, padding], dim=1)
    return durations[:, :sequence_length]


def load_model(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
    config: ModelConfig | None = None,
) -> HierarchicalVAE:
    resolved_config = config or ModelConfig()
    model = HierarchicalVAE(resolved_config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

