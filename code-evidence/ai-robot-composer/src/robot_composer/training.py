from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    notes: torch.Tensor
    durations: torch.Tensor
    velocities: torch.Tensor
    kl_short: torch.Tensor
    kl_long: torch.Tensor


def hierarchical_vae_loss(
    note_logits: torch.Tensor,
    target_notes: torch.Tensor,
    duration_logits: torch.Tensor,
    target_durations: torch.Tensor,
    output_velocities: torch.Tensor,
    target_velocities: torch.Tensor,
    mu_st: torch.Tensor,
    logvar_st: torch.Tensor,
    mu_lt: torch.Tensor,
    logvar_lt: torch.Tensor,
    beta_short: float = 0.5,
    beta_long: float = 2.0,
) -> LossBreakdown:
    note_loss = 1.3 * functional.cross_entropy(
        note_logits.reshape(-1, note_logits.shape[-1]),
        target_notes.reshape(-1),
    )
    duration_loss = 0.8 * functional.cross_entropy(
        duration_logits.reshape(-1, duration_logits.shape[-1]),
        target_durations.reshape(-1),
    )
    velocity_loss = functional.mse_loss(
        output_velocities.reshape(-1),
        target_velocities.reshape(-1),
    )
    kl_short = -0.5 * torch.mean(1 + logvar_st - mu_st.pow(2) - logvar_st.exp())
    kl_long = -0.5 * torch.mean(1 + logvar_lt - mu_lt.pow(2) - logvar_lt.exp())
    adaptive_beta_long = beta_long * (1 + (0.5 - kl_long).clamp(min=0))
    total = (
        note_loss
        + duration_loss
        + velocity_loss
        + beta_short * kl_short
        + adaptive_beta_long * kl_long
    )
    return LossBreakdown(total, note_loss, duration_loss, velocity_loss, kl_short, kl_long)


def train_epoch(
    model: nn.Module,
    batches: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    beta_short: float = 0.5,
    beta_long: float = 2.0,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in LossBreakdown.__dataclass_fields__}
    batch_count = 0

    for batch in batches:
        input_notes, target_notes, target_durations, vad, durations, velocities, _ = batch
        input_notes = input_notes.to(device)
        target_notes = target_notes.to(device)
        target_durations = target_durations.to(device)
        vad = vad.to(device)
        durations = durations.to(device)
        velocities = velocities.to(device)

        outputs = model(input_notes, durations, vad)
        losses = hierarchical_vae_loss(
            outputs[0],
            target_notes,
            outputs[1],
            target_durations,
            outputs[2],
            velocities,
            *outputs[3:],
            beta_short=beta_short,
            beta_long=beta_long,
        )
        optimizer.zero_grad()
        losses.total.backward()
        optimizer.step()

        for name in totals:
            totals[name] += float(getattr(losses, name).detach())
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Training requires at least one batch.")
    return {name: value / batch_count for name, value in totals.items()}

