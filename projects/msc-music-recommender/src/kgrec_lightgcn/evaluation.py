"""Top-k ranking evaluation for one held-out item per user."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def evaluate_topk(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    targets: dict[int, int],
    exclusions: list[set[int]],
    top_ks: list[int],
    batch_size: int,
) -> dict[str, float]:
    """Evaluate held-out targets after masking interactions already observed."""
    if not targets:
        raise ValueError("No evaluation targets were provided")
    if not top_ks or min(top_ks) <= 0:
        raise ValueError("top_ks must contain positive integers")
    if max(top_ks) > item_embeddings.shape[0]:
        raise ValueError("A requested k is greater than the number of items")

    users = sorted(targets)
    max_k = max(top_ks)
    hit_totals = {k: 0.0 for k in top_ks}
    ndcg_totals = {k: 0.0 for k in top_ks}
    reciprocal_rank_total = 0.0

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        user_tensor = torch.tensor(batch_users, device=user_embeddings.device)
        scores = user_embeddings[user_tensor] @ item_embeddings.T

        mask_rows: list[int] = []
        mask_items: list[int] = []
        for row, user in enumerate(batch_users):
            for item in exclusions[user]:
                if item != targets[user]:
                    mask_rows.append(row)
                    mask_items.append(item)
        if mask_rows:
            scores[
                torch.tensor(mask_rows, device=scores.device),
                torch.tensor(mask_items, device=scores.device),
            ] = -torch.inf

        recommendations = scores.topk(max_k, dim=1).indices
        target_tensor = torch.tensor(
            [targets[user] for user in batch_users], device=scores.device
        ).unsqueeze(1)
        matches = recommendations.eq(target_tensor)

        for row in range(len(batch_users)):
            match_positions = torch.nonzero(matches[row], as_tuple=False)
            if match_positions.numel() == 0:
                continue
            rank = int(match_positions[0, 0].item()) + 1
            reciprocal_rank_total += 1.0 / rank
            for k in top_ks:
                if rank <= k:
                    hit_totals[k] += 1.0
                    ndcg_totals[k] += 1.0 / math.log2(rank + 1)

    user_count = float(len(users))
    metrics: dict[str, float] = {"evaluated_users": user_count}
    for k in sorted(top_ks):
        # With one relevant held-out item, Recall@K and HitRate@K are identical.
        metrics[f"recall@{k}"] = hit_totals[k] / user_count
        metrics[f"precision@{k}"] = hit_totals[k] / (user_count * k)
        metrics[f"ndcg@{k}"] = ndcg_totals[k] / user_count
    metrics[f"mrr@{max_k}"] = reciprocal_rank_total / user_count
    return metrics
