"""Post-training popularity-bias reranking and exposure metrics.

The functions in this module keep the trained recommender unchanged. They
construct one candidate pool from its scores and can then apply either the
project-designed Global Popularity Discount or the paper-backed Smooth xQuAD
reranker of Abdollahpouri, Burke, and Mobasher (2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .data import KGRecData


@dataclass
class CandidatePool:
    """The same relevance-ranked candidates reused by every reranking policy."""

    users: list[int]
    item_ids: torch.Tensor
    relevance_scores: torch.Tensor

    def __post_init__(self) -> None:
        if self.item_ids.ndim != 2 or self.relevance_scores.ndim != 2:
            raise ValueError("candidate tensors must be two-dimensional")
        if self.item_ids.shape != self.relevance_scores.shape:
            raise ValueError("candidate item IDs and scores must have the same shape")
        if self.item_ids.shape[0] != len(self.users):
            raise ValueError("candidate tensors must contain one row per user")
        if len(set(self.users)) != len(self.users):
            raise ValueError("candidate users must be unique")


def train_item_popularity(data: KGRecData) -> torch.Tensor:
    """Count item popularity using training interactions only."""
    return torch.bincount(data.train_items, minlength=data.num_items).float()


def gini_index(values: torch.Tensor) -> float:
    """Return the Gini inequality of a non-negative vector."""
    if values.numel() == 0:
        raise ValueError("values must not be empty")
    values = values.detach().float().cpu()
    if torch.any(values < 0):
        raise ValueError("gini_index expects non-negative values")
    total = float(values.sum().item())
    if total == 0.0:
        return 0.0

    sorted_values = values.sort().values
    count = sorted_values.numel()
    index = torch.arange(1, count + 1, dtype=torch.float32)
    weighted_sum = float((index * sorted_values).sum().item())
    return (2.0 * weighted_sum) / (count * total) - (count + 1.0) / count


def _popularity_penalty(popularity: torch.Tensor, device: torch.device) -> torch.Tensor:
    penalty = torch.log1p(popularity.float()).to(device)
    max_penalty = penalty.max()
    if float(max_penalty.item()) == 0.0:
        return penalty
    return penalty / max_penalty


@torch.no_grad()
def build_candidate_pool(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: list[int],
    exclusions: list[set[int]],
    batch_size: int,
    candidate_pool_size: int,
) -> CandidatePool:
    """Score, mask, and retain the same top candidates for every policy."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")
    if len(exclusions) != user_embeddings.shape[0]:
        raise ValueError("exclusions must contain one set per user")
    if len(set(users)) != len(users):
        raise ValueError("users must not contain duplicates")
    if any(user < 0 or user >= user_embeddings.shape[0] for user in users):
        raise ValueError("users contains an out-of-range user ID")

    item_count = item_embeddings.shape[0]
    pool_size = min(candidate_pool_size, item_count)
    all_items: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        user_tensor = torch.tensor(batch_users, device=user_embeddings.device)
        scores = user_embeddings[user_tensor] @ item_embeddings.T

        mask_rows: list[int] = []
        mask_items: list[int] = []
        for row, user in enumerate(batch_users):
            for item in exclusions[user]:
                mask_rows.append(row)
                mask_items.append(item)
        if mask_rows:
            scores[
                torch.tensor(mask_rows, device=scores.device),
                torch.tensor(mask_items, device=scores.device),
            ] = -torch.inf

        candidate_scores, candidate_items = scores.topk(pool_size, dim=1)
        if torch.isneginf(candidate_scores).any():
            raise ValueError(
                "candidate_pool_size exceeds the eligible item count for a user"
            )
        all_items.append(candidate_items.cpu())
        all_scores.append(candidate_scores.cpu())

    if not all_items:
        return CandidatePool(
            users=[],
            item_ids=torch.empty((0, pool_size), dtype=torch.long),
            relevance_scores=torch.empty((0, pool_size), dtype=torch.float32),
        )
    return CandidatePool(
        users=list(users),
        item_ids=torch.cat(all_items),
        relevance_scores=torch.cat(all_scores),
    )


def _validate_candidate_top_k(candidate_pool: CandidatePool, top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > candidate_pool.item_ids.shape[1]:
        raise ValueError("top_k must not exceed the candidate pool size")


def _recommendations_from_items(
    users: list[int], selected_items: torch.Tensor
) -> dict[int, list[int]]:
    return {
        user: items
        for user, items in zip(users, selected_items.detach().cpu().tolist())
    }


def recommendations_from_candidate_pool(
    candidate_pool: CandidatePool, top_k: int
) -> dict[int, list[int]]:
    """Return the original base-recommender order without reranking."""
    _validate_candidate_top_k(candidate_pool, top_k)
    return _recommendations_from_items(
        candidate_pool.users, candidate_pool.item_ids[:, :top_k]
    )


def rerank_candidate_pool_with_popularity_penalty(
    candidate_pool: CandidatePool,
    popularity: torch.Tensor,
    top_k: int,
    fairness_weight: float,
) -> dict[int, list[int]]:
    """Apply the project-designed Global Popularity Discount to candidates."""
    _validate_candidate_top_k(candidate_pool, top_k)
    if fairness_weight < 0.0:
        raise ValueError("fairness_weight must be non-negative")
    if popularity.ndim != 1:
        raise ValueError("popularity must be one-dimensional")
    if candidate_pool.item_ids.numel() and (
        int(candidate_pool.item_ids.min().item()) < 0
        or int(candidate_pool.item_ids.max().item()) >= popularity.numel()
    ):
        raise ValueError("candidate pool contains an out-of-range item ID")
    if fairness_weight == 0.0:
        return recommendations_from_candidate_pool(candidate_pool, top_k)

    device = candidate_pool.relevance_scores.device
    scores = candidate_pool.relevance_scores
    minimum = scores.min(dim=1, keepdim=True).values
    maximum = scores.max(dim=1, keepdim=True).values
    relevance = (scores - minimum) / (maximum - minimum).clamp_min(1e-12)
    penalty = _popularity_penalty(popularity, device)
    rerank_scores = (
        relevance
        - fairness_weight * penalty[candidate_pool.item_ids.to(device)]
    )
    selected_positions = rerank_scores.topk(top_k, dim=1).indices
    selected_items = candidate_pool.item_ids.to(device).gather(
        1, selected_positions
    )
    return _recommendations_from_items(candidate_pool.users, selected_items)


def popularity_head_mask_by_interaction_share(
    popularity: torch.Tensor, head_interaction_share: float = 0.8
) -> torch.Tensor:
    """Identify the popularity head that accounts for a target event share.

    Abdollahpouri, Burke, and Mobasher (2019) define the short head as the
    popular items collectively responsible for 80% of training ratings. Items
    tied at the boundary are kept in the same category, so the achieved share
    can be slightly larger than the requested value.
    """
    if popularity.ndim != 1 or popularity.numel() == 0:
        raise ValueError("popularity must be a non-empty one-dimensional tensor")
    if torch.any(popularity < 0):
        raise ValueError("popularity must be non-negative")
    if not 0.0 < head_interaction_share < 1.0:
        raise ValueError("head_interaction_share must be between 0 and 1")

    popularity = popularity.detach().float().cpu()
    total = popularity.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("popularity must contain at least one interaction")

    sorted_popularity = popularity.sort(descending=True).values
    boundary = torch.searchsorted(
        sorted_popularity.cumsum(dim=0),
        total * head_interaction_share,
        right=False,
    )
    cutoff = sorted_popularity[min(int(boundary.item()), popularity.numel() - 1)]
    return popularity >= cutoff


def user_head_tail_preferences(
    train_seen: list[set[int]], head_mask: torch.Tensor
) -> torch.Tensor:
    """Estimate each user's tail/head preference from training history only.

    Column 0 is P(tail | user) and column 1 is P(head | user).
    """
    if head_mask.ndim != 1 or head_mask.dtype != torch.bool:
        raise ValueError("head_mask must be a one-dimensional boolean tensor")

    head_mask = head_mask.detach().cpu()
    preferences = torch.empty((len(train_seen), 2), dtype=torch.float32)
    for user, profile in enumerate(train_seen):
        if not profile:
            raise ValueError(f"user {user} has an empty training profile")
        if min(profile) < 0 or max(profile) >= head_mask.numel():
            raise ValueError(f"user {user} has an out-of-range training item")
        profile_items = torch.tensor(sorted(profile), dtype=torch.long)
        head_count = int(head_mask[profile_items].sum().item())
        profile_size = len(profile)
        preferences[user, 1] = head_count / profile_size
        preferences[user, 0] = (profile_size - head_count) / profile_size
    return preferences


def rerank_candidate_pool_with_smooth_xquad(
    candidate_pool: CandidatePool,
    head_mask: torch.Tensor,
    user_preferences: torch.Tensor,
    top_k: int,
    lambda_weight: float,
) -> dict[int, list[int]]:
    """Iteratively rerank candidates using popularity-aware Smooth xQuAD.

    This implements Equation 4 of Abdollahpouri, Burke, and Mobasher (2019):

    score(v) = (1-lambda) P(v|u)
               + lambda * sum_c P(c|u) P(v|c)
                          product_{i in S}(1-P(i|c,S))

    The two categories are long tail and short head. ``P(v|c)`` is a category
    indicator, ``P(c|u)`` is the category share in the user's training profile,
    and Smooth xQuAD defines category coverage from the proportion of already
    selected items in that category. LightGCN dot products are min-max
    normalized inside each candidate pool to operationalize ``P(v|u)`` on the
    same [0, 1] scale as the diversification term.
    """
    _validate_candidate_top_k(candidate_pool, top_k)
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError("lambda_weight must be between 0 and 1")
    if head_mask.ndim != 1 or head_mask.dtype != torch.bool:
        raise ValueError("head_mask must be a one-dimensional boolean tensor")
    if user_preferences.ndim != 2 or user_preferences.shape[1] != 2:
        raise ValueError("user_preferences must have shape [num_users, 2]")
    if candidate_pool.users and max(candidate_pool.users) >= user_preferences.shape[0]:
        raise ValueError("user_preferences does not contain every candidate user")
    if candidate_pool.item_ids.numel() and (
        int(candidate_pool.item_ids.min().item()) < 0
        or int(candidate_pool.item_ids.max().item()) >= head_mask.numel()
    ):
        raise ValueError("candidate pool contains an out-of-range item ID")
    if lambda_weight == 0.0:
        return recommendations_from_candidate_pool(candidate_pool, top_k)

    device = candidate_pool.relevance_scores.device
    item_ids = candidate_pool.item_ids.to(device)
    scores = candidate_pool.relevance_scores
    minimum = scores.min(dim=1, keepdim=True).values
    maximum = scores.max(dim=1, keepdim=True).values
    relevance = (scores - minimum) / (maximum - minimum).clamp_min(1e-12)

    # Category 0 is tail and category 1 is head.
    category_ids = head_mask.to(device)[item_ids].long()
    user_ids = torch.tensor(candidate_pool.users, dtype=torch.long, device=device)
    preferences = user_preferences.to(device)[user_ids]
    selected_mask = torch.zeros_like(item_ids, dtype=torch.bool)
    selected_counts = torch.zeros(
        (len(candidate_pool.users), 2), dtype=torch.float32, device=device
    )
    row_indices = torch.arange(len(candidate_pool.users), device=device)
    selected_positions: list[torch.Tensor] = []

    for position in range(top_k):
        if position == 0:
            novelty = torch.ones_like(selected_counts)
        else:
            coverage = selected_counts / float(position)
            # For selected items in category c, Smooth xQuAD assigns the
            # current category ratio; other selected items contribute zero.
            novelty = (1.0 - coverage).clamp(min=0.0).pow(selected_counts)

        category_bonus = preferences * novelty
        diversity = category_bonus.gather(1, category_ids)
        rerank_scores = (1.0 - lambda_weight) * relevance + lambda_weight * diversity
        rerank_scores = rerank_scores.masked_fill(selected_mask, -torch.inf)

        chosen_positions = rerank_scores.argmax(dim=1)
        selected_positions.append(chosen_positions)
        selected_mask[row_indices, chosen_positions] = True
        chosen_categories = category_ids[row_indices, chosen_positions]
        selected_counts.scatter_add_(
            1,
            chosen_categories.unsqueeze(1),
            torch.ones((len(candidate_pool.users), 1), device=device),
        )

    position_matrix = torch.stack(selected_positions, dim=1)
    selected_items = item_ids.gather(1, position_matrix)
    return _recommendations_from_items(candidate_pool.users, selected_items)


@torch.no_grad()
def rerank_with_popularity_penalty(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: list[int],
    exclusions: list[set[int]],
    popularity: torch.Tensor,
    top_k: int,
    batch_size: int,
    fairness_weight: float,
    candidate_pool_size: int,
) -> dict[int, list[int]]:
    """Build candidates and apply the Global Popularity Discount."""
    if candidate_pool_size < top_k:
        raise ValueError("candidate_pool_size must be at least top_k")
    if popularity.numel() != item_embeddings.shape[0]:
        raise ValueError("popularity must contain one value per item")
    candidate_pool = build_candidate_pool(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        users=users,
        exclusions=exclusions,
        batch_size=batch_size,
        candidate_pool_size=candidate_pool_size,
    )
    return rerank_candidate_pool_with_popularity_penalty(
        candidate_pool=candidate_pool,
        popularity=popularity,
        top_k=top_k,
        fairness_weight=fairness_weight,
    )


@torch.no_grad()
def rerank_with_smooth_xquad(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: list[int],
    exclusions: list[set[int]],
    train_seen: list[set[int]],
    popularity: torch.Tensor,
    top_k: int,
    batch_size: int,
    lambda_weight: float,
    candidate_pool_size: int,
    head_interaction_share: float = 0.8,
) -> dict[int, list[int]]:
    """Build candidates and apply paper-aligned Smooth xQuAD."""
    if candidate_pool_size < top_k:
        raise ValueError("candidate_pool_size must be at least top_k")
    if popularity.numel() != item_embeddings.shape[0]:
        raise ValueError("popularity must contain one value per item")
    if len(train_seen) != user_embeddings.shape[0]:
        raise ValueError("train_seen must contain one profile per user")

    candidate_pool = build_candidate_pool(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        users=users,
        exclusions=exclusions,
        batch_size=batch_size,
        candidate_pool_size=candidate_pool_size,
    )
    head_mask = popularity_head_mask_by_interaction_share(
        popularity, head_interaction_share
    )
    preferences = user_head_tail_preferences(train_seen, head_mask)
    return rerank_candidate_pool_with_smooth_xquad(
        candidate_pool=candidate_pool,
        head_mask=head_mask,
        user_preferences=preferences,
        top_k=top_k,
        lambda_weight=lambda_weight,
    )


def evaluate_recommendation_lists(
    recommendations: dict[int, list[int]],
    targets: dict[int, int],
    top_ks: list[int],
) -> dict[str, float]:
    """Evaluate precomputed recommendation lists against one target per user."""
    if not targets:
        raise ValueError("No evaluation targets were provided")
    if not top_ks or min(top_ks) <= 0:
        raise ValueError("top_ks must contain positive integers")
    max_k = max(top_ks)

    hit_totals = {k: 0.0 for k in top_ks}
    ndcg_totals = {k: 0.0 for k in top_ks}
    reciprocal_rank_total = 0.0

    for user, target in targets.items():
        if user not in recommendations:
            raise ValueError(f"Missing recommendations for user {user}")
        ranked_items = recommendations[user]
        if len(ranked_items) < max_k:
            raise ValueError("Each recommendation list must contain at least max(top_ks)")
        try:
            rank = ranked_items[:max_k].index(target) + 1
        except ValueError:
            continue

        reciprocal_rank_total += 1.0 / rank
        for k in top_ks:
            if rank <= k:
                hit_totals[k] += 1.0
                ndcg_totals[k] += 1.0 / math.log2(rank + 1)

    user_count = float(len(targets))
    metrics: dict[str, float] = {"evaluated_users": user_count}
    for k in sorted(top_ks):
        metrics[f"recall@{k}"] = hit_totals[k] / user_count
        metrics[f"precision@{k}"] = hit_totals[k] / (user_count * k)
        metrics[f"ndcg@{k}"] = ndcg_totals[k] / user_count
    metrics[f"mrr@{max_k}"] = reciprocal_rank_total / user_count
    return metrics


def _head_item_mask(popularity: torch.Tensor, head_fraction: float) -> torch.Tensor:
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must be between 0 and 1")
    item_count = popularity.numel()
    head_count = max(1, int(math.ceil(item_count * head_fraction)))
    head_indices = torch.topk(popularity.float(), head_count).indices
    mask = torch.zeros(item_count, dtype=torch.bool)
    mask[head_indices.cpu()] = True
    return mask


def exposure_metrics(
    recommendations: dict[int, list[int]],
    num_items: int,
    popularity: torch.Tensor,
    top_ks: list[int],
    head_fraction: float,
    head_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure catalogue exposure concentration for recommendation lists."""
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if popularity.numel() != num_items:
        raise ValueError("popularity must contain one value per item")
    if not recommendations:
        raise ValueError("recommendations must not be empty")
    if not top_ks or min(top_ks) <= 0:
        raise ValueError("top_ks must contain positive integers")

    max_k = max(top_ks)
    for items in recommendations.values():
        if len(items) < max_k:
            raise ValueError("Each recommendation list must contain at least max(top_ks)")

    popularity = popularity.detach().float().cpu()
    if head_mask is None:
        head_mask = _head_item_mask(popularity, head_fraction)
    else:
        if head_mask.ndim != 1 or head_mask.dtype != torch.bool:
            raise ValueError("head_mask must be a one-dimensional boolean tensor")
        if head_mask.numel() != num_items:
            raise ValueError("head_mask must contain one value per item")
        head_mask = head_mask.detach().cpu()
        if not torch.any(head_mask) or torch.all(head_mask):
            raise ValueError("head_mask must contain both head and tail items")
    tail_mask = ~head_mask
    metrics: dict[str, float] = {
        "head_fraction": float(head_mask.float().mean().item()),
        "head_item_count": float(head_mask.sum().item()),
        "tail_item_count": float(tail_mask.sum().item()),
    }

    for k in sorted(top_ks):
        counts = torch.zeros(num_items, dtype=torch.float32)
        for items in recommendations.values():
            for item in items[:k]:
                counts[item] += 1.0

        total_exposure = float(counts.sum().item())
        unique_items = float((counts > 0).sum().item())
        probabilities = counts[counts > 0] / total_exposure
        entropy = float(-(probabilities * torch.log2(probabilities)).sum().item())
        normalized_entropy = entropy / math.log2(num_items) if num_items > 1 else 0.0

        head_exposure = counts[head_mask]
        tail_exposure = counts[tail_mask]
        head_average = float(head_exposure.mean().item())
        tail_average = float(tail_exposure.mean().item())
        tail_to_head_ratio = (
            tail_average / head_average if head_average > 0.0 else float("inf")
        )
        exposure_gap = head_average - tail_average
        mean_popularity = float((counts * popularity).sum().item() / total_exposure)
        tail_total = float(tail_exposure.sum().item())
        tail_unique = float((tail_exposure > 0).sum().item())
        user_count = float(len(recommendations))

        metrics[f"unique_items@{k}"] = unique_items
        metrics[f"catalogue_coverage@{k}"] = unique_items / num_items
        metrics[f"exposure_entropy@{k}"] = entropy
        metrics[f"normalized_exposure_entropy@{k}"] = normalized_entropy
        metrics[f"exposure_gini@{k}"] = gini_index(counts)
        metrics[f"head_avg_exposure@{k}"] = head_average
        metrics[f"tail_avg_exposure@{k}"] = tail_average
        metrics[f"head_tail_exposure_gap@{k}"] = exposure_gap
        metrics[f"tail_to_head_exposure_ratio@{k}"] = tail_to_head_ratio
        metrics[f"head_exposure_share@{k}"] = float(head_exposure.sum().item() / total_exposure)
        metrics[f"tail_exposure_share@{k}"] = float(tail_exposure.sum().item() / total_exposure)
        metrics[f"mean_train_popularity@{k}"] = mean_popularity
        # Paper terminology from Abdollahpouri, Burke, and Mobasher (2019).
        metrics[f"arp@{k}"] = mean_popularity
        metrics[f"aplt@{k}"] = tail_total / total_exposure
        metrics[f"aclt@{k}"] = tail_total / user_count
        metrics[f"long_tail_catalogue_coverage@{k}"] = (
            tail_unique / float(tail_mask.sum().item())
        )

    return metrics
