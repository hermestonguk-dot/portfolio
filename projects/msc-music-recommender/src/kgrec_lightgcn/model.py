"""The LightGCN collaborative-filtering model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LightGCN(nn.Module):
    """Propagate user and item ID embeddings over a normalized graph."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers cannot be negative")

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Average the initial and layer-wise propagated embeddings."""
        embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        layer_embeddings = [embeddings]
        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layer_embeddings.append(embeddings)

        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return torch.split(final_embeddings, (self.num_users, self.num_items), dim=0)

    def bpr_loss(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        adjacency: torch.Tensor,
        regularization_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pairwise BPR ranking loss plus ego-embedding L2 penalty."""
        user_embeddings, item_embeddings = self.propagate(adjacency)
        users_final = user_embeddings[users]
        positives_final = item_embeddings[positive_items]
        negatives_final = item_embeddings[negative_items]

        positive_scores = (users_final * positives_final).sum(dim=1)
        negative_scores = (users_final * negatives_final).sum(dim=1)
        ranking_loss = F.softplus(negative_scores - positive_scores).mean()

        users_ego = self.user_embedding(users)
        positives_ego = self.item_embedding(positive_items)
        negatives_ego = self.item_embedding(negative_items)
        regularization = (
            users_ego.square().sum()
            + positives_ego.square().sum()
            + negatives_ego.square().sum()
        ) / (2.0 * users.shape[0])
        total_loss = ranking_loss + regularization_weight * regularization
        return total_loss, ranking_loss.detach(), regularization.detach()
