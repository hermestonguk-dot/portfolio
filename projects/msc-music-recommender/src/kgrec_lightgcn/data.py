"""Data loading and graph construction for KGRec-music LightGCN training."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class KGRecData:
    """The processed interactions required for training and evaluation."""

    num_users: int
    num_items: int
    train_users: torch.Tensor
    train_items: torch.Tensor
    train_seen: list[set[int]]
    validation_targets: dict[int, int]
    test_targets: dict[int, int]

    def test_exclusions(self) -> list[set[int]]:
        """Hide training and validation items when evaluating the test item."""
        exclusions = [set(items) for items in self.train_seen]
        for user, item in self.validation_targets.items():
            exclusions[user].add(item)
        return exclusions


def _read_user_item_rows(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")

    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_idx", "item_idx"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain user_idx and item_idx columns")
        for row in reader:
            rows.append((int(row["user_idx"]), int(row["item_idx"])))
    return rows


def _single_target_per_user(
    rows: list[tuple[int, int]], split_name: str
) -> dict[int, int]:
    targets: dict[int, int] = {}
    for user, item in rows:
        if user in targets:
            raise ValueError(
                f"Expected one {split_name} target per user, but user {user} has more"
            )
        targets[user] = item
    return targets


def load_kgrec_data(data_dir: Path) -> KGRecData:
    """Load the files produced by ``scripts/prepare_kgrec.py``."""
    stats_path = data_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run scripts/prepare_kgrec.py first."
        )

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    num_users = int(stats["num_users"])
    num_items = int(stats["num_items"])
    train_rows = _read_user_item_rows(data_dir / "train.csv")
    validation_rows = _read_user_item_rows(data_dir / "val.csv")
    test_rows = _read_user_item_rows(data_dir / "test.csv")

    if not train_rows:
        raise ValueError("The training split is empty")

    train_seen = [set() for _ in range(num_users)]
    for user, item in train_rows:
        if not 0 <= user < num_users or not 0 <= item < num_items:
            raise ValueError(f"Out-of-range training interaction: user={user}, item={item}")
        train_seen[user].add(item)

    return KGRecData(
        num_users=num_users,
        num_items=num_items,
        train_users=torch.tensor([user for user, _ in train_rows], dtype=torch.long),
        train_items=torch.tensor([item for _, item in train_rows], dtype=torch.long),
        train_seen=train_seen,
        validation_targets=_single_target_per_user(validation_rows, "validation"),
        test_targets=_single_target_per_user(test_rows, "test"),
    )


def build_normalized_adjacency(
    data: KGRecData, device: torch.device
) -> torch.Tensor:
    """Build D^-1/2 A D^-1/2 for the bipartite user-item graph."""
    user_nodes = data.train_users
    item_nodes = data.train_items + data.num_users
    sources = torch.cat((user_nodes, item_nodes))
    destinations = torch.cat((item_nodes, user_nodes))
    num_nodes = data.num_users + data.num_items

    degrees = torch.bincount(sources, minlength=num_nodes).float()
    if torch.any(degrees == 0):
        isolated = int((degrees == 0).sum().item())
        raise ValueError(f"Training graph contains {isolated} isolated nodes")

    weights = degrees[sources].rsqrt() * degrees[destinations].rsqrt()
    indices = torch.stack((sources, destinations))
    adjacency = torch.sparse_coo_tensor(
        indices,
        weights,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()
    return adjacency.to(device)


class BPRSampler:
    """Sample observed user-item pairs and unobserved negative items."""

    def __init__(self, data: KGRecData, seed: int) -> None:
        self.data = data
        self.rng = random.Random(seed)

    def sample(self, sample_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

        interaction_count = len(self.data.train_users)
        pair_indices = [self.rng.randrange(interaction_count) for _ in range(sample_count)]
        users = self.data.train_users[pair_indices]
        positive_items = self.data.train_items[pair_indices]

        negative_items: list[int] = []
        for user in users.tolist():
            candidate = self.rng.randrange(self.data.num_items)
            while candidate in self.data.train_seen[user]:
                candidate = self.rng.randrange(self.data.num_items)
            negative_items.append(candidate)

        return users, positive_items, torch.tensor(negative_items, dtype=torch.long)
