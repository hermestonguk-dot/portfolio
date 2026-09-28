"""Prepare KGRec-music for the recommender MVP.

This script converts the raw KGRec-music files into clean processed files:
- contiguous user/item id mappings
- deduplicated interactions
- per-user train/validation/test splits
- item metadata with tags, descriptions, and popularity
- LightGCN-ready graph edge CSV files

It intentionally uses only the Python standard library so it can run before the
ML environment is fully installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_RAW_DIR = Path("KGRec-dataset") / "KGRec-music"
DEFAULT_OUT_DIR = Path("data") / "processed" / "kgrec_music"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def find_interaction_file(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "implicit_lf_dataset.csv",
        raw_dir / "implicit_lf_dataset.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find KGRec-music interaction file. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def load_interactions(path: Path) -> tuple[list[tuple[str, str, int]], int]:
    """Load and deduplicate raw implicit feedback rows."""
    seen: set[tuple[str, str]] = set()
    interactions: list[tuple[str, str, int]] = []
    raw_rows = 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            raw_rows += 1
            if len(row) < 2:
                continue
            user_id = row[0].strip()
            item_id = row[1].strip()
            value = int(float(row[2])) if len(row) > 2 and row[2].strip() else 1
            key = (user_id, item_id)
            if key in seen:
                continue
            seen.add(key)
            interactions.append((user_id, item_id, value))

    return interactions, raw_rows


def make_id_maps(
    interactions: Iterable[tuple[str, str, int]],
) -> tuple[dict[str, int], dict[str, int]]:
    user_ids = sorted({user_id for user_id, _, _ in interactions}, key=lambda x: int(x))
    item_ids = sorted({item_id for _, item_id, _ in interactions}, key=lambda x: int(x))
    return (
        {user_id: idx for idx, user_id in enumerate(user_ids)},
        {item_id: idx for idx, item_id in enumerate(item_ids)},
    )


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def build_mapped_interactions(
    interactions: list[tuple[str, str, int]],
    user_map: dict[str, int],
    item_map: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for user_id, item_id, value in interactions:
        rows.append(
            {
                "raw_user_id": user_id,
                "raw_item_id": item_id,
                "value": value,
                "user_idx": user_map[user_id],
                "item_idx": item_map[item_id],
            }
        )
    return rows


def split_per_user(
    mapped_rows: list[dict[str, object]],
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Create a simple per-user split for top-k recommendation.

    Users with >=3 interactions get one validation and one test interaction.
    Users with 2 interactions get one train and one test interaction.
    Users with 1 interaction remain in train.
    """
    rng = random.Random(seed)
    by_user: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in mapped_rows:
        by_user[int(row["user_idx"])].append(row)

    train: list[dict[str, object]] = []
    val: list[dict[str, object]] = []
    test: list[dict[str, object]] = []

    for rows in by_user.values():
        rows = list(rows)
        rng.shuffle(rows)
        if len(rows) >= 3:
            test.append(rows[0])
            val.append(rows[1])
            train.extend(rows[2:])
        elif len(rows) == 2:
            test.append(rows[0])
            train.append(rows[1])
        else:
            train.extend(rows)

    return train, val, test


def load_item_metadata(
    raw_dir: Path,
    item_map: dict[str, int],
    popularity: dict[str, int],
    include_text: bool,
) -> list[dict[str, object]]:
    tags_dir = raw_dir / "tags"
    descriptions_dir = raw_dir / "descriptions"

    metadata: list[dict[str, object]] = []
    for raw_item_id, item_idx in sorted(item_map.items(), key=lambda kv: kv[1]):
        tag_path = tags_dir / f"{raw_item_id}.txt"
        description_path = descriptions_dir / f"{raw_item_id}.txt"
        has_tags = tag_path.exists()
        has_description = description_path.exists()
        tags = read_text(tag_path) if include_text and has_tags else ""
        description = (
            read_text(description_path) if include_text and has_description else ""
        )
        metadata.append(
            {
                "raw_item_id": raw_item_id,
                "item_idx": item_idx,
                "popularity": popularity.get(raw_item_id, 0),
                "has_tags": has_tags,
                "has_description": has_description,
                "tag_path": str(tag_path) if has_tags else "",
                "description_path": str(description_path) if has_description else "",
                "tags": tags,
                "description": description,
            }
        )
    return metadata


def write_graph_edges(
    out_dir: Path,
    train_rows: list[dict[str, object]],
    num_users: int,
) -> tuple[int, int]:
    bipartite_fields = ["user_idx", "item_idx"]
    bipartite_count = write_rows(
        out_dir / "train_edges_bipartite.csv",
        bipartite_fields,
        (
            {
                "user_idx": row["user_idx"],
                "item_idx": row["item_idx"],
            }
            for row in train_rows
        ),
    )

    undirected_rows = []
    for row in train_rows:
        user_node = int(row["user_idx"])
        item_node = num_users + int(row["item_idx"])
        undirected_rows.append({"src": user_node, "dst": item_node})
        undirected_rows.append({"src": item_node, "dst": user_node})

    undirected_count = write_rows(
        out_dir / "train_edges_undirected.csv",
        ["src", "dst"],
        undirected_rows,
    )
    return bipartite_count, undirected_count


def prepare_dataset(
    raw_dir: Path,
    out_dir: Path,
    seed: int,
    include_text: bool,
) -> dict[str, object]:
    interaction_file = find_interaction_file(raw_dir)
    interactions, raw_rows = load_interactions(interaction_file)
    user_map, item_map = make_id_maps(interactions)
    mapped_rows = build_mapped_interactions(interactions, user_map, item_map)
    train_rows, val_rows, test_rows = split_per_user(mapped_rows, seed)

    popularity: dict[str, int] = defaultdict(int)
    for _, item_id, _ in interactions:
        popularity[item_id] += 1

    metadata_rows = load_item_metadata(raw_dir, item_map, popularity, include_text)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        out_dir / "user_id_map.csv",
        ["raw_user_id", "user_idx"],
        ({"raw_user_id": user_id, "user_idx": idx} for user_id, idx in user_map.items()),
    )
    write_rows(
        out_dir / "item_id_map.csv",
        ["raw_item_id", "item_idx"],
        ({"raw_item_id": item_id, "item_idx": idx} for item_id, idx in item_map.items()),
    )

    interaction_fields = ["raw_user_id", "raw_item_id", "value", "user_idx", "item_idx"]
    write_rows(out_dir / "interactions.csv", interaction_fields, mapped_rows)
    write_rows(out_dir / "train.csv", interaction_fields, train_rows)
    write_rows(out_dir / "val.csv", interaction_fields, val_rows)
    write_rows(out_dir / "test.csv", interaction_fields, test_rows)
    write_rows(
        out_dir / "item_metadata.csv",
        [
            "raw_item_id",
            "item_idx",
            "popularity",
            "has_tags",
            "has_description",
            "tag_path",
            "description_path",
            "tags",
            "description",
        ],
        metadata_rows,
    )

    bipartite_edges, undirected_edges = write_graph_edges(out_dir, train_rows, len(user_map))

    stats = {
        "raw_dir": str(raw_dir),
        "interaction_file": str(interaction_file),
        "processed_dir": str(out_dir),
        "seed": seed,
        "include_text": include_text,
        "raw_interaction_rows": raw_rows,
        "unique_interactions": len(interactions),
        "duplicates_removed": raw_rows - len(interactions),
        "num_users": len(user_map),
        "num_items": len(item_map),
        "train_interactions": len(train_rows),
        "val_interactions": len(val_rows),
        "test_interactions": len(test_rows),
        "train_bipartite_edges": bipartite_edges,
        "train_undirected_edges": undirected_edges,
        "items_with_tags": sum(1 for row in metadata_rows if row["has_tags"]),
        "items_without_tags": sum(1 for row in metadata_rows if not row["has_tags"]),
        "items_with_descriptions": sum(1 for row in metadata_rows if row["has_description"]),
        "items_without_descriptions": sum(
            1 for row in metadata_rows if not row["has_description"]
        ),
    }

    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KGRec-music for the MVP.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Path to KGRec-music raw directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for processed files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-user split.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full tag and description text in item_metadata.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_dataset(args.raw_dir, args.out_dir, args.seed, args.include_text)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
