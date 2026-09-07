"""LightGCN components for the KGRec-music recommender project."""

from .data import KGRecData, build_normalized_adjacency, load_kgrec_data
from .model import LightGCN

__all__ = [
    "KGRecData",
    "LightGCN",
    "build_normalized_adjacency",
    "load_kgrec_data",
]
