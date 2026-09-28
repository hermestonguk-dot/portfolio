"""Small deterministic checks for the LightGCN implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kgrec_lightgcn.data import BPRSampler, KGRecData, build_normalized_adjacency
from kgrec_lightgcn.evaluation import evaluate_topk
from kgrec_lightgcn.fairness import (
    CandidatePool,
    evaluate_recommendation_lists,
    exposure_metrics,
    gini_index,
    popularity_head_mask_by_interaction_share,
    recommendations_from_candidate_pool,
    rerank_candidate_pool_with_smooth_xquad,
    rerank_with_popularity_penalty,
    train_item_popularity,
    user_head_tail_preferences,
)
from kgrec_lightgcn.model import LightGCN


def make_data() -> KGRecData:
    return KGRecData(
        num_users=2,
        num_items=3,
        train_users=torch.tensor([0, 0, 1]),
        train_items=torch.tensor([0, 1, 2]),
        train_seen=[{0, 1}, {2}],
        validation_targets={0: 2, 1: 1},
        test_targets={0: 2, 1: 0},
    )


class LightGCNTests(unittest.TestCase):
    def test_adjacency_is_symmetric_and_normalized(self) -> None:
        adjacency = build_normalized_adjacency(make_data(), torch.device("cpu"))
        dense = adjacency.to_dense()
        self.assertTrue(torch.allclose(dense, dense.T))
        self.assertAlmostEqual(dense[0, 2].item(), 2 ** -0.5, places=6)

    def test_propagation_and_bpr_loss(self) -> None:
        data = make_data()
        adjacency = build_normalized_adjacency(data, torch.device("cpu"))
        model = LightGCN(2, 3, embedding_dim=4, num_layers=2)
        users, items = model.propagate(adjacency)
        self.assertEqual(users.shape, (2, 4))
        self.assertEqual(items.shape, (3, 4))

        loss, _, _ = model.bpr_loss(
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
            torch.tensor([2, 0]),
            adjacency,
            regularization_weight=1e-4,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.user_embedding.weight.grad)

    def test_sampler_never_returns_seen_items(self) -> None:
        data = make_data()
        users, _, negatives = BPRSampler(data, seed=7).sample(100)
        for user, negative in zip(users.tolist(), negatives.tolist()):
            self.assertNotIn(negative, data.train_seen[user])

    def test_ranking_metrics_mask_seen_items(self) -> None:
        user_embeddings = torch.tensor([[1.0, 0.0]])
        item_embeddings = torch.tensor([[10.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
        metrics = evaluate_topk(
            user_embeddings,
            item_embeddings,
            targets={0: 1},
            exclusions=[{0}],
            top_ks=[1, 2],
            batch_size=1,
        )
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["ndcg@1"], 1.0)
        self.assertEqual(metrics["precision@2"], 0.5)

    def test_gini_index_handles_exposure_inequality(self) -> None:
        self.assertEqual(gini_index(torch.tensor([0.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(gini_index(torch.tensor([0.0, 0.0, 1.0, 1.0])), 0.5)

    def test_popularity_rerank_can_move_long_tail_candidate_up(self) -> None:
        user_embeddings = torch.tensor([[1.0, 0.0]])
        item_embeddings = torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.7, 0.0]])
        popularity = torch.tensor([100.0, 1.0, 1.0])

        base = rerank_with_popularity_penalty(
            user_embeddings,
            item_embeddings,
            users=[0],
            exclusions=[set()],
            popularity=popularity,
            top_k=2,
            batch_size=1,
            fairness_weight=0.0,
            candidate_pool_size=3,
        )
        fair = rerank_with_popularity_penalty(
            user_embeddings,
            item_embeddings,
            users=[0],
            exclusions=[set()],
            popularity=popularity,
            top_k=2,
            batch_size=1,
            fairness_weight=0.8,
            candidate_pool_size=3,
        )

        self.assertEqual(base[0][0], 0)
        self.assertEqual(fair[0][0], 1)

    def test_smooth_xquad_head_covers_80_percent_of_interactions(self) -> None:
        popularity = torch.tensor([5.0, 3.0, 1.0, 1.0])
        head_mask = popularity_head_mask_by_interaction_share(
            popularity, head_interaction_share=0.8
        )
        self.assertEqual(head_mask.tolist(), [True, True, False, False])
        self.assertAlmostEqual(
            float(popularity[head_mask].sum() / popularity.sum()), 0.8
        )

    def test_smooth_xquad_preferences_use_each_training_profile(self) -> None:
        head_mask = torch.tensor([True, True, False, False])
        preferences = user_head_tail_preferences(
            train_seen=[{0, 2, 3}, {0, 1}],
            head_mask=head_mask,
        )
        self.assertTrue(
            torch.allclose(preferences[0], torch.tensor([2 / 3, 1 / 3]))
        )
        self.assertTrue(torch.allclose(preferences[1], torch.tensor([0.0, 1.0])))

    def test_smooth_xquad_lambda_zero_reproduces_base_order(self) -> None:
        candidates = CandidatePool(
            users=[0],
            item_ids=torch.tensor([[0, 1, 2]]),
            relevance_scores=torch.tensor([[0.9, 0.8, 0.7]]),
        )
        base = recommendations_from_candidate_pool(candidates, top_k=2)
        smooth = rerank_candidate_pool_with_smooth_xquad(
            candidate_pool=candidates,
            head_mask=torch.tensor([True, False, True]),
            user_preferences=torch.tensor([[0.5, 0.5]]),
            top_k=2,
            lambda_weight=0.0,
        )
        self.assertEqual(smooth, base)

    def test_smooth_xquad_promotes_the_underrepresented_category(self) -> None:
        candidates = CandidatePool(
            users=[0],
            item_ids=torch.tensor([[0, 1, 2]]),
            relevance_scores=torch.tensor([[0.9, 0.8, 0.7]]),
        )
        smooth = rerank_candidate_pool_with_smooth_xquad(
            candidate_pool=candidates,
            head_mask=torch.tensor([True, False, True]),
            user_preferences=torch.tensor([[0.5, 0.5]]),
            top_k=2,
            lambda_weight=0.8,
        )
        self.assertEqual(smooth[0], [0, 1])

    def test_smooth_xquad_personalizes_the_first_selection(self) -> None:
        candidates = CandidatePool(
            users=[0, 1],
            item_ids=torch.tensor([[0, 1, 2], [0, 1, 2]]),
            relevance_scores=torch.tensor(
                [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]]
            ),
        )
        smooth = rerank_candidate_pool_with_smooth_xquad(
            candidate_pool=candidates,
            head_mask=torch.tensor([True, False, True]),
            user_preferences=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            top_k=1,
            lambda_weight=0.8,
        )
        self.assertEqual(smooth[0], [1])
        self.assertEqual(smooth[1], [0])

    def test_recommendation_list_metrics_and_exposure_metrics(self) -> None:
        recommendations = {0: [1, 2], 1: [2, 0]}
        ranking = evaluate_recommendation_lists(
            recommendations,
            targets={0: 1, 1: 0},
            top_ks=[1, 2],
        )
        self.assertEqual(ranking["recall@1"], 0.5)
        self.assertEqual(ranking["recall@2"], 1.0)

        data = make_data()
        popularity = train_item_popularity(data)
        exposure = exposure_metrics(
            recommendations,
            num_items=3,
            popularity=popularity,
            top_ks=[2],
            head_fraction=1 / 3,
            head_mask=torch.tensor([True, False, False]),
        )
        self.assertEqual(exposure["unique_items@2"], 3.0)
        self.assertAlmostEqual(exposure["catalogue_coverage@2"], 1.0)
        self.assertAlmostEqual(exposure["aplt@2"], 0.75)
        self.assertAlmostEqual(exposure["aclt@2"], 1.5)
        self.assertAlmostEqual(exposure["long_tail_catalogue_coverage@2"], 1.0)


if __name__ == "__main__":
    unittest.main()
