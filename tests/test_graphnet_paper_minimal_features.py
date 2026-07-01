from __future__ import annotations

import torch

from src.config import GameConfig, GraphNetConfig
from src.env import CooperativeNetworkGame
from src.graphnet import GraphNet, build_graph_batch, graph_feature_dimensions


def test_paper_minimal_feature_dimensions_match_paper_observation_set() -> None:
    assert graph_feature_dimensions("paper_minimal") == (2, 1, 1)
    assert graph_feature_dimensions("augmented") == (3, 1, 4)


def test_paper_minimal_graph_batch_excludes_degree_and_global_augmentation() -> None:
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    state = env.reset(42)
    batch = build_graph_batch([state], cfg, feature_mode="paper_minimal")

    assert batch.feature_mode == "paper_minimal"
    assert batch.node_attr.shape == (1, cfg.n_players, 2)
    assert batch.edge_attr.shape == (1, 2 * cfg.n_possible_edges, 1)
    assert batch.global_attr.shape == (1, 1)
    assert torch.allclose(batch.global_attr, torch.ones_like(batch.global_attr))
    assert torch.allclose(batch.node_attr[0, :, 0].cpu(), torch.tensor(state.capital, dtype=torch.float32))
    assert torch.allclose(batch.node_attr[0, :, 1].cpu(), torch.tensor(state.prev_actions, dtype=torch.float32))


def test_paper_minimal_graphnet_forward_returns_undirected_policy_and_scalar_value() -> None:
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    batch = build_graph_batch([env.reset(43)], cfg, feature_mode="paper_minimal")
    model = GraphNet(cfg, GraphNetConfig(feature_mode="paper_minimal"))
    logits, value = model(batch)

    assert model.feature_mode == "paper_minimal"
    assert logits.shape == (1, cfg.n_possible_edges, 2)
    assert value.shape == (1,)
