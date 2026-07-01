from __future__ import annotations

import torch

from src.config import GameConfig
from src.env import CooperativeNetworkGame
from src.graphnet import GraphNet, build_graph_batch, directed_edge_index


def test_directed_edge_mapping_for_16_players():
    cfg = GameConfig()
    directed_edges, mapping = directed_edge_index(cfg)
    assert cfg.n_possible_edges == 120
    assert len(directed_edges) == 240
    assert len(mapping) == 240
    assert all(i != j for i, j in directed_edges)
    counts = torch.bincount(torch.tensor(mapping), minlength=cfg.n_possible_edges)
    assert counts.shape == (cfg.n_possible_edges,)
    assert torch.all(counts == 2)


def test_graph_batch_uses_directed_messages_and_undirected_policy_shape():
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    batch = build_graph_batch([env.reset(11)], cfg)
    assert batch.edge_attr.shape == (1, 240, 1)
    assert batch.adjacency.shape == (1, 120)
    assert batch.senders.shape == (240,)
    assert batch.receivers.shape == (240,)
    assert batch.directed_to_undirected.shape == (240,)
    model = GraphNet(cfg)
    logits, value = model(batch)
    assert logits.shape == (1, 120, 2)
    assert value.shape == (1,)

