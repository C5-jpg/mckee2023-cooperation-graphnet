from __future__ import annotations

import numpy as np
import torch

from src.config import GameConfig
from src.env import CooperativeNetworkGame
from src.graphnet import target_edge_states_to_recommendations


def test_target_edge_states_map_to_add_delete_noop_numpy():
    cfg = GameConfig(n_players=3)
    env = CooperativeNetworkGame(cfg)
    adjacency = np.zeros((3, 3), dtype=bool)
    adjacency[0, 1] = True
    adjacency[1, 0] = True

    # Edges are (0,1), (0,2), (1,2). Desired states:
    # (0,1) active -> inactive means delete.
    # (0,2) inactive -> active means add.
    # (1,2) inactive -> inactive means no-op.
    recs = env.target_edge_states_to_recommendations(np.array([0, 1, 0]), adjacency)
    assert recs.tolist() == [-1, 1, 0]


def test_target_edge_states_map_to_add_delete_noop_torch():
    adjacency = torch.tensor([1.0, 0.0, 0.0])
    recs = target_edge_states_to_recommendations(torch.tensor([0, 1, 0]), adjacency)
    assert recs.tolist() == [-1, 1, 0]
