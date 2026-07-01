from __future__ import annotations

import numpy as np
import torch

from src.config import GameConfig
from src.env import CooperativeNetworkGame, GameState
from src.planners import GraphNetPlanner


class FixedLogitModel:
    feature_mode = "augmented"

    def __init__(self, actions: list[int]) -> None:
        self.actions = actions

    def eval(self) -> None:
        return None

    def __call__(self, batch):
        logits = torch.full((1, len(self.actions), 2), -10.0)
        for idx, action in enumerate(self.actions):
            logits[0, idx, action] = 10.0
        return logits, torch.zeros(1)


def small_state() -> GameState:
    adjacency = np.zeros((3, 3), dtype=bool)
    adjacency[0, 1] = True
    adjacency[1, 0] = True
    return GameState(
        round_index=1,
        adjacency=adjacency,
        capital=np.ones(3),
        prev_actions=np.array([1, 0, 0]),
        seed=42,
    )


def test_environment_target_state_semantics_maps_classes_to_target_edge_state() -> None:
    game = CooperativeNetworkGame(GameConfig(n_players=3))
    recs = game.target_edge_states_to_recommendations(np.array([0, 1, 0]), small_state().adjacency)
    assert recs.tolist() == [-1, 1, 0]


def test_environment_change_semantics_is_separate_ablation_mapping() -> None:
    game = CooperativeNetworkGame(GameConfig(n_players=3, action_semantics="change"))
    recs = game.binary_changes_to_recommendations(np.array([1, 1, 0]), small_state().adjacency)
    assert recs.tolist() == [-1, 1, 0]


def test_graphnet_planner_defaults_to_target_state_semantics() -> None:
    cfg = GameConfig(n_players=3)
    planner = GraphNetPlanner(model=FixedLogitModel([0, 1, 0]), config=cfg)
    recs = planner.recommend(small_state(), np.random.default_rng(7))
    assert planner.action_semantics == "target_state"
    assert recs.tolist() == [-1, 1, 0]


def test_graphnet_planner_change_semantics_requires_explicit_flag() -> None:
    cfg = GameConfig(n_players=3, action_semantics="change")
    planner = GraphNetPlanner(model=FixedLogitModel([1, 1, 0]), config=cfg, action_semantics="change")
    recs = planner.recommend(small_state(), np.random.default_rng(7))
    assert recs.tolist() == [-1, 1, 0]
