from __future__ import annotations

import numpy as np

from src.config import ENCOURAGEMENT_TABLES, NEUTRAL_TABLE, GameConfig
from src.env import GameState
from src.planners import (
    CooperativeClusteringPlanner,
    EncouragementPlanner,
    MaximumConnectivityPlanner,
    NeutralPlanner,
    RandomRecommendationPlanner,
    StaticPlanner,
)


def make_state(actions=None, adjacency=None, n=4, round_index=1):
    if actions is None:
        actions = np.array([1, 1, 0, 0], dtype=int)
    if adjacency is None:
        adjacency = np.zeros((n, n), dtype=bool)
    return GameState(round_index, adjacency, np.ones(n), np.asarray(actions, dtype=int), seed=1)


def test_static_planner_no_changes():
    cfg = GameConfig(n_players=4)
    recs = StaticPlanner(cfg).recommend(make_state(n=4), np.random.default_rng(1))
    assert np.count_nonzero(recs) == 0


def test_random_planner_approximately_30_percent():
    cfg = GameConfig(n_players=16)
    planner = RandomRecommendationPlanner(cfg)
    state = make_state(actions=np.ones(16, dtype=int), adjacency=np.zeros((16, 16), dtype=bool), n=16)
    rng = np.random.default_rng(2)
    rates = [np.count_nonzero(planner.recommend(state, rng)) / cfg.n_possible_edges for _ in range(200)]
    assert 0.25 < float(np.mean(rates)) < 0.35


def test_cooperative_clustering_disconnects_defecting_neighbor():
    cfg = GameConfig(n_players=4)
    adjacency = np.zeros((4, 4), dtype=bool)
    adjacency[0, 2] = adjacency[2, 0] = True
    planner = CooperativeClusteringPlanner(cfg, n_focal=1, random_fraction=0.0)
    planner.focal_players = [0]
    recs = planner.recommend(make_state(actions=np.array([1, 1, 0, 1]), adjacency=adjacency), np.random.default_rng(3))
    assert -1 in recs


def test_encouragement_table_for_defect_defect_delete():
    cfg = GameConfig(n_players=2)
    adjacency = np.array([[0, 1], [1, 0]], dtype=bool)
    state = make_state(actions=np.array([0, 0]), adjacency=adjacency, n=2, round_index=1)
    recs = EncouragementPlanner(cfg).recommend(state, np.random.default_rng(4))
    assert recs[0] == -1
    assert ENCOURAGEMENT_TABLES["dd"]["delete"][0] == 1.0


def test_neutral_table_bounds_and_max_connectivity():
    cfg = GameConfig(n_players=3)
    state = make_state(actions=np.ones(3, dtype=int), adjacency=np.zeros((3, 3), dtype=bool), n=3)
    recs = MaximumConnectivityPlanner(cfg).recommend(state, np.random.default_rng(5))
    assert np.all(recs == 1)
    assert 0.0 <= NEUTRAL_TABLE["add"][0] <= 1.0
    neutral = NeutralPlanner(cfg).recommend(state, np.random.default_rng(6))
    assert set(np.unique(neutral)).issubset({0, 1})

