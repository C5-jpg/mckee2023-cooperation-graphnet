from __future__ import annotations

import numpy as np

from src.bots import BotDecisionInfo, SimulatedHumanBots
from src.config import GameConfig
from src.env import CooperativeNetworkGame, GameState
from src.planners import BasePlanner


class FixedBots(SimulatedHumanBots):
    def __init__(self, actions, accept=True, game_config=None):
        cfg = game_config or GameConfig(n_players=len(actions))
        super().__init__(cfg.n_players, game_config=cfg)
        self.fixed_actions = np.asarray(actions, dtype=np.int64)
        self.accept = accept

    def reset_episode(self, rng):
        return None

    def choose_cooperation(self, round_index, adjacency, prev_actions, capital, rng):
        probs = self.fixed_actions.astype(float)
        return BotDecisionInfo(probs, self.fixed_actions.copy(), np.zeros_like(self.fixed_actions, dtype=bool))

    def accept_recommendation(self, recommendation, referent_cooperated, rng):
        return self.accept


class AllChangePlanner(BasePlanner):
    name = "all_change"

    def recommend(self, state, rng):
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            recs[idx] = -1 if state.adjacency[i, j] else 1
        return recs


def test_possible_edges_for_16_players():
    env = CooperativeNetworkGame(GameConfig())
    assert len(env.edges) == 120
    assert env.config.n_possible_edges == 120


def test_reset_has_no_self_loops_and_is_undirected():
    env = CooperativeNetworkGame(GameConfig())
    state = env.reset(20260624)
    assert not state.adjacency.diagonal().any()
    assert np.array_equal(state.adjacency, state.adjacency.T)
    assert np.allclose(state.capital, 1.0)


def test_initial_er_density_plausible():
    env = CooperativeNetworkGame(GameConfig())
    densities = []
    for seed in range(100, 140):
        state = env.reset(seed)
        densities.append(state.adjacency.sum() / (16 * 15))
    assert 0.20 < float(np.mean(densities)) < 0.40


def test_capital_update_matches_equation():
    cfg = GameConfig(n_players=3, initial_capital=1.0, benefit_b=0.1, cost_c=0.05)
    env = CooperativeNetworkGame(cfg)
    adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    state = GameState(1, adjacency, np.ones(3), np.zeros(3, dtype=int), seed=1)
    bots = FixedBots([1, 0, 1], game_config=cfg)
    trans = env.cooperation_transition(state, bots, np.random.default_rng(1))
    assert np.allclose(trans.payoffs, [0.0, 0.1, 0.05])
    assert np.allclose(trans.planner_state.capital, [1.0, 1.1, 1.05])


def test_capital_constraint_forces_defection():
    cfg = GameConfig(n_players=3, cost_c=0.05)
    bots = SimulatedHumanBots(3, game_config=cfg)
    bots.theta[:] = 100.0
    adjacency = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(adjacency, False)
    decision = bots.choose_cooperation(1, adjacency, np.zeros(3, dtype=int), np.array([0.01, 1.0, 1.0]), np.random.default_rng(3))
    assert decision.actions[0] == 0
    assert decision.constrained[0]


def test_accepted_add_delete_and_rejected_noop():
    cfg = GameConfig(n_players=3)
    env = CooperativeNetworkGame(cfg)
    adjacency = np.zeros((3, 3), dtype=bool)
    state = GameState(1, adjacency, np.ones(3), np.array([1, 1, 0]), seed=5)
    recs = np.zeros(len(env.edges), dtype=int)
    recs[0] = 1
    accepted = env.adjudicate_recommendations(state, recs, FixedBots([1, 1, 0], True, cfg), np.random.default_rng(1))
    assert accepted.next_adjacency[0, 1]
    recs[0] = -1
    deleted = env.adjudicate_recommendations(GameState(1, accepted.next_adjacency, np.ones(3), np.array([1, 1, 0]), 5), recs, FixedBots([1, 1, 0], True, cfg), np.random.default_rng(2))
    assert not deleted.next_adjacency[0, 1]
    rejected = env.adjudicate_recommendations(GameState(1, accepted.next_adjacency, np.ones(3), np.array([1, 1, 0]), 5), recs, FixedBots([1, 1, 0], False, cfg), np.random.default_rng(3))
    assert rejected.next_adjacency[0, 1]


def test_final_round_skips_recommendations():
    cfg = GameConfig(n_players=4, episode_length=2)
    env = CooperativeNetworkGame(cfg)
    state = env.reset(10)
    state.round_index = 2
    bots = FixedBots([1, 1, 1, 1], True, cfg)
    result = env.step(state, AllChangePlanner(cfg), bots, np.random.default_rng(10))
    assert result.recommendation_records == []
    assert result.done
