from __future__ import annotations

import numpy as np

from src.bots import BotDecisionInfo, SimulatedHumanBots
from src.config import GameConfig
from src.env import CooperativeNetworkGame, GameState
from src.planners import BasePlanner


class OneEdgePlanner(BasePlanner):
    name = "one_edge"

    def __init__(self, config=None, rec=1):
        super().__init__(config)
        self.rec = rec

    def recommend(self, state, rng):
        recs = np.zeros(len(self.edges), dtype=int)
        recs[0] = self.rec
        return recs


class FixedBots(SimulatedHumanBots):
    def __init__(self, actions, accept=True, game_config=None):
        cfg = game_config or GameConfig(n_players=len(actions))
        super().__init__(cfg.n_players, game_config=cfg)
        self.actions = np.asarray(actions, dtype=int)
        self.accept = accept

    def reset_episode(self, rng):
        return None

    def choose_cooperation(self, round_index, adjacency, prev_actions, capital, rng):
        return BotDecisionInfo(self.actions.astype(float), self.actions.copy(), np.zeros_like(self.actions, dtype=bool))

    def accept_recommendation(self, recommendation, referent_cooperated, rng):
        return self.accept


def test_recommendation_assignment_and_next_round_graph_change():
    cfg = GameConfig(n_players=3, episode_length=3)
    env = CooperativeNetworkGame(cfg)
    state = GameState(1, np.zeros((3, 3), dtype=bool), np.ones(3), np.zeros(3, dtype=int), seed=1)
    result = env.step(state, OneEdgePlanner(cfg, rec=1), FixedBots([1, 0, 1], True, cfg), np.random.default_rng(1))
    assert len(result.recommendation_records) == 1
    rec = result.recommendation_records[0]
    assert {rec["assigned_player"], rec["referent_player"]} == {0, 1}
    assert result.next_state.adjacency[0, 1]
    assert result.edge_records[0]["edge_active_before"] is False
    assert result.edge_records[0]["edge_active_after"] is True

