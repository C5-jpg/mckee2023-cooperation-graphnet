import numpy as np

from src.bots import SimulatedHumanBots
from src.config import BotConfig, GameConfig
from src.env import CooperativeNetworkGame, GameState


class AcceptAllBots(SimulatedHumanBots):
    def accept_recommendation(self, recommendation: int, referent_cooperated: int, rng: np.random.Generator) -> bool:
        return True


def _state_with_edge(active: bool) -> GameState:
    cfg = GameConfig()
    adjacency = np.zeros((cfg.n_players, cfg.n_players), dtype=bool)
    adjacency[0, 1] = active
    adjacency[1, 0] = active
    return GameState(
        round_index=3,
        adjacency=adjacency,
        capital=np.ones(cfg.n_players),
        prev_actions=np.zeros(cfg.n_players, dtype=np.int64),
        seed=123,
    )


def test_active_dd_delete_accepted_removes_edge_next_round() -> None:
    cfg = GameConfig()
    game = CooperativeNetworkGame(cfg)
    bots = AcceptAllBots(cfg.n_players, BotConfig(), cfg)
    recs = np.zeros(cfg.n_possible_edges, dtype=np.int64)
    recs[0] = -1

    result = game.adjudicate_recommendations(_state_with_edge(True), recs, bots, np.random.default_rng(1))

    assert result.next_adjacency[0, 1] is np.False_
    assert result.next_adjacency[1, 0] is np.False_
    assert result.recommendation_records[0]["accepted"] is True
    assert result.recommendation_records[0]["edge_active_before"] is True
    assert result.recommendation_records[0]["edge_active_after"] is False
    assert result.recommendation_records[0]["assigned_player"] in {0, 1}
    assert result.recommendation_records[0]["referent_player"] in {0, 1}
    assert result.recommendation_records[0]["assigned_player"] != result.recommendation_records[0]["referent_player"]


def test_inactive_dd_change_accepted_adds_edge_next_round() -> None:
    cfg = GameConfig()
    game = CooperativeNetworkGame(cfg)
    bots = AcceptAllBots(cfg.n_players, BotConfig(), cfg)
    recs = np.zeros(cfg.n_possible_edges, dtype=np.int64)
    recs[0] = 1

    result = game.adjudicate_recommendations(_state_with_edge(False), recs, bots, np.random.default_rng(2))

    assert result.next_adjacency[0, 1] is np.True_
    assert result.next_adjacency[1, 0] is np.True_
    assert result.recommendation_records[0]["accepted"] is True
    assert result.recommendation_records[0]["edge_active_before"] is False
    assert result.recommendation_records[0]["edge_active_after"] is True

