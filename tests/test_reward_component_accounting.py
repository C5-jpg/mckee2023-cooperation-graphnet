import numpy as np

from src.bots import SimulatedHumanBots
from src.config import BotConfig, GameConfig
from src.counterfactuals import reward_components
from src.env import CooperativeNetworkGame, GameState


class RejectAllBots(SimulatedHumanBots):
    def accept_recommendation(self, recommendation: int, referent_cooperated: int, rng: np.random.Generator) -> bool:
        return False


def test_reward_component_accounting_separates_capital_and_penalty() -> None:
    cfg = GameConfig()
    game = CooperativeNetworkGame(cfg)
    adjacency = np.zeros((cfg.n_players, cfg.n_players), dtype=bool)
    adjacency[0, 1] = True
    adjacency[1, 0] = True
    state = GameState(
        round_index=5,
        adjacency=adjacency,
        capital=np.ones(cfg.n_players) * 2.0,
        prev_actions=np.zeros(cfg.n_players, dtype=np.int64),
        seed=7,
    )
    recs = np.zeros(cfg.n_possible_edges, dtype=np.int64)
    recs[0] = -1
    bots = RejectAllBots(cfg.n_players, BotConfig(), cfg)

    adjudication = game.adjudicate_recommendations(state, recs, bots, np.random.default_rng(3))
    components = reward_components(state, adjudication, cfg)

    assert adjudication.rejected_nonzero == 1
    assert components["mean_capital_reward"] == 2.0
    assert components["recommendation_penalty"] == 1.0 / cfg.n_possible_edges
    assert components["total_reward"] == 2.0 - 1.0 / cfg.n_possible_edges

