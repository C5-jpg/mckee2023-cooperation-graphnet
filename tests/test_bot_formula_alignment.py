from __future__ import annotations

import numpy as np

from src.bots import SimulatedHumanBots
from src.config import BotConfig, GameConfig
from src.utils import sigmoid


def test_bot_table3_first_and_later_round_formula():
    cfg = GameConfig(n_players=3)
    bot_cfg = BotConfig()
    bots = SimulatedHumanBots(3, bot_cfg, cfg)
    bots.theta = np.array([0.0, 1.0, -1.0])
    adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    prev = np.array([1, 0, 1], dtype=int)
    first = bots.cooperation_probabilities(1, adjacency, prev, np.ones(3))
    assert np.allclose(first, sigmoid(bot_cfg.beta0_first + bot_cfg.beta1_first * bots.theta))
    degree = adjacency.sum(axis=1)
    coop_neighbors = adjacency @ prev
    frac = coop_neighbors / degree
    later_expected = sigmoid(bot_cfg.beta0_later + bot_cfg.beta_size * degree + bot_cfg.beta_num_coop * coop_neighbors + bot_cfg.beta_frac_coop * frac + bots.theta)
    later = bots.cooperation_probabilities(2, adjacency, prev, np.ones(3))
    assert np.allclose(later, later_expected)

