from __future__ import annotations

import numpy as np

from src.bots import SimulatedHumanBots
from src.config import BotConfig, GameConfig
from src.utils import sigmoid


def test_disposition_sampling_is_deterministic():
    bots1 = SimulatedHumanBots(4, BotConfig(), GameConfig(n_players=4))
    bots2 = SimulatedHumanBots(4, BotConfig(), GameConfig(n_players=4))
    bots1.reset_episode(np.random.default_rng(123))
    bots2.reset_episode(np.random.default_rng(123))
    assert np.allclose(bots1.theta, bots2.theta)


def test_first_round_probability_formula():
    cfg = BotConfig()
    bots = SimulatedHumanBots(2, cfg, GameConfig(n_players=2))
    bots.theta = np.array([0.0, 1.0])
    probs = bots.cooperation_probabilities(1, np.zeros((2, 2), dtype=bool), np.zeros(2, dtype=int))
    expected = sigmoid(cfg.beta0_first + cfg.beta1_first * bots.theta)
    assert np.allclose(probs, expected)


def test_later_round_probability_formula():
    cfg = BotConfig()
    bots = SimulatedHumanBots(3, cfg, GameConfig(n_players=3))
    bots.theta = np.array([0.2, 0.0, 0.0])
    adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    prev = np.array([0, 1, 0])
    probs = bots.cooperation_probabilities(2, adjacency, prev)
    degree = 2
    coop_neighbors = 1
    frac = 0.5
    expected0 = sigmoid(cfg.beta0_later + cfg.beta_size * degree + cfg.beta_num_coop * coop_neighbors + cfg.beta_frac_coop * frac + 0.2)
    assert np.isclose(probs[0], expected0)


def test_acceptance_probabilities_match_table():
    bots = SimulatedHumanBots(2)
    assert bots.acceptance_probability(-1, 0) == 0.774
    assert bots.acceptance_probability(-1, 1) == 0.085
    assert bots.acceptance_probability(1, 0) == 0.287
    assert bots.acceptance_probability(1, 1) == 0.909

