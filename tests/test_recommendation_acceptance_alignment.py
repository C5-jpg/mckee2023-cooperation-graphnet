from __future__ import annotations

from src.bots import SimulatedHumanBots
from src.config import BotConfig, GameConfig


def test_phi_mapping_for_recommendation_acceptance():
    bot_cfg = BotConfig()
    bots = SimulatedHumanBots(2, bot_cfg, GameConfig(n_players=2))
    assert bots.acceptance_probability(1, 1) == bot_cfg.phi_add_cooperator
    assert bots.acceptance_probability(1, 0) == bot_cfg.phi_add_defector
    assert bots.acceptance_probability(-1, 1) == bot_cfg.phi_delete_cooperator
    assert bots.acceptance_probability(-1, 0) == bot_cfg.phi_delete_defector

