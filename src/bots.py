from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import BotConfig, GameConfig
from .utils import sigmoid


@dataclass
class BotDecisionInfo:
    probabilities: np.ndarray
    actions: np.ndarray
    constrained: np.ndarray


class SimulatedHumanBots:
    """Hard-coded bot model fit from baseline human behavior."""

    def __init__(self, n_players: int, config: Optional[BotConfig] = None, game_config: Optional[GameConfig] = None) -> None:
        self.n_players = n_players
        self.config = config or BotConfig()
        self.game_config = game_config or GameConfig(n_players=n_players)
        self.theta = np.zeros(n_players, dtype=np.float64)

    def reset_episode(self, rng: np.random.Generator) -> None:
        self.theta = rng.normal(self.config.mu_theta, self.config.sigma_theta, size=self.n_players)

    def cooperation_probabilities(
        self,
        round_index: int,
        adjacency: np.ndarray,
        prev_actions: np.ndarray,
        capital: np.ndarray | None = None,
    ) -> np.ndarray:
        if round_index <= 1:
            logits = self.config.beta0_first + self.config.beta1_first * self.theta
        else:
            degree = adjacency.sum(axis=1).astype(np.float64)
            cooperating_neighbors = adjacency @ prev_actions.astype(np.float64)
            frac = np.divide(
                cooperating_neighbors,
                degree,
                out=np.zeros_like(cooperating_neighbors, dtype=np.float64),
                where=degree > 0,
            )
            logits = (
                self.config.beta0_later
                + self.config.beta_size * degree
                + self.config.beta_num_coop * cooperating_neighbors
                + self.config.beta_frac_coop * frac
                + self.theta
            )
        probs = sigmoid(logits)
        return np.asarray(probs, dtype=np.float64)

    def choose_cooperation(
        self,
        round_index: int,
        adjacency: np.ndarray,
        prev_actions: np.ndarray,
        capital: np.ndarray,
        rng: np.random.Generator,
    ) -> BotDecisionInfo:
        probs = self.cooperation_probabilities(round_index, adjacency, prev_actions, capital)
        sampled = (rng.random(self.n_players) < probs).astype(np.int64)
        degree = adjacency.sum(axis=1)
        required_cost = self.game_config.cost_c * degree
        constrained = (sampled == 1) & (required_cost > capital + 1e-12)
        sampled[constrained] = 0
        return BotDecisionInfo(probabilities=probs, actions=sampled, constrained=constrained)

    def acceptance_probability(self, recommendation: int, referent_cooperated: int) -> float:
        if recommendation == -1 and referent_cooperated == 0:
            return self.config.phi_delete_defector
        if recommendation == -1 and referent_cooperated == 1:
            return self.config.phi_delete_cooperator
        if recommendation == 1 and referent_cooperated == 0:
            return self.config.phi_add_defector
        if recommendation == 1 and referent_cooperated == 1:
            return self.config.phi_add_cooperator
        return 0.0

    def accept_recommendation(self, recommendation: int, referent_cooperated: int, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.acceptance_probability(recommendation, referent_cooperated))

