from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ENCOURAGEMENT_TABLES, NEUTRAL_TABLE, GameConfig, validate_action_semantics
from .env import GameState
from .utils import all_undirected_edges


class BasePlanner:
    name = "base"

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig()
        self.edges = all_undirected_edges(self.config.n_players)

    def reset_episode(self, state: GameState, rng: np.random.Generator) -> None:
        return None

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _change_recommendation(self, idx: int, adjacency: np.ndarray) -> int:
        i, j = self.edges[idx]
        return -1 if adjacency[i, j] else 1


class StaticPlanner(BasePlanner):
    name = "static"

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(len(self.edges), dtype=np.int64)


class RandomRecommendationPlanner(BasePlanner):
    name = "random"

    def __init__(self, config: Optional[GameConfig] = None, change_probability: float = 0.30) -> None:
        super().__init__(config)
        self.change_probability = change_probability

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        recs = np.zeros(len(self.edges), dtype=np.int64)
        mask = rng.random(len(self.edges)) < self.change_probability
        for idx in np.where(mask)[0]:
            recs[idx] = self._change_recommendation(int(idx), state.adjacency)
        return recs


class CooperativeClusteringPlanner(BasePlanner):
    name = "cooperative_clustering"

    def __init__(self, config: Optional[GameConfig] = None, n_focal: int = 5, random_fraction: float = 0.05) -> None:
        super().__init__(config)
        self.n_focal = n_focal
        self.random_fraction = random_fraction
        self.focal_players: list[int] = []

    def reset_episode(self, state: GameState, rng: np.random.Generator) -> None:
        self.focal_players = rng.choice(self.config.n_players, size=self.n_focal, replace=False).astype(int).tolist()

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        if not self.focal_players:
            self.reset_episode(state, rng)
        recs = np.zeros(len(self.edges), dtype=np.int64)
        next_focal = list(self.focal_players)
        edge_lookup = {edge: idx for idx, edge in enumerate(self.edges)}
        actions = state.prev_actions
        for focal in list(self.focal_players):
            neighbors = np.where(state.adjacency[focal])[0]
            if actions[focal] == 1:
                defecting_neighbors = [int(j) for j in neighbors if actions[j] == 0]
                if defecting_neighbors:
                    j = int(rng.choice(defecting_neighbors))
                    edge = (min(focal, j), max(focal, j))
                    recs[edge_lookup[edge]] = -1
                else:
                    candidates = [
                        j
                        for j in range(self.config.n_players)
                        if j != focal and actions[j] == 1 and not state.adjacency[focal, j]
                    ]
                    if candidates:
                        j = int(rng.choice(candidates))
                        edge = (min(focal, j), max(focal, j))
                        recs[edge_lookup[edge]] = 1
            else:
                candidates = [j for j in range(self.config.n_players) if j not in next_focal and actions[j] == 1]
                if candidates:
                    replacement = int(rng.choice(candidates))
                    if focal in next_focal:
                        next_focal[next_focal.index(focal)] = replacement
        self.focal_players = next_focal
        random_count = max(1, int(round(self.random_fraction * len(self.edges))))
        random_indices = rng.choice(len(self.edges), size=random_count, replace=False)
        for idx in random_indices:
            if recs[int(idx)] == 0:
                recs[int(idx)] = self._change_recommendation(int(idx), state.adjacency)
        return recs


class EncouragementPlanner(BasePlanner):
    name = "encouragement"

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        recs = np.zeros(len(self.edges), dtype=np.int64)
        table_idx = min(max(state.round_index, 1), 14) - 1
        for idx, (i, j) in enumerate(self.edges):
            key = "cc" if state.prev_actions[i] and state.prev_actions[j] else "dd" if not state.prev_actions[i] and not state.prev_actions[j] else "cd"
            if state.adjacency[i, j]:
                prob = ENCOURAGEMENT_TABLES[key]["delete"][table_idx]
                if rng.random() < prob:
                    recs[idx] = -1
            else:
                prob = ENCOURAGEMENT_TABLES[key]["add"][table_idx]
                if rng.random() < prob:
                    recs[idx] = 1
        return recs


class NeutralPlanner(BasePlanner):
    name = "neutral"

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        recs = np.zeros(len(self.edges), dtype=np.int64)
        table_idx = min(max(state.round_index, 1), 14) - 1
        for idx, (i, j) in enumerate(self.edges):
            if state.adjacency[i, j]:
                if rng.random() < NEUTRAL_TABLE["delete"][table_idx]:
                    recs[idx] = -1
            else:
                if rng.random() < NEUTRAL_TABLE["add"][table_idx]:
                    recs[idx] = 1
        return recs


class MaximumConnectivityPlanner(BasePlanner):
    name = "maximum_connectivity"

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            if not state.adjacency[i, j]:
                recs[idx] = 1
        return recs


@dataclass
class GraphNetPlanner(BasePlanner):
    model: object = field(default=None)
    device: str = "cpu"
    deterministic: bool = True
    action_semantics: str = "target_state"
    config: GameConfig = field(default_factory=GameConfig)
    name: str = "graphnet"

    def __post_init__(self) -> None:
        validate_action_semantics(self.action_semantics)
        self.edges = all_undirected_edges(self.config.n_players)

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        import torch

        from .graphnet import build_graph_batch

        if self.model is None:
            return np.zeros(len(self.edges), dtype=np.int64)
        self.model.eval()
        with torch.no_grad():
            batch = build_graph_batch([state], self.config, device=self.device, feature_mode=getattr(self.model, "feature_mode", "augmented"))
            logits, _ = self.model(batch)
            if self.deterministic:
                actions = logits.argmax(dim=-1)[0].cpu().numpy()
            else:
                dist = torch.distributions.Categorical(logits=logits[0])
                actions = dist.sample().cpu().numpy()
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            if self.action_semantics == "change":
                if int(actions[idx]) == 1:
                    recs[idx] = -1 if state.adjacency[i, j] else 1
            else:
                desired_active = bool(int(actions[idx]))
                current_active = bool(state.adjacency[i, j])
                if desired_active and not current_active:
                    recs[idx] = 1
                elif current_active and not desired_active:
                    recs[idx] = -1
        return recs


def all_rule_based_planners(config: Optional[GameConfig] = None) -> list[BasePlanner]:
    cfg = config or GameConfig()
    return [
        StaticPlanner(cfg),
        RandomRecommendationPlanner(cfg),
        CooperativeClusteringPlanner(cfg),
        EncouragementPlanner(cfg),
        NeutralPlanner(cfg),
        MaximumConnectivityPlanner(cfg),
    ]
