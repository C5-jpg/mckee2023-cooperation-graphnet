from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .a2c import load_checkpoint
from .config import GameConfig, GraphNetConfig, validate_action_semantics, validate_feature_mode
from .env import GameState
from .graphnet import GraphNet, build_graph_batch
from .planners import BasePlanner
from .utils import all_undirected_edges


TABLE6_ORACLE_RATES: dict[tuple[str, str], float] = {
    ("C-C", "add"): 0.994,
    ("C-C", "delete"): 0.030,
    ("C-D", "add"): 0.582,
    ("C-D", "delete"): 0.503,
    ("D-D", "add"): 0.000,
    ("D-D", "delete"): 1.000,
}


def pair_type_for_actions(a_i: int, a_j: int) -> str:
    if a_i == 1 and a_j == 1:
        return "C-C"
    if a_i == 0 and a_j == 0:
        return "D-D"
    return "C-D"


class Table6OraclePlanner(BasePlanner):
    """Diagnostic-only planner that samples recommendations from Table 6 rates.

    This planner is not a training target and must not be used as supervised
    labels for GraphNet. It is only a root-cause probe for whether the current
    simulation and reward landscape support Table 6-like recommendations.
    """

    name = "table6_oracle"

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            pair = pair_type_for_actions(int(state.prev_actions[i]), int(state.prev_actions[j]))
            change_type = "delete" if state.adjacency[i, j] else "add"
            prob = TABLE6_ORACLE_RATES[(pair, change_type)]
            if rng.random() < prob:
                recs[idx] = -1 if change_type == "delete" else 1
        return recs


class DDDeleteOraclePlanner(BasePlanner):
    """Diagnostic-only planner that deletes active D-D edges and does nothing else."""

    name = "dd_delete_oracle"

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            if state.adjacency[i, j] and state.prev_actions[i] == 0 and state.prev_actions[j] == 0:
                recs[idx] = -1
        return recs


class NoDeleteHighDensityPlanner(BasePlanner):
    """Diagnostic high-density planner that never deletes active edges.

    Inactive C-C and C-D opportunities are frequently added. This isolates
    whether dense networks remain rewarding even when D-D deletion is absent.
    """

    name = "no_delete_high_density"

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        cc_add_probability: float = 0.95,
        cd_add_probability: float = 0.75,
        dd_add_probability: float = 0.20,
    ) -> None:
        super().__init__(config)
        self.cc_add_probability = cc_add_probability
        self.cd_add_probability = cd_add_probability
        self.dd_add_probability = dd_add_probability

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            if state.adjacency[i, j]:
                continue
            pair = pair_type_for_actions(int(state.prev_actions[i]), int(state.prev_actions[j]))
            prob = {
                "C-C": self.cc_add_probability,
                "C-D": self.cd_add_probability,
                "D-D": self.dd_add_probability,
            }[pair]
            if rng.random() < prob:
                recs[idx] = 1
        return recs


@dataclass
class CheckpointGraphNetPlanner(BasePlanner):
    """GraphNet planner with explicit old/new action semantics for diagnostics."""

    checkpoint_path: str | Path
    device: str = "cpu"
    deterministic: bool = True
    action_semantics: str = "target_state"
    feature_mode: str = "augmented"
    config: GameConfig = field(default_factory=GameConfig)
    name: str = "graphnet_checkpoint"

    def __post_init__(self) -> None:
        validate_action_semantics(self.action_semantics)
        validate_feature_mode(self.feature_mode)
        self.edges = all_undirected_edges(self.config.n_players)
        self.model = GraphNet(self.config, GraphNetConfig(feature_mode=self.feature_mode)).to(self.device)
        load_checkpoint(self.checkpoint_path, self.model, optimizer=None, map_location=self.device)
        self.model.eval()

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        import torch

        with torch.no_grad():
            batch = build_graph_batch([state], self.config, device=self.device, feature_mode=self.feature_mode)
            logits, _ = self.model(batch)
            if self.deterministic:
                actions = logits.argmax(dim=-1)[0].detach().cpu().numpy()
            else:
                dist = torch.distributions.Categorical(logits=logits[0])
                actions = dist.sample().detach().cpu().numpy()
        if self.action_semantics == "change":
            recs = np.zeros(len(self.edges), dtype=np.int64)
            for idx, (i, j) in enumerate(self.edges):
                if int(actions[idx]) == 1:
                    recs[idx] = -1 if state.adjacency[i, j] else 1
            return recs
        if self.action_semantics == "target_state":
            recs = np.zeros(len(self.edges), dtype=np.int64)
            for idx, (i, j) in enumerate(self.edges):
                desired_active = bool(int(actions[idx]))
                current_active = bool(state.adjacency[i, j])
                if desired_active and not current_active:
                    recs[idx] = 1
                elif current_active and not desired_active:
                    recs[idx] = -1
            return recs
        raise ValueError(f"unknown action_semantics: {self.action_semantics!r}")
