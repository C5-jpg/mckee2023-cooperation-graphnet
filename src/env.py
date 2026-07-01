from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from .bots import SimulatedHumanBots
from .config import GameConfig
from .utils import all_undirected_edges


class PlannerProtocol(Protocol):
    name: str

    def reset_episode(self, state: "GameState", rng: np.random.Generator) -> None:
        ...

    def recommend(self, state: "GameState", rng: np.random.Generator) -> np.ndarray:
        ...


@dataclass
class GameState:
    round_index: int
    adjacency: np.ndarray
    capital: np.ndarray
    prev_actions: np.ndarray
    seed: int
    done: bool = False

    def copy(self) -> "GameState":
        return GameState(
            round_index=self.round_index,
            adjacency=self.adjacency.copy(),
            capital=self.capital.copy(),
            prev_actions=self.prev_actions.copy(),
            seed=self.seed,
            done=self.done,
        )


@dataclass
class CooperationTransition:
    planner_state: GameState
    actions: np.ndarray
    probabilities: np.ndarray
    constrained: np.ndarray
    payoffs: np.ndarray


@dataclass
class AdjudicationResult:
    next_adjacency: np.ndarray
    recommendation_records: List[Dict[str, Any]]
    rejected_nonzero: int
    nonzero_recommendations: int


@dataclass
class StepResult:
    next_state: GameState
    reward: float
    round_record: Dict[str, Any]
    player_records: List[Dict[str, Any]]
    edge_records: List[Dict[str, Any]]
    recommendation_records: List[Dict[str, Any]]
    done: bool


class CooperativeNetworkGame:
    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config or GameConfig()
        self.edges: list[tuple[int, int]] = all_undirected_edges(self.config.n_players)
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}

    def reset(self, seed: int) -> GameState:
        rng = np.random.default_rng(seed)
        n = self.config.n_players
        adjacency = np.zeros((n, n), dtype=bool)
        for i, j in self.edges:
            if rng.random() < self.config.initial_graph_p:
                adjacency[i, j] = True
                adjacency[j, i] = True
        capital = np.full(n, self.config.initial_capital, dtype=np.float64)
        prev_actions = np.zeros(n, dtype=np.int64)
        return GameState(round_index=1, adjacency=adjacency, capital=capital, prev_actions=prev_actions, seed=seed)

    def cooperation_transition(
        self,
        state: GameState,
        bots: SimulatedHumanBots,
        rng: np.random.Generator,
    ) -> CooperationTransition:
        decision = bots.choose_cooperation(
            state.round_index,
            state.adjacency,
            state.prev_actions,
            state.capital,
            rng,
        )
        actions = decision.actions
        degree = state.adjacency.sum(axis=1).astype(np.float64)
        cooperating_neighbors = state.adjacency @ actions.astype(np.float64)
        payoffs = self.config.benefit_b * cooperating_neighbors - self.config.cost_c * actions * degree
        new_capital = state.capital + payoffs
        planner_state = GameState(
            round_index=state.round_index,
            adjacency=state.adjacency.copy(),
            capital=new_capital,
            prev_actions=actions.copy(),
            seed=state.seed,
        )
        return CooperationTransition(
            planner_state=planner_state,
            actions=actions,
            probabilities=decision.probabilities,
            constrained=decision.constrained,
            payoffs=payoffs,
        )

    def normalize_recommendations(self, recommendations: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        recs = np.asarray(recommendations, dtype=np.int64).reshape(-1).copy()
        if recs.size != len(self.edges):
            raise ValueError(f"expected {len(self.edges)} recommendations, got {recs.size}")
        recs = np.clip(recs, -1, 1)
        for idx, (i, j) in enumerate(self.edges):
            if recs[idx] == 1 and adjacency[i, j]:
                recs[idx] = 0
            elif recs[idx] == -1 and not adjacency[i, j]:
                recs[idx] = 0
        return recs

    def binary_changes_to_recommendations(self, binary_actions: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        binary = np.asarray(binary_actions, dtype=np.int64).reshape(-1)
        if binary.size != len(self.edges):
            raise ValueError(f"expected {len(self.edges)} binary actions, got {binary.size}")
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            if binary[idx] == 1:
                recs[idx] = -1 if adjacency[i, j] else 1
        return recs

    def target_edge_states_to_recommendations(self, target_active: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Map target edge states to legal add/delete/no-op recommendations.

        GraphNet's two edge logits are interpreted as the desired next edge
        state: 0=inactive and 1=active. Given the current graph, this maps to
        add, delete, or no-op without requiring a separate three-class action.
        """

        target = np.asarray(target_active, dtype=np.int64).reshape(-1)
        if target.size != len(self.edges):
            raise ValueError(f"expected {len(self.edges)} target edge states, got {target.size}")
        target = np.clip(target, 0, 1)
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            current = bool(adjacency[i, j])
            desired = bool(target[idx])
            if desired and not current:
                recs[idx] = 1
            elif current and not desired:
                recs[idx] = -1
        return recs

    def adjudicate_recommendations(
        self,
        planner_state: GameState,
        recommendations: np.ndarray,
        bots: SimulatedHumanBots,
        rng: np.random.Generator,
        episode_id: int = 0,
        planner_name: str = "unknown",
    ) -> AdjudicationResult:
        recs = self.normalize_recommendations(recommendations, planner_state.adjacency)
        next_adjacency = planner_state.adjacency.copy()
        records: list[dict[str, Any]] = []
        rejected = 0
        nonzero = 0
        for idx, (i, j) in enumerate(self.edges):
            rec = int(recs[idx])
            if rec == 0:
                continue
            nonzero += 1
            assigned = int(i if rng.random() < 0.5 else j)
            referent = int(j if assigned == i else i)
            referent_coop = int(planner_state.prev_actions[referent])
            accepted = bots.accept_recommendation(rec, referent_coop, rng)
            before = bool(planner_state.adjacency[i, j])
            if accepted:
                if rec == 1:
                    next_adjacency[i, j] = True
                    next_adjacency[j, i] = True
                elif rec == -1:
                    next_adjacency[i, j] = False
                    next_adjacency[j, i] = False
            else:
                rejected += 1
            records.append(
                {
                    "episode_id": episode_id,
                    "round": planner_state.round_index,
                    "planner": planner_name,
                    "i": i,
                    "j": j,
                    "recommendation": rec,
                    "assigned_player": assigned,
                    "referent_player": referent,
                    "referent_cooperated": referent_coop,
                    "accepted": bool(accepted),
                    "edge_active_before": before,
                    "edge_active_after": bool(next_adjacency[i, j]),
                }
            )
        return AdjudicationResult(next_adjacency=next_adjacency, recommendation_records=records, rejected_nonzero=rejected, nonzero_recommendations=nonzero)

    def step(
        self,
        state: GameState,
        planner: PlannerProtocol,
        bots: SimulatedHumanBots,
        rng: np.random.Generator,
        episode_id: int = 0,
    ) -> StepResult:
        transition = self.cooperation_transition(state, bots, rng)
        planner_state = transition.planner_state
        if state.round_index < self.config.episode_length:
            recommendations = planner.recommend(planner_state, rng)
            adjudication = self.adjudicate_recommendations(
                planner_state,
                recommendations,
                bots,
                rng,
                episode_id=episode_id,
                planner_name=getattr(planner, "name", planner.__class__.__name__),
            )
        else:
            adjudication = AdjudicationResult(
                next_adjacency=planner_state.adjacency.copy(),
                recommendation_records=[],
                rejected_nonzero=0,
                nonzero_recommendations=0,
            )
        penalty = adjudication.rejected_nonzero / max(1, self.config.n_possible_edges)
        reward = float(np.mean(planner_state.capital) - self.config.recommendation_penalty_weight * penalty)
        done = state.round_index >= self.config.episode_length
        next_state = GameState(
            round_index=state.round_index + 1,
            adjacency=adjudication.next_adjacency,
            capital=planner_state.capital.copy(),
            prev_actions=transition.actions.copy(),
            seed=state.seed,
            done=done,
        )
        round_record = self._round_record(episode_id, planner, planner_state, reward, penalty, adjudication)
        player_records = self._player_records(episode_id, planner, state, planner_state, transition)
        edge_records = self._edge_records(episode_id, planner, planner_state, adjudication.next_adjacency)
        return StepResult(
            next_state=next_state,
            reward=reward,
            round_record=round_record,
            player_records=player_records,
            edge_records=edge_records,
            recommendation_records=adjudication.recommendation_records,
            done=done,
        )

    def run_episode(
        self,
        planner: PlannerProtocol,
        bots: SimulatedHumanBots,
        seed: int,
        episode_id: int = 0,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        state = self.reset(seed)
        bots.reset_episode(rng)
        if hasattr(planner, "reset_episode"):
            planner.reset_episode(state, rng)
        rounds: list[dict[str, Any]] = []
        players: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []
        rewards: list[float] = []
        while not state.done:
            result = self.step(state, planner, bots, rng, episode_id=episode_id)
            rounds.append(result.round_record)
            players.extend(result.player_records)
            edges.extend(result.edge_records)
            recommendations.extend(result.recommendation_records)
            rewards.append(result.reward)
            state = result.next_state
        return {
            "episode_id": episode_id,
            "planner": getattr(planner, "name", planner.__class__.__name__),
            "seed": seed,
            "rounds": rounds,
            "players": players,
            "edges": edges,
            "recommendations": recommendations,
            "final_state": state,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        }

    def _round_record(
        self,
        episode_id: int,
        planner: PlannerProtocol,
        planner_state: GameState,
        reward: float,
        penalty: float,
        adjudication: AdjudicationResult,
    ) -> Dict[str, Any]:
        adjacency = adjudication.next_adjacency
        return {
            "episode_id": episode_id,
            "round": planner_state.round_index,
            "planner": getattr(planner, "name", planner.__class__.__name__),
            "seed": planner_state.seed,
            "mean_cooperation": float(np.mean(planner_state.prev_actions)),
            "mean_capital": float(np.mean(planner_state.capital)),
            "network_density": float(adjacency.sum() / (self.config.n_players * (self.config.n_players - 1))),
            "reward": reward,
            "recommendation_penalty": penalty,
            "nonzero_recommendations": adjudication.nonzero_recommendations,
            "rejected_recommendations": adjudication.rejected_nonzero,
        }

    def _player_records(
        self,
        episode_id: int,
        planner: PlannerProtocol,
        prev_state: GameState,
        planner_state: GameState,
        transition: CooperationTransition,
    ) -> List[Dict[str, Any]]:
        degree = planner_state.adjacency.sum(axis=1)
        return [
            {
                "episode_id": episode_id,
                "round": planner_state.round_index,
                "planner": getattr(planner, "name", planner.__class__.__name__),
                "seed": planner_state.seed,
                "player_id": i,
                "capital": float(planner_state.capital[i]),
                "degree": int(degree[i]),
                "cooperated": int(transition.actions[i]),
                "prev_cooperated": int(prev_state.prev_actions[i]),
                "is_core_optional": False,
                "cooperation_probability": float(transition.probabilities[i]),
                "capital_constrained": bool(transition.constrained[i]),
                "payoff": float(transition.payoffs[i]),
            }
            for i in range(self.config.n_players)
        ]

    def _edge_records(
        self,
        episode_id: int,
        planner: PlannerProtocol,
        planner_state: GameState,
        next_adjacency: np.ndarray,
    ) -> List[Dict[str, Any]]:
        records = []
        actions = planner_state.prev_actions
        for i, j in self.edges:
            pair_type = "cooperate_cooperate" if actions[i] and actions[j] else "defect_defect" if not actions[i] and not actions[j] else "cooperate_defect"
            records.append(
                {
                    "episode_id": episode_id,
                    "round": planner_state.round_index,
                    "planner": getattr(planner, "name", planner.__class__.__name__),
                    "seed": planner_state.seed,
                    "i": i,
                    "j": j,
                    "edge_active_before": bool(planner_state.adjacency[i, j]),
                    "edge_active_after": bool(next_adjacency[i, j]),
                    "edge_active": bool(next_adjacency[i, j]),
                    "cooperation_pair_type": pair_type,
                }
            )
        return records
