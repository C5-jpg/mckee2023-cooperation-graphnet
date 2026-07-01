from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn

from .config import GameConfig, GraphNetConfig, validate_feature_mode
from .env import GameState
from .utils import all_undirected_edges


@dataclass
class GraphBatch:
    global_attr: torch.Tensor
    node_attr: torch.Tensor
    edge_attr: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    directed_to_undirected: torch.Tensor
    adjacency: torch.Tensor
    feature_mode: str


def directed_edge_index(config: GameConfig | None = None) -> tuple[list[tuple[int, int]], list[int]]:
    cfg = config or GameConfig()
    directed_edges: list[tuple[int, int]] = []
    directed_to_undirected: list[int] = []
    for edge_idx, (i, j) in enumerate(all_undirected_edges(cfg.n_players)):
        directed_edges.append((i, j))
        directed_edges.append((j, i))
        directed_to_undirected.extend([edge_idx, edge_idx])
    return directed_edges, directed_to_undirected


def graph_feature_dimensions(feature_mode: str) -> tuple[int, int, int]:
    mode = validate_feature_mode(feature_mode)
    if mode == "paper_minimal":
        return 2, 1, 1
    return 3, 1, 4


def build_graph_batch(
    states: List[GameState],
    config: GameConfig | None = None,
    device: str | torch.device = "cpu",
    feature_mode: str = "augmented",
) -> GraphBatch:
    cfg = config or GameConfig()
    mode = validate_feature_mode(feature_mode)
    undirected_edges = all_undirected_edges(cfg.n_players)
    directed_edges, directed_to_undirected = directed_edge_index(cfg)
    global_rows = []
    node_rows = []
    edge_rows = []
    adj_rows = []
    for state in states:
        edge_features = np.asarray([[1.0 if state.adjacency[i, j] else 0.0] for i, j in directed_edges], dtype=np.float32)
        if mode == "paper_minimal":
            node_features = np.stack(
                [
                    state.capital.astype(np.float32),
                    state.prev_actions.astype(np.float32),
                ],
                axis=1,
            )
            global_features = np.asarray([1.0], dtype=np.float32)
        else:
            degree = state.adjacency.sum(axis=1).astype(np.float32)
            density = state.adjacency.sum() / (cfg.n_players * (cfg.n_players - 1))
            node_features = np.stack(
                [
                    state.capital.astype(np.float32),
                    state.prev_actions.astype(np.float32),
                    degree / max(1, cfg.n_players - 1),
                ],
                axis=1,
            )
            global_features = np.asarray(
                [
                    state.round_index / cfg.episode_length,
                    float(np.mean(state.capital)),
                    float(np.mean(state.prev_actions)),
                    float(density),
                ],
                dtype=np.float32,
            )
        node_rows.append(node_features)
        edge_rows.append(edge_features)
        global_rows.append(global_features)
        adj_rows.append(np.asarray([1.0 if state.adjacency[i, j] else 0.0 for i, j in undirected_edges], dtype=np.float32))
    senders = torch.tensor([i for i, _ in directed_edges], dtype=torch.long, device=device)
    receivers = torch.tensor([j for _, j in directed_edges], dtype=torch.long, device=device)
    return GraphBatch(
        global_attr=torch.tensor(np.stack(global_rows), dtype=torch.float32, device=device),
        node_attr=torch.tensor(np.stack(node_rows), dtype=torch.float32, device=device),
        edge_attr=torch.tensor(np.stack(edge_rows), dtype=torch.float32, device=device),
        senders=senders,
        receivers=receivers,
        directed_to_undirected=torch.tensor(directed_to_undirected, dtype=torch.long, device=device),
        adjacency=torch.tensor(np.stack(adj_rows), dtype=torch.float32, device=device),
        feature_mode=mode,
    )


class SingleLayerMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh() if activation else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = max(1, self.linear.in_features)
        std = 1.0 / np.sqrt(fan_in)
        nn.init.trunc_normal_(self.linear.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class GraphNet(nn.Module):
    def __init__(
        self,
        game_config: GameConfig | None = None,
        graphnet_config: GraphNetConfig | None = None,
        node_in: int | None = None,
        edge_in: int | None = None,
        global_in: int | None = None,
        feature_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.game_config = game_config or GameConfig()
        self.graphnet_config = graphnet_config or GraphNetConfig()
        self.feature_mode = validate_feature_mode(feature_mode or self.graphnet_config.feature_mode)
        default_node_in, default_edge_in, default_global_in = graph_feature_dimensions(self.feature_mode)
        node_in = default_node_in if node_in is None else node_in
        edge_in = default_edge_in if edge_in is None else edge_in
        global_in = default_global_in if global_in is None else global_in
        h_edge = self.graphnet_config.edge_update_1_output_size
        h_node = self.graphnet_config.node_update_1_output_size
        h_global = self.graphnet_config.global_update_1_output_size
        edge2 = self.graphnet_config.edge_update_2_output_size
        node2 = self.graphnet_config.node_update_2_output_size
        self.phi_e1 = SingleLayerMLP(edge_in + 2 * node_in + global_in, h_edge, activation=True)
        self.phi_v1 = SingleLayerMLP(h_edge + node_in + global_in, h_node, activation=True)
        self.phi_u1 = SingleLayerMLP(h_edge + h_node + global_in, h_global, activation=True)
        self.phi_e2 = SingleLayerMLP(h_edge + 2 * h_node + h_global, edge2, activation=False)
        self.phi_v2 = SingleLayerMLP(h_node + h_global, node2, activation=True)
        self.phi_u2 = SingleLayerMLP(edge2 + node2 + h_global, 1, activation=False)

    def _aggregate_to_nodes(self, edge_values: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor, n_nodes: int) -> torch.Tensor:
        bsz, _, feat = edge_values.shape
        out = edge_values.new_zeros((bsz, n_nodes, feat))
        receiver_idx = receivers.view(1, -1, 1).expand(bsz, -1, feat)
        out.scatter_add_(1, receiver_idx, edge_values)
        return out

    def _aggregate_directed_logits_to_undirected(self, directed_logits: torch.Tensor, directed_to_undirected: torch.Tensor) -> torch.Tensor:
        bsz, _, feat = directed_logits.shape
        n_undirected = self.game_config.n_possible_edges
        out = directed_logits.new_zeros((bsz, n_undirected, feat))
        counts = directed_logits.new_zeros((bsz, n_undirected, 1))
        edge_idx = directed_to_undirected.view(1, -1, 1).expand(bsz, -1, feat)
        count_idx = directed_to_undirected.view(1, -1, 1).expand(bsz, -1, 1)
        out.scatter_add_(1, edge_idx, directed_logits)
        counts.scatter_add_(1, count_idx, torch.ones((bsz, directed_logits.shape[1], 1), dtype=directed_logits.dtype, device=directed_logits.device))
        return out / counts.clamp_min(1.0)

    def forward(self, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        u = batch.global_attr
        v = batch.node_attr
        e = batch.edge_attr
        bsz, n_edges, _ = e.shape
        n_nodes = v.shape[1]
        sender_v = v[:, batch.senders, :]
        receiver_v = v[:, batch.receivers, :]
        u_edges = u[:, None, :].expand(-1, n_edges, -1)
        e1 = self.phi_e1(torch.cat([e, sender_v, receiver_v, u_edges], dim=-1))
        node_agg = self._aggregate_to_nodes(e1, batch.senders, batch.receivers, n_nodes)
        u_nodes = u[:, None, :].expand(-1, n_nodes, -1)
        v1 = self.phi_v1(torch.cat([node_agg, v, u_nodes], dim=-1))
        e1_sum = e1.sum(dim=1)
        v1_sum = v1.sum(dim=1)
        u1 = self.phi_u1(torch.cat([e1_sum, v1_sum, u], dim=-1))
        sender_v1 = v1[:, batch.senders, :]
        receiver_v1 = v1[:, batch.receivers, :]
        u1_edges = u1[:, None, :].expand(-1, n_edges, -1)
        directed_logits = self.phi_e2(torch.cat([e1, sender_v1, receiver_v1, u1_edges], dim=-1))
        logits = self._aggregate_directed_logits_to_undirected(directed_logits, batch.directed_to_undirected)
        u1_nodes = u1[:, None, :].expand(-1, n_nodes, -1)
        v2 = self.phi_v2(torch.cat([v1, u1_nodes], dim=-1))
        value = self.phi_u2(torch.cat([directed_logits.sum(dim=1), v2.sum(dim=1), u1], dim=-1)).squeeze(-1)
        return logits, value


def binary_actions_to_recommendations(binary_actions: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Backward-compatible map for the old 1=change action semantics."""

    return torch.where(binary_actions.long() == 1, torch.where(adjacency > 0.5, -torch.ones_like(binary_actions), torch.ones_like(binary_actions)), torch.zeros_like(binary_actions))


def target_edge_states_to_recommendations(target_active: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Map 0=target inactive, 1=target active to add/delete/no-op."""

    target = target_active.long()
    current = adjacency > 0.5
    add = (target == 1) & (~current)
    delete = (target == 0) & current
    return torch.where(add, torch.ones_like(target), torch.where(delete, -torch.ones_like(target), torch.zeros_like(target)))
