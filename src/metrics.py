from __future__ import annotations

from typing import Dict

import numpy as np

from .config import GameConfig


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0.0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    return float((2.0 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def network_density(adjacency: np.ndarray) -> float:
    n = adjacency.shape[0]
    if n <= 1:
        return 0.0
    return float(adjacency.sum() / (n * (n - 1)))


def newman_assortativity(adjacency: np.ndarray, actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.int64)
    mixing = np.zeros((2, 2), dtype=np.float64)
    edge_count = 0
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                ai = int(actions[i])
                aj = int(actions[j])
                mixing[ai, aj] += 1.0
                mixing[aj, ai] += 1.0
                edge_count += 1
    if edge_count == 0:
        return 0.0
    mixing /= 2.0 * edge_count
    row = mixing.sum(axis=1)
    col = mixing.sum(axis=0)
    expected = float(row @ col)
    denom = 1.0 - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((np.trace(mixing) - expected) / denom)


def rand_assortativity(adjacency: np.ndarray, actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.int64)
    degree = adjacency.sum(axis=1).astype(np.float64)
    coop_neighbors = adjacency @ actions.astype(np.float64)
    frac = np.divide(coop_neighbors, degree, out=np.full_like(coop_neighbors, np.nan, dtype=np.float64), where=degree > 0)
    coop = actions.astype(bool)
    if not coop.any() or not (~coop).any():
        return float("nan")
    coop_vals = frac[coop]
    defect_vals = frac[~coop]
    if np.isnan(coop_vals).all() or np.isnan(defect_vals).all():
        return float("nan")
    return float(np.nanmean(coop_vals) - np.nanmean(defect_vals))


def shirado_assortativity(adjacency: np.ndarray, actions: np.ndarray) -> float:
    xs = []
    ys = []
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                xs.append(actions[i])
                ys.append(actions[j])
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def cooperator_defector_assortativity(adjacency: np.ndarray, actions: np.ndarray) -> float:
    return newman_assortativity(adjacency, actions)


def defector_disassortativity(adjacency: np.ndarray, actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.int64)
    n = actions.size
    active_edges = 0
    defect_defect_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                active_edges += 1
                if actions[i] == 0 and actions[j] == 0:
                    defect_defect_edges += 1
    if active_edges == 0 or n <= 1:
        return float("nan")
    n_defect = int(np.sum(actions == 0))
    expected = (n_defect / n) * ((n_defect - 1) / (n - 1)) if n_defect >= 2 else 0.0
    actual = defect_defect_edges / active_edges
    return float(expected - actual)


def relative_degree_cooperators(adjacency: np.ndarray, actions: np.ndarray) -> float:
    degree = adjacency.sum(axis=1).astype(np.float64)
    coop = actions.astype(bool)
    if not coop.any() or not (~coop).any():
        return float("nan")
    coop_degree = float(degree[coop].mean())
    defect_degree = float(degree[~coop].mean())
    return coop_degree - defect_degree


def core_periphery_approx_degree_concentration(adjacency: np.ndarray) -> float:
    """Approximate degree-concentration score in [0, 1]."""

    n = adjacency.shape[0]
    if n == 0:
        return 0.0
    degree = adjacency.sum(axis=1).astype(np.float64)
    if np.allclose(degree, 0.0):
        return 0.0
    order = np.argsort(-degree)
    core_size = max(1, n // 4)
    core = order[:core_size]
    periphery = order[core_size:]
    core_mean = degree[core].mean()
    periphery_mean = degree[periphery].mean() if periphery.size else 0.0
    return float(max(0.0, min(1.0, (core_mean - periphery_mean) / max(1.0, n - 1))))


def core_periphery_score(adjacency: np.ndarray) -> float:
    return core_periphery_approx_degree_concentration(adjacency)


def final_episode_summary(episode: dict, config: GameConfig | None = None) -> Dict[str, float]:
    final_state = episode["final_state"]
    rounds = episode["rounds"]
    final_round = rounds[-1] if rounds else {}
    actions = final_state.prev_actions
    adjacency = final_state.adjacency
    return {
        "mean_cooperation": float(np.mean([r["mean_cooperation"] for r in rounds])) if rounds else 0.0,
        "final_round_cooperation": float(final_round.get("mean_cooperation", 0.0)),
        "mean_capital": float(np.mean([r["mean_capital"] for r in rounds])) if rounds else 0.0,
        "final_mean_capital": float(np.mean(final_state.capital)),
        "gini_final": gini(final_state.capital),
        "network_density_final": network_density(adjacency),
        "cooperator_defector_assortativity_final": cooperator_defector_assortativity(adjacency, actions),
        "newman_assortativity_final": newman_assortativity(adjacency, actions),
        "rand_assortativity_final": rand_assortativity(adjacency, actions),
        "shirado_assortativity_final": shirado_assortativity(adjacency, actions),
        "defector_disassortativity_final": defector_disassortativity(adjacency, actions),
        "relative_degree_cooperators_final": relative_degree_cooperators(adjacency, actions),
        "core_periphery_approx_degree_concentration_final": core_periphery_approx_degree_concentration(adjacency),
        "core_periphery_score_final": core_periphery_score(adjacency),
        "mean_reward": float(episode.get("mean_reward", 0.0)),
    }
