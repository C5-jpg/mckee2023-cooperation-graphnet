from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .a2c import load_checkpoint
from .bots import SimulatedHumanBots
from .config import ACTION_SEMANTICS, FEATURE_MODES, BotConfig, GameConfig, GraphNetConfig, SEED, validate_action_semantics
from .counterfactuals import endpoint_context, make_planner
from .env import CooperativeNetworkGame, GameState
from .graphnet import GraphBatch, GraphNet, build_graph_batch, directed_edge_index
from .root_cause_planners import pair_type_for_actions
from .utils import ensure_dir, write_json


def load_model(checkpoint: str | Path, config: GameConfig, device: str, feature_mode: str = "augmented") -> tuple[GraphNet, dict[str, Any]]:
    model = GraphNet(config, GraphNetConfig(feature_mode=feature_mode)).to(device)
    payload = load_checkpoint(checkpoint, model, optimizer=None, map_location=device)
    model.eval()
    return model, payload


def logits_for_state(model: GraphNet, state: GameState, config: GameConfig, device: str) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        batch = build_graph_batch([state], config, device=device, feature_mode=getattr(model, "feature_mode", "augmented"))
        logits, value = model(batch)
        return logits[0].detach().cpu().numpy(), value.detach().cpu().numpy()


def probability_fields(logit_pair: np.ndarray, active: bool, action_semantics: str) -> dict[str, float]:
    semantics = validate_action_semantics(action_semantics)
    shifted = logit_pair - np.max(logit_pair)
    probs = np.exp(shifted) / np.exp(shifted).sum()
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    if semantics == "change":
        p_change = float(probs[1])
        change_logit = float(logit_pair[1])
        no_change_logit = float(logit_pair[0])
    elif semantics == "target_state":
        p_change = float(probs[0] if active else probs[1])
        change_logit = float(logit_pair[0] if active else logit_pair[1])
        no_change_logit = float(logit_pair[1] if active else logit_pair[0])
    else:
        raise ValueError(f"unknown action_semantics: {action_semantics!r}")
    return {
        "class0_logit": float(logit_pair[0]),
        "class1_logit": float(logit_pair[1]),
        "class0_probability": float(probs[0]),
        "class1_probability": float(probs[1]),
        "prob_target_inactive": float(probs[0]) if semantics == "target_state" else float("nan"),
        "prob_target_active": float(probs[1]) if semantics == "target_state" else float("nan"),
        "prob_recommend_change": p_change,
        "mean_change_logit": change_logit,
        "mean_no_change_logit": no_change_logit,
        "mean_p_change": p_change,
        "entropy": entropy,
    }


def sample_policy_states(
    game: CooperativeNetworkGame,
    bot_config: BotConfig,
    state_planner_names: list[str],
    n_samples: int,
    episodes: int,
    seed: int,
    checkpoint: Path | None,
    device: str,
    action_semantics: str,
    feature_mode: str,
) -> list[GameState]:
    states: list[GameState] = []
    for planner_name in state_planner_names:
        planner = make_planner(
            planner_name,
            game.config,
            checkpoint=checkpoint,
            device=device,
            action_semantics=action_semantics,
            feature_mode=feature_mode,
        )
        for episode_id in range(episodes):
            if len(states) >= n_samples:
                return states
            episode_seed = seed + episode_id * 9973 + len(states) * 31
            rng = np.random.default_rng(episode_seed)
            bots = SimulatedHumanBots(game.config.n_players, bot_config, game.config)
            bots.reset_episode(rng)
            state = game.reset(episode_seed)
            if hasattr(planner, "reset_episode"):
                planner.reset_episode(state, rng)
            while not state.done:
                transition = game.cooperation_transition(state, bots, rng)
                planner_state = transition.planner_state
                if planner_state.round_index < game.config.episode_length:
                    states.append(planner_state.copy())
                    if len(states) >= n_samples:
                        return states
                    recs = planner.recommend(planner_state, rng)
                    adjudication = game.adjudicate_recommendations(
                        planner_state,
                        recs,
                        bots,
                        rng,
                        episode_id=episode_id,
                        planner_name=planner_name,
                    )
                    next_adjacency = adjudication.next_adjacency
                else:
                    next_adjacency = planner_state.adjacency.copy()
                done = state.round_index >= game.config.episode_length
                state = GameState(
                    round_index=state.round_index + 1,
                    adjacency=next_adjacency,
                    capital=planner_state.capital.copy(),
                    prev_actions=planner_state.prev_actions.copy(),
                    seed=state.seed,
                    done=done,
                )
    return states


def real_state_probe_rows(
    model: GraphNet,
    states: list[GameState],
    config: GameConfig,
    device: str,
    action_semantics: str,
    max_edges_per_state: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    game = CooperativeNetworkGame(config)
    for state_idx, state in enumerate(states):
        logits, value = logits_for_state(model, state, config, device)
        edge_indices = np.arange(len(game.edges))
        if max_edges_per_state and len(edge_indices) > max_edges_per_state:
            edge_indices = rng.choice(edge_indices, size=max_edges_per_state, replace=False)
        for edge_idx in edge_indices:
            i, j = game.edges[int(edge_idx)]
            active = bool(state.adjacency[i, j])
            pair = pair_type_for_actions(int(state.prev_actions[i]), int(state.prev_actions[j]))
            row = {
                "probe_type": "real_state",
                "state_id": state_idx,
                "round": state.round_index,
                "edge_idx": int(edge_idx),
                "i": i,
                "j": j,
                "pair_type": pair,
                "active": active,
                "change_type": "delete" if active else "add",
                "action_semantics": action_semantics,
                "value_estimate": float(value.reshape(-1)[0]),
            }
            row.update(endpoint_context(state, i, j))
            row.update(probability_fields(logits[int(edge_idx)], active, action_semantics))
            rows.append(row)
    return rows


def mutate_edge_active_only(batch: GraphBatch, edge_idx: int, active: bool, config: GameConfig) -> GraphBatch:
    directed_edges, directed_to_undirected = directed_edge_index(config)
    edge_attr = batch.edge_attr.clone()
    adjacency = batch.adjacency.clone()
    for directed_idx, undirected_idx in enumerate(directed_to_undirected):
        if undirected_idx == edge_idx:
            edge_attr[:, directed_idx, 0] = 1.0 if active else 0.0
    adjacency[:, edge_idx] = 1.0 if active else 0.0
    return replace(batch, edge_attr=edge_attr, adjacency=adjacency)


def active_bit_probe_rows(
    model: GraphNet,
    config: GameConfig,
    device: str,
    action_semantics: str,
    feature_mode: str | None = None,
) -> list[dict[str, Any]]:
    game = CooperativeNetworkGame(config)
    base_actions = np.zeros(config.n_players, dtype=np.int64)
    base_actions[0] = 1
    base_actions[1] = 1
    base_actions[2] = 1
    base_actions[3] = 0
    base_actions[4] = 0
    base_actions[5] = 0
    base_state = GameState(
        round_index=8,
        adjacency=np.zeros((config.n_players, config.n_players), dtype=bool),
        capital=np.ones(config.n_players, dtype=np.float64) * 1.5,
        prev_actions=base_actions,
        seed=SEED,
    )
    mode = feature_mode or getattr(model, "feature_mode", "augmented")
    base_batch = build_graph_batch([base_state], config, device=device, feature_mode=mode)
    pair_edges = {
        "C-C": (0, 1),
        "C-D": (2, 3),
        "D-D": (4, 5),
    }
    edge_lookup = {edge: idx for idx, edge in enumerate(game.edges)}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for pair, edge in pair_edges.items():
            edge_idx = edge_lookup[tuple(sorted(edge))]
            for active in [False, True]:
                batch = mutate_edge_active_only(base_batch, edge_idx, active, config)
                logits, value = model(batch)
                logit_pair = logits[0, edge_idx].detach().cpu().numpy()
                row = {
                    "probe_type": "active_bit_synthetic",
                    "state_id": -1,
                    "round": base_state.round_index,
                    "edge_idx": edge_idx,
                    "i": edge[0],
                    "j": edge[1],
                    "pair_type": pair,
                    "active": active,
                    "change_type": "delete" if active else "add",
                    "action_semantics": action_semantics,
                    "value_estimate": float(value.detach().cpu().reshape(-1)[0]),
                    "node_features_held_constant": True,
                }
                row.update(endpoint_context(base_state, edge[0], edge[1]))
                row.update(probability_fields(logit_pair, active, action_semantics))
                rows.append(row)
    return rows


def summarize_probe(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "by_pair_active": [],
        "active_bit_probe": [],
    }
    if df.empty:
        return summary
    real = df[df["probe_type"] == "real_state"]
    if not real.empty:
        grouped = real.groupby(["pair_type", "active", "change_type"], as_index=False).agg(
            mean_p_change=("mean_p_change", "mean"),
            mean_change_logit=("mean_change_logit", "mean"),
            mean_no_change_logit=("mean_no_change_logit", "mean"),
            mean_entropy=("entropy", "mean"),
            n=("mean_p_change", "size"),
        )
        summary["by_pair_active"] = grouped.to_dict(orient="records")
    synth = df[df["probe_type"] == "active_bit_synthetic"]
    if not synth.empty:
        pivot = synth.pivot_table(index="pair_type", columns="active", values="mean_p_change", aggfunc="mean")
        active_rows = []
        for pair in pivot.index:
            inactive = float(pivot.loc[pair].get(False, np.nan))
            active = float(pivot.loc[pair].get(True, np.nan))
            active_rows.append(
                {
                    "pair_type": pair,
                    "p_change_inactive": inactive,
                    "p_change_active": active,
                    "active_minus_inactive": active - inactive if np.isfinite(active) and np.isfinite(inactive) else None,
                }
            )
        summary["active_bit_probe"] = active_rows
    return summary


def run_policy_probe(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out_dir = ensure_dir(root / args.out_dir)
    config = GameConfig(action_semantics=args.action_semantics)
    bot_config = BotConfig()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    model, payload = load_model(checkpoint, config, args.device, feature_mode=args.feature_mode)
    game = CooperativeNetworkGame(config)
    state_planners = [name.strip() for name in args.state_planners.split(",") if name.strip()]
    states = sample_policy_states(
        game,
        bot_config,
        state_planners,
        args.n_states,
        args.episodes,
        args.seed,
        checkpoint,
        args.device,
        args.action_semantics,
        args.feature_mode,
    )
    rows = real_state_probe_rows(
        model,
        states,
        config,
        args.device,
        args.action_semantics,
        args.max_edges_per_state,
        args.seed + 22_000,
    )
    rows.extend(active_bit_probe_rows(model, config, args.device, args.action_semantics, args.feature_mode))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "policy_logit_probe.csv"
    df.to_csv(csv_path, index=False)
    summary = summarize_probe(df)
    summary.update(
        {
            "checkpoint": str(checkpoint).replace("\\", "/"),
            "checkpoint_rounds": int(payload.get("simulated_rounds", -1)),
            "checkpoint_episodes": int(payload.get("episodes", -1)),
            "checkpoint_seed": int(payload.get("seed", -1)),
            "action_semantics": args.action_semantics,
            "feature_mode": args.feature_mode,
            "state_planners": state_planners,
            "n_states": len(states),
            "csv_path": str(csv_path).replace("\\", "/"),
        }
    )
    write_json(out_dir / "policy_logit_probe_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/root_cause")
    parser.add_argument("--checkpoint", default="outputs/train/checkpoints/step_1000005.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=GraphNetConfig().feature_mode)
    parser.add_argument("--state-planners", default="maximum_connectivity,neutral,graphnet")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--n-states", type=int, default=80)
    parser.add_argument("--max-edges-per-state", type=int, default=120)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_policy_probe(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
