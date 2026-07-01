from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .bots import SimulatedHumanBots
from .config import ACTION_SEMANTICS, FEATURE_MODES, BotConfig, GameConfig, GraphNetConfig, SEED, TrainingConfig
from .env import AdjudicationResult, CooperativeNetworkGame, GameState
from .planners import (
    CooperativeClusteringPlanner,
    EncouragementPlanner,
    MaximumConnectivityPlanner,
    NeutralPlanner,
    RandomRecommendationPlanner,
    StaticPlanner,
)
from .root_cause_planners import (
    CheckpointGraphNetPlanner,
    DDDeleteOraclePlanner,
    NoDeleteHighDensityPlanner,
    Table6OraclePlanner,
    pair_type_for_actions,
)
from .utils import ensure_dir, write_json


@dataclass
class SampledEdgeState:
    sample_id: int
    episode_id: int
    episode_seed: int
    state_planner: str
    round_index: int
    edge_idx: int
    i: int
    j: int
    pair_type: str
    planner_state: GameState
    bots: SimulatedHumanBots
    endpoint_context: dict[str, Any]


def clone_bots(bots: SimulatedHumanBots, config: BotConfig, game_config: GameConfig) -> SimulatedHumanBots:
    copied = SimulatedHumanBots(game_config.n_players, config, game_config)
    copied.theta = bots.theta.copy()
    return copied


def make_planner(
    name: str,
    config: GameConfig,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    action_semantics: str = "target_state",
    feature_mode: str = GraphNetConfig().feature_mode,
):
    if name == "static":
        return StaticPlanner(config)
    if name == "random":
        return RandomRecommendationPlanner(config)
    if name == "cooperative_clustering":
        return CooperativeClusteringPlanner(config)
    if name == "encouragement":
        return EncouragementPlanner(config)
    if name == "neutral":
        return NeutralPlanner(config)
    if name == "maximum_connectivity":
        return MaximumConnectivityPlanner(config)
    if name == "table6_oracle":
        return Table6OraclePlanner(config)
    if name == "dd_delete_oracle":
        return DDDeleteOraclePlanner(config)
    if name == "no_delete_high_density":
        return NoDeleteHighDensityPlanner(config)
    if name == "graphnet":
        if checkpoint is None:
            raise ValueError("graphnet planner requires --checkpoint")
        return CheckpointGraphNetPlanner(
            checkpoint_path=checkpoint,
            device=device,
            deterministic=True,
            action_semantics=action_semantics,
            feature_mode=feature_mode,
            config=config,
            name=f"graphnet_{action_semantics}",
        )
    raise ValueError(f"unknown planner: {name!r}")


def reward_components(planner_state: GameState, adjudication: AdjudicationResult, config: GameConfig) -> dict[str, float]:
    mean_capital = float(np.mean(planner_state.capital))
    penalty = float(adjudication.rejected_nonzero / max(1, config.n_possible_edges))
    reward = float(mean_capital - config.recommendation_penalty_weight * penalty)
    return {
        "mean_capital_reward": mean_capital,
        "rejected_nonzero": float(adjudication.rejected_nonzero),
        "nonzero_recommendations": float(adjudication.nonzero_recommendations),
        "recommendation_penalty": penalty,
        "total_reward": reward,
    }


def endpoint_context(state: GameState, i: int, j: int) -> dict[str, Any]:
    adjacency = state.adjacency
    actions = state.prev_actions.astype(np.float64)
    degree = adjacency.sum(axis=1).astype(np.float64)
    coop_neighbors = adjacency @ actions
    coop_frac = np.divide(coop_neighbors, degree, out=np.zeros_like(coop_neighbors), where=degree > 0)
    return {
        "degree_i": int(degree[i]),
        "degree_j": int(degree[j]),
        "capital_i": float(state.capital[i]),
        "capital_j": float(state.capital[j]),
        "action_i": int(state.prev_actions[i]),
        "action_j": int(state.prev_actions[j]),
        "coop_neighbors_i": float(coop_neighbors[i]),
        "coop_neighbors_j": float(coop_neighbors[j]),
        "coop_neighbor_fraction_i": float(coop_frac[i]),
        "coop_neighbor_fraction_j": float(coop_frac[j]),
    }


def delete_acceptance_probabilities(bots: SimulatedHumanBots, state: GameState, i: int, j: int) -> dict[str, float]:
    # If assigned to i, the referent is j; if assigned to j, the referent is i.
    p_if_assigned_i = bots.acceptance_probability(-1, int(state.prev_actions[j]))
    p_if_assigned_j = bots.acceptance_probability(-1, int(state.prev_actions[i]))
    return {
        "delete_acceptance_probability_assigned_i": float(p_if_assigned_i),
        "delete_acceptance_probability_assigned_j": float(p_if_assigned_j),
        "delete_acceptance_probability_mean": float(0.5 * (p_if_assigned_i + p_if_assigned_j)),
    }


def discounted_future_return(
    game: CooperativeNetworkGame,
    state: GameState,
    bots: SimulatedHumanBots,
    rng: np.random.Generator,
    planner,
    gamma: float,
) -> tuple[float, dict[str, float]]:
    total = 0.0
    discount = 1.0
    current = state
    final_mean_cooperation = float(np.mean(current.prev_actions))
    final_mean_capital = float(np.mean(current.capital))
    final_density = float(current.adjacency.sum() / (game.config.n_players * (game.config.n_players - 1)))
    while not current.done:
        result = game.step(current, planner, bots, rng, episode_id=0)
        total += discount * float(result.reward)
        discount *= gamma
        current = result.next_state
        if result.round_record:
            final_mean_cooperation = float(result.round_record["mean_cooperation"])
            final_mean_capital = float(result.round_record["mean_capital"])
            final_density = float(result.round_record["network_density"])
    return total, {
        "final_cooperation": final_mean_cooperation,
        "final_capital": final_mean_capital,
        "future_density": final_density,
    }


def apply_intervention(
    game: CooperativeNetworkGame,
    sample: SampledEdgeState,
    mode: str,
    rng: np.random.Generator,
    bot_config: BotConfig,
) -> tuple[GameState, dict[str, float]]:
    state = sample.planner_state.copy()
    bots = clone_bots(sample.bots, bot_config, game.config)
    recs = np.zeros(len(game.edges), dtype=np.int64)
    accepted_probability = 0.0
    accepted = 0.0
    if mode == "no_change" or mode == "forced_no_change":
        adjudication = AdjudicationResult(
            next_adjacency=state.adjacency.copy(),
            recommendation_records=[],
            rejected_nonzero=0,
            nonzero_recommendations=0,
        )
    elif mode == "delete_probabilistic":
        recs[sample.edge_idx] = -1
        accepted_probability = delete_acceptance_probabilities(bots, state, sample.i, sample.j)[
            "delete_acceptance_probability_mean"
        ]
        adjudication = game.adjudicate_recommendations(
            state,
            recs,
            bots,
            rng,
            episode_id=sample.episode_id,
            planner_name="counterfactual_delete_probabilistic",
        )
        if adjudication.recommendation_records:
            accepted = float(bool(adjudication.recommendation_records[0]["accepted"]))
    elif mode == "delete_forced":
        next_adj = state.adjacency.copy()
        next_adj[sample.i, sample.j] = False
        next_adj[sample.j, sample.i] = False
        accepted_probability = 1.0
        accepted = 1.0
        adjudication = AdjudicationResult(
            next_adjacency=next_adj,
            recommendation_records=[
                {
                    "episode_id": sample.episode_id,
                    "round": state.round_index,
                    "planner": "counterfactual_delete_forced",
                    "i": sample.i,
                    "j": sample.j,
                    "recommendation": -1,
                    "assigned_player": sample.i,
                    "referent_player": sample.j,
                    "referent_cooperated": int(state.prev_actions[sample.j]),
                    "accepted": True,
                    "edge_active_before": True,
                    "edge_active_after": False,
                }
            ],
            rejected_nonzero=0,
            nonzero_recommendations=1,
        )
    else:
        raise ValueError(f"unknown intervention mode: {mode!r}")

    components = reward_components(state, adjudication, game.config)
    done = state.round_index >= game.config.episode_length
    next_state = GameState(
        round_index=state.round_index + 1,
        adjacency=adjudication.next_adjacency,
        capital=state.capital.copy(),
        prev_actions=state.prev_actions.copy(),
        seed=state.seed,
        done=done,
    )
    components["accepted_probability"] = float(accepted_probability)
    components["accepted"] = float(accepted)
    return next_state, components


def estimate_intervention(
    game: CooperativeNetworkGame,
    sample: SampledEdgeState,
    mode: str,
    continuation_planner_name: str,
    n_rollouts: int,
    seed: int,
    gamma: float,
    bot_config: BotConfig,
) -> dict[str, float]:
    returns = []
    final_coops = []
    final_capitals = []
    densities = []
    penalties = []
    accepted = []
    accepted_probs = []
    immediate_rewards = []
    for rollout_idx in range(n_rollouts):
        rollout_seed = seed + sample.sample_id * 100_003 + rollout_idx * 9_973
        rng = np.random.default_rng(rollout_seed)
        next_state, components = apply_intervention(game, sample, mode, rng, bot_config)
        continuation_planner = make_planner(continuation_planner_name, game.config)
        future_bots = clone_bots(sample.bots, bot_config, game.config)
        future_return, final_metrics = discounted_future_return(
            game,
            next_state,
            future_bots,
            rng,
            continuation_planner,
            gamma,
        )
        returns.append(float(components["total_reward"] + gamma * future_return))
        immediate_rewards.append(float(components["total_reward"]))
        penalties.append(float(components["recommendation_penalty"]))
        accepted.append(float(components["accepted"]))
        accepted_probs.append(float(components["accepted_probability"]))
        final_coops.append(final_metrics["final_cooperation"])
        final_capitals.append(final_metrics["final_capital"])
        densities.append(final_metrics["future_density"])
    return {
        f"{mode}_return_mean": float(np.mean(returns)),
        f"{mode}_return_std": float(np.std(returns)),
        f"{mode}_immediate_reward_mean": float(np.mean(immediate_rewards)),
        f"{mode}_penalty_mean": float(np.mean(penalties)),
        f"{mode}_accepted_probability_mean": float(np.mean(accepted_probs)),
        f"{mode}_accepted_rate": float(np.mean(accepted)),
        f"{mode}_final_cooperation_mean": float(np.mean(final_coops)),
        f"{mode}_final_capital_mean": float(np.mean(final_capitals)),
        f"{mode}_future_density_mean": float(np.mean(densities)),
    }


def sample_edge_states(
    game: CooperativeNetworkGame,
    bot_config: BotConfig,
    pair_types: set[str],
    state_planner_names: Iterable[str],
    samples_per_pair: int,
    episodes: int,
    seed: int,
    checkpoint: str | Path | None,
    device: str,
    action_semantics: str,
    feature_mode: str,
) -> list[SampledEdgeState]:
    samples: list[SampledEdgeState] = []
    counts = {pair: 0 for pair in pair_types}
    sample_id = 0
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
            if all(counts[pair] >= samples_per_pair for pair in pair_types):
                return samples
            episode_seed = seed + episode_id * 9973 + len(samples) * 17
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
                    candidates = []
                    for edge_idx, (i, j) in enumerate(game.edges):
                        if not planner_state.adjacency[i, j]:
                            continue
                        pair = pair_type_for_actions(int(planner_state.prev_actions[i]), int(planner_state.prev_actions[j]))
                        if pair in pair_types and counts[pair] < samples_per_pair:
                            candidates.append((edge_idx, i, j, pair))
                    if candidates:
                        rng.shuffle(candidates)
                        for edge_idx, i, j, pair in candidates[: max(1, min(3, len(candidates)))]:
                            if counts[pair] >= samples_per_pair:
                                continue
                            ctx = endpoint_context(planner_state, i, j)
                            ctx.update(delete_acceptance_probabilities(bots, planner_state, i, j))
                            samples.append(
                                SampledEdgeState(
                                    sample_id=sample_id,
                                    episode_id=episode_id,
                                    episode_seed=episode_seed,
                                    state_planner=planner_name,
                                    round_index=planner_state.round_index,
                                    edge_idx=edge_idx,
                                    i=i,
                                    j=j,
                                    pair_type=pair,
                                    planner_state=planner_state.copy(),
                                    bots=clone_bots(bots, bot_config, game.config),
                                    endpoint_context=ctx,
                                )
                            )
                            counts[pair] += 1
                            sample_id += 1
                    recommendations = planner.recommend(planner_state, rng)
                    adjudication = game.adjudicate_recommendations(
                        planner_state,
                        recommendations,
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
    return samples


def summarize_counterfactuals(rows: list[dict[str, Any]], pair_type: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"pair_type": pair_type, "n_samples": 0, "status": "no_samples"}
    pair_df = df[df["pair_type"] == pair_type]
    if pair_df.empty:
        return {"pair_type": pair_type, "n_samples": 0, "status": "no_pair_samples"}
    summary = {
        "pair_type": pair_type,
        "n_samples": int(len(pair_df)),
        "mean_round": float(pair_df["round_index"].mean()),
        "delete_probabilistic_minus_no_change_return": float(pair_df["delta_return_delete_probabilistic"].mean()),
        "delete_forced_minus_no_change_return": float(pair_df["delta_return_delete_forced"].mean()),
        "delete_probabilistic_minus_no_change_final_cooperation": float(pair_df["delta_final_cooperation_delete_probabilistic"].mean()),
        "delete_forced_minus_no_change_final_cooperation": float(pair_df["delta_final_cooperation_delete_forced"].mean()),
        "delete_probabilistic_minus_no_change_final_capital": float(pair_df["delta_final_capital_delete_probabilistic"].mean()),
        "delete_forced_minus_no_change_final_capital": float(pair_df["delta_final_capital_delete_forced"].mean()),
        "delete_probabilistic_minus_no_change_density": float(pair_df["delta_future_density_delete_probabilistic"].mean()),
        "delete_forced_minus_no_change_density": float(pair_df["delta_future_density_delete_forced"].mean()),
        "accepted_probability_mean": float(pair_df["delete_probabilistic_accepted_probability_mean"].mean()),
        "accepted_rate_mean": float(pair_df["delete_probabilistic_accepted_rate"].mean()),
        "penalty_contribution_mean": float(pair_df["delete_probabilistic_penalty_mean"].mean()),
        "positive_probabilistic_return_share": float((pair_df["delta_return_delete_probabilistic"] > 0).mean()),
        "positive_forced_return_share": float((pair_df["delta_return_delete_forced"] > 0).mean()),
    }
    if summary["delete_forced_minus_no_change_return"] > 0 and summary["delete_probabilistic_minus_no_change_return"] <= 0:
        summary["diagnosis"] = "forced_delete_positive_but_probabilistic_not_positive_check_acceptance_penalty_assignment"
    elif summary["delete_forced_minus_no_change_return"] > 0 or summary["delete_probabilistic_minus_no_change_return"] > 0:
        summary["diagnosis"] = "delete_has_positive_counterfactual_value_learning_or_credit_assignment_suspect"
    else:
        summary["diagnosis"] = "delete_not_positive_environment_reward_landscape_suspect"
    return summary


def run_counterfactuals(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out_dir = ensure_dir(root / args.out_dir)
    game_config = GameConfig()
    bot_config = BotConfig()
    training_config = TrainingConfig()
    game = CooperativeNetworkGame(game_config)
    pair_types = {"D-D", "C-D"} if args.pair_type == "both" else {args.pair_type}
    state_planners = [name.strip() for name in args.state_planners.split(",") if name.strip()]
    samples = sample_edge_states(
        game,
        bot_config,
        pair_types,
        state_planners,
        args.samples_per_pair,
        args.episodes,
        args.seed,
        args.checkpoint,
        args.device,
        args.action_semantics,
        args.feature_mode,
    )
    modes = ["no_change", "delete_probabilistic", "delete_forced", "forced_no_change"]
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "episode_id": sample.episode_id,
            "episode_seed": sample.episode_seed,
            "state_planner": sample.state_planner,
            "continuation_planner": args.continuation_planner,
            "round_index": sample.round_index,
            "edge_idx": sample.edge_idx,
            "i": sample.i,
            "j": sample.j,
            "pair_type": sample.pair_type,
            "action_semantics": args.action_semantics,
        }
        row.update(sample.endpoint_context)
        for mode in modes:
            row.update(
                estimate_intervention(
                    game,
                    sample,
                    mode,
                    args.continuation_planner,
                    args.rollouts_per_sample,
                    args.seed + 11_000,
                    training_config.discount_gamma,
                    bot_config,
                )
            )
        row["delta_return_delete_probabilistic"] = row["delete_probabilistic_return_mean"] - row["no_change_return_mean"]
        row["delta_return_delete_forced"] = row["delete_forced_return_mean"] - row["no_change_return_mean"]
        row["delta_return_forced_no_change"] = row["forced_no_change_return_mean"] - row["no_change_return_mean"]
        row["delta_final_cooperation_delete_probabilistic"] = (
            row["delete_probabilistic_final_cooperation_mean"] - row["no_change_final_cooperation_mean"]
        )
        row["delta_final_cooperation_delete_forced"] = row["delete_forced_final_cooperation_mean"] - row["no_change_final_cooperation_mean"]
        row["delta_final_capital_delete_probabilistic"] = (
            row["delete_probabilistic_final_capital_mean"] - row["no_change_final_capital_mean"]
        )
        row["delta_final_capital_delete_forced"] = row["delete_forced_final_capital_mean"] - row["no_change_final_capital_mean"]
        row["delta_future_density_delete_probabilistic"] = (
            row["delete_probabilistic_future_density_mean"] - row["no_change_future_density_mean"]
        )
        row["delta_future_density_delete_forced"] = row["delete_forced_future_density_mean"] - row["no_change_future_density_mean"]
        rows.append(row)

    df = pd.DataFrame(rows)
    payload: dict[str, Any] = {
        "seed": args.seed,
        "pair_type_requested": args.pair_type,
        "state_planners": state_planners,
        "continuation_planner": args.continuation_planner,
        "episodes": args.episodes,
        "samples_per_pair": args.samples_per_pair,
        "rollouts_per_sample": args.rollouts_per_sample,
        "action_semantics": args.action_semantics,
        "feature_mode": args.feature_mode,
        "n_rows": int(len(df)),
        "summaries": {},
    }
    if not df.empty:
        if "D-D" in pair_types:
            dd = df[df["pair_type"] == "D-D"]
            dd.to_csv(out_dir / "dd_delete_counterfactuals.csv", index=False)
            payload["summaries"]["D-D"] = summarize_counterfactuals(rows, "D-D")
            write_json(out_dir / "dd_delete_counterfactual_summary.json", payload["summaries"]["D-D"])
        if "C-D" in pair_types:
            cd = df[df["pair_type"] == "C-D"]
            cd.to_csv(out_dir / "cd_delete_counterfactuals.csv", index=False)
            payload["summaries"]["C-D"] = summarize_counterfactuals(rows, "C-D")
            write_json(out_dir / "cd_delete_counterfactual_summary.json", payload["summaries"]["C-D"])
        df.to_csv(out_dir / "all_delete_counterfactuals.csv", index=False)
    else:
        for pair in pair_types:
            payload["summaries"][pair] = {"pair_type": pair, "n_samples": 0, "status": "no_samples"}
    write_json(out_dir / "counterfactual_run_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/root_cause")
    parser.add_argument("--pair-type", choices=["D-D", "C-D", "both"], default="both")
    parser.add_argument("--state-planners", default="maximum_connectivity,neutral,graphnet")
    parser.add_argument("--continuation-planner", default="static")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--samples-per-pair", type=int, default=80)
    parser.add_argument("--rollouts-per-sample", type=int, default=24)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint", default="outputs/train/checkpoints/step_1000005.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=GraphNetConfig().feature_mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(args.root) / checkpoint_path
        args.checkpoint = checkpoint_path
    summary = run_counterfactuals(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
