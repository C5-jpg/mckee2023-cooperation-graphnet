from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bots import SimulatedHumanBots
from .config import ACTION_SEMANTICS, FEATURE_MODES, BotConfig, GameConfig, GraphNetConfig, SEED
from .counterfactuals import make_planner
from .env import CooperativeNetworkGame
from .policy_audit import build_table6_policy_audit
from .utils import ensure_dir, write_json


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or np.allclose(x.mean(), 0.0):
        return 0.0
    diffs = np.abs(x[:, None] - x[None, :]).sum()
    return float(diffs / (2 * x.size * x.size * max(float(x.mean()), 1e-12)))


def evaluate_planner(
    game: CooperativeNetworkGame,
    planner,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    bot_config = BotConfig()
    all_rounds: list[dict[str, Any]] = []
    all_players: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    all_recs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for episode_id in range(episodes):
        episode_seed = seed + episode_id * 9973
        bots = SimulatedHumanBots(game.config.n_players, bot_config, game.config)
        result = game.run_episode(planner, bots, episode_seed, episode_id=episode_id)
        rounds = result["rounds"]
        players = result["players"]
        final_state = result["final_state"]
        all_rounds.extend(rounds)
        all_players.extend(players)
        all_edges.extend(result["edges"])
        all_recs.extend(result["recommendations"])
        summaries.append(
            {
                "planner": getattr(planner, "name", planner.__class__.__name__),
                "episode_id": episode_id,
                "mean_cooperation": float(np.mean([r["mean_cooperation"] for r in rounds])),
                "final_cooperation": float(rounds[-1]["mean_cooperation"]),
                "mean_capital": float(np.mean([r["mean_capital"] for r in rounds])),
                "final_capital": float(rounds[-1]["mean_capital"]),
                "mean_density": float(np.mean([r["network_density"] for r in rounds])),
                "final_density": float(rounds[-1]["network_density"]),
                "mean_reward": float(result["mean_reward"]),
                "final_gini": gini(final_state.capital),
            }
        )
    return {
        "rounds": pd.DataFrame(all_rounds),
        "players": pd.DataFrame(all_players),
        "edges": pd.DataFrame(all_edges),
        "recommendations": pd.DataFrame(all_recs),
        "episodes": pd.DataFrame(summaries),
    }


def planner_comparison_from_episode_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby("planner", as_index=False).agg(
        n_episodes=("episode_id", "size"),
        mean_cooperation=("mean_cooperation", "mean"),
        final_cooperation=("final_cooperation", "mean"),
        mean_capital=("mean_capital", "mean"),
        final_capital=("final_capital", "mean"),
        mean_density=("mean_density", "mean"),
        final_density=("final_density", "mean"),
        mean_reward=("mean_reward", "mean"),
        final_gini=("final_gini", "mean"),
    )


def run_oracle_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out_dir = ensure_dir(root / args.out_dir)
    config = GameConfig()
    game = CooperativeNetworkGame(config)
    planner_names = [
        "static",
        "random",
        "cooperative_clustering",
        "encouragement",
        "neutral",
        "maximum_connectivity",
        "graphnet",
        "table6_oracle",
        "dd_delete_oracle",
        "no_delete_high_density",
    ]
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    all_episode_rows = []
    audit_rows = []
    for planner_name in planner_names:
        planner = make_planner(
            planner_name,
            config,
            checkpoint=checkpoint,
            device=args.device,
            action_semantics=args.action_semantics,
            feature_mode=args.feature_mode,
        )
        result = evaluate_planner(game, planner, args.episodes, args.seed + len(all_episode_rows) * 101)
        all_episode_rows.append(result["episodes"])
        if not result["edges"].empty:
            audit = build_table6_policy_audit(result["edges"], result["recommendations"], planner=getattr(planner, "name", planner_name))
            audit.insert(0, "planner", getattr(planner, "name", planner_name))
            audit_rows.append(audit)
    episodes_df = pd.concat(all_episode_rows, ignore_index=True)
    comparison = planner_comparison_from_episode_rows(episodes_df)
    comparison.to_csv(out_dir / "oracle_planner_comparison.csv", index=False)
    if audit_rows:
        audit_df = pd.concat(audit_rows, ignore_index=True)
    else:
        audit_df = pd.DataFrame()
    audit_df.to_csv(out_dir / "oracle_table6_policy.csv", index=False)
    table6_reward = comparison.loc[comparison["planner"] == "table6_oracle", "mean_reward"]
    high_reward = comparison.loc[comparison["planner"] == "no_delete_high_density", "mean_reward"]
    max_reward = comparison.loc[comparison["planner"] == "maximum_connectivity", "mean_reward"]
    summary = {
        "episodes_per_planner": args.episodes,
        "seed": args.seed,
        "action_semantics_for_graphnet": args.action_semantics,
        "feature_mode_for_graphnet": args.feature_mode,
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "table6_oracle_mean_reward": float(table6_reward.iloc[0]) if not table6_reward.empty else None,
        "no_delete_high_density_mean_reward": float(high_reward.iloc[0]) if not high_reward.empty else None,
        "maximum_connectivity_mean_reward": float(max_reward.iloc[0]) if not max_reward.empty else None,
        "table6_oracle_beats_no_delete_high_density": bool(
            not table6_reward.empty and not high_reward.empty and float(table6_reward.iloc[0]) > float(high_reward.iloc[0])
        ),
        "table6_oracle_beats_maximum_connectivity": bool(
            not table6_reward.empty and not max_reward.empty and float(table6_reward.iloc[0]) > float(max_reward.iloc[0])
        ),
        "comparison_csv": str((out_dir / "oracle_planner_comparison.csv")).replace("\\", "/"),
        "audit_csv": str((out_dir / "oracle_table6_policy.csv")).replace("\\", "/"),
    }
    write_json(out_dir / "oracle_diagnostic_summary.json", summary)
    return summary


def random_adjacency_with_density(config: GameConfig, density: float, rng: np.random.Generator) -> np.ndarray:
    adjacency = np.zeros((config.n_players, config.n_players), dtype=bool)
    for i in range(config.n_players):
        for j in range(i + 1, config.n_players):
            if rng.random() < density:
                adjacency[i, j] = True
                adjacency[j, i] = True
    return adjacency


def run_max_connectivity_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out_dir = ensure_dir(root / args.out_dir)
    config = GameConfig()
    bot_config = BotConfig()
    game = CooperativeNetworkGame(config)
    by_round_rows: list[pd.DataFrame] = []
    for planner_name in ["static", "neutral", "maximum_connectivity"]:
        planner = make_planner(planner_name, config)
        result = evaluate_planner(game, planner, args.episodes, args.seed + len(by_round_rows) * 17)
        rounds = result["rounds"]
        grouped = rounds.groupby(["planner", "round"], as_index=False).agg(
            mean_cooperation=("mean_cooperation", "mean"),
            mean_capital=("mean_capital", "mean"),
            mean_density=("network_density", "mean"),
            mean_penalty=("recommendation_penalty", "mean"),
            mean_nonzero_recommendations=("nonzero_recommendations", "mean"),
            mean_rejected_recommendations=("rejected_recommendations", "mean"),
        )
        by_round_rows.append(grouped)
    by_round = pd.concat(by_round_rows, ignore_index=True)
    by_round.to_csv(out_dir / "maximum_connectivity_by_round.csv", index=False)

    rng = np.random.default_rng(args.seed + 90_000)
    density_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    for density in [0.3, 0.6, 0.9, 1.0]:
        constrained_rates = []
        prob_means = []
        degree_means = []
        capital = np.full(config.n_players, config.initial_capital)
        for replicate in range(args.surface_replicates):
            adjacency = random_adjacency_with_density(config, density, rng)
            prev_actions = (rng.random(config.n_players) < 0.5).astype(np.int64)
            bots = SimulatedHumanBots(config.n_players, bot_config, config)
            bots.theta = rng.normal(bot_config.mu_theta, bot_config.sigma_theta, size=config.n_players)
            decision = bots.choose_cooperation(2, adjacency, prev_actions, capital, rng)
            constrained_rates.append(float(np.mean(decision.constrained)))
            prob_means.append(float(np.mean(decision.probabilities)))
            degree_means.append(float(np.mean(adjacency.sum(axis=1))))
        density_rows.append(
            {
                "density": density,
                "mean_degree": float(np.mean(degree_means)),
                "mean_cooperation_probability": float(np.mean(prob_means)),
                "capital_constraint_rate": float(np.mean(constrained_rates)),
                "capital": config.initial_capital,
                "required_cost_at_mean_degree": float(config.cost_c * np.mean(degree_means)),
            }
        )
        degree = int(round(density * (config.n_players - 1)))
        for coop_fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            adjacency = np.zeros((config.n_players, config.n_players), dtype=bool)
            neighbors = list(range(1, min(config.n_players, degree + 1)))
            for j in neighbors:
                adjacency[0, j] = True
                adjacency[j, 0] = True
            prev_actions = np.zeros(config.n_players, dtype=np.int64)
            coop_count = int(round(len(neighbors) * coop_fraction))
            for j in neighbors[:coop_count]:
                prev_actions[j] = 1
            bots = SimulatedHumanBots(config.n_players, bot_config, config)
            bots.theta = np.zeros(config.n_players, dtype=np.float64)
            probs = bots.cooperation_probabilities(2, adjacency, prev_actions, np.full(config.n_players, config.initial_capital))
            required_cost = config.cost_c * adjacency.sum(axis=1)[0]
            surface_rows.append(
                {
                    "density": density,
                    "degree_player0": int(adjacency.sum(axis=1)[0]),
                    "cooperating_neighbor_fraction": coop_fraction,
                    "cooperating_neighbor_count": coop_count,
                    "theta": 0.0,
                    "capital": config.initial_capital,
                    "cooperation_probability_player0": float(probs[0]),
                    "required_cost_player0": float(required_cost),
                    "would_be_capital_constrained_if_cooperate": bool(required_cost > config.initial_capital + 1e-12),
                }
            )
    pd.DataFrame(density_rows).to_csv(out_dir / "capital_constraint_by_density.csv", index=False)
    pd.DataFrame(surface_rows).to_csv(out_dir / "bot_probability_surface.csv", index=False)

    final_max = by_round[(by_round["planner"] == "maximum_connectivity") & (by_round["round"] == config.episode_length)]
    final_neutral = by_round[(by_round["planner"] == "neutral") & (by_round["round"] == config.episode_length)]
    summary = {
        "episodes": args.episodes,
        "maximum_connectivity_final_cooperation": float(final_max["mean_cooperation"].iloc[0]) if not final_max.empty else None,
        "maximum_connectivity_final_density": float(final_max["mean_density"].iloc[0]) if not final_max.empty else None,
        "neutral_final_cooperation": float(final_neutral["mean_cooperation"].iloc[0]) if not final_neutral.empty else None,
        "capital_constraint_rows": density_rows,
        "by_round_csv": str((out_dir / "maximum_connectivity_by_round.csv")).replace("\\", "/"),
        "capital_constraint_csv": str((out_dir / "capital_constraint_by_density.csv")).replace("\\", "/"),
        "bot_probability_surface_csv": str((out_dir / "bot_probability_surface.csv")).replace("\\", "/"),
    }
    write_json(out_dir / "maximum_connectivity_anomaly_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".")
    common.add_argument("--out-dir", default="outputs/root_cause")
    common.add_argument("--episodes", type=int, default=512)
    common.add_argument("--seed", type=int, default=SEED)

    oracle = sub.add_parser("oracles", parents=[common])
    oracle.add_argument("--checkpoint", default="outputs/train/checkpoints/step_1000005.pt")
    oracle.add_argument("--device", default="cpu")
    oracle.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    oracle.add_argument("--feature-mode", choices=FEATURE_MODES, default=GraphNetConfig().feature_mode)

    maxconn = sub.add_parser("maxconn", parents=[common])
    maxconn.add_argument("--surface-replicates", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "oracles":
        summary = run_oracle_diagnostics(args)
    elif args.command == "maxconn":
        summary = run_max_connectivity_diagnostics(args)
    else:
        raise ValueError(args.command)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
