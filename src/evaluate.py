from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from .a2c import load_checkpoint
from .bots import SimulatedHumanBots
from .config import ACTION_SEMANTICS, FEATURE_MODES, BotConfig, EvalConfig, GameConfig, GraphNetConfig, SEED
from .data_logging import create_output_dirs
from .env import CooperativeNetworkGame
from .graphnet import GraphNet
from .metrics import final_episode_summary
from .policy_audit import build_table6_policy_audit, write_policy_audit_outputs
from .planners import GraphNetPlanner, all_rule_based_planners
from .utils import ensure_dir, write_json


def maximum_connectivity_diagnostic_status(root: Path) -> tuple[str, list[str]]:
    path = root / "outputs/diagnostics/maximum_connectivity_diagnostics.csv"
    if not path.exists():
        return "not_run", ["maximum connectivity diagnostics file is missing"]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return "warning", [f"could not read maximum connectivity diagnostics: {exc}"]
    if "planner" not in df.columns or "final_round_cooperation" not in df.columns:
        return "warning", ["maximum connectivity diagnostics missing required columns"]
    max_conn = df[df["planner"] == "maximum_connectivity"]
    if max_conn.empty:
        return "warning", ["maximum connectivity diagnostics missing maximum_connectivity rows"]
    final_coop = float(max_conn["final_round_cooperation"].mean())
    if final_coop > 0.50:
        return "warning", [f"maximum connectivity final cooperation remains high in simulated diagnostics: {final_coop:.3f}"]
    return "pass", [f"maximum connectivity final cooperation diagnostic: {final_coop:.3f}"]


def evaluate_planner(game: CooperativeNetworkGame, planner, n_episodes: int, seed: int) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    summaries = []
    all_rounds = []
    all_players = []
    all_edges = []
    all_recommendations = []
    for ep in range(n_episodes):
        bots = SimulatedHumanBots(game.config.n_players, BotConfig(), game.config)
        episode = game.run_episode(planner, bots, seed + ep * 7919, episode_id=ep)
        summary = final_episode_summary(episode, game.config)
        summary.update({"planner": planner.name, "seed": seed, "episode_id": ep})
        summaries.append(summary)
        all_rounds.extend(episode["rounds"])
        all_players.extend(episode["players"])
        all_edges.extend(episode["edges"])
        all_recommendations.extend(episode["recommendations"])
    aggregate = {
        "planner": planner.name,
        "seed": seed,
        "n_eval_episodes": n_episodes,
    }
    for key in [
        "mean_cooperation",
        "final_round_cooperation",
        "mean_capital",
        "final_mean_capital",
        "gini_final",
        "network_density_final",
        "cooperator_defector_assortativity_final",
        "newman_assortativity_final",
        "rand_assortativity_final",
        "shirado_assortativity_final",
        "defector_disassortativity_final",
        "relative_degree_cooperators_final",
        "core_periphery_approx_degree_concentration_final",
        "core_periphery_score_final",
        "mean_reward",
    ]:
        aggregate[key] = float(pd.Series([s[key] for s in summaries]).mean())
    return summaries, all_rounds, all_players, all_edges, all_recommendations, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-eval-episodes", type=int, default=16)
    parser.add_argument("--checkpoint", default="outputs/train/checkpoints/latest.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--raw-dir", default="outputs/raw_eval")
    parser.add_argument("--stage-summary", default="stage07_target_train_eval/eval_summary.json")
    parser.add_argument("--policy-audit-dir", default=None)
    parser.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=GraphNetConfig().feature_mode)
    args = parser.parse_args()
    root = Path(args.root)
    create_output_dirs(root)
    game_config = GameConfig(action_semantics=args.action_semantics)
    game = CooperativeNetworkGame(game_config)
    rows = []
    episode_rows = []
    round_rows = []
    player_rows = []
    edge_rows = []
    recommendation_rows = []
    baseline_summary: Dict[str, Any] = {}
    for planner in all_rule_based_planners(game_config):
        summaries, rounds, players, edges, recommendations, aggregate = evaluate_planner(game, planner, args.n_eval_episodes, args.seed)
        rows.append(aggregate)
        episode_rows.extend(summaries)
        round_rows.extend(rounds)
        player_rows.extend(players)
        edge_rows.extend(edges)
        recommendation_rows.extend(recommendations)
        baseline_summary[planner.name] = aggregate
    graphnet_summary = {"available": False, "reason": "checkpoint missing"}
    ckpt_path = root / args.checkpoint
    if ckpt_path.exists():
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = GraphNet(game_config, GraphNetConfig(feature_mode=args.feature_mode)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.0004)
        payload = load_checkpoint(ckpt_path, model, opt, map_location=device)
        planner = GraphNetPlanner(
            model=model,
            device=str(device),
            deterministic=True,
            action_semantics=args.action_semantics,
            config=game_config,
        )
        summaries, rounds, players, edges, recommendations, aggregate = evaluate_planner(game, planner, args.n_eval_episodes, args.seed + 12345)
        aggregate["checkpoint"] = str(ckpt_path).replace("\\", "/")
        aggregate["checkpoint_rounds"] = int(payload.get("simulated_rounds", 0))
        rows.append(aggregate)
        episode_rows.extend(summaries)
        round_rows.extend(rounds)
        player_rows.extend(players)
        edge_rows.extend(edges)
        recommendation_rows.extend(recommendations)
        graphnet_summary = aggregate
        graphnet_summary["available"] = True
    eval_dir = root / args.eval_dir
    raw_dir = root / args.raw_dir
    ensure_dir(eval_dir)
    ensure_dir(raw_dir)
    rows_df = pd.DataFrame(rows)
    episode_df = pd.DataFrame(episode_rows)
    round_df = pd.DataFrame(round_rows)
    player_df = pd.DataFrame(player_rows)
    edge_df = pd.DataFrame(edge_rows)
    recommendation_df = pd.DataFrame(recommendation_rows)
    rows_df.to_csv(eval_dir / "planner_comparison.csv", index=False)
    round_df.groupby(["planner", "round"], as_index=False).mean(numeric_only=True).to_csv(eval_dir / "cooperation_by_round.csv", index=False)
    round_df.groupby(["planner", "round"], as_index=False).mean(numeric_only=True).to_csv(eval_dir / "capital_by_round.csv", index=False)
    round_df.groupby(["planner", "round"], as_index=False).mean(numeric_only=True).to_csv(eval_dir / "network_metrics_by_round.csv", index=False)
    if recommendation_df.empty:
        pd.DataFrame(columns=["planner", "round", "recommendation", "size"]).to_csv(eval_dir / "recommendation_rates.csv", index=False)
    else:
        recommendation_df.groupby(["planner", "round", "recommendation"], as_index=False).size().to_csv(eval_dir / "recommendation_rates.csv", index=False)
    player_df.groupby(["planner", "round", "cooperated"], as_index=False).mean(numeric_only=True).to_csv(eval_dir / "payoff_by_choice_and_round.csv", index=False)
    episode_df.to_parquet(raw_dir / "episodes.parquet", index=False)
    round_df.to_parquet(raw_dir / "rounds.parquet", index=False)
    player_df.to_parquet(raw_dir / "players.parquet", index=False)
    edge_df.to_parquet(raw_dir / "edges.parquet", index=False)
    recommendation_df.to_parquet(raw_dir / "recommendations.parquet", index=False)
    policy_gate = {}
    if graphnet_summary.get("available"):
        audit = build_table6_policy_audit(edge_df, recommendation_df, planner="graphnet")
        audit.to_csv(eval_dir / "table6_style_policy.csv", index=False)
        write_json(eval_dir / "table6_style_policy.json", audit.where(pd.notna(audit), None).to_dict(orient="records"))
        checkpoint_rounds = int(graphnet_summary.get("checkpoint_rounds", 0))
        max_status, max_notes = maximum_connectivity_diagnostic_status(root)
        if args.policy_audit_dir:
            policy_gate = write_policy_audit_outputs(
                audit,
                root / args.policy_audit_dir,
                checkpoint_rounds=checkpoint_rounds,
                maximum_connectivity_diagnostic=max_status,
                action_semantics=args.action_semantics,
                notes=max_notes,
            )
        elif args.eval_dir == "outputs/eval":
            policy_dir = ensure_dir(root / "outputs/policy_audit")
            audit.to_csv(policy_dir / "table6_style_final.csv", index=False)
            write_json(policy_dir / "table6_style_final.json", audit.where(pd.notna(audit), None).to_dict(orient="records"))
            policy_gate = write_policy_audit_outputs(
                audit,
                policy_dir,
                checkpoint_rounds=checkpoint_rounds,
                maximum_connectivity_diagnostic=max_status,
                action_semantics=args.action_semantics,
                notes=max_notes,
            )
    baseline_summary["_run_config"] = {"action_semantics": args.action_semantics, "feature_mode": args.feature_mode}
    if isinstance(graphnet_summary, dict):
        graphnet_summary["action_semantics"] = args.action_semantics
        graphnet_summary["feature_mode"] = args.feature_mode
    write_json(eval_dir / "baseline_eval_summary.json", baseline_summary)
    write_json(eval_dir / "graphnet_eval_summary.json", graphnet_summary)
    write_json(root / args.stage_summary, {"baseline": baseline_summary, "graphnet": graphnet_summary, "policy_alignment_gate": policy_gate})
    print({"planner_rows": len(rows), "episodes": len(episode_rows), "rounds": len(round_rows)})


if __name__ == "__main__":
    main()
