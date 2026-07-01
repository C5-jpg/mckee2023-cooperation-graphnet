from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .bots import SimulatedHumanBots
from .config import BotConfig, GameConfig, SEED
from .env import CooperativeNetworkGame
from .metrics import final_episode_summary
from .planners import all_rule_based_planners
from .utils import ensure_dir


def run_rule_based_diagnostics(root: Path, n_episodes: int, seed: int) -> None:
    game_config = GameConfig()
    game = CooperativeNetworkGame(game_config)
    summaries = []
    round_rows = []
    player_rows = []
    for planner in all_rule_based_planners(game_config):
        for ep in range(n_episodes):
            bots = SimulatedHumanBots(game_config.n_players, BotConfig(), game_config)
            episode = game.run_episode(planner, bots, seed + ep * 7919, episode_id=ep)
            summary = final_episode_summary(episode, game_config)
            summary.update({"planner": planner.name, "episode_id": ep, "seed": seed})
            summaries.append(summary)
            round_rows.extend(episode["rounds"])
            player_rows.extend(episode["players"])

    out_dir = ensure_dir(root / "outputs/diagnostics")
    summary_df = pd.DataFrame(summaries)
    round_df = pd.DataFrame(round_rows)
    player_df = pd.DataFrame(player_rows)
    summary_df.to_csv(out_dir / "maximum_connectivity_diagnostics.csv", index=False)
    calibration = (
        player_df.groupby(["planner", "round"], as_index=False)
        .agg(
            cooperation_rate=("cooperated", "mean"),
            mean_degree=("degree", "mean"),
            mean_capital=("capital", "mean"),
            mean_cooperation_probability=("cooperation_probability", "mean"),
            capital_constrained_rate=("capital_constrained", "mean"),
        )
        .merge(
            round_df.groupby(["planner", "round"], as_index=False).agg(
                network_density=("network_density", "mean"),
                rejected_recommendations=("rejected_recommendations", "mean"),
                nonzero_recommendations=("nonzero_recommendations", "mean"),
            ),
            on=["planner", "round"],
            how="left",
        )
    )
    calibration.to_csv(out_dir / "bot_calibration_diagnostics.csv", index=False)
    payoff_doc = """# Payoff 与 Capital 顺序核查

本诊断按当前实现采用的论文游戏顺序解释 payoff/capital：

1. 玩家在本轮开始时的 graph 上选择 cooperate/defect。
2. payoff 按 `benefit_b * cooperating_neighbors - cost_c * own_action * degree` 计算。
3. payoff 后立即更新 capital。
4. planner 观察当前轮 choices 和更新后的 capital。
5. accepted recommendations 影响下一轮 graph。

在 16 人完全图中，一个合作者支付 `0.05 * 15 = 0.75`，每个合作邻居提供 `0.10`。如果 16 人全合作，每人本轮净 payoff 为 `1.5 - 0.75 = 0.75`。

因此，如果 maximum connectivity 诊断中合作仍然很高，直接 payoff 方程本身可以维持较高 capital；真正需要继续排查的是 bot logistic 方程在高 degree 完全图下是否应当让未来合作率下降到论文真人验证实验的量级。
"""
    (out_dir / "payoff_capital_order_check.md").write_text(payoff_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_rule_based_diagnostics(Path(args.root), args.n_episodes, args.seed)
    print({"diagnostic_episodes_per_planner": args.n_episodes})


if __name__ == "__main__":
    main()
