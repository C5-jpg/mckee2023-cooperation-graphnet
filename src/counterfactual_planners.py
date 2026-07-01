from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .a2c import load_checkpoint
from .bots import SimulatedHumanBots
from .config import FEATURE_MODES, BotConfig, GameConfig, GraphNetConfig, SEED
from .env import CooperativeNetworkGame, GameState
from .graphnet import GraphNet, build_graph_batch
from .metrics import final_episode_summary
from .policy_audit import build_table6_policy_audit
from .utils import ensure_dir, write_json


class GraphNetCounterfactualPlanner:
    def __init__(self, model: GraphNet, config: GameConfig, device: str, mode: str) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.mode = mode
        self.name = f"graphnet_{mode}"
        self.edges = CooperativeNetworkGame(config).edges

    def reset_episode(self, state: GameState, rng: np.random.Generator) -> None:
        return None

    def recommend(self, state: GameState, rng: np.random.Generator) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            batch = build_graph_batch([state], self.config, device=self.device, feature_mode=getattr(self.model, "feature_mode", "augmented"))
            logits, _ = self.model(batch)
            target_active = logits.argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int64)
        recs = np.zeros(len(self.edges), dtype=np.int64)
        for idx, (i, j) in enumerate(self.edges):
            desired_active = bool(int(target_active[idx]))
            current_active = bool(state.adjacency[i, j])
            if desired_active and not current_active:
                recs[idx] = 1
            elif current_active and not desired_active:
                recs[idx] = -1

        if self.mode == "force_dd_delete":
            for idx, (i, j) in enumerate(self.edges):
                if state.adjacency[i, j] and state.prev_actions[i] == 0 and state.prev_actions[j] == 0:
                    recs[idx] = -1
                elif (not state.adjacency[i, j]) and state.prev_actions[i] == 0 and state.prev_actions[j] == 0:
                    recs[idx] = 0
        elif self.mode == "force_table6_direction":
            for idx, (i, j) in enumerate(self.edges):
                ai = int(state.prev_actions[i])
                aj = int(state.prev_actions[j])
                if ai == 1 and aj == 1 and not state.adjacency[i, j]:
                    recs[idx] = 1
                elif ai == 1 and aj == 1 and state.adjacency[i, j]:
                    recs[idx] = 0
                elif ai == 0 and aj == 0 and state.adjacency[i, j]:
                    recs[idx] = -1
                elif ai == 0 and aj == 0 and not state.adjacency[i, j]:
                    recs[idx] = 0
        elif self.mode == "suppress_dd_change":
            for idx, (i, j) in enumerate(self.edges):
                if state.prev_actions[i] == 0 and state.prev_actions[j] == 0:
                    recs[idx] = 0
        return recs


def evaluate_counterfactuals(
    checkpoint: str | Path,
    out_dir: str | Path,
    n_episodes: int = 128,
    seed: int = SEED,
    device: str | None = None,
    feature_mode: str = GraphNetConfig().feature_mode,
) -> dict[str, Any]:
    cfg = GameConfig()
    game = CooperativeNetworkGame(cfg)
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GraphNet(cfg, GraphNetConfig(feature_mode=feature_mode)).to(device_obj)
    opt = torch.optim.Adam(model.parameters(), lr=0.0004)
    payload = load_checkpoint(checkpoint, model, opt, map_location=device_obj)
    modes = ["baseline", "force_dd_delete", "suppress_dd_change", "force_table6_direction"]

    all_summaries: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    all_recs: list[dict[str, Any]] = []
    for mode in modes:
        planner_mode = "force_dd_delete" if mode == "baseline" else mode
        planner = GraphNetCounterfactualPlanner(model, cfg, str(device_obj), planner_mode)
        if mode == "baseline":
            planner.name = "graphnet_baseline_argmax"
            planner.mode = "baseline"
        for ep in range(n_episodes):
            bots = SimulatedHumanBots(cfg.n_players, BotConfig(), cfg)
            episode = game.run_episode(planner, bots, seed + ep * 7919, episode_id=ep)
            summary = final_episode_summary(episode, cfg)
            summary.update({"planner": planner.name, "episode_id": ep, "seed": seed + ep * 7919})
            all_summaries.append(summary)
            all_edges.extend(episode["edges"])
            all_recs.extend(episode["recommendations"])

    out = ensure_dir(out_dir)
    summary_df = pd.DataFrame(all_summaries)
    edges_df = pd.DataFrame(all_edges)
    recs_df = pd.DataFrame(all_recs)
    aggregate = (
        summary_df.groupby("planner", as_index=False)
        .mean(numeric_only=True)
        .sort_values("planner")
    )
    aggregate.to_csv(out / "counterfactual_planner_summary.csv", index=False)
    audits = []
    for planner in sorted(summary_df["planner"].unique()):
        audit = build_table6_policy_audit(edges_df, recs_df, planner=planner)
        audit.insert(0, "planner", planner)
        audits.append(audit)
    audit_df = pd.concat(audits, ignore_index=True)
    audit_df.to_csv(out / "counterfactual_table6_policy.csv", index=False)
    details = {
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "checkpoint_rounds": int(payload.get("simulated_rounds", 0)),
        "n_episodes": int(n_episodes),
        "seed": int(seed),
            "device": str(device_obj),
            "action_semantics": "target_state",
            "feature_mode": feature_mode,
            "note": "Counterfactual overrides are diagnostics only and are not valid GraphNet reproduction results.",
    }
    write_json(out / "counterfactual_summary.json", details)
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/train/checkpoints/latest.pt")
    parser.add_argument("--out-dir", default="outputs/counterfactual_policy")
    parser.add_argument("--n-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=GraphNetConfig().feature_mode)
    args = parser.parse_args()
    print(
        evaluate_counterfactuals(
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            n_episodes=args.n_episodes,
            seed=args.seed,
            device=args.device,
            feature_mode=args.feature_mode,
        )
    )


if __name__ == "__main__":
    main()
