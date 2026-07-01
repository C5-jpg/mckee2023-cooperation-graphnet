from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .a2c import load_checkpoint
from .bots import SimulatedHumanBots
from .config import ACTION_SEMANTICS, FEATURE_MODES, BotConfig, GameConfig, GraphNetConfig, SEED, validate_action_semantics
from .env import CooperativeNetworkGame, GameState
from .graphnet import GraphNet, build_graph_batch
from .policy_audit import TABLE6_REFERENCE, direction_pass
from .utils import ensure_dir, write_json


def _pair_type(a_i: int, a_j: int) -> str:
    if a_i and a_j:
        return "C-C"
    if (not a_i) and (not a_j):
        return "D-D"
    return "C-D"


def policy_probability_fields(logit_pair: np.ndarray, active: bool, action_semantics: str) -> dict[str, float]:
    semantics = validate_action_semantics(action_semantics)
    shifted = logit_pair - np.max(logit_pair)
    probs = np.exp(shifted) / np.exp(shifted).sum()
    if semantics == "target_state":
        prob_recommend_change = float(probs[0] if active else probs[1])
        recommend_change_logit = float(logit_pair[0] if active else logit_pair[1])
        recommend_no_change_logit = float(logit_pair[1] if active else logit_pair[0])
    else:
        prob_recommend_change = float(probs[1])
        recommend_change_logit = float(logit_pair[1])
        recommend_no_change_logit = float(logit_pair[0])
    return {
        "prob_class0": float(probs[0]),
        "prob_class1": float(probs[1]),
        "prob_target_inactive": float(probs[0]) if semantics == "target_state" else float("nan"),
        "prob_target_active": float(probs[1]) if semantics == "target_state" else float("nan"),
        "prob_recommend_change": prob_recommend_change,
        "prob_change": prob_recommend_change,
        "recommend_change_logit": recommend_change_logit,
        "recommend_no_change_logit": recommend_no_change_logit,
        "logit_margin_recommend_change_minus_nochange": recommend_change_logit - recommend_no_change_logit,
        "logit_margin_change_minus_nochange": recommend_change_logit - recommend_no_change_logit,
    }


def policy_action_is_change(action: int, active: bool, action_semantics: str) -> int:
    semantics = validate_action_semantics(action_semantics)
    if semantics == "target_state":
        return int((active and int(action) == 0) or ((not active) and int(action) == 1))
    return int(int(action) == 1)


def diagnose_checkpoint(
    checkpoint: str | Path,
    out_dir: str | Path,
    n_episodes: int = 128,
    seed: int = SEED,
    device: str | None = None,
    stochastic_policy: bool = False,
    action_semantics: str = "target_state",
    feature_mode: str = "augmented",
) -> dict[str, Any]:
    semantics = validate_action_semantics(action_semantics)
    cfg = GameConfig(action_semantics=semantics)
    game = CooperativeNetworkGame(cfg)
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GraphNet(cfg, GraphNetConfig(feature_mode=feature_mode)).to(device_obj)
    opt = torch.optim.Adam(model.parameters(), lr=0.0004)
    payload = load_checkpoint(checkpoint, model, opt, map_location=device_obj)
    model.eval()

    opportunity_rows: list[dict[str, Any]] = []
    rec_rows: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        episode_seed = seed + ep * 7919
        rng = np.random.default_rng(episode_seed)
        bots = SimulatedHumanBots(cfg.n_players, BotConfig(), cfg)
        bots.reset_episode(rng)
        state = game.reset(episode_seed)
        while not state.done:
            transition = game.cooperation_transition(state, bots, rng)
            planner_state = transition.planner_state
            if state.round_index < cfg.episode_length:
                batch = build_graph_batch([planner_state], cfg, device=device_obj, feature_mode=feature_mode)
                with torch.no_grad():
                    logits, _ = model(batch)
                    probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
                if stochastic_policy:
                    target_active = np.asarray([rng.choice(2, p=prob) for prob in probs], dtype=np.int64)
                else:
                    target_active = probs.argmax(axis=-1).astype(np.int64)
                if semantics == "target_state":
                    recommendations = game.target_edge_states_to_recommendations(target_active, planner_state.adjacency)
                else:
                    recommendations = game.binary_changes_to_recommendations(target_active, planner_state.adjacency)

                edge_context: dict[tuple[int, int], dict[str, Any]] = {}
                for idx, (i, j) in enumerate(game.edges):
                    pair = _pair_type(int(planner_state.prev_actions[i]), int(planner_state.prev_actions[j]))
                    current_active = bool(planner_state.adjacency[i, j])
                    change = "delete" if current_active else "add"
                    fields = policy_probability_fields(logits[0, idx].detach().cpu().numpy(), current_active, semantics)
                    deterministic_action = int(np.argmax(probs[idx]))
                    deterministic_change = policy_action_is_change(deterministic_action, current_active, semantics)
                    policy_change = policy_action_is_change(int(target_active[idx]), current_active, semantics)
                    row = {
                        "episode_id": ep,
                        "round": planner_state.round_index,
                        "i": i,
                        "j": j,
                        "pair_type": pair,
                        "change_type": change,
                        "edge_active_before": bool(planner_state.adjacency[i, j]),
                        "action_semantics": semantics,
                        "deterministic_change": deterministic_change,
                        "policy_class": int(target_active[idx]),
                        "policy_target_active": int(target_active[idx]) if semantics == "target_state" else None,
                        "policy_action_change": policy_change,
                    }
                    row.update(fields)
                    opportunity_rows.append(row)
                    edge_context[(i, j)] = row

                adjudication = game.adjudicate_recommendations(
                    planner_state,
                    recommendations,
                    bots,
                    rng,
                    episode_id=ep,
                    planner_name="graphnet_diagnostic",
                )
                for rec in adjudication.recommendation_records:
                    ctx = edge_context[(int(rec["i"]), int(rec["j"]))]
                    rec_rows.append(
                        {
                            "episode_id": ep,
                            "round": rec["round"],
                            "i": rec["i"],
                            "j": rec["j"],
                            "pair_type": ctx["pair_type"],
                            "change_type": ctx["change_type"],
                            "recommendation": rec["recommendation"],
                            "accepted": bool(rec["accepted"]),
                            "referent_cooperated": rec["referent_cooperated"],
                            "action_semantics": semantics,
                            "prob_change": ctx["prob_change"],
                            "prob_recommend_change": ctx["prob_recommend_change"],
                            "logit_margin_change_minus_nochange": ctx["logit_margin_change_minus_nochange"],
                            "logit_margin_recommend_change_minus_nochange": ctx["logit_margin_recommend_change_minus_nochange"],
                        }
                    )
                next_adjacency = adjudication.next_adjacency
            else:
                next_adjacency = planner_state.adjacency.copy()
            done = state.round_index >= cfg.episode_length
            state = GameState(
                round_index=state.round_index + 1,
                adjacency=next_adjacency,
                capital=planner_state.capital.copy(),
                prev_actions=planner_state.prev_actions.copy(),
                seed=state.seed,
                done=done,
            )

    out = ensure_dir(out_dir)
    opp_df = pd.DataFrame(opportunity_rows)
    rec_df = pd.DataFrame(rec_rows)
    opp_df.to_csv(out / "policy_opportunity_logits.csv", index=False)
    rec_df.to_csv(out / "policy_recommendation_acceptance.csv", index=False)

    grouped = (
        opp_df.groupby(["pair_type", "change_type"], as_index=False)
        .agg(
            chances=("prob_change", "size"),
            mean_prob_change=("prob_change", "mean"),
            mean_prob_recommend_change=("prob_recommend_change", "mean"),
            median_prob_change=("prob_change", "median"),
            p10_prob_change=("prob_change", lambda s: float(np.quantile(s, 0.10))),
            p90_prob_change=("prob_change", lambda s: float(np.quantile(s, 0.90))),
            mean_logit_margin=("logit_margin_change_minus_nochange", "mean"),
            mean_recommend_change_logit_margin=("logit_margin_recommend_change_minus_nochange", "mean"),
            deterministic_change_rate=("deterministic_change", "mean"),
            policy_action_change_rate=("policy_action_change", "mean"),
        )
        .sort_values(["pair_type", "change_type"])
    )
    grouped["paper_rate_reference"] = grouped.apply(
        lambda r: TABLE6_REFERENCE[(str(r["pair_type"]), str(r["change_type"]))],
        axis=1,
    )
    grouped["direction_pass_deterministic"] = grouped.apply(
        lambda r: direction_pass(str(r["pair_type"]), str(r["change_type"]), float(r["deterministic_change_rate"])),
        axis=1,
    )
    grouped["direction_pass_probability"] = grouped.apply(
        lambda r: direction_pass(str(r["pair_type"]), str(r["change_type"]), float(r["mean_prob_change"])),
        axis=1,
    )
    grouped["action_semantics"] = semantics
    grouped.to_csv(out / "policy_probability_by_type.csv", index=False)

    if rec_df.empty:
        acceptance = pd.DataFrame(columns=["pair_type", "change_type", "recommendations", "acceptance_rate"])
    else:
        acceptance = (
            rec_df.groupby(["pair_type", "change_type"], as_index=False)
            .agg(
                recommendations=("accepted", "size"),
                acceptance_rate=("accepted", "mean"),
                mean_prob_change=("prob_change", "mean"),
                mean_prob_recommend_change=("prob_recommend_change", "mean"),
            )
            .sort_values(["pair_type", "change_type"])
        )
    acceptance.to_csv(out / "policy_acceptance_by_type.csv", index=False)

    summary = {
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "checkpoint_rounds": int(payload.get("simulated_rounds", 0)),
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "device": str(device_obj),
        "stochastic_policy": bool(stochastic_policy),
        "action_semantics": semantics,
        "feature_mode": feature_mode,
        "rows": {
            "opportunities": int(len(opp_df)),
            "recommendations": int(len(rec_df)),
        },
    }
    write_json(out / "policy_diagnostics_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/train/checkpoints/latest.pt")
    parser.add_argument("--out-dir", default="outputs/policy_diagnostics")
    parser.add_argument("--n-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=GraphNetConfig().feature_mode)
    args = parser.parse_args()
    summary = diagnose_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=args.device,
        stochastic_policy=args.stochastic_policy,
        action_semantics=args.action_semantics,
        feature_mode=args.feature_mode,
    )
    print(summary)


if __name__ == "__main__":
    main()
