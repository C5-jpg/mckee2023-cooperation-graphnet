from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .a2c import a2c_loss, load_checkpoint, save_checkpoint
from .bots import SimulatedHumanBots
from .config import ACTION_SEMANTICS, FEATURE_MODES, BotConfig, GameConfig, GraphNetConfig, ReproConfig, SEED, TrainingConfig
from .data_logging import create_output_dirs, write_figures_readme, write_input_manifest, write_resolved_config
from .env import CooperativeNetworkGame, GameState
from .graphnet import GraphNet, build_graph_batch
from .utils import append_jsonl, ensure_dir, seed_everything, utc_now, write_json


def joint_action_log_prob_and_entropy(
    dist: torch.distributions.Categorical,
    binary_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dist.log_prob(binary_actions).sum(), dist.entropy().sum()


class TrainingRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.root)
        self.game_config = GameConfig(action_semantics=args.action_semantics)
        self.bot_config = BotConfig()
        self.training_config = TrainingConfig(seed=args.seed, target_simulated_rounds=args.rounds)
        self.graphnet_config = GraphNetConfig(feature_mode=args.feature_mode)
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        seed_everything(args.seed)
        self.game = CooperativeNetworkGame(self.game_config)
        self.model = GraphNet(self.game_config, self.graphnet_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)
        self.start_rounds = 0
        self.start_episodes = 0
        if args.resume:
            payload = load_checkpoint(args.resume, self.model, self.optimizer, map_location=self.device)
            self.start_rounds = int(payload.get("simulated_rounds", 0))
            self.start_episodes = int(payload.get("episodes", 0))

    def rollout_episode(self, seed: int, episode_id: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        bots = SimulatedHumanBots(self.game_config.n_players, self.bot_config, self.game_config)
        bots.reset_episode(rng)
        state = self.game.reset(seed)
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []
        round_records = []
        recommendation_records = []
        sampled_round_records = []
        sampled_recommendations = []
        while not state.done:
            transition = self.game.cooperation_transition(state, bots, rng)
            planner_state = transition.planner_state
            if state.round_index < self.game_config.episode_length:
                batch = build_graph_batch([planner_state], self.game_config, device=self.device, feature_mode=self.graphnet_config.feature_mode)
                logits, value = self.model(batch)
                dist = torch.distributions.Categorical(logits=logits[0])
                edge_actions = dist.sample()
                log_prob, entropy = joint_action_log_prob_and_entropy(dist, edge_actions)
                if self.game_config.action_semantics == "target_state":
                    recommendations = self.game.target_edge_states_to_recommendations(edge_actions.detach().cpu().numpy(), planner_state.adjacency)
                else:
                    recommendations = self.game.binary_changes_to_recommendations(edge_actions.detach().cpu().numpy(), planner_state.adjacency)
                adjudication = self.game.adjudicate_recommendations(
                    planner_state,
                    recommendations,
                    bots,
                    rng,
                    episode_id=episode_id,
                    planner_name="graphnet_train",
                )
                penalty = adjudication.rejected_nonzero / self.game_config.n_possible_edges
                reward = float(np.mean(planner_state.capital) - self.game_config.recommendation_penalty_weight * penalty)
                log_probs.append(log_prob)
                values.append(value.squeeze(0))
                entropies.append(entropy)
                rewards.append(reward)
                recommendation_records.extend(adjudication.recommendation_records)
                next_adjacency = adjudication.next_adjacency
                nonzero = adjudication.nonzero_recommendations
                rejected = adjudication.rejected_nonzero
            else:
                penalty = 0.0
                reward = float(np.mean(planner_state.capital))
                if rewards:
                    rewards[-1] += self.training_config.discount_gamma * reward
                next_adjacency = planner_state.adjacency.copy()
                nonzero = 0
                rejected = 0
            round_record = {
                "episode_id": episode_id,
                "round": state.round_index,
                "planner": "graphnet_train",
                "seed": seed,
                "mean_cooperation": float(np.mean(planner_state.prev_actions)),
                "mean_capital": float(np.mean(planner_state.capital)),
                "network_density": float(next_adjacency.sum() / (self.game_config.n_players * (self.game_config.n_players - 1))),
                "reward": reward,
                "recommendation_penalty": penalty,
                "nonzero_recommendations": nonzero,
                "rejected_recommendations": rejected,
            }
            round_records.append(round_record)
            if episode_id % self.training_config.train_sample_interval_episodes == 0:
                sampled_round_records.append(round_record)
                sampled_recommendations.extend(recommendation_records[-nonzero:] if nonzero else [])
            done = state.round_index >= self.game_config.episode_length
            state = GameState(
                round_index=state.round_index + 1,
                adjacency=next_adjacency,
                capital=planner_state.capital.copy(),
                prev_actions=planner_state.prev_actions.copy(),
                seed=seed,
                done=done,
            )
        if log_probs:
            loss_out = a2c_loss(
                torch.stack(log_probs),
                torch.stack(values),
                rewards,
                torch.stack(entropies),
                gamma=self.training_config.discount_gamma,
                baseline_cost=self.training_config.baseline_cost,
                entropy_coeff=self.training_config.entropy_regularization,
            )
        else:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_out = a2c_loss(torch.zeros(0, device=self.device), zero.reshape(0), [], torch.zeros(0, device=self.device))
        return {
            "loss_out": loss_out,
            "round_records": round_records,
            "sampled_round_records": sampled_round_records,
            "sampled_recommendations": sampled_recommendations,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_cooperation": float(np.mean([r["mean_cooperation"] for r in round_records])) if round_records else 0.0,
            "mean_capital": float(np.mean([r["mean_capital"] for r in round_records])) if round_records else 0.0,
            "network_density": float(np.mean([r["network_density"] for r in round_records])) if round_records else 0.0,
            "recommendation_penalty": float(np.mean([r["recommendation_penalty"] for r in round_records])) if round_records else 0.0,
            "final_state": state,
        }

    def rollout_batch(self, start_episode: int, n_episodes: int) -> Dict[str, Any]:
        episode_ids = list(range(start_episode, start_episode + n_episodes))
        episode_seeds = [self.args.seed + episode_id * 9973 for episode_id in episode_ids]
        rngs = [np.random.default_rng(seed) for seed in episode_seeds]
        bots_list = [SimulatedHumanBots(self.game_config.n_players, self.bot_config, self.game_config) for _ in episode_ids]
        states = []
        for seed, rng, bots in zip(episode_seeds, rngs, bots_list):
            bots.reset_episode(rng)
            states.append(self.game.reset(seed))

        log_probs: list[list[torch.Tensor]] = [[] for _ in episode_ids]
        values: list[list[torch.Tensor]] = [[] for _ in episode_ids]
        entropies: list[list[torch.Tensor]] = [[] for _ in episode_ids]
        rewards: list[list[float]] = [[] for _ in episode_ids]
        round_records: list[list[dict[str, Any]]] = [[] for _ in episode_ids]
        recommendation_records: list[list[dict[str, Any]]] = [[] for _ in episode_ids]
        sampled_round_records: list[dict[str, Any]] = []
        sampled_recommendations: list[dict[str, Any]] = []

        for _ in range(self.game_config.episode_length):
            transitions = [
                self.game.cooperation_transition(state, bots, rng)
                for state, bots, rng in zip(states, bots_list, rngs)
            ]
            planner_states = [transition.planner_state for transition in transitions]
            is_action_round = states[0].round_index < self.game_config.episode_length
            if is_action_round:
                batch = build_graph_batch(planner_states, self.game_config, device=self.device, feature_mode=self.graphnet_config.feature_mode)
                logits, value_batch = self.model(batch)
                dist = torch.distributions.Categorical(logits=logits)
                edge_actions = dist.sample()
                log_prob_batch = dist.log_prob(edge_actions).sum(dim=1)
                entropy_batch = dist.entropy().sum(dim=1)
                action_np = edge_actions.detach().cpu().numpy()
            else:
                value_batch = None
                log_prob_batch = None
                entropy_batch = None
                action_np = None

            next_states: list[GameState] = []
            for idx, (episode_id, seed, state, planner_state, transition, bots, rng) in enumerate(
                zip(episode_ids, episode_seeds, states, planner_states, transitions, bots_list, rngs)
            ):
                if is_action_round:
                    if self.game_config.action_semantics == "target_state":
                        recommendations = self.game.target_edge_states_to_recommendations(action_np[idx], planner_state.adjacency)
                    else:
                        recommendations = self.game.binary_changes_to_recommendations(action_np[idx], planner_state.adjacency)
                    adjudication = self.game.adjudicate_recommendations(
                        planner_state,
                        recommendations,
                        bots,
                        rng,
                        episode_id=episode_id,
                        planner_name="graphnet_train",
                    )
                    penalty = adjudication.rejected_nonzero / self.game_config.n_possible_edges
                    reward = float(np.mean(planner_state.capital) - self.game_config.recommendation_penalty_weight * penalty)
                    log_probs[idx].append(log_prob_batch[idx])
                    values[idx].append(value_batch[idx])
                    entropies[idx].append(entropy_batch[idx])
                    rewards[idx].append(reward)
                    new_recommendations = adjudication.recommendation_records
                    recommendation_records[idx].extend(new_recommendations)
                    next_adjacency = adjudication.next_adjacency
                    nonzero = adjudication.nonzero_recommendations
                    rejected = adjudication.rejected_nonzero
                else:
                    penalty = 0.0
                    reward = float(np.mean(planner_state.capital))
                    if rewards[idx]:
                        rewards[idx][-1] += self.training_config.discount_gamma * reward
                    new_recommendations = []
                    next_adjacency = planner_state.adjacency.copy()
                    nonzero = 0
                    rejected = 0

                round_record = {
                    "episode_id": episode_id,
                    "round": state.round_index,
                    "planner": "graphnet_train",
                    "seed": seed,
                    "mean_cooperation": float(np.mean(planner_state.prev_actions)),
                    "mean_capital": float(np.mean(planner_state.capital)),
                    "network_density": float(next_adjacency.sum() / (self.game_config.n_players * (self.game_config.n_players - 1))),
                    "reward": reward,
                    "recommendation_penalty": penalty,
                    "nonzero_recommendations": nonzero,
                    "rejected_recommendations": rejected,
                }
                round_records[idx].append(round_record)
                if episode_id % self.training_config.train_sample_interval_episodes == 0:
                    sampled_round_records.append(round_record)
                    sampled_recommendations.extend(new_recommendations)
                done = state.round_index >= self.game_config.episode_length
                next_states.append(
                    GameState(
                        round_index=state.round_index + 1,
                        adjacency=next_adjacency,
                        capital=planner_state.capital.copy(),
                        prev_actions=planner_state.prev_actions.copy(),
                        seed=seed,
                        done=done,
                    )
                )
            states = next_states

        episode_metrics = []
        losses = []
        policy_losses = []
        value_losses = []
        entropy_values = []
        for idx, state in enumerate(states):
            if log_probs[idx]:
                loss_out = a2c_loss(
                    torch.stack(log_probs[idx]),
                    torch.stack(values[idx]),
                    rewards[idx],
                    torch.stack(entropies[idx]),
                    gamma=self.training_config.discount_gamma,
                    baseline_cost=self.training_config.baseline_cost,
                    entropy_coeff=self.training_config.entropy_regularization,
                )
            else:
                zero = torch.tensor(0.0, device=self.device, requires_grad=True)
                loss_out = a2c_loss(torch.zeros(0, device=self.device), zero.reshape(0), [], torch.zeros(0, device=self.device))
            losses.append(loss_out.loss)
            policy_losses.append(float(loss_out.policy_loss.detach().cpu()))
            value_losses.append(float(loss_out.value_loss.detach().cpu()))
            entropy_values.append(float(loss_out.entropy.detach().cpu()) if loss_out.entropy.numel() else 0.0)
            records = round_records[idx]
            episode_metrics.append(
                {
                    "loss_out": loss_out,
                    "round_records": records,
                    "mean_reward": float(np.mean(rewards[idx])) if rewards[idx] else 0.0,
                    "mean_cooperation": float(np.mean([r["mean_cooperation"] for r in records])) if records else 0.0,
                    "mean_capital": float(np.mean([r["mean_capital"] for r in records])) if records else 0.0,
                    "network_density": float(np.mean([r["network_density"] for r in records])) if records else 0.0,
                    "recommendation_penalty": float(np.mean([r["recommendation_penalty"] for r in records])) if records else 0.0,
                    "final_state": state,
                }
            )

        return {
            "n_episodes": n_episodes,
            "losses": losses,
            "policy_losses": policy_losses,
            "value_losses": value_losses,
            "entropy_values": entropy_values,
            "episode_metrics": episode_metrics,
            "sampled_round_records": sampled_round_records,
            "sampled_recommendations": sampled_recommendations,
        }

    def checkpoint(self, simulated_rounds: int, episodes: int, metrics: dict) -> str:
        ckpt_dir = self.root / "outputs/train/checkpoints"
        path = ckpt_dir / f"step_{simulated_rounds:07d}.pt"
        save_checkpoint(path, self.model, self.optimizer, simulated_rounds, episodes, self.args.seed, extra=metrics)
        latest = ckpt_dir / "latest.pt"
        save_checkpoint(latest, self.model, self.optimizer, simulated_rounds, episodes, self.args.seed, extra=metrics)
        return str(latest).replace("\\", "/")

    def run(self) -> dict:
        seed_everything(self.args.seed)
        create_output_dirs(self.root)
        write_resolved_config(
            self.root,
            ReproConfig(
                seed=self.args.seed,
                game=self.game_config,
                training=self.training_config,
                graphnet=self.graphnet_config,
            ),
        )
        write_input_manifest(self.root)
        write_figures_readme(self.root)
        target_rounds = self.args.rounds
        target_episodes = math.ceil(target_rounds / self.game_config.episode_length)
        max_wall_seconds = self.args.max_wall_minutes * 60 if self.args.max_wall_minutes else None
        start_time = time.time()
        simulated_rounds = self.start_rounds
        episodes = self.start_episodes
        sampled_rounds: list[dict[str, Any]] = []
        sampled_recommendations: list[dict[str, Any]] = []
        last_checkpoint = None
        last_checkpoint_rounds = self.start_rounds
        next_checkpoint_round = (
            (self.start_rounds // self.args.checkpoint_interval_rounds) + 1
        ) * self.args.checkpoint_interval_rounds
        while simulated_rounds < target_rounds and episodes < target_episodes:
            if max_wall_seconds and time.time() - start_time > max_wall_seconds:
                break
            batch_n = min(self.args.batch_size, target_episodes - episodes)
            result = self.rollout_batch(episodes, batch_n)
            sampled_rounds.extend(result["sampled_round_records"])
            sampled_recommendations.extend(result["sampled_recommendations"])
            simulated_rounds += result["n_episodes"] * self.game_config.episode_length
            episodes += result["n_episodes"]

            self.optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(result["losses"]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite A2C loss at {simulated_rounds} simulated rounds: {float(loss.detach().cpu())}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm).detach().cpu())
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at {simulated_rounds} simulated rounds: {grad_norm}")
            self.optimizer.step()
            recent = result["episode_metrics"]
            metrics = {
                "timestamp_utc": utc_now(),
                "seed": self.args.seed,
                "simulated_rounds": simulated_rounds,
                "episodes": episodes,
                "mean_reward": float(np.mean([x["mean_reward"] for x in recent])),
                "mean_cooperation": float(np.mean([x["mean_cooperation"] for x in recent])),
                "mean_capital": float(np.mean([x["mean_capital"] for x in recent])),
                "network_density": float(np.mean([x["network_density"] for x in recent])),
                "recommendation_penalty": float(np.mean([x["recommendation_penalty"] for x in recent])),
                "policy_loss": float(np.mean(result["policy_losses"])),
                "value_loss": float(np.mean(result["value_losses"])),
                "entropy": float(np.mean(result["entropy_values"])),
                "grad_norm": grad_norm,
                "rollout_mode": "batched",
                "action_semantics": self.game_config.action_semantics,
                "feature_mode": self.graphnet_config.feature_mode,
                "batch_episodes": result["n_episodes"],
                "checkpoint_path": None,
            }
            if simulated_rounds >= next_checkpoint_round or simulated_rounds >= target_rounds:
                last_checkpoint = self.checkpoint(simulated_rounds, episodes, metrics)
                last_checkpoint_rounds = simulated_rounds
                metrics["checkpoint_path"] = last_checkpoint
                while next_checkpoint_round <= simulated_rounds:
                    next_checkpoint_round += self.args.checkpoint_interval_rounds
            append_jsonl(self.root / "outputs/train/train_metrics.jsonl", metrics)
            append_jsonl(
                self.root / "outputs/train/loss_metrics.jsonl",
                {
                    "timestamp_utc": metrics["timestamp_utc"],
                    "simulated_rounds": simulated_rounds,
                    "episodes": episodes,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "grad_norm": grad_norm,
                    "rollout_mode": "batched",
                    "action_semantics": self.game_config.action_semantics,
                    "feature_mode": self.graphnet_config.feature_mode,
                },
            )
            write_json(
                self.root / "outputs/train/heartbeat.json",
                {
                    "timestamp_utc": utc_now(),
                    "seed": self.args.seed,
                    "target_simulated_rounds": target_rounds,
                    "completed_simulated_rounds": simulated_rounds,
                    "completed_episodes": episodes,
                    "last_checkpoint": last_checkpoint,
                    "mean_reward_recent": metrics["mean_reward"],
                    "mean_cooperation_recent": metrics["mean_cooperation"],
                    "policy_loss_recent": metrics["policy_loss"],
                    "value_loss_recent": metrics["value_loss"],
                    "entropy_recent": metrics["entropy"],
                    "rollout_mode": "batched",
                    "action_semantics": self.game_config.action_semantics,
                    "feature_mode": self.graphnet_config.feature_mode,
                },
            )
        if last_checkpoint is None or last_checkpoint_rounds < simulated_rounds:
            final_metrics = {"simulated_rounds": simulated_rounds, "episodes": episodes}
            last_checkpoint = self.checkpoint(simulated_rounds, episodes, final_metrics)
        if sampled_rounds:
            import pandas as pd

            pd.DataFrame(sampled_rounds).to_parquet(self.root / "outputs/raw_train_sample/sampled_rounds.parquet", index=False)
        else:
            import pandas as pd

            pd.DataFrame([]).to_parquet(self.root / "outputs/raw_train_sample/sampled_rounds.parquet", index=False)
        if sampled_recommendations:
            import pandas as pd

            pd.DataFrame(sampled_recommendations).to_parquet(self.root / "outputs/raw_train_sample/sampled_recommendations.parquet", index=False)
        else:
            import pandas as pd

            pd.DataFrame([]).to_parquet(self.root / "outputs/raw_train_sample/sampled_recommendations.parquet", index=False)
        summary = {
            "mode": self.args.mode,
            "seed": self.args.seed,
            "target_simulated_rounds": target_rounds,
            "completed_simulated_rounds": simulated_rounds,
            "completed_episodes": episodes,
            "target_finished": simulated_rounds >= target_rounds,
            "last_checkpoint": last_checkpoint,
            "device": str(self.device),
            "action_semantics": self.game_config.action_semantics,
            "feature_mode": self.graphnet_config.feature_mode,
            "wall_seconds": round(time.time() - start_time, 3),
        }
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "short", "target", "custom"], default="smoke")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-interval-rounds", type=int, default=10_000)
    parser.add_argument("--max-wall-minutes", type=float, default=None)
    parser.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=GraphNetConfig().feature_mode)
    args = parser.parse_args()
    defaults = {"smoke": 150, "short": 3000, "target": 1_000_000, "custom": 150}
    if args.rounds is None:
        args.rounds = defaults[args.mode]
    if args.batch_size is None:
        args.batch_size = 4 if args.mode == "smoke" else 8 if args.mode == "short" else 32
    return args


def main() -> None:
    args = parse_args()
    summary = TrainingRun(args).run()
    stage_dir = Path(args.root) / ("stage06_smoke_train" if args.mode in {"smoke", "short"} else "stage07_target_train_eval")
    ensure_dir(stage_dir)
    if args.mode == "smoke":
        write_json(stage_dir / "smoke_summary.json", summary)
    elif args.mode == "short":
        write_json(stage_dir / "short_train_summary.json", summary)
    elif args.mode == "target":
        write_json(stage_dir / "target_train_summary.json", summary)
    else:
        write_json(stage_dir / "custom_train_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
