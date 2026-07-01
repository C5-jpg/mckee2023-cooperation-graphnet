from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .a2c import load_checkpoint, save_checkpoint
from .bots import SimulatedHumanBots
from .config import ACTION_SEMANTICS, FEATURE_MODES, BotConfig, GameConfig, GraphNetConfig, ReproConfig, SEED, TrainingConfig
from .data_logging import create_output_dirs, write_figures_readme, write_input_manifest, write_resolved_config
from .env import CooperativeNetworkGame, GameState
from .graphnet import GraphNet, build_graph_batch
from .utils import append_jsonl, ensure_dir, seed_everything, utc_now, write_json
from .vtrace import VTraceLossOutput, vtrace_actor_critic_loss


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def drain_latest_weight_update(weight_queue: mp.Queue) -> dict[str, Any] | None:
    latest = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except queue.Empty:
            return latest


def send_weight_update(weight_queue: mp.Queue, payload: dict[str, Any]) -> None:
    while True:
        try:
            weight_queue.get_nowait()
        except queue.Empty:
            break
    try:
        weight_queue.put_nowait(payload)
    except queue.Full:
        pass


def recommendations_from_actions(
    game: CooperativeNetworkGame,
    action_semantics: str,
    actions: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    if action_semantics == "target_state":
        return game.target_edge_states_to_recommendations(actions, adjacency)
    if action_semantics == "change":
        return game.binary_changes_to_recommendations(actions, adjacency)
    raise ValueError(f"unknown action_semantics: {action_semantics!r}")


def rollout_actor_episode(
    model: GraphNet,
    game_config: GameConfig,
    bot_config: BotConfig,
    graphnet_config: GraphNetConfig,
    discount_gamma: float,
    seed: int,
    episode_id: int,
    actor_id: int,
    policy_version: int,
    train_sample_interval_episodes: int,
) -> dict[str, Any]:
    game = CooperativeNetworkGame(game_config)
    rng = np.random.default_rng(seed)
    bots = SimulatedHumanBots(game_config.n_players, bot_config, game_config)
    bots.reset_episode(rng)
    state = game.reset(seed)

    states: list[GameState] = []
    actions: list[np.ndarray] = []
    behavior_log_probs: list[float] = []
    behavior_entropies: list[float] = []
    behavior_values: list[float] = []
    rewards: list[float] = []
    round_records: list[dict[str, Any]] = []
    sampled_round_records: list[dict[str, Any]] = []
    sampled_recommendations: list[dict[str, Any]] = []

    model.eval()
    while not state.done:
        transition = game.cooperation_transition(state, bots, rng)
        planner_state = transition.planner_state
        if state.round_index < game_config.episode_length:
            with torch.no_grad():
                batch = build_graph_batch(
                    [planner_state],
                    game_config,
                    device="cpu",
                    feature_mode=graphnet_config.feature_mode,
                )
                logits, value = model(batch)
                dist = torch.distributions.Categorical(logits=logits[0])
                edge_actions = dist.sample()
                behavior_log_prob = float(dist.log_prob(edge_actions).sum().detach().cpu())
                behavior_entropy = float(dist.entropy().sum().detach().cpu())
            action_np = edge_actions.detach().cpu().numpy().astype(np.int64)
            recommendations = recommendations_from_actions(
                game,
                game_config.action_semantics,
                action_np,
                planner_state.adjacency,
            )
            adjudication = game.adjudicate_recommendations(
                planner_state,
                recommendations,
                bots,
                rng,
                episode_id=episode_id,
                planner_name="graphnet_train_vtrace",
            )
            penalty = adjudication.rejected_nonzero / game_config.n_possible_edges
            reward = float(np.mean(planner_state.capital) - game_config.recommendation_penalty_weight * penalty)
            states.append(planner_state.copy())
            actions.append(action_np)
            behavior_log_probs.append(behavior_log_prob)
            behavior_entropies.append(behavior_entropy)
            behavior_values.append(float(value.squeeze(0).detach().cpu()))
            rewards.append(reward)
            next_adjacency = adjudication.next_adjacency
            nonzero = adjudication.nonzero_recommendations
            rejected = adjudication.rejected_nonzero
            new_recommendations = adjudication.recommendation_records
        else:
            penalty = 0.0
            reward = float(np.mean(planner_state.capital))
            if rewards:
                rewards[-1] += discount_gamma * reward
            next_adjacency = planner_state.adjacency.copy()
            nonzero = 0
            rejected = 0
            new_recommendations = []

        round_record = {
            "episode_id": episode_id,
            "round": state.round_index,
            "planner": "graphnet_train_vtrace",
            "seed": seed,
            "actor_id": actor_id,
            "behavior_policy_version": policy_version,
            "mean_cooperation": float(np.mean(planner_state.prev_actions)),
            "mean_capital": float(np.mean(planner_state.capital)),
            "network_density": float(next_adjacency.sum() / (game_config.n_players * (game_config.n_players - 1))),
            "reward": reward,
            "recommendation_penalty": penalty,
            "nonzero_recommendations": nonzero,
            "rejected_recommendations": rejected,
        }
        round_records.append(round_record)
        if episode_id % train_sample_interval_episodes == 0:
            sampled_round_records.append(round_record)
            sampled_recommendations.extend(new_recommendations)
        done = state.round_index >= game_config.episode_length
        state = GameState(
            round_index=state.round_index + 1,
            adjacency=next_adjacency,
            capital=planner_state.capital.copy(),
            prev_actions=planner_state.prev_actions.copy(),
            seed=seed,
            done=done,
        )

    return {
        "actor_id": actor_id,
        "episode_id": episode_id,
        "seed": seed,
        "behavior_policy_version": policy_version,
        "states": states,
        "actions": np.stack(actions).astype(np.int64),
        "behavior_log_probs": np.asarray(behavior_log_probs, dtype=np.float32),
        "behavior_entropies": np.asarray(behavior_entropies, dtype=np.float32),
        "behavior_values": np.asarray(behavior_values, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "round_records": round_records,
        "sampled_round_records": sampled_round_records,
        "sampled_recommendations": sampled_recommendations,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_cooperation": float(np.mean([r["mean_cooperation"] for r in round_records])) if round_records else 0.0,
        "mean_capital": float(np.mean([r["mean_capital"] for r in round_records])) if round_records else 0.0,
        "network_density": float(np.mean([r["network_density"] for r in round_records])) if round_records else 0.0,
        "recommendation_penalty": float(np.mean([r["recommendation_penalty"] for r in round_records])) if round_records else 0.0,
    }


def actor_loop(
    actor_id: int,
    args_dict: dict[str, Any],
    initial_payload: dict[str, Any],
    trajectory_queue: mp.Queue,
    weight_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    torch.set_num_threads(int(args_dict["actor_torch_threads"]))
    seed_everything(int(args_dict["seed"]) + actor_id * 100_000)
    game_config = GameConfig(action_semantics=str(args_dict["action_semantics"]))
    bot_config = BotConfig()
    graphnet_config = GraphNetConfig(feature_mode=str(args_dict["feature_mode"]))
    model = GraphNet(game_config, graphnet_config).cpu()
    model.load_state_dict(initial_payload["state_dict"])
    policy_version = int(initial_payload["policy_version"])
    local_episode = 0

    while not stop_event.is_set():
        update = drain_latest_weight_update(weight_queue)
        if update is not None:
            model.load_state_dict(update["state_dict"])
            policy_version = int(update["policy_version"])
        episode_id = actor_id * 10_000_000 + local_episode
        episode_seed = int(args_dict["seed"]) + actor_id * 1_000_003 + local_episode * 9_973
        trajectory = rollout_actor_episode(
            model=model,
            game_config=game_config,
            bot_config=bot_config,
            graphnet_config=graphnet_config,
            discount_gamma=float(args_dict["discount_gamma"]),
            seed=episode_seed,
            episode_id=episode_id,
            actor_id=actor_id,
            policy_version=policy_version,
            train_sample_interval_episodes=int(args_dict["train_sample_interval_episodes"]),
        )
        while not stop_event.is_set():
            try:
                trajectory_queue.put(trajectory, timeout=1.0)
                break
            except queue.Full:
                continue
        local_episode += 1


class DistributedVTraceTrainingRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.root)
        self.game_config = GameConfig(action_semantics=args.action_semantics)
        self.bot_config = BotConfig()
        self.training_config = TrainingConfig(seed=args.seed, target_simulated_rounds=args.rounds)
        self.graphnet_config = GraphNetConfig(feature_mode=args.feature_mode)
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        seed_everything(args.seed)
        self.model = GraphNet(self.game_config, self.graphnet_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)
        self.start_rounds = 0
        self.start_episodes = 0
        if args.resume:
            payload = load_checkpoint(args.resume, self.model, self.optimizer, map_location=self.device)
            self.start_rounds = int(payload.get("simulated_rounds", 0))
            self.start_episodes = int(payload.get("episodes", 0))
        self.policy_version = 0

    def checkpoint(self, simulated_rounds: int, episodes: int, metrics: dict[str, Any]) -> str:
        ckpt_dir = self.root / "outputs/train/checkpoints"
        path = ckpt_dir / f"step_{simulated_rounds:07d}.pt"
        save_checkpoint(path, self.model, self.optimizer, simulated_rounds, episodes, self.args.seed, extra=metrics)
        latest = ckpt_dir / "latest.pt"
        save_checkpoint(latest, self.model, self.optimizer, simulated_rounds, episodes, self.args.seed, extra=metrics)
        return str(latest).replace("\\", "/")

    def learner_update(self, trajectories: list[dict[str, Any]]) -> tuple[VTraceLossOutput, dict[str, Any]]:
        bsz = len(trajectories)
        time_steps = trajectories[0]["actions"].shape[0]
        flat_states = [state for trajectory in trajectories for state in trajectory["states"]]
        batch = build_graph_batch(
            flat_states,
            self.game_config,
            device=self.device,
            feature_mode=self.graphnet_config.feature_mode,
        )
        logits_flat, values_flat = self.model(batch)
        logits = logits_flat.view(bsz, time_steps, self.game_config.n_possible_edges, 2)
        values = values_flat.view(bsz, time_steps)
        actions = torch.tensor(np.stack([t["actions"] for t in trajectories]), dtype=torch.long, device=self.device)
        behavior_log_probs = torch.tensor(
            np.stack([t["behavior_log_probs"] for t in trajectories]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(np.stack([t["rewards"] for t in trajectories]), dtype=torch.float32, device=self.device)
        dist = torch.distributions.Categorical(logits=logits)
        target_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        bootstrap_value = torch.zeros(bsz, dtype=torch.float32, device=self.device)
        loss_out = vtrace_actor_critic_loss(
            behavior_log_probs=behavior_log_probs,
            target_log_probs=target_log_probs,
            values=values,
            rewards=rewards,
            entropies=entropies,
            bootstrap_value=bootstrap_value,
            gamma=self.training_config.discount_gamma,
            baseline_cost=self.training_config.baseline_cost,
            entropy_coeff=self.training_config.entropy_regularization,
            clip_rho_threshold=self.args.vtrace_clip_rho,
            clip_pg_rho_threshold=self.args.vtrace_clip_pg_rho,
            clip_c_threshold=self.args.vtrace_clip_c,
        )
        behavior_versions = [int(t["behavior_policy_version"]) for t in trajectories]
        diagnostics = {
            "target_log_prob_mean": float(target_log_probs.detach().mean().cpu()),
            "behavior_log_prob_mean": float(behavior_log_probs.detach().mean().cpu()),
            "reward_mean": float(rewards.detach().mean().cpu()),
            "value_mean": float(values.detach().mean().cpu()),
            "vtrace_vs_mean": float(loss_out.vs.detach().mean().cpu()),
            "vtrace_advantage_mean": float(loss_out.policy_gradient_advantages.detach().mean().cpu()),
            "importance_rho_mean": float(loss_out.mean_rho.detach().cpu()),
            "importance_rho_max": float(loss_out.max_rho.detach().cpu()),
            "behavior_policy_version_min": min(behavior_versions),
            "behavior_policy_version_max": max(behavior_versions),
            "actor_lag_max": int(self.policy_version - min(behavior_versions)),
        }
        return loss_out, diagnostics

    def run(self) -> dict[str, Any]:
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
        stage_dir = ensure_dir(self.root / "stage20_distributed_vtrace")
        write_json(
            stage_dir / "run_config.json",
            {
                "algorithm": "distributed_actor_learner_vtrace",
                "seed": self.args.seed,
                "target_simulated_rounds": self.args.rounds,
                "batch_size": self.args.batch_size,
                "num_actors": self.args.num_actors,
                "checkpoint_interval_rounds": self.args.checkpoint_interval_rounds,
                "action_semantics": self.game_config.action_semantics,
                "feature_mode": self.graphnet_config.feature_mode,
                "device": str(self.device),
                "vtrace_clip_rho": self.args.vtrace_clip_rho,
                "vtrace_clip_pg_rho": self.args.vtrace_clip_pg_rho,
                "vtrace_clip_c": self.args.vtrace_clip_c,
            },
        )

        ctx = mp.get_context("spawn")
        trajectory_queue: mp.Queue = ctx.Queue(maxsize=max(self.args.batch_size * 4, self.args.num_actors * 2))
        stop_event = ctx.Event()
        weight_queues = [ctx.Queue(maxsize=1) for _ in range(self.args.num_actors)]
        initial_payload = {"policy_version": self.policy_version, "state_dict": cpu_state_dict(self.model)}
        args_dict = {
            "seed": self.args.seed,
            "action_semantics": self.game_config.action_semantics,
            "feature_mode": self.graphnet_config.feature_mode,
            "train_sample_interval_episodes": self.training_config.train_sample_interval_episodes,
            "actor_torch_threads": self.args.actor_torch_threads,
            "discount_gamma": self.training_config.discount_gamma,
        }
        actors = [
            ctx.Process(
                target=actor_loop,
                args=(actor_id, args_dict, initial_payload, trajectory_queue, weight_queues[actor_id], stop_event),
                daemon=True,
            )
            for actor_id in range(self.args.num_actors)
        ]
        for process in actors:
            process.start()

        target_rounds = self.args.rounds
        target_episodes = math.ceil(target_rounds / self.game_config.episode_length)
        checkpoint_interval_episodes = math.ceil(self.args.checkpoint_interval_rounds / self.game_config.episode_length)
        simulated_rounds = self.start_rounds
        episodes = self.start_episodes
        updates = 0
        start_time = time.time()
        last_checkpoint = None
        sampled_rounds: list[dict[str, Any]] = []
        sampled_recommendations: list[dict[str, Any]] = []
        next_checkpoint_episode = ((episodes // checkpoint_interval_episodes) + 1) * checkpoint_interval_episodes

        try:
            while simulated_rounds < target_rounds and episodes < target_episodes:
                desired_batch = min(self.args.batch_size, target_episodes - episodes)
                if episodes < next_checkpoint_episode:
                    desired_batch = min(desired_batch, next_checkpoint_episode - episodes)
                trajectories: list[dict[str, Any]] = []
                while len(trajectories) < desired_batch:
                    try:
                        trajectories.append(trajectory_queue.get(timeout=self.args.actor_timeout_seconds))
                    except queue.Empty as exc:
                        alive = [process.is_alive() for process in actors]
                        raise RuntimeError(f"timed out waiting for actor trajectories; alive={alive}") from exc

                self.optimizer.zero_grad(set_to_none=True)
                loss_out, update_diag = self.learner_update(trajectories)
                if not torch.isfinite(loss_out.loss):
                    raise FloatingPointError(f"non-finite V-trace loss at {simulated_rounds} simulated rounds")
                loss_out.loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm).detach().cpu())
                if not np.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at {simulated_rounds} simulated rounds: {grad_norm}")
                self.optimizer.step()
                self.policy_version += 1
                updates += 1

                payload = {"policy_version": self.policy_version, "state_dict": cpu_state_dict(self.model)}
                for weight_queue in weight_queues:
                    send_weight_update(weight_queue, payload)

                episodes += len(trajectories)
                simulated_rounds = episodes * self.game_config.episode_length
                sampled_rounds.extend(row for trajectory in trajectories for row in trajectory["sampled_round_records"])
                sampled_recommendations.extend(row for trajectory in trajectories for row in trajectory["sampled_recommendations"])
                metrics = {
                    "timestamp_utc": utc_now(),
                    "algorithm": "distributed_actor_learner_vtrace",
                    "seed": self.args.seed,
                    "simulated_rounds": simulated_rounds,
                    "episodes": episodes,
                    "updates": updates,
                    "policy_version": self.policy_version,
                    "actor_processes": self.args.num_actors,
                    "mean_reward": float(np.mean([t["mean_reward"] for t in trajectories])),
                    "mean_cooperation": float(np.mean([t["mean_cooperation"] for t in trajectories])),
                    "mean_capital": float(np.mean([t["mean_capital"] for t in trajectories])),
                    "network_density": float(np.mean([t["network_density"] for t in trajectories])),
                    "recommendation_penalty": float(np.mean([t["recommendation_penalty"] for t in trajectories])),
                    "loss": float(loss_out.loss.detach().cpu()),
                    "policy_loss": float(loss_out.policy_loss.detach().cpu()),
                    "value_loss": float(loss_out.value_loss.detach().cpu()),
                    "entropy": float(loss_out.entropy.detach().cpu()),
                    "grad_norm": grad_norm,
                    "rollout_mode": "distributed_actor_learner",
                    "vtrace": True,
                    "action_semantics": self.game_config.action_semantics,
                    "feature_mode": self.graphnet_config.feature_mode,
                    "batch_episodes": len(trajectories),
                    "checkpoint_path": None,
                }
                metrics.update(update_diag)
                if episodes >= next_checkpoint_episode or simulated_rounds >= target_rounds:
                    last_checkpoint = self.checkpoint(simulated_rounds, episodes, metrics)
                    metrics["checkpoint_path"] = last_checkpoint
                    while next_checkpoint_episode <= episodes:
                        next_checkpoint_episode += checkpoint_interval_episodes
                append_jsonl(self.root / "outputs/train/train_metrics.jsonl", metrics)
                append_jsonl(self.root / "outputs/train/vtrace_loss_metrics.jsonl", metrics)
                write_json(
                    self.root / "outputs/train/heartbeat.json",
                    {
                        "timestamp_utc": utc_now(),
                        "algorithm": "distributed_actor_learner_vtrace",
                        "seed": self.args.seed,
                        "target_simulated_rounds": target_rounds,
                        "completed_simulated_rounds": simulated_rounds,
                        "completed_episodes": episodes,
                        "updates": updates,
                        "last_checkpoint": last_checkpoint,
                        "mean_reward_recent": metrics["mean_reward"],
                        "mean_cooperation_recent": metrics["mean_cooperation"],
                        "policy_loss_recent": metrics["policy_loss"],
                        "value_loss_recent": metrics["value_loss"],
                        "entropy_recent": metrics["entropy"],
                        "importance_rho_mean_recent": metrics["importance_rho_mean"],
                        "actor_lag_max_recent": metrics["actor_lag_max"],
                        "rollout_mode": "distributed_actor_learner",
                        "vtrace": True,
                        "action_semantics": self.game_config.action_semantics,
                        "feature_mode": self.graphnet_config.feature_mode,
                    },
                )
        finally:
            stop_event.set()
            for process in actors:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()

        if last_checkpoint is None:
            final_metrics = {"simulated_rounds": simulated_rounds, "episodes": episodes, "algorithm": "distributed_actor_learner_vtrace"}
            last_checkpoint = self.checkpoint(simulated_rounds, episodes, final_metrics)

        import pandas as pd

        pd.DataFrame(sampled_rounds).to_parquet(self.root / "outputs/raw_train_sample/sampled_rounds.parquet", index=False)
        pd.DataFrame(sampled_recommendations).to_parquet(
            self.root / "outputs/raw_train_sample/sampled_recommendations.parquet",
            index=False,
        )
        summary = {
            "algorithm": "distributed_actor_learner_vtrace",
            "seed": self.args.seed,
            "target_simulated_rounds": target_rounds,
            "completed_simulated_rounds": simulated_rounds,
            "completed_episodes": episodes,
            "updates": updates,
            "target_finished": simulated_rounds >= target_rounds,
            "last_checkpoint": last_checkpoint,
            "device": str(self.device),
            "num_actors": self.args.num_actors,
            "action_semantics": self.game_config.action_semantics,
            "feature_mode": self.graphnet_config.feature_mode,
            "wall_seconds": round(time.time() - start_time, 3),
        }
        write_json(stage_dir / "distributed_vtrace_train_summary.json", summary)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-actors", type=int, default=8)
    parser.add_argument("--actor-torch-threads", type=int, default=1)
    parser.add_argument("--actor-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--checkpoint-interval-rounds", type=int, default=10_000)
    parser.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="paper_minimal")
    parser.add_argument("--vtrace-clip-rho", type=float, default=1.0)
    parser.add_argument("--vtrace-clip-pg-rho", type=float, default=1.0)
    parser.add_argument("--vtrace-clip-c", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = DistributedVTraceTrainingRun(args).run()
    print(summary)


if __name__ == "__main__":
    main()
