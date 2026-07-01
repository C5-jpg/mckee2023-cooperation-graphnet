from __future__ import annotations

import numpy as np
import torch

from src.config import BotConfig, GameConfig, GraphNetConfig
from src.distributed_vtrace_train import cpu_state_dict, recommendations_from_actions, rollout_actor_episode
from src.env import CooperativeNetworkGame
from src.graphnet import GraphNet


def test_recommendations_from_target_state_actions_respects_active_bit():
    cfg = GameConfig(n_players=3, action_semantics="target_state")
    game = CooperativeNetworkGame(cfg)
    adjacency = np.zeros((3, 3), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = True

    recs = recommendations_from_actions(
        game=game,
        action_semantics="target_state",
        actions=np.array([0, 1, 0]),
        adjacency=adjacency,
    )

    assert recs.tolist() == [-1, 1, 0]


def test_recommendations_from_change_actions_respects_active_bit():
    cfg = GameConfig(n_players=3, action_semantics="change")
    game = CooperativeNetworkGame(cfg)
    adjacency = np.zeros((3, 3), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = True

    recs = recommendations_from_actions(
        game=game,
        action_semantics="change",
        actions=np.array([1, 1, 0]),
        adjacency=adjacency,
    )

    assert recs.tolist() == [-1, 1, 0]


def test_actor_rollout_has_no_final_round_action_and_folds_final_reward():
    cfg = GameConfig(n_players=4, episode_length=3, action_semantics="target_state")
    graph_cfg = GraphNetConfig(feature_mode="paper_minimal")
    model = GraphNet(cfg, graph_cfg).cpu()

    trajectory = rollout_actor_episode(
        model=model,
        game_config=cfg,
        bot_config=BotConfig(),
        graphnet_config=graph_cfg,
        discount_gamma=0.99,
        seed=42,
        episode_id=0,
        actor_id=0,
        policy_version=0,
        train_sample_interval_episodes=1,
    )

    assert trajectory["actions"].shape == (cfg.episode_length - 1, cfg.n_possible_edges)
    assert trajectory["behavior_log_probs"].shape == (cfg.episode_length - 1,)
    assert trajectory["rewards"].shape == (cfg.episode_length - 1,)
    assert len(trajectory["round_records"]) == cfg.episode_length
    assert trajectory["round_records"][-1]["nonzero_recommendations"] == 0
    assert trajectory["round_records"][-1]["rejected_recommendations"] == 0
    assert np.isfinite(trajectory["rewards"]).all()


def test_cpu_state_dict_detaches_parameters_to_cpu():
    model = GraphNet(GameConfig(n_players=4), GraphNetConfig(feature_mode="paper_minimal"))
    state = cpu_state_dict(model)

    assert state
    assert all(not tensor.requires_grad for tensor in state.values())
    assert all(tensor.device.type == "cpu" for tensor in state.values())
    assert all(isinstance(tensor, torch.Tensor) for tensor in state.values())

