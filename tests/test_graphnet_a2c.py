from __future__ import annotations

import numpy as np
import torch

from src.a2c import a2c_loss
from src.bots import SimulatedHumanBots
from src.config import GameConfig
from src.env import CooperativeNetworkGame
from src.graphnet import GraphNet, build_graph_batch
from src.planners import GraphNetPlanner
from src.train import joint_action_log_prob_and_entropy


def test_graphnet_forward_shapes():
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    states = [env.reset(1), env.reset(2)]
    model = GraphNet(cfg)
    logits, value = model(build_graph_batch(states, cfg))
    assert logits.shape == (2, cfg.n_possible_edges, 2)
    assert value.shape == (2,)


def test_action_sampling_and_a2c_loss_finite():
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    state = env.reset(3)
    model = GraphNet(cfg)
    logits, value = model(build_graph_batch([state], cfg))
    dist = torch.distributions.Categorical(logits=logits[0])
    actions = dist.sample()
    assert actions.shape == (cfg.n_possible_edges,)
    log_prob, entropy = joint_action_log_prob_and_entropy(dist, actions)
    loss = a2c_loss(log_prob.reshape(1), value.reshape(1), [1.0], entropy.reshape(1))
    assert torch.isfinite(loss.loss)


def test_short_optimizer_update_changes_parameters():
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    state = env.reset(4)
    model = GraphNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    before = [p.detach().clone() for p in model.parameters()]
    logits, value = model(build_graph_batch([state], cfg))
    dist = torch.distributions.Categorical(logits=logits[0])
    actions = dist.sample()
    log_prob, entropy = joint_action_log_prob_and_entropy(dist, actions)
    loss = a2c_loss(log_prob.reshape(1), value.reshape(1), [1.0], entropy.reshape(1))
    opt.zero_grad()
    loss.loss.backward()
    opt.step()
    assert any(not torch.allclose(a, b) for a, b in zip(before, model.parameters()))


def test_graphnet_planner_rollout_no_nan():
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    model = GraphNet(cfg)
    planner = GraphNetPlanner(model=model, config=cfg)
    bots = SimulatedHumanBots(cfg.n_players, game_config=cfg)
    episode = env.run_episode(planner, bots, seed=5)
    capitals = episode["final_state"].capital
    assert not np.isnan(capitals).any()
    assert len(episode["rounds"]) == cfg.episode_length
