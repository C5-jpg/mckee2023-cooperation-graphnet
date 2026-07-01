from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.bots import SimulatedHumanBots
from src.config import GameConfig
from src.env import CooperativeNetworkGame
from src.planners import StaticPlanner


def test_static_episode_completes():
    cfg = GameConfig()
    env = CooperativeNetworkGame(cfg)
    bots = SimulatedHumanBots(cfg.n_players, game_config=cfg)
    episode = env.run_episode(StaticPlanner(cfg), bots, seed=20260624)
    assert len(episode["rounds"]) == cfg.episode_length
    assert episode["final_state"].done


def test_custom_training_cli_writes_metrics(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--mode",
        "custom",
        "--rounds",
        "15",
        "--batch-size",
        "1",
        "--checkpoint-interval-rounds",
        "15",
        "--root",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs/train/train_metrics.jsonl").exists()
    assert (tmp_path / "outputs/train/checkpoints/latest.pt").exists()

