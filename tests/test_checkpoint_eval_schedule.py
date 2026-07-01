from __future__ import annotations

from pathlib import Path

from scripts.evaluate_checkpoints import select_threshold_checkpoints


def test_checkpoint_selection_covers_100k_thresholds_with_episode_granularity(tmp_path: Path):
    ckpts = []
    for rounds in [100005, 200010, 300000]:
        path = tmp_path / f"step_{rounds:07d}.pt"
        path.write_text("x", encoding="utf-8")
        ckpts.append(path)
    selected = select_threshold_checkpoints(ckpts, min_rounds=100000, max_rounds=300000, interval_rounds=100000)
    assert [(threshold, rounds) for threshold, rounds, _ in selected] == [(100000, 100005), (200000, 200010), (300000, 300000)]

