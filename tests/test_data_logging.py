from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.a2c import load_checkpoint, save_checkpoint
from src.config import ReproConfig
from src.data_logging import create_output_dirs, write_input_manifest, write_resolved_config
from src.graphnet import GraphNet
from src.utils import append_jsonl


def test_output_dirs_jsonl_config_parquet_and_checkpoint(tmp_path: Path):
    create_output_dirs(tmp_path)
    for rel in ["outputs/config", "outputs/tests", "outputs/train/checkpoints", "outputs/eval", "outputs/raw_eval"]:
        assert (tmp_path / rel).exists()
    append_jsonl(tmp_path / "outputs/train/train_metrics.jsonl", {"seed": 20260624, "target_rounds": 1000000})
    line = (tmp_path / "outputs/train/train_metrics.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["seed"] == 20260624
    write_resolved_config(tmp_path, ReproConfig())
    cfg_text = (tmp_path / "outputs/config/resolved_config.yaml").read_text(encoding="utf-8")
    assert "seed: 20260624" in cfg_text
    paper_dir = tmp_path / "paper_inputs"
    paper_dir.mkdir()
    for name in ["main_paper.md", "supplement.md", "main_paper.pdf", "supplement.pdf"]:
        (paper_dir / name).write_text("x", encoding="utf-8")
    write_input_manifest(tmp_path)
    assert (tmp_path / "outputs/config/input_manifest.json").exists()
    parquet = tmp_path / "outputs/raw_eval/rounds.parquet"
    pd.DataFrame([{"round": 1, "mean_cooperation": 0.5}]).to_parquet(parquet, index=False)
    assert pd.read_parquet(parquet).iloc[0]["round"] == 1
    model = GraphNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    ckpt = tmp_path / "outputs/train/checkpoints/latest.pt"
    save_checkpoint(ckpt, model, opt, 15, 1, 20260624)
    loaded = load_checkpoint(ckpt, model, opt)
    assert loaded["simulated_rounds"] == 15

