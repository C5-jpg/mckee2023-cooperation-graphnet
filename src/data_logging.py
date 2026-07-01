from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .config import ReproConfig, resolved_config_dict
from .utils import ensure_dir, write_json


REQUIRED_OUTPUT_DIRS = [
    "outputs/config",
    "outputs/tests",
    "outputs/train/checkpoints",
    "outputs/eval",
    "outputs/raw_eval",
    "outputs/raw_train_sample",
    "outputs/figures_optional",
]


def create_output_dirs(root: str | Path = ".") -> None:
    root = Path(root)
    for rel in REQUIRED_OUTPUT_DIRS:
        ensure_dir(root / rel)


def write_resolved_config(root: str | Path = ".", config: ReproConfig | None = None) -> None:
    root = Path(root)
    create_output_dirs(root)
    cfg = resolved_config_dict(config)
    path = root / "outputs/config/resolved_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (root / "outputs/config/seed.txt").write_text(str(cfg["seed"]) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_input_manifest(root: str | Path = ".") -> None:
    root = Path(root)
    input_dir = root / "paper_inputs"
    rows = []
    for name in ["main_paper.md", "supplement.md", "main_paper.pdf", "supplement.pdf"]:
        path = input_dir / name
        rows.append(
            {
                "file": name,
                "sha256": file_sha256(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    write_json(root / "outputs/config/input_manifest.json", rows)


def write_dataframe(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(list(rows))
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported dataframe output: {path}")


def write_json_file(path: str | Path, data: Any) -> None:
    write_json(path, data)


def write_figures_readme(root: str | Path = ".") -> None:
    path = Path(root) / "outputs/figures_optional/README.md"
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text("Optional diagnostic figures may be written here.\n", encoding="utf-8")

