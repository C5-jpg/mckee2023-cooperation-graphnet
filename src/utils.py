from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def sigmoid(x: Any) -> Any:
    try:
        import numpy as np

        return 1.0 / (1.0 + np.exp(-x))
    except Exception:
        return 1.0 / (1.0 + math.exp(-float(x)))


def all_undirected_edges(n_players: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n_players) for j in range(i + 1, n_players)]


def edge_lookup(edges: Iterable[Tuple[int, int]]) -> dict[Tuple[int, int], int]:
    return {tuple(edge): idx for idx, edge in enumerate(edges)}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def relative_path(path: str | Path, root: str | Path | None = None) -> str:
    path_obj = Path(path)
    root_obj = Path(root or os.getcwd())
    try:
        return str(path_obj.resolve().relative_to(root_obj.resolve())).replace("\\", "/")
    except Exception:
        return str(path_obj).replace("\\", "/")

