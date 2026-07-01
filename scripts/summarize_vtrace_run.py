from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def checkpoint_round_from_name(name: str) -> int:
    return int(name.split("_", 1)[1])


def load_eval_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    graphnet = payload.get("graphnet", {})
    return graphnet if isinstance(graphnet, dict) else {}


def summarize_run(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_root = root / "outputs/eval_by_checkpoint"
    for checkpoint_dir in sorted(eval_root.glob("step_*")):
        audit_path = checkpoint_dir / "table6_style_policy.csv"
        if not audit_path.exists():
            continue
        audit = pd.read_csv(audit_path)
        row: dict[str, Any] = {
            "checkpoint": checkpoint_dir.name,
            "checkpoint_rounds": checkpoint_round_from_name(checkpoint_dir.name),
        }
        for _, record in audit.iterrows():
            key = f"{record['pair_type']}_{record['change_type']}".replace("-", "").lower()
            rate = record.get("recommendation_rate")
            row[f"{key}_rate"] = None if pd.isna(rate) else float(rate)
            row[f"{key}_direction_pass"] = bool(record.get("direction_pass"))
            row[f"{key}_count_recommended"] = int(record.get("count_recommended", 0))
            row[f"{key}_count_not_recommended"] = int(record.get("count_not_recommended", 0))
        graphnet = load_eval_summary(checkpoint_dir / "eval_summary.json")
        for key in [
            "mean_cooperation",
            "final_round_cooperation",
            "mean_capital",
            "final_mean_capital",
            "gini_final",
            "network_density_final",
            "relative_degree_cooperators_final",
            "newman_assortativity_final",
            "rand_assortativity_final",
            "shirado_assortativity_final",
            "defector_disassortativity_final",
        ]:
            if key in graphnet:
                row[key] = graphnet[key]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("checkpoint_rounds") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", default="stage20_distributed_vtrace")
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    summary = summarize_run(root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "checkpoint_audit_summary.csv"
    json_path = out_dir / "checkpoint_audit_summary.json"
    summary.to_csv(csv_path, index=False)
    records = summary.where(pd.notna(summary), None).to_dict(orient="records")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "root": str(root),
        "evaluated_checkpoints": int(len(summary)),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "latest": records[-1] if records else None,
        "tail": records[-args.tail :] if args.tail > 0 else records,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
