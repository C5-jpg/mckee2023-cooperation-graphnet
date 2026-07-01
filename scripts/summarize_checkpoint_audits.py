from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_table6(path: Path) -> dict[str, float]:
    table_path = path / "table6_style_policy.csv"
    if not table_path.exists():
        table_path = path / "eval" / "table6_style_policy.csv"
    out: dict[str, float] = {}
    if not table_path.exists():
        return out
    with table_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['pair_type']}_{row['change_type']}".replace("-", "")
            out[key] = float(row["recommendation_rate"])
    return out


def read_probability(path: Path) -> dict[str, float]:
    prob_path = path / "policy_diagnostics" / "policy_probability_by_type.csv"
    out: dict[str, float] = {}
    if not prob_path.exists():
        return out
    with prob_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['pair_type']}_{row['change_type']}".replace("-", "")
            out[f"{key}_prob"] = float(row["mean_prob_change"])
            out[f"{key}_margin"] = float(row["mean_logit_margin"])
            out[f"{key}_det"] = float(row["deterministic_change_rate"])
    return out


def read_graphnet_summary(path: Path) -> dict[str, float]:
    summary_path = path / "eval" / "graphnet_eval_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    fields = [
        "final_round_cooperation",
        "mean_cooperation",
        "network_density_final",
        "gini_final",
        "final_mean_capital",
    ]
    return {field: payload.get(field) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    root = Path(args.eval_root)
    rows: list[dict[str, object]] = []
    for step_dir in sorted(root.glob("step_*")):
        gate_path = step_dir / "alignment_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
        row: dict[str, object] = {
            "checkpoint": step_dir.name,
            "rounds": gate.get("checkpoint_rounds"),
            "severe_contradiction": gate.get("severe_contradiction"),
        }
        row.update(read_table6(step_dir))
        row.update(read_probability(step_dir))
        row.update(read_graphnet_summary(step_dir))
        rows.append(row)

    field_order = [
        "checkpoint",
        "rounds",
        "CC_add",
        "CC_delete",
        "CD_add",
        "CD_delete",
        "DD_add",
        "DD_delete",
        "DD_add_prob",
        "DD_delete_prob",
        "DD_add_margin",
        "DD_delete_margin",
        "DD_add_det",
        "DD_delete_det",
        "final_round_cooperation",
        "mean_cooperation",
        "network_density_final",
        "gini_final",
        "final_mean_capital",
        "severe_contradiction",
    ]
    extra_fields = sorted({key for row in rows for key in row if key not in field_order})
    fields = field_order + extra_fields

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
