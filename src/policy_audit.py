from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ACTION_SEMANTICS
from .utils import ensure_dir, write_json


TABLE6_REFERENCE: dict[tuple[str, str], float] = {
    ("C-C", "add"): 0.994,
    ("C-C", "delete"): 0.030,
    ("C-D", "add"): 0.582,
    ("C-D", "delete"): 0.503,
    ("D-D", "add"): 0.000,
    ("D-D", "delete"): 1.000,
}

PAIR_ORDER = ["C-C", "C-D", "D-D"]
CHANGE_ORDER = ["add", "delete"]


def normalize_pair_type(value: Any) -> str:
    text = str(value)
    if text in {"C-C", "cooperate_cooperate"}:
        return "C-C"
    if text in {"C-D", "cooperate_defect", "D-C"}:
        return "C-D"
    if text in {"D-D", "defect_defect"}:
        return "D-D"
    raise ValueError(f"unknown pair type: {value!r}")


def direction_pass(pair_type: str, change_type: str, rate: float) -> bool:
    if np.isnan(rate):
        return False
    if pair_type == "C-C" and change_type == "add":
        return bool(rate >= 0.70)
    if pair_type == "C-C" and change_type == "delete":
        return bool(rate <= 0.30)
    if pair_type == "D-D" and change_type == "add":
        return bool(rate <= 0.20)
    if pair_type == "D-D" and change_type == "delete":
        return bool(rate >= 0.70)
    return True


def direction_note(pair_type: str, change_type: str) -> str:
    if pair_type == "C-C" and change_type == "add":
        return "C-C add should be high; final engineering gate expects >= 0.70."
    if pair_type == "C-C" and change_type == "delete":
        return "C-C delete should be low; final engineering gate expects <= 0.30."
    if pair_type == "D-D" and change_type == "add":
        return "D-D add should be low; final engineering gate expects <= 0.20."
    if pair_type == "D-D" and change_type == "delete":
        return "D-D delete should be high; final engineering gate expects >= 0.70."
    return "C-D add/delete has no hard gate; inspect round trend."


def build_table6_policy_audit(
    edge_rows: pd.DataFrame,
    recommendation_rows: pd.DataFrame,
    planner: str = "graphnet",
    max_recommendation_round: int = 14,
    action_semantics: str | None = None,
) -> pd.DataFrame:
    if edge_rows.empty:
        return empty_policy_audit(action_semantics=action_semantics)
    edges = edge_rows[(edge_rows["planner"] == planner) & (edge_rows["round"] <= max_recommendation_round)].copy()
    if edges.empty:
        return empty_policy_audit(action_semantics=action_semantics)
    before_col = "edge_active_before" if "edge_active_before" in edges.columns else "edge_active"
    edges["edge_active_before_bool"] = edges[before_col].astype(bool)
    edges["pair_type"] = edges["cooperation_pair_type"].map(normalize_pair_type)
    edges["change_type"] = np.where(edges["edge_active_before_bool"], "delete", "add")

    rec_cols = ["planner", "episode_id", "round", "i", "j", "recommendation"]
    if recommendation_rows.empty:
        recs = pd.DataFrame(columns=rec_cols)
    else:
        recs = recommendation_rows[
            (recommendation_rows["planner"] == planner)
            & (recommendation_rows["round"] <= max_recommendation_round)
            & (recommendation_rows["recommendation"].astype(float) != 0.0)
        ][rec_cols].copy()
    merged = edges.merge(recs, on=["planner", "episode_id", "round", "i", "j"], how="left")
    merged["recommended"] = merged["recommendation"].notna()

    grouped = merged.groupby(["pair_type", "change_type"], as_index=False).agg(
        chances=("recommended", "size"),
        count_recommended=("recommended", "sum"),
    )
    rows = []
    for pair in PAIR_ORDER:
        for change in CHANGE_ORDER:
            row = grouped[(grouped["pair_type"] == pair) & (grouped["change_type"] == change)]
            chances = int(row["chances"].iloc[0]) if not row.empty else 0
            recommended = int(row["count_recommended"].iloc[0]) if not row.empty else 0
            not_recommended = chances - recommended
            rate = float(recommended / chances) if chances else float("nan")
            paper = TABLE6_REFERENCE[(pair, change)]
            out_row = {
                "pair_type": pair,
                "change_type": change,
                "count_recommended": recommended,
                "count_not_recommended": not_recommended,
                "recommendation_rate": rate,
                "paper_rate_reference": paper,
                "absolute_difference": abs(rate - paper) if not np.isnan(rate) else float("nan"),
                "direction_pass": direction_pass(pair, change, rate),
                "notes": direction_note(pair, change),
            }
            if action_semantics is not None:
                out_row["action_semantics"] = action_semantics
            rows.append(out_row)
    return pd.DataFrame(rows)


def empty_policy_audit(action_semantics: str | None = None) -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
        for change in CHANGE_ORDER:
            row = {
                "pair_type": pair,
                "change_type": change,
                "count_recommended": 0,
                "count_not_recommended": 0,
                "recommendation_rate": float("nan"),
                "paper_rate_reference": TABLE6_REFERENCE[(pair, change)],
                "absolute_difference": float("nan"),
                "direction_pass": False,
                "notes": "No graphnet edge opportunities were available.",
            }
            if action_semantics is not None:
                row["action_semantics"] = action_semantics
            rows.append(row)
    return pd.DataFrame(rows)


def build_alignment_gate(
    audit: pd.DataFrame,
    checkpoint_rounds: int,
    tests_passed: bool = True,
    nan_detected: bool = False,
    maximum_connectivity_diagnostic: str = "not_run",
    action_semantics: str = "target_state",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    def passed(pair: str, change: str) -> bool:
        row = audit[(audit["pair_type"] == pair) & (audit["change_type"] == change)]
        return bool(row["direction_pass"].iloc[0]) if not row.empty else False

    table6_direction = {
        "cc_add_high": passed("C-C", "add"),
        "cc_delete_low": passed("C-C", "delete"),
        "dd_add_low": passed("D-D", "add"),
        "dd_delete_high": passed("D-D", "delete"),
    }
    severe = not all(table6_direction.values())
    return {
        "checkpoint_rounds": int(checkpoint_rounds),
        "action_semantics": action_semantics,
        "tests_passed": bool(tests_passed),
        "nan_detected": bool(nan_detected),
        "table6_direction": table6_direction,
        "severe_contradiction": bool(severe),
        "maximum_connectivity_diagnostic": maximum_connectivity_diagnostic,
        "notes": notes or [],
    }


def write_policy_audit_outputs(
    audit: pd.DataFrame,
    out_dir: str | Path,
    checkpoint_rounds: int | None = None,
    tests_passed: bool = True,
    nan_detected: bool = False,
    maximum_connectivity_diagnostic: str = "not_run",
    action_semantics: str = "target_state",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    if action_semantics and "action_semantics" not in audit.columns:
        audit = audit.copy()
        audit["action_semantics"] = action_semantics
    csv_path = out / "table6_style_policy.csv"
    json_path = out / "table6_style_policy.json"
    audit.to_csv(csv_path, index=False)
    payload = audit.where(pd.notna(audit), None).to_dict(orient="records")
    write_json(json_path, payload)
    if checkpoint_rounds is not None:
        gate = build_alignment_gate(
            audit,
            checkpoint_rounds,
            tests_passed=tests_passed,
            nan_detected=nan_detected,
            maximum_connectivity_diagnostic=maximum_connectivity_diagnostic,
            action_semantics=action_semantics,
            notes=notes,
        )
        write_json(out / "alignment_gate.json", gate)
        return gate
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--raw-dir", default="outputs/raw_eval")
    parser.add_argument("--out-dir", default="outputs/policy_audit")
    parser.add_argument("--planner", default="graphnet")
    parser.add_argument("--checkpoint-rounds", type=int, default=None)
    parser.add_argument("--action-semantics", choices=ACTION_SEMANTICS, default="target_state")
    args = parser.parse_args()
    root = Path(args.root)
    raw_dir = root / args.raw_dir
    edges = pd.read_parquet(raw_dir / "edges.parquet")
    recs = pd.read_parquet(raw_dir / "recommendations.parquet")
    audit = build_table6_policy_audit(edges, recs, planner=args.planner, action_semantics=args.action_semantics)
    out_dir = root / args.out_dir
    write_policy_audit_outputs(audit, out_dir, checkpoint_rounds=args.checkpoint_rounds, action_semantics=args.action_semantics)
    final_dir = root / "outputs/policy_audit"
    if out_dir.resolve() == final_dir.resolve():
        audit.to_csv(final_dir / "table6_style_final.csv", index=False)
        write_json(final_dir / "table6_style_final.json", audit.where(pd.notna(audit), None).to_dict(orient="records"))
    print({"rows": len(audit), "out_dir": str(out_dir).replace("\\", "/")})


if __name__ == "__main__":
    main()
