from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


TAG_RE = re.compile(r"target_(\d+)_checkpoint_(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-csv", default="outputs/eval_by_checkpoint/checkpoint_eval_summary.csv")
    parser.add_argument("--out-json", default="outputs/eval_by_checkpoint/checkpoint_eval_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for summary_path in sorted((root / "outputs/eval_by_checkpoint").glob("*/eval_summary.json")):
        match = TAG_RE.match(summary_path.parent.name)
        if not match:
            continue
        target_rounds = int(match.group(1))
        checkpoint_rounds = int(match.group(2))
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        graphnet = payload.get("graphnet", {})
        row = {
            "target_rounds": target_rounds,
            "checkpoint_rounds": checkpoint_rounds,
            "available": graphnet.get("available", False),
            "mean_cooperation": graphnet.get("mean_cooperation"),
            "final_round_cooperation": graphnet.get("final_round_cooperation"),
            "mean_capital": graphnet.get("mean_capital"),
            "final_mean_capital": graphnet.get("final_mean_capital"),
            "network_density_final": graphnet.get("network_density_final"),
            "gini_final": graphnet.get("gini_final"),
            "mean_reward": graphnet.get("mean_reward"),
            "summary_path": str(summary_path.relative_to(root)).replace("\\", "/"),
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["target_rounds", "checkpoint_rounds"])
    out_csv = root / args.out_csv
    out_json = root / args.out_json
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print({"rows": len(rows), "csv": str(out_csv), "json": str(out_json)})


if __name__ == "__main__":
    main()
