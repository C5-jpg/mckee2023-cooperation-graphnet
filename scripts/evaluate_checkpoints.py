from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


STEP_RE = re.compile(r"step_(\d+)\.pt$")


def checkpoint_round(path: Path) -> int | None:
    match = STEP_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def existing_summary_complete(path: Path, expected_round: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    graphnet = payload.get("graphnet", {})
    gate = payload.get("policy_alignment_gate", {})
    return bool(graphnet.get("available")) and int(graphnet.get("checkpoint_rounds", -1)) == expected_round and bool(gate)


def existing_diagnostics_complete(path: Path, expected_round: int, min_episodes: int) -> bool:
    summary_path = path / "policy_diagnostics_summary.json"
    probability_path = path / "policy_probability_by_type.csv"
    if min_episodes <= 0:
        return True
    if not summary_path.exists() or not probability_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(payload.get("checkpoint_rounds", -1)) == expected_round and int(payload.get("n_episodes", 0)) >= min_episodes


def select_threshold_checkpoints(
    checkpoint_paths: list[Path],
    min_rounds: int,
    max_rounds: int,
    interval_rounds: int,
) -> list[tuple[int, int, Path]]:
    by_round: list[tuple[int, Path]] = []
    for checkpoint in checkpoint_paths:
        rounds = checkpoint_round(checkpoint)
        if rounds is not None:
            by_round.append((rounds, checkpoint))
    selected: list[tuple[int, int, Path]] = []
    threshold = min_rounds
    while threshold <= max_rounds:
        candidates = [(rounds, path) for rounds, path in by_round if rounds >= threshold]
        if candidates:
            rounds, path = min(candidates, key=lambda item: item[0])
            selected.append((threshold, rounds, path))
        threshold += interval_rounds
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--interval-rounds", type=int, default=100_000)
    parser.add_argument("--min-rounds", type=int, default=100_000)
    parser.add_argument("--max-rounds", type=int, default=1_000_000)
    parser.add_argument("--n-eval-episodes", type=int, default=64)
    parser.add_argument("--diagnostics-episodes", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--action-semantics", choices=["target_state", "change"], default="target_state")
    parser.add_argument("--feature-mode", choices=["augmented", "paper_minimal"], default="augmented")
    args = parser.parse_args()

    root = Path(args.root)
    checkpoint_dir = root / "outputs/train/checkpoints"
    checkpoint_paths = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: checkpoint_round(p) or -1)
    evaluated: list[dict[str, int]] = []
    skipped: list[dict[str, int]] = []
    diagnosed: list[dict[str, int]] = []
    for threshold, rounds, checkpoint in select_threshold_checkpoints(
        checkpoint_paths,
        min_rounds=args.min_rounds,
        max_rounds=args.max_rounds,
        interval_rounds=args.interval_rounds,
    ):
        tag = f"step_{rounds:07d}"
        summary_path = root / "outputs/eval_by_checkpoint" / tag / "eval_summary.json"
        diagnostics_dir = root / "outputs/eval_by_checkpoint" / tag / "policy_diagnostics"
        summary_complete = existing_summary_complete(summary_path, rounds)
        diagnostics_complete = existing_diagnostics_complete(diagnostics_dir, rounds, args.diagnostics_episodes)
        if summary_complete and diagnostics_complete:
            skipped.append({"target_rounds": threshold, "checkpoint_rounds": rounds})
            continue
        repo_cwd = Path(__file__).resolve().parents[1]
        if not summary_complete:
            eval_dir = f"outputs/eval_by_checkpoint/{tag}/eval"
            raw_dir = f"outputs/eval_by_checkpoint/{tag}/raw_eval"
            cmd = [
                sys.executable,
                "-m",
                "src.evaluate",
                "--root",
                str(root),
                "--checkpoint",
                str(checkpoint.relative_to(root)),
                "--n-eval-episodes",
                str(args.n_eval_episodes),
                "--eval-dir",
                eval_dir,
                "--raw-dir",
                raw_dir,
                "--stage-summary",
                f"outputs/eval_by_checkpoint/{tag}/eval_summary.json",
                "--policy-audit-dir",
                f"outputs/eval_by_checkpoint/{tag}",
                "--action-semantics",
                args.action_semantics,
                "--feature-mode",
                args.feature_mode,
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            subprocess.run(cmd, cwd=repo_cwd, check=True)
            evaluated.append({"target_rounds": threshold, "checkpoint_rounds": rounds})
        if args.diagnostics_episodes > 0 and not diagnostics_complete:
            diag_cmd = [
                sys.executable,
                "-m",
                "src.policy_diagnostics",
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(diagnostics_dir),
                "--n-episodes",
                str(args.diagnostics_episodes),
                "--action-semantics",
                args.action_semantics,
                "--feature-mode",
                args.feature_mode,
            ]
            if args.device:
                diag_cmd.extend(["--device", args.device])
            subprocess.run(diag_cmd, cwd=repo_cwd, check=True)
            diagnosed.append({"target_rounds": threshold, "checkpoint_rounds": rounds})
    print(json.dumps({"diagnosed": diagnosed, "evaluated": evaluated, "skipped": skipped}, sort_keys=True))


if __name__ == "__main__":
    main()
