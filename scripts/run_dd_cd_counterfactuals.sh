#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

python -m src.counterfactuals \
  --root "$ROOT" \
  --out-dir outputs/root_cause \
  --pair-type both \
  --state-planners "${STATE_PLANNERS:-maximum_connectivity,neutral,graphnet}" \
  --continuation-planner "${CONTINUATION_PLANNER:-static}" \
  --episodes "${EPISODES:-128}" \
  --samples-per-pair "${SAMPLES_PER_PAIR:-80}" \
  --rollouts-per-sample "${ROLLOUTS_PER_SAMPLE:-24}" \
  --seed "${SEED:-20260624}" \
  --checkpoint "${CHECKPOINT:-outputs/train/checkpoints/step_1000005.pt}" \
  --device "${DEVICE:-cpu}" \
  --action-semantics "${ACTION_SEMANTICS:-target_state}" \
  --feature-mode "${FEATURE_MODE:-augmented}"
