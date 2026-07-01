#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

python -m src.policy_probe \
  --root "$ROOT" \
  --out-dir outputs/root_cause \
  --checkpoint "${CHECKPOINT:-outputs/train/checkpoints/step_1000005.pt}" \
  --device "${DEVICE:-cpu}" \
  --action-semantics "${ACTION_SEMANTICS:-target_state}" \
  --feature-mode "${FEATURE_MODE:-augmented}" \
  --state-planners "${STATE_PLANNERS:-maximum_connectivity,neutral,graphnet}" \
  --episodes "${EPISODES:-64}" \
  --n-states "${N_STATES:-80}" \
  --max-edges-per-state "${MAX_EDGES_PER_STATE:-120}" \
  --seed "${SEED:-20260624}"
