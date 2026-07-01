#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

python -m src.root_cause_diagnostics oracles \
  --root "$ROOT" \
  --out-dir outputs/root_cause \
  --episodes "${EPISODES:-512}" \
  --seed "${SEED:-20260624}" \
  --checkpoint "${CHECKPOINT:-outputs/train/checkpoints/step_1000005.pt}" \
  --device "${DEVICE:-cpu}" \
  --action-semantics "${ACTION_SEMANTICS:-target_state}" \
  --feature-mode "${FEATURE_MODE:-augmented}"
