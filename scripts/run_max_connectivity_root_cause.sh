#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

python -m src.root_cause_diagnostics maxconn \
  --root "$ROOT" \
  --out-dir outputs/root_cause \
  --episodes "${EPISODES:-512}" \
  --seed "${SEED:-20260624}" \
  --surface-replicates "${SURFACE_REPLICATES:-512}"

