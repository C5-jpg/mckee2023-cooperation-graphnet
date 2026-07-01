#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
ROOT="${ROOT:-.}"
DIAGNOSTIC_EPISODES="${DIAGNOSTIC_EPISODES:-128}"

"${PYTHON_BIN}" -m src.diagnostics \
  --root "${ROOT}" \
  --n-episodes "${DIAGNOSTIC_EPISODES}"

