#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" -m src.evaluate --root . --n-eval-episodes "${N_EVAL_EPISODES:-16}" --checkpoint "${GRAPHNET_CHECKPOINT:-outputs/train/checkpoints/latest.pt}"

