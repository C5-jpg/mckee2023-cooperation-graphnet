#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
TARGET_ROUNDS="${TARGET_ROUNDS:-1000000}"
TARGET_MAX_WALL_MINUTES="${TARGET_MAX_WALL_MINUTES:-}"
RESUME_ARGS=()
if [ -n "${TARGET_RESUME_CHECKPOINT:-}" ]; then
  RESUME_ARGS=(--resume "${TARGET_RESUME_CHECKPOINT}")
fi
WALL_ARGS=()
if [ -n "${TARGET_MAX_WALL_MINUTES}" ]; then
  WALL_ARGS=(--max-wall-minutes "${TARGET_MAX_WALL_MINUTES}")
fi
"${PYTHON_BIN}" -m src.train --mode target --rounds "${TARGET_ROUNDS}" --batch-size "${TARGET_BATCH_SIZE:-32}" --checkpoint-interval-rounds "${TARGET_CHECKPOINT_INTERVAL_ROUNDS:-10000}" "${WALL_ARGS[@]}" --root . "${RESUME_ARGS[@]}"
