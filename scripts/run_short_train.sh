#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" -m src.train --mode short --rounds "${SHORT_ROUNDS:-3000}" --batch-size 8 --checkpoint-interval-rounds 1500 --root .

