#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" -m src.train --mode smoke --rounds "${SMOKE_ROUNDS:-150}" --batch-size 4 --checkpoint-interval-rounds 150 --root .

