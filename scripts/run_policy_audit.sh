#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
ROOT="${ROOT:-.}"
RAW_DIR="${RAW_DIR:-outputs/raw_eval}"
OUT_DIR="${OUT_DIR:-outputs/policy_audit}"

"${PYTHON_BIN}" -m src.policy_audit \
  --root "${ROOT}" \
  --raw-dir "${RAW_DIR}" \
  --out-dir "${OUT_DIR}"

