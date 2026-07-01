#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
SEED="${SEED:-20260624}"
TARGET_ROUNDS="${TARGET_ROUNDS:-1000000}"
TARGET_CHECKPOINT_INTERVAL_ROUNDS="${TARGET_CHECKPOINT_INTERVAL_ROUNDS:-10000}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-32}"
EVAL_INTERVAL_ROUNDS="${EVAL_INTERVAL_ROUNDS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-128}"
EVAL_MIN_ROUNDS="${EVAL_MIN_ROUNDS:-100000}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-2}"

RESUME_ARGS=()
if [ -n "${TARGET_RESUME_CHECKPOINT:-}" ]; then
  if [ "${ALLOW_RESUME_AFTER_FIX:-0}" != "1" ]; then
    echo "Refusing to resume from TARGET_RESUME_CHECKPOINT during post-fix formal retraining." >&2
    exit 2
  fi
  RESUME_ARGS=(--resume "${TARGET_RESUME_CHECKPOINT}")
fi

mkdir -p stage10_retrain_1m outputs/eval_by_checkpoint outputs/policy_audit

(
  export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
  "${PYTHON_BIN}" -m src.train \
    --mode target \
    --rounds "${TARGET_ROUNDS}" \
    --seed "${SEED}" \
    --batch-size "${TARGET_BATCH_SIZE}" \
    --checkpoint-interval-rounds "${TARGET_CHECKPOINT_INTERVAL_ROUNDS}" \
    --root . \
    "${RESUME_ARGS[@]}"
) > stage10_retrain_1m/target_1m_train.log 2>&1 &
train_pid=$!
echo "${train_pid}" > stage10_retrain_1m/target_1m_train.pid

while kill -0 "${train_pid}" 2>/dev/null; do
  (
    export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
    "${PYTHON_BIN}" scripts/evaluate_checkpoints.py \
      --root . \
      --interval-rounds "${EVAL_INTERVAL_ROUNDS}" \
      --min-rounds "${EVAL_MIN_ROUNDS}" \
      --max-rounds "${TARGET_ROUNDS}" \
      --n-eval-episodes "${EVAL_EPISODES}" \
      --device cuda
  ) >> stage10_retrain_1m/checkpoint_eval_monitor.log 2>&1 || true
  sleep "${EVAL_MONITOR_SLEEP_SECONDS:-60}"
done

wait "${train_pid}"

(
  export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
  "${PYTHON_BIN}" scripts/evaluate_checkpoints.py \
    --root . \
    --interval-rounds "${EVAL_INTERVAL_ROUNDS}" \
    --min-rounds "${EVAL_MIN_ROUNDS}" \
    --max-rounds "${TARGET_ROUNDS}" \
    --n-eval-episodes "${EVAL_EPISODES}" \
    --device cuda
  "${PYTHON_BIN}" -m src.evaluate \
    --root . \
    --checkpoint outputs/train/checkpoints/latest.pt \
    --n-eval-episodes "${FINAL_EVAL_EPISODES}" \
    --seed "${SEED}" \
    --device cuda
) >> stage10_retrain_1m/checkpoint_eval_monitor.log 2>&1
