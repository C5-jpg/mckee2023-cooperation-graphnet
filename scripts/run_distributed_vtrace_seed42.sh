#!/usr/bin/env bash
set -euo pipefail

# Single-seed paper-gold training launcher.
#
# This entrypoint starts a from-scratch distributed actor-learner/V-trace run
# and a checkpoint monitor that evaluates every 10k simulated game rounds.

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
TARGET_ROUNDS="${TARGET_ROUNDS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_ACTORS="${NUM_ACTORS:-16}"
ACTOR_TORCH_THREADS="${ACTOR_TORCH_THREADS:-1}"
CHECKPOINT_INTERVAL_ROUNDS="${CHECKPOINT_INTERVAL_ROUNDS:-10000}"
EVAL_INTERVAL_ROUNDS="${EVAL_INTERVAL_ROUNDS:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
DIAGNOSTICS_EPISODES="${DIAGNOSTICS_EPISODES:-32}"
MONITOR_SLEEP_SECONDS="${MONITOR_SLEEP_SECONDS:-120}"
ACTION_SEMANTICS="${ACTION_SEMANTICS:-target_state}"
FEATURE_MODE="${FEATURE_MODE:-paper_minimal}"
TRAIN_GPU="${TRAIN_GPU:-${GPU:-0}}"
EVAL_GPU="${EVAL_GPU:-$TRAIN_GPU}"
OUT_BASE="${OUT_BASE:-outputs/distributed_vtrace}"
RUN_ID="${RUN_ID:-seed_${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$OUT_BASE/$RUN_ID}"

ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$ROOT_DIR/$RUN_ROOT" ;;
esac

mkdir -p "$RUN_ROOT/stage20_distributed_vtrace" "$RUN_ROOT/logs"
if [[ ! -e "$RUN_ROOT/paper_inputs" ]]; then
  ln -s "$ROOT_DIR/paper_inputs" "$RUN_ROOT/paper_inputs"
fi

cat > "$RUN_ROOT/stage20_distributed_vtrace/launcher_config.json" <<JSON
{
  "algorithm": "distributed_actor_learner_vtrace",
  "from_scratch": true,
  "seed": $SEED,
  "target_rounds": $TARGET_ROUNDS,
  "batch_size": $BATCH_SIZE,
  "num_actors": $NUM_ACTORS,
  "actor_torch_threads": $ACTOR_TORCH_THREADS,
  "checkpoint_interval_rounds": $CHECKPOINT_INTERVAL_ROUNDS,
  "eval_interval_rounds": $EVAL_INTERVAL_ROUNDS,
  "eval_episodes": $EVAL_EPISODES,
  "diagnostics_episodes": $DIAGNOSTICS_EPISODES,
  "monitor_sleep_seconds": $MONITOR_SLEEP_SECONDS,
  "action_semantics": "$ACTION_SEMANTICS",
  "feature_mode": "$FEATURE_MODE",
  "train_gpu": "$TRAIN_GPU",
  "eval_gpu": "$EVAL_GPU",
  "run_root": "$RUN_ROOT"
}
JSON

echo "$$" > "$RUN_ROOT/stage20_distributed_vtrace/launcher.pid"
echo "[launcher] run_root=$RUN_ROOT"
echo "[launcher] train_gpu=$TRAIN_GPU eval_gpu=$EVAL_GPU seed=$SEED target_rounds=$TARGET_ROUNDS"

(
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON" -m src.distributed_vtrace_train \
    --root "$RUN_ROOT" \
    --rounds "$TARGET_ROUNDS" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --num-actors "$NUM_ACTORS" \
    --actor-torch-threads "$ACTOR_TORCH_THREADS" \
    --checkpoint-interval-rounds "$CHECKPOINT_INTERVAL_ROUNDS" \
    --action-semantics "$ACTION_SEMANTICS" \
    --feature-mode "$FEATURE_MODE" \
    --device cuda
) > "$RUN_ROOT/stage20_distributed_vtrace/train.stdout.log" 2> "$RUN_ROOT/stage20_distributed_vtrace/train.stderr.log" &

train_pid=$!
echo "$train_pid" > "$RUN_ROOT/stage20_distributed_vtrace/train.pid"

monitor_loop() {
  cd "$ROOT_DIR"
  while kill -0 "$train_pid" 2>/dev/null; do
    date -u +"%Y-%m-%dT%H:%M:%SZ monitor evaluation sweep"
    if CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/evaluate_checkpoints.py \
      --root "$RUN_ROOT" \
      --interval-rounds "$EVAL_INTERVAL_ROUNDS" \
      --min-rounds "$EVAL_INTERVAL_ROUNDS" \
      --max-rounds "$TARGET_ROUNDS" \
      --n-eval-episodes "$EVAL_EPISODES" \
      --diagnostics-episodes "$DIAGNOSTICS_EPISODES" \
      --action-semantics "$ACTION_SEMANTICS" \
      --feature-mode "$FEATURE_MODE" \
      --device cuda; then
      date -u +"%Y-%m-%dT%H:%M:%SZ monitor sweep complete"
    else
      status=$?
      date -u +"%Y-%m-%dT%H:%M:%SZ monitor sweep failed status=$status"
      cat > "$RUN_ROOT/stage20_distributed_vtrace/eval_last_error.json" <<JSON
{"timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "status": $status}
JSON
    fi
    sleep "$MONITOR_SLEEP_SECONDS"
  done

  date -u +"%Y-%m-%dT%H:%M:%SZ final evaluation sweep"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/evaluate_checkpoints.py \
    --root "$RUN_ROOT" \
    --interval-rounds "$EVAL_INTERVAL_ROUNDS" \
    --min-rounds "$EVAL_INTERVAL_ROUNDS" \
    --max-rounds "$TARGET_ROUNDS" \
    --n-eval-episodes "$EVAL_EPISODES" \
    --diagnostics-episodes "$DIAGNOSTICS_EPISODES" \
    --action-semantics "$ACTION_SEMANTICS" \
    --feature-mode "$FEATURE_MODE" \
    --device cuda
}

monitor_loop > "$RUN_ROOT/stage20_distributed_vtrace/eval_monitor.log" 2>&1 &
monitor_pid=$!
echo "$monitor_pid" > "$RUN_ROOT/stage20_distributed_vtrace/eval_monitor.pid"

set +e
wait "$train_pid"
train_status=$?
wait "$monitor_pid"
monitor_status=$?
set -e

cat > "$RUN_ROOT/stage20_distributed_vtrace/launcher_exit.json" <<JSON
{
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "train_status": $train_status,
  "monitor_status": $monitor_status,
  "run_root": "$RUN_ROOT"
}
JSON

if [[ "$train_status" -ne 0 ]]; then
  exit "$train_status"
fi
if [[ "$monitor_status" -ne 0 ]]; then
  exit "$monitor_status"
fi
