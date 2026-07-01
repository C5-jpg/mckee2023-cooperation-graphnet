#!/usr/bin/env bash
set -euo pipefail

# Seed-controlled GraphNet training launcher.
#
# The current formal reproduction keeps the existing project seed. Set SEEDS to
# a comma-separated list only for optional sanity checks.

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-outputs/paper_scale_replicates}"
SEEDS="${SEEDS:-20260624}"
TARGET_ROUNDS="${TARGET_ROUNDS:-50000000}"
CHECKPOINT_INTERVAL_ROUNDS="${CHECKPOINT_INTERVAL_ROUNDS:-10000}"
EVAL_INTERVAL_ROUNDS="${EVAL_INTERVAL_ROUNDS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-256}"
GPUS="${GPUS:-0,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
ACTION_SEMANTICS="${ACTION_SEMANTICS:-target_state}"
FEATURE_MODE="${FEATURE_MODE:-augmented}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs configured. Set GPUS=0,1,2,3 or similar." >&2
  exit 2
fi
if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL="${#GPU_LIST[@]}"
fi

mkdir -p "$OUT_ROOT"
mkdir -p "$OUT_ROOT/logs"

launch_one() {
  local rep="$1"
  local gpu="$2"
  local seed="$3"
  local rep_name
  rep_name="$(printf 'replicate_%02d' "$rep")"
  local rep_root="$OUT_ROOT/$rep_name"
  mkdir -p "$rep_root" "$rep_root/stage_paper_scale"

  if [[ ! -e "$rep_root/paper_inputs" ]]; then
    ln -s "$(realpath "$ROOT_DIR/paper_inputs")" "$rep_root/paper_inputs"
  fi

  cat > "$rep_root/stage_paper_scale/run_config.json" <<JSON
{
  "replicate": $rep,
  "seed": $seed,
  "target_rounds": $TARGET_ROUNDS,
  "checkpoint_interval_rounds": $CHECKPOINT_INTERVAL_ROUNDS,
  "eval_interval_rounds": $EVAL_INTERVAL_ROUNDS,
  "eval_episodes": $EVAL_EPISODES,
  "final_eval_episodes": $FINAL_EVAL_EPISODES,
  "action_semantics": "$ACTION_SEMANTICS",
  "feature_mode": "$FEATURE_MODE",
  "gpu": "$gpu"
}
JSON

  (
    set -euo pipefail
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[replicate $rep] start seed=$seed gpu=$gpu target_rounds=$TARGET_ROUNDS"
    "$PYTHON" -m src.train \
      --mode custom \
      --rounds "$TARGET_ROUNDS" \
      --seed "$seed" \
      --root "$rep_root" \
      --checkpoint-interval-rounds "$CHECKPOINT_INTERVAL_ROUNDS" \
      --batch-size 32 \
      --action-semantics "$ACTION_SEMANTICS" \
      --feature-mode "$FEATURE_MODE" \
      --device cuda

    "$PYTHON" scripts/evaluate_checkpoints.py \
      --root "$rep_root" \
      --interval-rounds "$EVAL_INTERVAL_ROUNDS" \
      --min-rounds "$EVAL_INTERVAL_ROUNDS" \
      --max-rounds "$TARGET_ROUNDS" \
      --n-eval-episodes "$EVAL_EPISODES" \
      --action-semantics "$ACTION_SEMANTICS" \
      --feature-mode "$FEATURE_MODE" \
      --device cuda

    "$PYTHON" -m src.evaluate \
      --root "$rep_root" \
      --checkpoint outputs/train/checkpoints/latest.pt \
      --eval-dir outputs/eval \
      --raw-dir outputs/raw_eval \
      --stage-summary stage_paper_scale/final_eval_summary.json \
      --n-eval-episodes "$FINAL_EVAL_EPISODES" \
      --seed "$seed" \
      --action-semantics "$ACTION_SEMANTICS" \
      --feature-mode "$FEATURE_MODE" \
      --device cuda

    "$PYTHON" -m src.policy_diagnostics \
      --checkpoint "$rep_root/outputs/train/checkpoints/latest.pt" \
      --out-dir "$rep_root/outputs/policy_diagnostics/final" \
      --n-episodes "$FINAL_EVAL_EPISODES" \
      --seed "$seed" \
      --action-semantics "$ACTION_SEMANTICS" \
      --feature-mode "$FEATURE_MODE" \
      --device cuda

    echo "[replicate $rep] complete"
  ) > "$rep_root/stage_paper_scale/run.log" 2>&1 &

  local pid=$!
  echo "$pid" > "$rep_root/stage_paper_scale/run.pid"
  echo "$rep_name seed=$seed gpu=$gpu pid=$pid" | tee -a "$OUT_ROOT/logs/launcher.log"
}

active_jobs() {
  jobs -pr | wc -l
}

for ((rep=0; rep<${#SEED_LIST[@]}; rep++)); do
  while [[ "$(active_jobs)" -ge "$MAX_PARALLEL" ]]; do
    sleep 30
  done
  gpu="${GPU_LIST[$((rep % ${#GPU_LIST[@]}))]}"
  launch_one "$rep" "$gpu" "${SEED_LIST[$rep]}"
done

wait
echo "All paper-scale replicates finished." | tee -a "$OUT_ROOT/logs/launcher.log"
