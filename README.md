# Paper Reproduction: GraphNet Social Planner

This repository contains a from-scratch Python/PyTorch implementation for the non-human computational reproduction of a cooperative network game social-planner paper.

The code focuses on:

- cooperative network game environment;
- simulated human bots;
- rule-based social planners;
- GraphNet planner;
- single-process A2C baseline;
- distributed actor-learner / V-trace training;
- evaluation metrics;
- Table 6-style policy audit;
- checkpointing and structured logging.

Human-subject infrastructure is intentionally out of scope: no Prolific flow, no participant UI, no questionnaires, no payment workflow, no OCR, and no paper-figure recreation.

## Repository Layout

```text
src/              Core implementation
scripts/          Training, evaluation, diagnostics, and checkpoint audit scripts
tests/            Mechanism and smoke tests
paper_inputs/     Place paper Markdown/PDF inputs here if running full provenance logging
docs/             Short validation notes
pyproject.toml    Python dependencies
```

## Install

```bash
python -m pip install -e .
```

Python 3.10+ is recommended. The main dependencies are PyTorch, NumPy, pandas, pyarrow, NetworkX, SciPy, scikit-learn, PyYAML, tqdm, pytest, and matplotlib.

## Test

```bash
python -m pytest
```

For a quick syntax check:

```bash
python -m compileall -q src scripts tests
```

Current release check:

- `compileall`: passed;
- mechanism/unit smoke runner: `60 passed, 0 failed`;
- tiny distributed V-trace smoke: `60` simulated rounds, `4` episodes, checkpoint generated.

## Quick Training Smoke

```bash
python -m src.distributed_vtrace_train \
  --root smoke_outputs/distributed_vtrace_seed42_tiny \
  --rounds 60 \
  --seed 42 \
  --device cpu \
  --batch-size 2 \
  --num-actors 2 \
  --actor-torch-threads 1 \
  --checkpoint-interval-rounds 30 \
  --action-semantics target_state \
  --feature-mode paper_minimal
```

## Paper-Alignment Training

The current paper-alignment pilot uses:

- seed: `42`;
- action semantics: `target_state`;
- feature mode: `paper_minimal`;
- algorithm: `distributed_actor_learner_vtrace`;
- checkpoint interval: approximately every `10,000` simulated game rounds;
- evaluation interval: approximately every `10,000` simulated game rounds.

Example server launcher:

```bash
TRAIN_GPU=0 \
EVAL_GPU=1 \
NUM_ACTORS=16 \
TARGET_ROUNDS=1000000 \
SEED=42 \
FEATURE_MODE=paper_minimal \
ACTION_SEMANTICS=target_state \
bash scripts/run_distributed_vtrace_seed42.sh
```

The 50M continuation entry point is:

```bash
RESUME_CHECKPOINT=/path/to/latest.pt \
TARGET_ROUNDS=50000000 \
TRAIN_GPU=0 \
EVAL_GPU=1 \
bash scripts/run_distributed_vtrace_resume.sh
```

## Notes

This is a research-engineering reproduction codebase, not a claim of full statistical reproduction of the paper. A single-seed 1M-round run can validate engineering and mechanism alignment gates, but it is not equivalent to the paper's full replicate-scale experiment.

See [docs/MECHANISM_VALIDATION.zh-CN.md](docs/MECHANISM_VALIDATION.zh-CN.md) for the mechanism-validation summary of the current implementation.
