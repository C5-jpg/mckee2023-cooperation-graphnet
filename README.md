# GraphNet 社会规划者：独立实现与单种子机制实验

基于 McKee et al. (2023), *Scaffolding cooperation in human groups with deep reinforcement learning*，用 Python 与 PyTorch 实现合作网络博弈、模拟玩家、GraphNet 规划者及 A2C / V-trace 训练，研究网络干预与合作行为。

> **范围与结论边界：** 这是独立研究实现，不是论文作者的官方代码，也不代表与原作者存在合作关系。现有结果记录来自 **bot 模拟、单 seed 42、约 1M game rounds**，不构成论文完整统计复现，也不能与人类被试条件等同。结果表是项目既有实验摘要；公开 main 当前未同时提供这些数值对应的原始评估输出与可下载 checkpoint。

[实验记录](#现有实验记录) · [快速开始](#快速开始) · [训练与评估](#训练与评估) · [方法与代码](#方法与代码) · [局限](#局限与后续工作)

## 研究问题与实现范围

规划者观察玩家状态和网络结构，对玩家之间的连边提出建立/断开建议。模拟玩家选择合作或背叛，并接受或拒绝建议；环境更新网络与资本，强化学习据此更新规划策略。

| 包含 | 不包含 |
|---|---|
| 合作网络环境、模拟玩家、6 种规则规划者 | 人类被试招募、知情同意、支付与问卷 |
| GraphNet、A2C、分布式 actor–learner V-trace | 原作者官方实现或官方合作关系 |
| 评估、Table 6 风格策略审计、结构化日志与测试 | 论文全部结果及 30 replicate 统计体系 |

核心实现不依赖 `torch_geometric`。论文来源：[DOI](https://doi.org/10.1038/s41562-023-01686-7) · [OSF](https://osf.io/8ahkg/)。

## 现有实验记录

### 配置与证据来源

[机制核验摘要](docs/MECHANISM_VALIDATION.zh-CN.md)记录了以下一次实验：

| 项目 | 记录值 |
|---|---|
| 训练 seed | 42 |
| 动作语义 | `target_state` |
| 输入特征 | `paper_minimal` |
| 算法 / actors | 分布式 actor–learner V-trace / 16 |
| 目标 / 实际 game rounds | 1,000,000 / 1,000,005 |
| 实际 episodes | 66,667 |
| checkpoint / 评估间隔 | 约每 10,000 rounds |

这里引用的是项目汇总记录，不是一次新的独立复算。公开结果的完整核验还需要原始评估 CSV/JSON、checkpoint、代码 commit、评估 seed/局数、运行环境和哈希对应关系。

### 合作率与网络统计

| 指标 | 项目摘要记录 |
|---|---:|
| 平均合作率 | 0.781576 |
| 末轮合作率 | 0.779297 |
| 末轮平均资本 | 7.667627 |
| 末轮 Gini | 0.039623 |
| 末轮网络密度 | 0.635286 |

### 连边建议统计

以下为 Table 6 风格的建议率统计，不是实际被接受后的边变化概率。C 表示合作玩家，D 表示背叛玩家。

| 边类型 | 建议 | 本项目单 seed 记录 | 历史 README 所列论文参考值 |
|---|---|---:|---:|
| C-C | add | 0.821120 | 0.994 |
| C-C | delete | 0.176608 | 0.030 |
| C-D | add | 0.319197 | 0.582 |
| C-D | delete | 0.988310 | 0.503 |
| D-D | add | 0.000686 | 0.000 |
| D-D | delete | 1.000000 | 1.000 |

本项目列来自[机制摘要](docs/MECHANISM_VALIDATION.zh-CN.md)；参考列保留自[该次实验的历史 README](https://github.com/C5-jpg/mckee2023-cooperation-graphnet/blob/96d5fd87a9b07e6d24b6f0114d1b538b9cc6dd60/README.md)，用于追踪已有比较记录。不同评价对象与条件不能据此认定统计等价。

这些记录显示：该 checkpoint 的 D-D 删除建议率接近 1；但 C-D delete（约 0.988 对 0.503）、C-C add（约 0.821 对 0.994）等仍有明显差异。因此可将其描述为**单种子机制实验**，不能概括为“所有策略与论文一致”。

历史记录中的训练阶段：

| 完成 rounds | D-D delete | 网络密度 | 末轮合作率 |
|---:|---:|---:|---:|
| 10,005 | 0.000 | 0.9999 | 0.642 |
| 250,125 | 0.725 | 0.5111 | 0.696 |
| 570,285 | 1.000 | 0.5391 | 0.748 |
| 1,000,005 | 1.000 | 0.6353 | 0.779 |

该阶段表同样保留自上述历史 README，未以公开原始日志重新计算。特征选择与训练算法都可能影响这些观察；在没有完整因子对照和多种子统计前，不把它们写成已确立的因果解释。

## 快速开始

### 安装与测试

[pyproject.toml](pyproject.toml)要求 Python >=3.10，依赖包含 PyTorch、NumPy、pandas、PyArrow、NetworkX、SciPy 等。

```bash
git clone https://github.com/C5-jpg/mckee2023-cooperation-graphnet.git
cd mckee2023-cooperation-graphnet
python -m venv .venv
```

Linux / macOS：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -e .
python -m pytest
python -m compileall -q src scripts tests
```

这些检查验证代码与测试所覆盖的工程行为，不替代研究结果验证。包名为 `src`，以下命令均从仓库根目录执行。

### 最小 CPU 训练与评估

```bash
python -m src.distributed_vtrace_train --root smoke_outputs/vtrace_tiny --rounds 60 --seed 42 --device cpu --batch-size 2 --num-actors 2 --actor-torch-threads 1 --checkpoint-interval-rounds 30 --action-semantics target_state --feature-mode paper_minimal
```

训练产物保存在指定 root 下，例如：

```text
smoke_outputs/vtrace_tiny/
  outputs/train/checkpoints/latest.pt
  outputs/train/train_metrics.jsonl
  outputs/train/vtrace_loss_metrics.jsonl
  stage20_distributed_vtrace/
```

使用**同一个 root 与特征设置**评估：

```bash
python -m src.evaluate --root smoke_outputs/vtrace_tiny --checkpoint outputs/train/checkpoints/latest.pt --n-eval-episodes 2 --seed 42 --device cpu --action-semantics target_state --feature-mode paper_minimal
```

这是短流程检查，不用于报告有效的合作提升。完成后检查：

- `outputs/eval/graphnet_eval_summary.json` 中 `available` 应为 `true`；
- `outputs/eval/planner_comparison.csv` 应包含 GraphNet 行；
- `outputs/eval/table6_style_policy.csv` 为该模型的策略审计；
- `outputs/raw_eval/` 包含 episode、round、player、edge、recommendation 的 Parquet 记录。

**评估器在 checkpoint 缺失时仍会运行规则基线，并写入 `available: false`。命令有输出不等于 GraphNet 已完成评估。**

## 训练与评估

### 1M rounds 主线配置

直接 CLI 适用于显式控制输出目录：

```bash
python -m src.distributed_vtrace_train --root outputs/distributed_vtrace/seed42_1m --rounds 1000000 --seed 42 --device cuda --batch-size 32 --num-actors 16 --actor-torch-threads 1 --checkpoint-interval-rounds 10000 --action-semantics target_state --feature-mode paper_minimal
```

上述配置对应既有实验的主要设置；实际运行环境、调度和随机性会影响结果，不能保证一次运行得到同一数值。

训练完成后：

```bash
python -m src.evaluate --root outputs/distributed_vtrace/seed42_1m --checkpoint outputs/train/checkpoints/latest.pt --n-eval-episodes 64 --seed 42 --device cpu --action-semantics target_state --feature-mode paper_minimal
```

当前评估器对 GraphNet 使用确定性动作，并为 GraphNet 的环境评估采用命令 seed + 12345；规则基线使用命令 seed。它不是逐局配对的相同随机流比较，统计分析时应记录这一差异。

### 自动训练与 checkpoint 监控

Linux / Bash 服务器可使用[启动脚本](scripts/run_distributed_vtrace_seed42.sh)：

```bash
RUN_ROOT=outputs/distributed_vtrace/seed42_1m_monitored TRAIN_GPU=0 EVAL_GPU=0 NUM_ACTORS=16 TARGET_ROUNDS=1000000 SEED=42 FEATURE_MODE=paper_minimal ACTION_SEMANTICS=target_state bash scripts/run_distributed_vtrace_seed42.sh
```

该脚本启动训练和评估监控两个进程。示例让它们共用 GPU 0；有第二张卡时可设置 `EVAL_GPU=1`。运行记录写在 `RUN_ROOT/stage20_distributed_vtrace/`，checkpoint 位于 `RUN_ROOT/outputs/train/checkpoints/`。

[scripts/run_eval.sh](scripts/run_eval.sh)使用固定 `--root .`，不会自动选择上面的嵌套运行目录，也未显式设置 `paper_minimal`。主线请使用前述完整评估命令，不要直接把默认脚本输出视为该模型结果。

### 继续训练

从前述 CLI 示例的 checkpoint 继续到 50M rounds，并使用新的输出 root：

```bash
python -m src.distributed_vtrace_train --root outputs/distributed_vtrace/seed42_continued --resume outputs/distributed_vtrace/seed42_1m/outputs/train/checkpoints/latest.pt --rounds 50000000 --seed 42 --device cuda --batch-size 32 --num-actors 16 --actor-torch-threads 1 --checkpoint-interval-rounds 10000 --action-semantics target_state --feature-mode paper_minimal
```

`--resume` 是实际 checkpoint 文件路径；`--root` 是本次输出目录，两者不同。保留原实验记录，按 checkpoint 中实际完成 rounds 解释后续训练，不声称恢复与不中断运行逐位一致。

需要自动评估监控时可用 [run_distributed_vtrace_resume.sh](scripts/run_distributed_vtrace_resume.sh)，显式提供 `RESUME_CHECKPOINT` 与 `RUN_ROOT`，并检查其 `MIN_EVAL_ROUNDS` 默认值是否适合该恢复节点。

## 方法与代码

### 环境与模型

默认环境为 16 玩家、15 轮，初始 ER 连边概率 0.3、初始资本 1.0、合作收益 b=0.1、成本 c=0.05。玩家先决定合作/背叛并结算收益，非末轮再由规划者提出建议、玩家裁决接受与否，更新网络及奖励；末轮不发推荐。

GraphNet 使用两步消息传递，策略输出每条候选无向边的 2 维 logits，价值输出为标量。`target_state` 将策略输出解释为目标边状态，再映射到建立、断开或保持；`paper_minimal` 与 `augmented` 是不同输入设置，训练与评估必须一致。

训练损失包含 policy gradient、baseline loss 与熵正则；分布式训练通过 V-trace 修正异步 actor 与 learner 的策略差异。

| 文件 | 内容 |
|---|---|
| [src/config.py](src/config.py) | 游戏、模型与训练参数 |
| [src/env.py](src/env.py) / [src/bots.py](src/bots.py) | 环境顺序、收益、模拟玩家行为与建议接受 |
| [src/graphnet.py](src/graphnet.py) | GraphNet、特征与动作映射 |
| [src/a2c.py](src/a2c.py) / [src/vtrace.py](src/vtrace.py) | 学习目标 |
| [src/distributed_vtrace_train.py](src/distributed_vtrace_train.py) | actors、learner、checkpoint 与日志 |
| [src/evaluate.py](src/evaluate.py) / [src/policy_audit.py](src/policy_audit.py) | 评估与 Table 6 风格策略审计 |
| [src/metrics.py](src/metrics.py) | 合作、网络与不平等指标 |
| [tests/](tests/) | 工程与机制检查 |
| [paper_inputs/README.md](paper_inputs/README.md) | 论文输入说明 |

当前主要训练配置：batch size 32、学习率 4e-4、折扣 0.99、推荐惩罚权重 1.0、熵正则 0.004、baseline cost 0.5、Adam、梯度裁剪 10.0。源码与实际运行配置是核验依据。

## 局限与后续工作

- 现有摘要来自单训练 seed，不能替代多 seed 的均值、方差与置信区间。
- bot 模拟与人类被试条件不同，不能把数值接近解释为统计等价。
- 当前公开结果缺少一套由 commit、配置、checkpoint hash 和原始评估文件组成的完整对应链。
- 现有工程审计门槛是项目内的检查标准，不是论文结论的独立验证。
- core-periphery 近似与论文的 bimodular surprise 指标不是同一个统计量，相关结论需明确口径。
- 特征与算法的因果作用需要统一预算的完整对照、多种子实验与失败结果记录。
- 50M 续训脚本存在不表示 30 × 50M 的论文级计算预算已经完成。

## 相关实现、来源与许可

[scaffolding-cooperation](https://github.com/C5-jpg/scaffolding-cooperation)是同一研究主题下的另一套独立实现，默认 seed 0、`bidirectional_mean`、RMSprop；本仓库记录的实验采用 seed 42、`paper_minimal + target_state`、Adam。两者分别维护，不共享结果归属或 checkpoint 格式保证。

原论文：McKee et al. (2023), *Scaffolding cooperation in human groups with deep reinforcement learning*. Nature Human Behaviour. [DOI: 10.1038/s41562-023-01686-7](https://doi.org/10.1038/s41562-023-01686-7)。

研究方法应引用原论文；使用本实现时同时记录本仓库 URL 与 commit。当前仓库未提供 LICENSE 文件，代码许可尚未明确。问题与复现记录可通过 [Issues](https://github.com/C5-jpg/mckee2023-cooperation-graphnet/issues) 交流。

