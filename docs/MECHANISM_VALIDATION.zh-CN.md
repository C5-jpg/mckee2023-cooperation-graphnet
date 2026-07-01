# 机制核验摘要

本仓库当前主线实现经过一次单 seed 机制核验 pilot。该摘要用于说明代码版本的机制对齐状态，不等同于论文完整统计复现结论。

## 当前主线

| 项目 | 值 |
|---|---|
| seed | `42` |
| action semantics | `target_state` |
| feature mode | `paper_minimal` |
| training algorithm | `distributed_actor_learner_vtrace` |
| actor processes | `16` |
| target simulated game rounds | `1,000,000` |
| completed simulated game rounds | `1,000,005` |
| completed episodes | `66,667` |
| checkpoint/evaluation cadence | approximately every `10,000` rounds |

## 最终 checkpoint 的 Table 6 风格 policy audit

| pair type | change type | recommendation rate | direction |
|---|---|---:|---|
| C-C | add | 0.821120 | pass |
| C-C | delete | 0.176608 | pass |
| C-D | add | 0.319197 | pass |
| C-D | delete | 0.988310 | pass |
| D-D | add | 0.000686 | pass |
| D-D | delete | 1.000000 | pass |

## 最终 GraphNet evaluation 摘要

| 指标 | 值 |
|---|---:|
| mean cooperation | 0.781576 |
| final round cooperation | 0.779297 |
| mean capital | 4.482044 |
| final mean capital | 7.667627 |
| Gini final | 0.039623 |
| final network density | 0.635286 |
| relative degree cooperators final | 5.141789 |
| Newman assortativity final | -0.100791 |
| Rand assortativity final | -0.094578 |
| Shirado assortativity final | -0.080179 |
| defector disassortativity final | 0.048802 |

## 结论边界

可以说：

- 工程训练链路真实跑通；
- checkpoint、evaluation、Table 6 风格 policy audit 链路可用；
- 单 seed 42、1M rounds 的最终 checkpoint 通过了本轮 engineering consistency gate。

不应说：

- 已完整复现论文所有 computational results；
- 已复现论文的 30 replicate 统计体系；
- Table 6 数值等价于 human participant GraphNet condition；
- 单 seed pilot 可以替代完整统计复现。

