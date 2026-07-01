from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


SEED = 20260624

ACTION_SEMANTICS = ("target_state", "change")
FEATURE_MODES = ("augmented", "paper_minimal")


def validate_action_semantics(value: str) -> str:
    if value not in ACTION_SEMANTICS:
        raise ValueError(f"action_semantics must be one of {ACTION_SEMANTICS}, got {value!r}")
    return value


def validate_feature_mode(value: str) -> str:
    if value not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {value!r}")
    return value


@dataclass(frozen=True)
class GameConfig:
    n_players: int = 16
    episode_length: int = 15
    initial_graph_p: float = 0.3
    initial_capital: float = 1.0
    benefit_b: float = 0.1
    cost_c: float = 0.05
    recommendation_penalty_weight: float = 1.0
    action_semantics: str = "target_state"

    def __post_init__(self) -> None:
        validate_action_semantics(self.action_semantics)

    @property
    def n_possible_edges(self) -> int:
        return self.n_players * (self.n_players - 1) // 2


@dataclass(frozen=True)
class BotConfig:
    mu_theta: float = -0.304
    sigma_theta: float = 2.410
    beta0_first: float = 1.807
    beta1_first: float = 0.818
    beta0_later: float = -0.010
    beta_size: float = -0.193
    beta_num_coop: float = 0.370
    beta_frac_coop: float = 1.521
    phi_delete_defector: float = 0.774
    phi_delete_cooperator: float = 0.085
    phi_add_defector: float = 0.287
    phi_add_cooperator: float = 0.909


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = SEED
    target_simulated_rounds: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 0.0004
    discount_gamma: float = 0.99
    entropy_regularization: float = 0.004
    baseline_cost: float = 0.5
    grad_clip_norm: float = 10.0
    checkpoint_interval_rounds: int = 10_000
    heartbeat_interval_seconds: int = 300
    train_sample_interval_episodes: int = 25


@dataclass(frozen=True)
class GraphNetConfig:
    message_passing_steps: int = 2
    edge_update_1_output_size: int = 128
    node_update_1_output_size: int = 128
    global_update_1_output_size: int = 128
    edge_update_2_output_size: int = 2
    node_update_2_output_size: int = 128
    global_update_2_output_size: int = 1
    feature_mode: str = "augmented"

    def __post_init__(self) -> None:
        validate_feature_mode(self.feature_mode)


@dataclass(frozen=True)
class EvalConfig:
    seed: int = SEED
    n_eval_episodes: int = 16


@dataclass(frozen=True)
class ReproConfig:
    seed: int = SEED
    game: GameConfig = field(default_factory=GameConfig)
    bots: BotConfig = field(default_factory=BotConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    graphnet: GraphNetConfig = field(default_factory=GraphNetConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


ENCOURAGEMENT_TABLES: Dict[str, Dict[str, List[float]]] = {
    "cc": {
        "add": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.991, 0.954, 1.000],
        "delete": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.010, 0.011, 0.028, 0.035, 0.073, 0.108],
    },
    "cd": {
        "add": [0.993, 0.973, 0.914, 0.791, 0.644, 0.594, 0.463, 0.429, 0.366, 0.372, 0.361, 0.371, 0.328, 0.408],
        "delete": [0.048, 0.029, 0.145, 0.213, 0.318, 0.508, 0.608, 0.745, 0.802, 0.753, 0.741, 0.774, 0.706, 0.722],
    },
    "dd": {
        "add": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "delete": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    },
}


NEUTRAL_TABLE: Dict[str, List[float]] = {
    "add": [0.891, 0.841, 0.656, 0.642, 0.608, 0.549, 0.545, 0.538, 0.520, 0.504, 0.532, 0.518, 0.529, 0.522],
    "delete": [0.119, 0.054, 0.084, 0.102, 0.117, 0.204, 0.215, 0.224, 0.239, 0.213, 0.215, 0.237, 0.232, 0.317],
}


def resolved_config_dict(config: ReproConfig | None = None) -> dict:
    cfg = config or ReproConfig()
    return {
        "seed": cfg.seed,
        "n_players": cfg.game.n_players,
        "episode_length": cfg.game.episode_length,
        "initial_graph_model": "erdos_renyi",
        "initial_graph_p": cfg.game.initial_graph_p,
        "initial_capital": cfg.game.initial_capital,
        "benefit_b": cfg.game.benefit_b,
        "cost_c": cfg.game.cost_c,
        "action_semantics": cfg.game.action_semantics,
        "target_simulated_rounds": cfg.training.target_simulated_rounds,
        "target_episodes": (cfg.training.target_simulated_rounds + cfg.game.episode_length - 1) // cfg.game.episode_length,
        "training": {
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "discount_gamma": cfg.training.discount_gamma,
            "entropy_regularization": cfg.training.entropy_regularization,
            "baseline_cost": cfg.training.baseline_cost,
            "recommendation_penalty_weight": cfg.game.recommendation_penalty_weight,
        },
        "graphnet": asdict(cfg.graphnet),
        "bots": asdict(cfg.bots),
        "evaluation": asdict(cfg.evaluation),
    }
