from __future__ import annotations

import math

import numpy as np

from src.metrics import (
    core_periphery_approx_degree_concentration,
    defector_disassortativity,
    newman_assortativity,
    rand_assortativity,
    relative_degree_cooperators,
    shirado_assortativity,
)


def test_relative_degree_missing_class_is_nan():
    adjacency = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(adjacency, False)
    actions = np.ones(4, dtype=int)
    assert math.isnan(relative_degree_cooperators(adjacency, actions))


def test_assortativity_metrics_are_named_separately():
    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    )
    actions = np.array([1, 1, 0, 0], dtype=int)
    assert isinstance(newman_assortativity(adjacency, actions), float)
    assert isinstance(rand_assortativity(adjacency, actions), float)
    assert isinstance(shirado_assortativity(adjacency, actions), float)
    assert isinstance(defector_disassortativity(adjacency, actions), float)
    assert 0.0 <= core_periphery_approx_degree_concentration(adjacency) <= 1.0

