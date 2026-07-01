from __future__ import annotations

import numpy as np

from src.policy_diagnostics import policy_action_is_change, policy_probability_fields
from src.policy_probe import probability_fields as probe_probability_fields


def test_target_state_probability_maps_active_edge_change_to_target_inactive() -> None:
    fields = policy_probability_fields(np.array([2.0, 0.0]), active=True, action_semantics="target_state")
    assert fields["prob_recommend_change"] == fields["prob_target_inactive"]
    assert fields["prob_recommend_change"] > 0.80
    assert fields["logit_margin_recommend_change_minus_nochange"] == 2.0


def test_target_state_probability_maps_inactive_edge_change_to_target_active() -> None:
    fields = policy_probability_fields(np.array([0.0, 2.0]), active=False, action_semantics="target_state")
    assert fields["prob_recommend_change"] == fields["prob_target_active"]
    assert fields["prob_recommend_change"] > 0.80
    assert fields["logit_margin_recommend_change_minus_nochange"] == 2.0


def test_change_semantics_keeps_class1_as_recommend_change() -> None:
    fields = policy_probability_fields(np.array([2.0, 0.0]), active=True, action_semantics="change")
    assert fields["prob_recommend_change"] < 0.20
    assert fields["logit_margin_recommend_change_minus_nochange"] == -2.0


def test_policy_action_change_indicator_respects_semantics() -> None:
    assert policy_action_is_change(0, active=True, action_semantics="target_state") == 1
    assert policy_action_is_change(1, active=True, action_semantics="target_state") == 0
    assert policy_action_is_change(1, active=False, action_semantics="target_state") == 1
    assert policy_action_is_change(0, active=False, action_semantics="target_state") == 0
    assert policy_action_is_change(1, active=True, action_semantics="change") == 1
    assert policy_action_is_change(0, active=True, action_semantics="change") == 0


def test_policy_probe_uses_same_target_state_probability_mapping() -> None:
    fields = probe_probability_fields(np.array([2.0, 0.0]), active=True, action_semantics="target_state")
    assert fields["mean_p_change"] == fields["prob_recommend_change"]
    assert fields["prob_target_inactive"] > fields["prob_target_active"]
