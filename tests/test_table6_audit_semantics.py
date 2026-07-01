from __future__ import annotations

import pandas as pd

from src.policy_audit import build_alignment_gate, build_table6_policy_audit


def test_table6_audit_counts_only_nonzero_actual_recommendations() -> None:
    edges = pd.DataFrame(
        [
            {
                "planner": "graphnet",
                "episode_id": 0,
                "round": 1,
                "i": 0,
                "j": 1,
                "edge_active_before": True,
                "cooperation_pair_type": "defect_defect",
            },
            {
                "planner": "graphnet",
                "episode_id": 0,
                "round": 1,
                "i": 0,
                "j": 2,
                "edge_active_before": False,
                "cooperation_pair_type": "defect_defect",
            },
        ]
    )
    recs = pd.DataFrame(
        [
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 0, "j": 1, "recommendation": 0},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 0, "j": 2, "recommendation": 1},
        ]
    )

    audit = build_table6_policy_audit(edges, recs, planner="graphnet", action_semantics="target_state")

    dd_delete = audit[(audit["pair_type"] == "D-D") & (audit["change_type"] == "delete")].iloc[0]
    dd_add = audit[(audit["pair_type"] == "D-D") & (audit["change_type"] == "add")].iloc[0]
    assert dd_delete["count_recommended"] == 0
    assert dd_delete["count_not_recommended"] == 1
    assert dd_add["count_recommended"] == 1
    assert dd_add["count_not_recommended"] == 0
    assert set(audit["action_semantics"]) == {"target_state"}


def test_alignment_gate_records_action_semantics_metadata() -> None:
    audit = build_table6_policy_audit(pd.DataFrame(), pd.DataFrame(), action_semantics="target_state")
    gate = build_alignment_gate(audit, checkpoint_rounds=100000, action_semantics="target_state")
    assert gate["action_semantics"] == "target_state"
