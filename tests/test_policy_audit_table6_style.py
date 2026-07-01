from __future__ import annotations

import pandas as pd

from src.policy_audit import build_alignment_gate, build_table6_policy_audit


def test_table6_policy_audit_counts_all_six_rows():
    edges = pd.DataFrame(
        [
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 0, "j": 1, "edge_active_before": False, "cooperation_pair_type": "cooperate_cooperate"},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 0, "j": 2, "edge_active_before": True, "cooperation_pair_type": "cooperate_cooperate"},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 0, "j": 3, "edge_active_before": False, "cooperation_pair_type": "cooperate_defect"},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 1, "j": 2, "edge_active_before": True, "cooperation_pair_type": "cooperate_defect"},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 1, "j": 3, "edge_active_before": False, "cooperation_pair_type": "defect_defect"},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 2, "j": 3, "edge_active_before": True, "cooperation_pair_type": "defect_defect"},
        ]
    )
    recs = pd.DataFrame(
        [
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 0, "j": 1, "recommendation": 1},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 0, "j": 3, "recommendation": 1},
            {"planner": "graphnet", "episode_id": 0, "round": 1, "i": 2, "j": 3, "recommendation": -1},
        ]
    )
    audit = build_table6_policy_audit(edges, recs)
    assert len(audit) == 6
    assert set(audit["pair_type"]) == {"C-C", "C-D", "D-D"}
    cc_add = audit[(audit["pair_type"] == "C-C") & (audit["change_type"] == "add")].iloc[0]
    dd_add = audit[(audit["pair_type"] == "D-D") & (audit["change_type"] == "add")].iloc[0]
    dd_delete = audit[(audit["pair_type"] == "D-D") & (audit["change_type"] == "delete")].iloc[0]
    assert cc_add["recommendation_rate"] == 1.0
    assert dd_add["recommendation_rate"] == 0.0
    assert dd_delete["recommendation_rate"] == 1.0
    gate = build_alignment_gate(audit, checkpoint_rounds=100000)
    assert gate["table6_direction"]["cc_add_high"]
    assert gate["table6_direction"]["dd_add_low"]
    assert gate["table6_direction"]["dd_delete_high"]

