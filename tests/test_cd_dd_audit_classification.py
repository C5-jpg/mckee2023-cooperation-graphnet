import pandas as pd

from src.policy_audit import build_table6_policy_audit


def test_audit_uses_pre_recommendation_edge_state_and_current_actions() -> None:
    edges = pd.DataFrame(
        [
            {
                "planner": "graphnet",
                "episode_id": 0,
                "round": 4,
                "i": 0,
                "j": 1,
                "edge_active_before": True,
                "cooperation_pair_type": "defect_defect",
            },
            {
                "planner": "graphnet",
                "episode_id": 0,
                "round": 4,
                "i": 2,
                "j": 3,
                "edge_active_before": False,
                "cooperation_pair_type": "defect_defect",
            },
            {
                "planner": "graphnet",
                "episode_id": 0,
                "round": 4,
                "i": 4,
                "j": 5,
                "edge_active_before": True,
                "cooperation_pair_type": "cooperate_defect",
            },
            {
                "planner": "graphnet",
                "episode_id": 0,
                "round": 4,
                "i": 6,
                "j": 7,
                "edge_active_before": False,
                "cooperation_pair_type": "cooperate_defect",
            },
        ]
    )
    recs = pd.DataFrame(
        [
            {"planner": "graphnet", "episode_id": 0, "round": 4, "i": 0, "j": 1, "recommendation": -1},
            {"planner": "graphnet", "episode_id": 0, "round": 4, "i": 6, "j": 7, "recommendation": 1},
        ]
    )

    audit = build_table6_policy_audit(edges, recs, planner="graphnet")

    dd_delete = audit[(audit["pair_type"] == "D-D") & (audit["change_type"] == "delete")].iloc[0]
    dd_add = audit[(audit["pair_type"] == "D-D") & (audit["change_type"] == "add")].iloc[0]
    cd_delete = audit[(audit["pair_type"] == "C-D") & (audit["change_type"] == "delete")].iloc[0]
    cd_add = audit[(audit["pair_type"] == "C-D") & (audit["change_type"] == "add")].iloc[0]

    assert dd_delete["count_recommended"] == 1
    assert dd_delete["count_not_recommended"] == 0
    assert dd_add["count_recommended"] == 0
    assert dd_add["count_not_recommended"] == 1
    assert cd_delete["count_recommended"] == 0
    assert cd_delete["count_not_recommended"] == 1
    assert cd_add["count_recommended"] == 1
    assert cd_add["count_not_recommended"] == 0

