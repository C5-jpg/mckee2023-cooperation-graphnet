from src.config import GameConfig
from src.graphnet import GraphNet
from src.policy_probe import active_bit_probe_rows


def test_active_bit_probe_changes_only_edge_feature_path() -> None:
    cfg = GameConfig()
    model = GraphNet(cfg)

    rows = active_bit_probe_rows(model, cfg, "cpu", "change")

    assert len(rows) == 6
    assert {(row["pair_type"], row["active"]) for row in rows} == {
        ("C-C", False),
        ("C-C", True),
        ("C-D", False),
        ("C-D", True),
        ("D-D", False),
        ("D-D", True),
    }
    assert all(row["node_features_held_constant"] is True for row in rows)
    assert all(0.0 <= row["mean_p_change"] <= 1.0 for row in rows)
    assert all(row["entropy"] >= 0.0 for row in rows)

