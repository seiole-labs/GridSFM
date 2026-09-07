import math
from pathlib import Path

import pytest

from benchmark.evaluate import branch_metrics, error_metrics, evaluate, reference_flows, split_prediction_flows, thermal_metrics


ROOT = Path(__file__).resolve().parent.parent


def test_circular_angle_error_wraps_at_two_pi():
    metrics = error_metrics([math.pi - 0.01], [-math.pi + 0.01], circular=True)
    assert metrics["mae"] == pytest.approx(0.02)
    assert metrics["rmse"] == pytest.approx(0.02)


def test_error_metrics_mse_and_rmse_are_not_mae():
    metrics = error_metrics([1.0, 4.0], [0.0, 0.0])
    assert metrics == pytest.approx({"mae": 2.5, "mse": 8.5, "rmse": math.sqrt(8.5)})


def test_branch_reference_export_order_is_reversed_by_end():
    predicted = {"ac_line": [[10.0, 20.0, 30.0, 40.0]]}
    # Exported solution is [Pji,Qji,Pij,Qij], unlike prediction ordering.
    solution = {"ac_line": {"features": [[30.0, 40.0, 10.0, 20.0]]}}
    metrics = branch_metrics(predicted, solution)
    assert metrics["P"]["mae"] == 0.0
    assert metrics["Q"]["mae"] == 0.0


def test_reference_flows_convert_export_order_for_physics_checks():
    converted = reference_flows({
        "ac_line": {"features": [[30.0, 40.0, 10.0, 20.0]]},
    })
    assert converted == {"ac_line": [[10.0, 20.0, 30.0, 40.0]]}


def test_prediction_flow_family_order_and_counts_are_enforced():
    pred = {
        "flow_edge_types": ["transformer", "ac_line"], "flow_edge_counts": [1, 1],
        "Pij": [1, 2], "Qij": [1, 2], "Pji": [1, 2], "Qji": [1, 2],
    }
    edges = {name: {"senders": [0], "receivers": [1], "features": [[0] * width]}
             for name, width in (("ac_line", 9), ("transformer", 11))}
    with pytest.raises(ValueError, match="families/order"):
        split_prediction_flows(pred, edges)


def test_thermal_metrics_use_apparent_power_at_worst_end_and_correct_rate_column():
    flows = {"ac_line": [[3.0, 4.0, 0.0, 0.0]], "transformer": [[0.0, 0.0, 6.0, 8.0]]}
    ac = [0.0] * 9; ac[6] = 4.0
    tr = [0.0] * 11; tr[4] = 20.0
    metrics = thermal_metrics(flows, {
        "ac_line": {"features": [ac]}, "transformer": {"features": [tr]},
    })
    assert metrics["rated_branch_count"] == 2
    assert metrics["overloaded_branch_count"] == 1
    assert metrics["overload_fraction"] == 0.5
    assert metrics["max_overload_pu"] == 1.0
    assert metrics["max_loading_ratio"] == pytest.approx(5.0 / math.sqrt(16.0001))


def test_saved_ohio_result_recomputes_deterministically_with_correct_units():
    paths = (
        ROOT / "artifacts/pilot/ohio_mixed_0001.pyg.json",
        ROOT / "artifacts/pilot/ohio_mixed_0001_prediction.json",
        ROOT / "artifacts/pilot/run_metadata.json",
    )
    first = evaluate(*paths)
    second = evaluate(*paths)
    assert first == second
    assert first["accuracy"]["cost"]["reference"] == pytest.approx(2705258.377864397)
    theta = first["accuracy"]["theta"]
    assert theta["mae_degrees"] == pytest.approx(math.degrees(theta["mae"]))
    assert first["counts"]["branches"] == 993 + 520
    assert first["physics"]["ac_opf"]["kcl"]["active"]["mae_pu"] < 1e-4
    assert first["physics"]["gridsfm"]["kcl"]["flow_source"] == "gridsfm_inference_pi_model_flows"
