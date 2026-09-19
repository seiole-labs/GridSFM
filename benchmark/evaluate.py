#!/usr/bin/env python3
"""Evaluate one saved GridSFM prediction against its AC-OPF reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch_geometric.data import Batch

from gridsfm import load_pyg_json
from gridsfm.loss import _kcl_residuals, _predicted_flows
from gridsfm.schema import GEN_CP0_IDX, GEN_CP1_IDX, GEN_CP2_IDX

THERMAL_OVERLOAD_TOL_PU = 1e-6


def error_metrics(predicted: Iterable[float], reference: Iterable[float], *, circular: bool = False) -> dict[str, float]:
    p = torch.as_tensor(list(predicted), dtype=torch.float64)
    r = torch.as_tensor(list(reference), dtype=torch.float64)
    if p.shape != r.shape or p.numel() == 0:
        raise ValueError(f"metric arrays must be non-empty and shape-aligned: {p.shape} != {r.shape}")
    delta = p - r
    if circular:
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    squared = delta.square()
    return {
        "mae": float(delta.abs().mean()),
        "mse": float(squared.mean()),
        "rmse": float(squared.mean().sqrt()),
    }


def split_prediction_flows(pred: dict[str, Any], grid_edges: dict[str, Any]) -> dict[str, list[list[float]]]:
    expected_types = [name for name in ("ac_line", "transformer") if name in grid_edges and grid_edges[name].get("senders")]
    types = pred.get("flow_edge_types", [])
    counts = pred.get("flow_edge_counts", [])
    if types != expected_types or len(types) != len(counts):
        raise ValueError(f"prediction flow families/order {types} do not match scenario {expected_types}")
    fields = [pred[name] for name in ("Pij", "Qij", "Pji", "Qji")]
    total = sum(counts)
    if any(len(values) != total for values in fields):
        raise ValueError("prediction flow arrays do not match flow_edge_counts")
    result: dict[str, list[list[float]]] = {}
    offset = 0
    for name, count in zip(types, counts):
        if count != len(grid_edges[name]["senders"]):
            raise ValueError(f"prediction has {count} {name} flows; scenario has {len(grid_edges[name]['senders'])}")
        result[name] = [list(row) for row in zip(*(values[offset:offset + count] for values in fields))]
        offset += count
    return result


def branch_metrics(predicted_by_type: dict[str, list[list[float]]], solution_edges: dict[str, Any]) -> dict[str, dict[str, float]]:
    predicted_p: list[float] = []
    predicted_q: list[float] = []
    reference_p: list[float] = []
    reference_q: list[float] = []
    for edge_type in ("ac_line", "transformer"):
        if edge_type not in predicted_by_type:
            continue
        # Predictions are [Pij,Qij,Pji,Qji]; exporter references are [Pji,Qji,Pij,Qij].
        for pred_row, ref_row in zip(predicted_by_type[edge_type], solution_edges[edge_type]["features"], strict=True):
            predicted_p.extend((pred_row[0], pred_row[2]))
            predicted_q.extend((pred_row[1], pred_row[3]))
            reference_p.extend((ref_row[2], ref_row[0]))
            reference_q.extend((ref_row[3], ref_row[1]))
    return {"P": error_metrics(predicted_p, reference_p), "Q": error_metrics(predicted_q, reference_q)}


def reference_flows(solution_edges: dict[str, Any]) -> dict[str, list[list[float]]]:
    """Convert exported [Pji,Qji,Pij,Qij] rows to evaluator prediction order."""
    return {
        edge_type: [[row[2], row[3], row[0], row[1]] for row in solution_edges[edge_type]["features"]]
        for edge_type in ("ac_line", "transformer")
        if edge_type in solution_edges
    }


def thermal_metrics(predicted_by_type: dict[str, list[list[float]]], grid_edges: dict[str, Any]) -> dict[str, float | int]:
    loadings: list[float] = []
    overloads: list[float] = []
    rate_index = {"ac_line": 6, "transformer": 4}
    for edge_type, rows in predicted_by_type.items():
        features = grid_edges[edge_type]["features"]
        for flow, attrs in zip(rows, features, strict=True):
            rate = float(attrs[rate_index[edge_type]])
            if rate <= 0.001:
                continue
            peak = max(math.hypot(flow[0], flow[1]), math.hypot(flow[2], flow[3]))
            loadings.append(peak / math.sqrt(rate * rate + 1e-4))
            overloads.append(max(peak - rate, 0.0))
    if not loadings:
        raise ValueError("scenario contains no rated branches")
    # Ignore solver-scale floating-point noise at a nominally binding limit.
    overloaded = sum(value > THERMAL_OVERLOAD_TOL_PU for value in overloads)
    return {
        "rated_branch_count": len(loadings),
        "overloaded_branch_count": overloaded,
        "max_loading_ratio": max(loadings),
        "overload_fraction": overloaded / len(loadings),
        "max_overload_pu": max(overloads),
    }


def kcl_metrics(
    scenario_path: Path,
    pred: dict[str, Any],
    flows_by_type: dict[str, list[list[float]]] | None = None,
    flow_source: str | None = None,
) -> dict[str, Any]:
    # KCL needs only the raw electrical graph. Avoid feature preparation here:
    # it would add unrelated cycle/PE work and a disk cache to evaluation.
    batch = Batch.from_data_list([load_pyg_json(scenario_path)])
    batch["bus"].pred = torch.tensor(list(zip(pred["theta"], pred["V"])), dtype=torch.float32)
    batch["generator"].pred = torch.tensor(list(zip(pred["Pg"], pred["Qg"])), dtype=torch.float32)
    if flows_by_type is None:
        i, j, pij, qij, pji, qji = _predicted_flows(batch)
        flow_source = flow_source or "loss_helper_pi_model_from_theta_V"
    else:
        i_rows, j_rows, flow_rows = [], [], []
        for edge_type in ("ac_line", "transformer"):
            rows = flows_by_type.get(edge_type, [])
            if not rows:
                continue
            key = ("bus", edge_type, "bus")
            i_rows.append(batch[key].edge_index[0])
            j_rows.append(batch[key].edge_index[1])
            flow_rows.extend(rows)
        i = torch.cat(i_rows)
        j = torch.cat(j_rows)
        flow_tensor = torch.tensor(flow_rows, dtype=torch.float32)
        pij, qij, pji, qji = flow_tensor.unbind(dim=1)
        flow_source = flow_source or "provided_branch_flows"
    rp, rq = _kcl_residuals(batch, i, j, pij, qij, pji, qji, batch["bus"].pred[:, 1])
    def summarize(values: torch.Tensor) -> dict[str, float]:
        values = values.double()
        return {"mae_pu": float(values.abs().mean()), "rmse_pu": float(values.square().mean().sqrt()), "max_abs_pu": float(values.abs().max())}
    return {"active": summarize(rp), "reactive": summarize(rq), "flow_source": flow_source}


def evaluate(scenario_path: Path, prediction_path: Path, run_metadata_path: Path) -> dict[str, Any]:
    scenario = json.loads(scenario_path.read_text())
    prediction_doc = json.loads(prediction_path.read_text())
    pred = prediction_doc["predictions"]
    nodes = scenario["grid"]["nodes"]
    solution_nodes = scenario["solution"]["nodes"]

    if len(pred["V"]) != len(solution_nodes["bus"]) or len(pred["Pg"]) != len(solution_nodes["generator"]):
        raise ValueError("prediction node dimensions do not match reference")
    ref_theta = [row[0] for row in solution_nodes["bus"]]
    ref_v = [row[1] for row in solution_nodes["bus"]]
    ref_pg = [row[0] for row in solution_nodes["generator"]]
    ref_qg = [row[1] for row in solution_nodes["generator"]]

    generator_features = nodes["generator"]
    def total_cost(pg: list[float]) -> float:
        return sum(row[GEN_CP2_IDX] * power**2 + row[GEN_CP1_IDX] * power + row[GEN_CP0_IDX]
                   for row, power in zip(generator_features, pg, strict=True))
    reference_cost = total_cost(ref_pg)
    predicted_cost = total_cost(pred["Pg"])
    cost_error = predicted_cost - reference_cost

    flows = split_prediction_flows(pred, scenario["grid"]["edges"])
    ac_opf_flows = reference_flows(scenario["solution"]["edges"])
    ac_opf_state = {
        "theta": ref_theta,
        "V": ref_v,
        "Pg": ref_pg,
        "Qg": ref_qg,
    }
    theta = error_metrics(pred["theta"], ref_theta, circular=True)
    theta["mae_degrees"] = math.degrees(theta["mae"])
    theta["rmse_degrees"] = math.degrees(theta["rmse"])
    prediction_timing = prediction_doc["timing"]
    # Older artifacts called the model-resident request "end_to_end_seconds".
    # Treat that field as resident timing unless an explicit E2E scope proves
    # it was recorded by the external request observer.
    resident_request_seconds = prediction_timing.get(
        "resident_request_seconds",
        prediction_timing.get("end_to_end_seconds"),
    )
    complete_e2e_seconds = (
        prediction_timing.get("end_to_end_seconds")
        if prediction_timing.get("end_to_end_scope")
        else None
    )
    return {
        "scenario": scenario_path.as_posix(),
        "prediction": prediction_path.as_posix(),
        "status": scenario["metadata"].get("termination_status"),
        "counts": {
            "buses": len(ref_v),
            "generators": len(ref_pg),
            "branches": sum(pred["flow_edge_counts"]),
            "scenario_size_kb": scenario_path.stat().st_size / 1024.0,
        },
        "accuracy": {
            "cost": {
                "reference": reference_cost, "predicted": predicted_cost,
                "absolute_error": abs(cost_error),
                "signed_error": cost_error,
                "mape_percent": abs(cost_error) / abs(reference_cost) * 100.0,
                "mse": cost_error**2, "rmse": abs(cost_error),
            },
            "V": error_metrics(pred["V"], ref_v), "theta": theta,
            "Pg": error_metrics(pred["Pg"], ref_pg), "Qg": error_metrics(pred["Qg"], ref_qg),
            "branch_flow": branch_metrics(flows, scenario["solution"]["edges"]),
        },
        "physics": {
            "gridsfm": {
                "kcl": kcl_metrics(
                    scenario_path, pred, flows,
                    flow_source="gridsfm_inference_pi_model_flows",
                ),
                "thermal": thermal_metrics(flows, scenario["grid"]["edges"]),
            },
            "ac_opf": {
                "kcl": kcl_metrics(
                    scenario_path, ac_opf_state, ac_opf_flows,
                    flow_source="ac_opf_exported_branch_flows",
                ),
                "thermal": thermal_metrics(ac_opf_flows, scenario["grid"]["edges"]),
            },
        },
        "timing_seconds": {
            "ac_opf_solve": scenario["metadata"].get("solve_time_seconds"),
            "ac_opf_ipopt_iterations": scenario["metadata"].get("ipopt_iterations"),
            "gridsfm_forward": prediction_timing["forward_seconds"],
            "gridsfm_resident_request": resident_request_seconds,
            "gridsfm_end_to_end": complete_e2e_seconds,
            "gridsfm_end_to_end_scope": prediction_timing.get("end_to_end_scope"),
        },
        "serving_metadata": json.loads(run_metadata_path.read_text()),
    }


def render_summary(result: dict[str, Any]) -> str:
    a, p, t = result["accuracy"], result["physics"], result["timing_seconds"]
    gp, ap = p["gridsfm"], p["ac_opf"]
    counts = result["counts"]
    signed_cost_percent = a["cost"]["signed_error"] / abs(a["cost"]["reference"]) * 100.0
    cost_direction = "higher" if signed_cost_percent > 0 else ("lower" if signed_cost_percent < 0 else "equal")
    e2e_display = (
        f"{t['gridsfm_end_to_end']:.3f} s"
        if t["gridsfm_end_to_end"] is not None
        else "not measured"
    )
    return f"""# GridSFM Pilot Result

Scenario: `{result['scenario']}`  
AC-OPF status: `{result['status']}`

| Metric | Value |
|---|---:|
| Buses / generators / branches | {counts['buses']} / {counts['generators']} / {counts['branches']} |
| AC-OPF reference cost | {a['cost']['reference']:.4f} |
| GridSFM predicted cost | {a['cost']['predicted']:.4f} |
| Cost MAPE | {a['cost']['mape_percent']:.4f}% |
| Signed cost bias | {signed_cost_percent:+.4f}% ({cost_direction}) |
| Cost absolute error | {a['cost']['absolute_error']:.4f} |
| V MAE | {a['V']['mae']:.6f} pu |
| Theta MAE | {a['theta']['mae']:.6f} rad ({a['theta']['mae_degrees']:.4f} deg) |
| Pg MAE | {a['Pg']['mae']:.6f} pu |
| Qg MAE | {a['Qg']['mae']:.6f} pu |
| Branch P MAE | {a['branch_flow']['P']['mae']:.6f} pu |
| Branch Q MAE | {a['branch_flow']['Q']['mae']:.6f} pu |
| GridSFM active KCL MAE | {gp['kcl']['active']['mae_pu']:.6f} pu |
| GridSFM active KCL max | {gp['kcl']['active']['max_abs_pu']:.6f} pu |
| AC-OPF active KCL MAE | {ap['kcl']['active']['mae_pu']:.6f} pu |
| AC-OPF active KCL max | {ap['kcl']['active']['max_abs_pu']:.6f} pu |
| GridSFM reactive KCL MAE | {gp['kcl']['reactive']['mae_pu']:.6f} pu |
| AC-OPF reactive KCL MAE | {ap['kcl']['reactive']['mae_pu']:.6f} pu |
| GridSFM maximum thermal loading | {gp['thermal']['max_loading_ratio']:.4f}x |
| AC-OPF maximum thermal loading | {ap['thermal']['max_loading_ratio']:.4f}x |
| GridSFM overloaded branches | {gp['thermal']['overloaded_branch_count']} / {gp['thermal']['rated_branch_count']} ({gp['thermal']['overload_fraction']:.2%}) |
| AC-OPF overloaded branches | {ap['thermal']['overloaded_branch_count']} / {ap['thermal']['rated_branch_count']} ({ap['thermal']['overload_fraction']:.2%}) |
| AC-OPF solve | {t['ac_opf_solve']:.3f} s |
| GridSFM forward | {t['gridsfm_forward']:.3f} s |
| GridSFM resident request | {t['gridsfm_resident_request']:.3f} s |
| GridSFM cold request E2E | {e2e_display} |

All power errors use the dataset's per-unit convention. GridSFM KCL uses the branch flows produced by the inference π-model; AC-OPF KCL uses the exported solver branch flows. Thermal overload counts ignore excesses of at most {THERMAL_OVERLOAD_TOL_PU:.0e} pu as numerical tolerance.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.scenario, args.prediction, args.run_metadata)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.summary.write_text(render_summary(result))
    print(json.dumps({"result": args.output.as_posix(), "summary": args.summary.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
