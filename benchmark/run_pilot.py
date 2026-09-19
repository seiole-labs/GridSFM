#!/usr/bin/env python3
"""Run the complete seeded GridSFM 5x2 pilot benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

from benchmark.evaluate import evaluate, render_summary
from benchmark.run_inference import run_inference, serving_metadata
from benchmark.validate_pyg_scenario import validate_scenario
from gridsfm import load_model


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "power_grid" / "US" / "topology_solver_pipeline"


def _docker_julia(image: str, expression: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{REPO_ROOT}:/workspace",
            "-w", "/workspace/power_grid/US/topology_solver_pipeline",
            image, "-c", expression,
        ],
        check=check,
    )


def _write_prediction(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def _row(result: dict[str, Any]) -> dict[str, Any]:
    accuracy = result["accuracy"]
    physics = result["physics"]
    timing = result["timing_seconds"]
    cost = accuracy["cost"]
    signed_cost_percent = cost["signed_error"] / abs(cost["reference"]) * 100.0
    return {
        "scenario": Path(result["scenario"]).name,
        "buses": result["counts"]["buses"],
        "generators": result["counts"]["generators"],
        "branches": result["counts"]["branches"],
        "cost_mape_percent": accuracy["cost"]["mape_percent"],
        "cost_signed_percent": signed_cost_percent,
        "cost_direction": "lower" if signed_cost_percent < 0 else ("higher" if signed_cost_percent > 0 else "equal"),
        "V_mae_pu": accuracy["V"]["mae"],
        "theta_mae_degrees": accuracy["theta"]["mae_degrees"],
        "Pg_mae_pu": accuracy["Pg"]["mae"],
        "Qg_mae_pu": accuracy["Qg"]["mae"],
        "branch_P_mae_pu": accuracy["branch_flow"]["P"]["mae"],
        "branch_Q_mae_pu": accuracy["branch_flow"]["Q"]["mae"],
        "gridsfm_kcl_P_mae_pu": physics["gridsfm"]["kcl"]["active"]["mae_pu"],
        "gridsfm_kcl_P_max_pu": physics["gridsfm"]["kcl"]["active"]["max_abs_pu"],
        "gridsfm_kcl_Q_mae_pu": physics["gridsfm"]["kcl"]["reactive"]["mae_pu"],
        "gridsfm_max_loading_ratio": physics["gridsfm"]["thermal"]["max_loading_ratio"],
        "gridsfm_overloaded_branches": physics["gridsfm"]["thermal"]["overloaded_branch_count"],
        "ac_opf_kcl_P_mae_pu": physics["ac_opf"]["kcl"]["active"]["mae_pu"],
        "ac_opf_kcl_P_max_pu": physics["ac_opf"]["kcl"]["active"]["max_abs_pu"],
        "ac_opf_kcl_Q_mae_pu": physics["ac_opf"]["kcl"]["reactive"]["mae_pu"],
        "ac_opf_max_loading_ratio": physics["ac_opf"]["thermal"]["max_loading_ratio"],
        "ac_opf_overloaded_branches": physics["ac_opf"]["thermal"]["overloaded_branch_count"],
        "ac_opf_seconds": timing["ac_opf_solve"],
        "gridsfm_forward_seconds": timing["gridsfm_forward"],
        "gridsfm_request_seconds": timing["gridsfm_resident_request"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric_keys = [
        key for key, value in rows[0].items()
        if key != "scenario" and isinstance(value, (int, float))
    ]
    return {
        key: {
            "mean": statistics.mean(float(row[key]) for row in rows),
            "median": statistics.median(float(row[key]) for row in rows),
            "max": max(float(row[key]) for row in rows),
        }
        for key in numeric_keys
    }


def _summary(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        f"# GridSFM Pilot Summary ({len(rows)} accepted scenarios)", "",
        "## System size and prediction accuracy", "",
        "| Scenario | Buses | Gens | Branches | Cost MAPE | Cost bias | V MAE | θ MAE | Pg MAE | Qg MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['buses']} | {row['generators']} | {row['branches']} | "
            f"{row['cost_mape_percent']:.3f}% | {row['cost_signed_percent']:+.3f}% ({row['cost_direction']}) | "
            f"{row['V_mae_pu']:.4f} | {row['theta_mae_degrees']:.3f}° | "
            f"{row['Pg_mae_pu']:.4f} | {row['Qg_mae_pu']:.4f} |"
        )
    lines.append(
        f"| **Overall mean ({len(rows)})** | {aggregate['buses']['mean']:.0f} | {aggregate['generators']['mean']:.1f} | "
        f"{aggregate['branches']['mean']:.0f} | {aggregate['cost_mape_percent']['mean']:.3f}% | "
        f"{aggregate['cost_signed_percent']['mean']:+.3f}% ({'lower' if aggregate['cost_signed_percent']['mean'] < 0 else 'higher'}) | "
        f"{aggregate['V_mae_pu']['mean']:.4f} | {aggregate['theta_mae_degrees']['mean']:.3f}° | "
        f"{aggregate['Pg_mae_pu']['mean']:.4f} | {aggregate['Qg_mae_pu']['mean']:.4f} |"
    )
    lines.extend([
        "", "## Physics and timing", "",
        "| Scenario | GridSFM KCL P MAE | GridSFM KCL P max | AC-OPF KCL P MAE | AC-OPF KCL P max | Max loading | AC-OPF solve | GridSFM forward | GridSFM request |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['gridsfm_kcl_P_mae_pu']:.4f} | "
            f"{row['gridsfm_kcl_P_max_pu']:.6f} | {row['ac_opf_kcl_P_mae_pu']:.6f} | {row['ac_opf_kcl_P_max_pu']:.6f} | "
            f"{row['gridsfm_max_loading_ratio']:.3f}x | "
            f"{row['ac_opf_seconds']:.3f}s | {row['gridsfm_forward_seconds']:.3f}s | "
            f"{row['gridsfm_request_seconds']:.3f}s |"
        )
    lines.append(
        f"| **Overall mean (10)** | {aggregate['gridsfm_kcl_P_mae_pu']['mean']:.5f} | "
        f"{aggregate['gridsfm_kcl_P_max_pu']['mean']:.5f} | {aggregate['ac_opf_kcl_P_mae_pu']['mean']:.6f} | "
        f"{aggregate['ac_opf_kcl_P_max_pu']['mean']:.6f} | {aggregate['gridsfm_max_loading_ratio']['mean']:.3f}x | "
        f"{aggregate['ac_opf_seconds']['mean']:.3f}s | {aggregate['gridsfm_forward_seconds']['mean']:.3f}s | "
        f"{aggregate['gridsfm_request_seconds']['mean']:.3f}s |"
    )
    lines.extend([
        "", "## Per-topology averages", "",
        "Each row averages the two accepted scenarios for one topology; raw generator/state vectors are not averaged.", "",
        "| Topology | Scenarios | Buses | Gens | Branches | Cost MAPE | Signed cost bias | GridSFM KCL P MAE | GridSFM KCL P max | AC-OPF KCL P MAE | AC-OPF KCL P max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["scenario"].split("_mixed_")[0], []).append(row)
    for topology, group in groups.items():
        mean = lambda key: statistics.mean(float(item[key]) for item in group)
        lines.append(
            f"| {topology} | {len(group)} | {mean('buses'):.0f} | {mean('generators'):.1f} | {mean('branches'):.0f} | "
            f"{mean('cost_mape_percent'):.3f}% | {mean('cost_signed_percent'):+.3f}% | "
            f"{mean('gridsfm_kcl_P_mae_pu'):.5f} | {mean('gridsfm_kcl_P_max_pu'):.5f} | "
            f"{mean('ac_opf_kcl_P_mae_pu'):.6f} | {mean('ac_opf_kcl_P_max_pu'):.6f} |"
        )
    lines.extend([
        "", "## Aggregate means", "",
        f"- Cost MAPE: {aggregate['cost_mape_percent']['mean']:.4f}%",
        f"- Signed cost bias: {aggregate['cost_signed_percent']['mean']:+.4f}% (positive means GridSFM predicts higher cost)",
        f"- V MAE: {aggregate['V_mae_pu']['mean']:.6f} pu",
        f"- Theta MAE: {aggregate['theta_mae_degrees']['mean']:.4f}°",
        f"- Pg/Qg MAE: {aggregate['Pg_mae_pu']['mean']:.6f} / {aggregate['Qg_mae_pu']['mean']:.6f} pu",
        f"- GridSFM request: {aggregate['gridsfm_request_seconds']['mean']:.4f}s",
        f"- AC-OPF solve: {aggregate['ac_opf_seconds']['mean']:.4f}s",
        f"- AC-OPF active KCL MAE/max: {aggregate['ac_opf_kcl_P_mae_pu']['mean']:.8f} / {aggregate['ac_opf_kcl_P_max_pu']['max']:.8f} pu",
        "", "Timing boundaries are defined in `benchmark/TIMING.md`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=REPO_ROOT / "artifacts/pilot/selection.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "model/checkpoints/gridsfm_open_v1.1.pt")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--docker-image", default="topo_solver_pipe")
    parser.add_argument("--scenarios-per-topology", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    selection_path = args.selection.resolve()
    args.output_root = args.output_root.resolve()
    selection = json.loads(selection_path.read_text())
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    model = load_model(args.checkpoint, device=device)
    model.eval()

    scenario_dir = args.output_root / "scenarios"
    prediction_dir = args.output_root / "predictions"
    result_dir = args.output_root / "results"
    for directory in (scenario_dir, prediction_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)
    run_metadata_path = args.output_root / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(serving_metadata(device, 1), indent=2) + "\n")

    results: list[dict[str, Any]] = []
    for topology_index, topology in enumerate(selection["selected"], start=1):
        accepted = 0
        candidate_index = 1
        while accepted < args.scenarios_per_topology:
            if candidate_index > args.max_attempts:
                raise RuntimeError(f"{topology['name']}: exhausted {args.max_attempts} attempts")
            sample_id = f"{topology['name']}_mixed_{candidate_index:04d}"
            print(f"\n=== {sample_id} ===", flush=True)
            _docker_julia(
                args.docker_image,
                "julia --project=. generate_mixed_scenario.jl "
                f"/workspace/{selection_path.relative_to(REPO_ROOT)} {candidate_index} "
                f"/workspace/{scenario_dir.relative_to(REPO_ROOT)} {topology_index}",
            )
            scenario_path = scenario_dir / f"{sample_id}.pyg.json"
            transformed_path = scenario_dir / f"{sample_id}_transformed.json"
            solve = _docker_julia(
                args.docker_image,
                "julia --project=. export_gridsfm_data.jl "
                f"/workspace/{transformed_path.relative_to(REPO_ROOT)} "
                f"/workspace/{scenario_path.relative_to(REPO_ROOT)}",
                check=False,
            )
            candidate_index += 1
            if solve.returncode != 0:
                print(f"rejected {sample_id}: AC-OPF did not solve", flush=True)
                continue
            validate_scenario(json.loads(scenario_path.read_text()))

            prediction_path = prediction_dir / f"{sample_id}_prediction.json"
            prediction = run_inference(model, scenario_path.relative_to(REPO_ROOT), device)
            _write_prediction(prediction_path, prediction)
            result = evaluate(scenario_path, prediction_path, run_metadata_path)
            result_path = result_dir / f"{sample_id}_result.json"
            summary_path = result_dir / f"{sample_id}_summary.md"
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            summary_path.write_text(render_summary(result))
            results.append(result)
            accepted += 1

    rows = [_row(result) for result in results]
    aggregate = _aggregate(rows)
    (args.output_root / "results.json").write_text(json.dumps({
        "selection": selection,
        "results": results,
        "aggregate": aggregate,
    }, indent=2) + "\n")
    with (args.output_root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "summary.md").write_text(_summary(rows, aggregate))
    print(f"\ncomplete: {len(results)} scenarios -> {args.output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
