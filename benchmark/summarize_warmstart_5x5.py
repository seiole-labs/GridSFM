#!/usr/bin/env python3
"""Summarize paired warm-start JSON files without third-party dependencies."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ARMS = ("cold", "gridsfm_warm", "dc_warm")
ERROR_KEYS = (
    "bus_theta_degrees",
    "bus_vm_pu",
    "generator_pg_pu",
    "generator_qg_pu",
    "all_branch_active_flows_pu",
    "all_branch_reactive_flows_pu",
)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    return "—" if value != value else f"{value:.{digits}f}"


def load_rows(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((output_dir / "config.json").read_text())
    rows: list[dict[str, Any]] = []
    for sample_id in config["sample_ids"]:
        path = output_dir / f"{sample_id}_runs.json"
        if not path.exists():
            continue
        document = json.loads(path.read_text())
        for triplet in document["measured_runs"]:
            cold_seconds = triplet["cold"]["ac_solver_reported_seconds"]
            for arm in ARMS:
                record = triplet[arm]
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "topology": sample_id.split("_mixed_")[0],
                    "perturbation_mode": document["perturbation_mode"],
                    "repetition": record["repetition"],
                    "arm": arm,
                    "termination_status": record["termination_status"],
                    "converged": record["converged"],
                    "objective": record["objective"],
                    "objective_signed_difference_vs_matched_cold": record[
                        "objective_signed_difference_vs_matched_cold"
                    ],
                    "objective_percent_difference_vs_matched_cold": record[
                        "objective_percent_difference_vs_matched_cold"
                    ],
                    "seed_preparation_seconds": record["seed_preparation_seconds"],
                    "ac_model_build_seconds": record["ac_model_build_seconds"],
                    "ac_solver_reported_seconds": record["ac_solver_reported_seconds"],
                    "ac_ipopt_iterations": record.get("ac_ipopt_iterations"),
                    "dc_seed_ipopt_iterations": record.get("dc_presolve", {}).get(
                        "ipopt_iterations"
                    ),
                    "ac_solve_wall_seconds": record["ac_solve_wall_seconds"],
                    "seeded_total_seconds": record["seeded_total_seconds"],
                    "workflow_e2e_seconds": record["workflow_e2e_seconds"],
                    "seeded_total_speedup_vs_cold": (
                        cold_seconds / record["seeded_total_seconds"]
                    ),
                    "ac_only_speedup_vs_cold": (
                        cold_seconds / record["ac_solver_reported_seconds"]
                    ),
                }
                errors = record.get("final_solution_differences_vs_matched_cold", {}).get(
                    "aggregate_errors", {}
                )
                for key in ERROR_KEYS:
                    for stat in ("mae", "rmse", "max_abs"):
                        row[f"{key}_{stat}"] = errors.get(key, {}).get(stat)
                rows.append(row)
    return config, rows


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], {}).setdefault(row["arm"], []).append(row)
    scenarios: dict[str, Any] = {}
    for sample_id, by_arm in grouped.items():
        scenario: dict[str, Any] = {}
        for arm, arm_rows in by_arm.items():
            numeric = (
                "seed_preparation_seconds",
                "ac_model_build_seconds",
                "ac_solver_reported_seconds",
                "ac_solve_wall_seconds",
                "seeded_total_seconds",
                "workflow_e2e_seconds",
                "seeded_total_speedup_vs_cold",
                "ac_only_speedup_vs_cold",
                "objective_percent_difference_vs_matched_cold",
            )
            scenario[arm] = {
                "observations": len(arm_rows),
                "all_converged": all(row["converged"] for row in arm_rows),
                **{f"median_{key}": median([float(row[key]) for row in arm_rows]) for key in numeric},
            }
            ac_iterations = [
                float(row["ac_ipopt_iterations"])
                for row in arm_rows
                if row["ac_ipopt_iterations"] is not None
            ]
            dc_seed_iterations = [
                float(row["dc_seed_ipopt_iterations"])
                for row in arm_rows
                if row["dc_seed_ipopt_iterations"] is not None
            ]
            scenario[arm]["median_ac_ipopt_iterations"] = median(ac_iterations)
            scenario[arm]["median_dc_seed_ipopt_iterations"] = median(dc_seed_iterations)
            for key in ERROR_KEYS:
                values = [row[f"{key}_max_abs"] for row in arm_rows if row[f"{key}_max_abs"] is not None]
                scenario[arm][f"maximum_{key}_absolute_difference"] = max(values, default=None)
        scenarios[sample_id] = scenario

    completed_ids = {
        sample_id
        for sample_id, by_arm in scenarios.items()
        if all(
            arm in by_arm and by_arm[arm]["observations"] == config["repetitions"]
            for arm in ARMS
        )
    }
    overall: dict[str, Any] = {}
    for arm in ARMS:
        available = [scenarios[sample_id][arm] for sample_id in completed_ids]
        overall[arm] = {
            "completed_scenarios": len(available),
            "all_converged": all(value["all_converged"] for value in available),
            "median_of_scenario_median_seeded_total_seconds": median(
                [value["median_seeded_total_seconds"] for value in available]
            ),
            "median_of_scenario_median_ac_solver_reported_seconds": median(
                [value["median_ac_solver_reported_seconds"] for value in available]
            ),
            "median_of_scenario_median_workflow_e2e_seconds": median(
                [value["median_workflow_e2e_seconds"] for value in available]
            ),
            "median_of_scenario_median_speedup_vs_cold": median(
                [value["median_seeded_total_speedup_vs_cold"] for value in available]
            ),
            "median_of_scenario_median_ac_only_speedup_vs_cold": median(
                [value["median_ac_only_speedup_vs_cold"] for value in available]
            ),
            "median_of_scenario_median_ac_ipopt_iterations": median(
                [
                    value["median_ac_ipopt_iterations"]
                    for value in available
                    if value["median_ac_ipopt_iterations"]
                    == value["median_ac_ipopt_iterations"]
                ]
            ),
        }
    comparisons = {
        "gridsfm_ac_only_faster_than_cold_scenarios": sum(
            scenarios[s]["gridsfm_warm"]["median_ac_solver_reported_seconds"]
            < scenarios[s]["cold"]["median_ac_solver_reported_seconds"]
            for s in completed_ids
        ),
        "gridsfm_seeded_total_faster_than_cold_scenarios": sum(
            scenarios[s]["gridsfm_warm"]["median_seeded_total_seconds"]
            < scenarios[s]["cold"]["median_seeded_total_seconds"]
            for s in completed_ids
        ),
        "dc_seeded_total_faster_than_cold_scenarios": sum(
            scenarios[s]["dc_warm"]["median_seeded_total_seconds"]
            < scenarios[s]["cold"]["median_seeded_total_seconds"]
            for s in completed_ids
        ),
        "gridsfm_seeded_total_faster_than_dc_scenarios": sum(
            scenarios[s]["gridsfm_warm"]["median_seeded_total_seconds"]
            < scenarios[s]["dc_warm"]["median_seeded_total_seconds"]
            for s in completed_ids
        ),
    }
    warm_rows = [row for row in rows if row["arm"] != "cold"]
    correctness = {
        "maximum_objective_percent_difference_vs_matched_cold": max(
            (float(row["objective_percent_difference_vs_matched_cold"]) for row in warm_rows),
            default=None,
        ),
        **{
            f"maximum_{key}_absolute_difference": max(
                (float(row[f"{key}_max_abs"]) for row in warm_rows if row[f"{key}_max_abs"] is not None),
                default=None,
            )
            for key in ERROR_KEYS
        },
    }
    topologies: dict[str, Any] = {}
    for sample_id in completed_ids:
        topology = sample_id.split("_mixed_")[0]
        topologies.setdefault(topology, {arm: [] for arm in ARMS})
        for arm in ARMS:
            topologies[topology][arm].append(scenarios[sample_id][arm])
    for by_arm in topologies.values():
        for arm, values in by_arm.items():
            by_arm[arm] = {
                "scenario_count": len(values),
                "median_ac_solver_reported_seconds": median(
                    [value["median_ac_solver_reported_seconds"] for value in values]
                ),
                "median_seeded_total_seconds": median(
                    [value["median_seeded_total_seconds"] for value in values]
                ),
                "median_ac_only_speedup_vs_cold": median(
                    [value["median_ac_only_speedup_vs_cold"] for value in values]
                ),
                "median_seeded_total_speedup_vs_cold": median(
                    [value["median_seeded_total_speedup_vs_cold"] for value in values]
                ),
            }
    return {
        "expected_scenarios": len(config["sample_ids"]),
        "completed_scenarios": len(completed_ids),
        "partial_scenarios": len(scenarios) - len(completed_ids),
        "measured_rows": len(rows),
        "scenarios": scenarios,
        "overall": overall,
        "comparisons": comparisons,
        "correctness": correctness,
        "topologies": dict(sorted(topologies.items())),
    }


def markdown(config: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        "# GridSFM and DC warm-start 5×5 benchmark",
        "",
        f"Completed **{result['completed_scenarios']}/{result['expected_scenarios']}** mixed-combination perturbation scenarios "
        f"({result['partial_scenarios']} currently partial). "
        f"Each completed scenario has up to {config['repetitions']} measured repetitions after one unmeasured warm-up.",
        "",
        "## Main readout",
        "",
        f"- GridSFM reduced the AC solver-only median from {fmt(result['overall']['cold']['median_of_scenario_median_ac_solver_reported_seconds'])} s to {fmt(result['overall']['gridsfm_warm']['median_of_scenario_median_ac_solver_reported_seconds'])} s and was faster on {result['comparisons']['gridsfm_ac_only_faster_than_cold_scenarios']}/25 scenarios.",
        f"- After adding recorded GridSFM inference and seed mapping, its median seeded total was {fmt(result['overall']['gridsfm_warm']['median_of_scenario_median_seeded_total_seconds'])} s versus {fmt(result['overall']['cold']['median_of_scenario_median_seeded_total_seconds'])} s cold; it was faster on {result['comparisons']['gridsfm_seeded_total_faster_than_cold_scenarios']}/25 scenarios.",
        f"- DC-warm's median seeded total was {fmt(result['overall']['dc_warm']['median_of_scenario_median_seeded_total_seconds'])} s and beat cold on {result['comparisons']['dc_seeded_total_faster_than_cold_scenarios']}/25 scenarios.",
        "",
        "## Five-grid summary",
        "",
        "| Grid | Perturbations | Cold AC | GridSFM seeded AC | GridSFM total | GridSFM total speedup | DC seeded AC | DC total | DC total speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topology, by_arm in result["topologies"].items():
        cold, grid, dc = (by_arm[arm] for arm in ARMS)
        lines.append(
            f"| {topology.title()} | {cold['scenario_count']} | {fmt(cold['median_ac_solver_reported_seconds'])} | "
            f"{fmt(grid['median_ac_solver_reported_seconds'])} | {fmt(grid['median_seeded_total_seconds'])} | "
            f"{fmt(grid['median_seeded_total_speedup_vs_cold'], 2)}× | {fmt(dc['median_ac_solver_reported_seconds'])} | "
            f"{fmt(dc['median_seeded_total_seconds'])} | {fmt(dc['median_seeded_total_speedup_vs_cold'], 2)}× |"
        )
    lines.extend([
        "",
        "## Median timing by scenario",
        "",
        "| Scenario | Cold AC solve | GridSFM presolve | GridSFM seeded AC | GridSFM total | GridSFM speedup | DC presolve | DC seeded AC | DC total | DC speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for sample_id, scenario in result["scenarios"].items():
        if not all(arm in scenario for arm in ARMS):
            continue
        cold, grid, dc = (scenario[arm] for arm in ARMS)
        lines.append(
            f"| `{sample_id}` | {fmt(cold['median_ac_solver_reported_seconds'])} | "
            f"{fmt(grid['median_seed_preparation_seconds'])} | {fmt(grid['median_ac_solver_reported_seconds'])} | "
            f"{fmt(grid['median_seeded_total_seconds'])} | {fmt(grid['median_seeded_total_speedup_vs_cold'], 2)}× | "
            f"{fmt(dc['median_seed_preparation_seconds'])} | {fmt(dc['median_ac_solver_reported_seconds'])} | "
            f"{fmt(dc['median_seeded_total_seconds'])} | {fmt(dc['median_seeded_total_speedup_vs_cold'], 2)}× |"
        )
    lines.extend([
        "",
        "Times are seconds. Seeded total is seed preparation plus the solver-reported seeded AC time. GridSFM presolve uses the saved resident-request measurement plus current ID mapping; this batch does not re-run neural inference inside Julia. DC presolve is measured directly and includes DC model construction, solve, extraction, and mapping.",
        "",
        "## Overall median of scenario medians",
        "",
        "| Arm | AC solver only | AC Ipopt iterations | AC-only speedup | Seeded total | Workflow E2E | Total speedup | All converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        value = result["overall"][arm]
        lines.append(
            f"| `{arm}` | {fmt(value['median_of_scenario_median_ac_solver_reported_seconds'])} s | "
            f"{fmt(value['median_of_scenario_median_ac_ipopt_iterations'], 1)} | "
            f"{fmt(value['median_of_scenario_median_ac_only_speedup_vs_cold'], 2)}× | "
            f"{fmt(value['median_of_scenario_median_seeded_total_seconds'])} s | "
            f"{fmt(value['median_of_scenario_median_workflow_e2e_seconds'])} s | "
            f"{fmt(value['median_of_scenario_median_speedup_vs_cold'], 2)}× | {value['all_converged']} |"
        )
    lines.extend([
        "",
        "## Correctness coverage",
        "",
        "Every warm result is compared with its matched cold solve for objective, bus angle, voltage magnitude, generator active/reactive output, and all four branch-flow channels (`pf`, `qf`, `pt`, `qt`) on both AC lines and transformers. The per-run JSON retains every signed per-component difference; `results.csv` contains the aggregate MAE/RMSE/maximum values.",
        "",
        "| Worst observed warm-versus-cold difference across all runs | Value |",
        "|---|---:|",
        f"| Objective | {result['correctness']['maximum_objective_percent_difference_vs_matched_cold']:.6g}% |",
        f"| Bus angle | {result['correctness']['maximum_bus_theta_degrees_absolute_difference']:.6g}° |",
        f"| Voltage magnitude | {result['correctness']['maximum_bus_vm_pu_absolute_difference']:.6g} p.u. |",
        f"| Generator `Pg` | {result['correctness']['maximum_generator_pg_pu_absolute_difference']:.6g} p.u. |",
        f"| Generator `Qg` | {result['correctness']['maximum_generator_qg_pu_absolute_difference']:.6g} p.u. |",
        f"| Branch active flow (`pf`/`pt`) | {result['correctness']['maximum_all_branch_active_flows_pu_absolute_difference']:.6g} p.u. |",
        f"| Branch reactive flow (`qf`/`qt`) | {result['correctness']['maximum_all_branch_reactive_flows_pu_absolute_difference']:.6g} p.u. |",
        "",
        "No objective acceptance range was used to hide or discard a result; raw differences and failures remain visible.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts_seed42_5x5/warmstart_5x5")
    config, rows = load_rows(output_dir)
    result = summarize(config, rows)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if rows:
        with (output_dir / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.md").write_text(markdown(config, result))
    print(f"summarized {result['completed_scenarios']}/{result['expected_scenarios']} scenarios")


if __name__ == "__main__":
    main()
