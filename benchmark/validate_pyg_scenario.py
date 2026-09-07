#!/usr/bin/env python3
"""Validate a solved GridSFM .pyg.json scenario and print its dimensions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SOLVED_STATUSES = {
    "LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED", "OPTIMAL", "ALMOST_OPTIMAL"
}


def _matrix(value: Any, name: str, width: int) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{name}[{index}] must contain {width} values")
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in row):
            raise ValueError(f"{name}[{index}] contains a non-finite value")
    return value


def _edge_count(edge: Any, name: str, feature_width: int) -> int:
    if not isinstance(edge, dict):
        raise ValueError(f"{name} must be an object")
    senders = edge.get("senders")
    receivers = edge.get("receivers")
    if not isinstance(senders, list) or not isinstance(receivers, list):
        raise ValueError(f"{name} senders/receivers must be arrays")
    features = _matrix(edge.get("features"), f"{name}.features", feature_width)
    if len(senders) != len(receivers) or len(senders) != len(features):
        raise ValueError(f"{name} sender/receiver/feature counts differ")
    return len(features)


def validate_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    grid = scenario.get("grid")
    solution = scenario.get("solution")
    metadata = scenario.get("metadata")
    if not isinstance(grid, dict) or not isinstance(solution, dict) or not isinstance(metadata, dict):
        raise ValueError("scenario requires grid, solution, and metadata objects")

    status = str(metadata.get("termination_status", "")).upper()
    if status not in SOLVED_STATUSES:
        raise ValueError(f"scenario is not solved: termination_status={status!r}")
    objective = metadata.get("objective")
    if not isinstance(objective, (int, float)) or not math.isfinite(objective):
        raise ValueError("metadata.objective must be finite")

    nodes = grid.get("nodes", {})
    solution_nodes = solution.get("nodes", {})
    buses = _matrix(nodes.get("bus"), "grid.nodes.bus", 4)
    generators = _matrix(nodes.get("generator"), "grid.nodes.generator", 11)
    bus_solution = _matrix(solution_nodes.get("bus"), "solution.nodes.bus", 2)
    generator_solution = _matrix(
        solution_nodes.get("generator"), "solution.nodes.generator", 2
    )
    if not buses:
        raise ValueError("scenario has no buses")
    if len(bus_solution) != len(buses):
        raise ValueError("bus input and solution counts differ")
    if len(generator_solution) != len(generators):
        raise ValueError("generator input and solution counts differ")

    grid_edges = grid.get("edges", {})
    solution_edges = solution.get("edges", {})
    edge_counts: dict[str, int] = {}
    for edge_type, input_width in (("ac_line", 9), ("transformer", 11)):
        input_count = _edge_count(
            grid_edges.get(edge_type, {}), f"grid.edges.{edge_type}", input_width
        )
        solution_count = _edge_count(
            solution_edges.get(edge_type, {}),
            f"solution.edges.{edge_type}",
            4,
        )
        if input_count != solution_count:
            raise ValueError(f"{edge_type} input and solution counts differ")
        edge_counts[edge_type] = input_count

    solve_time = metadata.get("solve_time_seconds")
    if solve_time is not None and (
        not isinstance(solve_time, (int, float)) or solve_time < 0 or not math.isfinite(solve_time)
    ):
        raise ValueError("metadata.solve_time_seconds must be finite and non-negative")

    return {
        "termination_status": status,
        "objective": float(objective),
        "solve_time_seconds": solve_time,
        "n_buses": len(buses),
        "n_generators": len(generators),
        "n_ac_lines": edge_counts["ac_line"],
        "n_transformers": edge_counts["transformer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    with args.scenario.open(encoding="utf-8") as handle:
        summary = validate_scenario(json.load(handle))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
