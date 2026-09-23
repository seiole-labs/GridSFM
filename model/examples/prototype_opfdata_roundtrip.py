"""PROTOTYPE: reconstruct one raw OPFData example as a PowerModels network.

Question answered by this prototype:
Can a self-contained OPFData JSON example be reconstructed, independently
re-solved with PowerModels/Ipopt, and compared with its stored AC-OPF labels?

This is intentionally small and lives beside the exploratory notebook.  It is
not part of the public ``gridsfm`` package.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable


def manifest_ids(n: int = 1_000) -> tuple[int, ...]:
    """Return an explicit, stable prefix manifest for the prototype."""
    return tuple(range(int(n)))


def manifest_sha256(ids: Iterable[int]) -> str:
    payload = json.dumps(list(ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def raw_example_path(root: Path, case_name: str, example_id: int) -> Path:
    """Locate a raw example extracted by PyG's OPFDataset downloader."""
    return (
        Path(root)
        / "dataset_release_1"
        / case_name
        / "raw"
        / "gridopt-dataset-tmp"
        / "dataset_release_1"
        / case_name
        / "group_0"
        / f"example_{int(example_id)}.json"
    )


def load_raw_example(path: Path) -> Dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _scalar(value: Any) -> float:
    while isinstance(value, list):
        value = value[0]
    return float(value)


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def opfdata_to_powermodels(
    sample: Dict[str, Any],
    *,
    initialization: str = "flat",
) -> Dict[str, Any]:
    """Convert one raw OPFData sample to PowerModels' single-network JSON.

    ``initialization='flat'`` does not use the stored optimum as a warm start.
    ``initialization='ground_truth'`` is useful as a diagnostic when the cold
    solve reaches a different local optimum.
    """
    if initialization not in {"flat", "ground_truth"}:
        raise ValueError("initialization must be 'flat' or 'ground_truth'")

    grid = sample["grid"]
    truth = sample["solution"]
    nodes = grid["nodes"]
    edges = grid["edges"]
    truth_nodes = truth["nodes"]
    base_mva = _scalar(grid["context"])

    bus: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(nodes["bus"]):
        idx = row_idx + 1
        base_kv, bus_type, vmin, vmax = map(float, row)
        if initialization == "ground_truth":
            va0, vm0 = map(float, truth_nodes["bus"][row_idx])
        else:
            va0, vm0 = 0.0, _clip(1.0, vmin, vmax)
        bus[str(idx)] = {
            "index": idx,
            "bus_i": idx,
            "bus_type": int(bus_type),
            "area": 1,
            "zone": 1,
            "base_kv": base_kv,
            "vmin": vmin,
            "vmax": vmax,
            "va": va0,
            "vm": vm0,
            "va_start": va0,
            "vm_start": vm0,
        }

    generator_bus = edges["generator_link"]["receivers"]
    gen: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(nodes["generator"]):
        idx = row_idx + 1
        (
            mbase,
            pg_initial,
            pmin,
            pmax,
            qg_initial,
            qmin,
            qmax,
            vg,
            c2,
            c1,
            c0,
        ) = map(float, row)
        if initialization == "ground_truth":
            pg0, qg0 = map(float, truth_nodes["generator"][row_idx])
        else:
            pg0 = _clip(pg_initial, pmin, pmax)
            qg0 = _clip(qg_initial, qmin, qmax)
        gen[str(idx)] = {
            "index": idx,
            "gen_bus": int(generator_bus[row_idx]) + 1,
            "gen_status": 1,
            "mbase": mbase,
            "pg": pg0,
            "qg": qg0,
            "pg_start": pg0,
            "qg_start": qg0,
            "pmin": pmin,
            "pmax": pmax,
            "qmin": qmin,
            "qmax": qmax,
            "vg": vg,
            "model": 2,
            "startup": 0.0,
            "shutdown": 0.0,
            "ncost": 3,
            "cost": [c2, c1, c0],
        }

    load_bus = edges["load_link"]["receivers"]
    load: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(nodes["load"]):
        idx = row_idx + 1
        pd, qd = map(float, row)
        load[str(idx)] = {
            "index": idx,
            "load_bus": int(load_bus[row_idx]) + 1,
            "status": 1,
            "pd": pd,
            "qd": qd,
        }

    shunt_bus = edges["shunt_link"]["receivers"]
    shunt: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(nodes["shunt"]):
        idx = row_idx + 1
        bs, gs = map(float, row)
        shunt[str(idx)] = {
            "index": idx,
            "shunt_bus": int(shunt_bus[row_idx]) + 1,
            "status": 1,
            "bs": bs,
            "gs": gs,
        }

    branch: Dict[str, Dict[str, Any]] = {}

    def add_branch(
        sender: int,
        receiver: int,
        *,
        angmin: float,
        angmax: float,
        br_r: float,
        br_x: float,
        rate_a: float,
        rate_b: float,
        rate_c: float,
        tap: float,
        shift: float,
        b_fr: float,
        b_to: float,
        transformer: bool,
    ) -> None:
        idx = len(branch) + 1
        branch[str(idx)] = {
            "index": idx,
            "f_bus": int(sender) + 1,
            "t_bus": int(receiver) + 1,
            "br_status": 1,
            "br_r": float(br_r),
            "br_x": float(br_x),
            "g_fr": 0.0,
            "g_to": 0.0,
            "b_fr": float(b_fr),
            "b_to": float(b_to),
            "rate_a": float(rate_a),
            "rate_b": float(rate_b),
            "rate_c": float(rate_c),
            "tap": float(tap),
            "shift": float(shift),
            "angmin": float(angmin),
            "angmax": float(angmax),
            "transformer": bool(transformer),
        }

    ac = edges["ac_line"]
    for sender, receiver, row in zip(ac["senders"], ac["receivers"], ac["features"]):
        angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c = row
        add_branch(
            sender,
            receiver,
            angmin=angmin,
            angmax=angmax,
            br_r=br_r,
            br_x=br_x,
            rate_a=rate_a,
            rate_b=rate_b,
            rate_c=rate_c,
            tap=1.0,
            shift=0.0,
            b_fr=b_fr,
            b_to=b_to,
            transformer=False,
        )

    transformer = edges["transformer"]
    for sender, receiver, row in zip(
        transformer["senders"],
        transformer["receivers"],
        transformer["features"],
    ):
        (
            angmin,
            angmax,
            br_r,
            br_x,
            rate_a,
            rate_b,
            rate_c,
            tap,
            shift,
            b_fr,
            b_to,
        ) = row
        add_branch(
            sender,
            receiver,
            angmin=angmin,
            angmax=angmax,
            br_r=br_r,
            br_x=br_x,
            rate_a=rate_a,
            rate_b=rate_b,
            rate_c=rate_c,
            tap=tap,
            shift=shift,
            b_fr=b_fr,
            b_to=b_to,
            transformer=True,
        )

    return {
        "name": "opfdata_roundtrip_prototype",
        "source_type": "OPFData",
        "source_version": "dataset_release_1",
        "per_unit": True,
        "baseMVA": base_mva,
        "bus": bus,
        "gen": gen,
        "load": load,
        "shunt": shunt,
        "branch": branch,
        "dcline": {},
        "storage": {},
        "switch": {},
    }


def compare_with_ground_truth(
    sample: Dict[str, Any],
    solver_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Return objective and per-channel errors for a PowerModels result."""
    solved = solver_result.get("solution", {})
    bus_solved = solved.get("bus", {})
    gen_solved = solved.get("gen", {})
    branch_solved = solved.get("branch", {})
    truth = sample["solution"]

    def stats(values: list[float]) -> Dict[str, float]:
        finite = [abs(float(v)) for v in values if math.isfinite(float(v))]
        if not finite:
            return {"mae": float("nan"), "max_abs": float("nan")}
        return {"mae": sum(finite) / len(finite), "max_abs": max(finite)}

    va_err, vm_err = [], []
    for row_idx, (va_true, vm_true) in enumerate(truth["nodes"]["bus"]):
        got = bus_solved[str(row_idx + 1)]
        delta = float(got["va"]) - float(va_true)
        va_err.append(math.atan2(math.sin(delta), math.cos(delta)))
        vm_err.append(float(got["vm"]) - float(vm_true))

    pg_err, qg_err = [], []
    for row_idx, (pg_true, qg_true) in enumerate(truth["nodes"]["generator"]):
        got = gen_solved[str(row_idx + 1)]
        pg_err.append(float(got["pg"]) - float(pg_true))
        qg_err.append(float(got["qg"]) - float(qg_true))

    flow_err = []
    truth_edges = truth["edges"]
    branch_idx = 1
    for edge_type in ("ac_line", "transformer"):
        # OPFData stores [pt, qt, pf, qf]; PowerModels names the channels.
        for pt_true, qt_true, pf_true, qf_true in truth_edges[edge_type]["features"]:
            got = branch_solved[str(branch_idx)]
            flow_err.extend(
                [
                    float(got["pt"]) - float(pt_true),
                    float(got["qt"]) - float(qt_true),
                    float(got["pf"]) - float(pf_true),
                    float(got["qf"]) - float(qf_true),
                ]
            )
            branch_idx += 1

    objective_true = float(sample["metadata"]["objective"])
    objective_solved = float(solver_result.get("objective", float("nan")))
    objective_abs = abs(objective_solved - objective_true)
    objective_rel = objective_abs / max(abs(objective_true), 1e-12)

    return {
        "termination_status": str(solver_result.get("termination_status", "UNKNOWN")),
        "objective": {
            "ground_truth": objective_true,
            "resolved": objective_solved,
            "abs_error": objective_abs,
            "relative_error": objective_rel,
        },
        "va_rad": stats(va_err),
        "vm_pu": stats(vm_err),
        "pg_pu": stats(pg_err),
        "qg_pu": stats(qg_err),
        "branch_flow_pu": stats(flow_err),
    }
