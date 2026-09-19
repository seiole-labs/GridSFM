#!/usr/bin/env python3
"""Run GridSFM on one solved .pyg.json and save standalone predictions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from gridsfm import batch_data_list, load_model, load_pyg_json, prepare_for_inference
from gridsfm.schema import AC_LINE_KEY, TRANSFORMER_KEY


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def serving_metadata(device: torch.device, batch_size: int) -> dict[str, Any]:
    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_model: str | None = properties.name
        gpu_memory_gb: float | None = properties.total_memory / 1024**3
    else:
        gpu_model = None
        gpu_memory_gb = None
    return {
        "cpu_model": _cpu_model(),
        "ram_gb": round(ram_bytes / 1024**3, 2),
        "gpu_model": gpu_model,
        "gpu_memory_gb": None if gpu_memory_gb is None else round(gpu_memory_gb, 2),
        "os": platform.platform(),
        "batch_size": batch_size,
    }


def run_inference(model, scenario_path: Path, device: torch.device) -> dict[str, Any]:
    resident_request_start = time.perf_counter()
    data = prepare_for_inference(load_pyg_json(scenario_path))
    batch = batch_data_list([data]).to(device)

    _synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        out = model(batch)
    _synchronize(device)
    forward_seconds = time.perf_counter() - forward_start

    bus_pred = out["bus"].pred.detach().cpu()
    gen_pred = out["generator"].pred.detach().cpu()
    flow_edge_types: list[str] = []
    flow_edge_counts: list[int] = []
    flow_rows: list[torch.Tensor] = []
    for edge_key in (AC_LINE_KEY, TRANSFORMER_KEY):
        if edge_key in out.edge_types and hasattr(out[edge_key], "edge_flow_pred"):
            flows = out[edge_key].edge_flow_pred.detach().cpu()
            flow_edge_types.append(edge_key[1])
            flow_edge_counts.append(int(flows.shape[0]))
            flow_rows.append(flows)
    all_flows = torch.cat(flow_rows, dim=0) if flow_rows else torch.zeros((0, 4))
    feasibility_logit = float(out.feas_logit.detach().cpu().item())
    resident_request_seconds = time.perf_counter() - resident_request_start

    return {
        "scenario": scenario_path.as_posix(),
        "device": str(device),
        "timing": {
            "forward_seconds": forward_seconds,
            "resident_request_seconds": resident_request_seconds,
        },
        "predictions": {
            "theta": bus_pred[:, 0].tolist(),
            "V": bus_pred[:, 1].tolist(),
            "Pg": gen_pred[:, 0].tolist(),
            "Qg": gen_pred[:, 1].tolist(),
            "Pij": all_flows[:, 0].tolist(),
            "Qij": all_flows[:, 1].tolist(),
            "Pji": all_flows[:, 2].tolist(),
            "Qji": all_flows[:, 3].tolist(),
            "flow_edge_types": flow_edge_types,
            "flow_edge_counts": flow_edge_counts,
            "feasibility_probability": float(torch.sigmoid(out.feas_logit).cpu().item()),
            "feasibility_logit": feasibility_logit,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    model = load_model(args.checkpoint, device=device)
    model.eval()
    result = run_inference(model, args.scenario, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.run_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.run_metadata.write_text(
        json.dumps(serving_metadata(device, batch_size=1), indent=2) + "\n"
    )
    print(json.dumps({
        "output": args.output.as_posix(),
        "device": str(device),
        **result["timing"],
        "feasibility_probability": result["predictions"]["feasibility_probability"],
    }, indent=2))


if __name__ == "__main__":
    main()
