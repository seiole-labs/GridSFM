#!/usr/bin/env python3
"""Print every executed GridSFM layer with runtime input/output shapes.

Examples:
    python examples/print_model_dag.py
    python examples/print_model_dag.py --full --leaf-only
    python examples/print_model_dag.py --output model_dag_case500.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gridsfm import (
    format_gridsfm_architecture,
    format_module_dag,
    load_model,
    load_pyg_json,
    prepare_for_inference,
    trace_module_dag,
)


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GridSFM once and print its executed module DAG with tensor shapes."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "gridsfm_open_v1.1.pt",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "samples" / "case500_goc.pyg.json",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu or cuda:0 (default: cpu).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print every runtime module call instead of the collapsed architecture.",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="With --full, hide composite containers and show leaf modules only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to this path instead of stdout.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.sample.is_file():
        parser.error(f"sample not found: {args.sample}")
    if args.leaf_only and not args.full:
        parser.error("--leaf-only requires --full")

    model = load_model(args.checkpoint, device=args.device)
    data = prepare_for_inference(load_pyg_json(args.sample)).to(args.device)
    with torch.inference_mode():
        trace = trace_module_dag(model, data)
    report = (
        format_module_dag(trace, leaf_only=args.leaf_only)
        if args.full
        else format_gridsfm_architecture(model, trace.output)
    )

    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(
            f"Wrote {'full runtime trace' if args.full else 'collapsed architecture'} "
            f"to {args.output.resolve()}"
        )


if __name__ == "__main__":
    main()
