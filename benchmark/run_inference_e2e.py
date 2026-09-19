#!/usr/bin/env python3
"""Measure one complete, cold GridSFM CLI request from launch to written output.

This is intentionally an external observer.  Measuring inside run_inference.py
cannot include Python startup, imports, or checkpoint loading, so it cannot
truthfully report complete request latency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "benchmark.run_inference",
        str(args.scenario),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--run-metadata",
        str(args.run_metadata),
        "--gpu",
        str(args.gpu),
    ]

    # The request boundary begins outside the inference process.  It therefore
    # includes worker startup, imports, checkpoint loading, inference, response
    # serialization, output writing, and worker shutdown.
    request_start = time.perf_counter()
    subprocess.run(command, check=True)
    end_to_end_seconds = time.perf_counter() - request_start

    if not args.output.is_file():
        raise RuntimeError(f"inference completed without output: {args.output}")

    # Annotating the already completed response is benchmark bookkeeping and is
    # deliberately outside the measured GridSFM request boundary.
    result = json.loads(args.output.read_text())
    result.setdefault("timing", {}).update({
        "end_to_end_seconds": end_to_end_seconds,
        "end_to_end_scope": (
            "external_cold_cli_request_from_worker_launch_through_"
            "prediction_file_written_and_worker_exit"
        ),
    })
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps({
        "output": args.output.as_posix(),
        "end_to_end_seconds": end_to_end_seconds,
        "end_to_end_scope": result["timing"]["end_to_end_scope"],
    }, indent=2))


if __name__ == "__main__":
    main()
