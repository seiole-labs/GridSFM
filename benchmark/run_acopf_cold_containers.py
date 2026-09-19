#!/usr/bin/env python3
"""Run accepted AC-OPF cases in one fresh Docker container per grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = "/workspace/power_grid/US/topology_solver_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="topo_solver_pipe:latest")
    parser.add_argument(
        "--artifact-root", type=Path, default=REPO_ROOT / "artifacts_seed42_5x5"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts_seed42_5x5" / "acopf_fresh_container",
    )
    return parser.parse_args()


def container_path(path: Path) -> str:
    resolved = path.resolve()
    relative = resolved.relative_to(REPO_ROOT)
    return f"/workspace/{relative.as_posix()}"


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id = subprocess.run(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    prediction_paths = sorted((artifact_root / "predictions").glob("*_prediction.json"))
    if not prediction_paths:
        raise RuntimeError(f"no accepted predictions found under {artifact_root}")

    for index, prediction_path in enumerate(prediction_paths, start=1):
        sample_id = prediction_path.name.removesuffix("_prediction.json")
        input_path = artifact_root / "scenarios" / f"{sample_id}_transformed.json"
        output_path = output_dir / f"{sample_id}.pyg.json"
        timing_path = output_dir / f"{sample_id}_cold_container_timing.json"
        if output_path.is_file() and timing_path.is_file():
            print(f"SKIP {index}/{len(prediction_paths)} {sample_id}", flush=True)
            continue
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        started_at = datetime.now(timezone.utc).isoformat()
        command = [
            "docker", "run", "--rm",
            "-v", f"{REPO_ROOT}:/workspace",
            "-w", PIPELINE_DIR,
            args.image,
            "-c",
            " ".join([
                "julia --project=. export_gridsfm_data.jl",
                container_path(input_path),
                container_path(output_path),
            ]),
        ]
        print(f"RUN {index}/{len(prediction_paths)} {sample_id}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(command, check=False)
        wall_seconds = time.perf_counter() - start
        timing = {
            "sample_id": sample_id,
            "docker_image": args.image,
            "docker_image_id": image_id,
            "container_policy": "fresh docker run --rm per grid",
            "started_at_utc": started_at,
            "cold_container_wall_seconds": wall_seconds,
            "cold_container_wall_scope": (
                "before docker run through container exit after output writing"
            ),
            "returncode": completed.returncode,
        }
        timing_path.write_text(json.dumps(timing, indent=2) + "\n")
        if completed.returncode != 0:
            raise RuntimeError(f"AC-OPF failed for {sample_id}: {completed.returncode}")
        print(f"DONE {sample_id} cold_wall={wall_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
