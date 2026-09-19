#!/usr/bin/env python3
"""Measure cold-container and AC-OPF latency for the six OOD grids."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PGLIB_ROOT = REPO_ROOT.parent / "pglib-opf-data"
PIPELINE_DIR = "/workspace/power_grid/US/topology_solver_pipeline"
CASES = (
    ("case6470_rte", "accepted"),
    ("case8387_pegase", "accepted"),
    ("case13659_pegase", "accepted"),
    ("case20758_epigrids", "accepted"),
    ("case24464_goc", "exploratory_roundtrip_failed"),
    ("case78484_epigrids", "exploratory_not_roundtrip_validated"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="topo_solver_pipe:latest")
    parser.add_argument("--pglib-root", type=Path, default=DEFAULT_PGLIB_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts_ood" / "acopf_fresh_container",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pglib_root = args.pglib_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.relative_to(REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id = subprocess.run(
        ["docker", "image", "ls", "--no-trunc", "--quiet", args.image],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if not image_id:
        raise RuntimeError(f"cached Docker image not found: {args.image}")

    failures = []
    for index, (sample_id, gate) in enumerate(CASES, start=1):
        source_path = pglib_root / f"pglib_opf_{sample_id}.m"
        output_path = output_dir / f"{sample_id}.pyg.json"
        timing_path = output_dir / f"{sample_id}_cold_container_timing.json"
        if output_path.is_file() and timing_path.is_file():
            existing = json.loads(timing_path.read_text())
            if existing.get("returncode") == 0:
                print(f"SKIP {index}/{len(CASES)} {sample_id}", flush=True)
                continue
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        output_in_container = (
            "/workspace/" + output_path.relative_to(REPO_ROOT).as_posix()
        )
        command = [
            "docker", "run", "--rm",
            "-v", f"{REPO_ROOT}:/workspace",
            "-v", f"{pglib_root}:/pglib:ro",
            "-w", PIPELINE_DIR,
            args.image,
            "-c",
            " ".join(
                [
                    "julia --project=. export_gridsfm_data.jl",
                    f"/pglib/{source_path.name}",
                    output_in_container,
                ]
            ),
        ]
        print(f"RUN {index}/{len(CASES)} {sample_id} gate={gate}", flush=True)
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        completed = subprocess.run(command, check=False)
        wall_seconds = time.perf_counter() - start
        timing = {
            "sample_id": sample_id,
            "gate": gate,
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
        if completed.returncode == 0:
            print(f"DONE {sample_id} cold_wall={wall_seconds:.2f}s", flush=True)
        else:
            failures.append(sample_id)
            print(
                f"FAILED {sample_id} returncode={completed.returncode} "
                f"cold_wall={wall_seconds:.2f}s",
                flush=True,
            )

    if failures:
        raise RuntimeError("OOD AC-OPF failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
