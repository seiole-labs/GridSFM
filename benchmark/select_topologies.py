#!/usr/bin/env python3
"""Discover and deterministically sample eligible GridSFM base topologies."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "gridsfm_data" / "16h"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "pilot" / "selection.json"
DEFAULT_POOL_MANIFEST = Path(__file__).resolve().parent / "topology_pool.json"
MIN_BUSES = 500
MAX_BUSES = 4661


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def describe_model(path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        model = json.load(handle)

    def count(key: str) -> int:
        value = model.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected {key!r} to be an object")
        return len(value)

    filename_suffix = "_model.json"
    if not path.name.endswith(filename_suffix):
        raise ValueError(f"{path}: expected a {filename_suffix} file")

    return {
        "name": path.name[: -len(filename_suffix)],
        "path": _repo_relative(path, repo_root),
        "n_buses": count("bus"),
        "n_generators": count("gen"),
        "n_loads": count("load"),
        "n_branches": count("branch"),
        "n_transformers": sum(
            1
            for branch in model.get("branch", {}).values()
            if bool(branch.get("transformer", False))
            or abs(float(branch.get("tap", 1.0)) - 1.0) > 1e-8
            or abs(float(branch.get("shift", 0.0))) > 1e-8
        ),
    }


def discover_eligible(
    data_dir: Path,
    repo_root: Path = REPO_ROOT,
    min_buses: int = MIN_BUSES,
    max_buses: int = MAX_BUSES,
) -> list[dict[str, Any]]:
    candidates = [
        describe_model(path, repo_root)
        for path in sorted(data_dir.glob("*_model.json"))
    ]
    return [
        item for item in candidates
        if min_buses <= item["n_buses"] <= max_buses
    ]


def load_pool(manifest_path: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    names = manifest.get("topology_ids")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{manifest_path}: topology_ids must be a non-empty string list")
    if len(names) != len(set(names)):
        raise ValueError(f"{manifest_path}: topology_ids contains duplicates")
    return names


def build_selection(
    data_dir: Path,
    seed: int,
    count: int,
    repo_root: Path = REPO_ROOT,
    pool_manifest: Path = DEFAULT_POOL_MANIFEST,
) -> dict[str, Any]:
    eligible = [
        describe_model(data_dir / f"{name}_model.json", repo_root)
        for name in load_pool(pool_manifest)
    ]
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > len(eligible):
        raise ValueError(
            f"requested {count} topologies, but only {len(eligible)} are eligible"
        )
    selected = random.Random(seed).sample(eligible, count)
    return {
        "seed": seed,
        "hour": data_dir.name,
        "eligibility": {"min_buses": MIN_BUSES, "max_buses": MAX_BUSES},
        "eligible_count": len(eligible),
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-manifest", type=Path, default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    selection = build_selection(
        args.data_dir, args.seed, args.count, pool_manifest=args.pool_manifest
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")
    print(
        f"selected {args.count}/{selection['eligible_count']} eligible topologies "
        f"with seed={args.seed} -> {args.output}"
    )
    for item in selection["selected"]:
        print(f"  {item['name']}: {item['n_buses']} buses ({item['path']})")


if __name__ == "__main__":
    main()
