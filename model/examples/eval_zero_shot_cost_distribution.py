#!/usr/bin/env python3
"""Record per-graph cost MAPE for saved zero-shot case6470 cohorts."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gridsfm import load_model
from gridsfm.loss import _per_graph_gen_cost

from eval_zero_shot_case6470 import (
    CHECKPOINT,
    SelectedGraphs,
    archive_path,
    make_transform,
    read_selected,
    sha256,
)


@torch.no_grad()
def cost_errors(model, loader, ids: list[int]) -> list[dict[str, float | int]]:
    model.eval()
    out = []
    position = 0
    for batch in loader:
        batch = batch.to("cuda:0")
        model(batch)
        predicted_cost = _per_graph_gen_cost(batch, batch["generator"].pred[:, 0])
        true_cost = _per_graph_gen_cost(batch, batch["generator"].y[:, 0])
        errors = ((predicted_cost - true_cost).abs() / true_cost.clamp_min(1.0)).tolist()
        for value in errors:
            out.append({"record_id": ids[position], "cost_mape_percent": 100.0 * value})
            position += 1
    if position != len(ids):
        raise RuntimeError(f"expected {len(ids)} graph errors, got {position}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    baseline = json.loads((args.run_dir / "metrics.json").read_text())
    if sha256(CHECKPOINT) != manifest["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash changed since baseline run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(manifest["seed"])
    model = load_model(CHECKPOINT, device="cuda:0")
    summary = {}
    all_errors = {}
    for variant, cohorts in manifest["cohorts"].items():
        selected = set().union(*(set(ids) for ids in cohorts.values()))
        print(f"Loading {variant}: {len(selected)} records", flush=True)
        graphs = read_selected(archive_path(variant), selected)
        summary[variant] = {}
        all_errors[variant] = {}
        for name, ids in cohorts.items():
            dataset = SelectedGraphs(graphs, ids, make_transform())
            loader = DataLoader(dataset, batch_size=8, shuffle=False,
                                num_workers=2, persistent_workers=True)
            rows = cost_errors(model, loader, ids)
            values = np.array([row["cost_mape_percent"] for row in rows], dtype=np.float64)
            mean = float(values.mean())
            expected = baseline[variant][name]["cost_mape"] * 100.0
            # GPU scatter reductions in the model can vary slightly between
            # passes. A 0.005 percentage-point bound catches a different
            # cohort/formula without rejecting this observed repeat drift.
            if not np.isclose(mean, expected, rtol=0, atol=0.005):
                raise RuntimeError(f"{variant}/{name}: mean {mean} != eval_pass {expected}")
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
            summary[variant][name] = {
                "n_graphs": len(rows),
                "mean_percent": mean,
                "eval_pass_mean_percent": expected,
                "mean_difference_percentage_points": mean - expected,
                "min_percent": float(values.min()),
                "q1_percent": float(q1),
                "median_percent": float(median),
                "q3_percent": float(q3),
                "max_percent": float(values.max()),
                "max_record_id": rows[int(values.argmax())]["record_id"],
                "quartile_method": "numpy.quantile, linear interpolation",
            }
            all_errors[variant][name] = rows
            (args.run_dir / "cost_distribution.json").write_text(
                json.dumps({"summary": summary, "per_graph": all_errors}, indent=2) + "\n"
            )
            print(f"{variant}/{name}: mean={mean:.6f}% max={values.max():.6f}%", flush=True)
            del loader, dataset
            gc.collect()
        del graphs
        gc.collect()


if __name__ == "__main__":
    main()
