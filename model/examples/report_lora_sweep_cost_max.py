#!/usr/bin/env python3
"""Report mean and maximum per-graph cost errors for a completed sweep.

Runs started before eval_pass recorded maxima are backfilled from their saved
best adapters. The report runs after training so it does not contend for GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
import yaml

from gridsfm import GridSFMLoRAModel, load_model
from gridsfm.loss import _per_graph_gen_cost
from train_lora_case6470 import _make_base_dataset, _make_loader, _make_transform, _path


@torch.no_grad()
def _max_cost_errors(model, loader, device: str) -> tuple[float, float]:
    model.eval()
    max_ape = max_abs = 0.0
    seen = False
    for batch in loader:
        batch = batch.to(device)
        model(batch)
        feasible = (batch.feasible > 0).view(-1)
        if not feasible.any():
            continue
        pred = _per_graph_gen_cost(batch, batch["generator"].pred[:, 0])
        truth = _per_graph_gen_cost(batch, batch["generator"].y[:, 0])
        absolute = (pred[feasible] - truth[feasible]).abs()
        relative = absolute / truth[feasible].clamp_min(1.0)
        max_ape = max(max_ape, float(relative.max()))
        max_abs = max(max_abs, float(absolute.max()))
        seen = True
    if not seen:
        raise ValueError("test loader contained no feasible graphs")
    return max_ape, max_abs


def _backfill(run_dir: Path, metrics: dict[tuple[str, str], dict]) -> None:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    device = config["run"]["device"]
    model = GridSFMLoRAModel(load_model(_path(config["model"]["checkpoint"]), device=device))
    adapter_name = model.load_adapter(run_dir / config["logging"]["best_adapter"])
    transform = _make_transform()

    for test_name, split_config in config["data"]["tests"].items():
        missing_models = [
            name for name in ("frozen_base", "best_lora")
            if "cost_max_ape" not in metrics[(test_name, name)]
            or "cost_max_abs_error" not in metrics[(test_name, name)]
        ]
        if not missing_models:
            continue
        dataset = _make_base_dataset(config, split_config, transform=transform)
        loader = _make_loader(config, dataset, shuffle=False)
        for name in missing_models:
            if name == "frozen_base":
                model.disable_adapter()
            else:
                model.enable_adapter(adapter_name)
            max_ape, max_abs = _max_cost_errors(model, loader, device)
            metrics[(test_name, name)]["cost_max_ape"] = max_ape
            metrics[(test_name, name)]["cost_max_abs_error"] = max_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir.resolve()
    rows = []
    for rank in (2, 4, 8):
        for graphs in (10, 100, 500, 1000):
            name = f"case6470_lora_r{rank}_a{rank}_n{graphs}_e8_b4"
            log_path = sweep_dir / f"{name}.log"
            log_text = log_path.read_text(encoding="utf-8")
            paths = re.findall(r"^run directory: (.+)$", log_text, re.MULTILINE)
            if len(paths) != 1:
                raise ValueError(f"expected one run directory in {log_path}")
            run_dir = Path(paths[0])
            records = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            if not any(record.get("event") == "run_complete" for record in records):
                raise ValueError(f"run did not complete: {run_dir}")
            metrics = {
                (record["test"], record["model"]): dict(record["metrics"])
                for record in records if record.get("event") == "test"
            }
            expected = {
                (test, model)
                for test in ("fulltop", "n1")
                for model in ("frozen_base", "best_lora")
            }
            if set(metrics) != expected:
                raise ValueError(f"missing test metrics in {run_dir}")
            if any(
                "cost_max_ape" not in value or "cost_max_abs_error" not in value
                for value in metrics.values()
            ):
                print(f"Backfilling maximum cost errors for {name}", flush=True)
                _backfill(run_dir, metrics)

            per_run = {}
            for (test, model), value in sorted(metrics.items()):
                per_run[f"{test}/{model}"] = {
                    "cost_mape_pct": 100.0 * value["cost_mape"],
                    "cost_max_ape_pct": 100.0 * value["cost_max_ape"],
                    "cost_max_abs_error": value["cost_max_abs_error"],
                }
                rows.append({
                    "rank": rank,
                    "training_graphs": graphs,
                    "test": test,
                    "model": model,
                    **per_run[f"{test}/{model}"],
                })
            (run_dir / "cost_max_error.json").write_text(
                json.dumps(per_run, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    destination = sweep_dir / "cost_max_error_summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "rank", "training_graphs", "test", "model", "cost_mape_pct",
            "cost_max_ape_pct", "cost_max_abs_error",
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
