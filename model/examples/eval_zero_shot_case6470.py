#!/usr/bin/env python3
"""Evaluate frozen case6470 weights on official and random held-out cohorts.

Uses the released fine-tuning notebook's preprocessing and ``eval_pass``.
Raw record IDs are selected before conversion so the sampled cohorts have an
auditable relationship to the LoRA training prefix.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from gridsfm import eval_pass, load_model
from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_
from gridsfm.opfdata_train import _opf_object_to_data
from gridsfm.pe_features import LaplacianFactorizationCache, attach_pe_features_


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / ".cache/opfdata_lora1000"
CASE = "pglib_opf_case6470_rte"
CHECKPOINT = ROOT / "model/checkpoints/gridsfm_open_v1.1.pt"
TRAINING_IDS = set(range(1000))
VALIDATION_IDS = set(range(13500, 14250))
OFFICIAL_TEST_IDS = set(range(14250, 15000))


def archive_path(variant: str) -> Path:
    release = "dataset_release_1" if variant == "fulltop" else "dataset_release_1_nminusone"
    return DATA_ROOT / release / CASE / "raw" / f"{CASE}_0.tar.gz"


def read_selected(path: Path, ids: set[int]) -> dict[int, object]:
    found = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            record_id = int(Path(member.name).stem.split("_")[1])
            if record_id not in ids:
                continue
            if record_id in found:
                raise RuntimeError(f"duplicate record ID {record_id} in {path}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read {member.name}")
            with source:
                graph = _opf_object_to_data(json.load(source))
            graph.feasible = torch.tensor(1, dtype=torch.long)
            found[record_id] = graph
            if len(found) % 100 == 0:
                print(f"{path.parent.parent.parent.name}: loaded {len(found)}/{len(ids)}", flush=True)
            if len(found) == len(ids):
                break
    missing = ids - found.keys()
    if missing:
        raise RuntimeError(f"missing {len(missing)} record IDs in {path}: {sorted(missing)[:10]}")
    return found


class SelectedGraphs(Dataset):
    def __init__(self, graphs: dict[int, object], ids: list[int], transform):
        self.graphs = graphs
        self.ids = ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        return self.transform(self.graphs[self.ids[index]].clone())


def make_transform():
    cycle_cache = CycleBasisCache()
    pe_cache = LaplacianFactorizationCache()

    def transform(graph):
        prepare_for_grid_transformer_(graph, cache=cycle_cache)
        attach_pe_features_(graph, cache=pe_cache)
        return graph

    return transform


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fulltop's random cohort is disjoint from training, validation, and the
    # official test cohort. N-1 was never used for LoRA training/validation.
    rng_fulltop = random.Random(args.seed)
    rng_n1 = random.Random(args.seed + 1)
    random_fulltop = sorted(rng_fulltop.sample(range(1000, 13500), 750))
    random_n1 = sorted(rng_n1.sample(range(0, 14250), 750))
    official = sorted(OFFICIAL_TEST_IDS)
    assert not (set(random_fulltop) & (TRAINING_IDS | VALIDATION_IDS | OFFICIAL_TEST_IDS))
    assert not (set(random_n1) & OFFICIAL_TEST_IDS)
    cohorts = {
        "fulltop": {"official_test": official, "random_heldout": random_fulltop},
        "n1": {"official_test": official, "random_heldout": random_n1},
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "device": torch.cuda.get_device_name(0),
        "batch_size": 8,
        "num_workers": 2,
        "scoring": "released notebook: load_model -> transform -> eval_pass",
        "training_fulltop_ids": "0..999",
        "validation_fulltop_ids": "13500..14249",
        "cohorts": cohorts,
        "archives": {variant: str(archive_path(variant).relative_to(ROOT)) for variant in cohorts},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    torch.manual_seed(args.seed)
    model = load_model(CHECKPOINT, device="cuda:0")
    results = {}
    for variant, by_cohort in cohorts.items():
        print(f"Loading {variant} selected raw records", flush=True)
        selected = set(by_cohort["official_test"]) | set(by_cohort["random_heldout"])
        graphs = read_selected(archive_path(variant), selected)
        results[variant] = {}
        for name, ids in by_cohort.items():
            dataset = SelectedGraphs(graphs, ids, make_transform())
            loader = DataLoader(dataset, batch_size=8, shuffle=False,
                                num_workers=2, persistent_workers=True)
            print(f"Evaluating {variant}/{name}: {len(ids)} graphs", flush=True)
            metrics = eval_pass(model, loader)
            if metrics["n_graphs"] != 750:
                raise RuntimeError(f"{variant}/{name}: expected 750, got {metrics['n_graphs']}")
            results[variant][name] = metrics
            (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{variant}/{name}: cost_mape={metrics['cost_mape'] * 100:.6f}%", flush=True)
            del loader, dataset
            gc.collect()
        del graphs
        gc.collect()
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
