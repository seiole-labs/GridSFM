"""OPFData adapter for GridSFM fine-tuning.

Wraps PyG's ``OPFDataset`` while building its tensor cache directly from the
compressed archive. This avoids materializing the much larger JSON payload on
disk before training.
"""
from __future__ import annotations

import io
import json
import os.path as osp
import tarfile
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data.download import download_url
from torch_geometric.datasets import OPFDataset
from torch_geometric.datasets.opf import extract_edge_index, extract_edge_index_rev
from tqdm import tqdm


def _opf_object_to_data(obj: dict) -> HeteroData:
    """Convert one OPFData JSON object without materializing it on disk."""
    grid = obj["grid"]
    solution = obj["solution"]
    metadata = obj["metadata"]

    data = HeteroData()
    data.x = torch.tensor(grid["context"]).view(-1)
    data.objective = torch.tensor(metadata["objective"])

    data["bus"].x = torch.tensor(grid["nodes"]["bus"])
    data["bus"].y = torch.tensor(solution["nodes"]["bus"])
    data["generator"].x = torch.tensor(grid["nodes"]["generator"])
    data["generator"].y = torch.tensor(solution["nodes"]["generator"])
    data["load"].x = torch.tensor(grid["nodes"]["load"])
    data["shunt"].x = torch.tensor(grid["nodes"]["shunt"])

    for edge_type in ("ac_line", "transformer"):
        store = data["bus", edge_type, "bus"]
        store.edge_index = extract_edge_index(obj, edge_type)
        store.edge_attr = torch.tensor(grid["edges"][edge_type]["features"])
        store.edge_label = torch.tensor(
            solution["edges"][edge_type]["features"]
        )

    for node_type, edge_type in (
        ("generator", "generator_link"),
        ("load", "load_link"),
        ("shunt", "shunt_link"),
    ):
        data[node_type, edge_type, "bus"].edge_index = extract_edge_index(
            obj, edge_type,
        )
        data["bus", edge_type, node_type].edge_index = extract_edge_index_rev(
            obj, edge_type,
        )
    return data


class _CachedOPFDataset(OPFDataset):
    """OPFDataset with a reusable, disk-bounded streaming cache builder."""

    def __init__(self, *args, cache_graphs=None, **kwargs):
        self.cache_graphs = (
            None
            if cache_graphs is None
            else {name: int(value) for name, value in cache_graphs.items()}
        )
        super().__init__(*args, **kwargs)

    @property
    def processed_dir(self) -> str:
        base = super().processed_dir
        if self.cache_graphs is None:
            return base
        suffix = "_".join(
            str(self.cache_graphs[name]) for name in ("train", "val", "test")
        )
        return f"{base}_subset_{suffix}"

    def download(self) -> None:
        """Download archives without expanding their large JSON payloads."""
        for name in self.raw_file_names:
            url = f"{self.url}/{self._release}/{name}"
            download_url(url, self.raw_dir)

    def process(self) -> None:
        """Stream JSON members directly into the three PyG tensor caches."""
        split_data = {"train": [], "val": [], "test": []}
        train_limit = int(15_000 * self.num_groups * 0.9)
        val_limit = train_limit + int(15_000 * self.num_groups * 0.05)
        split_starts = {"train": 0, "val": train_limit, "test": val_limit}
        full_counts = {
            "train": train_limit,
            "val": val_limit - train_limit,
            "test": 15_000 * self.num_groups - val_limit,
        }
        selected_counts = self.cache_graphs or full_counts
        if set(selected_counts) != set(full_counts):
            raise ValueError("cache_graphs must define train, val, and test")
        for split, count in selected_counts.items():
            if count < 0 or count > full_counts[split]:
                raise ValueError(
                    f"cache_graphs.{split} must be between 0 and "
                    f"{full_counts[split]}"
                )

        for archive_path in self.raw_paths:
            with tarfile.open(archive_path, mode="r:gz") as archive:
                members = (
                    member for member in archive
                    if member.isfile() and member.name.endswith(".json")
                )
                for member in tqdm(
                    members,
                    desc=f"Processing {osp.basename(archive_path)}",
                    unit="graph",
                ):
                    name = osp.basename(member.name)
                    index = int(osp.splitext(name)[0].split("_")[1])
                    if index < train_limit:
                        split = "train"
                    elif index < val_limit:
                        split = "val"
                    else:
                        split = "test"
                    if index >= split_starts[split] + selected_counts[split]:
                        continue

                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(
                            f"could not read JSON member {member.name!r}"
                        )
                    with io.TextIOWrapper(source, encoding="utf-8") as stream:
                        obj = json.load(stream)

                    data = _opf_object_to_data(obj)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    split_data[split].append((index, data))

        for split, path in zip(("train", "val", "test"), self.processed_paths):
            indexed = split_data[split]
            if len(indexed) != selected_counts[split]:
                raise RuntimeError(
                    f"archive yielded {len(indexed)} {split} graphs; "
                    f"expected {selected_counts[split]}"
                )
            indexed.sort(key=lambda item: item[0])
            self.save([data for _, data in indexed], path)

    def _download(self):
        if all(osp.exists(path) for path in self.processed_paths):
            return
        super()._download()

    def _process(self):
        if all(osp.exists(path) for path in self.processed_paths):
            return
        super()._process()


class OPFDataAdapterDataset(Dataset):
    """Yield OPFData samples adapted for GridSFM fine-tuning.

    ``n_graphs`` caps access after the underlying PyG cache is prepared. The
    cache builder streams the compressed archive into reusable tensor files,
    so the large intermediate JSON tree is never required.
    """

    def __init__(
        self,
        root: str,
        case_name: str,
        variant: str = "fulltop",
        split: str = "train",
        n_graphs: Optional[int] = None,
        num_groups: int = 1,
        cache_graphs: Optional[dict[str, int]] = None,
        transform=None,
    ):
        if variant not in ("fulltop", "n1"):
            raise ValueError(f"variant must be 'fulltop' or 'n1', got {variant!r}")
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train' | 'val' | 'test', got {split!r}"
            )
        self.case_name = case_name
        self.variant = variant
        self.split = split
        self.transform = transform
        self._inner = _CachedOPFDataset(
            root=root,
            split=split,
            case_name=case_name,
            num_groups=int(num_groups),
            topological_perturbations=(variant == "n1"),
            cache_graphs=cache_graphs,
        )
        self._n = (
            len(self._inner)
            if n_graphs is None
            else min(int(n_graphs), len(self._inner))
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> HeteroData:
        graph = self._inner[int(idx)]
        graph.feasible = torch.tensor(1, dtype=torch.long)
        if self.transform is not None:
            graph = self.transform(graph)
        return graph

    def __repr__(self) -> str:
        return (
            f"OPFDataAdapterDataset(case={self.case_name}, "
            f"variant={self.variant}, split={self.split}, "
            f"n={self._n}/{len(self._inner)})"
        )
