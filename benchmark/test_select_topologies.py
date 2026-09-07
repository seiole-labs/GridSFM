from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.select_topologies import build_selection, discover_eligible


class SelectTopologiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.data_dir = self.root / "16h"
        self.data_dir.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_model(self, name: str, n_buses: int) -> None:
        model = {
            "bus": {str(i): {} for i in range(n_buses)},
            "gen": {"1": {"gen_status": 1}},
            "load": {"1": {}},
            "branch": {
                "1": {"tap": 1.0, "shift": 0.0},
                "2": {"transformer": True},
            },
        }
        (self.data_dir / f"{name}_model.json").write_text(json.dumps(model))

    def write_pool(self, names: list[str]) -> Path:
        path = self.root / "topology_pool.json"
        path.write_text(json.dumps({"topology_ids": names}))
        return path

    def test_filters_bus_range_inclusively(self) -> None:
        self.write_model("too_small", 499)
        self.write_model("minimum", 500)
        self.write_model("maximum", 4661)
        self.write_model("too_large", 4662)

        eligible = discover_eligible(self.data_dir, repo_root=self.root)

        self.assertEqual([item["name"] for item in eligible], ["maximum", "minimum"])
        self.assertEqual(eligible[0]["n_transformers"], 1)

    def test_selection_is_reproducible(self) -> None:
        for index in range(5):
            self.write_model(f"case_{index}", 500 + index)
        pool = self.write_pool([f"case_{index}" for index in range(5)])

        first = build_selection(
            self.data_dir, seed=123, count=2, repo_root=self.root, pool_manifest=pool
        )
        second = build_selection(
            self.data_dir, seed=123, count=2, repo_root=self.root, pool_manifest=pool
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["selected"]), 2)
        self.assertEqual(len({item["name"] for item in first["selected"]}), 2)


if __name__ == "__main__":
    unittest.main()
