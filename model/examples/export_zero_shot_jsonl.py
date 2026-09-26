#!/usr/bin/env python3
"""Export the case6470 zero-shot run as a self-contained JSONL record set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    distribution = json.loads((run_dir / "cost_distribution.json").read_text())

    output = run_dir / "results.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        def record(item: dict) -> None:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")

        record({"event": "run", "manifest": manifest})
        for variant, cohorts in manifest["cohorts"].items():
            for cohort, ids in cohorts.items():
                rows = distribution["per_graph"][variant][cohort]
                summary = distribution["summary"][variant][cohort]
                if len(ids) != 750 or len(rows) != 750:
                    raise RuntimeError(f"{variant}/{cohort}: expected 750 IDs and errors")
                if [row["record_id"] for row in rows] != ids:
                    raise RuntimeError(f"{variant}/{cohort}: graph ID order mismatch")
                record({
                    "event": "cohort_summary",
                    "variant": variant,
                    "cohort": cohort,
                    "n_graphs": 750,
                    "eval_metrics": metrics[variant][cohort],
                    "cost_distribution_percent": summary,
                })
                for row in rows:
                    record({"event": "graph_cost", "variant": variant,
                            "cohort": cohort, **row})
    print(output)


if __name__ == "__main__":
    main()
