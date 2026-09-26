#!/usr/bin/env python3
"""Render learning-rate and train/validation loss histories as PNG files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gridsfm.experiment_logging import (
    write_learning_rate_png,
    write_train_validation_loss_png,
)


def load_history(metrics_path: Path):
    learning_rates = []
    training_losses = []
    validation_losses = []
    with metrics_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            event = record.get("event")
            if event == "train_step":
                learning_rates.append(
                    (int(record["optimizer_step"]), float(record["learning_rate"]))
                )
            elif event == "train_epoch":
                training_losses.append((int(record["epoch"]) + 1, float(record["loss"])))
            elif event == "validation":
                validation_losses.append(
                    (int(record["epoch"]) + 1, float(record["metrics"]["loss"]))
                )
    if not learning_rates or not training_losses or not validation_losses:
        raise RuntimeError(f"incomplete training history in {metrics_path}")
    return learning_rates, training_losses, validation_losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    output_dir = args.output_dir or args.metrics.parent
    learning_rates, training_losses, validation_losses = load_history(args.metrics)
    subtitle = None
    metadata_path = args.metrics.parent / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        subtitle = (
            f"rank={metadata['rank']}, alpha={metadata['alpha']:g} · "
            f"trainable params={metadata['trainable_parameters']:,} · "
            f"training graphs={metadata['training_graphs']:,}"
        )
    learning_rate_path = output_dir / "learning_rate.png"
    loss_path = output_dir / "train_validation_loss.png"
    write_learning_rate_png(
        learning_rates,
        learning_rate_path,
        width=args.width,
        height=args.height,
        subtitle=subtitle,
    )
    write_train_validation_loss_png(
        training_losses,
        validation_losses,
        loss_path,
        width=args.width,
        height=args.height,
        subtitle=subtitle,
    )
    print(learning_rate_path)
    print(loss_path)


if __name__ == "__main__":
    main()
