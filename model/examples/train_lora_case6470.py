#!/usr/bin/env python3
"""Train and evaluate the YAML-configured case6470 FFN LoRA adapter."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from gridsfm import (
    GridSFMLoRAModel,
    LoRAConfig,
    OPFDataAdapterDataset,
    SyntheticMixedDataset,
    compute_loss,
    eval_pass,
    load_model,
)
from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_
from gridsfm.experiment_logging import (
    JSONLLogger,
    cosine_learning_rate,
    write_learning_rate_png,
    write_train_validation_loss_png,
)
from gridsfm.pe_features import (
    LaplacianFactorizationCache,
    attach_pe_features_,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing configuration value {section}.{key}")
    return mapping[key]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    if config.get("schema_version") != 1:
        raise ValueError("configuration schema_version must be 1")

    run = _required(config, "run", "config")
    device = str(_required(run, "device", "run"))
    if device not in {"cpu", "cuda"}:
        raise ValueError("run.device must be 'cpu' or 'cuda'")
    loader = _required(config, "loader", "config")
    if int(_required(loader, "batch_size", "loader")) <= 0:
        raise ValueError("loader.batch_size must be positive")
    if bool(loader.get("persistent_workers")) and int(loader.get("num_workers", 0)) == 0:
        raise ValueError("persistent_workers requires num_workers > 0")

    training = _required(config, "training", "config")
    if int(_required(training, "epochs", "training")) <= 0:
        raise ValueError("training.epochs must be positive")
    if int(training.get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    optimizer = _required(training, "optimizer", "training")
    if optimizer.get("name") != "adamw":
        raise ValueError("only training.optimizer.name='adamw' is supported")
    scheduler = _required(training, "scheduler", "training")
    if scheduler != {"name": "cosine", "interval": "optimizer_step"}:
        raise ValueError(
            "training.scheduler must select cosine decay per optimizer step"
        )
    selection = _required(training, "checkpoint_selection", "training")
    if selection != {"metric": "val_loss", "mode": "min"}:
        raise ValueError("checkpoint selection must minimize val_loss")
    return config


def _make_transform():
    cycle_cache = CycleBasisCache()
    pe_cache = LaplacianFactorizationCache()

    def transform(data):
        prepare_for_grid_transformer_(data, cache=cycle_cache)
        attach_pe_features_(data, cache=pe_cache)
        return data

    return transform


def _make_base_dataset(
    config: dict[str, Any],
    split_config: dict[str, Any],
    *,
    transform=None,
):
    data = config["data"]
    return OPFDataAdapterDataset(
        root=str(_path(data["root"])),
        case_name=data["case_name"],
        variant=split_config["variant"],
        split=split_config["split"],
        n_graphs=int(split_config["graphs"]),
        num_groups=int(data["num_groups"]),
        cache_graphs=data.get("cache_graphs"),
        transform=transform,
    )


def _make_loader(config: dict[str, Any], dataset, *, shuffle: bool) -> DataLoader:
    loader = config["loader"]
    return DataLoader(
        dataset,
        batch_size=int(loader["batch_size"]),
        shuffle=shuffle,
        num_workers=int(loader["num_workers"]),
        persistent_workers=bool(loader["persistent_workers"]),
        pin_memory=bool(loader["pin_memory"]),
        drop_last=bool(loader["drop_last"]),
    )


def _build_datasets(config: dict[str, Any]):
    transform = _make_transform()
    train_base = _make_base_dataset(config, config["data"]["train"])
    synthetic = config["data"]["synthetic_training"]
    if synthetic["enabled"]:
        train_dataset = SyntheticMixedDataset(
            train_base,
            seed=int(config["run"]["seed"]),
            infeas_prob=float(synthetic["infeasible_probability"]),
            mode_weights=tuple(float(x) for x in synthetic["mode_weights"]),
            transform=transform,
        )
    else:
        train_base.transform = transform
        train_dataset = train_base

    validation_dataset = _make_base_dataset(
        config, config["data"]["validation"], transform=transform,
    )
    train_loader = _make_loader(
        config,
        train_dataset,
        shuffle=bool(config["loader"]["shuffle_training"]),
    )
    validation_loader = _make_loader(config, validation_dataset, shuffle=False)
    return train_dataset, train_loader, validation_loader, transform


def _build_lora_model(config: dict[str, Any]) -> tuple[GridSFMLoRAModel, str]:
    base = load_model(
        _path(config["model"]["checkpoint"]),
        device=config["run"]["device"],
    )
    model = GridSFMLoRAModel(base)
    adapter = config["adapter"]
    if adapter["checkpoint"] is not None:
        name = model.load_adapter(_path(adapter["checkpoint"]))
    else:
        name = model.add_adapter(LoRAConfig(
            target=adapter["target"],
            rank=int(adapter["rank"]),
            alpha=float(adapter["alpha"]),
            dropout=float(adapter["dropout"]),
        ))
    return model, name


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _train(
    config: dict[str, Any],
    model: GridSFMLoRAModel,
    adapter_name: str,
    train_dataset,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output_dir: Path,
    logger: JSONLLogger,
) -> tuple[Path, Path]:
    training = config["training"]
    optimizer_config = training["optimizer"]
    logging_config = config["logging"]
    trainable = list(model.adapter_parameters(adapter_name))
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    plot_subtitle = (
        f"rank={int(config['adapter']['rank'])}, "
        f"alpha={float(config['adapter']['alpha']):g} · "
        f"trainable params={trainable_parameter_count:,} · "
        f"training graphs={int(config['data']['train']['graphs']):,}"
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimizer_config["initial_learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
    )
    epochs = int(training["epochs"])
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    if total_steps == 0:
        raise ValueError("training loader is empty")

    best_path = output_dir / logging_config["best_adapter"]
    final_path = output_dir / logging_config["final_adapter"]
    plot_path = output_dir / logging_config["learning_rate_png"]
    loss_plot_path = output_dir / logging_config["train_validation_loss_png"]
    log_interval = int(logging_config["log_every_optimizer_steps"])
    if log_interval <= 0:
        raise ValueError("logging.log_every_optimizer_steps must be positive")

    initial_lr = float(optimizer_config["initial_learning_rate"])
    final_lr = float(optimizer_config["final_learning_rate"])
    loss_kwargs = dict(training["loss_kwargs"])
    global_step = 0
    batch_iteration = 0
    best_val_loss = math.inf
    lr_points: list[tuple[int, float]] = []
    train_loss_points: list[tuple[int, float]] = []
    validation_loss_points: list[tuple[int, float]] = []
    started_at = time.time()
    device = config["run"]["device"]

    logger.log(
        "training_start",
        adapter=adapter_name,
        epochs=epochs,
        batches_per_epoch=len(train_loader),
        gradient_accumulation_steps=accumulation_steps,
        effective_batch_size=(
            int(config["loader"]["batch_size"]) * accumulation_steps
        ),
        planned_optimizer_steps=total_steps,
        trainable_parameters=trainable_parameter_count,
        adapter_rank=int(config["adapter"]["rank"]),
        adapter_alpha=float(config["adapter"]["alpha"]),
        train_graphs=int(config["data"]["train"]["graphs"]),
    )
    print(
        f"training {adapter_name}: epochs={epochs}, "
        f"batches_per_epoch={len(train_loader)}, "
        f"gradient_accumulation_steps={accumulation_steps}, "
        f"effective_batch_size="
        f"{int(config['loader']['batch_size']) * accumulation_steps}, "
        f"planned_steps={total_steps}",
        flush=True,
    )

    for epoch in range(epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_loss_parts: dict[str, float] = {}
        completed_steps = 0
        completed_batches = 0
        skipped_batches = 0
        epoch_started_at = time.time()
        accumulated_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            batch_iteration += 1
            if accumulated_batches == 0:
                learning_rate = cosine_learning_rate(
                    global_step,
                    total_steps,
                    initial_lr,
                    final_lr,
                )
                _set_optimizer_lr(optimizer, learning_rate)
            batch = batch.to(device)
            model(batch)
            loss, parts = compute_loss(batch, **loss_kwargs)
            if not torch.isfinite(loss):
                skipped_batches += 1
                logger.log(
                    "train_batch_skipped",
                    epoch=epoch,
                    batch_index=batch_index,
                    batch_iteration=batch_iteration,
                    reason="non_finite_loss",
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                continue

            (loss / accumulation_steps).backward()
            accumulated_batches += 1
            completed_batches += 1
            epoch_loss += float(loss.item())
            for part_name, part_value in parts.items():
                epoch_loss_parts[part_name] = (
                    epoch_loss_parts.get(part_name, 0.0) + part_value
                )
            should_step = (
                accumulated_batches == accumulation_steps
                or batch_index + 1 == len(train_loader)
            )
            if not should_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_batches = 0
            global_step += 1
            completed_steps += 1
            lr_points.append((global_step, learning_rate))

            if global_step % log_interval == 0:
                logger.log(
                    "train_step",
                    epoch=epoch,
                    batch_index=batch_index,
                    batch_iteration=batch_iteration,
                    optimizer_step=global_step,
                    loss=float(loss.item()),
                    learning_rate=learning_rate,
                    gradient_norm=float(grad_norm),
                    loss_parts=parts,
                )
                print(
                    f"epoch={epoch + 1}/{epochs} batch={batch_index + 1}/"
                    f"{len(train_loader)} step={global_step}/{total_steps} "
                    f"loss={float(loss.item()):.6f} lr={learning_rate:.8g}",
                    flush=True,
                )

        train_loss = epoch_loss / completed_batches if completed_batches else math.nan
        train_loss_points.append((epoch + 1, train_loss))
        logger.log(
            "train_epoch",
            epoch=epoch,
            optimizer_step=global_step,
            loss=train_loss,
            loss_parts={
                name: total / completed_batches
                for name, total in sorted(epoch_loss_parts.items())
            } if completed_batches else {},
            completed_batches=completed_batches,
            completed_steps=completed_steps,
            skipped_batches=skipped_batches,
            epoch_seconds=time.time() - epoch_started_at,
            elapsed_seconds=time.time() - started_at,
        )

        validation = eval_pass(
            model,
            validation_loader,
            device=device,
            loss_kwargs=loss_kwargs,
            include_loss_parts=True,
        )
        logger.log(
            "validation",
            epoch=epoch,
            optimizer_step=global_step,
            metrics=validation,
        )
        validation_loss_points.append((epoch + 1, validation["loss"]))
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} "
            f"val_loss={validation['loss']:.6f}",
            flush=True,
        )
        if math.isfinite(validation["loss"]) and validation["loss"] < best_val_loss:
            best_val_loss = validation["loss"]
            model.save_adapter(adapter_name, best_path)
            logger.log(
                "checkpoint",
                kind="best",
                epoch=epoch,
                optimizer_step=global_step,
                metric="val_loss",
                value=best_val_loss,
                path=str(best_path),
            )
            print(
                f"saved best adapter: val_loss={best_val_loss:.6f} {best_path}",
                flush=True,
            )

        write_learning_rate_png(
            lr_points,
            plot_path,
            width=int(logging_config["plot_width"]),
            height=int(logging_config["plot_height"]),
            subtitle=plot_subtitle,
        )
        write_train_validation_loss_png(
            train_loss_points,
            validation_loss_points,
            loss_plot_path,
            width=int(logging_config["plot_width"]),
            height=int(logging_config["plot_height"]),
            subtitle=plot_subtitle,
        )

    model.save_adapter(adapter_name, final_path)
    logger.log(
        "checkpoint",
        kind="final",
        epoch=epochs - 1,
        optimizer_step=global_step,
        path=str(final_path),
    )
    if not best_path.exists():
        raise RuntimeError("no finite validation loss was observed; best adapter was not saved")
    return best_path, final_path


def _evaluate_best(
    config: dict[str, Any],
    best_path: Path,
    transform,
    logger: JSONLLogger,
) -> None:
    if not config["evaluation"]["compare_base_and_adapter"]:
        return
    model = GridSFMLoRAModel(
        load_model(
            _path(config["model"]["checkpoint"]),
            device=config["run"]["device"],
        )
    )
    adapter_name = model.load_adapter(best_path)
    loss_kwargs = dict(config["training"]["loss_kwargs"])

    for test_name, split_config in config["data"]["tests"].items():
        dataset = _make_base_dataset(config, split_config, transform=transform)
        loader = _make_loader(config, dataset, shuffle=False)

        model.disable_adapter()
        base_metrics = eval_pass(
            model,
            loader,
            device=config["run"]["device"],
            loss_kwargs=loss_kwargs,
        )
        logger.log(
            "test",
            test=test_name,
            model="frozen_base",
            adapter_enabled=False,
            metrics=base_metrics,
        )

        model.enable_adapter(adapter_name)
        adapter_metrics = eval_pass(
            model,
            loader,
            device=config["run"]["device"],
            loss_kwargs=loss_kwargs,
        )
        logger.log(
            "test",
            test=test_name,
            model="best_lora",
            adapter_enabled=True,
            metrics=adapter_metrics,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("lora_case6470.yaml"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--train-graphs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.run_name is not None:
        config["run"]["name"] = args.run_name
    if args.train_graphs is not None:
        if args.train_graphs <= 0:
            raise ValueError("--train-graphs must be positive")
        config["data"]["train"]["graphs"] = args.train_graphs
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    if args.rank is not None:
        if args.rank <= 0:
            raise ValueError("--rank must be positive")
        config["adapter"]["rank"] = args.rank
    if args.alpha is not None:
        if not math.isfinite(args.alpha) or args.alpha <= 0:
            raise ValueError("--alpha must be finite and positive")
        config["adapter"]["alpha"] = args.alpha
    if args.output_root is not None:
        config["run"]["output_root"] = args.output_root
    device = config["run"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device is 'cuda', but CUDA is not available")

    seed = int(config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime(config["run"]["timestamp_format"])
    output_dir = _path(config["run"]["output_root"]) / (
        f"{timestamp}_{config['run']['name']}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if config["run"]["fail_if_output_exists"]:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; choose a new run.output_dir"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / config["logging"]["config_snapshot"]
    snapshot_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    metrics_path = output_dir / config["logging"]["metrics_jsonl"]
    print(f"run directory: {output_dir}", flush=True)
    with JSONLLogger(metrics_path) as logger:
        try:
            logger.log(
                "run_start",
                name=config["run"]["name"],
                seed=seed,
                device=device,
                started_at=started_at.isoformat(),
                config=str(config_path),
                output_dir=str(output_dir),
                data_splits={
                    "train": config["data"]["train"],
                    "validation": config["data"]["validation"],
                    "tests": config["data"]["tests"],
                },
            )
            train_dataset, train_loader, validation_loader, transform = (
                _build_datasets(config)
            )
            model, adapter_name = _build_lora_model(config)
            model_summary = model.trainable_parameter_summary()
            run_metadata = {
                "run_name": config["run"]["name"],
                "adapter": adapter_name,
                "rank": int(config["adapter"]["rank"]),
                "alpha": float(config["adapter"]["alpha"]),
                "training_graphs": int(config["data"]["train"]["graphs"]),
                "epochs": int(config["training"]["epochs"]),
                "physical_batch_size": int(config["loader"]["batch_size"]),
                "gradient_accumulation_steps": int(
                    config["training"].get("gradient_accumulation_steps", 1)
                ),
                "effective_batch_size": int(config["loader"]["batch_size"])
                * int(config["training"].get("gradient_accumulation_steps", 1)),
                "target_modules": int(model_summary["target_modules"]),
                "trainable_parameters": int(model_summary["trainable_params"]),
                "total_parameters": int(model_summary["total_params"]),
            }
            (output_dir / "run_metadata.json").write_text(
                json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            logger.log(
                "model_ready",
                adapter=adapter_name,
                rank=run_metadata["rank"],
                alpha=run_metadata["alpha"],
                training_graphs=run_metadata["training_graphs"],
                trainable_parameters=run_metadata["trainable_parameters"],
                summary=model_summary,
            )
            print(
                f"adapter={adapter_name} rank={run_metadata['rank']} "
                f"alpha={run_metadata['alpha']:g} "
                f"trainable_parameters={run_metadata['trainable_parameters']:,}",
                flush=True,
            )
            best_path, final_path = _train(
                config,
                model,
                adapter_name,
                train_dataset,
                train_loader,
                validation_loader,
                output_dir,
                logger,
            )
            del model
            _evaluate_best(config, best_path, transform, logger)
            logger.log(
                "run_complete",
                best_adapter=str(best_path),
                final_adapter=str(final_path),
            )
        except Exception as exc:
            logger.log("run_error", error_type=type(exc).__name__, message=str(exc))
            raise

    print(f"run artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
