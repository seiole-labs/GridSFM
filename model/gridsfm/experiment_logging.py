"""Small, dependency-free helpers for reproducible training-run logs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, TextIO


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class JSONLLogger:
    """Append JSON records and flush each record so partial runs stay readable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None

    def __enter__(self) -> "JSONLLogger":
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def log(self, event: str, **values: Any) -> None:
        if self._file is None:
            raise RuntimeError("JSONLLogger must be used as a context manager")
        record = _json_safe({"event": event, **values})
        self._file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._file.flush()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def cosine_learning_rate(
    step: int,
    total_steps: int,
    initial_lr: float,
    final_lr: float,
) -> float:
    """Cosine decay including both configured endpoint learning rates."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")
    if initial_lr <= 0 or final_lr < 0 or final_lr > initial_lr:
        raise ValueError("require initial_lr > 0 and 0 <= final_lr <= initial_lr")
    if total_steps == 1:
        return initial_lr
    progress = step / (total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (initial_lr - final_lr) * cosine


def write_learning_rate_svg(
    points: Iterable[tuple[int, float]],
    path: str | Path,
    *,
    width: int,
    height: int,
) -> None:
    """Write a standalone SVG learning-rate curve without plotting packages."""
    values = list(points)
    if not values:
        raise ValueError("at least one learning-rate point is required")
    if width < 320 or height < 240:
        raise ValueError("plot width/height must be at least 320x240")

    margin_left, margin_right = 88, 28
    margin_top, margin_bottom = 32, 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = min(step for step, _ in values)
    x_max = max(step for step, _ in values)
    y_max = max(lr for _, lr in values)
    y_min = min(lr for _, lr in values)
    y_span = y_max - y_min

    def px(step: int) -> float:
        relative = (step - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        return margin_left + relative * plot_width

    def py(lr: float) -> float:
        relative = (lr - y_min) / y_span if y_span > 0 else 0.5
        return margin_top + (1.0 - relative) * plot_height

    polyline = " ".join(f"{px(step):.2f},{py(lr):.2f}" for step, lr in values)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#020510"/>',
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
            f'height="{plot_height}" rx="8" fill="#070D1F"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
            f'y2="{margin_top + plot_height}" stroke="#18234B"/>',
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
            f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
            'stroke="#18234B"/>',
            f'<polyline points="{polyline}" fill="none" stroke="#A970FF" '
            'stroke-width="3"/>',
            f'<text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle" '
            'fill="#8993BA" font-family="Inter, sans-serif" '
            'font-size="16">optimizer step</text>',
            f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" '
            'fill="#8993BA" font-family="Inter, sans-serif" font-size="16" '
            f'transform="rotate(-90 22 {height / 2:.1f})">learning rate</text>',
            f'<text x="{margin_left}" y="{margin_top - 10}" fill="#F5F2FF" '
            'font-family="monospace" '
            f'font-size="13">max {y_max:.6g}</text>',
            f'<text x="{margin_left}" y="{margin_top + plot_height + 22}" '
            f'fill="#8993BA" font-family="monospace" font-size="13">min {y_min:.6g}</text>',
            f'<text x="{margin_left + plot_width}" y="{margin_top + plot_height + 22}" '
            f'text-anchor="end" fill="#8993BA" font-family="monospace" '
            f'font-size="13">step {x_max}</text>',
            '</svg>',
        ]),
        encoding="utf-8",
    )


def write_train_validation_loss_svg(
    train_points: Iterable[tuple[int, float]],
    validation_points: Iterable[tuple[int, float]],
    path: str | Path,
    *,
    width: int,
    height: int,
) -> None:
    """Write an epoch-level train/validation loss plot as standalone SVG."""
    train = list(train_points)
    validation = list(validation_points)
    if not train or not validation:
        raise ValueError("train and validation loss points are required")
    if width < 320 or height < 240:
        raise ValueError("plot width/height must be at least 320x240")

    margin_left, margin_right = 88, 36
    margin_top, margin_bottom = 58, 68
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_points = train + validation
    x_min = min(epoch for epoch, _ in all_points)
    x_max = max(epoch for epoch, _ in all_points)
    raw_y_min = min(loss for _, loss in all_points)
    raw_y_max = max(loss for _, loss in all_points)
    padding = max((raw_y_max - raw_y_min) * 0.10, raw_y_max * 0.02, 1e-9)
    y_min = max(0.0, raw_y_min - padding)
    y_max = raw_y_max + padding

    def px(epoch: int) -> float:
        relative = (epoch - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        return margin_left + relative * plot_width

    def py(loss: float) -> float:
        return margin_top + (1.0 - (loss - y_min) / (y_max - y_min)) * plot_height

    def polyline(points: list[tuple[int, float]]) -> str:
        return " ".join(f"{px(epoch):.2f},{py(loss):.2f}" for epoch, loss in points)

    grid_lines = []
    tick_count = 5
    for index in range(tick_count):
        fraction = index / (tick_count - 1)
        y = margin_top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        grid_lines.extend([
            f'<line x1="{margin_left}" y1="{y:.2f}" '
            f'x2="{margin_left + plot_width}" y2="{y:.2f}" '
            'stroke="#1A2547" stroke-width="1"/>',
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            'fill="#8993BA" font-family="Inter, sans-serif" '
            f'font-size="13">{value:.3f}</text>',
        ])

    x_labels = []
    for epoch, _ in train:
        x_labels.append(
            f'<text x="{px(epoch):.2f}" y="{margin_top + plot_height + 26}" '
            'text-anchor="middle" fill="#8993BA" '
            f'font-family="Inter, sans-serif" font-size="13">{epoch}</text>'
        )

    train_markers = [
        f'<circle cx="{px(epoch):.2f}" cy="{py(loss):.2f}" r="4" '
        'fill="#A970FF"/>'
        for epoch, loss in train
    ]
    validation_markers = [
        f'<rect x="{px(epoch) - 4:.2f}" y="{py(loss) - 4:.2f}" '
        'width="8" height="8" rx="1" fill="#6878FF"/>'
        for epoch, loss in validation
    ]

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#020510"/>',
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
            f'height="{plot_height}" rx="8" fill="#070D1F"/>',
            f'<text x="{margin_left}" y="30" fill="#F5F2FF" '
            'font-family="Inter, sans-serif" font-size="21" '
            'font-weight="700">LoRA training and validation loss</text>',
            *grid_lines,
            f'<polyline points="{polyline(train)}" fill="none" stroke="#A970FF" '
            'stroke-width="3"/>',
            f'<polyline points="{polyline(validation)}" fill="none" '
            'stroke="#6878FF" stroke-width="3" stroke-dasharray="8 6"/>',
            *train_markers,
            *validation_markers,
            *x_labels,
            f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
            'fill="#8993BA" font-family="Inter, sans-serif" '
            'font-size="16">epoch</text>',
            f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" '
            'fill="#8993BA" font-family="Inter, sans-serif" font-size="16" '
            f'transform="rotate(-90 22 {height / 2:.1f})">loss</text>',
            f'<line x1="{width - 235}" y1="29" x2="{width - 205}" y2="29" '
            'stroke="#A970FF" stroke-width="3"/>',
            f'<text x="{width - 195}" y="34" fill="#F5F2FF" '
            'font-family="Inter, sans-serif" font-size="13">train</text>',
            f'<line x1="{width - 130}" y1="29" x2="{width - 100}" y2="29" '
            'stroke="#6878FF" stroke-width="3" stroke-dasharray="8 6"/>',
            f'<text x="{width - 90}" y="34" fill="#F5F2FF" '
            'font-family="Inter, sans-serif" font-size="13">validation</text>',
            '</svg>',
        ]),
        encoding="utf-8",
    )


__all__ = [
    "JSONLLogger",
    "cosine_learning_rate",
    "write_learning_rate_svg",
    "write_train_validation_loss_svg",
]
