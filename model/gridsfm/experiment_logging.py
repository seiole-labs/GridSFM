"""Small, dependency-free helpers for reproducible training-run logs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, TextIO

from PIL import Image, ImageDraw, ImageFont


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


_SANS_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_SANS_BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _font(size: int, *, bold: bool = False, mono: bool = False):
    path = _MONO_FONT if mono else (_SANS_BOLD_FONT if bold else _SANS_FONT)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _png_destination(path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise ValueError(f"PNG plot destination must end in .png: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _draw_vertical_label(
    image: Image.Image,
    text: str,
    *,
    center_y: float,
    x: int,
    font,
    color: str,
) -> None:
    box = font.getbbox(text)
    label = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8))
    ImageDraw.Draw(label).text((4 - box[0], 4 - box[1]), text, font=font, fill=color)
    label = label.rotate(90, expand=True)
    image.alpha_composite(label, (x, int(center_y - label.height / 2)))


def write_learning_rate_png(
    points: Iterable[tuple[int, float]],
    path: str | Path,
    *,
    width: int,
    height: int,
    subtitle: str | None = None,
) -> None:
    """Write a standalone PNG learning-rate curve."""
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

    image = Image.new("RGBA", (width, height), "#020510")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (margin_left, margin_top, margin_left + plot_width, margin_top + plot_height),
        radius=8,
        fill="#070D1F",
    )
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill="#18234B")
    draw.line(
        (margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height),
        fill="#18234B",
    )
    coordinates = [(px(step), py(lr)) for step, lr in values]
    if len(coordinates) == 1:
        x, y = coordinates[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#A970FF")
    else:
        draw.line(coordinates, fill="#A970FF", width=3, joint="curve")
    axis_font = _font(16)
    mono_font = _font(13, mono=True)
    draw.text((width / 2, height - 28), "optimizer step", font=axis_font, fill="#8993BA", anchor="mm")
    _draw_vertical_label(
        image, "learning rate", center_y=height / 2, x=14, font=axis_font, color="#8993BA"
    )
    draw.text((margin_left, margin_top - 8), f"max {y_max:.6g}", font=mono_font, fill="#F5F2FF", anchor="ls")
    if subtitle:
        draw.text((margin_left + 150, margin_top - 8), subtitle, font=_font(12), fill="#8993BA", anchor="ls")
    draw.text((margin_left, margin_top + plot_height + 8), f"min {y_min:.6g}", font=mono_font, fill="#8993BA", anchor="la")
    draw.text((margin_left + plot_width, margin_top + plot_height + 8), f"step {x_max}", font=mono_font, fill="#8993BA", anchor="ra")
    image.convert("RGB").save(_png_destination(path), format="PNG", optimize=True)


def write_train_validation_loss_png(
    train_points: Iterable[tuple[int, float]],
    validation_points: Iterable[tuple[int, float]],
    path: str | Path,
    *,
    width: int,
    height: int,
    subtitle: str | None = None,
) -> None:
    """Write an epoch-level train/validation loss plot as PNG."""
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

    image = Image.new("RGBA", (width, height), "#020510")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (margin_left, margin_top, margin_left + plot_width, margin_top + plot_height),
        radius=8,
        fill="#070D1F",
    )
    title_font = _font(21, bold=True)
    axis_font = _font(16)
    tick_font = _font(13)
    draw.text((margin_left, 14), "LoRA training and validation loss", font=title_font, fill="#F5F2FF")
    if subtitle:
        draw.text((margin_left, 42), subtitle, font=_font(12), fill="#8993BA")

    tick_count = 5
    for index in range(tick_count):
        fraction = index / (tick_count - 1)
        y = margin_top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        draw.line((margin_left, y, margin_left + plot_width, y), fill="#1A2547")
        draw.text((margin_left - 12, y), f"{value:.3f}", font=tick_font, fill="#8993BA", anchor="rm")

    for epoch, _ in train:
        draw.text(
            (px(epoch), margin_top + plot_height + 15),
            str(epoch),
            font=tick_font,
            fill="#8993BA",
            anchor="ma",
        )

    train_xy = [(px(epoch), py(loss)) for epoch, loss in train]
    validation_xy = [(px(epoch), py(loss)) for epoch, loss in validation]
    if len(train_xy) > 1:
        draw.line(train_xy, fill="#A970FF", width=3, joint="curve")
    if len(validation_xy) > 1:
        draw.line(validation_xy, fill="#6878FF", width=3, joint="curve")
    for x, y in train_xy:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#A970FF")
    for x, y in validation_xy:
        draw.rounded_rectangle((x - 4, y - 4, x + 4, y + 4), radius=1, fill="#6878FF")

    draw.text((width / 2, height - 24), "epoch", font=axis_font, fill="#8993BA", anchor="mm")
    _draw_vertical_label(image, "loss", center_y=height / 2, x=14, font=axis_font, color="#8993BA")
    legend_font = _font(13)
    draw.line((width - 235, 29, width - 205, 29), fill="#A970FF", width=3)
    draw.text((width - 195, 29), "train", font=legend_font, fill="#F5F2FF", anchor="lm")
    draw.line((width - 130, 29, width - 100, 29), fill="#6878FF", width=3)
    draw.text((width - 90, 29), "validation", font=legend_font, fill="#F5F2FF", anchor="lm")
    image.convert("RGB").save(_png_destination(path), format="PNG", optimize=True)


__all__ = [
    "JSONLLogger",
    "cosine_learning_rate",
    "write_learning_rate_png",
    "write_train_validation_loss_png",
]
