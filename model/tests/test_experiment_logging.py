from __future__ import annotations

import json

import pytest

from gridsfm.experiment_logging import (
    JSONLLogger,
    cosine_learning_rate,
    write_learning_rate_png,
    write_train_validation_loss_png,
)


def test_cosine_learning_rate_includes_endpoints_and_is_monotone():
    values = [cosine_learning_rate(i, 11, 1e-3, 1e-5) for i in range(11)]
    assert values[0] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-5)
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_jsonl_logger_flushes_structured_events(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with JSONLLogger(path) as logger:
        logger.log("train_step", optimizer_step=1, loss=2.5, learning_rate=1e-3)
        record = json.loads(path.read_text().strip())
        assert record == {
            "event": "train_step",
            "learning_rate": 1e-3,
            "loss": 2.5,
            "optimizer_step": 1,
        }


def test_jsonl_logger_serializes_non_finite_metrics_as_null(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with JSONLLogger(path) as logger:
        logger.log("train_epoch", loss=float("nan"), metrics={"x": float("inf")})
    assert json.loads(path.read_text()) == {
        "event": "train_epoch",
        "loss": None,
        "metrics": {"x": None},
    }


def test_learning_rate_png_is_valid_raster_image(tmp_path):
    path = tmp_path / "learning_rate.png"
    write_learning_rate_png(
        [(1, 1e-3), (2, 5e-4), (3, 1e-5)], path, width=640, height=360,
    )
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_train_validation_loss_png_is_valid_raster_image(tmp_path):
    path = tmp_path / "train_validation_loss.png"
    write_train_validation_loss_png(
        [(1, 8.0), (2, 7.0)],
        [(1, 8.5), (2, 7.5)],
        path,
        width=640,
        height=360,
    )
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
