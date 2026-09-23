from __future__ import annotations

import json

import pytest

from gridsfm.experiment_logging import (
    JSONLLogger,
    cosine_learning_rate,
    write_learning_rate_svg,
    write_train_validation_loss_svg,
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


def test_learning_rate_svg_contains_curve_and_axis_labels(tmp_path):
    path = tmp_path / "learning_rate.svg"
    write_learning_rate_svg(
        [(1, 1e-3), (2, 5e-4), (3, 1e-5)], path, width=640, height=360,
    )
    svg = path.read_text()
    assert "<polyline" in svg
    assert "optimizer step" in svg
    assert "learning rate" in svg


def test_train_validation_loss_svg_contains_both_series(tmp_path):
    path = tmp_path / "train_validation_loss.svg"
    write_train_validation_loss_svg(
        [(1, 8.0), (2, 7.0)],
        [(1, 8.5), (2, 7.5)],
        path,
        width=640,
        height=360,
    )
    svg = path.read_text()
    assert "LoRA training and validation loss" in svg
    assert "train" in svg
    assert "validation" in svg
    assert "#A970FF" in svg
    assert "#6878FF" in svg
