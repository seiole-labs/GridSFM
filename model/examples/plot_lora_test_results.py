#!/usr/bin/env python3
"""Render all GridSFM LoRA test metrics as one PNG dashboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SERIES = (
    ("Full base", "fulltop", "frozen_base", "#5E6A93"),
    ("Full LoRA", "fulltop", "best_lora", "#A970FF"),
    ("N-1 base", "n1", "frozen_base", "#8290C5"),
    ("N-1 LoRA", "n1", "best_lora", "#6878FF"),
)

METRICS = (
    ("Test loss", "loss", 1.0, ".3f"),
    ("Cost MAPE (%)", "cost_mape", 100.0, ".3f"),
    ("Pg MAE", "pg_mae", 1.0, ".4f"),
    ("Qg MAE", "qg_mae", 1.0, ".4f"),
    ("Voltage MAE", "V_mae", 1.0, ".4f"),
    ("Theta MAE", "theta_mae", 1.0, ".4f"),
    ("Branch P MAE", "brP_mae", 1.0, ".4f"),
    ("Branch Q MAE", "brQ_mae", 1.0, ".4f"),
    ("KCL P residual", "kcl_P_resid", 1.0, ".6f"),
    ("KCL Q residual", "kcl_Q_resid", 1.0, ".6f"),
    ("Thermal max loading", "thermal_max_loading", 1.0, ".3f"),
    ("Thermal overload edges (%)", "thermal_frac_overload", 100.0, ".3f"),
)


def load_results(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    results = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") == "test":
                results[(record["test"], record["model"])] = record["metrics"]
    expected = {(split, model) for _, split, model, _ in SERIES}
    missing = expected - results.keys()
    if missing:
        raise RuntimeError(f"missing test records: {sorted(missing)}")
    return results


def render(results: dict, destination: Path) -> None:
    from png_plotting import render_test_dashboard

    render_test_dashboard(results, destination, SERIES, METRICS)
    return
    width, height = 1600, 1220
    columns, rows = 3, 4
    outer_x, top = 52, 132
    gap_x, gap_y = 24, 28
    panel_w = (width - 2 * outer_x - (columns - 1) * gap_x) / columns
    panel_h = (height - top - 58 - (rows - 1) * gap_y) / rows
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#020510"/>',
        '<text x="52" y="47" fill="#F5F2FF" font-family="Inter, sans-serif" '
        'font-size="30" font-weight="700">GridSFM LoRA test results</text>',
        '<text x="52" y="78" fill="#8993BA" font-family="Inter, sans-serif" '
        'font-size="17">750 held-out graphs per split · lower is better except feasibility accuracy</text>',
    ]

    legend_x = 720
    for index, (label, _split, _model, color) in enumerate(SERIES):
        x = legend_x + index * 205
        elements.extend([
            f'<rect x="{x}" y="35" width="20" height="20" rx="4" fill="{color}"/>',
            f'<text x="{x + 29}" y="51" fill="#DDE2FF" '
            f'font-family="Inter, sans-serif" font-size="15">{label}</text>',
        ])

    for metric_index, (title, key, scale, value_format) in enumerate(METRICS):
        row, column = divmod(metric_index, columns)
        x = outer_x + column * (panel_w + gap_x)
        y = top + row * (panel_h + gap_y)
        chart_left, chart_right = x + 54, x + panel_w - 18
        chart_top, chart_bottom = y + 43, y + panel_h - 39
        chart_w, chart_h = chart_right - chart_left, chart_bottom - chart_top
        values = [results[(split, model)][key] * scale for _, split, model, _ in SERIES]
        y_max = max(values) * 1.23 if max(values) > 0 else 1.0

        elements.extend([
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{panel_w:.1f}" '
            f'height="{panel_h:.1f}" rx="12" fill="#070D1F" stroke="#172143"/>',
            f'<text x="{x + 18:.1f}" y="{y + 27:.1f}" fill="#F5F2FF" '
            f'font-family="Inter, sans-serif" font-size="17" font-weight="650">{title}</text>',
        ])
        for tick in range(4):
            fraction = tick / 3
            tick_y = chart_bottom - fraction * chart_h
            tick_value = fraction * y_max
            elements.extend([
                f'<line x1="{chart_left:.1f}" y1="{tick_y:.1f}" '
                f'x2="{chart_right:.1f}" y2="{tick_y:.1f}" stroke="#1A2547"/>',
                f'<text x="{chart_left - 8:.1f}" y="{tick_y + 4:.1f}" '
                f'text-anchor="end" fill="#7883AA" font-family="monospace" '
                f'font-size="11">{tick_value:{value_format}}</text>',
            ])

        slot_w = chart_w / len(SERIES)
        bar_w = min(58, slot_w * 0.62)
        for series_index, ((label, _split, _model, color), value) in enumerate(
            zip(SERIES, values)
        ):
            center = chart_left + slot_w * (series_index + 0.5)
            bar_h = value / y_max * chart_h
            bar_y = chart_bottom - bar_h
            short_label = ("FB", "FL", "NB", "NL")[series_index]
            elements.extend([
                f'<rect x="{center - bar_w / 2:.1f}" y="{bar_y:.1f}" '
                f'width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>',
                f'<text x="{center:.1f}" y="{max(chart_top + 11, bar_y - 7):.1f}" '
                f'text-anchor="middle" fill="#F5F2FF" font-family="monospace" '
                f'font-size="11">{value:{value_format}}</text>',
                f'<text x="{center:.1f}" y="{chart_bottom + 18:.1f}" text-anchor="middle" '
                f'fill="#8993BA" font-family="Inter, sans-serif" '
                f'font-size="12">{short_label}</text>',
            ])

    elements.extend([
        '<text x="52" y="1192" fill="#8993BA" font-family="Inter, sans-serif" '
        'font-size="15">Feasibility accuracy: 100% for all four evaluations. '
        'FB = Full base, FL = Full LoRA, NB = N-1 base, NL = N-1 LoRA.</text>',
        '</svg>',
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(load_results(args.metrics), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
