#!/usr/bin/env python3
"""Render a GridSFM-paper-style pre/post fine-tuning comparison PNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = (
    ("Cost MAPE", "cost_mape", 100.0, ".3f", "%"),
    ("Feasibility accuracy", "feas_acc", 1.0, ".3f", ""),
    ("Pg MAE", "pg_mae", 1.0, ".4f", ""),
    ("Qg MAE", "qg_mae", 1.0, ".4f", ""),
    ("V MAE", "V_mae", 1.0, ".4f", ""),
    ("θ MAE", "theta_mae", 1.0, ".4f", ""),
)

COLUMNS = (
    ("Frozen base", "fulltop", "frozen_base"),
    ("Frozen base", "n1", "frozen_base"),
    ("LoRA", "fulltop", "best_lora"),
    ("LoRA", "n1", "best_lora"),
)


def load_results(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    results = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") == "test":
                results[(record["test"], record["model"])] = record["metrics"]
    expected = {
        ("fulltop", "frozen_base"),
        ("n1", "frozen_base"),
        ("fulltop", "best_lora"),
        ("n1", "best_lora"),
    }
    missing = expected - results.keys()
    if missing:
        raise RuntimeError(f"missing test records: {sorted(missing)}")
    return results


def render(results: dict, destination: Path) -> None:
    from png_plotting import render_paper_comparison

    render_paper_comparison(results, destination, ROWS, COLUMNS)
    return
    width, height = 1540, 760
    left = 54
    table_top = 188
    header_h = 88
    row_h = 66
    widths = (300, 230, 230, 230, 230, 210)
    column_x = [left]
    for value in widths:
        column_x.append(column_x[-1] + value)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#020510"/>',
        '<text x="54" y="48" fill="#F5F2FF" font-family="Inter, sans-serif" '
        'font-size="30" font-weight="700">case6470 LoRA test comparison</text>',
        '<text x="54" y="82" fill="#AEB7D8" font-family="Inter, sans-serif" '
        'font-size="17">Microsoft GridSFM Table 9-style pre/post adaptation layout</text>',
        '<text x="54" y="116" fill="#8993BA" font-family="Inter, sans-serif" '
        'font-size="15">FFN LoRA · best validation checkpoint · '
        'effective batch 8 · 750 graphs per test split</text>',
        f'<rect x="{left}" y="{table_top}" width="{sum(widths)}" '
        f'height="{header_h + len(ROWS) * row_h}" rx="12" fill="#070D1F" '
        'stroke="#1B2851"/>',
    ]

    headers = (
        ("Metric", ""),
        ("Frozen base", "fulltop"),
        ("Frozen base", "N-1"),
        ("Best LoRA", "fulltop"),
        ("Best LoRA", "N-1 held out"),
        ("Fulltop", "relative change"),
    )
    for index, (line1, line2) in enumerate(headers):
        x0, x1 = column_x[index], column_x[index + 1]
        anchor = "start" if index == 0 else "middle"
        text_x = x0 + 20 if index == 0 else (x0 + x1) / 2
        color = "#73E6A1" if index in (3, 4) else "#F5F2FF"
        elements.append(
            f'<text x="{text_x:.1f}" y="{table_top + 35}" text-anchor="{anchor}" '
            f'fill="{color}" font-family="Inter, sans-serif" font-size="16" '
            f'font-weight="700">{line1}</text>'
        )
        if line2:
            elements.append(
                f'<text x="{text_x:.1f}" y="{table_top + 59}" '
                f'text-anchor="{anchor}" fill="#8993BA" '
                f'font-family="Inter, sans-serif" font-size="13">{line2}</text>'
            )

    for boundary in column_x[1:-1]:
        elements.append(
            f'<line x1="{boundary}" y1="{table_top}" x2="{boundary}" '
            f'y2="{table_top + header_h + len(ROWS) * row_h}" stroke="#1B2851"/>'
        )
    elements.append(
        f'<line x1="{left}" y1="{table_top + header_h}" '
        f'x2="{left + sum(widths)}" y2="{table_top + header_h}" '
        'stroke="#2A396D" stroke-width="2"/>'
    )

    for row_index, (label, key, scale, fmt, suffix) in enumerate(ROWS):
        y0 = table_top + header_h + row_index * row_h
        center_y = y0 + 41
        if row_index % 2:
            elements.append(
                f'<rect x="{left + 1}" y="{y0}" width="{sum(widths) - 2}" '
                f'height="{row_h}" fill="#0A1127"/>'
            )
        if row_index:
            elements.append(
                f'<line x1="{left}" y1="{y0}" x2="{left + sum(widths)}" '
                'y2="{y0}" stroke="#152044"/>'.replace('y2="{y0}"', f'y2="{y0}"')
            )
        elements.append(
            f'<text x="{left + 20}" y="{center_y}" fill="#E8EBFF" '
            f'font-family="Inter, sans-serif" font-size="16">{label}</text>'
        )

        values = []
        for _title, split, model in COLUMNS:
            values.append(results[(split, model)][key] * scale)
        for value_index, value in enumerate(values):
            x0, x1 = column_x[value_index + 1], column_x[value_index + 2]
            color = "#73E6A1" if value_index >= 2 else "#CDD3ED"
            elements.append(
                f'<text x="{(x0 + x1) / 2:.1f}" y="{center_y}" '
                f'text-anchor="middle" fill="{color}" font-family="monospace" '
                f'font-size="16" font-weight="600">{values[value_index]:{fmt}}{suffix}</text>'
            )

        base, adapted = values[0], values[2]
        if key == "feas_acc":
            delta = (adapted - base) * 100
            delta_text = f"{delta:+.1f} pp"
        else:
            delta = (adapted - base) / base * 100 if base else 0.0
            delta_text = f"{delta:+.1f}%"
        color = "#73E6A1" if delta <= 0 or key == "feas_acc" else "#FF8D8D"
        x0, x1 = column_x[5], column_x[6]
        elements.append(
            f'<text x="{(x0 + x1) / 2:.1f}" y="{center_y}" '
            f'text-anchor="middle" fill="{color}" font-family="monospace" '
            f'font-size="16" font-weight="700">{delta_text}</text>'
        )

    footer_y = table_top + header_h + len(ROWS) * row_h + 42
    elements.extend([
        f'<text x="54" y="{footer_y}" fill="#8993BA" font-family="Inter, sans-serif" '
        'font-size="14">Lower is better except feasibility accuracy. N-1 remained entirely '
        'held out during LoRA training.</text>',
        f'<text x="54" y="{footer_y + 25}" fill="#8993BA" '
        'font-family="Inter, sans-serif" font-size="14">Metric note: this run logs feasibility '
        'accuracy; Microsoft GridSFM Table 9 reports feasibility F1, so those values should not '
        'be compared directly.</text>',
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
