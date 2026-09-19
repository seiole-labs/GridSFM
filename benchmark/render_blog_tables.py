#!/usr/bin/env python3
"""Render Markdown tables in TECHNICAL_BLOG_DRAFT.md as styled SVG assets."""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TECHNICAL_BLOG_DRAFT.md"
OUTPUT = ROOT / "blog-assets"

TOKENS = {
    "canvas": "#020510",
    "surface": "#070D1F",
    "elevated": "#0B1228",
    "border": "#18234B",
    "primary_text": "#F5F2FF",
    "muted_text": "#8993BA",
    "primary": "#A970FF",
    "positive": "#55D6B4",
    "warning": "#F3C969",
}

FILENAMES = [
    "networks-tested.svg",
    "prediction-errors.svg",
    "active-kcl-violations.svg",
    "reactive-kcl-violations.svg",
    "feasibility-output.svg",
    "latency-and-iterations.svg",
]


def clean_cell(value: str) -> tuple[str, bool]:
    value = value.strip()
    emphasized = value.startswith("**") and value.endswith("**")
    value = value.replace("**", "").replace("`", "")
    return value, emphasized


def parse_tables(markdown: str) -> list[tuple[str, list[list[tuple[str, bool]]]]]:
    lines = markdown.splitlines()
    tables = []
    heading = ""
    i = 0
    while i < len(lines):
        if lines[i].startswith("#"):
            heading = lines[i].lstrip("# ")
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-:| ]+\|$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [clean_cell(cell) for cell in lines[i].strip("|").split("|")]
                rows.append(cells)
                i += 1
            del rows[1]
            tables.append((heading, rows))
            continue
        i += 1
    return tables


def column_widths(rows: list[list[tuple[str, bool]]]) -> list[int]:
    widths = []
    for col in range(len(rows[0])):
        longest = max(len(row[col][0]) for row in rows)
        minimum = 220 if col == 0 else 125
        maximum = 330 if col == 0 else 245
        widths.append(max(minimum, min(maximum, 34 + longest * 8)))
    return widths


def render_svg(title: str, rows: list[list[tuple[str, bool]]]) -> str:
    widths = column_widths(rows)
    margin = 28
    title_height = 68
    header_height = 52
    row_height = 44
    footer = 22
    width = sum(widths) + margin * 2
    height = title_height + header_height + (len(rows) - 1) * row_height + footer + margin
    x_positions = [margin]
    for col_width in widths[:-1]:
        x_positions.append(x_positions[-1] + col_width)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        '<desc id="desc">Data table from the GridSFM and AC-OPF benchmark.</desc>',
        "<style>",
        ".title{font:700 25px Inter,Arial,sans-serif;fill:#F5F2FF}",
        ".head{font:600 14px Inter,Arial,sans-serif;fill:#F5F2FF}",
        ".cell{font:500 15px Inter,Arial,sans-serif;fill:#F5F2FF;font-variant-numeric:tabular-nums}",
        ".em{font-weight:700;fill:#55D6B4}",
        ".summary{font-weight:700;fill:#A970FF}",
        ".warning{fill:#F3C969}",
        "</style>",
        f'<rect width="{width}" height="{height}" rx="18" fill="{TOKENS["canvas"]}"/>',
        f'<rect x="{margin}" y="{title_height}" width="{sum(widths)}" height="{height-title_height-footer}" rx="10" fill="{TOKENS["surface"]}" stroke="{TOKENS["border"]}"/>',
        f'<text class="title" x="{margin}" y="40">{html.escape(title)}</text>',
        f'<path d="M {margin} {title_height+header_height} H {width-margin}" stroke="{TOKENS["border"]}"/>',
        f'<path d="M {margin+6} 54 H {margin+74}" stroke="{TOKENS["primary"]}" stroke-width="3"/>',
    ]

    for col, ((label, _), x, col_width) in enumerate(zip(rows[0], x_positions, widths)):
        anchor = "start" if col == 0 else "end"
        tx = x + 14 if col == 0 else x + col_width - 14
        out.append(f'<text class="head" text-anchor="{anchor}" x="{tx}" y="{title_height+32}">{html.escape(label)}</text>')

    for row_index, row in enumerate(rows[1:]):
        y = title_height + header_height + row_index * row_height
        if row_index % 2:
            out.append(f'<rect x="{margin+1}" y="{y}" width="{sum(widths)-2}" height="{row_height}" fill="{TOKENS["elevated"]}" opacity="0.52"/>')
        if row_index:
            out.append(f'<path d="M {margin+14} {y} H {width-margin-14}" stroke="{TOKENS["border"]}" opacity="0.8"/>')
        summary_row = any(emphasized for _, emphasized in row)
        for col, ((value, emphasized), x, col_width) in enumerate(zip(row, x_positions, widths)):
            anchor = "start" if col == 0 else "end"
            tx = x + 14 if col == 0 else x + col_width - 14
            css = "cell"
            if emphasized:
                css += " em"
            elif summary_row:
                css += " summary"
            elif col == 0 and value.endswith("*"):
                css += " warning"
            out.append(f'<text class="{css}" text-anchor="{anchor}" x="{tx}" y="{y+28}">{html.escape(value)}</text>')

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    tables = parse_tables(SOURCE.read_text())
    if len(tables) != len(FILENAMES):
        raise SystemExit(f"Expected {len(FILENAMES)} tables, found {len(tables)}")
    OUTPUT.mkdir(exist_ok=True)
    for filename, (title, rows) in zip(FILENAMES, tables):
        (OUTPUT / filename).write_text(render_svg(title, rows))
        print(OUTPUT / filename)


if __name__ == "__main__":
    main()
