"""Pillow-based PNG renderers for LoRA evaluation summaries."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _font(size: int, *, bold: bool = False, mono: bool = False):
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _check_destination(destination: Path) -> None:
    if destination.suffix.lower() != ".png":
        raise ValueError("--output must end in .png")
    destination.parent.mkdir(parents=True, exist_ok=True)


def render_test_dashboard(results: dict, destination: Path, series, metrics) -> None:
    _check_destination(destination)
    width, height = 1600, 1220
    columns, rows = 3, 4
    outer_x, top = 52, 132
    gap_x, gap_y = 24, 28
    panel_w = (width - 2 * outer_x - (columns - 1) * gap_x) / columns
    panel_h = (height - top - 58 - (rows - 1) * gap_y) / rows
    image = Image.new("RGB", (width, height), "#020510")
    draw = ImageDraw.Draw(image)
    draw.text((52, 27), "GridSFM LoRA test results", fill="#F5F2FF", font=_font(30, bold=True))
    draw.text(
        (52, 66),
        "750 held-out graphs per split · lower is better except feasibility accuracy",
        fill="#8993BA",
        font=_font(17),
    )
    legend_x = 720
    for index, (label, _split, _model, color) in enumerate(series):
        x = legend_x + index * 205
        draw.rounded_rectangle((x, 35, x + 20, 55), radius=4, fill=color)
        draw.text((x + 29, 45), label, fill="#DDE2FF", font=_font(15), anchor="lm")

    for metric_index, (title, key, scale, value_format) in enumerate(metrics):
        row, column = divmod(metric_index, columns)
        x = outer_x + column * (panel_w + gap_x)
        y = top + row * (panel_h + gap_y)
        chart_left, chart_right = x + 54, x + panel_w - 18
        chart_top, chart_bottom = y + 43, y + panel_h - 39
        chart_w, chart_h = chart_right - chart_left, chart_bottom - chart_top
        values = [results[(split, model)][key] * scale for _, split, model, _ in series]
        y_max = max(values) * 1.23 if max(values) > 0 else 1.0
        draw.rounded_rectangle(
            (x, y, x + panel_w, y + panel_h),
            radius=12,
            fill="#070D1F",
            outline="#172143",
        )
        draw.text((x + 18, y + 15), title, fill="#F5F2FF", font=_font(17, bold=True))
        for tick in range(4):
            fraction = tick / 3
            tick_y = chart_bottom - fraction * chart_h
            tick_value = fraction * y_max
            draw.line((chart_left, tick_y, chart_right, tick_y), fill="#1A2547")
            draw.text(
                (chart_left - 8, tick_y),
                format(tick_value, value_format),
                fill="#7883AA",
                font=_font(11, mono=True),
                anchor="rm",
            )
        slot_w = chart_w / len(series)
        bar_w = min(58, slot_w * 0.62)
        for series_index, ((_label, _split, _model, color), value) in enumerate(zip(series, values)):
            center = chart_left + slot_w * (series_index + 0.5)
            bar_h = value / y_max * chart_h
            bar_y = chart_bottom - bar_h
            draw.rounded_rectangle(
                (center - bar_w / 2, bar_y, center + bar_w / 2, chart_bottom),
                radius=4,
                fill=color,
            )
            draw.text(
                (center, max(chart_top + 11, bar_y - 7)),
                format(value, value_format),
                fill="#F5F2FF",
                font=_font(11, mono=True),
                anchor="ms",
            )
            draw.text(
                (center, chart_bottom + 7),
                ("FB", "FL", "NB", "NL")[series_index],
                fill="#8993BA",
                font=_font(12),
                anchor="ma",
            )
    draw.text(
        (52, 1180),
        "Feasibility accuracy: 100% for all four evaluations. FB = Full base, FL = Full LoRA, NB = N-1 base, NL = N-1 LoRA.",
        fill="#8993BA",
        font=_font(15),
    )
    image.save(destination, format="PNG", optimize=True)


def render_paper_comparison(results: dict, destination: Path, rows, columns) -> None:
    _check_destination(destination)
    width, height = 1540, 760
    left, table_top, header_h, row_h = 54, 188, 88, 66
    widths = (300, 230, 230, 230, 230, 210)
    column_x = [left]
    for value in widths:
        column_x.append(column_x[-1] + value)
    image = Image.new("RGB", (width, height), "#020510")
    draw = ImageDraw.Draw(image)
    draw.text((54, 25), "case6470 LoRA test comparison", fill="#F5F2FF", font=_font(30, bold=True))
    draw.text(
        (54, 70),
        "Microsoft GridSFM Table 9-style pre/post adaptation layout",
        fill="#AEB7D8",
        font=_font(17),
    )
    draw.text(
        (54, 108),
        "FFN LoRA · best validation checkpoint · effective batch 8 · 750 graphs per test split",
        fill="#8993BA",
        font=_font(15),
    )
    table_bottom = table_top + header_h + len(rows) * row_h
    draw.rounded_rectangle(
        (left, table_top, left + sum(widths), table_bottom),
        radius=12,
        fill="#070D1F",
        outline="#1B2851",
    )
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
        text_x = x0 + 20 if index == 0 else (x0 + x1) / 2
        anchor = "lm" if index == 0 else "mm"
        color = "#73E6A1" if index in (3, 4) else "#F5F2FF"
        draw.text((text_x, table_top + 35), line1, fill=color, font=_font(16, bold=True), anchor=anchor)
        if line2:
            draw.text((text_x, table_top + 59), line2, fill="#8993BA", font=_font(13), anchor="mm")
    for boundary in column_x[1:-1]:
        draw.line((boundary, table_top, boundary, table_bottom), fill="#1B2851")
    draw.line((left, table_top + header_h, left + sum(widths), table_top + header_h), fill="#2A396D", width=2)

    for row_index, (label, key, scale, fmt, suffix) in enumerate(rows):
        y0 = table_top + header_h + row_index * row_h
        center_y = y0 + row_h / 2
        if row_index % 2:
            draw.rectangle((left + 1, y0, left + sum(widths) - 1, y0 + row_h), fill="#0A1127")
        if row_index:
            draw.line((left, y0, left + sum(widths), y0), fill="#152044")
        draw.text((left + 20, center_y), label, fill="#E8EBFF", font=_font(16), anchor="lm")
        values = [results[(split, model)][key] * scale for _title, split, model in columns]
        for value_index, value in enumerate(values):
            x0, x1 = column_x[value_index + 1], column_x[value_index + 2]
            color = "#73E6A1" if value_index >= 2 else "#CDD3ED"
            draw.text(
                ((x0 + x1) / 2, center_y),
                f"{value:{fmt}}{suffix}",
                fill=color,
                font=_font(16, bold=True, mono=True),
                anchor="mm",
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
        draw.text(
            ((x0 + x1) / 2, center_y),
            delta_text,
            fill=color,
            font=_font(16, bold=True, mono=True),
            anchor="mm",
        )
    footer_y = table_bottom + 32
    draw.text(
        (54, footer_y),
        "Lower is better except feasibility accuracy. N-1 remained entirely held out during LoRA training.",
        fill="#8993BA",
        font=_font(14),
    )
    draw.text(
        (54, footer_y + 25),
        "Metric note: this run logs feasibility accuracy; Microsoft GridSFM Table 9 reports feasibility F1, so those values should not be compared directly.",
        fill="#8993BA",
        font=_font(14),
    )
    image.save(destination, format="PNG", optimize=True)
