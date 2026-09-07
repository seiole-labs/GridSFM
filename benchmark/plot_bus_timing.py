#!/usr/bin/env python3
"""Plot benchmark runtime against grid bus count.

The plot compares the two timing fields used in the benchmark reports:
``ac_opf_solve`` and ``gridsfm_end_to_end``.  Despite its historical JSON
name, the latter is the model-resident GridSFM request, not cold-process
end-to-end latency; see benchmark/TIMING.md for the exact boundaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_RESULTS = REPO_ROOT / "artifacts_seed42_5x5" / "results"
OOD_RESULTS = (
    REPO_ROOT / "artifacts_ood" / "case6470_rte" / "phase5" / "case6470_rte_result.json",
    REPO_ROOT / "artifacts_ood" / "phase5" / "case8387_pegase_result.json",
    REPO_ROOT / "artifacts_ood" / "case13659_pegase" / "phase5" / "case13659_pegase_result.json",
    REPO_ROOT / "artifacts_ood" / "case20758_epigrids" / "phase5" / "case20758_epigrids_result.json",
    REPO_ROOT / "artifacts_ood" / "case24464_goc" / "phase5" / "case24464_goc_result.json",
    REPO_ROOT / "artifacts_ood" / "case78484_epigrids" / "phase5" / "case78484_epigrids_result.json",
)
EXPLORATORY_CASES = {"case24464_goc", "case78484_epigrids"}


@dataclass(frozen=True)
class TimingPoint:
    case: str
    buses: int
    ac_opf: float
    gridsfm: float
    group: str


def load_point(path: Path, group: str) -> TimingPoint:
    with path.open(encoding="utf-8") as result_file:
        result = json.load(result_file)
    timing = result["timing_seconds"]
    return TimingPoint(
        case=path.name.removesuffix("_result.json"),
        buses=int(result["counts"]["buses"]),
        ac_opf=float(timing["ac_opf_solve"]),
        gridsfm=float(timing["gridsfm_end_to_end"]),
        group=group,
    )


def load_results() -> tuple[list[TimingPoint], list[TimingPoint], list[TimingPoint]]:
    training = [load_point(path, "training") for path in sorted(TRAINING_RESULTS.glob("*_result.json"))]
    if len(training) != 25:
        raise RuntimeError(f"Expected 25 in-distribution 5x5 results, found {len(training)}")

    missing = [str(path) for path in OOD_RESULTS if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing OOD result files: " + ", ".join(missing))
    ood = [
        load_point(path, "exploratory" if path.name.removesuffix("_result.json") in EXPLORATORY_CASES else "ood")
        for path in OOD_RESULTS
    ]
    return training, [point for point in ood if point.group == "ood"], [point for point in ood if point.group == "exploratory"]


METHODS = (
    ("ac_opf", "AC-OPF solve", "#d1495b"),
    ("gridsfm", "GridSFM request", "#0077b6"),
)


def aggregate_training_runs(
    points: list[TimingPoint],
) -> tuple[list[TimingPoint], dict[tuple[int, str], float]]:
    """Average the five perturbation runs belonging to each topology/bus count."""
    by_bus: dict[int, list[TimingPoint]] = defaultdict(list)
    for point in points:
        by_bus[point.buses].append(point)

    means: list[TimingPoint] = []
    deviations: dict[tuple[int, str], float] = {}
    for buses, runs in sorted(by_bus.items()):
        if len(runs) != 5:
            raise RuntimeError(f"Expected five in-distribution runs at {buses} buses, found {len(runs)}")
        topology = runs[0].case.split("_mixed_", maxsplit=1)[0]
        means.append(
            TimingPoint(
                case=topology,
                buses=buses,
                ac_opf=fmean(point.ac_opf for point in runs),
                gridsfm=fmean(point.gridsfm for point in runs),
                group="training_mean",
            )
        )
        for field, _label, _color in METHODS:
            deviations[(buses, field)] = stdev(getattr(point, field) for point in runs)
    return means, deviations


def scatter_group(ax: plt.Axes, points: list[TimingPoint], marker: str, *, hollow: bool = False) -> None:
    for field, _label, color in METHODS:
        ax.scatter(
            [point.buses for point in points],
            [getattr(point, field) for point in points],
            marker=marker,
            s=58 if marker != "o" else 42,
            facecolors="none" if hollow else color,
            edgecolors=color,
            linewidths=1.6 if hollow else 0.7,
            alpha=0.9,
            zorder=3,
        )


def format_number(value: float, _position: float) -> str:
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def make_plot(training: list[TimingPoint], ood: list[TimingPoint], exploratory: list[TimingPoint]) -> plt.Figure:
    training_means, training_deviations = aggregate_training_runs(training)
    accepted = sorted(training_means + ood, key=lambda point: point.buses)
    exploratory = sorted(exploratory, key=lambda point: point.buses)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, (full_ax, zoom_ax) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.7),
        gridspec_kw={"width_ratios": (1.55, 1)},
    )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.25, top=0.87, wspace=0.18)
    figure.suptitle("Runtime scaling with power-grid size", fontsize=16, fontweight="bold")

    for field, _label, color in METHODS:
        full_ax.plot(
            [point.buses for point in accepted],
            [getattr(point, field) for point in accepted],
            color=color,
            linewidth=1.8,
            alpha=0.75,
            zorder=2,
        )
        full_ax.plot(
            [accepted[-1].buses] + [point.buses for point in exploratory],
            [getattr(accepted[-1], field)] + [getattr(point, field) for point in exploratory],
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            zorder=2,
        )
        full_ax.errorbar(
            [point.buses for point in training_means],
            [getattr(point, field) for point in training_means],
            yerr=[training_deviations[(point.buses, field)] for point in training_means],
            fmt="none",
            ecolor=color,
            elinewidth=1,
            capsize=2.5,
            alpha=0.55,
            zorder=1,
        )

    scatter_group(full_ax, training_means, "o")
    scatter_group(full_ax, ood, "D")
    scatter_group(full_ax, exploratory, "s", hollow=True)
    full_ax.set(xscale="log", yscale="log", xlabel="Number of buses (log scale)", ylabel="Recorded time (seconds, log scale)")
    full_ax.set_title("Training-scale and OOD grids")
    full_ax.grid(True, which="major", color="#d9dde3", linewidth=0.8)
    full_ax.grid(True, which="minor", color="#edf0f3", linewidth=0.5)
    full_ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    full_ax.xaxis.set_major_formatter(FuncFormatter(format_number))
    full_ax.xaxis.set_minor_formatter(NullFormatter())
    full_ax.yaxis.set_major_formatter(FuncFormatter(format_number))

    annotation_offsets = {
        "case6470_rte": (5, 6),
        "case8387_pegase": (5, -14),
        "case13659_pegase": (5, 6),
        "case20758_epigrids": (5, -14),
        "case24464_goc": (5, -12),
        "case78484_epigrids": (-5, 7),
    }
    for point in ood + exploratory:
        label = point.case.removeprefix("case")
        offset = annotation_offsets[point.case]
        full_ax.annotate(
            label,
            (point.buses, point.ac_opf),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] > 0 else "top",
            fontsize=8,
            color="#3b4045",
        )

    for field, label, color in METHODS:
        zoom_ax.errorbar(
            [point.buses for point in training_means],
            [getattr(point, field) for point in training_means],
            yerr=[training_deviations[(point.buses, field)] for point in training_means],
            color=color,
            marker="o",
            markersize=6.5,
            linewidth=1.8,
            elinewidth=1.2,
            capsize=3.5,
            label=label,
            zorder=3,
        )
    zoom_ax.set(xlabel="Number of buses", ylabel="Recorded time (seconds)")
    zoom_ax.set_title("In-distribution means (5 runs each)")
    zoom_ax.grid(True, color="#e1e4e8", linewidth=0.8)
    zoom_ax.set_xlim(610, 1310)
    zoom_ax.set_ylim(0, 15.2)
    zoom_ax.set_xticks(
        [point.buses for point in training_means],
        [f"{point.buses:,}\n{point.case.title()}" for point in training_means],
        rotation=25,
        ha="right",
    )
    zoom_ax.legend(loc="upper left", frameon=True)

    method_legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor=color, label=label)
        for _field, label, color in METHODS
    ]
    group_legend = [
        Line2D([0], [0], marker="o", linestyle="none", color="#555", label="5×5 mean (±1 SD)"),
        Line2D([0], [0], marker="D", linestyle="none", color="#555", label="OOD accepted"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor="#555", label="OOD exploratory"),
    ]
    first_legend = full_ax.legend(handles=method_legend, loc="upper left", title="Timing component", frameon=True)
    full_ax.add_artist(first_legend)
    full_ax.legend(handles=group_legend, loc="lower right", title="Benchmark set", frameon=True)

    figure.text(
        0.5,
        0.025,
        "Solid segments connect accepted measurements; dashed segments continue through exploratory probes (visual guide, not a fitted scaling law).\n"
        "GridSFM request is model-resident; AC-OPF includes PowerModels/JuMP construction + Ipopt. Boundaries are not matched end-to-end.",
        ha="center",
        fontsize=9,
        color="#50555a",
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark" / "plots",
        help="Directory for PNG and SVG output (default: benchmark/plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training, ood, exploratory = load_results()
    figure = make_plot(training, ood, exploratory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        output = args.output_dir / f"bus_count_vs_runtime.{extension}"
        figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
        print(output.relative_to(REPO_ROOT))
    plt.close(figure)


if __name__ == "__main__":
    main()
