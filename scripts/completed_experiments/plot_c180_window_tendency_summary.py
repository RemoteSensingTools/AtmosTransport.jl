#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(os.environ.get("OUT_DIR", "/temp1/c180_era_geos_window_tendencies"))

RUNS = ("advonly_ppm", "advdiff_ppm", "fullphysics_ppm")
TRACERS = ("co2_natural", "co2_fossil")
DIAGNOSTICS = ("column_mean", "surface", "lower_850hPa", "mid_500hPa", "upper_250hPa")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _num(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def _matrix(rows: list[dict[str, str]], dt_hours: str, field: str) -> list[list[float]]:
    labels = []
    matrix = []
    for tracer in TRACERS:
        for diagnostic in DIAGNOSTICS:
            labels.append(f"{tracer.replace('co2_', '')}\\n{diagnostic}")
            row = []
            for run in RUNS:
                hit = [
                    r for r in rows
                    if r["run_a"] == run
                    and r["run_b"] == run
                    and r["tracer"] == tracer
                    and r["diagnostic"] == diagnostic
                    and r["dt_hours"] == dt_hours
                ]
                row.append(_num(hit[0][field]) if hit else math.nan)
            matrix.append(row)
    return labels, matrix


def _plot_matched(rows: list[dict[str, str]], dt_hours: str, field: str,
                  title: str, cmap: str, vmin: float | None, vmax: float | None,
                  out_path: Path) -> None:
    ylabels, matrix = _matrix(rows, dt_hours, field)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(RUNS)), [r.replace("_ppm", "") for r in RUNS], rotation=20)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isfinite(value):
                text = f"{value:.2g}" if field != "mean_corr" else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(field)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _within_rows(rows: list[dict[str, str]], source: str,
                 run_a: str, run_b: str, dt_hours: str, field: str) -> list[float]:
    values = []
    for tracer in TRACERS:
        for diagnostic in ("column_mean", "surface"):
            hit = [
                r for r in rows
                if r["source_a"] == source
                and r["source_b"] == source
                and r["run_a"] == run_a
                and r["run_b"] == run_b
                and r["tracer"] == tracer
                and r["diagnostic"] == diagnostic
                and r["dt_hours"] == dt_hours
            ]
            values.append(_num(hit[0][field]) if hit else math.nan)
    return values


def _plot_within(rows: list[dict[str, str]], dt_hours: str, field: str,
                 title: str, cmap: str, vmin: float | None, vmax: float | None,
                 out_path: Path) -> None:
    comparisons = (
        ("ERA5", "advonly_ppm", "advdiff_ppm"),
        ("ERA5", "advdiff_ppm", "fullphysics_ppm"),
        ("GEOS", "advonly_ppm", "advdiff_ppm"),
        ("GEOS", "advdiff_ppm", "fullphysics_ppm"),
    )
    xlabels = [f"{s}\\n{a.replace('_ppm', '')}->{b.replace('_ppm', '')}"
               for s, a, b in comparisons]
    ylabels = [f"{tracer.replace('co2_', '')}\\n{diagnostic}"
               for tracer in TRACERS for diagnostic in ("column_mean", "surface")]
    matrix = []
    for y in range(len(ylabels)):
        matrix.append([
            _within_rows(rows, s, a, b, dt_hours, field)[y]
            for s, a, b in comparisons
        ])
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=20)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isfinite(value):
                text = f"{value:.2g}" if field != "mean_corr" else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(field)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    matched = _read_csv(OUT_DIR / "matched_era_geos_window_tendency_summary.csv")
    all_rows = _read_csv(OUT_DIR / "all_sim_window_tendency_summary.csv")
    for dt_hours in ("6", "24"):
        _plot_matched(
            matched, dt_hours, "mean_corr",
            f"Matched ERA5 vs GEOS mean dCO2/dt correlation ({dt_hours} h)",
            "viridis", 0.0, 1.0,
            OUT_DIR / f"matched_era_geos_mean_corr_{dt_hours}h.png",
        )
        _plot_matched(
            matched, dt_hours, "mean_rmse_ppm_hr",
            f"Matched ERA5 vs GEOS mean dCO2/dt RMSE ({dt_hours} h, ppm/hr)",
            "magma", 0.0, None,
            OUT_DIR / f"matched_era_geos_mean_rmse_{dt_hours}h.png",
        )
        _plot_within(
            all_rows, dt_hours, "mean_corr",
            f"Within-path mean dCO2/dt correlation ({dt_hours} h)",
            "viridis", 0.0, 1.0,
            OUT_DIR / f"within_path_mean_corr_{dt_hours}h.png",
        )
        _plot_within(
            all_rows, dt_hours, "mean_rmse_ppm_hr",
            f"Within-path mean dCO2/dt RMSE ({dt_hours} h, ppm/hr)",
            "magma", 0.0, None,
            OUT_DIR / f"within_path_mean_rmse_{dt_hours}h.png",
        )


if __name__ == "__main__":
    main()
