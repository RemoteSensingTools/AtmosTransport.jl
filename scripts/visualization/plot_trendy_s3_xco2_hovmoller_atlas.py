#!/usr/bin/env python3
"""Plot daily transported TRENDY XCO2 Hovmoller ensemble atlases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

from animate_trendy_s3_xco2_ensemble import (
    BACKGROUND_PPM,
    DEFAULT_INPUT,
    discover_daily_files,
    discover_models,
)


TRACERS = ("gpp", "ter", "nee")
ANOMALY_EDGES = np.arange(-4.25, 4.5, 0.5)
TITLES = {"gpp": "GPP", "ter": "TER", "nee": "NEE = GPP + TER"}
CMAPS = {"gpp": "YlGnBu_r", "ter": "YlOrRd", "nee": "RdBu_r"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--nbins", type=int, default=60,
                        help="number of equal-width sin(latitude) bands")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--tracers", default="gpp,ter,nee")
    parser.add_argument("--max-days", type=int,
                        help="testing aid: process at most this many days")
    return parser.parse_args()


def geometry(path: Path, nbins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_edges = np.linspace(-1.0, 1.0, nbins + 1)
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds["lats"][:], dtype=np.float64).ravel()
        area = np.asarray(ds["cell_area"][:], dtype=np.float64).ravel()
    indices = np.searchsorted(
        sin_edges, np.sin(np.deg2rad(lat)), side="right"
    ) - 1
    indices = np.clip(indices, 0, nbins - 1)
    area_by_bin = np.bincount(indices, weights=area, minlength=nbins)
    if np.any(area_by_bin <= 0.0):
        raise ValueError("one or more latitude bands contain no C90 cells")
    return sin_edges, indices, area_by_bin


def zonal_mean(field: np.ndarray, indices: np.ndarray,
               area: np.ndarray, area_by_bin: np.ndarray) -> np.ndarray:
    return np.bincount(
        indices, weights=field * area, minlength=len(area_by_bin)
    ) / area_by_bin


def aggregate(files_by_date: dict[str, list[Path]], models: list[str],
              dates: list[str], indices: np.ndarray,
              area: np.ndarray, area_by_bin: np.ndarray) -> dict[str, np.ndarray]:
    outputs = {
        name: np.empty((len(models), len(dates), len(area_by_bin)), dtype=np.float32)
        for name in ("gpp", "ter")
    }
    model_index = {model: index for index, model in enumerate(models)}
    for day_index, date in enumerate(dates):
        found: set[str] = set()
        for path in files_by_date[date]:
            with netCDF4.Dataset(path) as ds:
                for model in models:
                    if model in found:
                        continue
                    gpp_name = f"co2_trendy_{model}_gpp_column_mean"
                    ter_name = f"co2_trendy_{model}_ter_column_mean"
                    if gpp_name not in ds.variables or ter_name not in ds.variables:
                        continue
                    number = model_index[model]
                    for tracer, variable_name in (
                        ("gpp", gpp_name), ("ter", ter_name)
                    ):
                        values = np.ma.filled(
                            ds[variable_name][:], np.nan
                        ).astype(np.float32)
                        daily = np.mean(
                            values.reshape(values.shape[0], -1), axis=0,
                            dtype=np.float64,
                        ) * 1.0e6 - BACKGROUND_PPM
                        if not np.all(np.isfinite(daily)):
                            raise ValueError(
                                f"non-finite {variable_name} values in {path}"
                            )
                        outputs[tracer][number, day_index] = zonal_mean(
                            daily, indices, area, area_by_bin
                        )
                    found.add(model)
        missing = sorted(set(models) - found)
        if missing:
            raise KeyError(f"missing models on {date}: {missing}")
        if (day_index + 1) % 30 == 0 or day_index + 1 == len(dates):
            print(f"Aggregated {day_index + 1}/{len(dates)} days", flush=True)
    outputs["nee"] = outputs["gpp"] + outputs["ter"]
    return outputs


def limits(values: np.ndarray, tracer: str) -> tuple[tuple[float, float], float]:
    mean = np.mean(values, axis=0)
    departures = values - mean
    if tracer == "gpp":
        mean_range = (min(float(np.percentile(mean, 0.5)), -0.02), 0.0)
    elif tracer == "ter":
        mean_range = (0.0, max(float(np.percentile(mean, 99.5)), 0.02))
    else:
        bound = max(float(np.percentile(np.abs(mean), 99.5)), 0.02)
        mean_range = (-bound, bound)
    departure_bound = max(
        float(np.percentile(np.abs(departures), 99.5)), 0.01
    )
    return mean_range, departure_bound


def style_axis(axis: plt.Axes, row: int, column: int) -> None:
    latitudes = np.array([-90, -60, -30, 0, 30, 60, 90])
    axis.set_ylim(-1.0, 1.0)
    axis.set_yticks(np.sin(np.deg2rad(latitudes)))
    axis.set_yticklabels(
        [f"{latitude}°" for latitude in latitudes] if column == 0 else []
    )
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if row != 3:
        axis.set_xticklabels([])
    axis.tick_params(labelsize=6.5, length=2, pad=1)
    axis.axhline(0.0, color="0.25", linewidth=0.3, alpha=0.6)


def plot_atlas(path: Path, tracer: str, values: np.ndarray, models: list[str],
               dates: list[str], sin_edges: np.ndarray, dpi: int) -> None:
    mean = np.mean(values, axis=0)
    panels = [mean, *(values - mean)]
    labels = ["Multi-model mean", *models]
    mean_range, departure_bound = limits(values, tracer)
    mean_norm = (
        TwoSlopeNorm(vmin=mean_range[0], vcenter=0.0, vmax=mean_range[1])
        if mean_range[0] < 0.0 < mean_range[1] else None
    )
    departure_cmap = plt.get_cmap("RdBu_r", len(ANOMALY_EDGES) - 1)
    departure_norm = BoundaryNorm(
        ANOMALY_EDGES, departure_cmap.N, clip=True
    )
    date_centers = np.array(
        [datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
         for date in dates]
    )
    date_numbers = mdates.date2num(date_centers)
    date_edges = np.concatenate((
        [date_numbers[0] - 0.5],
        (date_numbers[:-1] + date_numbers[1:]) / 2.0,
        [date_numbers[-1] + 0.5],
    ))

    figure, axes = plt.subplots(4, 6, figsize=(19.2, 10.0),
                                sharex=True, sharey=True)
    figure.set_layout_engine(None)
    figure.subplots_adjust(left=0.045, right=0.992, bottom=0.105, top=0.925,
                           hspace=0.20, wspace=0.055)
    images = []
    for index, (label, panel) in enumerate(zip(labels, panels)):
        row, column = divmod(index, 6)
        axis = axes[row, column]
        options = (
            {"cmap": CMAPS[tracer], "norm": mean_norm}
            if mean_norm is not None and index == 0 else
            {"cmap": CMAPS[tracer], "vmin": mean_range[0], "vmax": mean_range[1]}
            if index == 0 else
            {"cmap": departure_cmap, "norm": departure_norm}
        )
        image = axis.pcolormesh(
            date_edges, sin_edges, panel.T, shading="flat", rasterized=True,
            **options,
        )
        axis.set_title(label, fontsize=8.2,
                       fontweight="bold" if index == 0 else None, pad=2)
        style_axis(axis, row, column)
        images.append(image)

    mean_cax = figure.add_axes((0.14, 0.052, 0.29, 0.014))
    departure_cax = figure.add_axes((0.58, 0.052, 0.29, 0.014))
    mean_bar = figure.colorbar(images[0], cax=mean_cax,
                               orientation="horizontal", extend="both")
    departure_bar = figure.colorbar(
        images[1], cax=departure_cax, orientation="horizontal", extend="both",
        boundaries=ANOMALY_EDGES, ticks=np.arange(-4.0, 4.1, 1.0),
    )
    mean_bar.set_label("Multi-model mean zonal-mean XCO₂ (ppm)", fontsize=8)
    departure_bar.set_label(
        "Model − multi-model mean zonal-mean XCO₂ (ppm)", fontsize=8
    )
    for bar in (mean_bar, departure_bar):
        bar.ax.tick_params(labelsize=7, length=2)
    figure.suptitle(
        f"TRENDYv14 S3 transported {TITLES[tracer]} XCO₂ — daily zonal means",
        fontsize=14, fontweight="bold", y=0.975,
    )
    figure.text(0.012, 0.51, "Latitude (equal-area sin(latitude) axis)",
                rotation="vertical", va="center", fontsize=9)
    figure.text(
        0.5, 0.012,
        f"23 models; daily means of 3-hourly output; "
        f"{BACKGROUND_PPM:g} ppm carrier removed from GPP and TER",
        ha="center", fontsize=8,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.nbins < 2 or args.dpi < 1:
        raise ValueError("nbins must be at least two and dpi must be positive")
    tracers = [item.strip().lower() for item in args.tracers.split(",")
               if item.strip()]
    invalid = sorted(set(tracers) - set(TRACERS))
    if invalid or not tracers:
        raise ValueError(f"invalid tracer selection: {invalid}")
    files_by_date = discover_daily_files(args.input_dir)
    dates = sorted(files_by_date)
    if args.max_days is not None:
        dates = dates[:args.max_days]
    models = discover_models(files_by_date)
    if len(models) != 23:
        raise ValueError(f"the 6x4 atlas requires 23 models; discovered {len(models)}")
    first_file = files_by_date[dates[0]][0]
    sin_edges, bin_indices, area_by_bin = geometry(first_file, args.nbins)
    with netCDF4.Dataset(first_file) as ds:
        area = np.asarray(ds["cell_area"][:], dtype=np.float64).ravel()
    values = aggregate(
        files_by_date, models, dates, bin_indices, area, area_by_bin
    )
    output_dir = args.output_dir or args.input_dir / "hovmoller" / "xco2_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)
    for tracer in tracers:
        path = output_dir / f"trendy_s3_2021_{tracer}_xco2_hovmoller_atlas.png"
        plot_atlas(path, tracer, values[tracer], models, dates, sin_edges, args.dpi)
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
