#!/usr/bin/env python3
"""Create TRENDYv14 S3 monthly GPP/TER Hovmoller ensemble atlases."""

from __future__ import annotations

import argparse
import calendar
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4
import numpy as np


DEFAULT_INPUT = Path(
    "/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/"
    "TRENDYv14/S3/C90/2021"
)
MODEL_PATTERN = re.compile(
    r"TRENDYv14_S3_(.+)_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021\.nc$"
)
KG_C_PER_KG_CO2 = 12.0 / 44.0
KG_PER_PG = 1.0e12
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def model_files(input_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for path in input_dir.glob("*.nc"):
        match = MODEL_PATTERN.match(path.name)
        if match:
            found.append((match.group(1), path))
    return sorted(found)


def aggregate_file(path: Path, nbins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_edges = np.linspace(-1.0, 1.0, nbins + 1)
    with netCDF4.Dataset(path) as ds:
        if len(ds.dimensions["time"]) != 8760:
            raise ValueError(f"incomplete time dimension in {path}")
        lat = np.asarray(ds["lats"][:], dtype=np.float64).ravel()
        area = np.asarray(ds["cell_area"][:], dtype=np.float64).ravel()
        bin_index = np.searchsorted(sin_edges, np.sin(np.deg2rad(lat)), side="right") - 1
        bin_index = np.clip(bin_index, 0, nbins - 1)
        outputs = []
        for variable, sign in (("GPP_CO2_FLUX", -1.0), ("TER_CO2_FLUX", 1.0)):
            zonal_mass = np.zeros((nbins, 12), dtype=np.float64)
            hour_start = 0
            for month in range(1, 13):
                nhours = calendar.monthrange(2021, month)[1] * 24
                values = np.ma.filled(
                    ds[variable][hour_start:hour_start + nhours], 0.0
                ).astype(np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite {variable} values in {path}, month {month}")
                if variable == "GPP_CO2_FLUX" and np.max(values) > 1.0e-15:
                    raise ValueError(f"positive atmospheric GPP flux in {path}, month {month}")
                if variable == "TER_CO2_FLUX" and np.min(values) < -1.0e-15:
                    raise ValueError(f"negative atmospheric TER flux in {path}, month {month}")
                carbon_per_cell = (
                    sign * values.sum(axis=0).ravel() * 3600.0
                    * KG_C_PER_KG_CO2 * area
                )
                zonal_mass[:, month - 1] = np.bincount(
                    bin_index, weights=carbon_per_cell, minlength=nbins
                )[:nbins]
                hour_start += nhours
            outputs.append(zonal_mass / KG_PER_PG / np.diff(sin_edges)[:, None])
    return sin_edges, outputs[0], outputs[1]


def style_axis(axis: plt.Axes, row: int, nrows: int) -> None:
    latitude_ticks = np.array([-90, -30, 0, 30, 60, 90])
    axis.set_xlim(0.5, 12.5)
    axis.set_ylim(-1.0, 1.0)
    axis.set_yticks(
        np.sin(np.deg2rad(latitude_ticks)), labels=[f"{value}°" for value in latitude_ticks]
    )
    axis.set_xticks(np.arange(1, 13))
    if row == nrows - 1:
        axis.set_xticklabels(MONTH_LABELS)
    else:
        axis.set_xticklabels([])
    axis.tick_params(labelsize=7, length=2)
    axis.axhline(0.0, color="0.35", linewidth=0.35, alpha=0.6)


def atlas(path: Path, data: np.ndarray, models: list[str], title: str,
          colorbar_label: str, anomaly: bool = False) -> None:
    ncols = 5
    nrows = math.ceil(len(models) / ncols)
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(18, 3.0 * nrows), squeeze=False,
        sharex=True, sharey=True,
    )
    figure.set_layout_engine(None)
    figure.subplots_adjust(left=0.055, right=0.91, bottom=0.055, top=0.90,
                           wspace=0.10, hspace=0.25)
    month_edges = np.arange(0.5, 13.5)
    sin_edges = np.linspace(-1.0, 1.0, data.shape[1] + 1)
    if anomaly:
        limit = max(float(np.nanpercentile(np.abs(data), 99.5)), 1.0e-12)
        normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        plot_options = {"cmap": "RdBu_r", "norm": normalization}
    else:
        limit = float(np.nanpercentile(data, 99.5))
        plot_options = {"cmap": "YlGn", "vmin": 0.0, "vmax": limit}
    image = None
    for index, (model, values) in enumerate(zip(models, data)):
        row, column = divmod(index, ncols)
        axis = axes[row, column]
        image = axis.pcolormesh(
            month_edges, sin_edges, values, shading="flat", **plot_options
        )
        annual = float(np.sum(values * np.diff(sin_edges)[:, None]))
        axis.set_title(f"{model}  ({annual:.1f} Pg C)", fontsize=9, fontweight="bold")
        style_axis(axis, row, nrows)
    for index in range(len(models), nrows * ncols):
        axes.flat[index].axis("off")
    if image is None:
        raise ValueError("empty ensemble")
    color_axis = figure.add_axes((0.93, 0.12, 0.015, 0.72))
    colorbar = figure.colorbar(image, cax=color_axis, extend="both" if anomaly else "max")
    colorbar.set_label(colorbar_label)
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.975)
    figure.text(0.5, 0.015, "Month of 2021", ha="center", fontsize=10)
    figure.text(0.012, 0.5, "Latitude (equal-area sin(latitude) axis)",
                va="center", rotation="vertical", fontsize=10)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def reference_plot(path: Path, mean: np.ndarray, median: np.ndarray,
                   quantity: str, signed: bool = False) -> None:
    difference = mean - median
    sin_edges = np.linspace(-1.0, 1.0, mean.shape[0] + 1)
    month_edges = np.arange(0.5, 13.5)
    common_max = max(float(np.max(mean)), float(np.max(median)))
    difference_max = max(float(np.max(np.abs(difference))), 1.0e-12)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    if signed:
        common_limit = max(float(np.max(np.abs(mean))), float(np.max(np.abs(median))), 1.0e-12)
        common_options = {
            "cmap": "RdBu_r",
            "norm": TwoSlopeNorm(vmin=-common_limit, vcenter=0.0, vmax=common_limit),
        }
    else:
        common_options = {"cmap": "YlGn", "vmin": 0.0, "vmax": common_max}
    image = axes[0].pcolormesh(month_edges, sin_edges, mean, shading="flat",
                               **common_options)
    axes[1].pcolormesh(month_edges, sin_edges, median, shading="flat",
                       **common_options)
    diff_image = axes[2].pcolormesh(
        month_edges, sin_edges, difference, cmap="RdBu_r", shading="flat",
        norm=TwoSlopeNorm(vmin=-difference_max, vcenter=0.0, vmax=difference_max),
    )
    for axis, label in zip(axes, ("Multi-model mean", "Multi-model median", "Mean minus median")):
        axis.set_title(label, loc="left", fontweight="bold")
        style_axis(axis, 2 if axis is axes[-1] else 0, 3)
        axis.set_ylabel("Latitude")
    axes[-1].set_xticklabels(MONTH_LABELS)
    axes[-1].set_xlabel("Month of 2021")
    figure.colorbar(image, ax=axes[:2], label=f"Zonal {quantity} (Pg C month⁻¹ per unit sin(latitude))")
    figure.colorbar(diff_image, ax=axes[2], label="Mean − median")
    figure.suptitle(f"TRENDYv14 S3 ensemble {quantity}, 2021", fontweight="bold")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--nbins", type=int, default=60)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "hovmoller"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = model_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"no TRENDY driver files under {args.input_dir}")

    models = []
    gpp, ter = [], []
    sin_edges = None
    for model, path in files:
        print(f"Aggregating {model}", flush=True)
        edges, model_gpp, model_ter = aggregate_file(path, args.nbins)
        if sin_edges is not None and not np.array_equal(edges, sin_edges):
            raise ValueError("inconsistent latitude bins")
        sin_edges = edges
        models.append(model)
        gpp.append(model_gpp)
        ter.append(model_ter)
    arrays = {"GPP": np.stack(gpp), "TER": np.stack(ter)}
    arrays["NEE"] = arrays["TER"] - arrays["GPP"]

    for quantity, values in arrays.items():
        lower = quantity.lower()
        ensemble_mean = np.mean(values, axis=0)
        ensemble_median = np.median(values, axis=0)
        units = f"Zonal {quantity} (Pg C month⁻¹ per unit sin(latitude))"
        is_nee = quantity == "NEE"
        atlas(
            output_dir / f"trendy_s3_2021_{lower}_absolute_atlas.png",
            values, models, f"TRENDYv14 S3 {quantity}: absolute monthly zonal totals", units,
            anomaly=is_nee,
        )
        atlas(
            output_dir / f"trendy_s3_2021_{lower}_deviation_from_mean_atlas.png",
            values - ensemble_mean[None, :, :], models,
            f"TRENDYv14 S3 {quantity}: deviation from multi-model mean",
            f"Δ {units}", anomaly=True,
        )
        atlas(
            output_dir / f"trendy_s3_2021_{lower}_deviation_from_median_atlas.png",
            values - ensemble_median[None, :, :], models,
            f"TRENDYv14 S3 {quantity}: deviation from multi-model median",
            f"Δ {units}", anomaly=True,
        )
        reference_plot(
            output_dir / f"trendy_s3_2021_{lower}_ensemble_mean_median.png",
            ensemble_mean, ensemble_median, quantity, signed=is_nee,
        )

    np.savez_compressed(
        output_dir / "trendy_s3_2021_monthly_zonal_hovmoller.npz",
        models=np.asarray(models), sinlat_edges=sin_edges,
        gpp_PgC_month_per_sinlat=arrays["GPP"],
        ter_PgC_month_per_sinlat=arrays["TER"],
        nee_PgC_month_per_sinlat=arrays["NEE"],
        gpp_ensemble_mean=np.mean(arrays["GPP"], axis=0),
        gpp_ensemble_median=np.median(arrays["GPP"], axis=0),
        ter_ensemble_mean=np.mean(arrays["TER"], axis=0),
        ter_ensemble_median=np.median(arrays["TER"], axis=0),
        nee_ensemble_mean=np.mean(arrays["NEE"], axis=0),
        nee_ensemble_median=np.median(arrays["NEE"], axis=0),
    )
    with (output_dir / "trendy_s3_2021_annual_totals.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "gpp_PgC", "ter_PgC", "nee_PgC"])
        width = np.diff(sin_edges)[:, None]
        for model, model_gpp, model_ter in zip(models, arrays["GPP"], arrays["TER"]):
            total_gpp = float(np.sum(model_gpp * width))
            total_ter = float(np.sum(model_ter * width))
            writer.writerow([model, f"{total_gpp:.8f}", f"{total_ter:.8f}",
                             f"{total_ter - total_gpp:.8f}"])
    print(f"Wrote ensemble products for {len(models)} models to {output_dir}")


if __name__ == "__main__":
    main()
