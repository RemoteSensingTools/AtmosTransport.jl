#!/usr/bin/env python3
"""Plot monthly zonal GPP totals for the 2021 SIF and FLUXCOM-X drivers."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4
import numpy as np


R_EARTH = 6_371_000.0
G_PER_PG = 1.0e15
PILOT_DEFAULT = Path("/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot")


def c90_cell_area(corner_lat: np.ndarray, corner_lon: np.ndarray) -> np.ndarray:
    """Return cubed-sphere cell areas from corner coordinates."""

    def unit(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        return np.stack(
            (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
            axis=-1,
        )

    def triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        numerator = np.abs(np.sum(a * np.cross(b, c), axis=-1))
        denominator = (
            1.0
            + np.sum(a * b, axis=-1)
            + np.sum(b * c, axis=-1)
            + np.sum(c * a, axis=-1)
        )
        return 2.0 * np.arctan2(numerator, denominator)

    p00 = unit(corner_lat[:, :-1, :-1], corner_lon[:, :-1, :-1])
    p10 = unit(corner_lat[:, :-1, 1:], corner_lon[:, :-1, 1:])
    p11 = unit(corner_lat[:, 1:, 1:], corner_lon[:, 1:, 1:])
    p01 = unit(corner_lat[:, 1:, :-1], corner_lon[:, 1:, :-1])
    return (triangle(p00, p10, p11) + triangle(p00, p11, p01)) * R_EARTH**2


def finite(data: np.ndarray) -> np.ndarray:
    """Convert a netCDF variable result to an ndarray with masked values zeroed."""
    if np.ma.isMaskedArray(data):
        data = data.filled(0.0)
    return np.nan_to_num(np.asarray(data), nan=0.0, posinf=0.0, neginf=0.0)


def add_zonal_mass(
    target: np.ndarray,
    month: int,
    carbon_per_cell_g: np.ndarray,
    bin_index: np.ndarray,
    nbins: int,
) -> None:
    target[:, month] += np.bincount(
        bin_index, weights=carbon_per_cell_g, minlength=nbins
    )[:nbins]


def aggregate(pilot: Path, nbins: int) -> tuple[np.ndarray, ...]:
    weekly_path = pilot / "sif_gpp_weekly_diurnal_c90_2021.nc"
    fluxcom_path = pilot / "fluxcom_x_c90_2021.npz"

    with netCDF4.Dataset(weekly_path) as ds:
        lat = np.asarray(ds["lats"][:]).ravel()
        area = c90_cell_area(ds["corner_lats"][:], ds["corner_lons"][:]).ravel()
        land_fraction = finite(ds["land_fraction"][:]).ravel()
        week_start_days = np.asarray(ds["time"][:])
        interval_days = np.asarray(ds["interval_days"][:], dtype=int)

        sin_edges = np.linspace(-1.0, 1.0, nbins + 1)
        bin_index = np.searchsorted(sin_edges, np.sin(np.deg2rad(lat)), side="right") - 1
        bin_index = np.clip(bin_index, 0, nbins - 1)
        area_land = area * land_fraction
        sif_mass = np.zeros((nbins, 12), dtype=np.float64)

        unix_epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        for week, start_day in enumerate(week_start_days):
            start = unix_epoch + timedelta(days=float(start_day))
            weekly_total = finite(ds["gpp_weekly_total"][week]).ravel()
            for day in range(interval_days[week]):
                date = start + timedelta(days=day)
                fraction = finite(ds["daily_fraction"][week, day]).ravel()
                add_zonal_mass(
                    sif_mass,
                    date.month - 1,
                    weekly_total * fraction * area_land,
                    bin_index,
                    nbins,
                )

    with np.load(fluxcom_path) as source:
        daily = source["daily"]
        daily_time = source["daily_time"]
        # The cached land fraction is the same field archived in the weekly file.
        if not np.allclose(source["land_fraction"], land_fraction, equal_nan=True):
            raise ValueError("FLUXCOM-X and weekly-product land fractions differ")
        fluxcom_mass = np.zeros((nbins, 12), dtype=np.float64)
        fluxcom_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
        for day, time_value in enumerate(daily_time):
            date = fluxcom_epoch + timedelta(days=float(time_value))
            gpp = np.maximum(finite(daily[day]), 0.0)
            add_zonal_mass(
                fluxcom_mass,
                date.month - 1,
                gpp * area_land,
                bin_index,
                nbins,
            )

    # Express zonal totals as density per unit sin(latitude). Multiplication by
    # the constant bin width and summation over latitude recovers Pg C month-1.
    delta_sinlat = np.diff(sin_edges)[:, None]
    sif_density = sif_mass / G_PER_PG / delta_sinlat
    fluxcom_density = fluxcom_mass / G_PER_PG / delta_sinlat
    sif_monthly = sif_mass.sum(axis=0) / G_PER_PG
    fluxcom_monthly = fluxcom_mass.sum(axis=0) / G_PER_PG
    return sin_edges, sif_density, fluxcom_density, sif_monthly, fluxcom_monthly


def write_csv(path: Path, sif: np.ndarray, fluxcom: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["month", "sif_gpp_PgC", "fluxcom_x_gpp_PgC", "sif_minus_fluxcom_PgC"])
        for month in range(12):
            writer.writerow(
                [month + 1, f"{sif[month]:.8f}", f"{fluxcom[month]:.8f}",
                 f"{sif[month] - fluxcom[month]:.8f}"]
            )


def make_plot(
    output: Path,
    sin_edges: np.ndarray,
    sif: np.ndarray,
    fluxcom: np.ndarray,
    sif_monthly: np.ndarray,
    fluxcom_monthly: np.ndarray,
) -> None:
    months = np.arange(1, 13)
    month_edges = np.arange(0.5, 13.5)
    difference = sif - fluxcom

    common_max = float(max(np.nanmax(sif), np.nanmax(fluxcom)))
    difference_max = float(np.nanmax(np.abs(difference)))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.0, 12.5),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.62], "hspace": 0.16},
        constrained_layout=False,
    )
    # Override site-wide matplotlib auto-layout settings so the shared
    # colorbars, title, and provenance footer keep deterministic spacing.
    fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.115, top=0.93, hspace=0.34)

    common_kwargs = dict(shading="flat", cmap="YlGn", vmin=0.0, vmax=common_max)
    image_sif = axes[0].pcolormesh(month_edges, sin_edges, sif, **common_kwargs)
    axes[1].pcolormesh(month_edges, sin_edges, fluxcom, **common_kwargs)
    image_difference = axes[2].pcolormesh(
        month_edges,
        sin_edges,
        difference,
        shading="flat",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-difference_max, vcenter=0.0, vmax=difference_max),
    )

    axes[0].set_title("SIF-constrained GPP", loc="left", fontweight="bold")
    axes[1].set_title("FLUXCOM-X GPP", loc="left", fontweight="bold")
    axes[2].set_title("Difference: SIF-constrained minus FLUXCOM-X", loc="left", fontweight="bold")

    lat_ticks = np.array([-90, -60, -30, 0, 30, 60, 90])
    for axis in axes[:3]:
        axis.set_xlim(0.5, 12.5)
        axis.set_ylim(-1.0, 1.0)
        axis.set_yticks(np.sin(np.deg2rad(lat_ticks)), labels=[f"{x}°" for x in lat_ticks])
        axis.set_ylabel("Latitude\n(equal-area sin(lat) axis)")
        axis.set_xticks(months)
        axis.tick_params(axis="x", labelbottom=False)
        axis.axhline(0.0, color="0.3", linewidth=0.5, alpha=0.5)
    axes[2].tick_params(axis="x", labelbottom=True)
    axes[2].set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    colorbar = fig.colorbar(image_sif, ax=axes[:2], pad=0.015, fraction=0.025)
    colorbar.set_label("Zonal GPP (Pg C month⁻¹ per unit sin(latitude))")
    diff_colorbar = fig.colorbar(image_difference, ax=axes[2], pad=0.015, fraction=0.025)
    diff_colorbar.set_label("Δ zonal GPP (Pg C month⁻¹ per unit sin(latitude))")

    axes[3].plot(months, sif_monthly, marker="o", linewidth=2.0, label="SIF-constrained")
    axes[3].plot(months, fluxcom_monthly, marker="s", linewidth=2.0, label="FLUXCOM-X")
    axes[3].axhline(0.0, color="0.3", linewidth=0.6)
    axes[3].set_xlim(0.5, 12.5)
    axes[3].set_xticks(months, labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axes[3].set_ylabel("Global GPP\n(Pg C month⁻¹)")
    axes[3].set_xlabel("2021")
    axes[3].grid(axis="y", color="0.85", linewidth=0.7)
    axes[3].legend(frameon=False, ncol=2, loc="upper right")

    fig.suptitle(
        "Monthly GPP: SIF-constrained pilot versus FLUXCOM-X (C90)",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.01,
        0.018,
        "Monthly carbon totals are area-integrated using C90 cell area and vegetated-land fraction. "
        "SIF timing uses its conserved daily allocation; negative FLUXCOM-X predictions are clipped to zero.",
        fontsize=8.5,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=PILOT_DEFAULT)
    parser.add_argument("--nbins", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.nbins < 4:
        parser.error("--nbins must be at least 4")

    output = args.output or args.pilot / "sif_vs_fluxcom_monthly_hovmoller_2021.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    values = aggregate(args.pilot, args.nbins)
    make_plot(output, *values)
    write_csv(output.with_suffix(".csv"), values[3], values[4])
    np.savez_compressed(
        output.with_suffix(".npz"),
        sinlat_edges=values[0],
        sif_gpp_PgC_month_per_sinlat=values[1],
        fluxcom_x_gpp_PgC_month_per_sinlat=values[2],
        difference_PgC_month_per_sinlat=values[1] - values[2],
        sif_monthly_PgC=values[3],
        fluxcom_x_monthly_PgC=values[4],
    )
    print(f"Wrote {output}")
    print(f"SIF annual total:       {values[3].sum():.6f} Pg C")
    print(f"FLUXCOM-X annual total: {values[4].sum():.6f} Pg C")
    print(f"Annual difference:      {(values[3] - values[4]).sum():.6f} Pg C")


if __name__ == "__main__":
    main()
