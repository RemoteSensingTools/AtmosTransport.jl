#!/usr/bin/env python3
"""Plot daily regional TransCom-tagged SIF-GPP XCO2 attribution diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from netCDF4 import Dataset
from scipy.spatial import cKDTree


@dataclass
class RegionalField:
    tracer: str
    label: str
    values: np.ndarray


LABELS = {
    "co2_sif_south_america_tropical": "S. America tropical",
    "co2_sif_south_america_temperate": "S. America temperate",
    "co2_sif_northern_africa": "N. Africa",
    "co2_sif_southern_africa": "S. Africa",
    "co2_sif_tropical_asia": "Tropical Asia",
    "co2_sif_australia": "Australia",
    "co2_sif_na_boreal": "N. America boreal",
    "co2_sif_na_temperate": "N. America temperate",
    "co2_sif_eurasia_boreal": "Eurasia boreal",
    "co2_sif_eurasia_temperate": "Eurasia temperate",
    "co2_sif_europe": "Europe",
    "co2_sif_unassigned": "Unassigned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", type=Path, required=True, help="compact group NetCDF; repeatable")
    parser.add_argument("--baseline", type=Path, help="single-tracer compact baseline NetCDF for linearity check")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.01, help="ppm magnitude threshold for dominance mask")
    return parser.parse_args()


def xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    coslat = np.cos(lat_r)
    return np.column_stack((coslat * np.cos(lon_r), coslat * np.sin(lon_r), np.sin(lat_r)))


def safe_to_tracer(varname: str) -> str:
    name = varname.removeprefix("xco2_anomaly_")
    return name if name.startswith("co2_sif_") else f"co2_sif_{name}"


def read_groups(paths: list[Path], time_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[RegionalField]]:
    fields: list[RegionalField] = []
    lon = lat = area = times = None
    for path in paths:
        with Dataset(path) as ds:
            this_lon = np.asarray(ds["cs_lon"][:], dtype=float)
            this_lat = np.asarray(ds["cs_lat"][:], dtype=float)
            this_area = np.asarray(ds["cs_area"][:], dtype=float)
            this_times = np.asarray(ds["time_hours"][:], dtype=float)
            if lon is None:
                lon, lat, area, times = this_lon, this_lat, this_area, this_times
            else:
                if not (np.allclose(lon, this_lon) and np.allclose(lat, this_lat) and np.allclose(times, this_times)):
                    raise ValueError(f"grid/time mismatch in {path}")

            ntime = len(this_times)
            idx = time_index if time_index >= 0 else ntime + time_index
            if idx < 0 or idx >= ntime:
                raise IndexError(f"time-index {time_index} out of range for {path}")

            for name in ds.variables:
                if not name.startswith("xco2_anomaly_"):
                    continue
                if name == "xco2_anomaly_total" or ds[name].dimensions != ("time", "cell"):
                    continue
                tracer = str(getattr(ds[name], "tracer", safe_to_tracer(name)))
                if not tracer.startswith("co2_sif_"):
                    tracer = f"co2_sif_{tracer}"
                label = LABELS.get(tracer, tracer.removeprefix("co2_sif_").replace("_", " ").title())
                fields.append(RegionalField(tracer=tracer, label=label, values=np.asarray(ds[name][idx, :], dtype=float)))
    assert lon is not None and lat is not None and area is not None and times is not None
    order = [f.tracer for f in fields]
    preferred = list(LABELS)
    fields.sort(key=lambda f: preferred.index(f.tracer) if f.tracer in preferred else len(preferred) + order.index(f.tracer))
    return lon, lat, area, times, fields


def target_grid(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_centres = np.arange(-179.5, 180.0, 1.0)
    lat_centres = np.arange(-89.5, 90.0, 1.0)
    target_lon, target_lat = np.meshgrid(lon_centres, lat_centres)
    tree = cKDTree(xyz(lon, lat))
    _, nearest = tree.query(xyz(target_lon.ravel(), target_lat.ravel()), workers=-1)
    return target_lon, target_lat, nearest


def plot_multipanel(out: Path, lon2: np.ndarray, lat2: np.ndarray, nearest: np.ndarray, fields: list[RegionalField], hour: float) -> None:
    sample = np.concatenate([f.values[np.isfinite(f.values)] for f in fields])
    vmin = min(float(np.percentile(sample, 0.5)), -0.01)
    vmax = 0.0
    fig = plt.figure(figsize=(15, 10), dpi=140)
    for i, field in enumerate(fields, start=1):
        ax = fig.add_subplot(3, 4, i, projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.35, color="0.25")
        mapped = field.values[nearest].reshape(lon2.shape)
        im = ax.pcolormesh(
            lon2, lat2, mapped, transform=ccrs.PlateCarree(), shading="nearest",
            cmap="YlGnBu_r", vmin=vmin, vmax=vmax, rasterized=True,
        )
        ax.set_title(field.label, fontsize=9)
    fig.suptitle(f"Regional SIF-GPP contributions to transported XCO₂, t={hour:.0f} h", fontsize=13)
    cbar = fig.colorbar(im, ax=fig.axes, orientation="horizontal", shrink=0.72, pad=0.04)
    cbar.set_label("XCO₂ anomaly from regional uptake tracer (ppm)")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_dominance(out: Path, lon2: np.ndarray, lat2: np.ndarray, nearest: np.ndarray, fields: list[RegionalField], threshold: float, hour: float) -> None:
    stack = np.stack([f.values for f in fields])
    total = np.sum(stack, axis=0)
    dominant = np.argmin(stack, axis=0)
    dominant[np.abs(total) < threshold] = -1
    mapped = dominant[nearest].reshape(lon2.shape)

    colors = ["0.85"] + list(plt.cm.tab20(np.linspace(0, 1, len(fields))))
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-1.5, len(fields) + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(13, 7), dpi=150)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.45, color="0.2")
    ax.pcolormesh(lon2, lat2, mapped, transform=ccrs.PlateCarree(), shading="nearest", cmap=cmap, norm=norm, rasterized=True)
    ax.set_title(f"Dominant regional source of SIF-GPP XCO₂ drawdown, t={hour:.0f} h")
    handles = [Patch(facecolor=colors[i + 1], edgecolor="none", label=f.label) for i, f in enumerate(fields)]
    handles.insert(0, Patch(facecolor=colors[0], edgecolor="none", label=f"|total| < {threshold:g} ppm"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, -0.12), ncol=4, fontsize=8, frameon=False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_linearity_check(out: Path, baseline: Path | None, lon: np.ndarray, lat: np.ndarray, area: np.ndarray, fields: list[RegionalField], time_index: int) -> None:
    regional_sum = np.sum(np.stack([f.values for f in fields]), axis=0)
    lines = []
    for f in fields:
        gm = float(np.sum(f.values * area) / np.sum(area))
        lines.append(f"{f.tracer:40s} global_mean_ppm={gm: .6f} min={np.nanmin(f.values): .6f} max={np.nanmax(f.values): .6f}")
    total_gm = float(np.sum(regional_sum * area) / np.sum(area))
    lines.append(f"{'regional_sum':40s} global_mean_ppm={total_gm: .6f} min={np.nanmin(regional_sum): .6f} max={np.nanmax(regional_sum): .6f}")
    if baseline is not None:
        with Dataset(baseline) as ds:
            b_times = np.asarray(ds["time_hours"][:], dtype=float)
            idx = time_index if time_index >= 0 else len(b_times) + time_index
            base = np.asarray(ds["xco2_anomaly"][idx, :], dtype=float)
        diff = regional_sum - base
        lines.append("")
        lines.append(f"baseline={baseline}")
        lines.append(f"regional_minus_baseline area_mean_ppm={float(np.sum(diff * area) / np.sum(area)): .8f}")
        lines.append(f"regional_minus_baseline rms_ppm={float(np.sqrt(np.mean(diff * diff))): .8f}")
        lines.append(f"regional_minus_baseline maxabs_ppm={float(np.max(np.abs(diff))): .8f}")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lon, lat, area, times, fields = read_groups(args.group, args.time_index)
    hour = float(times[args.time_index])
    lon2, lat2, nearest = target_grid(lon, lat)
    plot_multipanel(args.output_dir / "transcom_xco2_final_multipanel.png", lon2, lat2, nearest, fields, hour)
    plot_dominance(args.output_dir / "transcom_xco2_final_dominance.png", lon2, lat2, nearest, fields, args.threshold, hour)
    write_linearity_check(args.output_dir / "transcom_xco2_linearity_check.txt", args.baseline, lon, lat, area, fields, args.time_index)
    print(f"Wrote plots/checks to {args.output_dir}")


if __name__ == "__main__":
    main()
