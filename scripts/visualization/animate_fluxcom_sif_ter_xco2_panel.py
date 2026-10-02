#!/usr/bin/env python3
"""Animate the 2021 FLUXCOM-X, SIF-GPP, and TER XCO2 contributions.

The transport tracers use independent 400 ppm carriers.  This script removes
each carrier before forming sums or differences and renders a two-column by
three-row comparison panel from the daily split NetCDF output.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree


DEFAULT_INPUT = Path(
    "/home/cfranken/data/AtmosTransport/output/"
    "fluxcom_sif_gpp_ter_c90_2021_advdiff"
)
BACKGROUND_PPM = 400.0
START = datetime(2021, 1, 1, tzinfo=timezone.utc)
VARIABLES = {
    "sif": "co2_sif_gpp_column_mean",
    "fluxcom": "co2_fluxcom_gpp_column_mean",
    "ter": "co2_fluxcom_ter_column_mean",
}


@dataclass(frozen=True)
class Frame:
    path: Path
    local_index: int
    elapsed_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-step", type=int, default=1,
                        help="retain every Nth 3-hourly frame; the final frame is always included")
    parser.add_argument("--target-resolution", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--scale-sample-stride", type=int, default=8,
                        help="cell stride used while estimating robust color limits")
    return parser.parse_args()


def xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.deg2rad(np.asarray(lon))
    lat_rad = np.deg2rad(np.asarray(lat))
    coslat = np.cos(lat_rad)
    return np.column_stack(
        (coslat * np.cos(lon_rad), coslat * np.sin(lon_rad), np.sin(lat_rad))
    )


def discover_frames(input_dir: Path, frame_step: int) -> tuple[list[Frame], list[Path]]:
    paths = sorted(input_dir.glob("fluxcom_sif_gpp_ter_c90_????????.nc"))
    if len(paths) != 365:
        raise ValueError(f"expected 365 daily files, found {len(paths)} under {input_dir}")
    all_frames = []
    previous_time = -np.inf
    for path in paths:
        with netCDF4.Dataset(path) as ds:
            for name in VARIABLES.values():
                if name not in ds.variables:
                    raise KeyError(f"{name} missing from {path}")
            times = np.asarray(ds["time"][:], dtype=np.float64)
        for local_index, value in enumerate(times):
            if value <= previous_time:
                raise ValueError(f"non-increasing output time at {path}:{local_index}")
            previous_time = float(value)
            all_frames.append(Frame(path, local_index, float(value)))
    selected = all_frames[::frame_step]
    if selected[-1] != all_frames[-1]:
        selected.append(all_frames[-1])
    return selected, paths


def anomaly(variable: netCDF4.Variable, selection) -> np.ndarray:
    values = np.ma.filled(variable[selection], np.nan).astype(np.float32)
    values = values * np.float32(1.0e6) - np.float32(BACKGROUND_PPM)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values in {variable.name}")
    return values


class DailyCache:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.data: dict[str, np.ndarray] = {}

    def fields(self, frame: Frame) -> tuple[np.ndarray, ...]:
        if frame.path != self.path:
            with netCDF4.Dataset(frame.path) as ds:
                self.data = {
                    key: anomaly(ds[name], slice(None)).reshape(len(ds.dimensions["time"]), -1)
                    for key, name in VARIABLES.items()
                }
            self.path = frame.path
        fluxcom = self.data["fluxcom"][frame.local_index]
        sif = self.data["sif"][frame.local_index]
        ter = self.data["ter"][frame.local_index]
        direct_difference = sif - fluxcom
        net_sif = sif + ter
        net_fluxcom = fluxcom + ter
        net_difference = net_sif - net_fluxcom
        if np.max(np.abs(net_difference - direct_difference)) > 1.0e-4:
            raise ValueError("net-flux difference does not match the direct GPP difference")
        return fluxcom, sif, ter, direct_difference, net_sif, net_difference


def robust_limits(paths: list[Path], stride: int) -> dict[str, tuple[float, float]]:
    samples: list[list[np.ndarray]] = [[] for _ in range(6)]
    # The last 3-hourly state from each day resolves the annual accumulation
    # while keeping the color-limit prepass compact.
    for path in paths:
        with netCDF4.Dataset(path) as ds:
            sif = anomaly(ds[VARIABLES["sif"]], -1).ravel()[::stride]
            fluxcom = anomaly(ds[VARIABLES["fluxcom"]], -1).ravel()[::stride]
            ter = anomaly(ds[VARIABLES["ter"]], -1).ravel()[::stride]
        fields = (
            fluxcom, sif, ter, sif - fluxcom, sif + ter,
            (sif + ter) - (fluxcom + ter),
        )
        for destination, field in zip(samples, fields):
            destination.append(field)
    values = [np.concatenate(group) for group in samples]

    gpp = np.concatenate((values[0], values[1]))
    gpp_min = min(float(np.percentile(gpp, 0.25)), -0.05)
    ter_max = max(float(np.percentile(values[2], 99.75)), 0.05)
    difference_limit = max(
        float(np.percentile(np.abs(np.concatenate((values[3], values[5]))), 99.75)),
        0.02,
    )
    net_limit = max(float(np.percentile(np.abs(values[4]), 99.75)), 0.05)
    return {
        "gpp": (gpp_min, 0.0),
        "ter": (0.0, ter_max),
        "difference": (-difference_limit, difference_limit),
        "net": (-net_limit, net_limit),
    }


def main() -> None:
    args = parse_args()
    if args.frame_step < 1 or args.fps < 1 or args.scale_sample_stride < 1:
        raise ValueError("fps, frame-step, and scale-sample-stride must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg
    output = args.output or args.input_dir / "animations/xco2_fluxcom_sif_ter_3x2_2021.mp4"
    poster = args.poster or output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)

    frames, daily_paths = discover_frames(args.input_dir, args.frame_step)
    with netCDF4.Dataset(daily_paths[0]) as ds:
        source_lon = np.asarray(ds["lons"][:], dtype=np.float64).ravel()
        source_lat = np.asarray(ds["lats"][:], dtype=np.float64).ravel()

    resolution = args.target_resolution
    lon_centres = np.arange(-180.0 + resolution / 2.0, 180.0, resolution)
    lat_centres = np.arange(-90.0 + resolution / 2.0, 90.0, resolution)
    target_lon, target_lat = np.meshgrid(lon_centres, lat_centres)
    tree = cKDTree(xyz(source_lon, source_lat))
    nearest = tree.query(xyz(target_lon.ravel(), target_lat.ravel()), k=1)[1]
    limits = robust_limits(daily_paths, args.scale_sample_stride)
    print(f"Frames: {len(frames)}; robust scales: {limits}", flush=True)

    panel_specs = (
        ("FLUXCOM-X GPP contribution", "gpp", "YlGnBu_r"),
        ("SIF-constrained GPP contribution", "gpp", "YlGnBu_r"),
        ("FLUXCOM-X TER contribution", "ter", "YlOrRd"),
        ("SIF GPP − FLUXCOM-X GPP", "difference", "RdBu_r"),
        ("Net: SIF GPP + TER", "net", "RdBu_r"),
        ("Net(SIF+TER) − Net(FLUXCOM+TER)", "difference", "RdBu_r"),
    )
    figure = plt.figure(figsize=(16.0, 9.0))
    figure.set_layout_engine(None)
    outer = figure.add_gridspec(
        3, 2, left=0.025, right=0.975, bottom=0.085, top=0.925,
        hspace=0.22, wspace=0.06,
    )
    axes, images, colorbars = [], [], []
    cache = DailyCache()
    first_fields = cache.fields(frames[0])
    for index, (title, scale_name, cmap) in enumerate(panel_specs):
        row, column = divmod(index, 2)
        cell = outer[row, column].subgridspec(2, 1, height_ratios=(1.0, 0.06), hspace=0.04)
        axis = figure.add_subplot(cell[0], projection=ccrs.Robinson())
        color_axis = figure.add_subplot(cell[1])
        axis.set_global()
        axis.set_facecolor("0.90")
        axis.coastlines(linewidth=0.35, color="0.25")
        mapped = first_fields[index][nearest].reshape(target_lon.shape)
        vmin, vmax = limits[scale_name]
        if scale_name in ("difference", "net"):
            normalization = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            image = axis.pcolormesh(
                target_lon, target_lat, mapped, transform=ccrs.PlateCarree(),
                shading="nearest", cmap=cmap, norm=normalization, rasterized=True,
            )
        else:
            image = axis.pcolormesh(
                target_lon, target_lat, mapped, transform=ccrs.PlateCarree(),
                shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
            )
        axis.set_title(title, fontsize=10.5, fontweight="bold", pad=3)
        colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal", extend="both")
        colorbar.set_label("XCO₂ transport contribution (ppm)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7, length=2)
        axes.append(axis)
        images.append(image)
        colorbars.append(colorbar)

    main_title = figure.suptitle("", fontsize=15, fontweight="bold", y=0.982)
    figure.text(
        0.5, 0.015,
        "C90 ERA5-N320 advection + TM5 DKG diffusion; no convection • "
        "400 ppm numerical carrier removed independently from every tracer",
        ha="center", fontsize=8.5,
    )

    def update(frame_index: int):
        frame = frames[frame_index]
        fields = cache.fields(frame)
        for image, field in zip(images, fields):
            image.set_array(field[nearest].reshape(target_lon.shape).ravel())
        stamp = START + timedelta(hours=frame.elapsed_hours)
        main_title.set_text(
            f"Transported biospheric XCO₂ contributions — {stamp:%Y-%m-%d %H:%M UTC}"
        )
        if frame_index % 100 == 0 or frame_index == len(frames) - 1:
            print(f"Rendering frame {frame_index + 1}/{len(frames)}", flush=True)
        return [*images, main_title]

    animation = FuncAnimation(
        figure, update, frames=len(frames), interval=1000.0 / args.fps, blit=False,
    )
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=8500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "2021 FLUXCOM-X, SIF-GPP, and TER transported XCO2 contributions"},
    )
    animation.save(output, writer=writer, dpi=args.dpi)

    update(len(frames) - 1)
    figure.savefig(poster, dpi=160, bbox_inches="tight")
    plt.close(figure)
    duration = len(frames) / args.fps
    print(
        f"Wrote {output} ({len(frames)} frames, {duration:.1f} s at {args.fps} fps) "
        f"and {poster}", flush=True,
    )


if __name__ == "__main__":
    main()
