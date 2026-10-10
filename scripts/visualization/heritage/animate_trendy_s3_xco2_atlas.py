#!/usr/bin/env python3
"""Animate 24-panel TRENDY GPP, TER, and NEE XCO2 ensemble atlases.

Each atlas contains the multi-model mean followed by the 23 individual model
departures from that mean.  The three movies are rendered together so every
daily ensemble field is read from disk only once.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from scipy.spatial import cKDTree

# The shared helper module stays in scripts/visualization/, one level above heritage/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animate_trendy_s3_xco2_ensemble import (
    BACKGROUND_PPM,
    DEFAULT_INPUT,
    Frame,
    discover_daily_files,
    discover_frames,
    discover_models,
    read_trendy_frame,
    xyz,
)


START = datetime(2021, 1, 1, tzinfo=timezone.utc)
TRACERS = ("gpp", "ter", "nee")
ANOMALY_EDGES = np.arange(-4.25, 4.5, 0.5)
TITLES = {
    "gpp": "GPP",
    "ter": "TER",
    "nee": "NEE = GPP + TER",
}
CMAPS = {
    "gpp": "YlGnBu_r",
    "ter": "YlOrRd",
    "nee": "RdBu_r",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frame-step", type=int, default=8,
                        help="retain every Nth 3-hourly frame (default: daily)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--target-resolution", type=float, default=2.0,
                        help="regular plotting raster resolution in degrees")
    parser.add_argument("--scale-sample-frames", type=int, default=36)
    parser.add_argument("--scale-sample-stride", type=int, default=12)
    parser.add_argument("--tracers", default="gpp,ter,nee",
                        help="comma-separated subset of gpp,ter,nee")
    parser.add_argument("--max-frames", type=int,
                        help="testing aid: render at most this many selected frames")
    parser.add_argument("--poster-only", action="store_true",
                        help="write only the final-frame PNG atlases")
    return parser.parse_args()


def make_mapper(first_file: Path, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the cubed sphere once onto a regular grid used by all 72 panels."""
    with netCDF4.Dataset(first_file) as ds:
        source_lon = np.asarray(ds["lons"][:], dtype=np.float64).ravel()
        source_lat = np.asarray(ds["lats"][:], dtype=np.float64).ravel()
    lon = np.arange(-180.0 + resolution / 2.0, 180.0, resolution)
    lat = np.arange(-90.0 + resolution / 2.0, 90.0, resolution)
    target_lon, target_lat = np.meshgrid(lon, lat)
    nearest = cKDTree(xyz(source_lon, source_lat)).query(
        xyz(target_lon.ravel(), target_lat.ravel()), k=1, workers=-1
    )[1]
    return target_lon, target_lat, nearest


def stacks(fields: dict[str, tuple[np.ndarray, np.ndarray]],
           models: list[str]) -> dict[str, np.ndarray]:
    gpp = np.stack([fields[model][0] for model in models])
    ter = np.stack([fields[model][1] for model in models])
    return {"gpp": gpp, "ter": ter, "nee": gpp + ter}


def robust_limits(files_by_date: dict[str, list[Path]], models: list[str],
                  frames: list[Frame], sample_count: int,
                  cell_stride: int) -> dict[str, tuple[tuple[float, float], float]]:
    """Estimate separate MMM and model-departure scales from sparse samples."""
    indices = np.unique(np.linspace(0, len(frames) - 1, min(sample_count, len(frames)),
                                    dtype=int))
    mean_samples = {name: [] for name in TRACERS}
    departure_samples = {name: [] for name in TRACERS}
    for number, index in enumerate(indices, start=1):
        frame = frames[index]
        values = stacks(
            read_trendy_frame(files_by_date[frame.date], models, frame.local_index),
            models,
        )
        for name, stack in values.items():
            mean = np.mean(stack, axis=0)
            mean_samples[name].append(mean[::cell_stride])
            departure_samples[name].append(
                (stack[:, ::cell_stride] - mean[::cell_stride]).ravel()
            )
        if number % 12 == 0 or number == len(indices):
            print(f"Scale sampling {number}/{len(indices)}", flush=True)

    result = {}
    for name in TRACERS:
        means = np.concatenate(mean_samples[name])
        departures = np.concatenate(departure_samples[name])
        if name == "gpp":
            mean_range = (min(float(np.percentile(means, 0.5)), -0.02), 0.0)
        elif name == "ter":
            mean_range = (0.0, max(float(np.percentile(means, 99.5)), 0.02))
        else:
            bound = max(float(np.percentile(np.abs(means), 99.5)), 0.02)
            mean_range = (-bound, bound)
        departure_bound = max(
            float(np.percentile(np.abs(departures), 99.5)), 0.01
        )
        result[name] = (mean_range, departure_bound)
    return result


def panel_values(stack: np.ndarray) -> list[np.ndarray]:
    mean = np.mean(stack, axis=0)
    return [mean, *(stack - mean)]


def create_atlas(tracer: str, models: list[str], first_stack: np.ndarray,
                 target_lon: np.ndarray, target_lat: np.ndarray,
                 nearest: np.ndarray, mean_range: tuple[float, float],
                 departure_bound: float):
    figure = plt.figure(figsize=(19.2, 10.0))
    figure.set_layout_engine(None)
    grid = figure.add_gridspec(
        4, 6, left=0.008, right=0.992, bottom=0.105, top=0.925,
        hspace=0.12, wspace=0.015,
    )
    labels = ["Multi-model mean", *models]
    images = []
    first_values = panel_values(first_stack)
    mean_norm = (
        TwoSlopeNorm(vmin=mean_range[0], vcenter=0.0, vmax=mean_range[1])
        if mean_range[0] < 0.0 < mean_range[1] else None
    )
    anomaly_cmap = plt.get_cmap("RdBu_r", len(ANOMALY_EDGES) - 1)
    anomaly_norm = BoundaryNorm(ANOMALY_EDGES, anomaly_cmap.N, clip=True)
    for index, (label, field) in enumerate(zip(labels, first_values)):
        axis = figure.add_subplot(grid[index], projection=ccrs.Robinson())
        axis.set_global()
        axis.set_facecolor("0.88")
        axis.coastlines(linewidth=0.25, color="0.20")
        mapped = field[nearest].reshape(target_lon.shape)
        kwargs = (
            {"norm": mean_norm, "vmin": None if mean_norm else mean_range[0],
             "vmax": None if mean_norm else mean_range[1]}
            if index == 0 else {"norm": anomaly_norm}
        )
        image = axis.pcolormesh(
            target_lon, target_lat, mapped, transform=ccrs.PlateCarree(),
            shading="nearest", cmap=CMAPS[tracer] if index == 0 else anomaly_cmap,
            rasterized=True, **kwargs,
        )
        axis.set_title(label, fontsize=8.2, fontweight="bold" if index == 0 else None,
                       pad=1.5)
        images.append(image)

    mean_cax = figure.add_axes((0.14, 0.055, 0.29, 0.014))
    anomaly_cax = figure.add_axes((0.58, 0.055, 0.29, 0.014))
    mean_bar = figure.colorbar(images[0], cax=mean_cax, orientation="horizontal",
                               extend="both")
    anomaly_bar = figure.colorbar(images[1], cax=anomaly_cax,
                                  orientation="horizontal", extend="both",
                                  boundaries=ANOMALY_EDGES,
                                  ticks=np.arange(-4.0, 4.1, 1.0))
    mean_bar.set_label("Multi-model mean XCO₂ contribution (ppm)", fontsize=8)
    anomaly_bar.set_label("Model − multi-model mean XCO₂ (ppm)", fontsize=8)
    for bar in (mean_bar, anomaly_bar):
        bar.ax.tick_params(labelsize=7, length=2)
    title = figure.suptitle("", fontsize=14, fontweight="bold", y=0.975)
    figure.text(
        0.5, 0.012,
        f"23 TRENDYv14 S3 models; C90 ERA5-N320 advection + DKG diffusion; "
        f"no convection; {BACKGROUND_PPM:g} ppm carrier removed from GPP and TER",
        ha="center", fontsize=8,
    )
    return figure, images, title


def update_atlas(images, title, tracer: str, stack: np.ndarray,
                 nearest: np.ndarray, shape: tuple[int, int], stamp: datetime) -> None:
    for image, field in zip(images, panel_values(stack)):
        image.set_array(field[nearest].reshape(shape).ravel())
    title.set_text(
        f"TRENDYv14 S3 transported {TITLES[tracer]} XCO₂ — "
        f"{stamp:%Y-%m-%d %H:%M UTC}"
    )


def main() -> None:
    args = parse_args()
    if min(args.frame_step, args.fps, args.dpi, args.scale_sample_frames,
           args.scale_sample_stride) < 1 or args.target_resolution <= 0:
        raise ValueError("frame, rendering, sampling, and resolution options must be positive")
    selected_tracers = [item.strip().lower() for item in args.tracers.split(",")
                        if item.strip()]
    invalid = sorted(set(selected_tracers) - set(TRACERS))
    if invalid or not selected_tracers:
        raise ValueError(f"invalid tracer selection: {invalid}")
    if not args.poster_only:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required unless --poster-only is used")
        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg

    files_by_date = discover_daily_files(args.input_dir)
    models = discover_models(files_by_date)
    if len(models) != 23:
        raise ValueError(f"the 6x4 atlas requires 23 models; discovered {len(models)}")
    frames = discover_frames(files_by_date, args.frame_step)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    output_dir = args.output_dir or args.input_dir / "animations" / "trendy_s3_xco2_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_lon, target_lat, nearest = make_mapper(
        files_by_date[min(files_by_date)][0], args.target_resolution
    )
    limits = robust_limits(
        files_by_date, models, frames, args.scale_sample_frames,
        args.scale_sample_stride,
    )

    initial_frame = frames[-1] if args.poster_only else frames[0]
    initial = stacks(
        read_trendy_frame(files_by_date[initial_frame.date], models,
                          initial_frame.local_index),
        models,
    )
    atlases = {}
    for tracer in selected_tracers:
        atlases[tracer] = create_atlas(
            tracer, models, initial[tracer], target_lon, target_lat, nearest,
            *limits[tracer],
        )

    if args.poster_only:
        stamp = START + timedelta(hours=initial_frame.elapsed_hours)
        for tracer, (figure, images, title) in atlases.items():
            update_atlas(images, title, tracer, initial[tracer], nearest,
                         target_lon.shape, stamp)
            figure.savefig(
                output_dir / f"trendy_s3_2021_{tracer}_xco2_atlas.png",
                dpi=max(args.dpi, 150),
            )
            plt.close(figure)
        return

    writers = {}
    with ExitStack() as stack:
        for tracer, (figure, _, _) in atlases.items():
            writer = FFMpegWriter(
                fps=args.fps, codec="libx264", bitrate=9000,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-preset", "fast"],
                metadata={"title": f"TRENDYv14 S3 {tracer.upper()} XCO2 atlas"},
            )
            path = output_dir / f"trendy_s3_2021_{tracer}_xco2_atlas.mp4"
            stack.enter_context(writer.saving(figure, path, dpi=args.dpi))
            writers[tracer] = writer

        for index, frame in enumerate(frames, start=1):
            values = stacks(
                read_trendy_frame(files_by_date[frame.date], models,
                                  frame.local_index),
                models,
            )
            stamp = START + timedelta(hours=frame.elapsed_hours)
            for tracer, (figure, images, title) in atlases.items():
                update_atlas(images, title, tracer, values[tracer], nearest,
                             target_lon.shape, stamp)
                writers[tracer].grab_frame()
            if index % 10 == 0 or index == len(frames):
                print(f"Rendered {index}/{len(frames)} timesteps into "
                      f"{len(atlases)} atlases", flush=True)

    for tracer, (figure, _, _) in atlases.items():
        figure.savefig(output_dir / f"trendy_s3_2021_{tracer}_xco2_atlas.png",
                       dpi=150)
        plt.close(figure)


if __name__ == "__main__":
    main()
