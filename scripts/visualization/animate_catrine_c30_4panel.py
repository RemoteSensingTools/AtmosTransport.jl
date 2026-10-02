#!/usr/bin/env python3
"""Animate the four CATRINE column-mean tracers from daily C30 NetCDF files.

The default movie uses one frame per day. Pass ``--cadence-hours 6`` for a
six-hourly movie, or ``--cadence-hours 3`` when three-hourly snapshots exist.
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import Normalize, PowerNorm
from netCDF4 import Dataset

# The host currently has an older SciPy paired with a newer NumPy. cKDTree is
# compatible for this operation, but importing SciPy otherwise emits an
# irrelevant version warning into the terminal.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="A NumPy version.*is required")
    from scipy.spatial import cKDTree


TRACERS = (
    ("co2_natural", "Natural XCO₂", 1.0e6, "ppm", "viridis", "linear"),
    ("co2_fossil", "Fossil XCO₂", 1.0e6, "ppm", "YlOrRd", "power"),
    ("rn222", "²²²Rn column mean", 1.0e21, "10⁻²¹ mol mol⁻¹ dry", "magma", "power"),
    ("sf6", "SF₆ column mean", 1.0e12, "ppt", "viridis", "linear"),
)


@dataclass(frozen=True)
class Frame:
    path: Path
    index: int
    hour: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/temp2/catrine-runs/output/catrine_c30_2021_fullphysics_column"),
    )
    parser.add_argument(
        "--pattern", default="catrine_c30_2021_column_*.nc",
        help="glob pattern within --input-dir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/temp2/catrine-runs/output/catrine_c30_2021_fullphysics_column/"
            "visualization/catrine_c30_2021_4panel.mp4"
        ),
    )
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--start", type=datetime.fromisoformat,
                        default=datetime(2021, 1, 1))
    parser.add_argument("--movie-start", type=datetime.fromisoformat,
                        help="first timestamp to include (inclusive)")
    parser.add_argument("--movie-stop", type=datetime.fromisoformat,
                        help="last timestamp bound (exclusive)")
    parser.add_argument("--cadence-hours", type=float, default=24.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--expected-files", type=int, default=365)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="testing/QC limit; zero means all selected frames")
    return parser.parse_args()


def xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    coslat = np.cos(lat_r)
    return np.column_stack(
        (coslat * np.cos(lon_r), coslat * np.sin(lon_r), np.sin(lat_r))
    )


def discover_frames(args: argparse.Namespace) -> tuple[list[Path], list[Frame]]:
    files = sorted(args.input_dir.glob(args.pattern))
    if args.expected_files > 0 and len(files) != args.expected_files:
        raise RuntimeError(
            f"expected {args.expected_files} daily files matching {args.pattern!r}, "
            f"found {len(files)} in {args.input_dir}"
        )
    cadence = float(args.cadence_hours)
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("--cadence-hours must be positive")

    frames: list[Frame] = []
    seen: set[int] = set()
    for path in files:
        with Dataset(path) as ds:
            times = np.asarray(ds["time"][:], dtype=float)
            for index, hour in enumerate(times):
                stamp = args.start + timedelta(hours=float(hour))
                if args.movie_start is not None and stamp < args.movie_start:
                    continue
                if args.movie_stop is not None and stamp >= args.movie_stop:
                    continue
                multiple = round(float(hour) / cadence)
                if abs(float(hour) - multiple * cadence) > 1.0e-6:
                    continue
                key = round(float(hour) * 3600)
                if key in seen:
                    continue
                seen.add(key)
                frames.append(Frame(path, index, float(hour)))
    frames.sort(key=lambda frame: frame.hour)
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("no frames matched the requested cadence")
    return files, frames


class FrameReader:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.dataset: Dataset | None = None

    def read(self, frame: Frame, variable: str) -> np.ndarray:
        if frame.path != self.path:
            self.close()
            self.dataset = Dataset(frame.path)
            self.path = frame.path
        assert self.dataset is not None
        return np.asarray(self.dataset[variable][frame.index], dtype=np.float32)

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = None
        self.path = None


def color_norm(values: np.ndarray, kind: str):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return Normalize(0.0, 1.0)
    if kind == "power":
        positive = finite[finite > 0]
        vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        return PowerNorm(gamma=0.38, vmin=0.0, vmax=max(vmax, np.finfo(float).eps))
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not hi > lo:
        hi = lo + max(abs(float(lo)) * 1.0e-6, 1.0)
    return Normalize(float(lo), float(hi))


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    mpl.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg")
    files, frames = discover_frames(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(files[0]) as ds:
        lon = np.asarray(ds["lons"][:], dtype=float).ravel()
        lat = np.asarray(ds["lats"][:], dtype=float).ravel()
        missing = [f"{name}_column_mean" for name, *_ in TRACERS
                   if f"{name}_column_mean" not in ds.variables]
        if missing:
            raise RuntimeError(f"missing column variables in {files[0]}: {missing}")

    target_lon_1d = np.arange(-179.5, 180.0, 1.0)
    target_lat_1d = np.arange(-89.5, 90.0, 1.0)
    target_lon, target_lat = np.meshgrid(target_lon_1d, target_lat_1d)
    nearest = cKDTree(xyz(lon, lat)).query(
        xyz(target_lon.ravel(), target_lat.ravel()), workers=-1
    )[1]

    print(f"Found {len(files)} daily files and {len(frames)} movie frames")
    print("Scanning fields for robust per-panel color limits ...")
    samples: dict[str, list[np.ndarray]] = {name: [] for name, *_ in TRACERS}
    reader = FrameReader()
    try:
        for n, frame in enumerate(frames, start=1):
            for name, _title, scale, _unit, _cmap, _kind in TRACERS:
                values = reader.read(frame, f"{name}_column_mean").ravel()
                samples[name].append(values[::4].astype(np.float64) * scale)
            if n % 60 == 0:
                print(f"  scanned {n}/{len(frames)} frames")
    finally:
        reader.close()

    norms = {
        name: color_norm(np.concatenate(samples[name]), kind)
        for name, _title, _scale, _unit, _cmap, kind in TRACERS
    }
    del samples

    projection = ccrs.Robinson()
    plate_carree = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 2, figsize=(13.6, 7.8), subplot_kw={"projection": projection}
    )
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.07, top=0.91,
                        wspace=0.08, hspace=0.18)
    images = []
    reader = FrameReader()
    first = frames[0]
    for axis, (name, title, scale, unit, cmap_name, _kind) in zip(axes.flat, TRACERS):
        field = reader.read(first, f"{name}_column_mean").ravel()[nearest]
        field = field.reshape(target_lon.shape) * scale
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("0.9")
        image = axis.imshow(
            field,
            origin="lower",
            extent=(-180, 180, -90, 90),
            transform=plate_carree,
            cmap=cmap,
            norm=norms[name],
            interpolation="nearest",
        )
        axis.set_global()
        axis.coastlines(linewidth=0.4, color="0.2", alpha=0.75)
        axis.set_title(title, fontsize=12, fontweight="bold")
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal",
                                pad=0.035, shrink=0.78, aspect=35)
        colorbar.set_label(unit, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        images.append(image)

    title = fig.suptitle("", fontsize=14, fontweight="bold", y=0.975)
    footer = fig.text(
        0.5, 0.02,
        "AtmosTransport · experimental C90→C30 forcing · PPM + TM5 convection/PBL",
        ha="center", fontsize=8.5, color="0.3",
    )

    def update(frame_number: int):
        frame = frames[frame_number]
        for image, (name, _title, scale, _unit, _cmap, _kind) in zip(images, TRACERS):
            field = reader.read(frame, f"{name}_column_mean").ravel()[nearest]
            image.set_data(field.reshape(target_lon.shape) * scale)
        stamp = args.start + timedelta(hours=frame.hour)
        title.set_text(f"CATRINE C30 column means · {stamp:%Y-%m-%d %H:%M UTC}")
        if (frame_number + 1) % 60 == 0:
            print(f"  rendered {frame_number + 1}/{len(frames)} frames")
        return (*images, title, footer)

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / args.fps, blit=False
    )
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "CATRINE C30 four-tracer column means"},
    )
    try:
        animation.save(args.output, writer=writer, dpi=args.dpi)
        poster = args.poster or args.output.with_suffix(".png")
        update(len(frames) - 1)
        fig.savefig(poster, dpi=160, bbox_inches="tight")
    finally:
        reader.close()
        plt.close(fig)
    print(f"Wrote {args.output} and {poster}")


if __name__ == "__main__":
    main()
