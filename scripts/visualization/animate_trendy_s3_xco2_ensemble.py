#!/usr/bin/env python3
"""Animate TRENDYv14 S3 transported XCO2 by model against ensemble references."""

from __future__ import annotations

import argparse
import re
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
    "trendy_v14_s3_gpp_ter_c90_2021_advdiff"
)
DEFAULT_FLUXCOM = Path(
    "/home/cfranken/data/AtmosTransport/output/"
    "fluxcom_sif_gpp_ter_c90_2021_advdiff"
)
START = datetime(2021, 1, 1, tzinfo=timezone.utc)
BACKGROUND_PPM = 400.0
TRENDY_PATTERN = re.compile(r"co2_trendy_(.+)_(gpp|ter)_column_mean$")
DAILY_PATTERN = re.compile(r".*_(\d{8})\.nc$")


@dataclass(frozen=True)
class Frame:
    date: str
    local_index: int
    elapsed_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fluxcom-dir", type=Path, default=DEFAULT_FLUXCOM)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", default="all",
                        help="comma-separated model ids after co2_trendy_, or 'all'")
    parser.add_argument("--frame-step", type=int, default=8,
                        help="retain every Nth 3-hourly frame; 8 gives daily movies")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--target-resolution", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--scale-sample-stride", type=int, default=10)
    parser.add_argument("--no-fluxcom-reference", action="store_true")
    return parser.parse_args()


def xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.deg2rad(np.asarray(lon))
    lat_rad = np.deg2rad(np.asarray(lat))
    coslat = np.cos(lat_rad)
    return np.column_stack(
        (coslat * np.cos(lon_rad), coslat * np.sin(lon_rad), np.sin(lat_rad))
    )


def discover_daily_files(input_dir: Path) -> dict[str, list[Path]]:
    files_by_date: dict[str, list[Path]] = {}
    for path in sorted(input_dir.glob("batch*/trendy_v14_s3_batch*_c90_????????.nc")):
        match = DAILY_PATTERN.match(path.name)
        if match:
            files_by_date.setdefault(match.group(1), []).append(path)
    if len(files_by_date) != 365:
        raise ValueError(f"expected 365 daily groups, found {len(files_by_date)}")
    return files_by_date


def discover_models(files_by_date: dict[str, list[Path]]) -> list[str]:
    models: dict[str, set[str]] = {}
    for path in files_by_date[min(files_by_date)]:
        with netCDF4.Dataset(path) as ds:
            for name in ds.variables:
                match = TRENDY_PATTERN.match(name)
                if match:
                    models.setdefault(match.group(1), set()).add(match.group(2))
    paired = sorted(model for model, kinds in models.items() if kinds == {"gpp", "ter"})
    if not paired:
        raise ValueError("no paired TRENDY GPP/TER column-mean variables found")
    return paired


def discover_frames(files_by_date: dict[str, list[Path]], frame_step: int) -> list[Frame]:
    all_frames: list[Frame] = []
    previous = -np.inf
    for date in sorted(files_by_date):
        with netCDF4.Dataset(files_by_date[date][0]) as ds:
            times = np.asarray(ds["time"][:], dtype=np.float64)
        for local_index, value in enumerate(times):
            if value <= previous:
                raise ValueError(f"non-increasing time at {date}:{local_index}")
            previous = float(value)
            all_frames.append(Frame(date, local_index, float(value)))
    selected = all_frames[::frame_step]
    if selected[-1] != all_frames[-1]:
        selected.append(all_frames[-1])
    return selected


def anomaly(variable: netCDF4.Variable, selection) -> np.ndarray:
    values = np.ma.filled(variable[selection], np.nan).astype(np.float32)
    values = values * np.float32(1.0e6) - np.float32(BACKGROUND_PPM)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values in {variable.name}")
    return values.reshape(-1)


def read_trendy_frame(files: list[Path], models: list[str], local_index: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wanted = set(models)
    for path in files:
        with netCDF4.Dataset(path) as ds:
            for model in list(wanted):
                gpp_name = f"co2_trendy_{model}_gpp_column_mean"
                ter_name = f"co2_trendy_{model}_ter_column_mean"
                if gpp_name in ds.variables and ter_name in ds.variables:
                    result[model] = (
                        anomaly(ds[gpp_name], local_index),
                        anomaly(ds[ter_name], local_index),
                    )
                    wanted.remove(model)
        if not wanted:
            break
    if wanted:
        raise KeyError(f"missing model tracers for {sorted(wanted)}")
    return result


def read_fluxcom_nee(fluxcom_dir: Path, date: str, local_index: int) -> np.ndarray | None:
    path = fluxcom_dir / f"fluxcom_sif_gpp_ter_c90_{date}.nc"
    if not path.exists():
        return None
    with netCDF4.Dataset(path) as ds:
        gpp = anomaly(ds["co2_fluxcom_gpp_column_mean"], local_index)
        ter = anomaly(ds["co2_fluxcom_ter_column_mean"], local_index)
    return gpp + ter


class FrameCache:
    def __init__(self, files_by_date: dict[str, list[Path]], models: list[str],
                 fluxcom_dir: Path | None) -> None:
        self.files_by_date = files_by_date
        self.models = models
        self.fluxcom_dir = fluxcom_dir
        self.key: tuple[str, int] | None = None
        self.payload: dict[str, object] = {}

    def fields(self, frame: Frame) -> dict[str, object]:
        key = (frame.date, frame.local_index)
        if key != self.key:
            model_fields = read_trendy_frame(
                self.files_by_date[frame.date], self.models, frame.local_index
            )
            gpp_stack = np.stack([model_fields[model][0] for model in self.models])
            ter_stack = np.stack([model_fields[model][1] for model in self.models])
            nee_stack = gpp_stack + ter_stack
            payload: dict[str, object] = {
                "model_fields": model_fields,
                "mean_gpp": np.mean(gpp_stack, axis=0),
                "mean_ter": np.mean(ter_stack, axis=0),
                "mean_nee": np.mean(nee_stack, axis=0),
                "fluxcom_nee": None,
            }
            if self.fluxcom_dir is not None:
                payload["fluxcom_nee"] = read_fluxcom_nee(
                    self.fluxcom_dir, frame.date, frame.local_index
                )
            self.payload = payload
            self.key = key
        return self.payload


def robust_limits(files_by_date: dict[str, list[Path]], models: list[str],
                  frames: list[Frame], fluxcom_dir: Path | None, stride: int) -> dict[str, tuple[float, float]]:
    gpp_samples: list[np.ndarray] = []
    ter_samples: list[np.ndarray] = []
    nee_samples: list[np.ndarray] = []
    dgpp_samples: list[np.ndarray] = []
    dter_samples: list[np.ndarray] = []
    dnee_samples: list[np.ndarray] = []
    fluxcom_dnee_samples: list[np.ndarray] = []
    sample_frames = frames[::max(1, len(frames) // 48)] + [frames[-1]]
    cache = FrameCache(files_by_date, models, fluxcom_dir)
    for frame in sample_frames:
        payload = cache.fields(frame)
        model_fields = payload["model_fields"]
        for model in models:
            gpp, ter = model_fields[model]
            nee = gpp + ter
            gpp_samples.append(gpp[::stride])
            ter_samples.append(ter[::stride])
            nee_samples.append(nee[::stride])
            dgpp_samples.append((gpp - payload["mean_gpp"])[::stride])
            dter_samples.append((ter - payload["mean_ter"])[::stride])
            dnee_samples.append((nee - payload["mean_nee"])[::stride])
            if payload["fluxcom_nee"] is not None:
                fluxcom_dnee_samples.append((nee - payload["fluxcom_nee"])[::stride])

    def signed_limit(samples: list[np.ndarray], floor: float) -> tuple[float, float]:
        limit = max(float(np.percentile(np.abs(np.concatenate(samples)), 99.5)), floor)
        return -limit, limit

    gpp_values = np.concatenate(gpp_samples)
    ter_values = np.concatenate(ter_samples)
    return {
        "gpp": (min(float(np.percentile(gpp_values, 0.5)), -0.02), 0.0),
        "ter": (0.0, max(float(np.percentile(ter_values, 99.5)), 0.02)),
        "nee": signed_limit(nee_samples, 0.02),
        "dgpp": signed_limit(dgpp_samples, 0.01),
        "dter": signed_limit(dter_samples, 0.01),
        "dnee": signed_limit(fluxcom_dnee_samples or dnee_samples, 0.01),
    }


def make_mapper(first_file: Path, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(first_file) as ds:
        source_lon = np.asarray(ds["lons"][:], dtype=np.float64).ravel()
        source_lat = np.asarray(ds["lats"][:], dtype=np.float64).ravel()
    lon_centres = np.arange(-180.0 + resolution / 2.0, 180.0, resolution)
    lat_centres = np.arange(-90.0 + resolution / 2.0, 90.0, resolution)
    target_lon, target_lat = np.meshgrid(lon_centres, lat_centres)
    nearest = cKDTree(xyz(source_lon, source_lat)).query(
        xyz(target_lon.ravel(), target_lat.ravel()), k=1
    )[1]
    return target_lon, target_lat, nearest


def render_model(model: str, frames: list[Frame], cache: FrameCache,
                 target_lon: np.ndarray, target_lat: np.ndarray, nearest: np.ndarray,
                 limits: dict[str, tuple[float, float]], output: Path, poster: Path,
                 fps: int, dpi: int, use_fluxcom_reference: bool) -> None:
    figure = plt.figure(figsize=(16.0, 8.4))
    figure.set_layout_engine(None)
    grid = figure.add_gridspec(
        2, 3, left=0.025, right=0.975, bottom=0.105, top=0.90,
        hspace=0.20, wspace=0.055,
    )
    specs = (
        ("GPP XCO2", "gpp", "YlGnBu_r"),
        ("TER XCO2", "ter", "YlOrRd"),
        ("NEE XCO2 = GPP + TER", "nee", "RdBu_r"),
        ("GPP minus multi-model mean", "dgpp", "RdBu_r"),
        ("TER minus multi-model mean", "dter", "RdBu_r"),
        ("NEE minus FLUXCOM-X NEE" if use_fluxcom_reference else "NEE minus multi-model mean", "dnee", "RdBu_r"),
    )
    images = []
    first = cache.fields(frames[0])

    def model_panel_fields(payload: dict[str, object]) -> tuple[np.ndarray, ...]:
        gpp, ter = payload["model_fields"][model]
        nee = gpp + ter
        reference_nee = payload["fluxcom_nee"] if use_fluxcom_reference and payload["fluxcom_nee"] is not None else payload["mean_nee"]
        return (
            gpp, ter, nee,
            gpp - payload["mean_gpp"],
            ter - payload["mean_ter"],
            nee - reference_nee,
        )

    first_fields = model_panel_fields(first)
    for index, (title, scale_name, cmap) in enumerate(specs):
        cell = grid[index].subgridspec(2, 1, height_ratios=(1.0, 0.065), hspace=0.04)
        axis = figure.add_subplot(cell[0], projection=ccrs.Robinson())
        color_axis = figure.add_subplot(cell[1])
        axis.set_global()
        axis.set_facecolor("0.90")
        axis.coastlines(linewidth=0.35, color="0.25")
        mapped = first_fields[index][nearest].reshape(target_lon.shape)
        vmin, vmax = limits[scale_name]
        if scale_name in {"nee", "dgpp", "dter", "dnee"}:
            image = axis.pcolormesh(
                target_lon, target_lat, mapped, transform=ccrs.PlateCarree(),
                shading="nearest", cmap=cmap,
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
                rasterized=True,
            )
        else:
            image = axis.pcolormesh(
                target_lon, target_lat, mapped, transform=ccrs.PlateCarree(),
                shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
            )
        axis.set_title(title, fontsize=10.5, fontweight="bold", pad=3)
        colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal", extend="both")
        colorbar.set_label("XCO2 contribution (ppm)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7, length=2)
        images.append(image)

    main_title = figure.suptitle("", fontsize=14.5, fontweight="bold", y=0.972)
    figure.text(
        0.5, 0.020,
        "C90 ERA5-N320 advection + TM5 DKG diffusion; no convection; "
        "400 ppm numerical carrier removed per tracer",
        ha="center", fontsize=8.5,
    )

    def update(frame_index: int):
        frame = frames[frame_index]
        fields = model_panel_fields(cache.fields(frame))
        for image, field in zip(images, fields):
            image.set_array(field[nearest].reshape(target_lon.shape).ravel())
        stamp = START + timedelta(hours=frame.elapsed_hours)
        main_title.set_text(f"TRENDYv14 S3 {model} transported XCO2 - {stamp:%Y-%m-%d %H:%M UTC}")
        if frame_index % 50 == 0 or frame_index == len(frames) - 1:
            print(f"{model}: rendering frame {frame_index + 1}/{len(frames)}", flush=True)
        return [*images, main_title]

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000.0 / fps, blit=False)
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=7600,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": f"TRENDYv14 S3 {model} transported XCO2"},
    )
    animation.save(output, writer=writer, dpi=dpi)
    update(len(frames) - 1)
    figure.savefig(poster, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.frame_step < 1 or args.fps < 1 or args.scale_sample_stride < 1:
        raise ValueError("frame-step, fps, and scale-sample-stride must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg

    files_by_date = discover_daily_files(args.input_dir)
    all_models = discover_models(files_by_date)
    if args.models == "all":
        models = all_models
    else:
        requested = [item.strip() for item in args.models.split(",") if item.strip()]
        missing = sorted(set(requested) - set(all_models))
        if missing:
            raise ValueError(f"requested models not found: {missing}")
        models = requested
    frames = discover_frames(files_by_date, args.frame_step)
    output_dir = args.output_dir or args.input_dir / "animations" / "trendy_s3_xco2_per_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    fluxcom_dir = None if args.no_fluxcom_reference else args.fluxcom_dir
    first_file = files_by_date[min(files_by_date)][0]
    target_lon, target_lat, nearest = make_mapper(first_file, args.target_resolution)
    limits = robust_limits(files_by_date, models, frames, fluxcom_dir, args.scale_sample_stride)
    print(f"Animating {len(models)} models, {len(frames)} frames each; scales={limits}", flush=True)
    cache = FrameCache(files_by_date, models, fluxcom_dir)
    for index, model in enumerate(models, start=1):
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        suffix = "vs_fluxcom_nee" if fluxcom_dir is not None else "vs_mmm"
        movie = output_dir / f"trendy_s3_2021_xco2_{safe_model}_{suffix}.mp4"
        poster = output_dir / f"trendy_s3_2021_xco2_{safe_model}_{suffix}.png"
        print(f"[{index}/{len(models)}] Writing {movie}", flush=True)
        render_model(
            model, frames, cache, target_lon, target_lat, nearest, limits,
            movie, poster, args.fps, args.dpi, fluxcom_dir is not None,
        )


if __name__ == "__main__":
    main()
