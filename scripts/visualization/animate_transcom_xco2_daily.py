#!/usr/bin/env python3
"""Animate daily TransCom-region SIF-GPP contributions to transported XCO2."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from netCDF4 import Dataset
from scipy.spatial import cKDTree


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
    parser.add_argument("--group", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--start", type=datetime.fromisoformat, default=datetime(2021, 12, 1))
    parser.add_argument("--vmin", type=float)
    return parser.parse_args()


def xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    coslat = np.cos(lat_r)
    return np.column_stack((coslat * np.cos(lon_r), coslat * np.sin(lon_r), np.sin(lat_r)))


def tracer_from_var(name: str, variable) -> str:
    tracer = str(getattr(variable, "tracer", "") or name.removeprefix("xco2_anomaly_"))
    return tracer if tracer.startswith("co2_sif_") else f"co2_sif_{tracer}"


def read_inputs(paths: list[Path]):
    lon = lat = times = None
    fields: list[tuple[str, str, np.ndarray]] = []
    for path in paths:
        ds = Dataset(path)
        this_lon = np.asarray(ds["cs_lon"][:], dtype=float)
        this_lat = np.asarray(ds["cs_lat"][:], dtype=float)
        this_times = np.asarray(ds["time_hours"][:], dtype=float)
        if lon is None:
            lon, lat, times = this_lon, this_lat, this_times
        elif not (np.allclose(lon, this_lon) and np.allclose(lat, this_lat) and np.allclose(times, this_times)):
            raise ValueError(f"grid/time mismatch in {path}")
        for name in ds.variables:
            if not name.startswith("xco2_anomaly_"):
                continue
            if name == "xco2_anomaly_total" or ds[name].dimensions != ("time", "cell"):
                continue
            tracer = tracer_from_var(name, ds[name])
            label = LABELS.get(tracer, tracer.removeprefix("co2_sif_").replace("_", " ").title())
            fields.append((tracer, label, np.asarray(ds[name][:], dtype=np.float32)))
        ds.close()

    if lon is None or lat is None or times is None:
        raise ValueError("no input data")
    preferred = list(LABELS)
    fields.sort(key=lambda f: preferred.index(f[0]) if f[0] in preferred else len(preferred))
    return lon, lat, times, fields


def main() -> None:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lon, lat, times, fields = read_inputs(args.group)
    nframes = len(times)

    lon_centres = np.arange(-179.5, 180.0, 1.0)
    lat_centres = np.arange(-89.5, 90.0, 1.0)
    target_lon, target_lat = np.meshgrid(lon_centres, lat_centres)
    tree = cKDTree(xyz(lon, lat))
    _, nearest = tree.query(xyz(target_lon.ravel(), target_lat.ravel()), workers=-1)

    if args.vmin is None:
        sample = np.concatenate([data[:, ::16].ravel() for _, _, data in fields])
        sample = sample[np.isfinite(sample)]
        vmin = min(float(np.percentile(sample, 0.5)), -0.01)
    else:
        vmin = args.vmin
    vmax = 0.0

    fig = plt.figure(figsize=(15, 10), dpi=120)
    axes = []
    images = []
    for i, (_, label, data) in enumerate(fields, start=1):
        ax = fig.add_subplot(3, 4, i, projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.35, color="0.25")
        mapped = data[0, nearest].reshape(target_lon.shape)
        image = ax.pcolormesh(
            target_lon, target_lat, mapped, transform=ccrs.PlateCarree(),
            shading="nearest", cmap="YlGnBu_r", vmin=vmin, vmax=vmax, rasterized=True,
        )
        ax.set_title(label, fontsize=9)
        axes.append(ax)
        images.append(image)

    title = fig.suptitle("", fontsize=13)
    cbar = fig.colorbar(images[0], ax=axes, orientation="horizontal", shrink=0.72, pad=0.04)
    cbar.set_label("Regional contribution to XCO₂ anomaly from SIF-GPP uptake (ppm)")

    def update(frame: int):
        for image, (_, _, data) in zip(images, fields):
            mapped = data[frame, nearest].reshape(target_lon.shape)
            image.set_array(mapped.ravel())
        stamp = args.start + timedelta(hours=float(times[frame]))
        title.set_text(f"TransCom regional SIF-GPP XCO₂ attribution — {stamp:%Y-%m-%d %H:%M UTC}")
        return [*images, title]

    animation = FuncAnimation(fig, update, frames=nframes, interval=1000 / args.fps, blit=False)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=5500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Daily TransCom regional SIF-GPP XCO2 attribution, December 2021"},
    )
    animation.save(args.output, writer=writer, dpi=120)

    poster = args.poster or args.output.with_suffix(".png")
    update(nframes - 1)
    fig.savefig(poster, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output} ({nframes} frames at {args.fps} fps) and {poster}; scale={vmin:.3f}..0 ppm")


if __name__ == "__main__":
    main()
