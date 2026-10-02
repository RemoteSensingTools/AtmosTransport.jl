#!/usr/bin/env python3
"""Animate an hourly cubed-sphere XCO2 anomaly time series on a global map."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--start", type=datetime.fromisoformat, default=datetime(2021, 12, 1))
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--vmin", type=float)
    return parser.parse_args()


def xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    coslat = np.cos(lat_r)
    return np.column_stack((coslat * np.cos(lon_r), coslat * np.sin(lon_r), np.sin(lat_r)))


def main() -> None:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = Dataset(args.input)
    lon = np.asarray(ds["cs_lon"][:])
    lat = np.asarray(ds["cs_lat"][:])
    times = np.asarray(ds["time_hours"][:], dtype=float)
    anomaly = ds["xco2_anomaly"]
    nframes = len(times)

    lon_centres = np.arange(-179.5, 180.0, 1.0)
    lat_centres = np.arange(-89.5, 90.0, 1.0)
    target_lon, target_lat = np.meshgrid(lon_centres, lat_centres)
    tree = cKDTree(xyz(lon, lat))
    _, nearest = tree.query(xyz(target_lon.ravel(), target_lat.ravel()), workers=-1)

    if args.vmin is None:
        sample_t = np.unique(np.linspace(0, nframes - 1, min(64, nframes), dtype=int))
        sample_cells = np.arange(0, len(lon), 8)
        sample = np.asarray(anomaly[sample_t, :])[:, sample_cells].ravel()
        sample = sample[np.isfinite(sample)]
        vmin = float(np.percentile(sample, 0.5))
        vmin = min(vmin, -0.1)
    else:
        vmin = args.vmin
    vmax = 0.0

    fig = plt.figure(figsize=(12.8, 7.2), dpi=120)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.45, color="0.25")
    ax.set_facecolor("0.92")
    first = np.asarray(anomaly[0, :])[nearest].reshape(target_lon.shape)
    image = ax.pcolormesh(
        target_lon,
        target_lat,
        first,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        cmap="YlGnBu_r",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.045, shrink=0.78)
    colorbar.set_label("SIF-GPP contribution to XCO₂ (ppm; 400 ppm carrier removed)")
    title = ax.set_title("")
    subtitle = fig.text(
        0.5,
        0.035,
        "Hourly length-of-day-corrected SIF uptake • GEOS-IT C180 full-physics transport",
        ha="center",
        fontsize=9,
    )

    def update(frame: int):
        field = np.asarray(anomaly[frame, :])[nearest].reshape(target_lon.shape)
        image.set_array(field.ravel())
        stamp = args.start + timedelta(hours=float(times[frame]))
        title.set_text(f"Transported SIF-GPP XCO₂ anomaly — {stamp:%Y-%m-%d %H:%M UTC}")
        return image, title, subtitle

    animation = FuncAnimation(fig, update, frames=nframes, interval=1000 / args.fps, blit=False)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Hourly transported SIF-GPP XCO2 anomaly, December 2021"},
    )
    animation.save(args.output, writer=writer, dpi=120)

    poster = args.poster or args.output.with_suffix(".png")
    update(nframes - 1)
    fig.savefig(poster, dpi=160, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    print(f"Wrote {args.output} ({nframes} frames at {args.fps} fps) and {poster}; scale={vmin:.3f}..0 ppm")


if __name__ == "__main__":
    main()
