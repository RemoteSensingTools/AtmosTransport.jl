#!/usr/bin/env python3
"""Daily-cadence XCO2 animation from the C30 JULES-ES NEE and control runs.

Two panels, both true daily model output (no monthly aggregation, no
interpolation, no deseasonalising):

  left    NEE-driven XCO2, instantaneous spatial pattern: the real tracer with
          the (area-weighted) global mean removed at every step. Shows the full
          seasonal sweep plus genuine synoptic weather.
  right   flux-driven XCO2: real minus the climatological-flux control. Exact
          by linearity, and seasonal-free by construction (the flux difference
          has zero calendar-month climatology), so it is shown raw with only
          each cell's time mean removed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from animate_jules_components_xco2 import make_mapper

INK = "#1f2933"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots/sif_gpp_iav"))
    ap.add_argument("--resolution", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth day (1 = every day)")
    ap.add_argument("--dpi", type=int, default=105)
    args = ap.parse_args()

    z = np.load(args.scratch / "c30_daily_grids.npz", allow_pickle=True)
    dates, lons, lats, area = z["dates"], z["lons"], z["lats"], z["area"]
    w = (area.ravel() / area.sum()).astype(np.float32)
    real = z["real"]
    pattern = real - (real * w[None, :]).sum(axis=1, keepdims=True)
    fluxdrv = real - z["clim"]
    fluxdrv = fluxdrv - fluxdrv.mean(axis=0, keepdims=True)   # each cell's time mean

    sel = np.arange(0, len(dates), args.stride)
    v1 = float(np.percentile(np.abs(pattern[sel[::10]]), 99.5))
    v2 = float(np.percentile(np.abs(fluxdrv[sel[::10]]), 99.5))
    print(f"{len(sel)} frames; pattern scale +/-{v1:.1f} ppm, "
          f"flux-driven scale +/-{v2:.2f} ppm")

    tlon, tlat, nearest = make_mapper(lons, lats, args.resolution)
    fig = plt.figure(figsize=(15.6, 5.6))
    grid = fig.add_gridspec(1, 2, left=0.02, right=0.98, bottom=0.17, top=0.82,
                            wspace=0.04)
    titles = ("NEE-driven XCO2, global mean removed (seasonal + synoptic)",
              "flux-driven XCO2: real - climatological control (exact)")
    images = []
    for i, (field0, vmax, title) in enumerate(
            ((pattern[0], v1, titles[0]), (fluxdrv[0], v2, titles[1]))):
        ax = fig.add_subplot(grid[i], projection=ccrs.Robinson())
        ax.set_global()
        ax.set_facecolor("0.92")
        ax.coastlines(linewidth=0.35, color="0.25")
        im = ax.pcolormesh(tlon, tlat, field0[nearest].reshape(tlon.shape),
                           transform=ccrs.PlateCarree(), shading="nearest",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=4, color=INK)
        images.append(im)
    cax1 = fig.add_axes([0.10, 0.085, 0.32, 0.028])
    fig.colorbar(images[0], cax=cax1, orientation="horizontal", extend="both",
                 label="ppm (seasonal pattern)")
    cax2 = fig.add_axes([0.60, 0.085, 0.32, 0.028])
    fig.colorbar(images[1], cax=cax2, orientation="horizontal", extend="both",
                 label="ppm (flux-driven IAV)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "jules_nee_xco2_daily.mp4"
    writer = FFMpegWriter(fps=args.fps, bitrate=5000,
                          metadata={"title": "JULES-ES XCO2, daily"})
    with writer.saving(fig, str(out), args.dpi):
        for n, k in enumerate(sel):
            images[0].set_array(pattern[k][nearest].reshape(tlon.shape).ravel())
            images[1].set_array(fluxdrv[k][nearest].reshape(tlon.shape).ravel())
            fig.suptitle("JULES-ES transported NEE XCO2, C30, true daily output"
                         f" -- {dates[k]}", fontsize=13.5, fontweight="bold",
                         y=0.97, color=INK)
            writer.grab_frame()
            if n % 300 == 0:
                print(f"  frame {n}/{len(sel)} {dates[k]}", flush=True)
    print(f"wrote {out} ({len(sel)} frames, {len(sel) / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
