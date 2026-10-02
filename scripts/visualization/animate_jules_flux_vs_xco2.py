#!/usr/bin/env python3
"""Animate JULES-ES flux anomalies above the XCO2 anomalies they produce.

A 3 x 2 matrix, fire excluded:

  columns   GPP - Ra (NPP term) | Rh | NEE = Ra + Rh - GPP
  row 1     transported column XCO2 anomaly, ppm
  row 2     the surface flux anomaly that produced it, g C m-2 day-1,
            straight from the driver file

The point is to separate what the land model actually does from what transport
does to it. In XCO2 the two components look like near-perfect mirror images,
but that partly reflects the smoothing and accumulation of transport; the flux
row shows how much of the anti-correlation is real, and where only one of the
two terms is anomalous.

Both rows are detrended and deseasonalised by the same joint backfit, and each
row carries one colour scale across its three panels so the columns are
comparable. NEE is the exact sum of the two components in both rows.
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

from animate_jules_components_xco2 import backfit, interpolate_frames, make_mapper

COLUMNS = ("GPP - Ra  (NPP term)", "Rh", "NEE  (no fire)")
INK = "#1f2933"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots/sif_gpp_iav"))
    ap.add_argument("--resolution", type=float, default=1.5)
    ap.add_argument("--frames-per-month", type=int, default=6)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    x = np.load(args.scratch / "c30_maps_jules_fire.npz", allow_pickle=True)
    f = np.load(args.scratch / "c30_jules_flux_maps.npz", allow_pickle=True)
    months, lons, lats = x["months"], x["lons"], x["lats"]

    # Sign convention: both rows are the contribution to atmospheric CO2, so
    # red always means "adds carbon to the atmosphere". The gpp tracer and the
    # GPP flux variable already carry the uptake sign.
    flux = [f["GPP_CO2_FLUX"] + f["RA_CO2_FLUX"], f["RH_CO2_FLUX"]]
    flux.append(flux[0] + flux[1])
    xco2 = [x["co2_jules_es_gpp"] + x["co2_jules_es_ra"], x["co2_jules_es_rh"]]
    xco2.append(xco2[0] + xco2[1])

    rows = []
    for series, unit in ((xco2, "ppm"), (flux, "g C m$^{-2}$ day$^{-1}$")):
        fields = [backfit(s) for s in series]
        vmax = float(np.percentile(np.abs(np.concatenate(
            [g.ravel() for g in fields])), 99.5))
        rows.append((fields, vmax, unit))
        print(f"{unit:26s} scale +/-{vmax:.3f}; rms "
              + ", ".join(f"{c.split('(')[0].strip()} {np.sqrt((g ** 2).mean()):.3f}"
                          for c, g in zip(COLUMNS, fields)))
    for label, (fields, _, _) in zip(("xco2", "flux"), rows):
        r = np.corrcoef(fields[0].ravel(), fields[1].ravel())[0, 1]
        print(f"corr(NPP term, Rh) in {label} anomalies: {r:+.3f}  "
              f"(variance of the sum / sum of variances: "
              f"{fields[2].var() / (fields[0].var() + fields[1].var()):.3f})")

    play = [[interpolate_frames(g, args.frames_per_month) for g in fields]
            for fields, _, _ in rows]
    stamps = interpolate_frames(np.arange(len(months))[:, None].astype(float),
                                args.frames_per_month).ravel()
    nframe = play[0][0].shape[0]

    tlon, tlat, nearest = make_mapper(lons, lats, args.resolution)
    fig = plt.figure(figsize=(16.8, 8.2))
    grid = fig.add_gridspec(2, 3, left=0.035, right=0.995, bottom=0.10, top=0.86,
                            wspace=0.03, hspace=0.12)
    images = []
    for r, (fields, vmax, unit) in enumerate(rows):
        for c in range(3):
            ax = fig.add_subplot(grid[r, c], projection=ccrs.Robinson())
            ax.set_global()
            ax.set_facecolor("0.92")
            ax.coastlines(linewidth=0.35, color="0.25")
            im = ax.pcolormesh(tlon, tlat,
                               play[r][c][0][nearest].reshape(tlon.shape),
                               transform=ccrs.PlateCarree(), shading="nearest",
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               rasterized=True)
            rms = np.sqrt((fields[c] ** 2).mean())
            ax.set_title(f"{COLUMNS[c]}   rms {rms:.2f}", fontsize=11,
                         fontweight="bold", pad=3, color=INK)
            images.append(im)
        cax = fig.add_axes([0.30, 0.525 - r * 0.475, 0.40, 0.020])
        fig.colorbar(images[3 * r], cax=cax, orientation="horizontal",
                     extend="both",
                     label=("column XCO2 anomaly (" + unit + ")") if r == 0
                     else ("surface flux anomaly (" + unit + ")"))

    out = args.outdir / "jules_flux_vs_xco2_anomaly.mp4"
    writer = FFMpegWriter(fps=args.fps, bitrate=4600,
                          metadata={"title": "JULES-ES flux vs XCO2 anomalies"})
    with writer.saving(fig, str(out), args.dpi):
        for k in range(nframe):
            for r in range(2):
                for c in range(3):
                    images[3 * r + c].set_array(
                        play[r][c][k][nearest].reshape(tlon.shape).ravel())
            fig.suptitle("JULES-ES column XCO2 anomaly (top) and the surface "
                         "flux anomaly that produced it (bottom), C30\n"
                         "detrended + deseasonalised, fire excluded -- "
                         f"{months[int(round(stamps[k]))]}",
                         fontsize=13, fontweight="bold", y=0.985, color=INK)
            writer.grab_frame()
            if k % 100 == 0:
                print(f"  frame {k}/{nframe}", flush=True)
    print(f"wrote {out} ({nframe} frames, {nframe / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
