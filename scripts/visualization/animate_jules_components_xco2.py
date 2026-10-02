#!/usr/bin/env python3
"""Animate the JULES-ES component XCO2 fields against GFED5 fire, on C30.

Three panels, all on one colour scale so their magnitudes are comparable:

  GPP - Ra    the NPP term, transported as the sum of the gpp and ra tracers
              (the gpp tracer already carries the uptake sign)
  Rh          heterotrophic respiration
  GFED5 fire  the standalone observed-fire tracer

Two movies:

  --mode absolute   each component's spatial XCO2 pattern with the global mean
                    removed at every step. Keeps the full seasonal march; the
                    global mean is dropped only because each tracer accumulates
                    a large secular ramp (GPP-Ra to -400 ppm, Rh to +360, fire
                    to +16) that would otherwise swamp the colour scale.
  --mode anomaly    detrended and deseasonalised per grid cell, i.e. the
                    interannual signal only.

Input is monthly-mean gridded column XCO2; frames are interpolated between
monthly midpoints purely so the movie plays smoothly.
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
from scipy.spatial import cKDTree

PANELS = (("GPP - Ra  (NPP term)", ("co2_jules_es_gpp", "co2_jules_es_ra")),
          ("Rh", ("co2_jules_es_rh",)),
          ("GFED5 fire", ("co2_gfed_fire",)),
          ("NEE + fire  (total)", None))          # exact sum of the three above
INK = "#1f2933"


def xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    rlon, rlat = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack([np.cos(rlat) * np.cos(rlon),
                            np.cos(rlat) * np.sin(rlon), np.sin(rlat)])


def make_mapper(lons: np.ndarray, lats: np.ndarray, resolution: float):
    lon_c = np.arange(-180.0 + resolution / 2, 180.0, resolution)
    lat_c = np.arange(-90.0 + resolution / 2, 90.0, resolution)
    tlon, tlat = np.meshgrid(lon_c, lat_c)
    nearest = cKDTree(xyz(lons.ravel(), lats.ravel())).query(
        xyz(tlon.ravel(), tlat.ravel()), k=1)[1]
    return tlon, tlat, nearest


def backfit(x: np.ndarray, iters: int = 5) -> np.ndarray:
    """Joint linear trend + mean seasonal cycle removal along the month axis.

    Joint rather than sequential: with a seasonal cycle this large, detrending
    first leaks season into the trend.
    """
    n = x.shape[0]
    t = np.arange(n, dtype=float)
    t -= t.mean()
    season = np.zeros_like(x)
    for _ in range(iters):
        resid = x - season
        slope = (resid * t[:, None]).sum(axis=0) / (t ** 2).sum()
        trend = t[:, None] * slope
        resid = x - trend
        season = np.concatenate([resid[m::12].mean(axis=0)[None] for m in range(12)])
        season = np.tile(season, (n // 12, 1))
    out = x - trend - season
    return out - out.mean(axis=0)


def interpolate_frames(monthly: np.ndarray, per_month: int) -> np.ndarray:
    """Linear interpolation between monthly midpoints, for playback only."""
    n = monthly.shape[0]
    src = np.arange(n, dtype=float)
    dst = np.linspace(0, n - 1, (n - 1) * per_month + 1)
    lo = np.clip(np.floor(dst).astype(int), 0, n - 2)
    w = (dst - src[lo])[:, None]
    return monthly[lo] * (1 - w) + monthly[lo + 1] * w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots/sif_gpp_iav"))
    ap.add_argument("--mode", choices=("absolute", "anomaly"), required=True)
    ap.add_argument("--only", default=None,
                    help="comma-separated panel names to keep, e.g. 'npp,rh'; "
                         "default shows all four")
    ap.add_argument("--suffix", default="", help="appended to the output filename")
    ap.add_argument("--resolution", type=float, default=1.5)
    ap.add_argument("--frames-per-month", type=int, default=6)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    keep = None if args.only is None else [k.strip() for k in args.only.split(",")]
    z = np.load(args.scratch / "c30_maps_jules_fire.npz", allow_pickle=True)
    months, lons, lats, area = z["months"], z["lons"], z["lats"], z["area"]
    w = area.ravel() / area.sum()

    fields = []
    for _, keys in PANELS:
        if keys is None:
            fields.append(None)          # NEE, filled once the panels are chosen
            continue
        f = sum(z[k] for k in keys)
        if args.mode == "absolute":
            f = f - (f * w[None, :]).sum(axis=1, keepdims=True)   # spatial pattern
        else:
            f = backfit(f)
        fields.append(f)

    alias = {"npp": 0, "gpp-ra": 0, "rh": 1, "fire": 2, "nee": 3}
    pick = list(range(4)) if keep is None else [alias[k] for k in keep]
    # Both processings are linear, so NEE is the exact sum of the components on
    # display; with fire dropped it is the fire-free NEE, and it is labelled so.
    shown = [i for i in pick if i != 3]
    with_fire = 2 in shown
    fields[3] = sum(fields[i] for i in (shown if shown else [0, 1, 2]))
    panels = [(("NEE + fire  (total)" if with_fire else "NEE  (no fire)")
               if i == 3 else PANELS[i][0], None) for i in pick]
    fields = [fields[i] for i in pick]
    fire_index = pick.index(2) if 2 in pick else None

    # The two land-flux terms share a scale because they are directly
    # comparable and largely compensate. Fire is 15-30x smaller, so forcing it
    # onto that scale would render it blank; it gets its own colour bar, with
    # the ratio stated on the panel.
    land = [f for i, f in enumerate(fields) if i != fire_index]
    vmax = float(np.percentile(np.abs(np.concatenate(
        [f.ravel() for f in land])), 99.5))
    vfire = (float(np.percentile(np.abs(fields[fire_index]), 99.5))
             if fire_index is not None else vmax)
    scales = [vfire if i == fire_index else vmax for i in range(len(fields))]
    print(f"{args.mode}: land scale +/-{vmax:.2f} ppm, fire scale "
          f"+/-{vfire:.2f} ppm (1/{vmax / vfire:.0f}); per-panel rms "
          + ", ".join(f"{t.split('(')[0].strip()} {np.sqrt((f ** 2).mean()):.3f}"
                      for (t, _), f in zip(panels, fields)))

    play = [interpolate_frames(f, args.frames_per_month) for f in fields]
    stamps = interpolate_frames(np.arange(len(months))[:, None].astype(float),
                                args.frames_per_month).ravel()
    nframe = play[0].shape[0]

    tlon, tlat, nearest = make_mapper(lons, lats, args.resolution)
    n = len(fields)
    if n <= 3:
        fig = plt.figure(figsize=(5.5 * n + 0.6, 5.6))
        grid = fig.add_gridspec(1, n, left=0.02, right=0.98, bottom=0.16,
                                top=0.80, wspace=0.04)
    else:
        fig = plt.figure(figsize=(13.6, 8.6))
        grid = fig.add_gridspec(2, 2, left=0.02, right=0.98, bottom=0.115,
                                top=0.87, wspace=0.04, hspace=0.10)
    images = []
    for i, (title, _) in enumerate(panels):
        ax = fig.add_subplot(grid[i], projection=ccrs.Robinson())
        ax.set_global()
        ax.set_facecolor("0.92")
        ax.coastlines(linewidth=0.35, color="0.25")
        im = ax.pcolormesh(tlon, tlat, play[i][0][nearest].reshape(tlon.shape),
                           transform=ccrs.PlateCarree(), shading="nearest",
                           cmap="RdBu_r", vmin=-scales[i], vmax=scales[i],
                           rasterized=True)
        rms = np.sqrt((fields[i] ** 2).mean())
        extra = (f"   [scale 1/{vmax / vfire:.0f}]" if i == fire_index else "")
        ax.set_title(f"{title}   rms {rms:.2f} ppm{extra}", fontsize=11.5,
                     fontweight="bold", pad=4, color=INK)
        images.append(im)
    if fire_index is None:
        cax = fig.add_axes([0.32, 0.075, 0.36, 0.026])
        fig.colorbar(images[0], cax=cax, orientation="horizontal", extend="both",
                     label="XCO2 contribution (ppm)")
    else:
        cax = fig.add_axes([0.12, 0.065, 0.36, 0.022])
        fig.colorbar(images[0], cax=cax, orientation="horizontal", extend="both",
                     label="XCO2 contribution, land terms and total (ppm)")
        cax2 = fig.add_axes([0.64, 0.065, 0.24, 0.022])
        fig.colorbar(images[fire_index], cax=cax2, orientation="horizontal",
                     extend="both", label="XCO2 contribution, fire (ppm)")
    kind = ("spatial pattern, global mean removed, seasonal cycle retained"
            if args.mode == "absolute" else "detrended + deseasonalised anomaly")
    fig.suptitle("", fontsize=14, fontweight="bold", y=0.965, color=INK)

    out = args.outdir / f"jules_components_xco2_{args.mode}{args.suffix}.mp4"
    writer = FFMpegWriter(fps=args.fps, bitrate=4200,
                          metadata={"title": f"JULES-ES components, {kind}"})
    with writer.saving(fig, str(out), args.dpi):
        for k in range(nframe):
            for i, im in enumerate(images):
                im.set_array(play[i][k][nearest].reshape(tlon.shape).ravel())
            label = str(months[int(round(stamps[k]))])
            fig.suptitle(f"JULES-ES transported XCO2 vs GFED5 fire, C30 -- {kind}"
                         f"\n{label}", fontsize=13, fontweight="bold", y=0.98,
                         color=INK)
            writer.grab_frame()
            if k % 100 == 0:
                print(f"  frame {k}/{nframe} {label}", flush=True)
    print(f"wrote {out} ({nframe} frames, {nframe / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
