#!/usr/bin/env python3
"""Hovmoller of the SIF-derived GPP driver itself (the model input).

Three stacked panels on equal-area sin(latitude) bands, sharing a time axis:

  1. raw               total carbon uptake per band
  2. mean seasonal cycle   the day-of-year climatology computed from the data,
                           tiled across years -- i.e. exactly what panel 3
                           subtracts
  3. deseasonalised    panel 1 minus panel 2

By default the vertical unit is **total C uptake per band** (Pg C/yr), not flux
density, so a band's value reflects how much land it actually contains. Land
fraction per band ranges from 0.2% (the Southern Ocean near 51 S) to 75%, so
this is a very different quantity from a per-area mean even though the equal-area
binning makes the two similar in shape (band areas vary by only 2.8% rms).

The seasonal cycle is empirical by default -- a day-of-year climatology averaged
over complete years and lightly smoothed -- rather than a harmonic fit. Four
harmonics cannot represent the sharply non-sinusoidal boreal growing season, and
the residual aliases into the anomaly as horizontal striping poleward of ~60 N.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "plotmod", os.path.join(_HERE, "plot_sif_gpp_iav_xco2.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", [
    "#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])

KGCO2_M2_S_TO_GC_M2_DAY = 86400.0 * (12.0 / 44.0) * 1e3
DAYS_PER_YEAR = 365.25


def doy_key(d):
    """(month, day) key with 29 Feb folded onto 28 Feb."""
    return d.month * 100 + (28 if (d.month == 2 and d.day == 29) else d.day)


def empirical_seasonal_cycle(hov, dates, years, smooth=31):
    """Day-of-year climatology from complete years, circularly smoothed."""
    keys = np.array([doy_key(d) for d in dates])
    yr = np.array([d.year for d in dates])
    ndays = {Y: int((yr == Y).sum()) for Y in set(yr)}
    use = np.array([ndays[y] >= 350 for y in yr])
    if use.sum() < 365:
        use = np.ones(len(dates), bool)
    uniq = np.unique(keys)
    idx_of = {k: i for i, k in enumerate(uniq)}
    clim = np.zeros((hov.shape[0], len(uniq)))
    cnt = np.zeros(len(uniq))
    for i in np.where(use)[0]:
        j = idx_of[keys[i]]
        clim[:, j] += hov[:, i]
        cnt[j] += 1
    assert cnt.min() > 0, "empty day-of-year slot"
    clim /= cnt[None, :]
    if smooth > 1:                      # circular running mean over day-of-year
        n = clim.shape[1]
        pad = np.concatenate([clim[:, n - smooth // 2:], clim,
                              clim[:, :smooth - smooth // 2 - 1]], axis=1)
        kern = np.ones(smooth) / smooth
        clim = np.vstack([np.convolve(r, kern, mode="valid") for r in pad])
    tiled = np.stack([clim[:, idx_of[k]] for k in keys], axis=1)
    return tiled, int(cnt.min()), int(cnt.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux", default="/temp1/cfranken/sif_gpp_iav/fluxes/"
                                      "sif_gpp_daily_co2flux_c90_2018_2025.nc")
    ap.add_argument("--var", default="GPP_CO2_FLUX")
    ap.add_argument("--outdir", default="plots/sif_gpp_iav")
    ap.add_argument("--nband", type=int, default=40)
    ap.add_argument("--mode", choices=("total", "density"), default="total",
                    help="total = Pg C/yr summed per band (default); "
                         "density = g C m-2 d-1 per total cell area")
    ap.add_argument("--season", choices=("empirical", "harmonic"),
                    default="empirical")
    ap.add_argument("--clim-smooth", type=int, default=31,
                    help="circular smoothing of the day-of-year climatology")
    ap.add_argument("--smooth-days", type=int, default=31,
                    help="running mean on the anomaly panel; daily SIF "
                         "retrieval noise exceeds the interannual signal")
    ap.add_argument("--detrend", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ds = nc.Dataset(args.flux)
    lats = np.asarray(ds.variables["lats"][:], np.float64)
    area = np.asarray(ds.variables["cell_area"][:], np.float64)
    lf = np.asarray(ds.variables["land_fraction"][:], np.float64)
    tv = ds.variables["time"]
    t0 = dt.datetime.fromisoformat(tv.units.split("since")[1].strip().split()[0])
    dates = [t0 + dt.timedelta(hours=float(h)) for h in np.asarray(tv[:], float)]
    nt = len(dates)

    edges = np.linspace(-1.0, 1.0, args.nband + 1)
    zb = np.clip(np.digitize(np.sin(np.deg2rad(lats.ravel())), edges) - 1,
                 0, args.nband - 1)
    aw = area.ravel()
    band_area = np.bincount(zb, weights=aw, minlength=args.nband)
    band_land = np.bincount(zb, weights=(lf * area).ravel(), minlength=args.nband)
    assert (band_area > 0).all(), "empty sin(lat) band"

    v = ds.variables[args.var]
    hov = np.empty((args.nband, nt))
    for s in range(0, nt, 200):
        e = min(nt, s + 200)
        blk = v[s:e]
        if np.ma.isMaskedArray(blk):
            blk = blk.filled(0.0)
        gpp = -np.asarray(blk, np.float64) * KGCO2_M2_S_TO_GC_M2_DAY
        for i in range(e - s):
            tot = np.bincount(zb, weights=gpp[i].ravel() * aw,
                              minlength=args.nband)     # g C / day per band
            hov[:, s + i] = tot
    ds.close()

    if args.mode == "total":
        hov *= DAYS_PER_YEAR / 1e15        # -> Pg C / yr per band
        unit, unit_a = "Pg C yr$^{-1}$ per band", "Pg C yr$^{-1}$ anomaly"
    else:
        hov /= band_area[:, None]          # -> g C m-2 d-1
        unit = unit_a = "g C m$^{-2}$ d$^{-1}$"

    years = np.array([d.year for d in dates])
    t = mdates.date2num(dates)
    if args.season == "empirical":
        clim, cmin, cmax = empirical_seasonal_cycle(hov, dates, years,
                                                    args.clim_smooth)
        print(f"[season] empirical day-of-year climatology from complete years, "
              f"{cmin}-{cmax} samples/slot, {args.clim_smooth}-day circular smooth")
    else:
        clim = hov - P._deseason(hov, dates)
        print("[season] 4-harmonic fit")
    des = hov - clim
    if args.detrend:
        des = P._detrend(des, t)

    raw_rms = des.std()
    if args.smooth_days > 1:
        k = args.smooth_days
        pad = k // 2
        padded = np.pad(des, ((0, 0), (pad, k - pad - 1)), mode="edge")
        kern = np.ones(k) / k
        des = np.vstack([np.convolve(r, kern, mode="valid") for r in padded])
        print(f"[smooth] {k}-day running mean on the anomaly: rms "
              f"{raw_rms:.4f} -> {des.std():.4f} ({100 * des.std() / raw_rms:.0f}% "
              "kept; the rest was daily retrieval noise)")

    print(f"[input] {nt} days, {args.nband} equal-area bands, mode={args.mode}")
    print(f"  band area spread {100 * band_area.std() / band_area.mean():.2f}% rms; "
          f"land fraction {100 * (band_land / band_area).min():.1f}%"
          f"..{100 * (band_land / band_area).max():.1f}%")
    if args.mode == "total":
        print(f"  record-mean total over all bands: {hov.sum(axis=0).mean():.1f} Pg C/yr")
    print(f"  raw range {hov.min():.4f} .. {hov.max():.4f} {unit}")
    print(f"  anomaly rms {des.std():.4f}, range {des.min():+.4f} .. {des.max():+.4f}")

    tedge = np.append(t, t[-1] + (t[-1] - t[-2]))
    ticks = np.array([-90, -60, -30, 0, 30, 60, 90])
    fig, axs = plt.subplots(3, 1, figsize=(12.5, 11.2), sharex=True,
                            constrained_layout=True)
    vmax = float(np.nanpercentile(hov, 99.5))

    im1 = axs[0].pcolormesh(tedge, edges, hov, cmap=SEQUENTIAL, vmin=0.0,
                            vmax=vmax, shading="flat")
    axs[0].set_title(f"Raw — total SIF-derived GPP per latitude band "
                     f"(peak {hov.max():.2f} {unit.split(' per')[0]})",
                     color=P.INK, fontsize=11, loc="left", fontweight="bold")
    im2 = axs[1].pcolormesh(tedge, edges, clim, cmap=SEQUENTIAL, vmin=0.0,
                            vmax=vmax, shading="flat")
    axs[1].set_title("Mean seasonal cycle — day-of-year climatology computed "
                     "from the data itself, tiled across years",
                     color=P.INK, fontsize=11, loc="left", fontweight="bold")
    vv = max(float(np.nanpercentile(np.abs(des), 99)), 1e-12)
    lab = "Anomaly (deseasonalised and detrended)" if args.detrend else \
          "Anomaly (deseasonalised; long-term trend retained)"
    sm = f", {args.smooth_days}-day running mean" if args.smooth_days > 1 else ""
    im3 = axs[2].pcolormesh(tedge, edges, des, cmap=P.DIVERGING,
                            norm=TwoSlopeNorm(0.0, -vv, vv), shading="flat")
    axs[2].set_title(f"{lab} — panel 1 minus panel 2{sm}",
                     color=P.INK, fontsize=11, loc="left", fontweight="bold")

    for ax, im, lb in ((axs[0], im1, unit), (axs[1], im2, unit),
                       (axs[2], im3, unit_a)):
        ax.set_yticks(np.sin(np.deg2rad(ticks)))
        ax.set_yticklabels([f"{x}°" for x in ticks])
        ax.set_ylabel("latitude (equal-area)", color=P.INK, fontsize=10)
        ax.tick_params(colors=P.INK2, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(P.GRID)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(lb, color=P.INK, fontsize=9)
        cb.ax.tick_params(colors=P.INK2, labelsize=8)
    P.date_axis(axs[2])
    what = ("total carbon uptake per band"
            if args.mode == "total" else "flux density")
    fig.suptitle(f"Model input: zonal SIF-derived GPP, 2018-2025 — {what}\n"
                 "equal-area sin(latitude) bands",
                 color=P.INK, fontsize=12.5, fontweight="bold")
    tag = f"_{args.mode}" + ("_detrended" if args.detrend else "")
    f = f"{args.outdir}/sif_gpp_input_hovmoller{tag}.png"
    fig.savefig(f, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
