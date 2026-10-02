#!/usr/bin/env python3
"""Figures for the SIF-GPP interannual-variability XCO2 experiment.

Consumes the time-series NetCDF written by analyze_sif_gpp_iav_xco2.py.
Sign convention: positive XCO2 anomaly = anomalously WEAK GPP uptake.
"""
from __future__ import annotations

import argparse
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Validated categorical slots (blue, orange) and the blue<->red diverging pair
# with a neutral gray midpoint.
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [
    "#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#b7d3f6",
    "#f0efec",
    "#f4b3b3", "#e87b7b", "#e34948", "#c72f2f", "#8f1f1f",
])


def date_axis(ax):
    """Robust for spans from weeks to a decade."""
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def to_latlon(field, lats, lons, area, dlat=3.0, dlon=3.0):
    """Area-weighted binning of C90 cells onto a regular lat-lon grid."""
    nlat, nlon = int(180 / dlat), int(360 / dlon)
    la = np.clip(((lats.ravel() + 90) / dlat).astype(int), 0, nlat - 1)
    lo = np.clip((((lons.ravel() + 180) % 360) / dlon).astype(int), 0, nlon - 1)
    idx = la * nlon + lo
    w = area.ravel()
    num = np.zeros(nlat * nlon)
    den = np.zeros(nlat * nlon)
    np.add.at(num, idx, field.ravel() * w)
    np.add.at(den, idx, w)
    out = np.divide(num, den, out=np.full(nlat * nlon, np.nan),
                    where=den > 0).reshape(nlat, nlon)
    # Near the poles a 3-degree longitude bin can contain no C90 cell centre
    # (~6% of bins, all poleward of 72 deg), which would render as speckle.
    # Fill each such bin from the nearest filled bin in its own latitude row.
    for r in range(nlat):
        row = out[r]
        good = np.flatnonzero(np.isfinite(row))
        if good.size == 0 or good.size == nlon:
            continue
        bad = np.flatnonzero(~np.isfinite(row))
        # circular nearest neighbour in longitude
        d = np.abs(bad[:, None] - good[None, :])
        d = np.minimum(d, nlon - d)
        row[bad] = row[good[np.argmin(d, axis=1)]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="/temp1/cfranken/sif_gpp_iav/"
                                    "sif_gpp_iav_c90_xco2_timeseries.nc")
    # Default into the repo's gitignored /plots tree so the figures are visible
    # in the editor without polluting `git status`.
    ap.add_argument("--outdir", default="plots/sif_gpp_iav")
    args = ap.parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)

    ds = nc.Dataset(args.ts)
    tu = ds.variables["time"].units
    t0 = dt.datetime.fromisoformat(tu.split("since")[1].strip().split()[0])
    hours = np.asarray(ds.variables["time"][:], float)
    dates = np.array([t0 + dt.timedelta(hours=float(h)) for h in hours])
    area = np.asarray(ds.variables["cell_area"][:], np.float64)
    lats = np.asarray(ds.variables["lats"][:], np.float64)
    lons = np.asarray(ds.variables["lons"][:], np.float64)
    anom = np.asarray(ds.variables["xco2_anom"][:], np.float64)
    has_full = "xco2_full" in ds.variables
    full = np.asarray(ds.variables["xco2_full"][:], np.float64) if has_full else None
    ds.close()

    w = area / area.sum()
    g_anom = np.einsum("tfyx,fyx->t", anom, w)
    years = np.array([d.year for d in dates])

    # ---- Figure 1: global-mean time series ---------------------------------
    nrow = 2 if has_full else 1
    fig, axs = plt.subplots(nrow, 1, figsize=(11, 3.2 * nrow), sharex=True,
                            constrained_layout=True)
    axs = np.atleast_1d(axs)
    ax = axs[0]
    style(ax)
    ax.axhline(0, color=INK2, lw=1.0, zorder=1)
    ax.plot(dates, g_anom, color=C_BLUE, lw=2.0, zorder=3)
    ann = np.array([g_anom[years == Y].mean() for Y in sorted(set(years))])
    for Y, v in zip(sorted(set(years)), ann):
        m = years == Y
        ax.plot([dates[m][0], dates[m][-1]], [v, v], color=C_ORANGE, lw=2.0, zorder=4)
        ax.annotate(f"{v:+.2f}", (dates[m][len(dates[m]) // 2], v),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_ylabel("XCO$_2$ anomaly (ppm)", color=INK, fontsize=10)
    ax.set_title("Global-mean XCO$_2$ response to interannual variability in "
                 "SIF-derived GPP\npositive = anomalously weak uptake; "
                 "orange = annual mean",
                 color=INK, fontsize=11.5, loc="left", fontweight="bold")
    if has_full:
        g_full = np.einsum("tfyx,fyx->t", full, w)
        ax2 = axs[1]
        style(ax2)
        ax2.plot(dates, g_full, color=C_BLUE, lw=2.0)
        ax2.set_ylabel("XCO$_2$ drawdown (ppm)", color=INK, fontsize=10)
        ax2.set_title("Full SIF-GPP uptake tracer (undetrended): cumulative "
                      "drawdown, no respiration",
                      color=INK, fontsize=11.5, loc="left", fontweight="bold")
        date_axis(ax2)
    date_axis(axs[-1])
    f1 = f"{args.outdir}/sif_gpp_iav_global_timeseries.png"
    fig.savefig(f1, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("wrote", f1)

    # ---- Figure 2: annual-mean anomaly maps -------------------------------
    ys = [Y for Y in sorted(set(years)) if (years == Y).sum() > 300]
    if not ys:                      # partial record: fall back to whatever exists
        ys = sorted(set(years))
    n = len(ys)
    ncol = min(3, n)
    nr = max(1, int(np.ceil(n / ncol)))
    # The global mean (2-3.5 ppm) dwarfs the spatial structure (~0.3 ppm), so a
    # map of the raw field is uniformly one colour and shows nothing. Plot each
    # year's DEPARTURE from its own global mean: that has genuine polarity, and
    # the global mean itself is in the panel title and in Figure 1.
    vmax = 0.0
    maps, gmeans = {}, {}
    for Y in ys:
        fld = np.nanmean(anom[years == Y].astype(np.float64), axis=0)
        gmeans[Y] = float(np.einsum("fyx,fyx->", np.nan_to_num(fld), w))
        m = to_latlon(fld - gmeans[Y], lats, lons, area)
        maps[Y] = m
        vmax = max(vmax, np.nanpercentile(np.abs(m), 99))
    fig, axs = plt.subplots(nr, ncol, figsize=(max(9.5, 4.4 * ncol), 2.5 * nr + 0.5),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    vmax = max(float(vmax), 1e-6)          # TwoSlopeNorm rejects a zero range
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    for i, Y in enumerate(ys):
        ax = axs[i]
        im = ax.pcolormesh(np.linspace(-180, 180, maps[Y].shape[1] + 1),
                           np.linspace(-90, 90, maps[Y].shape[0] + 1),
                           maps[Y], cmap=DIVERGING, norm=norm, shading="flat")
        ax.set_title(f"{Y}   global mean {gmeans[Y]:+.2f} ppm",
                     fontsize=10, color=INK, loc="left")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-60, 0, 60])
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
    for j in range(n, len(axs)):
        axs[j].axis("off")
    cb = fig.colorbar(im, ax=axs.tolist(), shrink=0.8, pad=0.01)
    cb.set_label("departure from that year's global mean (ppm)",
                 color=INK, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle("Spatial structure of the SIF-GPP XCO$_2$ anomaly, shown as the "
                 "departure from each year's global mean\n"
                 "(red = locally more CO$_2$ than the global mean, i.e. locally "
                 "weaker uptake)",
                 color=INK, fontsize=11.5, fontweight="bold")
    f2 = f"{args.outdir}/sif_gpp_iav_annual_maps.png"
    fig.savefig(f2, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("wrote", f2)

    # ---- Figure 3: zonal-mean Hovmoller ----------------------------------
    edges = np.arange(-90, 91, 5.0)
    zb = np.clip(np.digitize(lats.ravel(), edges) - 1, 0, len(edges) - 2)
    aw = area.ravel()
    den = np.bincount(zb, weights=aw, minlength=len(edges) - 1)
    # Again: subtract the contemporaneous global mean so the meridional
    # structure is visible and the diverging map encodes real polarity.
    hov = np.empty((len(edges) - 1, anom.shape[0]))
    for i in range(anom.shape[0]):
        fld = anom[i].astype(np.float64)
        gmu = float(np.einsum("fyx,fyx->", np.nan_to_num(fld), w))
        num = np.bincount(zb, weights=(fld.ravel() - gmu) * aw,
                          minlength=len(edges) - 1)
        hov[:, i] = num / den
    v = max(float(np.nanpercentile(np.abs(hov), 99)), 1e-6)
    fig, ax = plt.subplots(figsize=(11, 4.0), constrained_layout=True)
    # shading="flat" needs cell edges: one more time point than columns.
    tnum = mdates.date2num(dates)
    tedge = np.append(tnum, tnum[-1] + (tnum[-1] - tnum[-2] if len(tnum) > 1 else 1.0))
    im = ax.pcolormesh(tedge, edges, hov, cmap=DIVERGING,
                       norm=TwoSlopeNorm(0.0, -v, v), shading="flat")
    ax.xaxis_date()
    ax.set_ylabel("latitude", color=INK, fontsize=10)
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.tick_params(colors=INK2, labelsize=9)
    date_axis(ax)
    ax.set_title("Meridional structure of the SIF-GPP XCO$_2$ anomaly\n"
                 "zonal mean minus the global mean at each time (ppm)",
                 color=INK, fontsize=11.5, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("ppm", color=INK, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    f3 = f"{args.outdir}/sif_gpp_iav_zonal_hovmoller.png"
    fig.savefig(f3, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("wrote", f3)

    # ---- Figure 4: equal-area sin(lat) Hovmollers --------------------------
    hovmoller_sinlat(dates, anom, lats, area, args.outdir, tracer="anom")
    if has_full:
        # The "run the raw sink, detrend afterwards" route. Its structure should
        # match the anomaly tracer -- the linearity check, made visible.
        hovmoller_sinlat(dates, full, lats, area, args.outdir, tracer="full")


def _sinlat_bands(anom, lats, area, nband=40):
    """Zonal means on equal-area sin(latitude) bands -> (nband, ntime), edges."""
    edges = np.linspace(-1.0, 1.0, nband + 1)
    sl = np.sin(np.deg2rad(lats.ravel()))
    zb = np.clip(np.digitize(sl, edges) - 1, 0, nband - 1)
    aw = area.ravel()
    den = np.bincount(zb, weights=aw, minlength=nband)
    assert (den > 0).all(), "empty sin(lat) band"
    hov = np.empty((nband, anom.shape[0]))
    for i in range(anom.shape[0]):
        fld = np.nan_to_num(anom[i].astype(np.float64)).ravel()
        hov[:, i] = np.bincount(zb, weights=fld * aw, minlength=nband) / den
    return hov, edges


def _detrend(hov, t):
    """Remove a least-squares line in time from every band independently."""
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, hov.T, rcond=None)
    return hov - (A @ coef).T


def _deseason(hov, dates, nharm=4):
    """Remove each band's mean seasonal cycle via a harmonic fit.

    A day-of-year climatology built from only 7-8 samples per day injects its own
    sampling noise, which shows up as vertical striping. Fitting `nharm`
    harmonics of the annual cycle (the standard approach for CO2 curve fitting)
    gives a smooth seasonal cycle instead. What remains is the interannual
    signal.
    """
    t = np.array([(d - dates[0]).total_seconds() / 86400.0 for d in dates])
    cols = [np.ones_like(t)]
    for k in range(1, nharm + 1):
        w = 2 * np.pi * k * t / 365.2425
        cols += [np.sin(w), np.cos(w)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, hov.T, rcond=None)
    return hov - (A @ coef).T


def _panel(ax, tedge, edges, hov, title):
    v = max(float(np.nanpercentile(np.abs(hov), 99)), 1e-6)
    im = ax.pcolormesh(tedge, edges, hov, cmap=DIVERGING,
                       norm=TwoSlopeNorm(0.0, -v, v), shading="flat")
    ticks = np.array([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticks(np.sin(np.deg2rad(ticks)))
    ax.set_yticklabels([f"{x}\u00b0" for x in ticks])
    ax.set_ylabel("latitude (equal-area)", color=INK, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    return im


def hovmoller_sinlat(dates, anom, lats, area, outdir, nband=40, tracer="anom"):
    """Two stacked equal-area Hovmollers: detrended, and also deseasonalised."""
    hov, edges = _sinlat_bands(anom, lats, area, nband)
    t = mdates.date2num(dates)
    tedge = np.append(t, t[-1] + (t[-1] - t[-2] if len(t) > 1 else 1.0))

    det = _detrend(hov, t)
    des = _deseason(det, dates)

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True,
                            constrained_layout=True)
    im1 = _panel(axs[0], tedge, edges, det,
                 "Detrended \u2014 linear trend in time removed per band "
                 f"(\u00b1{np.nanpercentile(np.abs(det), 99):.2f} ppm)")
    im2 = _panel(axs[1], tedge, edges, des,
                 "Detrended and deseasonalised \u2014 mean seasonal cycle also "
                 f"removed (\u00b1{np.nanpercentile(np.abs(des), 99):.2f} ppm)")
    for im, ax in ((im1, axs[0]), (im2, axs[1])):
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("ppm", color=INK, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8)
    date_axis(axs[1])
    fig.suptitle("Zonal-mean XCO$_2$ anomaly from interannual variability in "
                 "SIF-derived GPP\n"
                 "equal-area sin(latitude) bands; red = anomalously weak uptake",
                 color=INK, fontsize=12.5, fontweight="bold")
    f = f"{outdir}/sif_gpp_iav_hovmoller_sinlat_{tracer}.png"
    fig.savefig(f, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("wrote", f)
    return det, des, edges


if __name__ == "__main__":
    main()
