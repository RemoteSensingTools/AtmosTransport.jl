#!/usr/bin/env python3
"""Separate flux-driven from transport-driven XCO2 interannual variability.

Compares the flux-driven anomaly tracer (`xco2_anom`, from
config/runs/sif_gpp_iav_c90_2018_2025.toml) against the transport-only control
tracer (`co2_gpp_clim`, from config/runs/sif_gpp_clim_control_c90_2018_2025.toml).

The control is driven by the day-of-year GPP climatology with each cell's annual
mean removed, so its forcing is byte-identical in every year. Any interannual
signal it carries is therefore produced by transport alone, and it sets the
floor below which zonal XCO2 interannual variability cannot be attributed to
GPP.

Do NOT substitute `xco2_full - xco2_anom` for this: differencing two Float32
tracers that carry a 900 ppm background injects noise comparable to the signal,
and the two respond to different fluxes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import importlib.util
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "plotmod", os.path.join(_HERE, "plot_sif_gpp_iav_xco2.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

_ispec = importlib.util.spec_from_file_location(
    "inpmod", os.path.join(_HERE, "plot_sif_gpp_input_hovmoller.py"))
I = importlib.util.module_from_spec(_ispec)
_ispec.loader.exec_module(I)


def _doy_key(dates):
    return np.array([d.month * 100 + (28 if (d.month == 2 and d.day == 29)
                                      else d.day) for d in dates])


def interannual(x, dates):
    """Isolate interannual variability: per-day-of-year linear detrend.

    For each calendar day, fit and remove a straight line across years. This
    removes the mean seasonal cycle AND any slow evolution of that cycle in one
    step.

    Do NOT detrend the whole series first and deseasonalise afterwards. These
    tracers carry a seasonal cycle far larger than their trend (the control's
    seasonal std is 2.55 ppm against a true drift of +0.009 ppm/yr), and the
    seasonal cycle is not exactly orthogonal to a linear basis over a finite
    window: a global linear fit to the control picks up a spurious
    -0.118 ppm/yr, and removing it leaves a ~0.25 ppm sawtooth in the
    day-of-year residual that looks like interannual variability but is not.

    Removing a slow evolution matters here for a second reason: the control
    starts from a uniform field, and its seasonal cycle is still equilibrating
    at the end of the record (the stratosphere fills slowly), which is residual
    spin-up rather than variability.
    """
    key = _doy_key(dates)
    years = np.array([d.year for d in dates], float)
    out = np.array(x, np.float64, copy=True)
    for q in np.unique(key):
        m = np.flatnonzero(key == q)
        yy = years[m] - years[m].mean()
        A = np.vstack([yy, np.ones_like(yy)]).T
        coef, *_ = np.linalg.lstsq(A, np.asarray(x)[..., m].T, rcond=None)
        out[..., m] = np.asarray(x)[..., m] - (A @ coef).T
    return out

DATE_RE = re.compile(r"(\d{8})\.nc$")


def load_daily(outdir, varname, carrier_ppm):
    files = sorted(f for f in glob.glob(os.path.join(outdir, "*.nc"))
                   if DATE_RE.search(f))
    dates, out, bad = [], [], 0
    for f in files:
        try:
            d = nc.Dataset(f)
        except OSError:
            bad += 1
            continue
        v = d.variables.get(varname)
        if v is None:
            d.close()
            bad += 1
            continue
        a = v[:]
        a = np.ma.filled(a, np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
        a = np.asarray(a, np.float64)
        for att in ("_FillValue", "missing_value"):
            if att in v.ncattrs():
                fv = float(v.getncattr(att))
                if np.isfinite(fv):
                    a[a == fv] = np.nan
        if a.ndim == 4:
            a = a[-1]
        s = DATE_RE.search(f).group(1)
        dates.append(dt.datetime(int(s[:4]), int(s[4:6]), int(s[6:8])))
        out.append(a * 1e6 - carrier_ppm)
        d.close()
    if bad:
        print(f"[warn] skipped {bad} unreadable/incomplete file(s) in {outdir}")
    return dates, np.array(out, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="/temp1/cfranken/sif_gpp_iav/"
                                    "sif_gpp_iav_c90_xco2_timeseries.nc")
    ap.add_argument("--control", default="/temp1/cfranken/sif_gpp_iav/output/"
                                         "sif_gpp_clim_c90")
    ap.add_argument("--control-var", default="co2_gpp_clim_column_mean")
    ap.add_argument("--control-carrier", type=float, default=300.0)
    ap.add_argument("--outdir", default="plots/sif_gpp_iav")
    ap.add_argument("--start", default="2019-01-01",
                    help="drop everything before this date. The control tracer "
                         "starts from a uniform field and needs roughly a year "
                         "to build its equilibrium seasonal cycle; including "
                         "that transient inflates its apparent interannual "
                         "variability.")
    args = ap.parse_args()
    start = dt.datetime.fromisoformat(args.start)
    os.makedirs(args.outdir, exist_ok=True)

    cdates, ctrl = load_daily(args.control, args.control_var, args.control_carrier)
    if len(cdates) < 400:
        raise SystemExit(f"only {len(cdates)} control days available; run incomplete")
    # The harmonic deseasonalisation needs several full annual cycles to be
    # identifiable; anything shorter reports fit residuals, not variability.

    ds = nc.Dataset(args.ts)
    tu = ds.variables["time"].units
    t0 = dt.datetime.fromisoformat(tu.split("since")[1].strip().split()[0])
    adates = [t0 + dt.timedelta(hours=float(h))
              for h in np.asarray(ds.variables["time"][:], float)]
    area = np.asarray(ds.variables["cell_area"][:], np.float64)
    lats = np.asarray(ds.variables["lats"][:], np.float64)
    idx = {d: i for i, d in enumerate(adates)}
    keep = [i for i, d in enumerate(cdates) if d in idx and cdates[i] >= start]
    dropped = sum(1 for d in cdates if d < start)
    if dropped:
        print(f"[spin-up] dropped {dropped} day(s) before {start.date()}")
    dates = [cdates[i] for i in keep]
    ctrl = ctrl[keep]
    anom = np.asarray(ds.variables["xco2_anom"][:], np.float64)[[idx[d] for d in dates]]
    ds.close()
    print(f"[read] {len(dates)} common days: {dates[0].date()} .. {dates[-1].date()}")
    span_yr = (dates[-1] - dates[0]).days / 365.25
    if span_yr < 3.0:
        print(f"[warn] only {span_yr:.1f} yr after spin-up removal. The linear "
              "detrend plus 4-harmonic fit is under-constrained on a window this "
              "short, so the rms values below are fit residuals rather than "
              "interannual variability. Wait for the full record.")

    t = mdates.date2num(dates)
    w = area / area.sum()
    ha, _ = P._sinlat_bands(anom, lats, area)
    hc, edges = P._sinlat_bands(ctrl, lats, area)
    da = interannual(ha, dates)
    dc = interannual(hc, dates)

    ga = np.einsum("tfyx,fyx->t", anom, w)
    gc = np.einsum("tfyx,fyx->t", ctrl, w)
    gda = interannual(ga[None, :], dates)[0]
    gdc = interannual(gc[None, :], dates)[0]

    print("\nInterannual variability (per-day-of-year linear detrend; rms, ppm)")
    print(f"  zonal bands   flux-driven {da.std():7.3f} | transport-only "
          f"{dc.std():7.3f} | ratio {dc.std() / da.std():5.2f}")
    print(f"  global mean   flux-driven {gda.std():7.3f} | transport-only "
          f"{gdc.std():7.3f} | ratio {gdc.std() / gda.std():5.2f}")
    print(f"  correlation between the two zonal fields: r = "
          f"{np.corrcoef(da.ravel(), dc.ravel())[0, 1]:+.3f}")
    yrs = np.array([d.year for d in dates])
    print("\n  per-year zonal rms (ppm)   flux-driven / transport-only")
    for Y in sorted(set(yrs)):
        m = yrs == Y
        n = int(m.sum())
        flag = "" if n >= 350 else "*"
        print(f"    {Y}{flag} n={n:3d}   {da[:, m].std():6.3f} / {dc[:, m].std():6.3f}")
    print("  * partial year")

    tedge = np.append(t, t[-1] + (t[-1] - t[-2]))
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True,
                            constrained_layout=True)
    im1 = P._panel(axs[0], tedge, edges, da,
                   "Flux-driven — SIF-GPP interannual variability "
                   f"(±{np.nanpercentile(np.abs(da), 99):.2f} ppm)")
    im2 = P._panel(axs[1], tedge, edges, dc,
                   "Transport-only control — identical GPP seasonality every "
                   f"year (±{np.nanpercentile(np.abs(dc), 99):.2f} ppm)")
    for im, ax in ((im1, axs[0]), (im2, axs[1])):
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("ppm", color=P.INK, fontsize=9)
        cb.ax.tick_params(colors=P.INK2, labelsize=8)
    P.date_axis(axs[1])
    fig.suptitle("Separating flux-driven from transport-driven XCO$_2$ "
                 "interannual variability\n"
                 "equal-area sin(latitude) bands, per-day-of-year detrended; "
                 "red = anomalously weak uptake",
                 color=P.INK, fontsize=12.5, fontweight="bold")
    f = f"{args.outdir}/sif_gpp_iav_flux_vs_transport.png"
    fig.savefig(f, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
