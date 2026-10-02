#!/usr/bin/env python3
"""Single-panel Hovmoller of the transported XCO2 anomaly.

Equal-area sin(latitude) bands. By default the only operation applied is a
per-band linear detrend in time -- the seasonal cycle is left in. Add
--deseason to remove the mean seasonal cycle as well (empirical day-of-year
climatology, not a harmonic fit).

--tracer anom      the flux-driven SIF-GPP interannual signal (default)
--tracer full      the full uptake tracer, whose ~468 ppm cumulative drawdown
                   the detrend removes
--tracer balanced  an annually balanced, NEE-like tracer built by linearity as
                   xco2_anom + xco2_clim. The anomaly tracer is forced by
                   F - C and the control by C - M, so the sum is forced by
                   F - M: the full flux with each cell's annual mean removed.
                   That has the real seasonal cycle and the real interannual
                   variability but NO secular drawdown, which removes the
                   inflated Southern-Hemisphere seasonal signal that the raw
                   uptake tracer shows (see memo section 4i). Requires the
                   control run output.

--minus-global     subtract the contemporaneous global mean from every band,
                   leaving only the meridional structure. Works on any tracer
                   and needs no extra run.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "plotmod", os.path.join(_HERE, "plot_sif_gpp_iav_xco2.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

_ispec = importlib.util.spec_from_file_location(
    "inpmod", os.path.join(_HERE, "plot_sif_gpp_input_hovmoller.py"))
I = importlib.util.module_from_spec(_ispec)
_ispec.loader.exec_module(I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="/temp1/cfranken/sif_gpp_iav/"
                                    "sif_gpp_iav_c90_xco2_timeseries.nc")
    ap.add_argument("--tracer", choices=("anom", "full", "balanced"),
                    default="anom")
    ap.add_argument("--control", default="/temp1/cfranken/sif_gpp_iav/output/"
                                         "sif_gpp_clim_c90",
                    help="control-run output dir (needed by --tracer balanced)")
    ap.add_argument("--control-carrier", type=float, default=300.0)
    ap.add_argument("--minus-global", action="store_true",
                    help="subtract the contemporaneous global mean from each band")
    ap.add_argument("--outdir", default="plots/sif_gpp_iav")
    ap.add_argument("--nband", type=int, default=40)
    ap.add_argument("--deseason", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="no processing at all: plot the field as transported, "
                         "with no detrend, no deseasonalisation and no "
                         "global-mean removal")
    ap.add_argument("--clim-smooth", type=int, default=31)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ds = nc.Dataset(args.ts)
    tu = ds.variables["time"].units
    t0 = dt.datetime.fromisoformat(tu.split("since")[1].strip().split()[0])
    dates = [t0 + dt.timedelta(hours=float(h))
             for h in np.asarray(ds.variables["time"][:], float)]
    area = np.asarray(ds.variables["cell_area"][:], np.float64)
    lats = np.asarray(ds.variables["lats"][:], np.float64)
    if args.tracer == "balanced":
        _cspec = importlib.util.spec_from_file_location(
            "cmpmod", os.path.join(_HERE, "compare_sif_gpp_transport_control.py"))
        C = importlib.util.module_from_spec(_cspec)
        _cspec.loader.exec_module(C)
        cdates, ctrl = C.load_daily(args.control, "co2_gpp_clim_column_mean",
                                    args.control_carrier)
        idx = {d: i for i, d in enumerate(dates)}
        keep = [i for i, d in enumerate(cdates) if d in idx]
        if not keep:
            raise SystemExit("no overlap between control run and time series")
        sub = [idx[cdates[i]] for i in keep]
        anom = np.asarray(ds.variables["xco2_anom"][:], np.float64)[sub]
        fld = anom + ctrl[keep]
        dates = [cdates[i] for i in keep]
        print(f"[balanced] xco2_anom + xco2_clim over {len(dates)} common days "
              f"({dates[0].date()} .. {dates[-1].date()})")
        if len(dates) < 0.9 * len(idx):
            print(f"[warn] control run covers only {len(dates)} of {len(idx)} "
                  "days; this is a partial record")
    else:
        fld = np.asarray(ds.variables[f"xco2_{args.tracer}"][:], np.float64)
    ds.close()

    if args.minus_global:
        w = area / area.sum()
        gm = np.einsum("tfyx,fyx->t", np.nan_to_num(fld), w)
        fld = fld - gm[:, None, None, None]
        print(f"[minus-global] removed the contemporaneous global mean "
              f"(range {gm.min():+.2f} .. {gm.max():+.2f} ppm)")

    t = mdates.date2num(dates)
    hov, edges = P._sinlat_bands(fld, lats, area, args.nband)
    det = hov if args.raw else P._detrend(hov, t)
    if args.deseason and args.raw:
        raise SystemExit("--raw and --deseason are mutually exclusive")
    if args.deseason:
        years = np.array([d.year for d in dates])
        clim, cmin, cmax = I.empirical_seasonal_cycle(det, dates, years,
                                                      args.clim_smooth)
        det = det - clim
        print(f"[season] empirical day-of-year climatology, {cmin}-{cmax} "
              f"samples/slot, {args.clim_smooth}-day circular smooth")

    print(f"[xco2_{args.tracer}] {len(dates)} days, {args.nband} equal-area bands")
    print(f"  band range  {hov.min():+9.3f} .. {hov.max():+9.3f} ppm")
    if not args.raw:
        print(f"  after detrend rms {det.std():9.3f} ppm, "
              f"range {det.min():+.3f} .. {det.max():+.3f}")

    tedge = np.append(t, t[-1] + (t[-1] - t[-2]))
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    lo, hi = float(det.min()), float(det.max())
    one_signed = (lo >= 0) or (hi <= 0) or \
                 (min(abs(lo), abs(hi)) < 0.12 * max(abs(lo), abs(hi)))
    if one_signed:
        # A magnitude, not a polarity: use the sequential single-hue ramp.
        # Flip it for a purely negative field so "more drawdown" reads darker.
        flip = hi <= 0 or abs(lo) > abs(hi)
        data = -det if flip else det
        im = ax.pcolormesh(tedge, edges, data, cmap=I.SEQUENTIAL,
                           vmin=max(0.0, float(data.min())),
                           vmax=float(np.nanpercentile(data, 99.8)),
                           shading="flat")
        cmap_note = "sequential (field is effectively one-signed"
        cmap_note += "; sign flipped so darker = more drawdown)" if flip else ")"
        ax.set_yticks(np.sin(np.deg2rad(np.array([-90, -60, -30, 0, 30, 60, 90]))))
        ax.set_yticklabels([f"{x}°" for x in (-90, -60, -30, 0, 30, 60, 90)])
        ax.set_ylabel("latitude (equal-area)", color=P.INK, fontsize=10)
        ax.tick_params(colors=P.INK2, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(P.GRID)
    else:
        im = P._panel(ax, tedge, edges, det, "")
        cmap_note = "diverging about zero"
    print(f"  colour map: {cmap_note}")
    what = ("As transported, unprocessed" if args.raw else
            "Detrended and deseasonalised" if args.deseason else "Detrended")
    name = {"anom": "SIF-GPP interannual variability",
            "full": "full SIF-GPP uptake tracer",
            "balanced": "annually balanced (NEE-like) SIF-GPP tracer"}[args.tracer]
    if args.minus_global:
        name += ", global mean removed"
    sub = ("equal-area sin(latitude) bands; no detrending, no deseasonalisation"
           if args.raw else
           "equal-area sin(latitude) bands; a linear trend in time is removed "
           "from each band; red = anomalously weak uptake")
    ax.set_title(f"{what} zonal-mean XCO$_2$ — {name}\n{sub}",
                 color=P.INK, fontsize=11.5, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("ppm", color=P.INK, fontsize=9)
    cb.ax.tick_params(colors=P.INK2, labelsize=8)
    P.date_axis(ax)
    tag = ("_raw" if args.raw else
           "_deseason" if args.deseason else "") + \
          ("_minusglobal" if args.minus_global else "")
    stem = "xco2_hovmoller" if args.raw else "xco2_hovmoller_detrended"
    f = f"{args.outdir}/{stem}_{args.tracer}{tag}.png"
    fig.savefig(f, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
