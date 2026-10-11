#!/usr/bin/env python3
"""Figures of a Float32 vs Float64 twin-run comparison.

Reads float32_vs_float64.nc, written by
scripts/diagnostics/compare_float32_float64_runs.jl, and draws:

  float32_vs_float64.png      per tracer, (M32 - M64)/M64 of the global total
                              net of the first snapshot (top), and the
                              area-weighted mean and RMS of the column-mean
                              difference F32 - F64 once per day (bottom);
  float32_vs_float64_map.png  the column-mean difference F32 - F64 at the last
                              daily snapshot, per tracer (Robinson projection,
                              colour limit the 99.5th percentile of |d|).

Values are shown in each variable's `display_units` (`display_scale` times the
stored mol mol-1). No numbers are computed here beyond that scaling.

  python3 scripts/visualization/plot_float32_vs_float64.py DIR [--out DIR] [--dpi 110]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

NETCDF_NAME = "float32_vs_float64.nc"
SUFFIX = "_column_mean_diff"


def times(var):
    return netCDF4.num2date(var[:], var.units, only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)


def tracers(ds):
    """Tracer names in file order: the variables `<tracer>_column_mean_diff`."""
    return [name[:-len(SUFFIX)] for name in ds.variables if name.endswith(SUFFIX)]


def scaled(var):
    return np.asarray(var[:], float) * float(var.display_scale)


def time_series_figure(ds, names, path, dpi):
    when, field_days = times(ds["time"]), times(ds["sample_time"])
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 6.4), squeeze=False)
    for c, v in enumerate(names):
        rms, mean = ds[f"{v}_column_mean_diff_rms"], ds[f"{v}_column_mean_diff_mean"]
        ax = axes[0][c]
        ax.plot(when, np.asarray(ds[f"{v}_total_rel_diff_net"][:], float), lw=0.8)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(rms.display_label, fontsize=10)
        ax.set_ylabel("(F32 − F64)/F64 of global total" if c == 0 else "")
        ax = axes[1][c]
        ax.plot(field_days, scaled(rms), lw=0.8, label="RMS")
        ax.plot(field_days, scaled(mean), lw=0.8, label="mean")
        ax.set_ylabel(f"column-mean F32 − F64 ({rms.display_units})")
        if c == 0:
            ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    label = f", {ds.label}" if getattr(ds, "label", "") else ""
    fig.suptitle(f"Float32 − Float64{label} ({ds.first_day} .. {ds.last_day})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def map_figure(ds, names, path, dpi):
    import cartopy.crs as ccrs
    lon, lat = np.asarray(ds["lons"][:], float), np.asarray(ds["lats"][:], float)
    last = times(ds["sample_time"])[-1]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 2.8),
                             subplot_kw={"projection": ccrs.Robinson()}, squeeze=False)
    for ax, v in zip(axes[0], names):
        var = ds[f"{v}{SUFFIX}"]
        d = scaled(var)
        finite = np.abs(d[np.isfinite(d)])
        lim = float(np.percentile(finite, 99.5)) if finite.size else 0.0
        lim = lim if lim > 0 else 1.0           # all zero or all missing: any symmetric range
        im = ax.scatter(lon.ravel(), lat.ravel(), c=d.ravel(), s=0.3, cmap="RdBu_r", vmin=-lim, vmax=lim,
                        transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.3)
        ax.set_title(f"{var.display_label} ({var.display_units})", fontsize=9)
        plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    fig.suptitle(f"Column-mean Float32 − Float64 on {last:%Y-%m-%d %H}z", fontsize=11)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="output folder of compare_float32_float64_runs.jl")
    ap.add_argument("--out", help="folder for the figures (default: the input folder)")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    src = os.path.expanduser(args.input)
    out = os.path.expanduser(args.out or src)
    os.makedirs(out, exist_ok=True)
    with netCDF4.Dataset(os.path.join(src, NETCDF_NAME)) as ds:
        names = tracers(ds)
        time_series_figure(ds, names, os.path.join(out, "float32_vs_float64.png"), args.dpi)
        map_figure(ds, names, os.path.join(out, "float32_vs_float64_map.png"), args.dpi)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
