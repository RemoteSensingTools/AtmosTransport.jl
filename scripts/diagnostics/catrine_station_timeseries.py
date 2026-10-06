#!/usr/bin/env python3
"""Station time series, AtmosTransport vs the GEOS-Chem CATRINE standard.

Both models are sampled in the cell whose centre is nearest the station, and
their dry mole-fraction profiles are interpolated in log(dry pressure) to the
station's pressure level. Mountain stations sit far above their cell's mean
surface (Mauna Loa: station ~680 hPa, model surfaces ~910-960 hPa), so the
lowest model layer would sample the marine boundary layer instead.

  python3 scripts/diagnostics/catrine_station_timeseries.py \
      --dataset 'AtmosTransport C90=/temp1/.../catrine_c90_*.nc' \
      --fossil 'AtmosTransport C90=co2_fossil_from_dec2021' \
      --station 'Mauna Loa,19.536,-155.576,680' --out mlo.png

With --column-npz (the fields_c90.npz written by catrine_compare_vs_geoschem.py)
it plots dry-air-mass weighted column means of the station's C90 cell
instead, GEOS-Chem aggregated mass-weighted onto the same cell.
"""
import argparse, datetime as dt, glob, os
import numpy as np
from netCDF4 import Dataset

GC_DIR = os.path.expanduser("~/data/AtmosTransport/catrine-geoschem-runs")
T0 = dt.datetime(2021, 12, 1)
G = 9.80665
# key: (AtmosTransport variable, GEOS-Chem variable, label, unit, scale)
TRACERS = [("co2_natural", "SpeciesConcVV_CO2", "CO$_2$ (total)", "ppm", 1e6),
           ("co2_fossil", "SpeciesConcVV_FossilCO2", "fossil CO$_2$", "ppm", 1e6),
           ("sf6", "SpeciesConcVV_SF6", "SF$_6$", "ppt", 1e12),
           ("rn222", "SpeciesConcVV_Rn222", "$^{222}$Rn", "10$^{-21}$ mol/mol", 1e21)]


def nearest_cell(lons, lats, lon, lat):
    lo, la, t, p = map(np.radians, (lons, lats, lon, lat))
    d = np.arccos(np.clip(np.sin(la) * np.sin(p) + np.cos(la) * np.cos(p) * np.cos(lo - t), -1, 1))
    return np.unravel_index(np.argmin(d), d.shape)


def at_level(q, p_mid, p_target):
    """Interpolate a top-first profile to p_target in log p."""
    lp = np.log(p_mid)
    return float(np.interp(np.log(p_target), lp, q))   # p_mid increases downward


def read_at(pattern, fossil, cell, p_target):
    times, out = [], {k[0]: [] for k in TRACERS}
    names = {k[0]: k[0] for k in TRACERS}
    if fossil:
        names["co2_fossil"] = fossil
    p, j, i_x = cell
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        with Dataset(f) as d:
            for i, h in enumerate(np.asarray(d["time"][:], float)):
                t = T0 + dt.timedelta(hours=float(h))
                if times and t <= times[-1]:
                    continue
                m = np.asarray(d["air_mass_per_area"][i, :, p, j, i_x], float)   # read one column only
                p_half = np.concatenate([[0.0], G * np.cumsum(m)])
                p_mid = 0.5 * (p_half[:-1] + p_half[1:])
                times.append(t)
                for k, v in names.items():
                    out[k].append(at_level(np.asarray(d[v][i, :, p, j, i_x], float), p_mid, p_target))
    return times, {k: np.array(v) for k, v in out.items()}


def read_gc(cell, p_target):
    times, out = [], {k[0]: [] for k in TRACERS}
    p, j, i_x = cell
    for f in sorted(glob.glob(f"{GC_DIR}/GEOSChem.CATRINE_inst.*.nc4")):
        stamp = dt.datetime.strptime(os.path.basename(f)[-18:-5], "%Y%m%d_%H%M")
        try:
            with Dataset(f) as d:
                p_mid = np.asarray(d["Met_PMIDDRY"][0, :, p, j, i_x], float)[::-1] * 100.0   # top first
                vals = {k: at_level(np.asarray(d[g][0, :, p, j, i_x], float)[::-1], p_mid, p_target)
                        for k, g, *_ in TRACERS}
        except (OSError, RuntimeError, KeyError):
            continue                                     # one corrupt GEOS-Chem file
        times.append(stamp)
        for k, v in vals.items():
            out[k].append(v)
    return times, {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="label=glob of AtmosTransport daily files")
    ap.add_argument("--fossil", default=None, help="label=fossil tracer variable")
    ap.add_argument("--station", default="Mauna Loa,19.536,-155.576,680", help="name,lat,lon,dry pressure hPa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--column-npz", default=None, help="fields_c90.npz: plot column means instead")
    ap.add_argument("--npz-label", default=None, help="dataset label inside the npz")
    args = ap.parse_args()
    label, pattern = args.dataset.split("=", 1)
    fossil = args.fossil.split("=", 1)[1] if args.fossil else None
    name, lat, lon, p_hpa = args.station.split(",")
    lat, lon, p_target = float(lat), float(lon), float(p_hpa) * 100.0

    with Dataset(sorted(glob.glob(os.path.expanduser(pattern)))[0]) as d:
        at_cell = nearest_cell(np.asarray(d["lons"][:]), np.asarray(d["lats"][:]), lon, lat)
    if args.column_npz:
        return plot_columns(args, label, name, lat, lon, at_cell)
    with Dataset(sorted(glob.glob(f"{GC_DIR}/GEOSChem.CATRINE_inst.*.nc4"))[0]) as d:
        gc_cell = nearest_cell(np.asarray(d["lons"][:]), np.asarray(d["lats"][:]), lon, lat)
    t_at, at = read_at(pattern, fossil, at_cell, p_target)
    t_gc, gc = read_gc(gc_cell, p_target)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(len(TRACERS), 1, figsize=(11, 2.4 * len(TRACERS)), sharex=True)
    common = sorted(set(t_at) & set(t_gc))
    ia = [t_at.index(t) for t in common]; ig = [t_gc.index(t) for t in common]
    for ax, (k, _, lab, unit, scale) in zip(axes, TRACERS):
        ax.plot(t_gc, gc[k] * scale, color="black", lw=1.4, label="GEOS-Chem (C180 L72)")
        ax.plot(t_at, at[k] * scale, color="tab:red", lw=1.4, label=label)
        a, g = at[k][ia] * scale, gc[k][ig] * scale
        r = np.corrcoef(a, g)[0, 1] if np.std(a) > 0 and np.std(g) > 0 else np.nan
        ax.set_ylabel(f"{lab}\n({unit})")
        ax.text(0.01, 0.95, f"r = {r:.3f}   mean diff = {np.mean(a - g):+.3g} {unit}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"{name} ({lat:.2f}°N, {lon:.2f}°E): dry mole fraction at {p_target / 100:.0f} hPa dry pressure, "
                 f"3-hourly, nearest model cell", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out, "cells: AT", at_cell, "GC", gc_cell, f"{len(common)} common times")


def plot_columns(args, label, name, lat, lon, cell):
    """Column means of one C90 cell from the comparison's fields_c90.npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    z = np.load(args.column_npz)
    times = [dt.datetime.fromisoformat(t) for t in z["times"]]
    keys = {"co2_natural": "co2", "co2_fossil": "fossil", "sf6": "sf6", "rn222": "rn222"}
    model = args.npz_label
    fig, axes = plt.subplots(len(TRACERS), 1, figsize=(11, 2.4 * len(TRACERS)), sharex=True)
    for ax, (k, _, lab, unit, scale) in zip(axes, TRACERS):
        g = z[f"gc__{keys[k]}__column"][(slice(None),) + cell] * scale
        a = z[f"{model}__{keys[k]}__column"][(slice(None),) + cell] * scale
        ax.plot(times, g, color="black", lw=1.4, label="GEOS-Chem (C180 L72, on the C90 cell)")
        ax.plot(times, a, color="tab:red", lw=1.4, label=label)
        r = np.corrcoef(a, g)[0, 1]
        ax.text(0.01, 0.95, f"r = {r:.3f}   mean diff = {np.mean(a - g):+.3g} {unit}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
        ax.set_ylabel(f"{lab}\n({unit})"); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"{name} cell ({lat:.2f}°N, {lon:.2f}°E): total-column dry-air mean mole fraction, 3-hourly",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print("wrote", args.out, "cell", cell)


if __name__ == "__main__":
    main()
