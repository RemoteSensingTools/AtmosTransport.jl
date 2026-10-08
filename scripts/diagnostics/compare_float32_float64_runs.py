#!/usr/bin/env python3
"""Float32 vs Float64 twin runs: global totals and column-mean differences.

Both runs must have the same configuration apart from `float_type`, with daily
files carrying `<tracer>_total_mass` (Float64, compensated, written at every
snapshot) and `<tracer>_column_mean` fields. For every common snapshot this
reports, per tracer:
  * (M32 - M64) / M64 of the global total, raw and net of the first snapshot
    (the initial Float32 rounding of the mixing ratios);
  * the area-weighted mean and RMS of the column-mean difference F32 - F64
    (once per day, at --hour UTC, to bound the reading).
Writes <out>/float32_vs_float64.png, <out>/float32_vs_float64_map.png (column
mean difference at the last common day) and <out>/float32_vs_float64.json.

  python3 scripts/diagnostics/compare_float32_float64_runs.py \\
      --f32 '/temp1/.../merra2_full_f32/*.nc' --f64 '~/data/.../catrine_merra2_f64/*.nc' --out DIR
"""
import argparse, datetime as dt, glob, json, os
import numpy as np
from netCDF4 import Dataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACERS = {  # variable: (label, scale, unit)
    "co2_natural": ("CO₂", 1e6, "ppm"),
    "co2_fossil_from_dec2021": ("Fossil CO₂ (from 2021-12-01)", 1e6, "ppm"),
    "sf6": ("SF₆", 1e12, "ppt"),
    "rn222": ("Rn-222", 1e21, "10⁻²¹ mol/mol"),
}


def by_date(pattern):
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    return {os.path.basename(f).rsplit("_", 1)[-1][:8]: f for f in files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f32", required=True); ap.add_argument("--f64", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--hour", type=int, default=0)
    args = ap.parse_args()
    out = os.path.expanduser(args.out); os.makedirs(out, exist_ok=True)
    f32, f64 = by_date(args.f32), by_date(args.f64)
    days = sorted(set(f32) & set(f64))
    print(f"{len(days)} common days {days[0]} .. {days[-1]}", flush=True)
    t0 = dt.datetime.strptime(days[0], "%Y%m%d")       # the time axis counts hours from the run start

    totals = {v: ([], []) for v in TRACERS}
    fields = {v: ([], []) for v in TRACERS}
    when, field_days, area, last = [], [], None, {}
    for day in days:
        with Dataset(f32[day]) as a, Dataset(f64[day]) as b:
            n = min(len(a["time"]), len(b["time"]))
            if area is None:
                area = np.asarray(b["cell_area"][:], float); w = area / area.sum()
            hours = np.asarray(b["time"][:n], float)
            when += [t0 + dt.timedelta(hours=float(h)) for h in hours]
            for v in TRACERS:
                totals[v][0].extend(np.asarray(a[f"{v}_total_mass"][:n], float))
                totals[v][1].extend(np.asarray(b[f"{v}_total_mass"][:n], float))
            i = int(np.argmin(np.abs((hours % 24) - args.hour)))
            field_days.append(t0 + dt.timedelta(hours=float(hours[i])))
            for v, (_, sc, _) in TRACERS.items():
                d = (np.asarray(a[f"{v}_column_mean"][i], float) - np.asarray(b[f"{v}_column_mean"][i], float)) * sc
                fields[v][0].append(float(np.sum(w * d))); fields[v][1].append(float(np.sqrt(np.sum(w * d * d))))
                if day == days[-1]:
                    last[v] = (d, np.asarray(b["lons"][:], float), np.asarray(b["lats"][:], float))

    summary = {"days": [days[0], days[-1], len(days)], "tracers": {}}
    fig, axes = plt.subplots(2, len(TRACERS), figsize=(4.2 * len(TRACERS), 6.4), squeeze=False)
    for c, (v, (label, sc, unit)) in enumerate(TRACERS.items()):
        m32, m64 = map(np.asarray, totals[v])
        ok = m64 != 0
        rel = np.full(m64.shape, np.nan); rel[ok] = (m32[ok] - m64[ok]) / m64[ok]
        k0 = int(np.argmax(ok))
        net = rel - rel[k0] if m64[0] > 0 else rel
        ax = axes[0][c]; ax.plot(when[:len(rel)], net, lw=0.8)
        ax.axhline(0, color="0.6", lw=0.5); ax.set_title(label, fontsize=10)
        ax.set_ylabel("(F32 − F64)/F64 of global total" if c == 0 else "")
        ax = axes[1][c]
        ax.plot(field_days, fields[v][1], lw=0.8, label="RMS"); ax.plot(field_days, fields[v][0], lw=0.8, label="mean")
        ax.set_ylabel(f"column-mean F32 − F64 ({unit})")
        if c == 0: ax.legend(fontsize=8)
        summary["tracers"][v] = {"final_total_rel_net_of_start": float(net[-1]),
                                 "final_total_rel_raw": float(rel[-1]),
                                 "final_column_mean_rms": fields[v][1][-1], "final_column_mean_mean": fields[v][0][-1],
                                 "unit": unit}
    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.suptitle(f"Float32 − Float64, MERRA-2 C90 ({days[0]} .. {days[-1]})", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/float32_vs_float64.png", dpi=110); plt.close(fig)

    import cartopy.crs as ccrs
    fig, axes = plt.subplots(1, len(TRACERS), figsize=(4.4 * len(TRACERS), 2.8),
                             subplot_kw={"projection": ccrs.Robinson()}, squeeze=False)
    for ax, (v, (label, sc, unit)) in zip(axes[0], TRACERS.items()):
        d, lon, lat = last[v]
        lim = float(np.nanpercentile(np.abs(d), 99.5)) or 1.0
        im = ax.scatter(lon.ravel(), lat.ravel(), c=d.ravel(), s=0.3, cmap="RdBu_r", vmin=-lim, vmax=lim,
                        transform=ccrs.PlateCarree())
        ax.set_global(); ax.coastlines(linewidth=0.3); ax.set_title(f"{label} ({unit})", fontsize=9)
        plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    fig.suptitle(f"Column-mean Float32 − Float64 on {field_days[-1]:%Y-%m-%d %H}z", fontsize=11)
    fig.savefig(f"{out}/float32_vs_float64_map.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    with open(f"{out}/float32_vs_float64.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    for v, s in summary["tracers"].items():
        print(f"{v:26s} total (net) {s['final_total_rel_net_of_start']:+.2e}  column-mean RMS "
              f"{s['final_column_mean_rms']:.2e} {s['unit']}  mean {s['final_column_mean_mean']:+.2e}")


if __name__ == "__main__":
    main()
