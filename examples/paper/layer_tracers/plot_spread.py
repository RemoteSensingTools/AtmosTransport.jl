#!/usr/bin/env python3
"""Vertical spread of the layered tracers with time, from the files of sigma_spread.py.

    python3 examples/paper/layer_tracers/plot_spread.py --spread <dir of .npz> --figures <dir> \
        --month 202207 [--arm APC] [--region "all latitudes"] [--cases era5 merra2] [--time-scale sqrt|linear]

One panel per tracer, with all cases overlaid in one colour each (`--style bands`: one row of
filled bands per case): the bounds of the sigma range that holds the central 68.3 % of the tracer
(from the 15.9 % to the 84.1 % point of its vertical distribution, the range of plus and minus one
standard deviation of a Gaussian), of the central 95.4 % (2.3 % to 97.7 %, two standard deviations),
the median and the mean, against time. Percentiles do not assume a shape, so the upward and the
downward spread from the starting level are read separately. A table of the same numbers is
printed and written as CSV.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TRACERS = [("sfc", "surface", 1.0), ("s095", "σ = 0.95", 0.95), ("s085", "σ = 0.85", 0.85), ("s070", "σ = 0.70", 0.70),
           ("s050", "σ = 0.50", 0.50), ("s025", "σ = 0.25", 0.25), ("s007", "σ = 0.07", 0.07)]
CASES = {"era5": "ERA5 fields, TM5 {}", "merra2": "MERRA-2 fields, GEOS-Chem {}",
         "era5ifs": "ERA5 fields, IFS solver", "merra2tm5matrix": "MERRA-2 fields, TM5 matrix solver",
         "merra2era5conv": "MERRA-2 run, ERA5 convective fields"}
SCHEME_WORDS = {"C": "convection", "P": "mixing", "PC": "schemes"}
PROCESS = {"A": "advection", "P": "boundary-layer mixing", "C": "convection"}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
# Fractions of a Gaussian within one and two standard deviations of its mean.
P1 = (0.158655, 0.841345)
P2 = (0.022750, 0.977250)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BAND2, BAND1, MEDIAN = "#cde2fb", "#86b6ef", "#0d366b"
# One colour per case, fixed, in the order of the categorical palette (validated for colour-vision deficiencies).
CASE_COLORS = {"era5": "#2a78d6", "merra2": "#eb6834", "merra2era5conv": "#1baf7a", "era5ifs": "#6250d6",
               "merra2tm5matrix": "#e87ba4"}

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 9})


def case_label(case, arm):
    physics = "".join(c for c in arm if c in "PC")
    label = CASES[case]
    if "{}" not in label:
        return label
    return label.format(SCHEME_WORDS[physics]) if physics else label.split(",")[0]


def percentiles(z, tracer, region):
    """Sigma of the 2.3, 15.9, 50, 84.1 and 97.7 % points and the mean, at every output time."""
    edges = z["sigma_edges"]
    # Small negative mole fractions can make the stored distribution non-monotone; the running
    # maximum restores the order and moves a percentile by less than 1e-3 of the tracer amount.
    cdf = np.maximum.accumulate(np.asarray(z[tracer])[:, region, :], axis=1)
    out = np.zeros((cdf.shape[0], 6))
    for i, c in enumerate(cdf):
        out[i, :5] = np.interp([P2[0], P1[0], 0.5, P1[1], P2[1]], c, edges)
    out[:, 5] = np.asarray(z[f"mean_{tracer}"])[:, region]
    return out


def draw_case(ax, days, p, color, overlay):
    """One case in one panel: `p[:, 0:5]` are the 2.3, 15.9, 50, 84.1 and 97.7 % points, `p[:, 5]` the mean."""
    if overlay:
        ax.fill_between(days, p[:, 1], p[:, 3], color=color, alpha=0.10, lw=0)
        for k in (0, 4):
            ax.plot(days, p[:, k], color=color, lw=0.9, ls=(0, (1, 1.6)), alpha=0.9)
        for k in (1, 3):
            ax.plot(days, p[:, k], color=color, lw=1.0, alpha=0.9)
        ax.plot(days, p[:, 2], color=color, lw=2.3, alpha=0.9)
        ax.plot(days, p[:, 5], color=color, lw=1.3, ls=(0, (4, 2)), alpha=0.9)
    else:
        ax.fill_between(days, p[:, 0], p[:, 4], color=BAND2, lw=0)
        ax.fill_between(days, p[:, 1], p[:, 3], color=BAND1, lw=0)
        ax.plot(days, p[:, 2], color=MEDIAN, lw=1.6)
        ax.plot(days, p[:, 5], color=INK, lw=1.0, ls=(0, (3, 2)))


def style_axis(ax, days, center, top, time_scale, ticks):
    ax.axhline(center, color=MUTED, lw=0.7, ls=":")
    ax.set_ylim(top, 0.0)
    if time_scale == "sqrt":
        ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_xlim(0, days[-1])
    ax.set_xticks([t for t in ticks if t <= days[-1]])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend_handles(cases, arm, overlay):
    if not overlay:
        return [Patch(color=BAND1, label="central 68 % of the tracer (±1σ of a Gaussian)"),
                Patch(color=BAND2, label="central 95 % (±2σ)"),
                Line2D([], [], color=MEDIAN, lw=1.6, label="median"),
                Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)), label="mean"),
                Line2D([], [], color=MUTED, lw=0.7, ls=":", label="starting level")]
    handles = [Line2D([], [], color=CASE_COLORS[c], lw=2.3, label=case_label(c, arm)) for c in cases]
    handles += [Line2D([], [], color=INK2, lw=2.3, label="median"),
                Line2D([], [], color=INK2, lw=1.3, ls=(0, (4, 2)), label="mean"),
                Line2D([], [], color=INK2, lw=1.0, label="bounds of the central 68 % (±1σ of a Gaussian)"),
                Line2D([], [], color=INK2, lw=0.9, ls=(0, (1, 1.6)), label="bounds of the central 95 % (±2σ)"),
                Line2D([], [], color=MUTED, lw=0.7, ls=":", label="starting level")]
    return handles


def plot(data, month, arm, region, resolution, path, time_scale, overlay):
    cases = list(data)
    region_name = str(next(iter(data.values()))["regions"][region])
    n_rows = 1 if overlay else len(cases)
    panel = 3.3 if overlay else 2.55
    height = panel * n_rows + 1.75
    fig, axes = plt.subplots(n_rows, len(TRACERS), figsize=(13.0, height), sharex=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.14, hspace=0.16, left=0.055, right=0.965,
                                              top=1 - 1.5 / height, bottom=0.6 / height))
    for r, case in enumerate(cases):
        z = data[case]
        days = z["hours"] / 24.0
        for c, (tracer, label, center) in enumerate(TRACERS):
            ax = axes[0 if overlay else r, c]
            draw_case(ax, days, percentiles(z, tracer, region), CASE_COLORS[case], overlay)
            if overlay and r > 0:
                continue
            style_axis(ax, days, center, 0.3 if tracer == "s007" else 1.0, time_scale, (0, 1, 3, 7, 14, 21, 28))
            if r == 0:
                ax.set_title(f"released at\n{label}" if tracer != "sfc" else "released at\nthe surface", color=INK)
            if overlay or r == len(cases) - 1:
                ax.set_xlabel("days")
            if c == 0:
                ax.set_ylabel("σ = p / p$_s$" if overlay else f"{case_label(case, arm)}\nσ = p / p$_s$", color=INK2)
            elif tracer != "s007":
                ax.set_yticklabels([])
            else:
                ax.yaxis.tick_right()
    handles = legend_handles(cases, arm, overlay)
    if not overlay:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 1 - 0.62 / height), frameon=False,
                   ncol=5, handlelength=2.6, columnspacing=1.8)
    else:            # cases on the first row, line styles on the second
        fig.legend(handles=handles[:len(cases)], loc="upper left", bbox_to_anchor=(0.05, 1 - 0.58 / height), frameon=False,
                   ncol=len(cases), handlelength=2.6, columnspacing=1.8)
        fig.legend(handles=handles[len(cases):], loc="upper left", bbox_to_anchor=(0.05, 1 - 0.86 / height), frameon=False,
                   ncol=5, handlelength=2.6, columnspacing=1.8)
    processes = " + ".join(PROCESS[c] for c in arm)
    fig.suptitle(f"Vertical spread of the tracers: {MONTH_NAMES[int(month[4:]) - 1]} {month[:4]}, {processes}, "
                 f"{resolution.upper()}, {region_name}", x=0.055, ha="left", fontsize=11, fontweight="bold",
                 y=1 - 0.12 / height)
    limit = float(next(iter(data.values()))["min_surface_pressure"])
    note = "Percentiles of the tracer amount in σ." + (f" Columns with a surface pressure above {limit:g} hPa only." if limit > 0 else "")
    note += " The σ = 0.07 panel shows σ from 0 to 0.3." + (" Time axis: square-root scale." if time_scale == "sqrt" else "")
    fig.text(0.055, 1 - 0.40 / height, note, color=INK2, fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_regions(data, month, arm, tracer, regions, resolution, path, time_scale, overlay, columns=None):
    """One tracer in several regions: one panel per region."""
    cases = list(data)
    names = [str(r) for r in next(iter(data.values()))["regions"]]
    label, center = next((l, c) for t, l, c in TRACERS if t == tracer)
    if overlay:
        n_cols = columns or min(len(regions), 7)
        n_rows = int(np.ceil(len(regions) / n_cols))
        slots = [(k // n_cols, k % n_cols) for k in range(len(regions))]
    else:
        n_cols, n_rows = len(regions), len(cases)
    panel = 2.75 if overlay else 2.55
    height = panel * n_rows + 1.85
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols + 1.0, height), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.10, hspace=0.42 if overlay else 0.16, left=0.065, right=0.99,
                                              top=1 - 1.6 / height, bottom=0.6 / height))
    for r, case in enumerate(cases):
        z = data[case]
        days = z["hours"] / 24.0
        for c, name in enumerate(regions):
            row, col = slots[c] if overlay else (r, c)
            ax = axes[row, col]
            draw_case(ax, days, percentiles(z, tracer, names.index(name)), CASE_COLORS[case], overlay)
            if overlay and r > 0:
                continue
            style_axis(ax, days, center, 1.0, time_scale, (0, 1, 3, 7, 14, 28))
            if overlay or r == 0:
                box = z["region_boxes"][names.index(name)]
                where = "" if box[2] == -180 and box[3] == 180 else f"\n{lat(box[0])}–{lat(box[1])}, {lon(box[2])}–{lon(box[3])}"
                ax.set_title(name.replace(" (", "\n(").replace(" storm", "\nstorm") + where, color=INK, fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("days")
            if col == 0:
                ax.set_ylabel("σ = p / p$_s$" if overlay else f"{case_label(case, arm)}\nσ = p / p$_s$", color=INK2)
    if overlay:
        for k in range(len(regions), n_rows * n_cols):
            axes[k // n_cols, k % n_cols].set_visible(False)
    handles = legend_handles(cases, arm, overlay)
    if overlay:
        fig.legend(handles=handles[:len(cases)], loc="upper left", bbox_to_anchor=(0.06, 1 - 0.58 / height), frameon=False,
                   ncol=len(cases), handlelength=2.6, columnspacing=1.8)
        fig.legend(handles=handles[len(cases):], loc="upper left", bbox_to_anchor=(0.06, 1 - 0.86 / height), frameon=False,
                   ncol=5, handlelength=2.6, columnspacing=1.8)
    else:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 1 - 0.55 / height), ncol=5, frameon=False,
                   handlelength=2.2, columnspacing=1.8)
    processes = " + ".join(PROCESS[c] for c in arm)
    released = "at the surface" if tracer == "sfc" else f"at {label}"
    fig.suptitle(f"Vertical spread by region, tracer released {released}: {MONTH_NAMES[int(month[4:]) - 1]} {month[:4]}, "
                 f"{processes}, {resolution.upper()}", x=0.06, ha="left", fontsize=11, fontweight="bold", y=1 - 0.12 / height)
    note = "Percentiles of the tracer amount in σ within each region." + (" Time axis: square-root scale." if time_scale == "sqrt" else "")
    if "A" in arm:
        note += " With advection the tracer in a region includes what the winds bring in."
    fig.text(0.06, 1 - 0.38 / height, note, color=INK2, fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def lat(v):
    return f"{abs(v):g}°{'N' if v > 0 else 'S' if v < 0 else ''}"


def lon(v):
    return f"{abs(v):g}°{'E' if v > 0 else 'W' if v < 0 else ''}"


def table(data, month, arm, region, path):
    rows = []
    region_name = str(next(iter(data.values()))["regions"][region])
    print(f"\n{month} {arm}, {region_name}: sigma of the mean, the median and the percentile points, and the spread from the starting level")
    print("up = toward the model top (lower sigma), down = toward the surface; 1σ, 2σ: the 15.9/84.1 and 2.3/97.7 % points")
    print(f"{'tracer':6s} {'case':16s} {'day':>4s} | {'mean':>6s} {'median':>6s} | {'1σ up':>6s} {'1σ down':>7s} | {'2σ up':>6s} {'2σ down':>7s}")
    for tracer, label, center in TRACERS:
        for case, z in data.items():
            p = percentiles(z, tracer, region)
            for day in (0, 1, 7, 14, 28):
                hit = np.where(z["hours"] == 24 * day)[0]
                if len(hit) == 0:
                    continue
                q = p[hit[0]]
                values = dict(start=center, mean=q[5], median=q[2], p02=q[0], p16=q[1], p84=q[3], p98=q[4],
                              up_1sigma=center - q[1], down_1sigma=q[3] - center,
                              up_2sigma=center - q[0], down_2sigma=q[4] - center)
                row = dict(month=month, arm=arm, region=region_name, tracer=tracer, case=case, day=day,
                           **{k: round(float(v), 5) for k, v in values.items()})
                rows.append(row)
                print(f"{tracer:6s} {case:16s} {day:4d} | {q[5]:6.3f} {q[2]:6.3f} | {center - q[1]:6.3f} {q[3] - center:7.3f} | {center - q[0]:6.3f} {q[4] - center:7.3f}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spread", required=True, nargs="+", help="one or more folders with the .npz files")
    p.add_argument("--figures", required=True)
    p.add_argument("--month", required=True, help="YYYYMM")
    p.add_argument("--arm", default="APC")
    p.add_argument("--region", default="all latitudes", help="a region name of sigma_spread.py, e.g. 'Amazon'")
    p.add_argument("--cases", nargs="+", default=["era5", "merra2"], choices=sorted(CASES))
    p.add_argument("--resolution", default="c90")
    p.add_argument("--precision", default="32")
    p.add_argument("--time-scale", default="sqrt", choices=["sqrt", "linear"])
    p.add_argument("--style", default="overlay", choices=["overlay", "bands"],
                   help="overlay: all cases in one panel, one colour each; bands: one row of filled bands per case")
    p.add_argument("--columns", type=int, help="panels per row in the figure by region (overlay style)")
    p.add_argument("--tracer", help="with --regions: draw this tracer in several regions (e.g. sfc)")
    p.add_argument("--regions", nargs="+", help="region names for the figure by region")
    a = p.parse_args()
    data = {}
    for case in a.cases:
        name = f"{case}_{a.resolution}_{a.month}_{a.arm}_f{a.precision}.npz"
        hits = [Path(d) / name for d in a.spread if (Path(d) / name).exists()]
        if not hits:
            raise SystemExit(f"{name} not found in {a.spread}")
        data[case] = np.load(hits[0])
    Path(a.figures).mkdir(parents=True, exist_ok=True)
    regions = [str(r) for r in next(iter(data.values()))["regions"]]
    if a.tracer:
        chosen = a.regions or [r for r in regions if r != "all latitudes"]
        missing = [r for r in chosen if r not in regions]
        if missing:
            raise SystemExit(f"unknown regions {missing}; available: {', '.join(regions)}")
        stem = f"spread_regions_{a.month}_{a.arm}_{a.tracer}" + ("" if a.cases == ["era5", "merra2"] else "_" + "-".join(a.cases))
        stem += "" if a.style == "overlay" else "_bands"
        plot_regions(data, a.month, a.arm, a.tracer, chosen, a.resolution, f"{a.figures}/{stem}.png", a.time_scale,
                     a.style == "overlay", a.columns)
        print(f"wrote {a.figures}/{stem}.png")
        return
    if a.region not in regions:
        raise SystemExit(f"unknown region {a.region!r}; available: {', '.join(regions)}")
    region = regions.index(a.region)
    slug = "".join(ch if ch.isalnum() else "_" for ch in a.region.lower()).strip("_")
    stem = f"spread_{a.month}_{a.arm}" + (f"_{slug}" if region else "") + ("" if a.cases == ["era5", "merra2"] else "_" + "-".join(a.cases))
    stem += "" if a.style == "overlay" else "_bands"
    plot(data, a.month, a.arm, region, a.resolution, f"{a.figures}/{stem}.png", a.time_scale, a.style == "overlay")
    table(data, a.month, a.arm, region, f"{a.figures}/{stem}.csv")
    print(f"wrote {a.figures}/{stem}.png")


if __name__ == "__main__":
    main()
