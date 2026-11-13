#!/usr/bin/env python3
"""Figures of the process experiments with layered tracers for the paper.

    python3 examples/paper/layer_tracers/paper_figures.py --spread <dir> --maps <dir> --figures <dir> [--month 202207]

Reads the percentile files of sigma_spread.py and the column maps of column_lift.py and writes, as
PDF and PNG:

* `layer_processes`: vertical spread with time for advection + mixing, all processes and convection
  only, two cases overlaid;
* `layer_attribution`: convection only, with the cases that exchange the convective fields or the solver;
* `layer_lift_map`: where convection lifts the surface tracer, both cases, January and July;
* `layer_regions`: the surface tracer in selected regions, convection only.
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from plot_spread import CASE_COLORS, CASES, GRID, INK, INK2, MUTED, TRACERS, case_label, draw_case, percentiles

WIDTH = 7.2                                   # inches, full page width
TRACER_LABELS = {name: label for name, label, _ in TRACERS}
CENTERS = {name: center for name, _, center in TRACERS}
CASE_NAMES = {case: case_label(case, "PC") for case in CASES}      # "... schemes" for the two main cases
RUN = "{case}_c90_{month}_{arm}_f32.npz"                           # the paper figures use the C90 Float32 runs
ARM_NAMES = {"AP": "advection + mixing", "APC": "all processes", "C": "convection only"}
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LIFT_LEVELS = [0, 1, 2, 5, 10, 20, 30, 40, 60]
MONTHS = {"01": "January", "04": "April", "07": "July", "10": "October"}

plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.5, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.titlesize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5})


def load(folder, case, month, arm):
    return np.load(Path(folder) / RUN.format(case=case, month=month, arm=arm))


def panel(ax, z, tracer, region, color):
    draw_case(ax, z["hours"] / 24.0, percentiles(z, tracer, region), color, True)


def finish(ax, tracer, days_end, top=1.0):
    ax.axhline(CENTERS[tracer], color=MUTED, lw=0.6, ls=":")
    ax.set_ylim(top, 0.0)
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_xlim(0, days_end)
    ax.set_xticks([0, 1, 3, 7, 14, 28])
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def style_handles():
    return [Line2D([], [], color=INK2, lw=2.0, label="median"),
            Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label="mean"),
            Line2D([], [], color=INK2, lw=0.9, label="bounds of central 68 %"),
            Line2D([], [], color=INK2, lw=0.9, ls=(0, (1, 1.6)), label="bounds of central 95 %"),
            Line2D([], [], color=MUTED, lw=0.6, ls=":", label="starting level")]


def legends(fig, cases, top, ncol=None):
    fig.legend(handles=[Line2D([], [], color=CASE_COLORS[c], lw=2.0, label=CASE_NAMES[c]) for c in cases],
               loc="upper left", bbox_to_anchor=(0.06, top), frameon=False, ncol=ncol or len(cases), handlelength=2.2,
               columnspacing=1.5)
    rows = int(np.ceil(len(cases) / (ncol or len(cases))))
    fig.legend(handles=style_handles(), loc="upper left", bbox_to_anchor=(0.06, top - 0.21 * rows / fig.get_figheight()),
               frameon=False, ncol=5, handlelength=2.2, columnspacing=1.2)


def save(fig, folder, name):
    for ext in ("pdf", "png"):
        fig.savefig(Path(folder) / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {folder}/{name}.pdf")


def figure_processes(spread, figures, month):
    tracers, arms, cases = ("sfc", "s085", "s050", "s025"), ("AP", "APC", "C"), ("era5", "merra2")
    height = 6.3
    fig, axes = plt.subplots(len(arms), len(tracers), figsize=(WIDTH, height), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.08, hspace=0.14, left=0.085, right=0.99, top=1 - 0.78 / height,
                                              bottom=0.42 / height))
    for r, arm in enumerate(arms):
        for case in cases:
            z = load(spread, case, month, arm)
            for c, tracer in enumerate(tracers):
                panel(axes[r, c], z, tracer, 0, CASE_COLORS[case])
        for c, tracer in enumerate(tracers):
            ax = axes[r, c]
            finish(ax, tracer, z["hours"][-1] / 24.0)
            if r == 0:
                ax.set_title(f"released at {TRACER_LABELS[tracer]}" if tracer != "sfc" else "released at the surface", color=INK)
            if r == len(arms) - 1:
                ax.set_xlabel("days")
            if c == 0:
                ax.set_ylabel("σ = p / p$_\\mathrm{s}$")
            if c == 0:
                ax.text(0.97, 0.94, f"({'abc'[r]}) {ARM_NAMES[arm]}", transform=ax.transAxes, ha="right", va="top",
                        color=INK, fontsize=7.5, fontweight="bold")
    legends(fig, cases, 1 - 0.02 / height)
    save(fig, figures, "layer_processes")


def figure_attribution(spread, figures, month):
    tracers = ("sfc", "s070", "s050")
    cases = [c for c in ("era5", "merra2", "merra2era5conv", "era5ifs", "merra2tm5matrix")
             if (Path(spread) / RUN.format(case=c, month=month, arm="C")).exists()]
    if len(cases) < 5:
        print(f"layer_attribution: drawn without the missing cases, with {', '.join(cases)}")
    height = 3.25
    fig, axes = plt.subplots(1, len(tracers), figsize=(WIDTH, height), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.085, right=0.99, top=1 - 1.0 / height, bottom=0.42 / height))
    for case in cases:
        z = load(spread, case, month, "C")
        for c, tracer in enumerate(tracers):
            panel(axes[c], z, tracer, 0, CASE_COLORS[case])
    for c, tracer in enumerate(tracers):
        finish(axes[c], tracer, z["hours"][-1] / 24.0)
        axes[c].set_title(f"released at {TRACER_LABELS[tracer]}" if tracer != "sfc" else "released at the surface", color=INK)
        axes[c].set_xlabel("days")
    axes[0].set_ylabel("σ = p / p$_\\mathrm{s}$")
    legends(fig, cases, 1 - 0.02 / height, ncol=3)
    save(fig, figures, "layer_attribution")


def figure_regions(spread, figures, month, regions):
    cases = ("era5", "merra2")
    n_cols = 4
    n_rows = int(np.ceil(len(regions) / n_cols))
    height = 2.05 * n_rows + 0.95
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH, height), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.08, hspace=0.30, left=0.085, right=0.99, top=1 - 0.95 / height,
                                              bottom=0.42 / height))
    data = {case: load(spread, case, month, "C") for case in cases}
    names = [str(r) for r in data["era5"]["regions"]]
    for k, name in enumerate(regions):
        ax = axes[k // n_cols, k % n_cols]
        for case in cases:
            panel(ax, data[case], "sfc", names.index(name), CASE_COLORS[case])
        finish(ax, "sfc", data["era5"]["hours"][-1] / 24.0)
        ax.set_title(f"({'abcdefghijkl'[k]}) {name}", color=INK, loc="left", pad=3)
        if k // n_cols == n_rows - 1:
            ax.set_xlabel("days")
        if k % n_cols == 0:
            ax.set_ylabel("σ = p / p$_\\mathrm{s}$")
    legends(fig, cases, 1 - 0.02 / height)
    save(fig, figures, "layer_regions")


def figure_lift_map(maps, spread, figures, months, regions):
    cases = ("era5", "merra2")
    height = 1.95 * len(months) + 0.75
    fig = plt.figure(figsize=(WIDTH, height))
    cmap = ListedColormap(BLUES)
    norm = BoundaryNorm(LIFT_LEVELS, cmap.N)
    boxes = np.load(Path(spread) / RUN.format(case="era5", month=months[0], arm="C"))
    names = [str(r) for r in boxes["regions"]]
    mesh = None
    for r, month in enumerate(months):
        for c, case in enumerate(cases):
            z = np.load(Path(maps) / RUN.format(case=case, month=month, arm="C"))
            i = int(np.where(z["hours"] == 72)[0][0])
            ax = fig.add_axes([0.01 + 0.495 * c, 0.75 / height + (len(months) - 1 - r) * 1.95 / height, 0.485, 1.72 / height],
                              projection=ccrs.Robinson(central_longitude=-150))
            mesh = ax.pcolormesh(z["lon_edges"], z["lat_edges"], 100 * z["upper_sfc"][i], cmap=cmap, norm=norm,
                                 transform=ccrs.PlateCarree(), rasterized=True)
            ax.coastlines(resolution="110m", lw=0.3, color=INK2)
            ax.spines["geo"].set_linewidth(0.4)
            ax.spines["geo"].set_edgecolor(MUTED)
            for name in regions:
                south, north, west, east = boxes["region_boxes"][names.index(name)]
                width = east - west if east >= west else east - west + 360
                ax.add_patch(Rectangle((west, south), width, north - south, fill=False, ec="#eb6834", lw=0.7,
                                       transform=ccrs.PlateCarree()))
            ax.set_title(f"({'abcdefgh'[2 * r + c]}) {MONTHS[month[4:]]}: {'ERA5' if case == 'era5' else 'MERRA-2'} convection",
                         color=INK, loc="left", pad=2)
    cax = fig.add_axes([0.25, 0.42 / height, 0.5, 0.09 / height])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=LIFT_LEVELS)
    bar.set_label("share of the surface tracer above σ = 0.5 after 3 days (%)")
    bar.outline.set_linewidth(0.4)
    save(fig, figures, "layer_lift_map")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spread", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--figures", required=True)
    p.add_argument("--month", default="202207")
    p.add_argument("--map-months", nargs="+", default=["202201", "202207"])
    p.add_argument("--regions", nargs="+", default=["Amazon", "Maritime Continent", "West Pacific warm pool", "Pacific 8°N (ITCZ)",
                                                    "Pacific 26°N", "Pacific 48°S", "North Atlantic storm track", "Europe"])
    a = p.parse_args()
    Path(a.figures).mkdir(parents=True, exist_ok=True)
    figure_processes(a.spread, a.figures, a.month)
    figure_attribution(a.spread, a.figures, a.month)
    figure_regions(a.spread, a.figures, a.month, a.regions)
    figure_lift_map(a.maps, a.spread, a.figures, a.map_months, a.regions)


if __name__ == "__main__":
    main()
