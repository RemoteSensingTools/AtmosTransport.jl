#!/usr/bin/env python3
"""Figures of the layered-tracer runs from the zonal sums written by zonal_reduce.py.

    python3 examples/paper/layer_tracers/plot_zonal.py --zonal <dir of .npz> --figures <dir> \
        --month 202207 [--arm APC] [--resolution c90] [--precision 32] [--section-hours 168 672]

Two figures per month and process combination, the ERA5 case against the MERRA-2 case. The cases
differ in the meteorological fields and in the convection and mixing schemes: ERA5 fields drive the
TM5 schemes, MERRA-2 fields the GEOS-Chem schemes. The labels name both, because the result of
"ERA5 convection" is that of the ERA5 mass fluxes in the TM5 scheme, not of the ECMWF model.

* `sections_<month>_<arm>_h<hour>.png`: latitude-pressure sections of the zonal-mean mole fraction,
  one row per tracer, columns ERA5 case, MERRA-2 case and their difference.
* `profiles_<month>_<arm>.png`: mean profiles in three latitude bands after one to four weeks.

All mole fractions are divided by the well-mixed value (tracer amount over the dry air mass of the
whole atmosphere), so 1 means fully mixed and the scale is the same for every tracer.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

# Release levels, lowest first. `sfc` and `s095` evolve alike once mixing acts; `s095` is not drawn.
TRACERS = [("sfc", "surface", 1.0), ("s085", "σ = 0.85", 0.85), ("s070", "σ = 0.70", 0.70),
           ("s050", "σ = 0.50", 0.50), ("s025", "σ = 0.25", 0.25), ("s007", "σ = 0.07", 0.07)]
STRATOSPHERIC = "s007"
METS = (("era5", "ERA5"), ("merra2", "MERRA-2"))
SCHEMES = {"era5": "TM5", "merra2": "GEOS-Chem"}      # origin of the convection and mixing schemes
SCHEME_WORDS = {"C": "convection", "P": "mixing", "PC": "schemes"}
PROCESS = {"A": "advection", "P": "boundary-layer mixing", "C": "convection"}
BANDS = (("30°N–90°N", 30, 90), ("30°S–30°N", -30, 30), ("90°S–30°S", -90, -30))
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
         "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
WEEK_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # one hue, light to dark with time
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b3b0", "#e34948", "#8a1c1c"])

LEVELS = {False: [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2, 3],
          True: [0, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40]}
DIFF_LEVELS = {False: [-1, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 1],
               True: [-10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10]}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                     "axes.linewidth": 0.6, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlesize": 8, "legend.frameon": False})


def sequential(levels):
    colors = [SURFACE] + BLUES[:len(levels) - 2]
    cmap = ListedColormap(colors)
    cmap.set_over(BLUES[min(len(levels) - 2, len(BLUES) - 1)])
    return cmap, BoundaryNorm(levels, cmap.N)


def diverging(levels):
    n = len(levels) + 1
    colors = DIVERGING(np.linspace(0.02, 0.98, n))
    cmap = ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    return cmap, BoundaryNorm(levels, cmap.N)


def load(zonal, met, resolution, month, arm, precision):
    z = np.load(f"{zonal}/{met}_{resolution}_{month}_{arm}_f{precision}.npz")
    return {k: z[k] for k in z.files}


def index(z, hour):
    return int(np.where(z["hours"] == hour)[0][0])


def coverage(air, dp):
    """Share of a bin that holds air, from the air mass per hPa relative to a full bin of that latitude or band."""
    per_hpa = air / dp[(slice(None),) + (None,) * (air.ndim - 1)]
    return per_hpa / per_hpa.max(axis=0, keepdims=True)


def well_mixed(z, tracer, i):
    """Mole fraction of the tracer mixed over the whole atmosphere (all columns)."""
    return z[f"total_{tracer}"][i] / z["total_air"][i]


def section(z, tracer, hour):
    """Zonal-mean mole fraction over the well-mixed value, [pressure, latitude].

    Masked: bins below ground and latitudes where less than a tenth of the area is kept."""
    i = index(z, hour)
    air, amount = z["air"][i], z[tracer][i]
    ratio = amount / np.maximum(air, 1e-30) / well_mixed(z, tracer, i)
    with np.errstate(invalid="ignore", divide="ignore"):
        empty = ~(coverage(air, np.diff(z["p_edges"])) >= 0.05) | (z["kept_area_fraction"] < 0.1)[None, :]
    return np.ma.masked_where(empty, ratio)


def profile(z, tracer, hour, south, north):
    i = index(z, hour)
    lat = 0.5 * (z["lat_edges"][1:] + z["lat_edges"][:-1])
    sel = (lat > south) & (lat < north)
    air, amount = z["air"][i][:, sel].sum(axis=1), z[tracer][i][:, sel].sum(axis=1)
    ratio = amount / np.maximum(air, 1e-30) / well_mixed(z, tracer, i)
    return np.ma.masked_where(coverage(air, np.diff(z["p_edges"])) < 0.25, ratio)


def surface_pressure(z):
    """Global mean pressure of the dry air column, hPa."""
    area_weight = z["air"][0].sum(axis=0)
    centers = 0.5 * (z["p_edges"][1:] + z["p_edges"][:-1])
    return 2 * (z["air"][0] * centers[:, None]).sum() / area_weight.sum()


def title(month, arm, resolution):
    processes = " + ".join(PROCESS[c] for c in arm)
    return f"{MONTH_NAMES[int(month[4:]) - 1]} {month[:4]}, {processes}, {resolution.upper()}"


def case_label(met, arm):
    """Name of a case: the meteorological fields and, if a scheme acts, where the scheme is from."""
    fields = dict(METS)[met]
    physics = "".join(c for c in arm if c in "PC")
    return f"{fields} fields, {SCHEMES[met]} {SCHEME_WORDS[physics]}" if physics else fields


def zonal_note(z):
    limit = float(z["min_surface_pressure"])
    columns = f" over columns with a surface pressure above {limit:g} hPa" if limit > 0 else ""
    grey = "below the surface or no columns" if limit > 0 else "below the surface"
    return f"Zonal mean{columns}. Black tick at the left edge: release level. Grey: {grey}."


def columns_note(z):
    limit = float(z["min_surface_pressure"])
    return f" Columns with a surface pressure above {limit:g} hPa only." if limit > 0 else ""


def plot_sections(data, month, arm, resolution, hour, path):
    rows = TRACERS[::-1]                                   # highest release on top
    fig = plt.figure(figsize=(9.2, 11.6))
    # stratospheric row, its colour bars, a gap, the tropospheric rows, their colour bars
    heights = [1.0, 0.07, 0.42] + [1.0] * (len(rows) - 1) + [0.07]
    gs = fig.add_gridspec(len(heights), 3, height_ratios=heights, hspace=0.42, wspace=0.07,
                          left=0.075, right=0.985, top=0.905, bottom=0.05)
    row_slot = [0] + list(range(3, 3 + len(rows) - 1))
    ps = surface_pressure(data["era5"])
    meshes = {}
    for r, (tracer, label, sigma) in enumerate(rows):
        strat = tracer == STRATOSPHERIC
        cmap, norm = sequential(LEVELS[strat])
        dcmap, dnorm = diverging(DIFF_LEVELS[strat])
        fields = [section(data[m], tracer, hour) for m, _ in METS]
        fields.append(fields[0] - fields[1])
        for c, field in enumerate(fields):
            ax = fig.add_subplot(gs[row_slot[r], c])
            z = data["era5"]
            ax.set_facecolor("#c9c8c1")                    # below ground
            mesh = ax.pcolormesh(z["lat_edges"], z["p_edges"], field, cmap=cmap if c < 2 else dcmap,
                                 norm=norm if c < 2 else dnorm, rasterized=True)
            meshes[(strat, c < 2)] = mesh
            ax.set_ylim(250 if strat else 1040, 0)
            ax.set_xlim(-90, 90)
            ax.set_xticks([-60, -30, 0, 30, 60])
            ax.tick_params(length=2.5, width=0.5)
            if sigma < 1:
                ax.plot([-90, -84], [sigma * ps] * 2, color=INK, lw=1.4, solid_capstyle="butt", clip_on=False)
            else:
                ax.plot([-90, -84], [ps] * 2, color=INK, lw=1.4, solid_capstyle="butt", clip_on=False)
            last = r in (0, len(rows) - 1)
            ax.set_xticklabels(["60°S", "30°S", "0°", "30°N", "60°N"] if last else [])
            if c == 0:
                ax.set_ylabel("pressure (hPa)")
                ax.set_title(f"released at the {label}" if sigma == 1 else f"released at {label}", loc="left", pad=3, color=INK)
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.annotate([case_label("era5", arm), case_label("merra2", arm), "ERA5 case − MERRA-2 case"][c], (0.5, 1.0), xytext=(0, 16), xycoords="axes fraction",
                            textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    for strat, slot in ((True, 1), (False, len(heights) - 1)):
        cax = fig.add_subplot(gs[slot, 0:2])
        cb = fig.colorbar(meshes[(strat, True)], cax=cax, orientation="horizontal", extend="max", ticks=LEVELS[strat])
        cb.ax.tick_params(length=2, width=0.5)
        cb.outline.set_linewidth(0.4)
        cb.set_label("zonal-mean mole fraction / well-mixed value", labelpad=2)
        cax = fig.add_subplot(gs[slot, 2])
        cb = fig.colorbar(meshes[(strat, False)], cax=cax, orientation="horizontal", extend="both", ticks=DIFF_LEVELS[strat])
        cb.ax.tick_params(length=2, width=0.5, labelsize=6.5)
        cb.outline.set_linewidth(0.4)
        cb.set_label("difference (in units of the well-mixed value)", labelpad=2)
    fig.suptitle(f"Day {hour // 24}: {title(month, arm, resolution)}", x=0.075, ha="left",
                 fontsize=11, fontweight="bold", y=0.988)
    fig.text(0.075, 0.962, zonal_note(data["era5"]), color=INK2, fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_profiles(data, month, arm, resolution, hours, path, scale="auto", x_max=None, linear_to=2.0):
    fig, axes = plt.subplots(len(BANDS), len(TRACERS), figsize=(12.5, 8.2), sharex="col",
                             gridspec_kw=dict(left=0.055, right=0.99, top=0.84, bottom=0.07, hspace=0.12, wspace=0.16))
    z = data["era5"]
    centers = 0.5 * (z["p_edges"][1:] + z["p_edges"][:-1])
    ps = surface_pressure(z)
    tops = []
    for c, (tracer, label, sigma) in enumerate(TRACERS):
        strat = tracer == STRATOSPHERIC
        top = 0.0
        for r, (band, south, north) in enumerate(BANDS):
            ax = axes[r, c]
            ax.axvline(1, color="#c3c2b7", lw=0.8, zorder=1)
            ax.axhline(min(sigma * ps, ps), color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
            for h, color in zip(hours, WEEK_COLORS):
                for (met, _), style in zip(METS, ("-", (0, (3.2, 1.6)))):
                    p = profile(data[met], tracer, h, south, north)
                    ax.plot(p, centers, color=color, lw=1.5, ls=style, zorder=3)
                    top = max(top, float(p.max()))
            ax.set_ylim(250 if strat else 1040, 0)
            ax.grid(axis="x", color=GRID, lw=0.5)
            ax.tick_params(length=2.5, width=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{band}\npressure (hPa)")
            elif c not in (len(TRACERS) - 1,):
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(f"released in the\n{label}" if sigma == 1 else f"released at\n{label}", color=INK, pad=5)
        tops.append(top)
        axes[-1, c].set_xlabel("mole fraction / well-mixed")
    # Without advection the release layer keeps most of the tracer: a logarithmic axis shows both it and the rest.
    log_axis = scale == "log" or (scale == "auto" and max(tops[:-1]) > 6)
    clipped = False
    for c, top in enumerate(tops):
        strat = c == len(tops) - 1
        if scale == "symlog":
            # linear up to `linear_to`, logarithmic beyond; the linear part takes about 60 % of the width
            limit = next(v for v in (5, 10, 20, 50, 100, 200, 500) if v >= top * 1.02 or v == 500)
            for ax in axes[:, c]:
                ax.axvspan(linear_to, limit, color="#f1f0eb", lw=0, zorder=0)
            axes[0, c].set_xscale("symlog", linthresh=linear_to, linscale=1.5 * np.log10(limit / linear_to))
            axes[0, c].set_xlim(0, limit)
            ticks = [0, 1, 2] + [v for v in (5, 10, 100) if v < limit] + [limit]
            axes[0, c].set_xticks(ticks)
            axes[0, c].set_xticklabels([f"{v:g}" for v in ticks])
            axes[0, c].minorticks_off()
        elif log_axis:
            axes[0, c].set_xscale("log")
            axes[0, c].set_xlim(0.02, 10 ** np.ceil(np.log10(top * 1.02)))
        elif x_max and not strat and top > x_max:
            axes[0, c].set_xlim(0, x_max)
            clipped = True
        else:
            step = 10 if strat else 0.5
            axes[0, c].set_xlim(0, step * np.ceil(top / step * 1.02))
    handles = [Line2D([], [], color=col, lw=2.2, label=f"after {h // 168} week{'s' if h > 168 else ''}")
               for h, col in zip(hours, WEEK_COLORS)]
    handles += [Line2D([], [], color=INK2, lw=1.5, ls="-", label=case_label("era5", arm)),
                Line2D([], [], color=INK2, lw=1.5, ls=(0, (3.2, 1.6)), label=case_label("merra2", arm))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.925), ncol=6, handlelength=2.6, columnspacing=1.8)
    fig.suptitle(f"Mean tracer profiles by latitude band: {title(month, arm, resolution)}", x=0.055, ha="left",
                 fontsize=11, fontweight="bold", y=0.985)
    note = "Vertical line: fully mixed. Dotted line: release level. The σ = 0.07 panels show 250–0 hPa." + columns_note(z)
    if clipped:
        note += f" The peak at the release level runs off the axis (cut at {x_max:g})."
    if scale == "symlog":
        note += f" Axis linear up to {linear_to:g}, logarithmic beyond (shaded)."
    fig.text(0.055, 0.945, note, color=INK2, fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--zonal", required=True)
    p.add_argument("--figures", required=True)
    p.add_argument("--month", required=True, help="YYYYMM")
    p.add_argument("--arm", default="APC")
    p.add_argument("--resolution", default="c90")
    p.add_argument("--precision", default="32")
    p.add_argument("--section-hours", nargs="+", type=int, default=[168, 672])
    p.add_argument("--profile-hours", nargs="+", type=int, default=[168, 336, 504, 672])
    p.add_argument("--profile-scale", default="auto", choices=["auto", "linear", "log", "symlog"],
                   help="auto: logarithmic when the release layer keeps more than six times the well-mixed value; "
                        "symlog: linear up to --profile-linear-to, logarithmic beyond")
    p.add_argument("--profile-linear-to", type=float, default=2.0)
    p.add_argument("--profile-max", type=float, help="upper limit of a linear profile axis (tropospheric tracers)")
    a = p.parse_args()
    Path(a.figures).mkdir(parents=True, exist_ok=True)
    data = {met: load(a.zonal, met, a.resolution, a.month, a.arm, a.precision) for met, _ in METS}
    for hour in a.section_hours if a.profile_scale == "auto" else []:
        plot_sections(data, a.month, a.arm, a.resolution, hour, f"{a.figures}/sections_{a.month}_{a.arm}_h{hour}.png")
    suffix = "" if a.profile_scale == "auto" else f"_{a.profile_scale}"
    plot_profiles(data, a.month, a.arm, a.resolution, a.profile_hours, f"{a.figures}/profiles_{a.month}_{a.arm}{suffix}.png",
                  a.profile_scale, a.profile_max, a.profile_linear_to)


if __name__ == "__main__":
    main()
