#!/usr/bin/env python3
"""Figures of the idealized advection tests (GMD paper, Sect. 4.1).

    python3 examples/paper/idealized_advection/plot_figures.py --workdir <work dir> \
        --results <results dir> --figures <figures dir>

convergence.pdf       l2 at T against grid spacing (log-log). Top row: smooth tracer (Gaussian
                      hill(s)) of flows A, B, C; bottom row: cosine bell(s), with the "minimal
                      resolution" level l2 = 0.033 of Lauritzen et al. (2012) in flow B. Colour =
                      scheme; cubed sphere: filled circles, FP64 solid, FP32 dashed; lat-lon: open
                      squares, thin line; reduced Gaussian: open triangles, thin line (FP64). All
                      at their true grid spacing. Tagged variants are left out.
deformation_maps.pdf  flow B at T/2 and T, slotted cylinders and cosine bells, all at 0.94 deg:
                      exact cell means (C96), upwind on C96 and N96, PPM (default limiter) on C96
                      and the lat-lon grid, Lin-Rood 7 on C96.
conservation.pdf      Marker shape = grid as in convergence.pdf, filled = FP64, open = FP32.
                      (a, b) relative change of the mass of a uniform tracer (model budget)
                      against time in flow B at 0.94 deg, in units of 1e-7: (a) C96, (b) lat-lon
                      and N96; (c) largest |chi - 1| of the uniform tracer per scheme and grid
                      (all flows and resolutions), with the machine epsilon of each precision
                      (dotted); (d) l2 error of the FP64 run divided by the FP32 - FP64 l2 difference:
                      smallest ratio per scheme and grid (all flows, resolutions, non-uniform
                      tracers, T/2 and T).
tilt.pdf              flow A l2 at T against the tilt of the rotation axis (0, 45, 90 degrees).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from analyze import is_current     # same directory: finished run with a current fingerprint

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
# Categorical slots in fixed order (validated reference palette, adjacent pairs).
SCHEMES = [("upwind", "upwind", "#2a78d6"), ("slopes", "slopes", "#eb6834"),
           ("ppm", "PPM (default)", "#1baf7a"), ("ppm_cw84", "PPM CW84", "#eda100"),
           ("linrood5", "Lin–Rood 5", "#e87ba4"), ("linrood7", "Lin–Rood 7", "#008300")]
COLOR = {k: c for k, _, c in SCHEMES}
LABEL = {k: l for k, l, _ in SCHEMES}
BLUES = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1d4f8f", "#143766"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)
TITLES = {"A": "A: solid-body rotation (45°)", "B": "B: deformational, non-divergent",
          "C": "C: deformational, divergent"}
SMOOTH = {"A": "gaussian_hill", "B": "gaussian_hills", "C": "gaussian_hills"}
BELLS = {"A": "cosine_bell", "B": "cosine_bells", "C": "cosine_bells"}
FAMILY_STYLE = {"cs": dict(marker="o", ms=4, lw=1.6, mfc=None),
                "ll": dict(marker="s", ms=5, lw=0.8, mfc="none"),
                "rg": dict(marker="^", ms=6, lw=0.8, mfc="none")}
FAMILY_LABEL = {"cs": "cubed sphere", "ll": "lat–lon", "rg": "reduced Gaussian"}
MINIMAL_RESOLUTION_L2 = 0.033           # Lauritzen et al. (2012), cosine bells of flow B
TIME_TOLERANCE_HOURS = 0.01
HOURLY_WINDOW = 1.0         # window length [h] of the untagged cases (24 windows per day)
SPACING_TICKS = [0.46875, 0.9375, 1.875, 3.75]

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "pdf.fonttype": 42,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6})


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.4)
    ax.set_facecolor(SURFACE)


def save(fig, figures, name, png_dpi=200):
    fig.savefig(figures / f"{name}.pdf", dpi=200, facecolor=SURFACE)
    fig.savefig(figures / f"{name}.png", dpi=png_dpi, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- convergence
def _convergence_panel(ax, rows, flow, tracer, wind_error=None):
    sel = [r for r in rows if r["flow"] == flow and r["tracer"] == tracer and r["time"] == "T" and not r["tag"]]
    for key, _, color in SCHEMES:
        for family, st in FAMILY_STYLE.items():
            for prec, ls in (("Float64", "-"), ("Float32", "--")):
                if family != "cs" and prec == "Float32":
                    continue
                pts = sorted((r["spacing_deg"], r["l2"]) for r in sel
                             if r["scheme"] == key and r["precision"] == prec and r["family"] == family)
                if not pts:
                    continue
                x, y = zip(*pts)
                ax.plot(x, y, ls, color=color, lw=st["lw"] if prec == "Float64" else 0.9, marker=st["marker"],
                        ms=st["ms"], mfc=st["mfc"] or color, mec=color, mew=1.0)
    c = [r["l2"] for r in sel if r["family"] == "cs" and r["precision"] == "Float64"
         and abs(r["spacing_deg"] - 1.875) < 1e-6]
    if c:                                  # reference slopes below the best and the worst C48 error
        x = np.array([0.46875, 1.875])
        for order, y0 in ((1, 0.55 * max(c)), (2, 0.45 * min(c))):
            ax.plot(x, y0 * (x / 1.875) ** order, ":", color=INK2, lw=0.7)
            if order == 1:                 # left end: free space between upwind and the rest
                ax.text(0.48, y0 * (0.48 / 1.875) * 1.12, "order 1", fontsize=6.5, color=INK2, ha="left", va="bottom")
            else:
                ax.text(1.95, y0, "order 2", fontsize=6.5, color=INK2, ha="left", va="center")
    if wind_error is not None:             # error of hourly window-mean winds (forcing_window_error.jl)
        ax.plot([0.38, 1.2], [wind_error, wind_error], "-", color=INK2, lw=0.6)
        ax.text(0.40, wind_error * 0.9, "hourly-wind error", fontsize=6, color=INK2, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if wind_error is not None:             # keep its label inside the panel
        ax.set_ylim(bottom=min(ax.get_ylim()[0], 0.45 * wind_error))
    ax.set_xticks(SPACING_TICKS)
    ax.set_xticklabels(["0.47°", "0.94°", "1.875°", "3.75°"])
    ax.minorticks_off()
    ax.set_xlim(0.38, 4.6)
    _style(ax)


def convergence_figure(rows, figures, wind_errors):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.6), sharey="row", facecolor=SURFACE)
    for col, flow in enumerate("ABC"):
        _convergence_panel(axes[0, col], rows, flow, SMOOTH[flow], wind_errors.get((flow, SMOOTH[flow])))
        _convergence_panel(axes[1, col], rows, flow, BELLS[flow], wind_errors.get((flow, BELLS[flow])))
        axes[0, col].set_title(TITLES[flow], fontsize=8, color=INK, loc="left")
        axes[0, col].set_xticklabels([])
        axes[1, col].set_xlabel("grid spacing")
    ax = axes[1, 1]
    ax.axhline(MINIMAL_RESOLUTION_L2, color=INK, lw=0.8, ls="-.")
    ax.text(4.4, MINIMAL_RESOLUTION_L2 * 0.88, "ℓ2 = 0.033", fontsize=6.5, color=INK, ha="right", va="top")
    axes[0, 0].set_ylabel("ℓ2 at T, Gaussian hill(s)")
    axes[1, 0].set_ylabel("ℓ2 at T, cosine bell(s)")
    handles = [plt.Line2D([], [], color=c, lw=1.6) for _, _, c in SCHEMES]
    labels = [l for _, l, _ in SCHEMES]
    handles += [plt.Line2D([], [], color=INK2, ls="none", marker=st["marker"], mfc=st["mfc"] or INK2, mec=INK2)
                for st in FAMILY_STYLE.values()]
    labels += list(FAMILY_LABEL.values())
    handles += [plt.Line2D([], [], color=INK2, ls="-", lw=1.6), plt.Line2D([], [], color=INK2, ls="--", lw=0.9)]
    labels += ["FP64", "FP32 (CS, under FP64)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7, handlelength=2.2,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.22, wspace=0.08, hspace=0.12)
    save(fig, figures, "convergence")


# ----------------------------------------------------------------------------- maps
def cs_polygons(path):
    with nc.Dataset(path) as ds:
        lon = np.asarray(ds["corner_lons"][:]) % 360       # (nf, Ycorner, Xcorner)
        lat = np.asarray(ds["corner_lats"][:])
    polys, keep = [], []
    nf, nyc, nxc = lon.shape
    for p in range(nf):
        for j in range(nyc - 1):
            for i in range(nxc - 1):
                lo = np.array([lon[p, j, i], lon[p, j, i + 1], lon[p, j + 1, i + 1], lon[p, j + 1, i]])
                la = np.array([lat[p, j, i], lat[p, j, i + 1], lat[p, j + 1, i + 1], lat[p, j + 1, i]])
                ok = lo.max() - lo.min() < 180
                keep.append(ok)
                if ok:
                    polys.append(np.column_stack([lo, la]))
    return polys, np.array(keep)


def ll_polygons(path):
    with nc.Dataset(path) as ds:
        lonb = np.asarray(ds["lon_bounds"][:])
        latb = np.asarray(ds["lat_bounds"][:])
    # (lon, nv) in the file's Julia order is (nv, lon) here; put the 2 bounds last.
    lonb = (lonb.T if lonb.shape[0] == 2 else lonb) % 360
    latb = latb.T if latb.shape[0] == 2 else latb
    polys, keep = [], []
    for j in range(latb.shape[0]):
        for i in range(lonb.shape[0]):
            a, b = lonb[i]
            if b <= a:
                b += 360
            ok = b - a < 180 and b <= 360
            keep.append(ok)
            if ok:
                polys.append(np.array([[a, latb[j, 0]], [b, latb[j, 0]], [b, latb[j, 1]], [a, latb[j, 1]]]))
    return polys, np.array(keep)


def rg_polygons(path):
    with nc.Dataset(path) as ds:
        lonb = np.asarray(ds["cell_lon_bounds"][:])        # corners SW, SE, NE, NW
        latb = np.asarray(ds["cell_lat_bounds"][:])
    lonb = (lonb.T if lonb.shape[0] == 4 else lonb) % 360  # (cell, 4)
    latb = latb.T if latb.shape[0] == 4 else latb
    east = np.where(lonb[:, 1] <= lonb[:, 0], lonb[:, 1] + 360, lonb[:, 1])
    keep = (east - lonb[:, 0] < 180) & (east <= 360)
    polys = [np.array([[w, s], [e, s], [e, n], [w, n]])
             for w, e, s, n in zip(lonb[keep, 0], east[keep], latb[keep, 0], latb[keep, 2])]
    return polys, keep


POLYGONS = {"c": cs_polygons, "l": ll_polygons, "n": rg_polygons}       # by first letter of the grid name


def run_field(workdir, run_id, tracer, hour):
    with nc.Dataset(workdir / "runs" / run_id / "output.nc") as ds:
        hours = np.asarray(ds["time"][:])
        it = int(np.argmin(np.abs(hours - hour)))
        if abs(hours[it] - hour) > TIME_TOLERANCE_HOURS:
            raise ValueError(f"{run_id}: no snapshot at {hour} h (nearest {hours[it]} h)")
        a = np.asarray(ds[tracer][it, -1], dtype=np.float64)
    return a.reshape(-1)


def maps_figure(workdir, figures, cs="c96", ll="ll384x192", rg="n96"):
    """All three grids at the same equatorial spacing (0.94 degrees for the defaults)."""
    panels = [("exact", None, cs), (f"C{cs[1:]} upwind", f"{cs}_B_upwind_f64", cs),
              (f"N{rg[1:]} upwind", f"{rg}_B_upwind_f64", rg),
              (f"C{cs[1:]} PPM", f"{cs}_B_ppm_f64", cs),
              ("lat–lon PPM", f"{ll}_B_ppm_f64", ll),
              (f"C{cs[1:]} Lin–Rood 7", f"{cs}_B_linrood7_f64", cs)]
    missing = [p[1] for p in panels if p[1] is not None and not is_current(workdir / "runs" / p[1])]
    if missing:
        raise RuntimeError(f"runs not finished or stale: {missing}")
    tracers = (("slotted_cylinders", "slotted cylinders"), ("cosine_bells", "cosine bells"))
    exact = {}
    with nc.Dataset(workdir / "inputs" / f"{cs}_B" / "reference.nc") as ref:
        for tracer, _ in tracers:
            exact[tracer] = {t: np.asarray(ref[tracer][t]) for t in (1, 2)}
        period = float(ref["time"][2])
    geometry = {}
    fig, axes = plt.subplots(len(panels), 4, figsize=(7.1, 1.0 * len(panels) + 0.75), facecolor=SURFACE)
    norm = Normalize(0.0, 1.05)
    for row, (title, run_id, grid) in enumerate(panels):
        if grid not in geometry:
            src = workdir / "runs" / (run_id or f"{cs}_B_upwind_f64") / "output.nc"
            geometry[grid] = POLYGONS[grid[0]](src)
        polys, keep = geometry[grid]
        for block, (tracer, tname) in enumerate(tracers):
            for k, (t, label) in enumerate(((1, "T/2"), (2, "T"))):
                col = 2 * block + k
                ax = axes[row, col]
                values = exact[tracer][t] if run_id is None else run_field(workdir, run_id, tracer, period * t / 2)
                pc = PolyCollection(polys, array=values[keep], cmap=SEQ, norm=norm, edgecolors="none",
                                    antialiased=False)
                pc.set_rasterized(True)             # cell fills as an image, axes and text stay vector
                ax.add_collection(pc)
                ax.set_xlim(0, 360)
                ax.set_ylim(-90, 90)
                ax.set_aspect("equal")
                ax.set_xticks([90, 180, 270])
                ax.set_yticks([-60, 0, 60])
                ax.tick_params(labelsize=6, length=2)
                if row < len(panels) - 1:
                    ax.set_xticklabels([])
                if col > 0:
                    ax.set_yticklabels([])
                ax.text(0.98, 0.04, f"{values.min():.2f} … {values.max():.2f}", transform=ax.transAxes,
                        fontsize=5.5, ha="right", va="bottom", color=INK2,
                        bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
                if row == 0:
                    ax.set_title(f"{tname}, {label}", fontsize=7, color=INK)
        axes[row, 0].text(-0.27, 0.5, title, transform=axes[row, 0].transAxes, fontsize=6.5, color=INK,
                          rotation=90, ha="center", va="center")
    cax = fig.add_axes([0.3, 0.065, 0.4, 0.010])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("mixing ratio (background 0.1; range of each panel in its corner)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.96, bottom=0.13, hspace=0.06, wspace=0.04)
    save(fig, figures, "deformation_maps", png_dpi=130)


# ----------------------------------------------------------------------------- conservation
FAMILY_GRID = {"cs": "c96", "ll": "ll384x192", "rg": "n96"}       # the three grids at 0.94 degrees
FAMILY_OFFSET = {"cs": -0.27, "ll": 0.0, "rg": 0.27}             # rows of one scheme in panels (c, d)


def conservation_figure(rows, series, figures):
    fig = plt.figure(figsize=(7.1, 4.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.3, left=0.15, right=0.98, top=0.945, bottom=0.095)
    marker = {family: st["marker"] for family, st in FAMILY_STYLE.items()}
    # (a, b) uniform tracer in flow B at 0.94 degrees, linear axes in units of 1e-7: FP64 stays on the zero line
    exponent = -7
    # per line: grid family, scheme, label of the FP32 line and its place (day, offset in points, ha, va)
    panels = (("(a) tracer-mass change, uniform tracer, flow B, C96",
               [("cs", "upwind", "upwind", (9, (0, -6), "center", "top")),
                ("cs", "ppm", "PPM", (9, (-4, 5), "right", "bottom")),
                ("cs", "linrood7", "Lin–Rood 7", (9, (-8, -2), "right", "top"))]),
              ("(b) the same on lat–lon 0.94° and N96",
               [("ll", "upwind", "lat–lon upwind", (4.5, (8, 3), "left", "bottom")),
                ("ll", "ppm", "lat–lon PPM", (9, (-4, 5), "right", "bottom")),
                ("rg", "upwind", "N96 upwind", (10.5, (0, -7), "center", "top"))]))
    for col, (title, picks) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        for family, scheme, text, (day, offset, ha, va) in picks:
            for prec, ls, tag in (("Float64", "-", "f64"), ("Float32", "--", "f32")):
                rid = f"{FAMILY_GRID[family]}_B_{scheme}_{tag}"
                pts = [(r["hour"] / 24, r["budget_rel"]) for r in series if r["run_id"] == rid and r["tracer"] == "uniform"]
                if not pts:
                    raise RuntimeError(f"{rid}: no uniform-tracer mass series in mass_series.csv")
                x, y = zip(*sorted(pts))
                fp64 = prec == "Float64"
                ax.plot(x, np.array(y) / 10.0**exponent, ls, color=COLOR[scheme], lw=1.4 if fp64 else 1.0,
                        marker=None if fp64 else marker[family], markevery=(6, 6), ms=4.5, mfc=SURFACE,
                        mec=COLOR[scheme], mew=0.9)
                if not fp64:               # name the line itself: two lines of (b) share the upwind colour
                    k = int(np.argmin(np.abs(np.array(x) - day)))
                    ax.annotate(text, (x[k], y[k] / 10.0**exponent), xytext=offset, textcoords="offset points",
                                fontsize=6.5, color=INK, ha=ha, va=va)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xlim(0, 12)
        ax.set_xticks(range(0, 13, 3))
        ax.set_xlabel("day")
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.set_ylabel(f"relative change ($10^{{{exponent}}}$)")
        ax.tick_params(labelsize=6.5)
        _style(ax)
        if col == 0:                       # the unit of the axis as a mole fraction of CO2
            ax.text(0.03, 0.05, f"$10^{{{exponent}}}$ of a 420 ppm tracer = {420 * 10.0**exponent * 1e3:.2f} ppb",
                    transform=ax.transAxes, fontsize=6.5, color=INK2, ha="left", va="bottom")
    schemes = ("upwind", "ppm", "linrood7")
    handles = [plt.Line2D([], [], color=COLOR[s], lw=1.4) for s in schemes]
    handles += [plt.Line2D([], [], color=INK2, ls="-", lw=1.4), plt.Line2D([], [], color=INK2, ls="--", lw=1.0)]
    handles += [plt.Line2D([], [], color=INK2, ls="none", marker=marker[f], mfc="none", mec=INK2, ms=4.5)
                for f in FAMILY_STYLE]
    labels = [LABEL[s] for s in schemes] + ["FP64", "FP32"] + [FAMILY_LABEL[f] for f in FAMILY_STYLE]
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, fontsize=6.5, handlelength=1.8,
               columnspacing=1.0, handletextpad=0.5, bbox_to_anchor=(0.5, 0.535))

    # (c) uniform tracer: largest |chi - 1| per scheme, grid and precision, all flows and resolutions
    ax = fig.add_subplot(gs[1, 0])
    for i, (scheme, label, color) in enumerate(SCHEMES):
        for family, dy in FAMILY_OFFSET.items():
            for prec in ("Float64", "Float32"):
                v = [r["max_abs_chi_minus_1"] for r in rows if r["family"] == family and r["scheme"] == scheme
                     and r["precision"] == prec and r["tracer"] == "uniform" and r["time"] == "T" and not r["tag"]]
                if v:
                    ax.plot([max(v)], [i + dy], marker[family], color=color,
                            mfc=color if prec == "Float64" else "none", ms=4.5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(SCHEMES)))
    ax.set_yticklabels([l for _, l, _ in SCHEMES])
    ax.set_ylim(len(SCHEMES) - 0.45, -0.55)
    # machine epsilon of each precision: the symbols of a precision lie a similar factor to its right
    for prec, eps in (("FP64", np.finfo(np.float64).eps), ("FP32", np.finfo(np.float32).eps)):
        ax.axvline(eps, color=INK2, lw=0.7, ls=":")
        ax.text(eps * 1.6, len(SCHEMES) - 0.6, f"{prec} machine epsilon", fontsize=6, color=INK2, rotation=90,
                ha="left", va="bottom")
    ax.set_xlim(left=np.finfo(np.float64).eps / 4)
    ax.set_xticks(10.0 ** np.arange(-15, -2, 3))
    ax.set_xlabel("max |χ − 1|, all flows and resolutions")
    ax.set_title("(c) uniform tracer: filled FP64, open FP32", fontsize=7.5, color=INK, loc="left")
    ax.tick_params(labelsize=6.5)
    _style(ax)

    # (d) how many times the FP32 - FP64 difference is below the discretization error: smallest ratio per
    # scheme and grid over all flows, resolutions, non-uniform tracers and both times (T/2, T)
    ax = fig.add_subplot(gs[1, 1])
    for i, (scheme, label, color) in enumerate(SCHEMES):
        for family, dy in FAMILY_OFFSET.items():
            v = [r["fp64_l2_same_case"] / r["fp32_minus_fp64_l2"] for r in rows
                 if r["family"] == family and r["scheme"] == scheme and r["precision"] == "Float32"
                 and r["tracer"] != "uniform" and not r["tag"] and r["fp32_minus_fp64_l2"] > 0]
            if v:
                ax.plot([min(v)], [i + dy], marker[family], color=color, ms=4.5)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1e6)
    ax.axvline(1, color=INK2, lw=0.7, ls=":")
    ax.text(1.35, len(SCHEMES) - 0.6, "precision error = discretization error", fontsize=6, color=INK2,
            rotation=90, ha="left", va="bottom")
    ax.set_yticks(range(len(SCHEMES)))
    ax.set_yticklabels([])
    ax.set_ylim(len(SCHEMES) - 0.45, -0.55)
    ax.set_xlabel("smallest ratio, all flows and resolutions")
    ax.set_title("(d) discretization error ÷ FP32 − FP64 difference", fontsize=7.5, color=INK, loc="left")
    ax.tick_params(labelsize=6.5)
    _style(ax)
    save(fig, figures, "conservation")


# ----------------------------------------------------------------------------- tilt
def tilt_figure(tilt, figures):
    series = [("c96", "ppm", "C96 PPM"), ("c96", "linrood7", "C96 Lin–Rood 7"),
              ("ll192x96", "ppm", "lat–lon 1.875° PPM"), ("n48", "upwind", "N48 upwind")]
    family_marker = {"c": "o", "l": "s", "n": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), facecolor=SURFACE)
    for ax, tracer, title in ((axes[0], "gaussian_hill", "Gaussian hill"), (axes[1], "cosine_bell", "cosine bell")):
        for grid, scheme, label in series:
            pts = sorted((r["tilt_deg"], r["l2"]) for r in tilt
                         if r["grid"] == grid and r["scheme"] == scheme and r["tracer"] == tracer)
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, "-", color=COLOR[scheme], marker=family_marker[grid[0]], ms=5,
                    mfc="none" if grid[0] != "c" else COLOR[scheme], lw=1.2)
            ax.annotate(label, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                        color=INK, va="center")
        ax.set_yscale("log")
        ax.set_xticks([0, 45, 90])
        ax.set_xlim(-8, 135)
        ax.set_xlabel("tilt of the rotation axis (°)")
        ax.set_title(f"flow A, {title}, ℓ2 at T (FP64)", fontsize=7.5, color=INK, loc="left")
        _style(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.18, wspace=0.25)
    save(fig, figures, "tilt")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workdir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--figures", required=True)
    a = p.parse_args()
    workdir, results, figures = Path(a.workdir), Path(a.results), Path(a.figures)
    figures.mkdir(parents=True, exist_ok=True)
    runs = [Path(line.strip()).parent for line in open(workdir / "runs" / "runs.txt") if line.strip()]
    stale = [run.name for run in runs if not is_current(run)]
    if stale:
        raise RuntimeError(f"runs not finished or stale (rerun run_matrix.sh run and analyze): {stale[:5]}")
    newest = max((run / "fingerprint").stat().st_mtime for run in runs)
    if (results / "norms.csv").stat().st_mtime < newest:
        raise RuntimeError("results/norms.csv is older than the newest run; rerun run_matrix.sh analyze")
    rows = read_csv(results / "norms.csv")
    for r in rows:
        r["tag"] = "" if r["tag"] in ("", None) or (isinstance(r["tag"], float) and np.isnan(r["tag"])) else r["tag"]
    wind_file = results / "forcing_window_error.csv"        # written by forcing_window_error.jl
    wind_errors = {(r["flow"], r["tracer"]): r["l2"] for r in (read_csv(wind_file) if wind_file.exists() else [])
              if r["time"] == "T" and r["window_hours"] == HOURLY_WINDOW}
    convergence_figure(rows, figures, wind_errors)
    maps_figure(workdir, figures)
    conservation_figure(rows, read_csv(results / "mass_series.csv"), figures)
    tilt_figure(read_csv(results / "tilt.csv"), figures)
    print(f"wrote figures to {figures}")


if __name__ == "__main__":
    main()
