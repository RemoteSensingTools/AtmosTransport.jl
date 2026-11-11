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
deformation_maps.pdf  flow B at T/2 and T, slotted cylinders and cosine bells: exact cell means
                      and C96 upwind, PPM (default limiter), Lin-Rood 7, lat-lon 1.875 deg PPM.
conservation.pdf      (a, b) relative change of the tracer mass (model budget) against time for
                      the uniform tracer and the Gaussian hills, flow B at C96, linear axes
                      in units of 1e-7 and 1e-8; (c) largest |chi - 1| of the uniform tracer per scheme (cubed sphere,
                      all flows and resolutions); (d) FP32 - FP64 l2 difference next to the FP64
                      l2 error per scheme (flow B, smooth tracer, cubed sphere).
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


def run_field(workdir, run_id, tracer, hour):
    with nc.Dataset(workdir / "runs" / run_id / "output.nc") as ds:
        hours = np.asarray(ds["time"][:])
        it = int(np.argmin(np.abs(hours - hour)))
        if abs(hours[it] - hour) > TIME_TOLERANCE_HOURS:
            raise ValueError(f"{run_id}: no snapshot at {hour} h (nearest {hours[it]} h)")
        a = np.asarray(ds[tracer][it, -1], dtype=np.float64)
    return a.reshape(-1)


def maps_figure(workdir, figures, cs="c96", ll="ll192x96"):
    panels = [("exact", None, cs), (f"C{cs[1:]} upwind", f"{cs}_B_upwind_f64", cs),
              (f"C{cs[1:]} PPM", f"{cs}_B_ppm_f64", cs),
              (f"C{cs[1:]} Lin–Rood 7", f"{cs}_B_linrood7_f64", cs),
              ("lat–lon 1.875° PPM", f"{ll}_B_ppm_f64", ll)]
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
            geometry[grid] = cs_polygons(src) if grid.startswith("c") else ll_polygons(src)
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
    cax = fig.add_axes([0.3, 0.075, 0.4, 0.012])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("mixing ratio (background 0.1; range of each panel in its corner)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.95, bottom=0.15, hspace=0.06, wspace=0.04)
    save(fig, figures, "deformation_maps", png_dpi=130)


# ----------------------------------------------------------------------------- conservation
def conservation_figure(rows, series, figures, grid="c96"):
    fig = plt.figure(figsize=(7.1, 4.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3, left=0.15, right=0.98, top=0.94, bottom=0.1)
    picks = ("upwind", "ppm", "linrood7")
    # (a, b) linear axes, each in the unit of its FP32 signal: FP64 stays on the zero line
    for col, (tracer, exponent) in enumerate((("uniform", -7), ("gaussian_hills", -8))):
        ax = fig.add_subplot(gs[0, col])
        for scheme in picks:
            for prec, ls, tag in (("Float64", "-", "f64"), ("Float32", "--", "f32")):
                rid = f"{grid}_B_{scheme}_{tag}"
                pts = [(r["hour"] / 24, r["budget_rel"]) for r in series if r["run_id"] == rid and r["tracer"] == tracer]
                if pts:
                    x, y = zip(*sorted(pts))
                    ax.plot(x, np.array(y) / 10.0**exponent, ls, color=COLOR[scheme],
                            lw=1.4 if prec == "Float64" else 1.0)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xlim(0, 12)
        ax.set_xticks(range(0, 13, 3))
        ax.set_xlabel("day")
        name = "uniform tracer" if tracer == "uniform" else "Gaussian hills"
        ax.set_title(f"({'ab'[col]}) tracer-mass change, {name}, flow B {grid.upper()}", fontsize=7.5,
                     color=INK, loc="left")
        ax.set_ylabel(f"relative change ($10^{{{exponent}}}$)")
        ax.tick_params(labelsize=6.5)
        _style(ax)
    handles = [plt.Line2D([], [], color=COLOR[s], lw=1.4) for s in picks]
    handles += [plt.Line2D([], [], color=INK2, ls="-", lw=1.4), plt.Line2D([], [], color=INK2, ls="--", lw=1.0)]
    fig.legend(handles, [LABEL[s] for s in picks] + ["FP64", "FP32"], loc="upper center", ncol=5,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.525))

    # (c) uniform tracer: largest |chi - 1| per scheme and precision, cubed sphere, all flows
    ax = fig.add_subplot(gs[1, 0])
    for i, (scheme, label, color) in enumerate(SCHEMES):
        for prec, mk, dy in (("Float64", "o", -0.15), ("Float32", "s", 0.15)):
            v = [r["max_abs_chi_minus_1"] for r in rows if r["family"] == "cs" and r["scheme"] == scheme
                 and r["precision"] == prec and r["tracer"] == "uniform" and r["time"] == "T" and not r["tag"]]
            if v:
                ax.plot([max(v)], [i + dy], mk, color=color, mfc=color if prec == "Float64" else "none", ms=5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(SCHEMES)))
    ax.set_yticklabels([l for _, l, _ in SCHEMES])
    ax.invert_yaxis()
    ax.set_xlabel("max |χ − 1|, cubed sphere, all flows and resolutions")
    ax.set_title("(c) uniform tracer: ● FP64, □ FP32", fontsize=7.5, color=INK, loc="left")
    ax.tick_params(labelsize=6.5)
    _style(ax)

    # (d) FP32 - FP64 next to the FP64 discretization error, flow B smooth tracer, cubed sphere
    ax = fig.add_subplot(gs[1, 1])
    row_labels = []
    for i, (scheme, label, color) in enumerate(SCHEMES):
        sel = [r for r in rows if r["family"] == "cs" and r["scheme"] == scheme and r["flow"] == "B"
               and r["tracer"] == "gaussian_hills" and r["time"] == "T" and not r["tag"]]
        for grid_label, dy in (("c48", -0.28), ("c96", 0.0), ("c192", 0.28)):
            e64 = [r["l2"] for r in sel if r["grid"] == grid_label and r["precision"] == "Float64"]
            d = [r["fp32_minus_fp64_l2"] for r in sel if r["grid"] == grid_label and r["precision"] == "Float32"]
            if e64:
                ax.plot([e64[0]], [i + dy], "o", color=color, ms=4)
            if d:
                ax.plot([d[0]], [i + dy], "s", color=color, mfc="none", ms=4)
            if e64 and d:
                ax.plot([d[0], e64[0]], [i + dy, i + dy], "-", color=color, lw=0.5)
            if d and i == 0:               # name the three rows once, at the first scheme
                row_labels.append((i + dy, grid_label.upper()))
    ax.set_xscale("log")
    ax.set_yticks(range(len(SCHEMES)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(left=ax.get_xlim()[0] * 0.1)
    for y, text in row_labels:
        ax.text(ax.get_xlim()[0] * 1.3, y, text, fontsize=5, color=INK2, ha="left", va="center")
    ax.set_xlabel("normalized ℓ2, flow B Gaussian hills at T")
    ax.set_title("(d) ● FP64 error, □ FP32 − FP64", fontsize=7.5, color=INK, loc="left")
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
