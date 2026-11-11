#!/usr/bin/env python3
"""Error norms, extrema, mass drift and FP32 - FP64 differences of the idealized advection runs.

    python3 examples/paper/idealized_advection/analyze.py --workdir <work dir> --results <dir>

Reads the runs listed in <workdir>/runs/runs.txt ({run.toml, output.nc}; every run must have
finished with a current fingerprint, see run_matrix.sh) and <workdir>/inputs/<case>/reference.nc
and writes
  norms.csv          one row per run x tracer x time (T/2, T): l1, l2, linf against the analytic
                     cell means (area weighted, Williamson et al. 1992 normalization), extrema,
                     normalized under/overshoot (Lauritzen et al. 2012; not for the uniform
                     tracer, whose initial range is rounding), layer identity, mass drift
                     (budget series and fields), air-mass drift, max |chi - 1| of the uniform
                     tracer, and the FP32 - FP64 difference next to the error of the FP64 run;
  convergence.csv    observed order between consecutive resolutions of each grid family (cubed
                     sphere, lat-lon, reduced Gaussian) from the actual grid-spacing ratio
                     (untagged cases);
  minimal_resolution.csv  grid spacing at which flow B's cosine bells reach l2 = 0.033 (Lauritzen
                     et al. 2012, Sect. 3.2), log-log interpolated between the bracketing
                     resolutions;
  summary.csv        per grid family x scheme: observed orders, minimal resolution, extrema,
                     mass drift, FP32 - FP64 difference;
  table_convergence.tex  LaTeX tabular body for the paper (cubed sphere, schemes as rows);
  substep_sensitivity.csv  `cfl*` variants (a smaller Courant number) next to the base case;
  window_sensitivity.csv   `w*` variants (shorter met windows) next to the base case, with the
                     observed order between consecutive resolutions of the variant;
  tilt.csv           flow A at the rotation-axis tilts 0, 45 and 90 degrees (`tilt*` variants);
  mass_series.csv    relative tracer-mass change of every snapshot (budget output and fields).
The level used for the norms is the bottom output level; the top output level is compared
with it ("layer_diff"), as all layers are the same 2-D test.
"""
import argparse
import csv
import hashlib
import math
import os
try:
    import tomllib
except ImportError:          # Python < 3.11
    import tomli as tomllib
from pathlib import Path

import numpy as np
import netCDF4 as nc

HALF, FULL = 1, 2           # reference time indices (0, T/2, T)
TIME_TOLERANCE_HOURS = 0.01 # a snapshot must be this close to 0, T/2 and T


def snapshot_index(hours, target, run_id):
    """Index of the snapshot at `target` hours; an error if there is none within the tolerance."""
    i = int(np.argmin(np.abs(hours - target)))
    if abs(hours[i] - target) > TIME_TOLERANCE_HOURS:
        raise ValueError(f"{run_id}: no snapshot at {target} h (nearest {hours[i]} h)")
    return i


def flat(var, it):
    """Field at snapshot `it`, shape (level, cell) in the reference's cell order."""
    a = np.asarray(var[it], dtype=np.float64)
    return a.reshape(a.shape[0], -1)


def norms(q, ref, area):
    e = q - ref
    l1 = (area * np.abs(e)).sum() / (area * np.abs(ref)).sum()
    l2 = math.sqrt((area * e**2).sum() / (area * ref**2).sum())
    linf = np.abs(e).max() / np.abs(ref).max()
    return l1, l2, linf


def fingerprint(run_dir, case_file):
    """Hash of the run's config and of its case file, as `fingerprint` in run_matrix.sh computes it."""
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest() + "  -\n"
    text = digest(run_dir / "config.toml") + digest(case_file) + f"{int(os.stat(case_file).st_mtime)}\n"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def is_current(run_dir):
    """True if the run finished and its stored fingerprint matches its config and case file."""
    log, stored, meta = run_dir / "log.txt", run_dir / "fingerprint", run_dir / "run.toml"
    if not ((run_dir / "output.nc").exists() and log.exists() and stored.exists() and meta.exists()):
        return False
    case_file = tomllib.load(open(meta, "rb"))["case_file"]
    return ("Forward run wall" in log.read_text(errors="ignore") and os.path.exists(case_file)
            and stored.read_text().strip() == fingerprint(run_dir, case_file))


def load_run(run_dir):
    if not is_current(run_dir):
        raise RuntimeError(f"{run_dir.name}: run not finished or stale; rerun run_matrix.sh run")
    meta = tomllib.load(open(run_dir / "run.toml", "rb"))
    log = run_dir / "log.txt"
    meta["dir"] = run_dir
    meta["wall_s"] = _wall_seconds(log.read_text(errors="ignore"))
    return meta


def _wall_seconds(text):
    for line in text.splitlines():
        if "Forward run wall" in line:
            tok = line.split("Forward run wall")[1].split()[0]
            return float(tok.rstrip("s"))
    return float("nan")


def analyze_run(meta, workdir):
    ref = nc.Dataset(workdir / "inputs" / meta["case"] / "reference.nc")
    out = nc.Dataset(meta["dir"] / "output.nc")
    area = np.asarray(ref["cell_area"][:], dtype=np.float64)
    hours = np.asarray(out["time"][:])
    period = float(ref["time"][FULL])
    idx = {t: snapshot_index(hours, h, meta["run_id"]) for t, h in ((0, 0.0), (HALF, period / 2), (FULL, period))}
    air = np.asarray(out["air_mass"][:], dtype=np.float64)
    air = air.reshape(air.shape[0], air.shape[1], -1)                    # (time, level, cell)
    air_total = air.sum(axis=2)
    air_drift = np.abs(air_total / air_total[0] - 1).max()
    rows, series, fields = [], [], {}
    for tracer in meta["tracers"]:
        var = out[tracer]
        budget = np.asarray(out[f"{tracer}_total_mass"][:], dtype=np.float64)
        q_all = np.asarray(var[:], dtype=np.float64)
        q_all = q_all.reshape(q_all.shape[0], q_all.shape[1], -1)
        field_mass = (q_all * air).sum(axis=2)                           # per output level
        budget_rel = budget / budget[0] - 1
        field_rel = field_mass[:, -1] / field_mass[0, -1] - 1
        for h, b, f in zip(hours, budget_rel, field_rel):
            series.append(dict(run_id=meta["run_id"], tracer=tracer, hour=h, budget_rel=b, field_rel=f))
        q0_ref = np.asarray(ref[tracer][0], dtype=np.float64)
        ic_error = np.abs(q_all[idx[0], -1] - q0_ref).max()
        for t_ref, label in ((HALF, "T/2"), (FULL, "T")):
            q = q_all[idx[t_ref]]
            r = np.asarray(ref[tracer][t_ref], dtype=np.float64)
            uniform = tracer == "uniform"
            l1, l2, linf = norms(q[-1], r, area) if not uniform else (np.nan,) * 3
            fields[(tracer, label)] = q[-1]
            span = q0_ref.max() - q0_ref.min()
            rows.append(dict(
                run_id=meta["run_id"], case=meta["case"], grid=meta["grid"], flow=meta["flow"],
                tag=meta.get("tag", ""), courant=meta.get("courant", 0.5),
                windows_per_day=meta.get("windows_per_day", 24),
                family=grid_family(meta["grid"]), spacing_deg=grid_spacing(meta["grid"]),
                tilt_deg=meta.get("tilt_deg", np.nan),
                scheme=meta["scheme"], precision=meta["precision"], tracer=tracer, time=label,
                l1=l1, l2=l2, linf=linf,
                phi_min=q[-1].min(), phi_max=q[-1].max(),
                undershoot=(q[-1].min() - r.min()) / span if not uniform else np.nan,
                overshoot=(q[-1].max() - r.max()) / span if not uniform else np.nan,
                max_abs_chi_minus_1=np.abs(q_all[: idx[t_ref] + 1] - 1).max() if uniform else np.nan,
                layer_diff=np.abs(q[0] - q[-1]).max(),
                ic_error=ic_error,
                mass_drift_budget_max=np.abs(budget_rel[: idx[t_ref] + 1]).max(),
                mass_drift_budget=budget_rel[idx[t_ref]],
                mass_drift_fields_max=np.abs(field_rel[: idx[t_ref] + 1]).max(),
                air_mass_drift_max=air_drift,
                wall_s=meta["wall_s"]))
    out.close()
    ref.close()
    return rows, series, fields


def add_precision_differences(rows, fields):
    """FP32 - FP64 difference (l2, linf, normalized like the error norms) per matching run pair."""
    by_key = {(r["case"], r["scheme"], r["precision"], r["tracer"], r["time"]): r for r in rows}
    for r in rows:
        r["fp32_minus_fp64_l2"] = r["fp32_minus_fp64_linf"] = np.nan
        r["fp64_l2_same_case"] = np.nan
        if r["precision"] != "Float32":
            continue
        key64 = (r["case"], r["scheme"], "Float64", r["tracer"], r["time"])
        if key64 not in by_key:
            continue
        q32 = fields[(r["run_id"], r["tracer"], r["time"])]
        q64 = fields[(by_key[key64]["run_id"], r["tracer"], r["time"])]
        area = fields[("area", r["case"])]
        scale_l2 = math.sqrt((area * q64**2).sum() / area.sum())
        d = q32 - q64
        r["fp32_minus_fp64_l2"] = math.sqrt((area * d**2).sum() / area.sum()) / scale_l2
        r["fp32_minus_fp64_linf"] = np.abs(d).max() / np.abs(q64).max()
        r["fp64_l2_same_case"] = by_key[key64]["l2"]


MINIMAL_RESOLUTION_L2 = 0.033          # Lauritzen et al. (2012), flow B cosine bells
SCHEME_ORDER = ("upwind", "slopes", "ppm", "ppm_cw84", "linrood5", "linrood7")
SCHEME_TEX = {"upwind": "Upwind", "slopes": "Slopes", "ppm": "PPM, default limiter",
              "ppm_cw84": "PPM, CW84", "linrood5": "Lin--Rood, order 5", "linrood7": "Lin--Rood, order 7"}
FAMILIES = ("cs", "ll", "rg")
SMOOTH = {"A": "gaussian_hill", "B": "gaussian_hills", "C": "gaussian_hills"}


def grid_family(label):
    """'cs' (c<N>), 'll' (ll<Nx>x<Ny>) or 'rg' (n<N>)."""
    return "ll" if label.startswith("ll") else ("cs" if label.startswith("c") else "rg")


def grid_spacing(label):
    """Nominal grid spacing [degrees]: 90/N on C<N>, 360/Nx on lat-lon, 90/N on N<N>."""
    if label.startswith("ll"):
        return 360.0 / int(label[2:].split("x")[0])
    return 90.0 / int(label[1:])


def convergence(rows):
    """Observed order log(e_coarse/e_fine)/log(h_coarse/h_fine) between consecutive resolutions."""
    groups = {}
    for r in rows:
        if r["tag"] or r["tracer"] == "uniform":
            continue
        key = (r["family"], r["flow"], r["scheme"], r["precision"], r["tracer"], r["time"])
        groups.setdefault(key, []).append(r)
    out = []
    for (family, flow, scheme, prec, tracer, time), members in sorted(groups.items()):
        members = sorted(members, key=lambda r: -r["spacing_deg"])
        for coarse, fine in zip(members, members[1:]):
            ratio = coarse["spacing_deg"] / fine["spacing_deg"]
            row = dict(family=family, flow=flow, scheme=scheme, precision=prec, tracer=tracer, time=time,
                       coarse=coarse["grid"], fine=fine["grid"], spacing_ratio=ratio)
            for n in ("l1", "l2", "linf"):
                row[f"{n}_coarse"], row[f"{n}_fine"] = coarse[n], fine[n]
                row[f"order_{n}"] = (math.log(coarse[n] / fine[n]) / math.log(ratio)
                                     if coarse[n] > 0 and fine[n] > 0 else np.nan)
            out.append(row)
    return out


def minimal_resolution(rows):
    """Grid spacing where flow B's cosine-bell l2 at T equals 0.033 (FP64), per family and scheme."""
    out = []
    for family in FAMILIES:
        for scheme in SCHEME_ORDER:
            pts = sorted(((r["spacing_deg"], r["l2"], r["grid"]) for r in rows
                          if r["family"] == family and r["scheme"] == scheme and r["flow"] == "B"
                          and r["tracer"] == "cosine_bells" and r["time"] == "T"
                          and r["precision"] == "Float64" and not r["tag"]), reverse=True)
            if not pts:
                continue
            row = dict(family=family, scheme=scheme, grids=" ".join(p[2] for p in pts),
                       l2=" ".join(f"{p[1]:.4f}" for p in pts), minimal_spacing_deg=np.nan, note="")
            for (h1, e1, g1), (h2, e2, g2) in zip(pts, pts[1:]):
                if e1 > MINIMAL_RESOLUTION_L2 >= e2:
                    w = math.log(MINIMAL_RESOLUTION_L2 / e1) / math.log(e2 / e1)
                    row["minimal_spacing_deg"] = math.exp(math.log(h1) + w * math.log(h2 / h1))
                    row["note"] = f"between {g1} and {g2}"
                    break
            else:
                if pts[-1][1] > MINIMAL_RESOLUTION_L2:
                    row["note"] = f"finer than {pts[-1][2]} ({pts[-1][0]:.3g} deg)"
                elif pts[0][1] <= MINIMAL_RESOLUTION_L2:
                    row["note"] = f"coarser than {pts[0][2]} ({pts[0][0]:.3g} deg)"
            out.append(row)
    return out


def _finest_order(conv, family, flow, scheme, norm, precision="Float64"):
    c = [r for r in conv if r["family"] == family and r["flow"] == flow and r["scheme"] == scheme
         and r["tracer"] == SMOOTH[flow] and r["time"] == "T" and r["precision"] == precision]
    if not c:
        return np.nan, ""
    best = min(c, key=lambda r: grid_spacing(r["fine"]))
    return best[f"order_{norm}"], f"{best['coarse']}-{best['fine']}"


def summary(rows, conv, minres):
    """One row per grid family x scheme."""
    out = []
    for family in FAMILIES:
        for scheme in SCHEME_ORDER:
            sel = [r for r in rows if r["family"] == family and r["scheme"] == scheme and not r["tag"]]
            if not sel:
                continue
            row = dict(family=family, scheme=scheme)
            for flow in "ABC":
                for norm in ("l2", "linf"):
                    row[f"order_{norm}_{flow}"], pair = _finest_order(conv, family, flow, scheme, norm)
                row[f"pair_{flow}"] = pair
                fine = [r for r in sel if r["flow"] == flow and r["tracer"] == SMOOTH[flow] and r["time"] == "T"
                        and r["precision"] == "Float64"]
                best = min(fine, key=lambda r: r["spacing_deg"]) if fine else None
                row[f"l2_finest_{flow}"] = best["l2"] if best else np.nan
                row[f"finest_{flow}"] = best["grid"] if best else ""
            m = [r for r in minres if r["family"] == family and r["scheme"] == scheme]
            row["minimal_spacing_deg"] = m[0]["minimal_spacing_deg"] if m else np.nan
            row["minimal_note"] = m[0]["note"] if m else ""
            tracers = [r for r in sel if r["tracer"] != "uniform" and r["precision"] == "Float64"]
            row["undershoot_min"] = min((r["undershoot"] for r in tracers), default=np.nan)
            row["overshoot_max"] = max((r["overshoot"] for r in tracers), default=np.nan)
            for prec, tag in (("Float64", "fp64"), ("Float32", "fp32")):
                p = [r for r in sel if r["precision"] == prec and r["time"] == "T"]
                row[f"mass_drift_max_{tag}"] = max((r["mass_drift_budget_max"] for r in p), default=np.nan)
                u = [r["max_abs_chi_minus_1"] for r in p if r["tracer"] == "uniform"]
                row[f"uniform_max_abs_chi_minus_1_{tag}"] = max(u, default=np.nan)
            d = [r["fp32_minus_fp64_l2"] for r in sel if r["precision"] == "Float32" and r["time"] == "T"
                 and r["tracer"] != "uniform" and np.isfinite(r["fp32_minus_fp64_l2"])]
            row["fp32_minus_fp64_l2_max"] = max(d, default=np.nan)
            out.append(row)
    return out


def _tex_number(v, digits=2):
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    if -2 <= e <= 2:
        return f"{v:.{max(0, digits - 1 - e)}f}" if abs(v) < 100 else f"{v:.0f}"
    m = v / 10**e
    return f"${m:.1f}\\times10^{{{e}}}$"


FAMILY_TEX = {"cs": "Cubed sphere", "ll": "Latitude--longitude", "rg": "Reduced Gaussian"}


def _grid_tex(grid):
    """`ll384x192` -> `$384\\times192$`; cubed-sphere and reduced-Gaussian labels in capitals."""
    return "$" + grid[2:].replace("x", "\\times") + "$" if grid.startswith("ll") else grid.upper()


def table_convergence(rows, conv, minres, path):
    """LaTeX tabular (booktabs rules) for the paper: grid families as groups, schemes as rows."""
    lines = ["% Generated by examples/paper/idealized_advection/analyze.py; do not edit by hand.",
             "% FP64: normalized l2 at T of the smooth tracer at the finest resolution of each grid and",
             "% the observed order between the two finest resolutions. A value marked $^a$ is from a",
             "% coarser grid than the group header names (flow C ran at one resolution there). Minimal",
             "% resolution: grid spacing (degrees) where the flow B cosine bells reach l2 = 0.033. FP32",
             "% drift: largest relative tracer-mass change of any tracer, flow and resolution.",
             "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "Grid and scheme & \\multicolumn{2}{c}{Flow A} & \\multicolumn{2}{c}{Flow B} & "
             "\\multicolumn{2}{c}{Flow C} & Min.\\ res. & FP32 drift \\\\",
             " & $\\ell_2$ & order & $\\ell_2$ & order & $\\ell_2$ & order & ($^\\circ$) & \\\\"]
    for family in FAMILIES:
        sel = [r for r in rows if r["family"] == family and not r["tag"]]
        if not sel:
            continue
        finest = min(sel, key=lambda r: r["spacing_deg"])
        lines += ["\\midrule",
                  f"\\multicolumn{{9}}{{l}}{{{FAMILY_TEX[family]}, {_grid_tex(finest['grid'])} "
                  f"(${finest['spacing_deg']:.2f}^\\circ$)}} \\\\"]
        for scheme in SCHEME_ORDER:
            mine = [r for r in sel if r["scheme"] == scheme]
            if not mine:
                continue
            cells = ["\\quad " + SCHEME_TEX[scheme]]
            for flow in "ABC":
                fine = [r for r in mine if r["flow"] == flow and r["tracer"] == SMOOTH[flow] and r["time"] == "T"
                        and r["precision"] == "Float64"]
                best = min(fine, key=lambda r: r["spacing_deg"]) if fine else None
                coarser = best is not None and best["spacing_deg"] > finest["spacing_deg"] * 1.01
                cells.append((_tex_number(best["l2"]) + ("$^a$" if coarser else "")) if best else "--")
                cells.append(_tex_number(_finest_order(conv, family, flow, scheme, "l2")[0]))
            m = [r for r in minres if r["family"] == family and r["scheme"] == scheme]
            mv = m[0]["minimal_spacing_deg"] if m else np.nan
            if np.isfinite(mv):
                cells.append(_tex_number(mv))
            elif m and m[0]["note"]:
                relation, grid = m[0]["note"].split(" (")[0].split(" than ")
                cells.append(("$<$" if relation == "finer" else "$>$") + f"{grid_spacing(grid):.2f}")
            else:
                cells.append("--")
            drift = [r["mass_drift_budget_max"] for r in mine if r["precision"] == "Float32" and r["time"] == "T"]
            cells.append(_tex_number(max(drift)) if drift else "--")
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def tilt_table(rows):
    """Flow A, l2 and linf at T of the smooth tracer and the cosine bell for every tilt (FP64)."""
    out = []
    for r in rows:
        if r["flow"] != "A" or r["time"] != "T" or r["precision"] != "Float64" or r["tracer"] == "uniform":
            continue
        if r["tag"] and not r["tag"].startswith("tilt"):
            continue
        out.append(dict(grid=r["grid"], family=r["family"], scheme=r["scheme"], tilt_deg=r["tilt_deg"],
                        tracer=r["tracer"], l2=r["l2"], linf=r["linf"], undershoot=r["undershoot"]))
    return sorted(out, key=lambda r: (r["grid"], r["scheme"], r["tracer"], r["tilt_deg"]))


def variant_rows(rows, prefix):
    """(variant, base) pairs: tagged runs and the untagged run of the same grid, flow, scheme,
    precision, tracer and time."""
    base = {(r["grid"], r["flow"], r["scheme"], r["precision"], r["tracer"], r["time"]): r
            for r in rows if not r["tag"]}
    for r in rows:
        if not r["tag"].startswith(prefix) or r["tracer"] == "uniform":
            continue
        b = base.get((r["grid"], r["flow"], r["scheme"], r["precision"], r["tracer"], r["time"]))
        if b is not None:
            yield r, b


def substep_sensitivity(rows):
    """`cfl*` variants (more substeps per window) next to their base case."""
    return [dict(grid=r["grid"], flow=r["flow"], tag=r["tag"], scheme=r["scheme"],
                 precision=r["precision"], tracer=r["tracer"], time=r["time"],
                 courant_base=b["courant"], courant_variant=r["courant"],
                 l2_base=b["l2"], l2_variant=r["l2"], linf_base=b["linf"], linf_variant=r["linf"])
            for r, b in variant_rows(rows, "cfl")]


def window_sensitivity(rows):
    """`w*` variants (shorter met windows) next to their base case; `order_l2_*` is the observed
    order from the next coarser resolution of the same family, for the base and the variant."""
    pairs = list(variant_rows(rows, "w"))
    out = []
    for r, b in pairs:
        coarser = [(r2, b2) for r2, b2 in pairs
                   if (r2["family"], r2["flow"], r2["tag"], r2["scheme"], r2["precision"], r2["tracer"], r2["time"])
                   == (r["family"], r["flow"], r["tag"], r["scheme"], r["precision"], r["tracer"], r["time"])
                   and r2["spacing_deg"] > r["spacing_deg"]]
        order_base = order_variant = float("nan")
        if coarser:
            r2, b2 = min(coarser, key=lambda p: p[0]["spacing_deg"])
            ratio = math.log(r2["spacing_deg"] / r["spacing_deg"])
            order_base = math.log(b2["l2"] / b["l2"]) / ratio
            order_variant = math.log(r2["l2"] / r["l2"]) / ratio
        out.append(dict(grid=r["grid"], flow=r["flow"], tag=r["tag"], scheme=r["scheme"],
                        precision=r["precision"], tracer=r["tracer"], time=r["time"],
                        windows_per_day_base=b["windows_per_day"], windows_per_day_variant=r["windows_per_day"],
                        l2_base=b["l2"], l2_variant=r["l2"], linf_base=b["linf"], linf_variant=r["linf"],
                        order_l2_base=order_base, order_l2_variant=order_variant))
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workdir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--runs", default=None, help="list of run configs (default <workdir>/runs/runs.txt)")
    a = p.parse_args()
    workdir, results = Path(a.workdir), Path(a.results)
    results.mkdir(parents=True, exist_ok=True)
    rows, series, fields = [], [], {}
    runs_list = Path(a.runs) if a.runs else workdir / "runs" / "runs.txt"
    listed = [Path(line.strip()).parent for line in open(runs_list) if line.strip()]
    for run_dir in listed:
        meta = load_run(run_dir)
        case = tomllib.load(open(workdir / "inputs" / meta["case"] / "case.toml", "rb"))
        meta["tracers"] = case["tracers"]
        meta["tilt_deg"] = case.get("tilt_deg", 45.0) if meta["flow"] == "A" else np.nan
        r, s, f = analyze_run(meta, workdir)
        rows += r
        series += s
        for (tracer, label), v in f.items():
            fields[(meta["run_id"], tracer, label)] = v
        if ("area", meta["case"]) not in fields:
            with nc.Dataset(workdir / "inputs" / meta["case"] / "reference.nc") as ref:
                fields[("area", meta["case"])] = np.asarray(ref["cell_area"][:], dtype=np.float64)
        print(f"analyzed {meta['run_id']}")
    add_precision_differences(rows, fields)
    write_csv(results / "norms.csv", rows)
    conv = convergence(rows)
    minres = minimal_resolution(rows)
    write_csv(results / "convergence.csv", conv)
    write_csv(results / "minimal_resolution.csv", minres)
    write_csv(results / "summary.csv", summary(rows, conv, minres))
    table_convergence(rows, conv, minres, results / "table_convergence.tex")
    write_csv(results / "substep_sensitivity.csv", substep_sensitivity(rows))
    write_csv(results / "window_sensitivity.csv", window_sensitivity(rows))
    write_csv(results / "tilt.csv", tilt_table(rows))
    write_csv(results / "mass_series.csv", series)
    print(f"wrote {len(rows)} norm rows to {results}")


if __name__ == "__main__":
    main()
