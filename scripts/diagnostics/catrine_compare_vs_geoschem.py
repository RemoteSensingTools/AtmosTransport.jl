#!/usr/bin/env python3
"""Compare AtmosTransport CATRINE runs with the GEOS-Chem CATRINE standard.

The GEOS-Chem reference is 3-hourly instantaneous output on the GMAO cube,
levels surface first: GEOSChem.CATRINE_inst.YYYYMMDD_HHMMz.nc4, either flat in
--gc-dir or under --gc-dir/YYYY/MM/. Its resolution (C90 or C180) is read from
the files. Each AtmosTransport dataset is a cubed sphere (C90 or C180, levels
top first) on the same cube, so every field is compared on the common C90 grid,
summing 2x2 blocks of C180 cells (nesting is checked from the cell corners).

For every matched 3-hourly time and tracer this computes, per C90 column:
  * the dry-air-mass weighted column mean,
  * band means over fractions of the column air mass counted from the
    surface: 0-0.1 (~surface-910 hPa), 0.1-0.6 (~910-400 hPa),
    0.6-0.9 (~400-100 hPa), 0.9-1.0 (above ~100 hPa); these do not depend on
    the two models' different level sets,
plus zonal means on fixed dry-pressure levels (each model on its own grid)
and global burdens. Statistics are area weighted: R^2, OLS slope of model on
GEOS-Chem, mean bias, RMSE. They are reported for every time, for monthly
mean fields, and for the mean fields over --period-start..--end. Only monthly
sums are kept in memory, so multi-year comparisons are fine.

  python3 scripts/diagnostics/catrine_compare_vs_geoschem.py \\
      --gc-dir ~/data/AtmosTransport/catrine-geoschem-runs/C90 \\
      --dataset 'era5=/temp1/.../output/full/catrine_c90_*.nc' \\
      --extra 'era5=/temp1/.../output/full_fossil_dec2021/*.nc' \\
      --fossil era5=co2_fossil_from_dec2021 \\
      --start 2021-12-01T03:00 --end 2024-01-01T00:00 \\
      --out /temp1/cfranken/catrine_protocol/compare_c90_full

--extra adds files whose variables (here a tracer from a separate run on the
same met) are looked up by time next to the dataset's own; air mass always
comes from the main files.
"""
import argparse, datetime as dt, glob, json, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from netCDF4 import Dataset

GC_DIR = os.path.expanduser("~/data/AtmosTransport/catrine-geoschem-runs/C90")
T0 = dt.datetime(2021, 12, 1)
G = 9.80665
M_AIR = 0.0289644
TRACERS = {  # name: (default AtmosTransport variable, GEOS-Chem variable, molar mass kg/mol, display unit, scale)
    "co2":    ("co2_natural", "SpeciesConcVV_CO2",       0.0440095, "ppm", 1e6),
    "fossil": ("co2_fossil",  "SpeciesConcVV_FossilCO2", 0.0440095, "ppm", 1e6),
    "sf6":    ("sf6",         "SpeciesConcVV_SF6",       0.146055,  "ppt", 1e12),
    "rn222":  ("rn222",       "SpeciesConcVV_Rn222",     0.222,     "1e-21 mol/mol", 1e21),
}
BANDS = {"column": (0.0, 1.0), "sfc_910hPa": (0.0, 0.1), "910_400hPa": (0.1, 0.6),
         "400_100hPa": (0.6, 0.9), "above_100hPa": (0.9, 1.0)}
METRICS = ("r2", "slope", "bias", "rmse")
# 900 hPa and above: below that, which columns reach a level depends on each
# model's surface pressure, so the two zonal means would average different
# columns (near-surface differences are covered by the 0-10% mass band).
P_LEVELS = np.array([900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10], float) * 100.0
LAT_EDGES = np.arange(-90, 91, 4.0)


# ---------------------------------------------------------------- readers --

def as_float(x):
    """Masked or fill-valued NetCDF data as float64 with NaN for missing values."""
    return np.ma.filled(np.ma.asarray(x, dtype=float), np.nan)


def gc_path(t, gc_dir=None):
    gc_dir = gc_dir or GC_DIR
    name = f"GEOSChem.CATRINE_inst.{t:%Y%m%d}_{t:%H%M}z.nc4"
    nested = f"{gc_dir}/{t:%Y}/{t:%m}/{name}"
    return nested if os.path.exists(nested) else f"{gc_dir}/{name}"


def time_origin(units):
    """Origin of an hours axis: bare "hours" counts from the run start T0."""
    if "since" not in units:
        return T0
    stamp = units.split("since", 1)[1].strip().replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(stamp, fmt)
        except ValueError:
            pass
    raise SystemExit(f"cannot parse time units {units!r}")


def time_index(files, origin):
    """{datetime: (file, index)}; a time repeated across files keeps the first."""
    index = {}
    for f in files:
        with Dataset(f) as d:
            base = origin or time_origin(d["time"].units)
            for i, h in enumerate(np.asarray(d["time"][:], float)):
                index.setdefault(base + dt.timedelta(hours=float(h)), (f, i))
    return index


class AtmosDataset:
    """An AtmosTransport output: one or many NetCDF files with a time axis, plus
    optional extra files (same grid and times) holding some of the tracers."""

    def __init__(self, label, pattern, var_overrides, origin=None, extra_patterns=()):
        self.label = label
        files = sorted(glob.glob(os.path.expanduser(pattern)))
        if not files:
            raise SystemExit(f"{label}: no files match {pattern}")
        self.vars = {k: var_overrides.get(k, v[0]) for k, v in TRACERS.items()}
        self.indexes = [time_index(files, origin)]
        heads = [files[0]]
        for pat in extra_patterns:
            ef = sorted(glob.glob(os.path.expanduser(pat)))
            if not ef:
                raise SystemExit(f"{label}: no extra files match {pat}")
            self.indexes.append(time_index(ef, origin))
            heads.append(ef[0])
        self.source = {}                       # tracer -> position in self.indexes
        for k, v in self.vars.items():
            for s, f in enumerate(heads):
                with Dataset(f) as d:
                    if v in d.variables:
                        self.source[k] = s
                        break
            else:
                raise SystemExit(f"{label}: variable {v} is in none of {heads}")
        self.times = set(self.indexes[0]).intersection(*(set(self.indexes[s]) for s in set(self.source.values())))
        with Dataset(files[0]) as d:
            self.Nc = d.dimensions["Xdim"].size
            self.area = np.asarray(d["cell_area"][:], float)
            self.corner_lons = np.asarray(d["corner_lons"][:], float)
            self.corner_lats = np.asarray(d["corner_lats"][:], float)
            self.lats = np.asarray(d["lats"][:], float)

    def read(self, t):
        """Profiles top first: dict tracer -> (L, nf, Y, X) VMR, plus air mass (kg) and p_mid (Pa)."""
        f, i = self.indexes[0][t]
        with Dataset(f) as d:
            m = as_float(d["air_mass_per_area"][i])                   # (L, nf, Y, X) kg/m2
        q = {}
        for k, v in self.vars.items():
            f, i = self.indexes[self.source[k]][t]
            with Dataset(f) as d:
                q[k] = as_float(d[v][i])
            if q[k].shape != m.shape:
                raise ValueError(f"{self.label}: {v} shape {q[k].shape} != air mass {m.shape}")
        p_half = np.concatenate([np.zeros((1,) + m.shape[1:]), G * np.cumsum(m, axis=0)])
        p_mid = 0.5 * (p_half[:-1] + p_half[1:])
        return q, m * self.area[None], p_mid


def read_gc(t, gc_dir):
    path = gc_path(t, gc_dir)
    with Dataset(path) as d:
        ad = as_float(d["Met_AD"][0])[::-1]                             # top first
        p_mid = as_float(d["Met_PMIDDRY"][0])[::-1] * 100.0              # hPa -> Pa
        q = {k: as_float(d[v[1]][0])[::-1] for k, v in TRACERS.items()}
        area = as_float(d["Met_AREAM2"][0])
        lats = as_float(d["lats"][:])
    if not np.all(p_mid[0] < p_mid[-1]):
        raise ValueError(f"{path}: levels are not surface first")
    return q, ad, p_mid, area, lats


# ------------------------------------------------------------- reductions --

def band_sums(q, m):
    """Per column: {band: (sum q*m over the band, sum m over the band)}; arrays top first."""
    total = m.sum(axis=0)
    below = np.cumsum(m[::-1], axis=0)[::-1] - m          # mass below each layer
    f_bot = below / total
    f_top = (below + m) / total
    out = {}
    for name, (a, b) in BANDS.items():
        w = np.clip(np.minimum(b, f_top) - np.maximum(a, f_bot), 0.0, None) * total
        out[name] = ((w * q).sum(axis=0), w.sum(axis=0))
    return out


def block(x, r):
    """Sum r x r blocks of (nf, Y, X) into (nf, Y/r, X/r)."""
    if r == 1:
        return x
    nf, ny, nx = x.shape
    return x.reshape(nf, ny // r, r, nx // r, r).sum(axis=(2, 4))


def zonal_on_pressure(q, p_mid, lats, area):
    """Area-weighted zonal means (lat bins x P_LEVELS) of VMR interpolated in log p."""
    L = q.shape[0]
    qf, pf = q.reshape(L, -1), p_mid.reshape(L, -1)
    lp, lpt = np.log(np.maximum(pf, 1e-3)), np.log(P_LEVELS)
    out = np.full((len(LAT_EDGES) - 1, len(P_LEVELS)), np.nan)
    vals = np.full((len(P_LEVELS), qf.shape[1]), np.nan)
    for j, target in enumerate(lpt):
        k = (lp < target).sum(axis=0)                     # levels above the target pressure
        ok = (k > 0) & (k < L)
        k0 = np.clip(k - 1, 0, L - 1); k1 = np.clip(k, 0, L - 1)
        cols = np.arange(qf.shape[1])
        x0, x1 = lp[k0, cols], lp[k1, cols]
        w = np.where(x1 > x0, (target - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
        v = qf[k0, cols] * (1 - w) + qf[k1, cols] * w
        vals[j] = np.where(ok, v, np.nan)
    a, la = area.ravel(), lats.ravel()
    bins = np.digitize(la, LAT_EDGES) - 1
    for b in range(len(LAT_EDGES) - 1):
        sel = bins == b
        for j in range(len(P_LEVELS)):
            v = vals[j, sel]; ww = a[sel]; good = np.isfinite(v)
            if good.any():
                out[b, j] = np.sum(v[good] * ww[good]) / np.sum(ww[good])
    return out


def wstats(x, y, w):
    """Model y vs reference x, area weights w: R^2, slope, bias, RMSE."""
    good = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[good], y[good], w[good]
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    sxx, syy, sxy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2), np.sum(w * (x - mx) * (y - my))
    r2 = sxy ** 2 / (sxx * syy) if sxx > 0 and syy > 0 else np.nan
    return dict(r2=r2, slope=sxy / sxx if sxx > 0 else np.nan, bias=my - mx,
                rmse=np.sqrt(np.sum(w * (y - x) ** 2)), ref_mean=mx, ref_std=np.sqrt(sxx))


def reduce_fields(q, m, p_mid, lats, area, r):
    """Band means on C90, zonal means, burdens and air mass of one model state."""
    res = {"air_mass": float(m.sum())}
    for k in TRACERS:
        sums = band_sums(q[k], m)
        res[k] = {b: block(s, r) / block(w, r) for b, (s, w) in sums.items()}
        res[k]["zonal"] = zonal_on_pressure(q[k], p_mid, lats, area)
        res[k]["burden"] = float((q[k] * m).sum() * TRACERS[k][2] / M_AIR)
    return res


def reduce_one(args):
    """All reductions for one time: GEOS-Chem plus every dataset, and their statistics."""
    t, datasets, gc_dir = args       # gc_dir passed explicitly: workers may not fork
    try:
        q, m, p, area, gc_lats = read_gc(t, gc_dir)
    except (OSError, KeyError, ValueError, RuntimeError) as err:   # e.g. a corrupt GEOS-Chem file
        return t, None, str(err)
    rg = q["co2"].shape[-1] // 90
    res = {"gc": reduce_fields(q, m, p, gc_lats, area, rg)}
    w = block(area, rg).ravel()
    stats = {}
    for ds in datasets:
        try:
            qd, md, pd = ds.read(t)
        except (OSError, KeyError, ValueError, RuntimeError) as err:
            return t, None, f"{ds.label}: {err}"
        res[ds.label] = reduce_fields(qd, md, pd, ds.lats, ds.area, ds.Nc // 90)
        stats[ds.label] = {k: {b: wstats(res["gc"][k][b].ravel(), res[ds.label][k][b].ravel(), w)
                               for b in BANDS} for k in TRACERS}
    return t, res, stats


def check_nesting(ds, gc_file):
    """Largest C90-level corner offset (deg) between a dataset and the GEOS-Chem cube."""
    with Dataset(gc_file) as g:
        gl, ga = np.asarray(g["corner_lons"][:], float), np.asarray(g["corner_lats"][:], float)
    rg, r = (gl.shape[-1] - 1) // 90, ds.Nc // 90
    gl, ga = gl[:, ::rg, ::rg], ga[:, ::rg, ::rg]
    dlon = np.abs(((ds.corner_lons[:, ::r, ::r] - gl) + 180) % 360 - 180)
    dlat = np.abs(ds.corner_lats[:, ::r, ::r] - ga)
    off_pole = np.abs(ga) < 89.9
    worst = max(dlon[off_pole].max(), dlat.max())
    if worst > 1e-2:
        raise SystemExit(f"{ds.label}: grid does not nest in the GEOS-Chem cube (corner offset {worst:.3g} deg)")
    return worst


# ---------------------------------------------------------- accumulation --

class MeanAccumulator:
    """Running sums of arrays keyed by (group, label, tracer, field); NaN-aware."""

    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key, x):
        good = np.isfinite(x)
        if key not in self.s:
            self.s[key], self.n[key] = np.zeros(x.shape), np.zeros(x.shape)
        self.s[key][good] += x[good]
        self.n[key] += good

    def mean(self, key):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n[key] > 0, self.s[key] / np.maximum(self.n[key], 1), np.nan)


def month_key(t):
    return f"{t:%Y-%m}"


# ------------------------------------------------------------------ plots --

def plots(out, period, times, series, burdens, acc, labels, lon90, lat90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    import cartopy.crs as ccrs

    def xyz(lon, lat):
        lo, la = np.radians(lon), np.radians(lat)
        return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    glon, glat = np.meshgrid(np.arange(-179.5, 180, 1.0), np.arange(-89.5, 90, 1.0))
    _, nearest = cKDTree(xyz(lon90.ravel(), lat90.ravel())).query(xyz(glon.ravel(), glat.ravel()))
    regular = lambda f: f.ravel()[nearest].reshape(glon.shape)

    for k, (_, _, _, unit, scale) in TRACERS.items():
        # period-mean column maps: each model, GEOS-Chem, and differences
        ref = acc.mean(("period", "gc", k, "column")) * scale
        panels = [("GEOS-Chem", ref, "viridis", None)]
        for lab in labels:
            mod = acc.mean(("period", lab, k, "column")) * scale
            panels += [(lab, mod, "viridis", None), (f"{lab} - GEOS-Chem", mod - ref, "RdBu_r", "diff")]
        fig = plt.figure(figsize=(6.2 * len(panels), 3.6))
        lo, hi = np.nanpercentile(ref, [1, 99])
        dmax = max((np.nanpercentile(np.abs(p[1]), 99) for p in panels if p[3]), default=1.0)
        for i, (title, f, cmap, kind) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1, projection=ccrs.Robinson())
            ax.set_global(); ax.coastlines(linewidth=0.4)
            vmin, vmax = (-dmax, dmax) if kind else (lo, hi)
            im = ax.pcolormesh(glon, glat, regular(f), transform=ccrs.PlateCarree(), cmap=cmap,
                               vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
            ax.set_title(f"{title}", fontsize=10)
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, label=unit)
        fig.suptitle(f"{k}: {period} mean column-average dry mole fraction")
        fig.savefig(f"{out}/map_column_{k}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

        # zonal-mean cross sections
        zref = acc.mean(("period", "gc", k, "zonal")) * scale
        rows = [("GEOS-Chem", zref, None)] + [(lab, acc.mean(("period", lab, k, "zonal")) * scale, zref) for lab in labels]
        fig, axes = plt.subplots(1, 1 + 2 * len(labels), figsize=(5.5 * (1 + 2 * len(labels)), 3.8), squeeze=False)
        latc = 0.5 * (LAT_EDGES[:-1] + LAT_EDGES[1:])
        lo, hi = np.nanpercentile(zref, [2, 98])
        col = 0
        for title, z, base in rows:
            ax = axes[0, col]; col += 1
            im = ax.pcolormesh(latc, P_LEVELS / 100, z.T, vmin=lo, vmax=hi, shading="auto")
            ax.set_ylim(1000, 10); ax.set_yscale("log"); ax.set_title(title, fontsize=10); ax.set_ylabel("hPa")
            fig.colorbar(im, ax=ax, label=unit)
            if base is not None:
                ax = axes[0, col]; col += 1
                d = z - base; dm = np.nanpercentile(np.abs(d), 98) or 1.0
                im = ax.pcolormesh(latc, P_LEVELS / 100, d.T, cmap="RdBu_r", vmin=-dm, vmax=dm, shading="auto")
                ax.set_ylim(1000, 10); ax.set_yscale("log"); ax.set_title(f"{title} - GEOS-Chem", fontsize=10)
                fig.colorbar(im, ax=ax, label=unit)
        fig.suptitle(f"{k}: {period} mean zonal mean on dry pressure")
        fig.savefig(f"{out}/zonal_{k}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # time series of R^2 and bias per band
    fig, axes = plt.subplots(len(TRACERS), 2, figsize=(15, 2.6 * len(TRACERS)), squeeze=False)
    for i, (k, (_, _, _, unit, scale)) in enumerate(TRACERS.items()):
        for lab in labels:
            for b in BANDS:
                ls = "-" if b == "column" else ":"
                axes[i, 0].plot(times, series[lab][k][b]["r2"], ls, lw=0.8, label=f"{lab} {b}")
                axes[i, 1].plot(times, np.asarray(series[lab][k][b]["bias"]) * scale, ls, lw=0.8, label=f"{lab} {b}")
        axes[i, 0].set_ylabel(f"{k} R²"); axes[i, 1].set_ylabel(f"{k} bias ({unit})")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(f"{out}/timeseries_r2_bias.png", dpi=110); plt.close(fig)

    # global burdens, and their ratio to GEOS-Chem
    fig, axes = plt.subplots(2, len(TRACERS), figsize=(4.2 * len(TRACERS), 6.0), squeeze=False)
    for j, k in enumerate(TRACERS):
        ref = np.asarray(burdens["gc"][k])
        axes[0, j].plot(times, ref, "k-", label="GEOS-Chem")
        for lab in labels:
            axes[0, j].plot(times, burdens[lab][k], label=lab)
            with np.errstate(invalid="ignore", divide="ignore"):
                axes[1, j].plot(times, np.asarray(burdens[lab][k]) / ref, label=lab)
        axes[0, j].set_title(f"{k} burden (kg)"); axes[1, j].set_title(f"{k} burden / GEOS-Chem")
        for ax in axes[:, j]:
            ax.tick_params(axis="x", rotation=45)
    axes[0, 0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{out}/burdens.png", dpi=110); plt.close(fig)


def monthly_plot(out, months, monthly, labels):
    """R^2 and bias of the monthly mean fields, per tracer and band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [dt.datetime.strptime(mo, "%Y-%m") for mo in months]
    fig, axes = plt.subplots(len(TRACERS), 2, figsize=(14, 2.6 * len(TRACERS)), squeeze=False)
    for i, (k, (_, _, _, unit, scale)) in enumerate(TRACERS.items()):
        for lab in labels:
            for b in BANDS:
                ls = "o-" if b == "column" else ".:"
                axes[i, 0].plot(x, [monthly[mo][lab][k][b]["r2"] for mo in months], ls, ms=3, label=f"{lab} {b}")
                axes[i, 1].plot(x, [monthly[mo][lab][k][b]["bias"] * scale for mo in months], ls, ms=3, label=f"{lab} {b}")
        axes[i, 0].set_ylabel(f"{k} R²"); axes[i, 1].set_ylabel(f"{k} bias ({unit})")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(f"{out}/monthly_mean_field_r2_bias.png", dpi=110); plt.close(fig)


# ------------------------------------------------------------------- main --

def main():
    global GC_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gc-dir", default=GC_DIR, help="GEOS-Chem CATRINE_inst files (flat or YYYY/MM/)")
    ap.add_argument("--dataset", action="append", required=True, help="label=glob of AtmosTransport files")
    ap.add_argument("--extra", action="append", default=[],
                    help="label=glob of extra files with more tracer variables for that dataset")
    ap.add_argument("--fossil", action="append", default=[], help="label=variable for the fossil tracer")
    ap.add_argument("--origin", action="append", default=[],
                    help="label=ISO time: origin of that dataset's time axis when its units are wrong")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2021-12-01T03:00")
    ap.add_argument("--end", default="2021-12-31T21:00")
    ap.add_argument("--period-start", default=None,
                    help="start of the period for the mean-field maps and statistics (default --start)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    GC_DIR = os.path.expanduser(args.gc_dir)
    os.makedirs(args.out, exist_ok=True)
    fossil = dict(s.split("=", 1) for s in args.fossil)
    origins = dict(s.split("=", 1) for s in args.origin)
    extras = defaultdict(list)
    for s in args.extra:
        lab, pat = s.split("=", 1)
        extras[lab].append(pat)
    # The protocol's co2_fossil starts on 2022-01-01; comparisons with GEOS-Chem
    # need a tracer that, like GEOS-Chem's, emits from 2021-12-01.
    datasets = []
    for spec in args.dataset:
        lab, pat = spec.split("=", 1)
        overrides = {"fossil": fossil[lab]} if lab in fossil else {}
        origin = dt.datetime.fromisoformat(origins[lab]) if lab in origins else None
        datasets.append(AtmosDataset(lab, pat, overrides, origin, extras.get(lab, ())))
    labels = [d.label for d in datasets]
    if set(extras) - set(labels):
        raise SystemExit(f"--extra for unknown datasets {sorted(set(extras) - set(labels))}")
    t, end, times = dt.datetime.fromisoformat(args.start), dt.datetime.fromisoformat(args.end), []
    while t <= end:
        if os.path.exists(gc_path(t)) and all(t in d.times for d in datasets):
            times.append(t)
        t += dt.timedelta(hours=3)
    if not times:
        raise SystemExit("no matched times")
    period_start = dt.datetime.fromisoformat(args.period_start) if args.period_start else times[0]
    nest = {d.label: check_nesting(d, gc_path(times[0])) for d in datasets}
    print(f"{len(times)} matched times {times[0]} .. {times[-1]}; corner offsets {nest}", flush=True)

    acc = MeanAccumulator()
    all_labels = ["gc"] + labels
    done, skipped = [], []
    series = {lab: {k: {b: {m: [] for m in METRICS} for b in BANDS} for k in TRACERS} for lab in labels}
    burdens = {lab: {k: [] for k in TRACERS} for lab in all_labels}
    air = {lab: [] for lab in all_labels}
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (tt, res, stats) in enumerate(pool.map(reduce_one, [(tt, datasets, GC_DIR) for tt in times], chunksize=4)):
            if res is None:
                print(f"skip {tt}: {stats}", flush=True)
                skipped.append(str(tt))
                continue
            done.append(tt)
            groups = [month_key(tt)] + (["period"] if tt >= period_start else [])
            for lab in all_labels:
                air[lab].append(res[lab]["air_mass"])
                for k in TRACERS:
                    burdens[lab][k].append(res[lab][k]["burden"])
                    for fld in list(BANDS) + ["zonal"]:
                        for g in groups:
                            acc.add((g, lab, k, fld), res[lab][k][fld])
            for lab in labels:
                for k in TRACERS:
                    for b in BANDS:
                        for m in METRICS:
                            series[lab][k][b][m].append(stats[lab][k][b][m])
            if (n + 1) % 200 == 0:
                print(f"  {n + 1}/{len(times)} {tt}", flush=True)
    times = done
    months = sorted({month_key(tt) for tt in times})
    with Dataset(gc_path(times[0])) as d:
        rg = d.dimensions["Xdim"].size // 90
        w = block(as_float(d["Met_AREAM2"][0]), rg).ravel()
        lon90 = as_float(d["lons"][:])[:, ::rg, ::rg]; lat90 = as_float(d["lats"][:])[:, ::rg, ::rg]

    def field_stats(group):
        return {lab: {k: {b: wstats(acc.mean((group, "gc", k, b)).ravel(), acc.mean((group, lab, k, b)).ravel(), w)
                          for b in BANDS} for k in TRACERS} for lab in labels}
    monthly = {mo: field_stats(mo) for mo in months}
    period = f"{max(period_start, times[0]):%Y-%m-%d} to {times[-1]:%Y-%m-%d}"
    month_of = np.array([month_key(tt) for tt in times])
    summary = {"times": [str(times[0]), str(times[-1]), len(times)], "skipped": skipped,
               "gc_dir": GC_DIR, "corner_offset_deg": nest, "period": period,
               "air_mass_ratio_to_gc": {lab: float(np.mean(np.array(air[lab]) / np.array(air["gc"]))) for lab in labels},
               "period_mean_fields": field_stats("period"),
               "monthly_mean_fields": monthly,
               "monthly_mean_of_instantaneous": {
                   mo: {lab: {k: {b: {m: float(np.nanmean(np.asarray(series[lab][k][b][m])[month_of == mo]))
                                      for m in METRICS} for b in BANDS} for k in TRACERS} for lab in labels}
                   for mo in months},
               "final_time": {lab: {k: {b: {m: series[lab][k][b][m][-1] for m in METRICS} for b in BANDS}
                                    for k in TRACERS} for lab in labels},
               "final_burden_ratio_to_gc": {lab: {k: burdens[lab][k][-1] / burdens["gc"][k][-1] for k in TRACERS}
                                            for lab in labels}}
    with open(f"{args.out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    np.savez_compressed(
        f"{args.out}/monthly_fields_c90.npz", months=np.array(months),
        **{f"{lab}__{k}__{fld}": np.stack([acc.mean((mo, lab, k, fld)) for mo in months]).astype(np.float32)
           for lab in all_labels for k in TRACERS for fld in list(BANDS) + ["zonal"]})
    np.savez_compressed(
        f"{args.out}/timeseries.npz", times=np.array([str(x) for x in times]),
        **{f"{lab}__{k}__{b}__{m}": np.asarray(series[lab][k][b][m]) for lab in labels for k in TRACERS
           for b in BANDS for m in METRICS},
        **{f"{lab}__{k}__burden": np.asarray(burdens[lab][k]) for lab in all_labels for k in TRACERS},
        **{f"{lab}__air_mass": np.asarray(air[lab]) for lab in all_labels})
    plots(args.out, period, times, series, burdens, acc, labels, lon90, lat90)
    monthly_plot(args.out, months, monthly, labels)
    for lab in labels:
        print(f"\n{lab}: {period} mean fields vs GEOS-Chem (area weighted)")
        for k, (_, _, _, unit, scale) in TRACERS.items():
            for b in BANDS:
                st = summary["period_mean_fields"][lab][k][b]
                print(f"  {k:7s} {b:13s} R2 {st['r2']:.3f}  slope {st['slope']:.3f}  "
                      f"bias {st['bias'] * scale:+.4g} {unit}  rmse {st['rmse'] * scale:.4g} {unit}")


if __name__ == "__main__":
    main()
