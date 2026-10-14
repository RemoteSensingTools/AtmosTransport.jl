#!/usr/bin/env python3
"""Build the AtmosTransport v0.5 CATRINE benchmark web site (GCHP vs the v0.5 ERA5 and MERRA-2 runs).

The site is a set of static pages (D3 charts, no server code) written into --out:

  index.html         overview: signoff verdict, run progress, key numbers, agreement table, criteria
  timeseries.html    monthly band/region means and daily agreement statistics
  vertical.html      latitude-pressure cross sections and mean profiles
  maps.html          band-mean maps: GCHP, each run minus GCHP, ERA5 minus MERRA-2
  surface.html       near-surface band (lowest 10 % of the air mass): statistics, maps, zonal means
  stratosphere.html  above-100 hPa and 400-100 hPa biases, drift, identical-start check
  mass.html          global burdens, conservation residuals, dry-air mass
  methods.html       runs, comparison method, known differences (GCHP-side and ours)

Inputs (read only) are the products of the v0.5 signoff chain, never the raw model output:
  <signoff-dir>/compare/<label>/<season>/{summary.json,timeseries.npz,monthly_fields_c90.npz,DONE}
      written by scripts/diagnostics/catrine_compare_vs_geoschem.py, one season per directory;
  <signoff-dir>/massbal/<label>/<YYYY-MM>.csv  (scripts/diagnostics/catrine_true_mass_balance.jl, per month chunk);
  <signoff-dir>/SIGNOFF_TABLE.json             (signoff_summary.py: verdict and criteria);
  one GCHP C90 file for the grid and cell areas, and four GCHP files per month for the tropopause
  (cached in <out>/_cache); the GCHP mass-balance csv; the run output folders only to count days.

Only seasons with a DONE marker are read, so partly finished runs give a consistent site that
shows the compared months and an "as of" time; re-running the command picks up new seasons.

  python3 scripts/diagnostics/catrine_benchmark_v05_site.py --out /temp1/cfranken/plan/w3/signoff/site_staging

scripts/diagnostics/catrine_benchmark_site/publish.sh builds into the staging folder and copies
it to the public folder; README.md there describes the rebuild.
"""
import argparse, base64, datetime as dt, glob, hashlib, html, json, os, re, shutil, subprocess, sys, urllib.request
from collections import OrderedDict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "catrine_benchmark_site")

D3_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
D3_SHA256 = "f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539"   # same file on jsdelivr and unpkg
D3_LICENSE = """D3.js 7.9.0 (https://d3js.org), served with this site as assets/d3.min.js.

Copyright 2010-2023 Mike Bostock

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
"""
ASSET_FILES = ("site.css", "site.js", "d3.min.js", "d3.LICENSE.txt")   # everything else in <out>/assets is removed

# Tracers in display units. `scale` converts mol/mol to the unit; `key` is the comparison key.
TRACERS = OrderedDict([   # name/unit are HTML (sub/sup tags render in every font); text/unit_text are plain
    ("co2",    dict(name="CO<sub>2</sub>", text="CO2", long="CO₂ (CAMS posterior fluxes, fossil included)", unit="ppm", unit_text="ppm",
                    scale=1e6, digits=3, mb="co2_natural", gc_mb="co2_natural")),
    ("fossil", dict(name="Fossil CO<sub>2</sub>", text="Fossil CO2", long="Fossil CO₂ (GridFED, emitted from 2021-12-01)",
                    unit="ppm", unit_text="ppm", scale=1e6, digits=3, mb="co2_fossil_from_dec2021", gc_mb="co2_fossil")),
    ("sf6",    dict(name="SF<sub>6</sub>", text="SF6", long="SF₆ (EDGAR scaled to NOAA growth)", unit="ppt", unit_text="ppt",
                    scale=1e12, digits=4, mb="sf6", gc_mb="sf6")),
    ("rn222",  dict(name="<sup>222</sup>Rn", text="Rn-222", long="Radon-222 (half-life 3.8235 d)",
                    unit="10<sup>−21</sup> mol/mol", unit_text="1e-21 mol/mol", scale=1e21, digits=3, mb="rn222", gc_mb="rn222")),
])
BANDS = OrderedDict([
    ("column",       dict(name="Column", long="whole column (dry-air-mass weighted)", short="column")),
    ("sfc_910hPa",   dict(name="Surface–910 hPa", long="lowest 10 % of the air mass (surface to ≈ 910 hPa)", short="0–10 %")),
    ("910_400hPa",   dict(name="910–400 hPa", long="10–60 % of the air mass (≈ 910–400 hPa)", short="10–60 %")),
    ("400_100hPa",   dict(name="400–100 hPa", long="60–90 % of the air mass (≈ 400–100 hPa)", short="60–90 %")),
    ("above_100hPa", dict(name="Above 100 hPa", long="top 10 % of the air mass (above ≈ 100 hPa)", short="90–100 %")),
])
REGIONS = OrderedDict([
    ("global", dict(name="Global", lo=-90.0, hi=90.0)),
    ("nh", dict(name="30–90° N", lo=30.0, hi=90.0)),
    ("tropics", dict(name="30° S–30° N", lo=-30.0, hi=30.0)),
    ("sh", dict(name="90–30° S", lo=-90.0, hi=-30.0)),
    ("t20", dict(name="20° S–20° N", lo=-20.0, hi=20.0, hidden=True)),   # deep tropics: stratosphere page only
])
MAP_BANDS = list(BANDS)                    # every band gets maps
LAT_EDGES = np.arange(-90, 91, 4.0)        # zonal bins of catrine_compare_vs_geoschem.py
P_LEVELS = [900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]   # hPa, same script
MAP_RES = 2.0                              # degrees of the display grid
SPINUP = "2021-12"                         # first month: spin-up, left out of the period statistics
T_ORIGIN = dt.datetime(2021, 12, 1)       # protocol start; station series count hours from here
ALL_MONTHS = [f"{y}-{m:02d}" for y, m in [(2021, 12)] + [(y, m) for y in (2022, 2023) for m in range(1, 13)]]

PAGES = [  # file, nav label, title
    ("index.html", "Overview", "Overview"),
    ("timeseries.html", "Time series", "Time series"),
    ("vertical.html", "Vertical structure", "Vertical structure"),
    ("maps.html", "Maps", "Maps"),
    ("surface.html", "Near surface", "Near surface and boundary layer"),
    ("stratosphere.html", "Stratosphere", "Stratosphere and UTLS"),
    ("mass.html", "Mass budget", "Mass budget"),
    ("stations.html", "Stations", "Stations"),
    ("methods.html", "Methods", "Methods and known differences"),
]


# ------------------------------------------------------------------ helpers --

def sig(x, digits=6):
    """JSON number with `digits` significant figures; NaN/inf -> None."""
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{digits}g}") if np.isfinite(x) else None


def sig_list(values, digits=6):
    return [sig(v, digits) for v in np.asarray(values, dtype=float).ravel()]


def encode_int16(field):
    """Field -> {lo, step, b64}: int16 codes, -32768 = missing; value = lo + (code + 32767) * step."""
    f = np.asarray(field, dtype=np.float64)
    good = np.isfinite(f)
    if not good.any():
        return {"lo": 0.0, "step": 1.0, "b64": base64.b64encode(np.full(f.size, -32768, "<i2").tobytes()).decode()}
    lo, hi = float(f[good].min()), float(f[good].max())
    step = (hi - lo) / 65534.0 if hi > lo else 1.0
    code = np.full(f.shape, -32768, dtype=np.int64)
    code[good] = np.rint((f[good] - lo) / step).astype(np.int64) - 32767
    return {"lo": lo, "step": step, "b64": base64.b64encode(code.astype("<i2").ravel().tobytes()).decode()}


def write_js(path, var, key, payload):
    """A data file that registers itself: works from a web server and from disk (no fetch)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{var}({json.dumps(key)},{text});\n")


def git_revision():
    try:
        rev = subprocess.run(["git", "-C", HERE, "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", HERE], capture_output=True,
                               text=True, check=True).stdout.strip()
        return rev + (" + local changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def month_name(mo):
    return dt.datetime.strptime(mo, "%Y-%m").strftime("%b %Y")


# --------------------------------------------------------------- the inputs --

class Grid:
    """GCHP C90 cell centres and areas, region masks, and the 2-degree display grid."""

    def __init__(self, gc_file):
        from netCDF4 import Dataset
        with Dataset(gc_file) as d:
            self.lat = np.ma.filled(np.ma.asarray(d["lats"][:], dtype=float), np.nan)
            self.lon = np.ma.filled(np.ma.asarray(d["lons"][:], dtype=float), np.nan)
            self.area = np.ma.filled(np.ma.asarray(d["Met_AREAM2"][0], dtype=float), np.nan)
        if self.lat.shape != (6, 90, 90):
            raise SystemExit(f"{gc_file}: expected a C90 grid, got lats {self.lat.shape}")
        lat, lon, a = self.lat.ravel(), ((self.lon.ravel() + 180.0) % 360.0) - 180.0, self.area.ravel()
        self.region_w = {r: np.where((lat >= v["lo"]) & (lat <= v["hi"]), a, 0.0) for r, v in REGIONS.items()}
        # 2-degree display grid: area-weighted mean of the C90 cells whose centre falls in a bin; bins without a
        # centre (near the poles) take the nearest cell.
        self.nlon, self.nlat = int(360 / MAP_RES), int(180 / MAP_RES)
        self.mlon = -180 + MAP_RES * (np.arange(self.nlon) + 0.5)
        self.mlat = -90 + MAP_RES * (np.arange(self.nlat) + 0.5)
        ilon = np.clip(((lon + 180.0) / MAP_RES).astype(int), 0, self.nlon - 1)
        ilat = np.clip(((lat + 90.0) / MAP_RES).astype(int), 0, self.nlat - 1)
        self.bin = ilat * self.nlon + ilon
        self.cell_area = a
        self.bin_area = np.bincount(self.bin, weights=a, minlength=self.nlat * self.nlon)
        empty = np.flatnonzero(self.bin_area == 0)
        glat, glon = np.meshgrid(self.mlat, self.mlon, indexing="ij")
        xyz = lambda la, lo: np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                                       np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))], -1)
        cells = xyz(lat, lon)
        targets = xyz(glat.ravel()[empty], glon.ravel()[empty])
        self.empty = empty
        self.nearest = np.concatenate([np.argmax(targets[i:i + 256] @ cells.T, axis=1)          # chunked: bounded memory
                                       for i in range(0, len(empty), 256)]) if len(empty) else np.array([], int)

    def nearest_cell(self, lat, lon):
        """C90 cell (face, j, i) whose centre is nearest to (lat, lon), and the distance in km."""
        xyz = lambda la, lo: np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                                       np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))], -1)
        dots = xyz(self.lat.ravel(), self.lon.ravel()) @ xyz(np.array(lat), np.array(lon))
        k = int(np.argmax(dots))
        f, j, i = np.unravel_index(k, self.lat.shape)
        return int(f), int(j), int(i), float(6371.0 * np.arccos(np.clip(dots[k], -1.0, 1.0)))

    def to_map(self, field):
        f = np.asarray(field, dtype=float).ravel()
        good = np.isfinite(f)
        num = np.bincount(self.bin[good], weights=(self.cell_area * f)[good], minlength=self.bin_area.size)
        den = np.bincount(self.bin[good], weights=self.cell_area[good], minlength=self.bin_area.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        out[self.empty] = f[self.nearest]
        return out.reshape(self.nlat, self.nlon)

    def region_mean(self, field, region):
        f = np.asarray(field, dtype=float).ravel()
        w = self.region_w[region]
        good = np.isfinite(f) & (w > 0)
        return float(np.sum(f[good] * w[good]) / np.sum(w[good])) if good.any() else np.nan


class RunProducts:
    """Every finished season of one run's comparison with GCHP, merged by month."""

    def __init__(self, label, compare_dir):
        self.label = label
        self.seasons, self.months = [], []
        self.stats = {}            # month -> tracer -> band -> wstats dict (mol/mol)
        self.fields = {}           # month -> (npz path, index)
        self.snapshots = {}        # month -> number of 3-hourly snapshots
        self.times, self.series = [], {}   # 3-hourly statistics, burdens
        dirs = sorted(d for d in glob.glob(os.path.join(compare_dir, label, "*")) if os.path.exists(os.path.join(d, "DONE")))
        for d in dirs:
            try:
                with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
                    s = json.load(fh)
                ts = np.load(os.path.join(d, "timeseries.npz"))
                mf = os.path.join(d, "monthly_fields_c90.npz")
                with np.load(mf) as z:
                    npz_months = [str(m) for m in z["months"]]
            except (OSError, ValueError, KeyError) as err:
                print(f"warning: {label}: skipping {d}: {err}")
                continue
            self.seasons.append(os.path.basename(d))
            times = [str(t) for t in ts["times"]]
            per_month = {}
            for t in times:
                per_month[t[:7]] = per_month.get(t[:7], 0) + 1
            for i, mo in enumerate(npz_months):
                if per_month.get(mo, 0) < 8:            # single-snapshot "months" such as 2024-01
                    continue
                if mo in self.fields and per_month[mo] <= self.snapshots[mo]:
                    continue
                self.fields[mo] = (mf, i)
                self.snapshots[mo] = per_month[mo]
                self.stats[mo] = s["monthly_mean_fields"][mo][label]
            keys = [k for k in ts.files if k != "times"]
            order = np.argsort(times)
            for k in keys:
                self.series.setdefault(k, []).append(np.asarray(ts[k], float)[order])
            self.times.extend(sorted(times))
        self.months = sorted(self.fields)
        if self.times:
            idx = np.argsort(self.times)            # seasons are disjoint; keep the order of time
            self.times = [self.times[i] for i in idx]
            self.series = {k: np.concatenate(v)[idx] for k, v in self.series.items()}

    def field(self, month, who, tracer, fld):
        """Monthly-mean field (6, 90, 90) or zonal (45, 17); `who` is 'run' or 'gc' (paired GCHP)."""
        path, i = self.fields[month]
        name = f"{self.label if who == 'run' else 'gc'}__{tracer}__{fld}"
        with np.load(path) as z:
            return np.asarray(z[name][i], dtype=float)


class FieldCache:
    """Loads every monthly field of a run once, as float32 (about 8 MB per month and dataset)."""

    def __init__(self, run):
        self.run, self.data = run, {}
        by_path = {}
        for mo, (path, i) in run.fields.items():
            by_path.setdefault(path, []).append((mo, i))
        for path, items in by_path.items():
            with np.load(path) as z:
                for mo, i in items:
                    for who, lab in (("run", run.label), ("gc", "gc")):
                        for k in TRACERS:
                            for fld in list(BANDS) + ["zonal"]:
                                self.data[(mo, who, k, fld)] = np.asarray(z[f"{lab}__{k}__{fld}"][i], dtype=np.float32)

    def get(self, mo, who, k, fld):
        return self.data[(mo, who, k, fld)]


def tropopause(gc_dir, months, cache_dir):
    """GCHP zonal-mean tropopause pressure (hPa) per month on the 4-degree bins: Met_PMIDDRY at Met_TropLev,
    mean of days 1, 8, 15, 22 at 12 UTC; cached per month (and GCHP folder) once all four files were read."""
    from netCDF4 import Dataset
    out = {}
    os.makedirs(cache_dir, exist_ok=True)
    latc = 0.5 * (LAT_EDGES[:-1] + LAT_EDGES[1:])
    for mo in months:
        tag = hashlib.sha1(os.path.abspath(gc_dir).encode()).hexdigest()[:8]
        cache = os.path.join(cache_dir, f"tropopause_{tag}_{mo}.json")
        if os.path.exists(cache):
            with open(cache) as fh:
                out[mo] = json.load(fh)
            continue
        acc, n = np.zeros(len(latc)), 0
        for day in (1, 8, 15, 22):
            t = dt.datetime.strptime(f"{mo}-{day:02d}", "%Y-%m-%d")
            path = os.path.join(gc_dir, f"{t:%Y}", f"{t:%m}", f"GEOSChem.CATRINE_inst.{t:%Y%m%d}_1200z.nc4")
            if not os.path.exists(path):
                continue
            try:
                with Dataset(path) as d:
                    lev = np.asarray(d["Met_TropLev"][0], float)               # 1-based, surface first
                    p = np.asarray(d["Met_PMIDDRY"][0], float)                 # (72, 6, 90, 90) hPa
                    lats = np.asarray(d["lats"][:], float)
                    area = np.asarray(d["Met_AREAM2"][0], float)
            except (OSError, KeyError, RuntimeError) as err:
                print(f"warning: tropopause {path}: {err}")
                continue
            k = np.clip(np.rint(lev).astype(int) - 1, 0, p.shape[0] - 1)
            pt = np.take_along_axis(p, k[None], axis=0)[0]
            b = np.clip(np.digitize(lats.ravel(), LAT_EDGES) - 1, 0, len(latc) - 1)
            num = np.bincount(b, weights=(pt * area).ravel(), minlength=len(latc))
            den = np.bincount(b, weights=area.ravel(), minlength=len(latc))
            acc += num / den
            n += 1
        if n:
            out[mo] = sig_list(acc / n, 4)
            if n == 4:
                with open(cache, "w") as fh:
                    json.dump(out[mo], fh)
    return out


def days_present(folder, prefix):
    """Dates of the daily output files of a run (YYYY-MM-DD), to report progress."""
    days = []
    for f in glob.glob(os.path.join(folder, f"{prefix}*.nc")):
        m = re.search(r"_(\d{8})\.nc$", f)
        if m:
            days.append(dt.datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d"))
    return sorted(days)


def read_csv(path):
    import csv
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in (rows[0].keys() if rows else [])}


# ------------------------------------------------------------ the products --

class Site:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(os.path.expanduser(a.out))
        self.data_dir = os.path.join(self.out, "data")
        self.now = dt.datetime.now(dt.timezone.utc)
        sd = a.signoff_dir
        # Datasets: MERRA-2 always (unless --datasets leaves it out); ERA5 with --datasets ...,era5, or with "auto"
        # once all protocol months are compared; the identical-start MERRA-2 run only with MERRA-2 and only when
        # it has compared months (it appears in one collapsed panel).
        want = {"merra2", "era5_if_complete"} if a.datasets == "auto" else set(a.datasets.split(","))
        self.runs, self.excluded = OrderedDict(), {}
        for key, name, label, color, dash, role, folder, prefix in (
                ("era5", "ERA5 v0.5", a.era5_label, "#2a78d6", "solid", "signoff", a.era5_output, "catrine_c90_era5_"),
                ("merra2", "MERRA-2 v0.5", a.merra2_label, "#eb6834", "solid", "signoff", a.merra2_output,
                 "catrine_c90_merra2_"),
                ("merra2_gcic", "MERRA-2 v0.5, GCHP start", a.gcic_label, "#eb6834", "dot", "supporting",
                 a.gcic_output, "catrine_c90_merra2_")):
            prod = RunProducts(label, os.path.join(sd, "compare"))
            days = days_present(os.path.expanduser(folder), prefix)
            n = len(prod.months)
            use = {"era5": "era5" in want or ("era5_if_complete" in want and n == len(ALL_MONTHS)),
                   "merra2": "merra2" in want,
                   "merra2_gcic": "merra2" in want and n > 0}[key]
            print(f"{key:12s} {label:14s} months compared {n:2d} ({', '.join(prod.seasons) or 'none'}); "
                  f"output days {len(days)}; {'INCLUDED' if use else 'left out'}")
            if use:
                self.runs[key] = dict(key=key, name=name, label=label, color=color, dash=dash, role=role, prod=prod,
                                      days=days, folder=os.path.basename(os.path.normpath(folder)))
            else:
                self.excluded[key] = n
        self.main = [k for k in ("era5", "merra2") if k in self.runs and self.runs[k]["prod"].months]
        if not self.main:
            raise SystemExit(f"no compared months for the selected datasets ({a.datasets})")
        self.cache = {k: FieldCache(r["prod"]) for k, r in self.runs.items() if r["prod"].months}
        gc_file = sorted(glob.glob(os.path.join(os.path.expanduser(a.gc_dir), "2021", "12",
                                                "GEOSChem.CATRINE_inst.20211201_0300z.nc4")))
        if not gc_file:
            raise SystemExit(f"no GCHP grid file under {a.gc_dir}/2021/12")
        self.grid = Grid(gc_file[0])
        self.gc_file = gc_file[0]
        # months on the site = months of the main runs; the identical-start run is shown only within them
        self.months = sorted(set().union(*(set(self.runs[k]["prod"].months) for k in self.main)))
        common = sorted(set.intersection(*(set(self.runs[k]["prod"].months) for k in self.main)))
        self.common = [m for m in common if m != SPINUP] or common
        self.signoff = self.read_signoff()

    # Paired GCHP field for a month: the one compared with the given run, else any run's.
    def gc_field(self, mo, k, fld, prefer=None):
        for key in ([prefer] if prefer else []) + list(self.cache):
            c = self.cache.get(key)
            if c and (mo, "gc", k, fld) in c.data:
                return c.get(mo, "gc", k, fld)
        return None

    def weights(self, run_key, months):
        prod = self.runs[run_key]["prod"]
        w = np.array([prod.snapshots[m] for m in months], float)
        return w / w.sum()

    def mean_field(self, run_key, months, who, k, fld):
        w = self.weights(run_key, months)
        return sum(wi * self.cache[run_key].get(m, who, k, fld).astype(float) for wi, m in zip(w, months))

    # ---- signoff
    def read_signoff(self):
        path = os.path.join(self.a.signoff_dir, "SIGNOFF_TABLE.json")
        if not os.path.exists(path):
            print("warning: no SIGNOFF_TABLE.json; the overview shows no criteria")
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ---- data files
    def write_all(self):
        if os.path.isdir(self.data_dir):            # every data file is rewritten; none may survive from an older build
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        meta = self.meta()
        write_js(os.path.join(self.data_dir, "meta.js"), "SITE.put", "meta", meta)
        write_js(os.path.join(self.data_dir, "overview.js"), "SITE.put", "overview", self.overview())
        write_js(os.path.join(self.data_dir, "monthly.js"), "SITE.put", "monthly", self.monthly())
        for k in TRACERS:
            write_js(os.path.join(self.data_dir, f"daily_{k}.js"), "SITE.put", f"daily_{k}", self.daily(k))
        trop = tropopause(os.path.expanduser(self.a.gc_dir), self.months, os.path.join(self.out, "_cache"))
        for k in TRACERS:
            write_js(os.path.join(self.data_dir, f"zonal_{k}.js"), "SITE.put", f"zonal_{k}", self.zonal(k, trop))
        self.maps()
        write_js(os.path.join(self.data_dir, "mass.js"), "SITE.put", "mass", self.mass())
        write_js(os.path.join(self.data_dir, "coast.js"), "SITE.put", "coast", coastlines())
        if self.a.station_workers > 0:
            Stations(self, self.a.station_workers).write(self.data_dir)
        return meta

    def meta(self):
        runs = []
        for k, r in self.runs.items():
            p = r["prod"]
            runs.append(dict(key=k, name=r["name"], label=r["label"], color=r["color"], dash=r["dash"], role=r["role"],
                             months=p.months, seasons=p.seasons, folder=r["folder"],
                             days=len(r["days"]), last_day=r["days"][-1] if r["days"] else None,
                             months_transported=len({d[:7] for d in r["days"]}),
                             snapshots={m: p.snapshots[m] for m in p.months}))
        mtimes = [os.path.getmtime(os.path.join(self.a.signoff_dir, "compare", r["label"], s, "DONE"))
                  for r in self.runs.values() for s in r["prod"].seasons]
        latest = dt.datetime.fromtimestamp(max(mtimes), dt.timezone.utc) if mtimes else None
        return dict(
            generated=self.now.strftime("%Y-%m-%d %H:%M UTC"),
            data_time=latest.strftime("%Y-%m-%d %H:%M UTC") if latest else None,
            revision=git_revision(), all_months=ALL_MONTHS, months=self.months, common=self.common,
            spinup=SPINUP, runs=runs,
            gc=dict(key="gc", name="GCHP", color="#1f1f1d", dash="solid"),
            tracers=[dict(key=k, **{f: v[f] for f in ("name", "text", "long", "unit", "unit_text", "digits")})
                     for k, v in TRACERS.items()],
            bands=[dict(key=k, **v) for k, v in BANDS.items()],
            regions=[dict(key=k, name=v["name"]) for k, v in REGIONS.items() if not v.get("hidden")]
                    + [dict(key="nh_sh", name="NH − SH")],
            map_bands=MAP_BANDS,
            plevels=P_LEVELS, latc=sig_list(0.5 * (LAT_EDGES[:-1] + LAT_EDGES[1:]), 4),
            mlon=sig_list(self.grid.mlon, 5), mlat=sig_list(self.grid.mlat, 5),
            signoff_time=self.signoff.get("generated") if self.signoff else None,
            verdict=self.signoff.get("verdict") if self.signoff else None,
            datasets=list(self.runs), excluded=self.excluded)

    def overview(self):
        """Agreement statistics of the monthly-mean fields, averaged over months: bias = mean bias,
        RMSE = root of the mean squared RMSE, r = mean correlation; the GCHP field mean and spatial
        standard deviation for scale. Two month sets: the months both runs have compared (without the
        spin-up month) and every compared month of each run."""
        out = {"sets": {}}
        sets = {"common": {k: self.common for k in self.main}}
        sets["own"] = {k: [m for m in self.runs[k]["prod"].months if m != SPINUP] or self.runs[k]["prod"].months
                       for k in self.main}
        for name, months_of in sets.items():
            tab = {}
            for rk in self.main:
                months = [m for m in months_of[rk] if m in self.runs[rk]["prod"].stats]
                if not months:
                    continue
                st = self.runs[rk]["prod"].stats
                for k, t in TRACERS.items():
                    for b in BANDS:
                        rows = [st[m][k][b] for m in months]
                        r = [np.sign(x["slope"]) * np.sqrt(max(x["r2"], 0.0)) for x in rows]
                        tab[f"{rk}|{k}|{b}"] = dict(
                            bias=sig(np.mean([x["bias"] for x in rows]) * t["scale"], 5),
                            rmse=sig(np.sqrt(np.mean([x["rmse"] ** 2 for x in rows])) * t["scale"], 5),
                            r=sig(np.mean(r), 6),
                            ref_mean=sig(np.mean([x["ref_mean"] for x in rows]) * t["scale"], 7),
                            ref_std=sig(np.mean([x["ref_std"] for x in rows]) * t["scale"], 5))
            out["sets"][name] = dict(months={rk: [m for m in months_of[rk]] for rk in self.main}, table=tab)
        out["signoff"] = self.signoff_view()
        return out

    def signoff_view(self):
        s = self.signoff
        if not s:
            return None
        labels = {k: r["label"] for k, r in self.runs.items()}
        if "merra2" in labels:                       # C6a of MERRA-2 lives on the identical-start run
            labels.setdefault("merra2_gcic", self.a.gcic_label)
        crit = OrderedDict()
        for cid, per in s.get("criteria", {}).items():
            row = {}
            for key, lab in labels.items():
                if lab in per:
                    row[key] = dict(status=per[lab]["status"], detail=per[lab]["detail"],
                                    blocking=per[lab].get("blocking", False))
            if row:
                crit[cid] = row
        drift = {key: s["labels"][lab]["drift"] for key, lab in labels.items() if lab in s.get("labels", {})}
        air = {key: s["labels"][lab]["air"] for key, lab in labels.items() if lab in s.get("labels", {})}
        massv = {key: s["labels"][lab]["mass"] for key, lab in labels.items() if lab in s.get("labels", {})}
        clean = lambda x: json.loads(json.dumps(x, default=float).replace("NaN", "null"))
        return dict(verdict=s.get("verdict"), generated=s.get("generated"), code=s.get("code"),
                    fails=s.get("fails", []), criteria=crit, drift=clean(drift), air=clean(air), mass=clean(massv))

    def monthly(self):
        """Monthly area-weighted means per dataset, tracer, band and region (display units), the run minus
        paired-GCHP differences, and the monthly-mean-field statistics (global)."""
        g = self.grid
        out = {"abs": {}, "diff": {}, "stats": {}}
        regions = list(REGIONS)

        def means(f):
            vals = {r: g.region_mean(f, r) for r in regions}
            vals["nh_sh"] = vals["nh"] - vals["sh"]
            return vals
        for k, t in TRACERS.items():
            for b in BANDS:
                gc_vals = {r: [] for r in regions + ["nh_sh"]}
                for mo in self.months:
                    f = self.gc_field(mo, k, b, prefer="merra2")
                    v = means(f) if f is not None else {r: np.nan for r in gc_vals}
                    for r in gc_vals:
                        gc_vals[r].append(v[r] * t["scale"])
                for r in gc_vals:
                    out["abs"][f"gc|{k}|{b}|{r}"] = sig_list(gc_vals[r], 7)
                for rk, run in self.runs.items():
                    if rk not in self.cache:
                        continue
                    a_vals = {r: [] for r in gc_vals}
                    d_vals = {r: [] for r in gc_vals}
                    for mo in self.months:
                        if mo in run["prod"].fields:
                            fr = means(self.cache[rk].get(mo, "run", k, b))
                            fg = means(self.cache[rk].get(mo, "gc", k, b))
                        else:
                            fr = fg = {r: np.nan for r in gc_vals}
                        for r in gc_vals:
                            a_vals[r].append(fr[r] * t["scale"])
                            d_vals[r].append((fr[r] - fg[r]) * t["scale"])
                    for r in gc_vals:
                        out["abs"][f"{rk}|{k}|{b}|{r}"] = sig_list(a_vals[r], 7)
                        out["diff"][f"{rk}|{k}|{b}|{r}"] = sig_list(d_vals[r], 5)
                    st = run["prod"].stats
                    for stat in ("bias", "rmse", "r"):
                        vals = []
                        for mo in self.months:
                            if mo not in st:
                                vals.append(np.nan)
                                continue
                            x = st[mo][k][b]
                            vals.append(np.sign(x["slope"]) * np.sqrt(max(x["r2"], 0)) if stat == "r"
                                        else x[stat] * t["scale"])
                        out["stats"][f"{rk}|{k}|{b}|{stat}"] = sig_list(vals, 5)
        return out

    def daily(self, k):
        """Daily means of the 3-hourly agreement statistics (global, per band) and of the burdens."""
        t = TRACERS[k]
        days = sorted({tt[:10] for r in self.runs.values() for tt in r["prod"].times})
        index = {d: i for i, d in enumerate(days)}
        out = {"days": days, "stats": {}, "burden_ratio": {}}

        def daily_mean(times, values):
            v = np.asarray(values, float)
            idx = np.array([index[tt[:10]] for tt in times])
            ok = np.isfinite(v)
            n = np.bincount(idx[ok], minlength=len(days))
            s = np.bincount(idx[ok], weights=v[ok], minlength=len(days))
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(n > 0, s / np.maximum(n, 1), np.nan)
        for rk, r in self.runs.items():
            p = r["prod"]
            if not p.times:
                continue
            for b in BANDS:
                for stat in ("bias", "rmse", "r2"):
                    v = p.series[f"{p.label}__{k}__{b}__{stat}"] * (t["scale"] if stat in ("bias", "rmse") else 1.0)
                    out["stats"][f"{rk}|{b}|{stat}"] = sig_list(daily_mean(p.times, v), 5)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = p.series[f"{p.label}__{k}__burden"] / p.series[f"gc__{k}__burden"]
            out["burden_ratio"][rk] = sig_list(daily_mean(p.times, ratio), 8)
        return out

    def zonal(self, k, trop):
        """Latitude-pressure zonal means (45 x 17, 4-degree bins x P_LEVELS): GCHP absolute and each run
        minus its paired GCHP, per month and for the period of the common months."""
        t = TRACERS[k]
        out = {"months": {}, "tropopause": {}}
        keys = self.months + ["period"]
        for mo in keys:
            entry = {}
            if mo == "period":
                ref = self.mean_field(self.main[-1], self.common, "gc", k, "zonal") if self.common else None
                if ref is None:
                    continue
                entry["gc"] = sig_list(ref.T * t["scale"], 6)
                for rk in self.main:
                    d = self.mean_field(rk, self.common, "run", k, "zonal") - self.mean_field(rk, self.common, "gc", k,
                                                                                          "zonal")
                    entry[rk] = sig_list(d.T * t["scale"], 4)
                tp = [trop[m] for m in self.common if m in trop]
                if tp:
                    out["tropopause"]["period"] = sig_list(np.mean(np.array(tp, float), axis=0), 4)
            else:
                ref = self.gc_field(mo, k, "zonal", prefer="merra2")
                entry["gc"] = sig_list(ref.T * t["scale"], 6)
                for rk in self.main:
                    if mo in self.runs[rk]["prod"].fields:
                        d = self.cache[rk].get(mo, "run", k, "zonal") - self.cache[rk].get(mo, "gc", k, "zonal")
                        entry[rk] = sig_list(d.T * t["scale"], 4)
                if mo in trop:
                    out["tropopause"][mo] = trop[mo]
            out["months"][mo] = entry        # arrays are [pressure][latitude], P_LEVELS order
        return out

    def maps(self):
        """One file per tracer, band and month (and 'period'): GCHP on the 2-degree grid and each run minus its
        paired GCHP, int16-coded."""
        g = self.grid
        n = 0
        for k, t in TRACERS.items():
            for b in MAP_BANDS:
                for mo in self.months + ["period"]:
                    payload = {}
                    if mo == "period":
                        if not self.common:
                            continue
                        ref = self.mean_field(self.main[-1], self.common, "gc", k, b)
                        payload["gc"] = encode_int16(g.to_map(ref * t["scale"]))
                        for rk in self.main:
                            d = self.mean_field(rk, self.common, "run", k, b) - self.mean_field(rk, self.common, "gc",
                                                                                            k, b)
                            payload[rk] = encode_int16(g.to_map(d * t["scale"]))
                    else:
                        ref = self.gc_field(mo, k, b, prefer="merra2")
                        payload["gc"] = encode_int16(g.to_map(ref * t["scale"]))
                        for rk in self.main:
                            if mo in self.runs[rk]["prod"].fields:
                                d = self.cache[rk].get(mo, "run", k, b) - self.cache[rk].get(mo, "gc", k, b)
                                payload[rk] = encode_int16(g.to_map(d * t["scale"]))
                    write_js(os.path.join(self.data_dir, "maps", f"{k}_{b}_{mo}.js"), "SITE.put", f"map|{k}|{b}|{mo}",
                             payload)
                    n += 1
        # stale map files of months no longer present are removed
        keep = {f"{k}_{b}_{mo}.js" for k in TRACERS for b in MAP_BANDS for mo in self.months + ["period"]}
        for f in glob.glob(os.path.join(self.data_dir, "maps", "*.js")):
            if os.path.basename(f) not in keep:
                os.remove(f)
        print(f"wrote {n} map files")

    def mass(self):
        """Burden ratios (daily, from daily_*), true mass balance residuals and dry-air ratios."""
        out = {"runs": {}, "gc": {}, "c9": {}, "air": {}}
        origin = dt.datetime(2021, 12, 1)
        for rk, r in self.runs.items():
            folder = os.path.join(self.a.signoff_dir, "massbal", r["label"])
            chunks = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".DONE")) if os.path.isdir(folder) else []
            series = {k: {"t": [], "v": []} for k in TRACERS}
            offset = {k: 0.0 for k in TRACERS}
            for mo in chunks:
                path = os.path.join(folder, f"{mo}.csv")
                if not os.path.exists(path):
                    continue
                c = read_csv(path)
                if not c:
                    continue
                t0 = dt.datetime.strptime(mo, "%Y-%m")
                times = [(t0 + dt.timedelta(hours=float(h))) for h in c["hours"]]
                keep = np.arange(0, len(times), 8)          # daily at the chunk's own start hour
                for k, t in TRACERS.items():
                    col = t["mb"]
                    if f"{col}_residual" not in c:
                        continue
                    if k == "rn222":         # decays in days: each month's final residual relative to the amount
                        src = c[f"{col}_source_cum"][-1]            # emitted in that month (signoff C9)
                        series[k]["t"].append(times[-1].strftime("%Y-%m-%d %H:%M"))
                        series[k]["v"].append(sig(c[f"{col}_residual"][-1] / src, 5) if src > 0 else None)
                        continue
                    else:                    # mol/mol, chunks chained
                        v = offset[k] + c[f"{col}_residual"] / c["dry_air_kg"]
                        offset[k] = float(v[-1])
                    series[k]["t"].extend(times[i].strftime("%Y-%m-%d %H:%M") for i in keep)
                    series[k]["v"].extend(sig_list(v[keep], 5))
            out["runs"][rk] = dict(chunks=chunks, series=series)
        gc_csv = os.path.expanduser(self.a.gc_massbal)
        if os.path.exists(gc_csv):
            c = read_csv(gc_csv)
            keep = np.arange(0, len(c["hours"]), 8)
            times = [(origin + dt.timedelta(hours=float(c["hours"][i]))).strftime("%Y-%m-%d %H:%M") for i in keep]
            for k, t in TRACERS.items():
                col = t["gc_mb"]
                if k == "rn222" or f"{col}_residual" not in c:
                    continue
                out["gc"][k] = dict(t=times, v=sig_list((c[f"{col}_residual"] / c["dry_air_kg"])[keep], 5))
        else:
            print(f"warning: no GCHP mass balance {gc_csv}")
        so = self.signoff_view()
        if so:
            out["c9"], out["air"] = so["mass"], so["air"]
        return out


# ------------------------------------------------------------- stations --
# NOAA sites, sampled in the nearest C90 cell. Coordinates, elevation and intake height come from the
# ObsPack (GLOBALVIEWplus v10.1) metadata of the listed dataset when it is available, otherwise from
# the fallback values below (copied from the same metadata). Sampling: "lowest" = lowest model layer;
# "asl" = at elevation + intake above sea level; "agl" = at intake height above the model surface.
STATIONS = [
    dict(code="MLO", mode="asl", obs="co2_mlo_surface-insitu_1_allvalid",
         fallback=dict(name="Mauna Loa, Hawaii", lat=19.5362, lon=-155.5763, elev=3397.0, intake=40.0)),
    dict(code="SPO", mode="lowest", obs="co2_spo_surface-insitu_1_allvalid",
         fallback=dict(name="South Pole, Antarctica", lat=-89.98, lon=-24.8, elev=2810.0, intake=11.0)),
    dict(code="BRW", mode="lowest", obs="co2_brw_surface-insitu_1_allvalid",
         fallback=dict(name="Barrow Atmospheric Baseline Observatory, Alaska", lat=71.3230, lon=-156.6114, elev=11.0, intake=16.0)),
    dict(code="ALT", mode="lowest", obs="co2_alt_surface-flask_1_representative",
         fallback=dict(name="Alert, Nunavut", lat=82.4508, lon=-62.5072, elev=185.0, intake=None)),
    dict(code="MHD", mode="lowest", obs="co2_mhd_surface-flask_1_representative",
         fallback=dict(name="Mace Head, County Galway", lat=53.3260, lon=-9.8990, elev=5.0, intake=None)),
    dict(code="LEF", mode="agl", obs="co2_lef_tower-insitu_1_allvalid-396magl",
         fallback=dict(name="Park Falls, Wisconsin", lat=45.9451, lon=-90.2732, elev=472.0, intake=396.0)),
    dict(code="SMO", mode="lowest", obs="co2_smo_surface-insitu_1_allvalid",
         fallback=dict(name="Tutuila, American Samoa", lat=-14.2474, lon=-170.5644, elev=42.0, intake=18.0)),
    dict(code="CGO", mode="lowest", obs="co2_cgo_surface-flask_1_representative",
         fallback=dict(name="Cape Grim, Tasmania", lat=-40.6830, lon=144.6900, elev=94.0, intake=None)),
]
STATION_TRACERS = [("co2", "SpeciesConcVV_CO2", "co2_natural"), ("fossil", "SpeciesConcVV_FossilCO2", "co2_fossil_from_dec2021"),
                   ("sf6", "SpeciesConcVV_SF6", "sf6"), ("rn222", "SpeciesConcVV_Rn222", "rn222")]
STATION_LEVELS = 40          # lowest levels read for altitude sampling (GCHP ~ up to 300 hPa; enough for 3.4 km)
G0 = 9.80665                 # m s-2, STANDARD_GRAVITY of src/Parameters/PhysicalConstants.jl (dry pressure from air mass)
OBSPACK_PRODUCT = "obspack_co2_1_GLOBALVIEWplus_v10.1_2024-11-13"


def read_obspack_metadata(obspack_dir, dataset):
    """key -> value of an ObsPack dataset metadata file (None if the file is missing)."""
    path = os.path.join(obspack_dir, "metadata", OBSPACK_PRODUCT, f"{dataset}.txt")
    if not os.path.exists(path):
        return None
    meta = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if " : " in line:
                k, v = line.split(" : ", 1)
                meta[k.strip()] = v.strip()
    return meta


def _intake(text, default):
    """Mid value of an ObsPack intake height ("39.0-41.0" -> 40.0); `default` (may be None) when not stated."""
    if text is None:
        return default
    try:
        parts = [float(x) for x in str(text).replace("magl", "").split("-") if x.strip()]
        return sum(parts) / len(parts) if parts else default
    except ValueError:
        return default


def _gc_station_file(args):
    """One GCHP 3-hourly file: per site the sampled values (4 tracers), the target dry pressure (hPa) and
    the model surface height. Columns are read surface first, only the lowest STATION_LEVELS levels."""
    path, sites = args
    from netCDF4 import Dataset
    out = np.full((len(sites), len(STATION_TRACERS)), np.nan)
    pstar = np.full(len(sites), np.nan)
    try:
        with Dataset(path) as d:
            for s, site in enumerate(sites):
                f, j, i = site["cell"]
                if site["mode"] == "lowest":
                    for k, (_, gv, _) in enumerate(STATION_TRACERS):
                        out[s, k] = float(d[gv][0, 0, f, j, i])
                    pstar[s] = float(d["Met_PMIDDRY"][0, 0, f, j, i])
                    continue
                K = STATION_LEVELS
                H = np.asarray(d["Height_asl"][0, :K, f, j, i], float)            # layer centres, m asl
                P = np.asarray(d["Met_PMIDDRY"][0, :K, f, j, i], float)           # hPa, dry
                z = site["target_asl"] if site["mode"] == "asl" else float(d["Met_PHIS"][0, f, j, i]) + site["intake"]
                if z <= H[0]:
                    lev = None
                    pstar[s] = P[0]
                else:
                    pstar[s] = float(np.exp(np.interp(z, H, np.log(P))))
                for k, (_, gv, _) in enumerate(STATION_TRACERS):
                    q = np.asarray(d[gv][0, :K, f, j, i], float)
                    out[s, k] = q[0] if z <= H[0] else float(np.interp(z, H, q))
    except (OSError, KeyError, RuntimeError, IndexError) as err:
        return path, None, None, str(err)
    return path, out, pstar, None


def _run_station_file(args):
    """One AtmosTransport daily file: per time and site the values at the site (lowest layer, or the
    GCHP target dry pressure interpolated in log p on the run's own dry pressure, from its air mass)."""
    path, sites, pstar_of = args
    from netCDF4 import Dataset
    rows = []
    try:
        with Dataset(path) as d:
            tv = d["time"]
            base = time_origin_of(tv.units)
            hours = [float(h) for h in np.asarray(tv[:], float)]
            times = [base + dt.timedelta(hours=h) for h in hours]
            L = d.dimensions["lev"].size
            vals = np.full((len(times), len(sites), len(STATION_TRACERS)), np.nan)
            for s, site in enumerate(sites):
                f, j, i = site["cell"]
                if site["mode"] == "lowest":
                    for k, (_, _, rv) in enumerate(STATION_TRACERS):
                        vals[:, s, k] = np.asarray(d[rv][:, L - 1, f, j, i], float)
                    continue
                K = min(L, site["levels"])
                m = np.asarray(d["air_mass_per_area"][:, L - K:, f, j, i], float)               # top first, kg m-2
                ps = G0 * np.asarray(d["column_air_mass_per_area"][:, f, j, i], float)          # Pa, dry surface pressure
                below = np.cumsum(m[:, ::-1], axis=1)[:, ::-1]                                   # mass at and below each layer
                pmid = (ps[:, None] - G0 * (below - 0.5 * m)) / 100.0                            # hPa, layer midpoints
                qs = [np.asarray(d[rv][:, L - K:, f, j, i], float) for _, _, rv in STATION_TRACERS]
                for t, tt in enumerate(times):
                    p = pstar_of.get((tt, site["code"]))
                    if p is None or not np.isfinite(p):
                        continue
                    lp = np.log(pmid[t])                                                         # increasing downward
                    for k in range(len(STATION_TRACERS)):
                        col = qs[k][t]
                        vals[t, s, k] = col[-1] if p >= pmid[t, -1] else float(np.interp(np.log(p), lp, col))
            rows = [(tt, vals[t]) for t, tt in enumerate(times)]
    except (OSError, KeyError, RuntimeError, IndexError) as err:
        return path, None, str(err)
    return path, rows, None


def time_origin_of(units):
    stamp = units.split("since", 1)[1].strip().replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(stamp, fmt)
        except ValueError:
            pass
    raise ValueError(f"cannot parse time units {units!r}")


def _obs_file(args):
    """One ObsPack daily file: (dataset, epoch seconds, value, obs_flag) of the wanted datasets."""
    path, wanted = args
    from netCDF4 import Dataset, chartostring
    try:
        with Dataset(path) as d:
            ids = chartostring(d["obspack_id"][:])
            names = np.array([x.split("~")[1] if "~" in x else "" for x in ids])
            keep = np.isin(names, list(wanted))
            if not keep.any():
                return []
            t = np.asarray(d["time"][:], float)[keep]
            v = np.ma.filled(np.ma.asarray(d["value"][:], float), np.nan)[keep]
            fl = np.ma.filled(np.ma.asarray(d["obs_flag"][:], int), -9)[keep]
            return list(zip(names[keep], t, v, fl))
    except (OSError, KeyError, RuntimeError):
        return []


class Stations:
    """Station time series of GCHP and the runs (cached per month) and NOAA observations (CO2, CC0 only)."""

    def __init__(self, site_obj, workers):
        self.s, self.a, self.workers = site_obj, site_obj.a, workers
        self.cache = os.path.join(site_obj.out, "_cache", "stations")
        os.makedirs(self.cache, exist_ok=True)
        obspack = os.path.expanduser(self.a.obspack_dir) if self.a.obspack_dir else ""
        self.obspack = obspack if obspack and os.path.isdir(obspack) else None
        g = site_obj.grid
        with __import__("netCDF4").Dataset(site_obj.gc_file) as d:
            phis = np.asarray(d["Met_PHIS"][0], float)                         # m (surface height)
        self.sites, self.skipped = [], []
        for st in STATIONS:
            md = read_obspack_metadata(self.obspack, st["obs"]) if self.obspack else None
            fb = st["fallback"]
            lat = float(md["site_latitude"]) if md else fb["lat"]
            lon = float(md["site_longitude"]) if md else fb["lon"]
            elev = float(md["site_elevation"]) if md else fb["elev"]
            intake = _intake(md.get("dataset_intake_ht"), fb["intake"]) if md else fb["intake"]
            f, j, i, dist = g.nearest_cell(lat, lon)
            lic = md.get("dataset_provider_license", "") if md else ""
            cc0 = "CC0" in lic
            site = dict(code=st["code"], name=(md or {}).get("site_name", fb["name"]), lat=lat, lon=lon, elev=elev, intake=intake,
                        mode=st["mode"], cell=(int(f), int(j), int(i)), cell_lat=float(g.lat[f, j, i]),
                        cell_lon=float(((g.lon[f, j, i] + 180) % 360) - 180), dist_km=dist,
                        surface_m=float(phis[f, j, i]), target_asl=elev + (intake or 0.0),
                        source="ObsPack metadata" if md else "fallback (ObsPack metadata values)",
                        obs=dict(dataset=st["obs"], license_cc0=cc0, selection=(md or {}).get("dataset_selection", ""),
                                 kind="flask" if "flask" in st["obs"] else "insitu",
                                 citations=[md[k] for k in sorted(md) if k.startswith("dataset_provider_citation_")
                                            and not k.startswith("dataset_provider_citation_identifier")
                                            and not k.endswith("_total")] if md else [],
                                 usage_url=(md or {}).get("dataset_usage_url", "")) if md else None)
            self.sites.append(site)
            if md and not cc0:
                self.skipped.append((st["obs"], "provider licence is not CC0" if lic else "no provider licence stated"))
        # datasets of the same sites that are present but not used (for the report and the page)
        if self.obspack:
            mdir = os.path.join(self.obspack, "metadata", OBSPACK_PRODUCT)
            used = {s["obs"]["dataset"] for s in self.sites if s["obs"]}
            for s in self.sites:
                for path in sorted(glob.glob(os.path.join(mdir, f"co2_{s['code'].lower()}_*.txt"))):
                    name = os.path.basename(path)[:-4]
                    if name in used:
                        continue
                    md = read_obspack_metadata(self.obspack, name)
                    lic = md.get("dataset_provider_license", "") if md else ""
                    self.skipped.append((name, "not selected (one NOAA record per site)" if "CC0" in lic
                                         else "provider licence is not CC0" if lic else "no provider licence stated"))
        self.key = hashlib.sha1(json.dumps([(s["code"], s["cell"], s["mode"], round(s["target_asl"], 1), s["intake"])
                                            for s in self.sites]).encode()).hexdigest()[:10]

    # ---- per month extraction with cache
    def _cached(self, name, compute):
        path = os.path.join(self.cache, f"{name}_{self.key}.npz")
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        res = compute()
        if res is not None:
            np.savez_compressed(path, **res)
        return res

    def _pool(self, fn, jobs):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.workers) as pool:
            return list(pool.map(fn, jobs, chunksize=2))

    def gc_month(self, mo):
        def compute():
            y, m = map(int, mo.split("-"))
            gc_dir = os.path.expanduser(self.a.gc_dir)
            files = sorted(glob.glob(os.path.join(gc_dir, f"{y:04d}", f"{m:02d}", "GEOSChem.CATRINE_inst.*.nc4")))
            res = self._pool(_gc_station_file, [(f, self.sites) for f in files])
            times, vals, ps = [], [], []
            for path, out, pstar, err in res:
                if err:
                    print(f"warning: {path}: {err}")
                    continue
                stamp = re.search(r"(\d{8})_(\d{4})z", path)
                tt = dt.datetime.strptime(stamp.group(1) + stamp.group(2), "%Y%m%d%H%M")
                if tt.strftime("%Y-%m") != mo:
                    continue
                times.append(tt); vals.append(out); ps.append(pstar)
            if not times:
                return None
            return dict(hours=np.array([(t - T_ORIGIN).total_seconds() / 3600 for t in times]),
                        values=np.array(vals, np.float32), pstar=np.array(ps, np.float32))
        return self._cached(f"gc_{mo}", compute)

    def run_month(self, rk, mo, gc):
        r = self.s.runs[rk]
        folder = os.path.expanduser({"era5": self.a.era5_output, "merra2": self.a.merra2_output}[rk])
        prefix = "catrine_c90_era5_" if rk == "era5" else "catrine_c90_merra2_"

        def compute():
            y, m = map(int, mo.split("-"))
            first = dt.date(y, m, 1)
            files = []
            for k in range(-1, 32):                                 # the previous day holds this month's 00 UTC
                day = first + dt.timedelta(days=k)
                p = os.path.join(folder, f"{prefix}{day:%Y%m%d}.nc")
                if os.path.exists(p):
                    files.append(p)
            pstar_of = {}
            if gc is not None:
                for h, p in zip(gc["hours"], gc["pstar"]):
                    tt = T_ORIGIN + dt.timedelta(hours=float(h))
                    for s, site in enumerate(self.sites):
                        pstar_of[(tt, site["code"])] = float(p[s])
            sites = [dict(site, levels=self._levels(rk)) for site in self.sites]
            res = self._pool(_run_station_file, [(f, sites, pstar_of) for f in files])
            got = {}
            for path, rows, err in res:
                if err:
                    print(f"warning: {path}: {err}")
                    continue
                for tt, v in rows:
                    if tt.strftime("%Y-%m") == mo and tt not in got:
                        got[tt] = v
            if not got:
                return None
            times = sorted(got)
            return dict(hours=np.array([(t - T_ORIGIN).total_seconds() / 3600 for t in times]),
                        values=np.array([got[t] for t in times], np.float32))
        return self._cached(f"{r['label']}_{mo}", compute)

    def _levels(self, rk):
        return 40 if rk == "merra2" else 70        # bottom levels covering the lowest ~ 4 km (L72 / L117)

    def obs_month(self, mo):
        if not self.obspack:
            return None
        wanted = {s["obs"]["dataset"] for s in self.sites if s["obs"] and s["obs"]["license_cc0"]}

        def compute():
            y, m = map(int, mo.split("-"))
            files = sorted(glob.glob(os.path.join(self.obspack, "data", "daily", f"*.{y:04d}{m:02d}??.nc")))
            recs = [r for chunk in self._pool(_obs_file, [(f, wanted) for f in files]) for r in chunk]
            if not recs:
                return dict(site=np.array([], "U8"), sec=np.array([]), value=np.array([]), flag=np.array([], int))
            code_of = {s["obs"]["dataset"]: s["code"] for s in self.sites if s["obs"]}
            return dict(site=np.array([code_of[r[0]] for r in recs]), sec=np.array([r[1] for r in recs]),
                        value=np.array([r[2] for r in recs]), flag=np.array([r[3] for r in recs], int))
        return self._cached(f"obs_{mo}", compute)

    # ---- the data files
    def write(self, data_dir):
        months = self.s.months
        if not months:
            return
        t0 = dt.datetime.strptime(months[0], "%Y-%m")
        y, m = map(int, months[-1].split("-"))
        t1 = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        n = int((t1 - t0).total_seconds() // 10800)
        idx = lambda hours: np.rint((T_ORIGIN - t0).total_seconds() / 10800 + np.asarray(hours) / 3).astype(int)
        series = {"gc": np.full((n, len(self.sites), len(STATION_TRACERS)), np.nan)}
        for rk in self.s.main:
            series[rk] = np.full_like(series["gc"], np.nan)
        pstar = np.full((n, len(self.sites)), np.nan)
        obs = {s["code"]: [] for s in self.sites}
        for mo in months:
            gc = self.gc_month(mo)
            if gc is not None:
                k = idx(gc["hours"]); ok = (k >= 0) & (k < n)
                series["gc"][k[ok]] = gc["values"][ok]; pstar[k[ok]] = gc["pstar"][ok]
            for rk in self.s.main:
                if mo not in self.s.runs[rk]["prod"].months:
                    continue
                rv = self.run_month(rk, mo, gc)
                if rv is not None:
                    k = idx(rv["hours"]); ok = (k >= 0) & (k < n)
                    series[rk][k[ok]] = rv["values"][ok]
            ob = self.obs_month(mo)
            if ob is not None:
                for c, sec, v, fl in zip(ob["site"], ob["sec"], ob["value"], ob["flag"]):
                    obs[str(c)].append((float(sec), float(v), int(fl)))
        stats, meta_sites, ostats = {}, [], {}
        period = set(self._obs_period())
        in_period = np.array([(t0 + dt.timedelta(days=d)).strftime("%Y-%m") in period for d in range(n // 8)], bool)
        for s, site in enumerate(self.sites):
            payload = {"t0": t0.strftime("%Y-%m-%dT%H:%M"), "step_h": 3, "n": n, "series": {}}
            for key, arr in series.items():
                payload["series"][key] = {tk: encode_int16(arr[:, s, k] * TRACERS[tk]["scale"])
                                          for k, (tk, _, _) in enumerate(STATION_TRACERS)}
            payload["obs"] = self._obs_payload(site, obs[site["code"]])
            if payload["obs"]:
                ostats[site["code"]] = self._obs_stats(payload["obs"], series, s, t0, n)
            write_js(os.path.join(data_dir, "stations", f"{site['code']}.js"), "SITE.put", f"station|{site['code']}", payload)
            # daily-mean statistics against GCHP per run and tracer (for the summary table)
            for rk in self.s.main:
                for k, (tk, _, _) in enumerate(STATION_TRACERS):
                    a = self._daily(series[rk][:, s, k], n) * TRACERS[tk]["scale"]
                    b = self._daily(series["gc"][:, s, k], n) * TRACERS[tk]["scale"]
                    ok = np.isfinite(a) & np.isfinite(b) & in_period[:len(a)]   # same days for every run
                    if ok.sum() < 3:
                        continue
                    stats[f"{site['code']}|{rk}|{tk}"] = dict(gc_mean=sig(b[ok].mean(), 6), bias=sig((a - b)[ok].mean(), 4),
                                                             rmse=sig(np.sqrt(((a - b)[ok] ** 2).mean()), 4),
                                                             r=sig(np.corrcoef(a[ok], b[ok])[0, 1], 4), days=int(ok.sum()))
            ps = pstar[:, s]
            meta_sites.append(dict({k: site[k] for k in ("code", "name", "lat", "lon", "elev", "intake", "mode", "cell", "cell_lat",
                                                          "cell_lon", "surface_m", "target_asl", "source")},
                                   dist_km=sig(site["dist_km"], 3), pstar_median=sig(np.nanmedian(ps), 4) if np.isfinite(ps).any() else None,
                                   obs=site["obs"]))
        write_js(os.path.join(data_dir, "stations.js"), "SITE.put", "stations",
                 dict(sites=meta_sites, stats=stats, obs_stats=ostats, obs_period=self._obs_period(),
                      skipped=[dict(dataset=a, reason=b) for a, b in self.skipped],
                      obspack=dict(available=self.obspack is not None, product=OBSPACK_PRODUCT,
                                   doi="https://doi.org/10.25925/20241101",
                                   citation_url="https://gml.noaa.gov/ccgg/obspack/citation.php")))
        print(f"wrote station series for {len(self.sites)} sites ({', '.join(s['code'] for s in self.sites)})")

    def _obs_period(self):
        """Months scored against the observations: the months every run on the site has compared (without the
        spin-up month), so every model is scored on the same days."""
        return list(self.s.common)

    def _obs_stats(self, ob, series, s, t0, n):
        """CO2 against the observations, per model (gc and the runs), over _obs_period(): in situ = daily means of
        the obs_flag 1 hours against the model's daily mean on the same days; flask = the model's 3-hourly series
        interpolated linearly in time to each sample. n, bias (model - obs), RMSE, centred RMSE, r, and the
        seasonal amplitude (peak to trough of linearly detrended monthly means of the matched pairs, >= 12 months)."""
        months = set(self._obs_period())
        k_co2 = [tk for tk, _, _ in STATION_TRACERS].index("co2")
        out = {}
        for key, arr in series.items():
            v = arr[:, s, k_co2].astype(float) * 1e6
            pairs = []                                     # (month, model, obs)
            if ob["kind"] == "insitu":
                daily = self._daily(v, n)
                for day, o in zip(ob["t"], ob["v"]):
                    if day[:7] not in months or o is None:
                        continue
                    d = (dt.datetime.strptime(day, "%Y-%m-%d") - t0).days
                    if 0 <= d < len(daily) and np.isfinite(daily[d]) and np.isfinite(v[8 * d:8 * d + 8]).sum() >= 4:
                        pairs.append((day[:7], daily[d], o))
            else:
                for when, o in zip(ob["t"], ob["v"]):
                    if when[:7] not in months or o is None:
                        continue
                    x = (dt.datetime.strptime(when, "%Y-%m-%d %H:%M") - t0).total_seconds() / 10800.0
                    k = int(np.floor(x))
                    if 0 <= k < n - 1 and np.isfinite(v[k]) and np.isfinite(v[k + 1]):
                        w = x - k
                        pairs.append((when[:7], (1 - w) * v[k] + w * v[k + 1], o))
            if len(pairs) < 5:
                continue
            m = np.array([p[1] for p in pairs]); o = np.array([p[2] for p in pairs]); d = m - o
            res = dict(n=len(pairs), bias=sig(d.mean(), 4), rmse=sig(np.sqrt((d ** 2).mean()), 4),
                       crmse=sig(np.sqrt(((d - d.mean()) ** 2).mean()), 4), r=sig(np.corrcoef(m, o)[0, 1], 4))
            mons = sorted({p[0] for p in pairs})
            if len(mons) >= 12:
                idx = {mo: i for i, mo in enumerate(mons)}
                mm = np.array([np.mean([p[1] for p in pairs if p[0] == mo]) for mo in mons])
                mo_ = np.array([np.mean([p[2] for p in pairs if p[0] == mo]) for mo in mons])
                xm = np.array([int(mo[:4]) * 12 + int(mo[5:]) for mo in mons], float)
                amp = lambda y: float(np.ptp(y - np.polyval(np.polyfit(xm, y, 1), xm)))
                res.update(amp_model=sig(amp(mm), 4), amp_obs=sig(amp(mo_), 4), months=len(mons))
            out[key] = res
        return out

    @staticmethod
    def _daily(x, n):
        import warnings
        d = n // 8
        with warnings.catch_warnings():                 # all-missing days give NaN without a warning
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.asarray(x[:d * 8], float).reshape(d, 8), axis=1) if d else np.array([])

    def _obs_payload(self, site, recs):
        o = site["obs"]
        if not o or not o["license_cc0"] or not recs:
            return None
        recs = sorted(recs)
        if o["kind"] == "insitu":            # daily means of the hours the provider marks representative (obs_flag 1)
            days = {}
            for sec, v, fl in recs:
                if fl == 1 and np.isfinite(v):
                    day = dt.datetime.utcfromtimestamp(sec).strftime("%Y-%m-%d")
                    days.setdefault(day, []).append(v)
            keys = sorted(days)
            return dict(kind="insitu", dataset=o["dataset"], t=keys, v=[sig(np.mean(days[k]) * 1e6, 6) for k in keys],
                        n=[len(days[k]) for k in keys])
        pts = [(sec, v) for sec, v, fl in recs if fl == 1 and np.isfinite(v)]
        return dict(kind="flask", dataset=o["dataset"],
                    t=[dt.datetime.utcfromtimestamp(sec).strftime("%Y-%m-%d %H:%M") for sec, _ in pts],
                    v=[sig(v * 1e6, 6) for _, v in pts])


def coastlines():
    """Natural Earth 110 m coastlines as lon/lat polylines (null separates lines); empty without cartopy."""
    try:
        import cartopy.io.shapereader as shpreader
        path = shpreader.natural_earth(resolution="110m", category="physical", name="coastline")
        lon, lat = [], []
        for rec in shpreader.Reader(path).geometries():
            geoms = getattr(rec, "geoms", [rec])
            for gm in geoms:
                xy = np.asarray(gm.coords)
                lon.extend(np.round(xy[:, 0], 2).tolist() + [None])
                lat.extend(np.round(xy[:, 1], 2).tolist() + [None])
        return {"lon": lon, "lat": lat}
    except Exception as err:        # optional decoration
        print(f"warning: no coastlines ({err})")
        return {"lon": [], "lat": []}


# ---------------------------------------------------------------- the pages --

def d3_js(dest, local):
    """The D3 bundle next to the pages (pinned version, checksum verified), with its ISC licence text."""
    with open(os.path.join(os.path.dirname(dest), "d3.LICENSE.txt"), "w", encoding="utf-8") as fh:
        fh.write(D3_LICENSE)
    if os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == D3_SHA256:
        return
    if local:
        code = open(os.path.expanduser(local), "rb").read()
    else:
        with urllib.request.urlopen(D3_URL, timeout=60) as response:
            code = response.read()
    if hashlib.sha256(code).hexdigest() != D3_SHA256:
        raise SystemExit("D3 bundle checksum mismatch; pass --d3-js with d3.min.js v7.9.0")
    with open(dest, "wb") as fh:
        fh.write(code)


def render_pages(out, meta):
    with open(os.path.join(ASSETS, "layout.html"), encoding="utf-8") as fh:
        layout = fh.read()
    for fname, label, title in PAGES:
        with open(os.path.join(ASSETS, "pages", fname), encoding="utf-8") as fh:
            body = fh.read()
        nav = "".join(f'<a href="{f}"{" aria-current=page" if f == fname else ""}>{html.escape(lab)}</a>'
                      for f, lab, _ in PAGES)
        page = (layout.replace("{{TITLE}}", html.escape(title)).replace("{{NAV}}", nav)
                .replace("{{BODY}}", body).replace("{{PAGE}}", fname[:-5])
                .replace("{{GENERATED}}", html.escape(meta["generated"]))
                .replace("{{REVISION}}", html.escape(meta["revision"]))
                .replace("{{VERSION}}", meta["generated"].replace(" ", "").replace(":", "")))
        with open(os.path.join(out, fname), "w", encoding="utf-8") as fh:
            fh.write(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="site directory to write (created; data/ is rewritten)")
    ap.add_argument("--signoff-dir", default="/temp1/cfranken/jobs/catrine_signoff_v05",
                    help="run directory of the signoff chain (compare/, massbal/, SIGNOFF_TABLE.json)")
    ap.add_argument("--era5-label", default="era5_protic")
    ap.add_argument("--merra2-label", default="m2_protic_fix")
    ap.add_argument("--gcic-label", default="m2_gcic_fix", help="MERRA-2 run started from GCHP's initial state")
    root = "~/data/AtmosTransport/catrine_protocol_output_v05"
    ap.add_argument("--era5-output", default=f"{root}/era5_l117_li_v05_protic", help="only to count transported days")
    ap.add_argument("--merra2-output", default=f"{root}/merra2_l72_hm_gchp_v05_protic_fix")
    ap.add_argument("--gcic-output", default=f"{root}/merra2_l72_hm_gchp_v05_gcic_fix")
    ap.add_argument("--gc-dir", default="~/data/AtmosTransport/catrine-geoschem-runs/C90",
                    help="GCHP CATRINE_inst files under YYYY/MM/ (grid, areas, tropopause)")
    ap.add_argument("--gc-massbal", default="/temp1/cfranken/catrine_protocol/true_mass_balance/geoschem.csv",
                    help="GCHP true mass balance csv (catrine_true_mass_balance.jl --model geoschem)")
    ap.add_argument("--datasets", default="auto",
                    help="'auto' (default): MERRA-2, plus ERA5 once all 25 months are compared; or a comma list of "
                         "merra2 and era5, e.g. --datasets merra2,era5 to show ERA5 before it is complete")
    ap.add_argument("--obspack-dir", default="/kiwi-data/Data/model/OCO2MIP/observation_input/"
                    "obspack_co2_1_OCO2MIP_v5.0.1_2024-11-14",
                    help="ObsPack (GLOBALVIEWplus v10.1 records) for site metadata and CO2 observations; '' = none")
    ap.add_argument("--station-workers", type=int, default=8,
                    help="processes for the station extraction (I/O bound; 0 = no station page data)")
    ap.add_argument("--d3-js", default=None, help="local d3.min.js v7.9.0 (default: download once)")
    a = ap.parse_args()
    a.signoff_dir = os.path.expanduser(a.signoff_dir)
    if a.datasets != "auto" and (not a.datasets or not set(a.datasets.split(",")) <= {"merra2", "era5"}):
        ap.error("--datasets: 'auto' or a comma list of merra2 and era5")

    site = Site(a)
    os.makedirs(os.path.join(site.out, "assets"), exist_ok=True)
    meta = site.write_all()
    assets = os.path.join(site.out, "assets")
    d3_js(os.path.join(assets, "d3.min.js"), a.d3_js)
    for f in ("site.css", "site.js"):
        shutil.copy(os.path.join(ASSETS, f), os.path.join(assets, f))
    for f in os.listdir(assets):           # e.g. the Plotly bundle of earlier builds
        if f not in ASSET_FILES:
            path = os.path.join(assets, f)
            shutil.rmtree(path) if os.path.isdir(path) and not os.path.islink(path) else os.remove(path)
    render_pages(site.out, meta)
    with open(os.path.join(site.out, ".catrine_benchmark_site"), "w") as fh:     # marker checked by publish.sh
        fh.write(f"built {meta['generated']} by catrine_benchmark_v05_site.py {meta['revision']}\n")
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(site.out) for f in fs
               if "_cache" not in dp)
    print(f"site written to {site.out} ({size / 1e6:.1f} MB); months: "
          + ", ".join(f"{k} {len(r['prod'].months)}" for k, r in site.runs.items())
          + f"; common (period) {', '.join(site.common) or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
