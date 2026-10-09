#!/usr/bin/env python3
"""Write the interactive CATRINE benchmark web page from a catrine_compare_vs_geoschem.py output.

Reads <compare-dir>/summary.json and timeseries.npz (and, optionally, the csv files of
catrine_true_mass_balance*.{jl,py}) and writes a self-contained web directory --out:
  * index.html - Chart.js charts with check boxes for the runs and selectors for tracer, band
                 and statistic (the selection is kept in the URL, so a view can be linked):
                 daily means of the 3-hourly agreement statistics with GEOS-Chem (R^2, OLS
                 slope, bias, RMSE), statistics of the monthly mean fields, a period table,
                 global burdens (absolute or relative to GEOS-Chem) and the true mass balance;
                 the code versions; maps, zonal means, animations and extra figures in
                 collapsible sections. The data are embedded in display units (bias and RMSE
                 in ppm / ppt / 1e-21 mol/mol), so the page also works from disk and offline;
  * chart.umd.min.js (downloaded once), summary.json (raw statistics, mol/mol), figures/.

  python3 scripts/diagnostics/catrine_benchmark_page.py \\
      --compare-dir /temp1/cfranken/catrine_protocol/compare_c90_full_hm_gchp \\
      --out ~/www/catrine/benchmark_v2 \\
      --run 'merra2_old=MERRA-2, mass-weighted closure' --code 'merra2_old=transport 4d2379f0' \\
      --mass-balance 'gc=geoschem.csv' --mass-balance 'merra2_old=merra2_old.csv'
"""
import argparse, csv, datetime as dt, html, json, os, re, shutil, subprocess, urllib.request
from collections import namedtuple
import numpy as np

Tracer = namedtuple("Tracer", "key name unit scale digits ratio_digits burden_unit burden_scale column gc_column")
TRACERS = [  # scale: mol/mol → unit; burden_scale: kg → burden_unit; mass-balance csv columns (AtmosTransport, GEOS-Chem)
    Tracer("co2", "CO₂", "ppm", 1e6, 3, 4, "Pg CO₂", 1e-12, "co2_natural", "co2_natural"),
    Tracer("fossil", "Fossil CO₂ (from 2021-12-01)", "ppm", 1e6, 3, 5, "Pg CO₂", 1e-12,
           "co2_fossil_from_dec2021", "co2_fossil"),    # our co2_fossil is the protocol tracer from 2022-01-01
    Tracer("sf6", "SF₆", "ppt", 1e12, 4, 4, "Gg", 1e-6, "sf6", "sf6"),
    Tracer("rn222", "Rn-222", "10⁻²¹ mol/mol", 1e21, 4, 4, "g", 1e3, "rn222", "rn222")]
BANDS = [("column", "column"), ("sfc_910hPa", "0–10 % of air mass (surface–~910 hPa)"),
         ("910_400hPa", "10–60 % (~910–400 hPa)"), ("400_100hPa", "60–90 % (~400–100 hPa)"),
         ("above_100hPa", "90–100 % (above ~100 hPa)")]
STATS = [("bias", "bias", True), ("rmse", "RMSE", True), ("r2", "R²", False), ("slope", "slope", False)]  # key, name, in tracer units
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
MAPS = ["map_column_co2.png", "map_column_fossil.png", "map_column_sf6.png", "map_column_rn222.png",
        "zonal_co2.png", "zonal_fossil.png", "zonal_sf6.png", "zonal_rn222.png"]
CHART_JS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def script_revision():
    """Commit of the repository holding this script, with '+ local changes' if the tree is dirty."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))          # repository root: scripts/diagnostics/..
    if not os.path.exists(os.path.join(root, ".git")):     # an exported tree; never ask a parent repository
        path = os.path.join(root, "src", "REVISION")
        rev = open(path).read().strip() if os.path.exists(path) else ""
        return rev[:8] if re.fullmatch(r"[0-9a-f]{40}", rev) else "unknown"
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "--short=8", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", here],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + (" + local changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def chart_js(out):
    """Chart.js next to the page (works offline and from disk); the CDN URL if it cannot be downloaded."""
    path = os.path.join(out, os.path.basename(CHART_JS))
    if not os.path.exists(path):
        try:
            with urllib.request.urlopen(CHART_JS, timeout=30) as response:
                code = response.read()
            with open(path, "wb") as fh:
                fh.write(code)
        except OSError as e:
            print(f"warning: Chart.js not downloaded ({e}); the page loads it from {CHART_JS}")
            return CHART_JS
    return os.path.basename(CHART_JS)


class Days:
    """Calendar days of sub-daily times ('YYYY-mm-dd ...' strings) and daily means on them."""
    def __init__(self, times):
        days, self.day_of = np.unique([t[:10] for t in times], return_inverse=True)
        self.days = list(days)

    def mean(self, values):
        """Daily means, NaNs skipped (a day without finite values is NaN)."""
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(v)
        n = np.bincount(self.day_of[ok], minlength=len(self.days))
        total = np.bincount(self.day_of[ok], weights=v[ok], minlength=len(self.days))
        with np.errstate(invalid="ignore", divide="ignore"):
            return total / n


def number(x, digits=6):
    """JSON number with `digits` significant figures; NaN → null (a gap in the charts, '–' in tables)."""
    return float(f"{x:.{digits}g}") if np.isfinite(x) else None


def rounded(values, digits=5):
    return [number(x, digits) for x in values]


def mass_balance(path, origin, gc):
    """First time and daily residual per tracer of a true-mass-balance csv (GEOS-Chem's if `gc`).

    The residual is a global-mean dry mole fraction, residual / dry-air mass (both csv writers store
    kg dry-air equivalent), and for Rn-222 a percentage of its burden at the same time."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: empty csv")
    times = [(origin + dt.timedelta(hours=float(r["hours"]))).strftime("%Y-%m-%d %H:%M") for r in rows]
    days, out = Days(times), {}
    values = lambda name: np.array([float(r[name]) for r in rows])
    for t in TRACERS:
        name = t.gc_column if gc else t.column
        if f"{name}_residual" not in rows[0]:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            y = (100 * values(f"{name}_residual") / values(f"{name}_burden") if t.key == "rn222"
                 else t.scale * values(f"{name}_residual") / values("dry_air_kg"))
        out[t.key] = dict(zip(days.days, days.mean(y)))
    return times[0], out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-dir", required=True, help="output directory of catrine_compare_vs_geoschem.py")
    ap.add_argument("--out", required=True, help="web directory to write")
    ap.add_argument("--run", action="append", required=True, help="key=description, in display order")
    ap.add_argument("--code", action="append", default=[],
                    help="key=code versions of a run (met preprocessing and transport)")
    ap.add_argument("--mass-balance", action="append", default=[],
                    help="key=csv[,csv...] of catrine_true_mass_balance*.{jl,py} (tracers merged); key 'gc' for GEOS-Chem")
    ap.add_argument("--mass-balance-note", default="", help="HTML after the mass-balance chart")
    ap.add_argument("--origin", default="2021-12-01T00:00", help="time origin of the mass-balance csv hours")
    ap.add_argument("--title", default="CATRINE C90 benchmark: AtmosTransport vs GEOS-Chem (GCHP)")
    ap.add_argument("--intro", default="", help="HTML paragraph(s) after the title")
    ap.add_argument("--reference", help="code version of the reference run",
                    default="GEOS-Chem CATRINE C90 standard run (GCHP driven by MERRA-2; files say GEOS-Chem_devel)")
    ap.add_argument("--extra-figure", action="append", default=[],
                    help="title|path|caption(HTML): a further figure, copied to figures/")
    ap.add_argument("--video", action="append", default=[],
                    help="file|caption of an animation already in --out (poster: <file>_last.png)")
    a = ap.parse_args()

    for opt, sep, values in (("--run", "=", a.run), ("--code", "=", a.code), ("--mass-balance", "=", a.mass_balance),
                             ("--video", "|", a.video), ("--extra-figure", "|", a.extra_figure)):
        for v in values:
            if sep not in v:
                ap.error(f"{opt} {v!r}: expected 'name{sep}text'")
    summary_path = os.path.join(a.compare_dir, "summary.json")
    if not os.path.exists(summary_path):
        ap.error(f"no summary.json in {a.compare_dir}")
    with open(summary_path, encoding="utf-8") as fh:
        s = json.load(fh)
    ts = np.load(os.path.join(a.compare_dir, "timeseries.npz"))
    times = [str(t) for t in ts["times"]]
    t0, t1, n = s["times"]
    codes = dict(c.split("=", 1) for c in a.code)
    for opt, values in (("--run", a.run), ("--code", a.code), ("--mass-balance", a.mass_balance)):
        seen = [v.split("=", 1)[0] for v in values]
        if len(set(seen)) < len(seen):
            ap.error(f"{opt}: a key is given twice")
        if any(c in k for k in seen for c in ",|#&"):
            ap.error(f"{opt}: keys may not contain , | # or &")
    runs = []
    for i, spec in enumerate(a.run):
        key, desc = spec.split("=", 1)
        if not all(key in s[f] for f in ("period_mean_fields", "air_mass_ratio_to_gc", "final_burden_ratio_to_gc")):
            ap.error(f"--run {key!r} is not in {summary_path} (has {', '.join(s['period_mean_fields'])})")
        runs.append({"key": key, "desc": desc, "code": codes.get(key, "not recorded"), "color": COLORS[i % len(COLORS)],
                     "air_ratio": number(s["air_mass_ratio_to_gc"][key]),
                     "final_ratio": {t: number(v, 7) for t, v in s["final_burden_ratio_to_gc"][key].items()}})

    # Agreement statistics (3-hourly → daily means, monthly, period) and burdens, in display units.
    days, months = Days(times), sorted(s["monthly_mean_fields"])
    data = {"runs": runs, "days": days.days, "months": months, "final_time": t1, "reference": a.reference,
            "scripts": script_revision(), "daily": {}, "monthly": {}, "period": {}, "burden": {},
            "tracers": [{"key": t.key, "name": t.name, "unit": t.unit, "digits": t.digits,
                         "ratio_digits": t.ratio_digits, "burden_unit": t.burden_unit} for t in TRACERS],
            "bands": [{"key": k, "name": n} for k, n in BANDS],
            "stats": [{"key": k, "name": n, "in_units": u} for k, n, u in STATS]}
    for r in [r["key"] for r in runs] + ["gc"]:
        for t in TRACERS:
            data["burden"][f"{r}|{t.key}"] = rounded(days.mean(ts[f"{r}__{t.key}__burden"]) * t.burden_scale, 7)
            if r == "gc":
                continue
            for b, _ in BANDS:
                for st, _, in_units in STATS:
                    scale = t.scale if in_units else 1.0
                    data["daily"][f"{r}|{t.key}|{b}|{st}"] = rounded(days.mean(ts[f"{r}__{t.key}__{b}__{st}"]) * scale)
                    data["monthly"][f"{r}|{t.key}|{b}|{st}"] = rounded(
                        [s["monthly_mean_fields"][m][r][t.key][b][st] * scale for m in months])
                data["period"][f"{r}|{t.key}|{b}"] = {st: number(s["period_mean_fields"][r][t.key][b][st] *
                                                                 (t.scale if in_units else 1.0))
                                                      for st, _, in_units in STATS}

    # True mass balance: daily residuals on the union of the days of all csv files.
    origin, balance, starts = dt.datetime.fromisoformat(a.origin), {}, []
    keys = {r["key"] for r in runs}
    for c in codes:
        if c not in keys:
            ap.error(f"--code {c!r} is not a --run key")
    for spec in a.mass_balance:
        key, path = spec.split("=", 1)
        if key != "gc" and key not in keys:
            ap.error(f"--mass-balance {key!r} is neither 'gc' nor a --run key")
        balance[key] = {}
        for one in path.split(","):
            start, per = mass_balance(os.path.expanduser(one), origin, key == "gc")
            if balance[key].keys() & per.keys():
                ap.error(f"--mass-balance {key}: tracers {sorted(balance[key].keys() & per.keys())} in more than one csv")
            balance[key].update(per); starts.append(start)
        for t in TRACERS:
            if t.key not in balance[key]:
                print(f"warning: no {t.gc_column if key == 'gc' else t.column}_residual for {key}; no {t.key} mass balance")
    mb_days = sorted({d for per in balance.values() for series in per.values() for d in series})
    data["mass_balance"] = {"days": mb_days, "series": {f"{k}|{t}": rounded([series.get(d, np.nan) for d in mb_days])
                                                         for k, per in balance.items() for t, series in per.items()}}

    out = os.path.expanduser(a.out)
    if os.path.abspath(out) == os.path.abspath(a.compare_dir):
        ap.error("--out must differ from --compare-dir")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    have = []
    for f in MAPS:
        src = os.path.join(a.compare_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, "figures", f)); have.append(f)
        else:
            print("warning: missing figure", src)
    extra = ""
    for spec in a.extra_figure:
        title, path, caption = (spec.split("|", 2) + [""])[:3]
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print("warning: missing extra figure", path)
            continue
        shutil.copy(path, os.path.join(out, "figures", os.path.basename(path)))
        extra += (f"<details><summary>{html.escape(title)}</summary>{caption}"
                  f'<img src="figures/{html.escape(os.path.basename(path), quote=True)}"></details>')
    shutil.copy(summary_path, os.path.join(out, "summary.json"))

    for f in (v.split("|", 1)[0] for v in a.video):
        for need in (f, os.path.splitext(f)[0] + "_last.png"):
            if not os.path.exists(os.path.join(out, need)):
                print("warning: missing animation file", os.path.join(out, need))
    videos = "".join(f'<p>{html.escape(cap)}</p><video controls preload="metadata" src="{html.escape(f, quote=True)}" '
                     f'poster="{html.escape(os.path.splitext(f)[0], quote=True)}_last.png"></video>'
                     for f, cap in (v.split("|", 1) for v in a.video))
    fill = {"TITLE": html.escape(a.title), "INTRO": a.intro, "CHART_JS": html.escape(chart_js(out), quote=True),
            "TIMES": html.escape(f"{n} three-hourly instantaneous snapshots, {t0} to {t1}"),
            "PERIOD": html.escape(s["period"]), "REFERENCE": html.escape(a.reference),
            "MB_START": html.escape(min(starts) + " UTC" if starts else ""), "MB_NOTE": a.mass_balance_note,
            "MB_HIDDEN": "" if a.mass_balance else "hidden",
            "VIDEOS": f"<details><summary>Animations (one frame per day, 00 UTC)</summary>{videos}</details>" if videos else "",
            "MAPS": "<details><summary>Mean column maps and zonal means on dry pressure</summary>" +
                    "".join(f'<img src="figures/{f}">' for f in have) + "</details>" if have else "",
            "EXTRA": extra,
            # strict JSON (NaN raises), safe inside <script>
            "DATA": json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")}
    page = re.sub(r"\{\{(\w+)\}\}", lambda m: fill[m.group(1)], PAGE)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", os.path.join(out, "index.html"), f"({os.path.getsize(os.path.join(out, 'index.html')) / 1e6:.1f} MB)")


# The page: static text with placeholders {{NAME}}; charts and tables are drawn from the embedded data D.
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>{{TITLE}}</title>
<script src="{{CHART_JS}}"></script>
<style>
body{font-family:sans-serif;max-width:1400px;margin:auto;padding:1em;color:#222}
.controls{position:sticky;top:0;background:#f6f6f6;border:1px solid #ddd;padding:.4em .8em;z-index:10;font-size:13px}
.controls div{margin:.15em 0}.controls b{display:inline-block;width:5.5em}
label{margin-right:.9em;white-space:nowrap;cursor:pointer;font-size:13px}
.swatch{display:inline-block;width:.8em;height:.8em;border-radius:2px;margin:0 .25em 0 .1em}
.chart{position:relative;height:330px;margin:.4em 0 1.2em}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}
td.l,th.l{text-align:left}img,video{max-width:100%}h2{margin:1.4em 0 .3em}
details{margin:.6em 0}summary{cursor:pointer;font-weight:bold}
</style></head><body>
<h1>{{TITLE}}</h1>
<p>{{TIMES}}. Reference: {{REFERENCE}}. All fields are compared on the common C90 cube; statistics are area
weighted (bias = run − GEOS-Chem); bands are fractions of the column dry-air mass counted from the surface, so they
do not depend on the models' level sets.</p>
{{INTRO}}
<p>Choose runs, tracer, band and statistic in the bar below (it stays at the top while scrolling); hover a run name
for its description, and see the run table at the end for descriptions and code versions.</p>
<div class="controls">
<div><b>Runs</b><span id="runs"></span></div>
<div><b>Tracer</b><span id="tracer"></span></div>
<div><b>Band</b><span id="band"></span></div>
<div><b>Statistic</b><span id="stat"></span></div>
</div>
<h2>Agreement with GEOS-Chem through time</h2>
<p>Daily mean of the statistic of the 3-hourly instantaneous fields (selected tracer and band); the first and last
days are partial.</p>
<div class="chart"><canvas id="daily"></canvas></div>
<h2>Monthly mean fields</h2>
<p>The statistic of the monthly mean fields; December 2021 is the spin-up month. Band 90–100 % shows the
stratospheric drift.</p>
<div class="chart"><canvas id="monthly"></canvas></div>
<h2>Period mean fields</h2>
<p>Statistics of the fields averaged over {{PERIOD}}. Best value of each row in bold (R² highest, slope closest to 1,
|bias| and RMSE smallest).</p><div id="table"></div>
<h2>Global burden</h2>
<p><label><input type="radio" name="burden" value="ratio" checked>run / GEOS-Chem</label>
<label><input type="radio" name="burden" value="absolute">absolute</label>
Daily mean of the 3-hourly global burdens, or the ratio of these daily means to GEOS-Chem's. The dry-air mass ratio to GEOS-Chem's
<code>Met_AD</code> (run table below) sets the level of the long-lived tracers.</p>
<div class="chart"><canvas id="burden"></canvas></div>
<section {{MB_HIDDEN}}>
<h2>True mass balance</h2>
<p>Conservation residual: global burden change since {{MB_START}} minus the run's own applied emissions (Rn-222: and
radioactive decay), as a global-mean dry mole fraction (Rn-222: % of its burden); daily means. A conserving model
stays at zero. AtmosTransport: the run's compensated Float64 tracer totals and its sources rebuilt by the model
(<code>scripts/diagnostics/catrine_true_mass_balance.jl</code>); its Rn-222 reference follows the model's own
emit-then-decay order per met window. GEOS-Chem (black): <code>SpeciesConcVV × Met_AD</code> and its own
<code>Emis*</code> diagnostics (<code>catrine_true_mass_balance_gc.py</code>); its Rn-222 reference is the exact
solution of B' = E − λB, so GEOS-Chem's Rn-222 residual includes its own emission/decay splitting.</p>
<div class="chart"><canvas id="massbal"></canvas></div>{{MB_NOTE}}
</section>
<h2>Runs and code versions</h2><div id="runtable"></div>
{{VIDEOS}}{{MAPS}}{{EXTRA}}
<p><small>Generated by <code>scripts/diagnostics/catrine_compare_vs_geoschem.py</code> and
<code>scripts/diagnostics/catrine_benchmark_page.py</code> (AtmosTransport). Raw statistics (mol/mol):
<a href="summary.json">summary.json</a>. Charts: Chart.js.</small></p>
<script>
const D = {{DATA}};
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const byKey = (list, k) => list.find(x => x.key === k);
const GC = {key: "gc", label: "GEOS-Chem", color: "#000"};

// Selection, restored from and written to the URL hash (#runs=a,b&tracer=co2&band=column&stat=bias&burden=ratio).
const H = new URLSearchParams(location.hash.slice(1));
const pick = (list, k) => byKey(list, H.get(k)) ? H.get(k) : list[0].key;
const known = D.runs.map(r => r.key).concat("gc");
const sel = {runs: new Set(H.has("runs") ? H.get("runs").split(",").filter(k => known.includes(k)) : known),
             tracer: pick(D.tracers, "tracer"), band: pick(D.bands, "band"), stat: pick(D.stats, "stat"),
             burden: H.get("burden") === "absolute" ? "absolute" : "ratio"};
const box = (r, label, title) => `<label title="${esc(title)}"><input type="checkbox" value="${esc(r.key)}"
    ${sel.runs.has(r.key) ? "checked" : ""}><span class="swatch" style="background:${r.color}"></span>${esc(label)}</label>`;
document.getElementById("runs").innerHTML = D.runs.map(r => box(r, r.key, r.desc)).join("") +
    box(GC, "GEOS-Chem (absolute burden, mass balance)", "GEOS-Chem's absolute burden and its own mass balance");
for (const [id, list] of [["tracer", D.tracers], ["band", D.bands], ["stat", D.stats]])
  document.getElementById(id).innerHTML = list.map(x => `<label><input type="radio" name="${id}" value="${esc(x.key)}"
      ${x.key === sel[id] ? "checked" : ""}>${esc(x.name)}</label>`).join("");
document.querySelector(`input[name=burden][value=${sel.burden}]`).checked = true;

// One line chart per canvas, created once and updated in place; null values are gaps.
const charts = {};
const isZero = (scale, v) => Math.abs(v) < 1e-9 * (scale.max - scale.min);    // tick values carry rounding
const tick = function (v, i, ticks) {         // decimals from the tick spacing; exponent notation for tiny or huge axes
  if (isZero(this, v)) return "0";
  const step = ticks.length > 1 ? Math.abs(ticks[1].value - ticks[0].value) : Math.abs(v);
  const big = Math.max(...ticks.map(t => Math.abs(t.value)));
  const e = x => Math.floor(Math.log10(x) + 1e-9);
  return big < 1e-3 || big >= 1e6 ? v.toExponential(Math.max(0, e(big) - e(step))) : v.toFixed(Math.max(0, -e(step)));
};
function plot(id, labels, datasets, ytitle) {
  if (!charts[id]) charts[id] = new Chart(document.getElementById(id), {type: "line", data: {labels, datasets},
    options: {animation: false, maintainAspectRatio: false, spanGaps: false,
      interaction: {mode: "index", intersect: false}, elements: {point: {radius: 0}, line: {borderWidth: 1.3}},
      scales: {x: {ticks: {maxTicksLimit: 14}},
               y: {title: {display: true, text: ytitle}, ticks: {callback: tick},  // darker zero line: bias and residuals
                   grid: {color: c => isZero(c.scale, c.tick.value) ? "#999" : "rgba(0,0,0,0.08)"}}}}});
  const ch = charts[id];
  ch.data.labels = labels; ch.data.datasets = datasets; ch.options.scales.y.title.text = ytitle; ch.update();
}
const line = (r, data, extra = {}) => ({label: r.label || r.key, borderColor: r.color, backgroundColor: r.color, data, ...extra});
const runs = () => D.runs.filter(r => sel.runs.has(r.key));

function draw() {
  const t = byKey(D.tracers, sel.tracer), st = byKey(D.stats, sel.stat);
  const k = r => `${r.key}|${t.key}|${sel.band}|${st.key}`;
  const ytitle = st.in_units ? `${st.name} (${t.unit})` : st.name;
  plot("daily", D.days, runs().map(r => line(r, D.daily[k(r)])), ytitle);
  plot("monthly", D.months, runs().map(r => line(r, D.monthly[k(r)], {pointRadius: 2.5})), ytitle);
  const gc = sel.runs.has("gc") ? [GC] : [];
  if (sel.burden === "ratio") {
    const ref = D.burden[`gc|${t.key}`];
    const ratio = r => D.burden[`${r.key}|${t.key}`].map((x, i) => x === null || !ref[i] ? null : x / ref[i]);
    plot("burden", D.days, runs().map(r => line(r, ratio(r))), `${t.name} burden / GEOS-Chem`);
  } else {
    plot("burden", D.days, [...runs(), ...gc].map(r => line(r, D.burden[`${r.key}|${t.key}`])), `${t.name} burden (${t.burden_unit})`);
  }
  const mb = [...runs(), ...gc].filter(r => D.mass_balance.series[`${r.key}|${t.key}`])
    .map(r => line(r, D.mass_balance.series[`${r.key}|${t.key}`]));
  plot("massbal", D.mass_balance.days, mb, t.key === "rn222" ? "conservation residual (% of burden)" : `conservation residual (${t.unit})`);
  drawTable(t);
  history.replaceState(null, "", "#" + new URLSearchParams({runs: [...sel.runs].join(","), tracer: t.key,
                                                           band: sel.band, stat: st.key, burden: sel.burden}));
}

function drawTable(t) {
  const rs = runs();
  if (!rs.length) { document.getElementById("table").innerHTML = "<p>No run selected.</p>"; return; }
  const fmt = {r2: x => x.toFixed(4), slope: x => x.toFixed(3), rmse: x => x.toFixed(t.digits),
               bias: x => (x >= 0 ? "+" : "") + x.toFixed(t.digits)};
  const score = {r2: x => -x, slope: x => Math.abs(x - 1), bias: x => Math.abs(x), rmse: x => x};  // smaller is better
  const rank = (s, x) => x === null ? Infinity : score[s](x);
  let h = `<table><tr><th class="l">${esc(t.name)} (${esc(t.unit)})</th>` +
          rs.map(r => `<th colspan="${D.stats.length}" style="color:${r.color}">${esc(r.key)}</th>`).join("") + `</tr><tr><th class="l">band</th>` +
          rs.map(() => D.stats.map(s => `<th>${esc(s.name)}</th>`).join("")).join("") + "</tr>";
  for (const b of D.bands) {
    const v = rs.map(r => D.period[`${r.key}|${t.key}|${b.key}`]);
    const best = Object.fromEntries(D.stats.map(s => [s.key, Math.min(...v.map(x => rank(s.key, x[s.key])))]));
    h += `<tr><td class="l">${esc(b.name)}</td>` + v.map(x => D.stats.map(s => {
      const txt = x[s.key] === null ? "–" : fmt[s.key](x[s.key]);
      return rs.length > 1 && x[s.key] !== null && rank(s.key, x[s.key]) === best[s.key] ? `<td><b>${txt}</b></td>` : `<td>${txt}</td>`;
    }).join("")).join("") + "</tr>";
  }
  document.getElementById("table").innerHTML = h + "</table>";
}

const num = (x, d) => x === null ? "–" : x.toFixed(d);
const ratios = r => D.tracers.map(t => `${esc(t.name.split(" ")[0])}&nbsp;${num(r.final_ratio[t.key], t.ratio_digits)}`).join("<br>");
document.getElementById("runtable").innerHTML = `<table><tr><th class="l">run</th><th class="l">description</th>
    <th class="l">met preprocessing and transport code</th><th>dry air / GEOS-Chem</th>
    <th class="l">burden / GEOS-Chem at ${esc(D.final_time)}</th></tr>` +
  D.runs.map(r => `<tr><td class="l" style="color:${r.color}"><b>${esc(r.key)}</b></td><td class="l">${esc(r.desc)}</td>
    <td class="l">${esc(r.code)}</td><td>${num(r.air_ratio, 5)}</td><td class="l">${ratios(r)}</td></tr>`).join("") +
  `<tr><td class="l">reference</td><td class="l" colspan="4">${esc(D.reference)}</td></tr>
   <tr><td class="l">comparison</td><td class="l" colspan="4">page built by catrine_benchmark_page.py at ${esc(D.scripts)}</td></tr></table>`;

document.body.addEventListener("change", e => {
  const i = e.target;
  if (i.type === "checkbox") i.checked ? sel.runs.add(i.value) : sel.runs.delete(i.value);
  else if (i.name in sel) sel[i.name] = i.value;
  else return;
  draw();
});
draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
