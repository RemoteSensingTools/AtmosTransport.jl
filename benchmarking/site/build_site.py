#!/usr/bin/env python3
"""Build the AtmosTransport performance benchmark web site from the timing records.

    python3 benchmarking/site/build_site.py --data <data dir> --out <site dir>

<data dir>/<device>/records.csv holds one row per timed run (schema: FIELDS); an optional
<data dir>/<device>/PUBLIC_NOTES.md is shown on the Methods page (NOTES.md is the internal run log and is
never published). Devices, versions, grids, tracer counts and
operator sets that have no successful run are left out of the site (no empty axes or columns), so the
build can be repeated while the runs are still in progress. Python 3 standard library only.
"""
import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("device,backend,float_type,version,commit,pass,grid,tracers,operators,sim_days,status,"
          "steady_seconds_per_day,warm_seconds_per_day,peak_mem_mib,mean_util_percent,mean_power_w,"
          "load_avg_1m,utc,result_dir").split(",")
MARKER = ".atmostr_bench_site"
ASSET_FILES = ("site.css", "site.js", "d3.min.js", "d3.LICENSE.txt")
D3_SHA256 = "f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539"   # d3 v7.9.0 dist/d3.min.js
PAGES = [  # file, nav label, page title
    ("index.html", "Overview", "Overview"),
    ("releases.html", "Releases", "Release comparison"),
    ("scaling.html", "Scaling", "Resolution scaling"),
    ("operators.html", "Operators", "Operator cost breakdown"),
    ("devices.html", "Devices", "Devices"),
    ("methods.html", "Methods", "Methods and data"),
]

# Fixed colours, site-wide. Versions: owner rule (v0.5.0 grey, v0.6.0 blue, main orange). Devices: green,
# violet, magenta, amber, validated as a categorical set for colour-vision deficiency; every device chart
# also uses a marker shape and a text label, so identity never rests on colour alone.
VERSIONS = OrderedDict([
    ("v0.5.0", dict(color="#8f8d86", note="release v0.5.0")),
    ("v0.6.0", dict(color="#2a78d6", note="release v0.6.0")),
    ("main", dict(color="#eb6834", note="main with the performance pull requests #90 to #97; the 50 to 200 tracer runs also with the decay fix #99")),
])
DEVICES = OrderedDict([
    ("L40S", dict(name="L40S", long="NVIDIA L40S 48 GB (wurst)", kind="gpu", color="#008300", shape="circle")),
    ("A100", dict(name="A100", long="NVIDIA A100 40 GB PCIe (curry)", kind="gpu", color="#6250d6", shape="square")),
    ("M5Pro", dict(name="M5 Pro", long="Apple M5 Pro, Metal (MacBook)", kind="gpu", color="#d55181", shape="diamond")),
    ("CPU-wurst", dict(name="CPU", long="CPU, 32 Julia threads (wurst)", kind="cpu", color="#c98500", shape="triangle")),
])
OPS = OrderedDict([
    ("none", dict(name="none", long="met input and stepping only")),
    ("adv", dict(name="adv", long="advection")),
    ("diff", dict(name="diff", long="diffusion")),
    ("conv", dict(name="conv", long="convection")),
    ("adv+emis", dict(name="adv+emis", long="advection and emissions")),
    ("all", dict(name="all", long="all operators")),
])
# Operator breakdown components (s/day). rest = all - (input + the four operator costs); it can be negative.
COMPONENTS = OrderedDict([
    ("input", dict(name="input floor", long="met input and stepping (none)", color="#bdbbb1")),
    ("advection", dict(name="advection", long="adv − none", color="#1baf7a")),
    ("diffusion", dict(name="diffusion", long="diff − none", color="#6250d6")),
    ("convection", dict(name="convection", long="conv − none", color="#eda100")),
    ("emissions", dict(name="emissions", long="(adv+emis) − adv", color="#e34948")),
    ("rest", dict(name="interaction / rest", long="all − (none + the four costs)", color="hatch")),
])
LEVELS = 72
MET = {720: "GEOS-FP"}           # grids whose binaries are not MERRA-2 derived
SPREAD_FLAG = 0.10               # |pass 1 - pass 2| / mean above this is flagged
INPUT_BOUND = 0.5                # 'none' / selected operator set at or above this = input-bound
FLOAT_TYPE = "Float32"           # the site compares this precision; other rows stay in records.csv only
# Planned matrix per device (RUN_SPEC.md), counted in runs (configs x passes); C360/C720 are extra.
PLAN = dict(release=dict(versions=3, grids=("C24", "C48", "C90", "C180"), tracers=(1, 12), ops=("none", "adv", "all"), passes=2),
            opbreak=dict(versions=3, grids=("C90", "C180"), tracers=(12,), ops=("diff", "conv", "adv+emis"), passes=2),
            scaling=dict(versions=1, grids=("C24", "C48", "C90", "C180"), tracers=(4,), ops=("adv", "all"), passes=1))
# Devices with a reduced matrix: the CPU runs the release grids up to C90 with adv and all, one pass.
DEVICE_PLAN = {"CPU-wurst": dict(release=dict(versions=3, grids=("C24", "C48", "C90"), tracers=(1, 12), ops=("adv", "all"), passes=1))}
HEADLINES = [("C90", 12, "all"), ("C180", 12, "all"), ("C48", 12, "all"), ("C24", 12, "all"),
             ("C90", 1, "all"), ("C48", 1, "all"), ("C24", 1, "all")]
SECONDS_PER_YEAR_DAYS = 365.0


def plan_for(d):
    return DEVICE_PLAN.get(d, PLAN)


def planned_runs(plan=PLAN):
    return sum(p["versions"] * len(p["grids"]) * len(p["tracers"]) * len(p["ops"]) * p["passes"] for p in plan.values())


def sypd(s_per_day):
    """Simulated years per wall-clock day."""
    return 86400.0 / (SECONDS_PER_YEAR_DAYS * s_per_day)


def grid_n(grid):
    m = re.fullmatch(r"C(\d+)", grid)
    return int(m.group(1)) if m else None


def cells(n):
    return 6 * n * n * LEVELS


def num(text):
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def sig(x, s=3):
    """Significant figures without an exponent for ordinary magnitudes."""
    if x is None or not math.isfinite(x):
        return "–"
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    return f"{x:.{max(0, s - 1 - e)}f}"


def times(x):
    return f"{x:.2f}×" if x < 10 else f"{x:.1f}×"


def pct(x):
    return f"{100 * x:.0f} %"


def cfg_label(grid, tracers, ops):
    return f"{grid}, {tracers} tracer{'s' if tracers != 1 else ''}, {OPS[ops]['long'] if ops in OPS else ops}"


# ---------------------------------------------------------------- reading

def read_records(data_dir):
    rows, problems = [], []
    if not os.path.isdir(data_dir):
        raise SystemExit(f"data directory not found: {data_dir}")
    for dev in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, dev, "records.csv")
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [f for f in ("device", "version", "pass", "grid", "tracers", "operators", "status",
                                   "steady_seconds_per_day", "utc") if f not in (reader.fieldnames or [])]
            if missing:
                problems.append(f"{path}: missing columns {', '.join(missing)}; skipped")
                continue
            for k, r in enumerate(reader, start=2):
                r = {f: (r.get(f) or "").strip() for f in FIELDS}
                if r["device"] != dev:
                    problems.append(f"{path}:{k}: device {r['device']!r} in the {dev} folder; row skipped")
                    continue
                rows.append(r)
    return rows, problems


def read_notes(data_dir):
    notes = {}
    for dev in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, dev, "PUBLIC_NOTES.md")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                notes[dev] = markdown(fh.read())
    return notes


def markdown(text):
    """Small Markdown subset (headings, bullet lists, paragraphs, `code`, **bold**, *italic*, links) to HTML."""
    def inline(s):
        # Code spans and link targets are set aside first so that * inside them stays literal; *italic*
        # needs a non-word character on both sides, so 2*3*4 and file*name* stay as written.
        keep = []
        def stash(h):
            keep.append(h)
            return f"\x00{len(keep) - 1}\x00"
        s = html.escape(s, quote=False)
        s = re.sub(r"`([^`]+)`", lambda m: stash(f"<code>{m[1]}</code>"), s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", lambda m: stash(f'<a href="{m[2]}">') + m[1] + stash("</a>"), s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\w*])\*([^*\s](?:[^*]*[^*\s])?)\*(?![\w*])", r"<i>\1</i>", s)
        return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m[1])], s)
    out, para, items = [], [], []
    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("#"):
            flush()
            out.append(f"<h4>{inline(s.lstrip('#').strip())}</h4>")
        elif re.match(r"^[-*] ", s):
            if para:
                flush()
            items.append(s[2:].strip())
        elif items and line.startswith((" ", "\t")):
            items[-1] += " " + s
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return "\n".join(out)


# ---------------------------------------------------------------- aggregation

class Bench:
    def __init__(self, rows):
        self.rows = rows
        self.unknown = sorted({r["device"] for r in rows if r["device"] not in DEVICES})
        # one row per (config, pass): the latest successful run; a failure only when no run succeeded
        latest = {}
        for r in rows:
            n, t = grid_n(r["grid"]), num(r["tracers"])
            if n is None or t is None or r["operators"] not in OPS or r["version"] not in VERSIONS:
                continue
            ok = r["status"] == "ok" and (num(r["steady_seconds_per_day"]) or 0) > 0
            key = (r["device"], r["version"], r["grid"], int(t), r["operators"], r["pass"] or "1")
            rank = (ok, r["utc"])
            if key not in latest or rank > latest[key][0]:
                latest[key] = (rank, r)
        self.runs = [v[1] for v in latest.values()]
        groups = defaultdict(list)
        self.failures = []
        for r in self.runs:
            key = (r["device"], r["version"], r["grid"], int(num(r["tracers"])), r["operators"])
            if r["status"] == "ok" and (num(r["steady_seconds_per_day"]) or 0) > 0:
                groups[key].append(r)
            else:
                self.failures.append(dict(d=key[0], v=key[1], g=key[2], t=key[3], o=key[4], p=r["pass"],
                                          status=r["status"] or "no steady value", utc=r["utc"]))
        self.failures.sort(key=lambda f: (f["d"], f["v"], grid_n(f["g"]), f["t"], f["o"]))
        self.cfg = {}
        for key, rs in groups.items():
            rs.sort(key=lambda r: (r["pass"], r["utc"]))
            vals = [num(r["steady_seconds_per_day"]) for r in rs]
            mean = statistics.fmean(vals)
            n = grid_n(key[2])
            spread = (max(vals) - min(vals)) / mean if len(vals) > 1 else None
            mems = [num(r["peak_mem_mib"]) for r in rs if num(r["peak_mem_mib"])]
            utils = [num(r["mean_util_percent"]) for r in rs if num(r["mean_util_percent"]) is not None]
            power = [num(r["mean_power_w"]) for r in rs if num(r["mean_power_w"]) is not None]
            loads = [num(r["load_avg_1m"]) for r in rs if num(r["load_avg_1m"]) is not None]
            self.cfg[key] = dict(
                d=key[0], v=key[1], g=key[2], t=key[3], o=key[4], s=mean,
                p=[dict(n=r["pass"], s=v, utc=r["utc"]) for r, v in zip(rs, vals)],
                spread=spread, flag=bool(spread is not None and spread > SPREAD_FLAG),
                sypd=sypd(mean), cost_ns=1e9 * mean / (cells(n) * key[3]),
                mem=max(mems) if mems else None, util=statistics.fmean(utils) if utils else None,
                power=statistics.fmean(power) if power else None, load=loads,
                commit=rs[-1]["commit"], backend=rs[-1]["backend"], float_type=rs[-1]["float_type"],
                sim_days=rs[-1]["sim_days"])
        self.devices = [d for d in list(DEVICES) + self.unknown if any(k[0] == d for k in self.cfg)]
        self.versions = [v for v in VERSIONS if any(k[1] == v for k in self.cfg)]

    def get(self, d, v, g, t, o):
        return self.cfg.get((d, v, g, t, o))

    def s(self, d, v, g, t, o):
        c = self.get(d, v, g, t, o)
        return c["s"] if c else None

    def grids(self, d=None, v=None):
        return sorted({k[2] for k in self.cfg if (d is None or k[0] == d) and (v is None or k[1] == v)}, key=grid_n)

    def best_version(self, d):
        for v in reversed(VERSIONS):
            if any(k[0] == d and k[1] == v for k in self.cfg):
                return v
        return None

    def breakdowns(self):
        done, partial = [], []
        for (d, v, g, t) in sorted({k[:4] for k in self.cfg}, key=lambda k: (k[0], k[1], grid_n(k[2]), k[3])):
            vals = {o: self.s(d, v, g, t, o) for o in OPS}
            if vals["all"] is None and vals["none"] is None:
                continue
            missing = [o for o, x in vals.items() if x is None]
            if missing:
                ob = plan_for(d).get("opbreak")
                planned = ob is not None and g in ob["grids"] and t in ob["tracers"]
                if vals["all"] is not None and (planned or any(vals[o] is not None for o in ("diff", "conv", "adv+emis"))):
                    partial.append(dict(d=d, v=v, g=g, t=t, missing=missing))
                continue
            parts = OrderedDict(input=vals["none"], advection=vals["adv"] - vals["none"],
                                diffusion=vals["diff"] - vals["none"], convection=vals["conv"] - vals["none"],
                                emissions=vals["adv+emis"] - vals["adv"])
            parts["rest"] = vals["all"] - sum(parts.values())
            done.append(dict(d=d, v=v, g=g, t=t, all=vals["all"], parts=parts))
        return done, partial

    def progress(self, d):
        counted = 0
        for name, p in plan_for(d).items():
            vs = list(VERSIONS) if p["versions"] == 3 else ["main"]
            for v in vs:
                for g in p["grids"]:
                    for t in p["tracers"]:
                        for o in p["ops"]:
                            c = self.get(d, v, g, t, o)
                            if c:
                                counted += min(len(c["p"]), p["passes"])
        extra = sum(len(c["p"]) for k, c in self.cfg.items() if k[0] == d and grid_n(k[2]) > 180)
        runs = [r for r in self.runs if r["device"] == d]
        return dict(done=counted, planned=planned_runs(plan_for(d)), extra=extra,
                    failed=sum(1 for f in self.failures if f["d"] == d),
                    last=max((r["utc"] for r in runs), default=""))


# ---------------------------------------------------------------- headline numbers

def nice_utc(stamp):
    try:
        t = dt.datetime.strptime(stamp, "%Y%m%dT%H%M%S")
    except ValueError:
        return stamp
    return t.strftime("%-d %b %H:%M UTC")


def dev_name(d):
    return DEVICES[d]["name"] if d in DEVICES else d


def headline(B):
    """The headline configuration: on the first device (DEVICES order) with runs, the first of HEADLINES
    that has a run, preferring main, then v0.6.0, then v0.5.0."""
    for d in B.devices:
        for g, t, o in HEADLINES:
            for v in reversed(VERSIONS):
                if B.get(d, v, g, t, o):
                    return dict(d=d, v=v, g=g, t=t, o=o)
    if not B.cfg:
        return None
    # no preferred configuration measured yet: the newest version, all operators, most tracers, grid nearest C90
    vo, oo = list(VERSIONS), list(OPS)
    c = max(B.cfg.values(), key=lambda c: (-B.devices.index(c["d"]), vo.index(c["v"]), oo.index(c["o"]), c["t"],
                                           -abs(math.log(grid_n(c["g"]) / 90))))
    return dict(d=c["d"], v=c["v"], g=c["g"], t=c["t"], o=c["o"])


def shared_speedups(B, d, old, new):
    out = []
    for k, c in B.cfg.items():
        if k[0] == d and k[1] == new:
            o = B.get(d, old, k[2], k[3], k[4])
            if o:
                out.append((o["s"] / c["s"], k))
    return out


def tiles_and_points(B, breakdowns):
    tiles, points = [], []
    H = headline(B)
    if H is None:
        return [dict(l="Status", v="No runs yet", s="The timing runs have not produced a successful record.")], [], None
    d, v, g, t, o = H["d"], H["v"], H["g"], H["t"], H["o"]
    c = B.get(d, v, g, t, o)
    tiles.append(dict(l=f"{g} · {t} tracer{'s' if t > 1 else ''} · {OPS[o]['long']} · {dev_name(d)}",
                      v=f"{sig(c['sypd'])}", u="SYPD", c=VERSIONS[v]["color"],
                      s=f"{v}: {sig(c['s'])} s per simulated day" + ("" if v == "main" else "; main not measured yet")))
    # speedup since v0.5.0 for the headline configuration, else the median over shared configurations
    sp = None
    for new in ("main", "v0.6.0"):
        old = B.get(d, "v0.5.0", g, t, o)
        cur = B.get(d, new, g, t, o)
        if old and cur:
            sp = dict(l=f"Speedup since v0.5.0 · {dev_name(d)}", v=times(old["s"] / cur["s"]), u="", c=VERSIONS[new]["color"],
                      s=f"{new}, {g}, {t} tracers, {OPS[o]['long']}: {sig(old['s'])} → {sig(cur['s'])} s/day")
            break
        sh = shared_speedups(B, d, "v0.5.0", new)
        if sh:
            sp = dict(l=f"Speedup since v0.5.0 · {dev_name(d)}", v=times(statistics.median(x for x, _ in sh)), u="",
                      c=VERSIONS[new]["color"], s=f"median {new} / v0.5.0 over {len(sh)} shared configurations")
            break
    tiles.append(sp or dict(l="Speedup since v0.5.0", v="pending", u="", c=None,
                            s="needs v0.5.0 and main runs of the same configuration"))
    # largest grid run
    big = max(B.cfg.values(), key=lambda c: (grid_n(c["g"]), c["o"] == "all", c["v"] == "main", c["t"], -list(B.devices).index(c["d"])))
    met = MET.get(grid_n(big["g"]), "MERRA-2")
    mem = f", {sig(big['mem'] / 1024)} GiB peak memory" if big["mem"] else ""
    tiles.append(dict(l="Largest grid run", v=big["g"], u=f"{sig(cells(grid_n(big['g'])) / 1e6, 2)} M cells", c=None,
                      s=f"{dev_name(big['d'])}, {big['v']}, {big['t']} tracer{'s' if big['t'] > 1 else ''}, {OPS[big['o']]['long']}: "
                        f"{sig(big['sypd'])} SYPD{mem}; {met} binaries"))
    tiles.append(dict(l="Devices measured", v=f"{len(B.devices)}", u=f"of {len(DEVICES)}", c=None,
                      s=", ".join(f"{dev_name(x)} ({B.progress(x)['done']} runs)" for x in B.devices) +
                        ("; waiting for " + ", ".join(dev_name(x) for x in DEVICES if x not in B.devices) if len(B.devices) < len(DEVICES) else "")))

    # ---- key points, each only when the data supports it
    for dd in B.devices:                                   # release speedup
        sh = shared_speedups(B, dd, "v0.5.0", "main")
        if not sh:
            continue
        x = [s for s, _ in sh]
        hc = B.get(dd, "main", g, t, o) and B.get(dd, "v0.5.0", g, t, o)
        head = (f"For {g} with {t} tracers and {OPS[o]['long']}, main takes <b>{sig(B.s(dd, 'main', g, t, o))} s</b> per simulated day "
                f"against {sig(B.s(dd, 'v0.5.0', g, t, o))} s for v0.5.0 (<b>{times(B.s(dd, 'v0.5.0', g, t, o) / B.s(dd, 'main', g, t, o))}</b>). ") if hc else ""
        points.append(f"<b>{dev_name(dd)}, main against v0.5.0.</b> {head}Over the {len(x)} configurations measured with both, main is "
                      f"median <b>{times(statistics.median(x))}</b> as fast (range {times(min(x))} – {times(max(x))}).")
        break
    for dd in B.devices:                                   # what the perf PRs added
        sh = shared_speedups(B, dd, "v0.6.0", "main")
        if not sh:
            continue
        x = [s for s, _ in sh]
        best = max(sh)
        points.append(f"<b>Since v0.6.0 ({dev_name(dd)}).</b> The performance pull requests merged after v0.6.0 give median "
                      f"<b>{times(statistics.median(x))}</b> over {len(x)} configurations; the largest gain is {times(best[0])} "
                      f"for {cfg_label(best[1][2], best[1][3], best[1][4])}.")
        break
    # devices against each other (main, else best common version) at the headline configuration
    for vv in reversed(VERSIONS):
        have = [(dd, B.get(dd, vv, g, t, o)) for dd in B.devices if B.get(dd, vv, g, t, o)]
        if len(have) < 2:
            continue
        have.sort(key=lambda x: x[1]["s"])
        fast = have[0]
        cpu = next((x for x in have if DEVICES.get(x[0], {}).get("kind") == "cpu"), None)
        txt = ", ".join(f"{dev_name(x[0])} {sig(x[1]['sypd'])}" for x in have)
        cpu_txt = (f" The fastest GPU is <b>{times(cpu[1]['s'] / fast[1]['s'])}</b> as fast as the {DEVICES[cpu[0]]['long']}."
                   if cpu and cpu is not fast else "")
        points.append(f"<b>Devices ({vv}, {g}, {t} tracers, {OPS[o]['long']}).</b> Simulated years per day: {txt}.{cpu_txt}")
        break
    # small grids are input bound: share of 'none' in 'all'
    for vv in reversed(VERSIONS):
        fr = []
        for gg in B.grids(d, vv):
            a, n0 = B.get(d, vv, gg, t, "all"), B.get(d, vv, gg, t, "none")
            if a and n0:
                fr.append((grid_n(gg), gg, n0["s"] / a["s"], a["cost_ns"]))
        if len(fr) >= 2:
            lo, hi = fr[0], fr[-1]
            title = "Small grids are input-bound" if lo[2] >= INPUT_BOUND else "Share of met input and stepping"
            points.append(f"<b>{title} ({dev_name(d)}, {vv}, {t} tracers).</b> Met input and stepping alone "
                          f"(operator set <i>none</i>) take <b>{pct(lo[2])}</b> of the all-operator time at {lo[1]} and "
                          f"<b>{pct(hi[2])}</b> at {hi[1]}; the cost per cell and tracer goes from {sig(lo[3])} ns to "
                          f"{sig(hi[3])} ns per simulated day.")
            break
    bd = [b for b in breakdowns if b["d"] == d]
    if bd:
        b = max(bd, key=lambda b: (b["v"] == "main", b["v"] == "v0.6.0", grid_n(b["g"])))
        comp = [(k, x) for k, x in b["parts"].items() if k not in ("input", "rest")]
        k, x = max(comp, key=lambda kx: kx[1])
        points.append(f"<b>Where the time goes ({dev_name(d)}, {b['v']}, {b['g']}, {b['t']} tracers).</b> Of "
                      f"{sig(b['all'])} s per simulated day with all operators, {COMPONENTS[k]['name']} costs the most "
                      f"(<b>{pct(x / b['all'])}</b>); the input floor is {pct(b['parts']['input'] / b['all'])} and the "
                      f"interaction remainder {'+' if b['parts']['rest'] >= 0 else '−'}{pct(abs(b['parts']['rest']) / b['all'])}.")
    two = [c for c in B.cfg.values() if len(c["p"]) > 1]
    flagged = [c for c in two if c["flag"]]
    if two:
        worst = max(two, key=lambda c: c["spread"])
        points.append(f"<b>Repeatability.</b> {len(two) - len(flagged)} of {len(two)} configurations with two passes agree within "
                      f"{pct(SPREAD_FLAG)}" + (f"; the largest spread is {pct(worst['spread'])} ({dev_name(worst['d'])}, {worst['v']}, "
                                                f"{cfg_label(worst['g'], worst['t'], worst['o'])})." if flagged else "."))
    else:
        points.append(f"<b>Repeatability.</b> Only first passes so far ({len(B.cfg)} configurations); the second, interleaved pass "
                      f"will show the run-to-run spread.")
    return tiles, points[:5], H


def caveats(B):
    out = []
    for d in B.devices:
        loads = [x for c in B.cfg.values() if c["d"] == d for x in c["load"]]
        if loads:
            out.append(dict(d=d, html=f"Host 1-minute load average at the start of the runs: {sig(min(loads), 2)} – "
                                      f"{sig(max(loads), 2)} (median {sig(statistics.median(loads), 2)})."))
        flagged = [c for c in B.cfg.values() if c["d"] == d and c["flag"]]
        if flagged:
            out.append(dict(d=d, html=f"{len(flagged)} configuration{'s' if len(flagged) > 1 else ''} with passes more than "
                                      f"{pct(SPREAD_FLAG)} apart (s/day of pass 1 / pass 2): " + "; ".join(
                                          f"{c['v']} {c['g']} {c['t']} tr {c['o']} ({' / '.join(sig(p['s']) for p in c['p'])})"
                                          for c in flagged) + "."))
        bound = []
        for v in VERSIONS:
            for t in sorted({k[3] for k in B.cfg if k[0] == d and k[1] == v}):
                gs = [g for g in B.grids(d, v) if B.get(d, v, g, t, "all") and B.get(d, v, g, t, "none")
                      and B.s(d, v, g, t, "none") >= INPUT_BOUND * B.s(d, v, g, t, "all")]
                if gs:
                    bound.append(f"{v} {t} tr: " + ", ".join(
                        f"{g} ({pct(B.s(d, v, g, t, 'none') / B.s(d, v, g, t, 'all'))})" for g in gs))
        if bound:
            out.append(dict(d=d, html=f"Input-bound with all operators (<i>none</i> takes at least {pct(INPUT_BOUND)} of the "
                                      f"<i>all</i> time; share in brackets): " + "; ".join(bound) + "."))
        inv = []
        for c in B.cfg.values():
            if c["d"] == d and c["o"] != "none":
                n0 = B.get(d, c["v"], c["g"], c["t"], "none")
                if n0 and n0["s"] > c["s"] * 1.02:
                    inv.append(f"{c['v']} {c['g']} {c['t']} tr {c['o']}")
        if inv:
            out.append(dict(d=d, html="Measured faster than the <i>none</i> run (no operators) of the same version, grid and tracer "
                                      "count, so their operator costs come out negative: " + ", ".join(inv) + "."))
    return out


# ---------------------------------------------------------------- output

def git_revision():
    try:
        rev = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short=9", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", HERE, "status", "--porcelain", "--", "."], text=True, stderr=subprocess.DEVNULL).strip())
        return rev + (" (local changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def round_floats(x, digits=6):
    if isinstance(x, float):
        return float(f"{x:.{digits}g}") if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: round_floats(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_floats(v, digits) for v in x]
    return x


def payload(B, notes, now, revision, problems, other_float):
    breakdowns, partial = B.breakdowns()
    tiles, points, H = tiles_and_points(B, breakdowns)
    grids = sorted({c["g"] for c in B.cfg.values()}, key=grid_n)
    commits = defaultdict(set)
    for c in B.cfg.values():
        commits[c["v"]].add(c["commit"])
    devs = []
    for d in B.devices:
        meta = DEVICES.get(d, dict(name=d, long=d, kind="gpu", color="#52514e", shape="circle"))
        cs = [c for c in B.cfg.values() if c["d"] == d]
        devs.append(dict(key=d, **meta, backend=cs[0]["backend"], float_type=cs[0]["float_type"], progress=B.progress(d)))
    return dict(
        generated=now.strftime("%Y-%m-%d %H:%M UTC"), revision=revision,
        versions=[dict(key=v, name=v, color=VERSIONS[v]["color"], note=VERSIONS[v]["note"],
                       commits=sorted(x for x in commits[v] if x)) for v in B.versions],
        devices=devs, waiting=[dict(key=d, name=DEVICES[d]["name"], long=DEVICES[d]["long"]) for d in DEVICES if d not in B.devices],
        grids=[dict(key=g, n=grid_n(g), cells=cells(grid_n(g)), met=MET.get(grid_n(g), "MERRA-2")) for g in grids],
        tracers=sorted({c["t"] for c in B.cfg.values()}),
        ops=[dict(key=o, **OPS[o]) for o in OPS if any(c["o"] == o for c in B.cfg.values())],
        components=[dict(key=k, **v) for k, v in COMPONENTS.items()],
        configs=sorted(B.cfg.values(), key=lambda c: (c["d"], list(VERSIONS).index(c["v"]), grid_n(c["g"]), c["t"], list(OPS).index(c["o"]))),
        breakdown=breakdowns, partial=partial, failures=B.failures,
        tiles=tiles, points=points, headline=H, notes=notes, caveats=caveats(B), problems=problems,
        float_type=FLOAT_TYPE, other_float=other_float, spread_flag=SPREAD_FLAG, input_bound=INPUT_BOUND, planned=planned_runs(), levels=LEVELS)


def write_site(out, data, rows, version):
    os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, MARKER)
    if os.listdir(out) and not os.path.exists(marker):
        raise SystemExit(f"{out} is not empty and has no {MARKER} marker; refusing to write there")
    assets, ddir = os.path.join(out, "assets"), os.path.join(out, "data")
    for d in (assets, ddir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for f in ASSET_FILES:
        shutil.copyfile(os.path.join(HERE, "assets", f), os.path.join(assets, f))
    import hashlib
    with open(os.path.join(assets, "d3.min.js"), "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != D3_SHA256:
            raise SystemExit("assets/d3.min.js is not the vendored D3 v7.9.0 bundle (checksum mismatch)")
    js = json.dumps(round_floats(data), ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(ddir, "bench.js"), "w", encoding="utf-8") as fh:
        fh.write(f"SITE.put(\"bench\", {js});\n")
    with open(os.path.join(ddir, "records.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        # publish only the run label (run directory / timing / config), not the host's absolute path
        w.writerows(sorted(({**r, "result_dir": "/".join(r["result_dir"].split("/")[-3:])} for r in rows),
                           key=lambda r: (r["device"], r["utc"])))
    with open(os.path.join(ddir, "configs.json"), "w", encoding="utf-8") as fh:
        json.dump(round_floats(dict(generated=data["generated"], note="mean of the passes; s = steady seconds per simulated day",
                                    configs=data["configs"], breakdown=data["breakdown"], failures=data["failures"])), fh, indent=1)
    with open(os.path.join(HERE, "layout.html"), encoding="utf-8") as fh:
        layout = fh.read()
    for fname, label, title in PAGES:
        with open(os.path.join(HERE, "pages", fname), encoding="utf-8") as fh:
            body = fh.read()
        nav = "".join(f'<a href="{f}"{" aria-current=page" if f == fname else ""}>{html.escape(lab)}</a>' for f, lab, _ in PAGES)
        page = (layout.replace("{{TITLE}}", html.escape(title)).replace("{{NAV}}", nav).replace("{{PAGE}}", fname[:-5])
                .replace("{{VERSION}}", version).replace("{{GENERATED}}", data["generated"])
                .replace("{{REVISION}}", html.escape(data["revision"])).replace("{{BODY}}", body))
        with open(os.path.join(out, fname), "w", encoding="utf-8") as fh:
            fh.write(page)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(f"built {data['generated']} by benchmarking/site/build_site.py {data['revision']}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data"),
                    help="folder with <device>/records.csv and optional PUBLIC_NOTES.md (default: %(default)s)")
    ap.add_argument("--out", required=True, help="output folder (created; must be empty or an earlier build)")
    a = ap.parse_args(argv)
    rows, problems = read_records(a.data)
    B = Bench([r for r in rows if r["float_type"] == FLOAT_TYPE])
    other_float = Counter(f"{r['device']} {r['float_type'] or 'unknown'}" for r in rows if r["float_type"] != FLOAT_TYPE)
    for d in B.unknown:
        problems.append(f"device {d!r} has no fixed colour or description in build_site.py (shown in grey)")
    now = dt.datetime.now(dt.timezone.utc)
    data = payload(B, read_notes(a.data), now, git_revision(), problems, dict(sorted(other_float.items())))
    write_site(a.out, data, rows, now.strftime("%Y%m%d%H%M%S"))
    for p in problems:
        print("warning:", p, file=sys.stderr)
    print(f"{len(B.rows)} {FLOAT_TYPE} records ({len(rows)} in all), {len(B.cfg)} configurations, {len(B.failures)} failed runs, devices: "
          f"{', '.join(B.devices) or 'none'} -> {a.out}")


if __name__ == "__main__":
    main()
