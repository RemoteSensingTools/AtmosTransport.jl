#!/usr/bin/env python3
"""Write the CATRINE benchmark web page from a catrine_compare_vs_geoschem.py output.

Reads <compare-dir>/summary.json and the figures next to it, copies them to
--out, and writes --out/index.html with:
  * the period-mean agreement of every run with GEOS-Chem (R^2, OLS slope,
    bias, RMSE in ppm / ppt / 1e-21 mol/mol), per tracer and air-mass band;
    the best value of each row is bold;
  * the monthly bias above ~100 hPa (the stratospheric drift) for CO2 and SF6;
  * global burden ratios and the figures of the comparison.

  python3 scripts/diagnostics/catrine_benchmark_page.py \\
      --compare-dir /temp1/cfranken/catrine_protocol/compare_c90_merra2_interim_2026_10_08 \\
      --out ~/www/catrine/benchmark_merra2 \\
      --run 'merra2_old=MERRA-2, mass-weighted closure (2026-10-06)' \\
      --run 'hm_gchp_gcic=MERRA-2, GCHP-matched fluxes, from GCHP initial state'
"""
import argparse, html, json, os, re, shutil, subprocess

# key, name, unit, scale from mol/mol, decimals of bias and RMSE
TRACERS = [("co2", "CO₂", "ppm", 1e6, 3), ("fossil", "Fossil CO₂ (from 2021-12-01)", "ppm", 1e6, 3),
           ("sf6", "SF₆", "ppt", 1e12, 4), ("rn222", "Rn-222", "10⁻²¹ mol/mol", 1e21, 4)]
BANDS = [("column", "column"), ("sfc_910hPa", "0–10 % of air mass (surface–~910 hPa)"),
         ("910_400hPa", "10–60 % (~910–400 hPa)"), ("400_100hPa", "60–90 % (~400–100 hPa)"),
         ("above_100hPa", "90–100 % (above ~100 hPa)")]
FIGURES = [("timeseries_r2_bias.png", "R² and bias of the instantaneous fields, per band:"),
           ("monthly_mean_field_r2_bias.png", "The same for monthly mean fields:"),
           ("burdens.png", "Global burdens and their ratio to GEOS-Chem:")]
MAPS = ["map_column_co2.png", "map_column_fossil.png", "map_column_sf6.png", "map_column_rn222.png"]
ZONAL = ["zonal_co2.png", "zonal_fossil.png", "zonal_sf6.png", "zonal_rn222.png"]


def stat_cells(stats, runs, scale, digits):
    """One table row: R², slope, bias, RMSE for every run; best value per statistic in bold."""
    vals = {r: stats[r] for r in runs}
    best = {"r2": max(runs, key=lambda r: vals[r]["r2"]),
            "slope": min(runs, key=lambda r: abs(vals[r]["slope"] - 1)),
            "bias": min(runs, key=lambda r: abs(vals[r]["bias"])),
            "rmse": min(runs, key=lambda r: vals[r]["rmse"])}
    cells = []
    for r in runs:
        v = vals[r]
        for key, text in (("r2", f"{v['r2']:.3f}"), ("slope", f"{v['slope']:.3f}"),
                          ("bias", f"{v['bias'] * scale:+.{digits}f}"), ("rmse", f"{v['rmse'] * scale:.{digits}f}")):
            cells.append(f"<td><b>{text}</b></td>" if len(runs) > 1 and best[key] == r else f"<td>{text}</td>")
    return "".join(cells)


def script_revision():
    """Commit of the repository holding this script, with '+local changes' if the tree is dirty."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))          # repository root: scripts/diagnostics/..
    if not os.path.exists(os.path.join(root, ".git")):     # an exported tree; never ask a parent repository
        rev = open(os.path.join(root, "src", "REVISION")).read().strip() \
            if os.path.exists(os.path.join(root, "src", "REVISION")) else ""
        return rev[:8] if re.fullmatch(r"[0-9a-f]{40}", rev) else "unknown"
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "--short=8", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", here],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + (" + local changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-dir", required=True, help="output directory of catrine_compare_vs_geoschem.py")
    ap.add_argument("--out", required=True, help="web directory to write")
    ap.add_argument("--run", action="append", required=True, help="label=description, in table order")
    ap.add_argument("--title", default="CATRINE C90 benchmark: AtmosTransport vs GEOS-Chem (GCHP)")
    ap.add_argument("--intro", default="", help="HTML paragraph(s) after the title")
    ap.add_argument("--code", action="append", default=[],
                    help="label=code versions of a run (met preprocessing and transport), shown in a table")
    ap.add_argument("--reference", default="GEOS-Chem CATRINE C90 standard run (GCHP; files say GEOS-Chem_devel)",
                    help="code version of the reference run")
    ap.add_argument("--video", action="append", default=[],
                    help="file|caption of an animation already in --out (poster: <file>_last.png)")
    args = ap.parse_args()

    summary = os.path.join(args.compare_dir, "summary.json")
    if not os.path.exists(summary):
        ap.error(f"no summary.json in {args.compare_dir}")
    with open(summary) as fh:
        s = json.load(fh)
    for opt, sep, values in (("--run", "=", args.run), ("--code", "=", args.code), ("--video", "|", args.video)):
        for v in values:
            if sep not in v:
                ap.error(f"{opt} {v!r}: expected 'name{sep}text'")
    runs, names = [], {}
    for spec in args.run:
        label, desc = spec.split("=", 1)
        if label not in s["period_mean_fields"]:
            ap.error(f"--run {label!r} is not in {summary} (has {', '.join(s['period_mean_fields'])})")
        runs.append(label); names[label] = desc
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    have = set()                                          # figures present in the comparison
    for f in [f for f, _ in FIGURES] + MAPS + ZONAL:
        src = os.path.join(args.compare_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, "figures", f)); have.add(f)
        else:
            print("warning: missing figure", src)
    shutil.copy(os.path.join(args.compare_dir, "summary.json"), os.path.join(out, "summary.json"))

    t0, t1, n = s["times"]
    videos = ""
    if args.video:
        videos = "<h2>Animations (one frame per day, 00 UTC)</h2>" + "".join(
            f'<p>{html.escape(cap)}</p><video controls preload="metadata" src="{html.escape(f, quote=True)}" '
            f'poster="{html.escape(os.path.splitext(f)[0], quote=True)}_last.png"></video>'
            for f, cap in (v.split("|", 1) for v in args.video))
    pm = s["period_mean_fields"]
    h = [f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:auto;padding:1em;color:#222}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:3px 7px;text-align:right}}
td:nth-child(2),th{{text-align:left}}img,video{{max-width:100%}}h2{{margin-top:1.6em}}</style></head><body>
<h1>{html.escape(args.title)}</h1>
<p>{n} three-hourly instantaneous snapshots, {html.escape(str(t0))} to {html.escape(str(t1))}; period means over {html.escape(s['period'])}.
Reference: the GEOS-Chem CATRINE C90 standard run (GCHP, MERRA-2).</p>
{args.intro}
<ul>""" + "".join(f"<li><b>{html.escape(r)}</b>: {html.escape(names[r])}</li>" for r in runs) + f"""</ul>
<p>All fields are compared on the common C90 cube. Statistics are area weighted; bands are fractions of the
column dry-air mass counted from the surface, so they do not depend on the models' level sets. Bold: the best
value of each row (R² highest, slope closest to 1, |bias| and RMSE smallest).</p>
{videos}
<h2>Agreement over the period (mean fields)</h2>
<table><tr><th rowspan="2">Tracer</th><th rowspan="2">Band</th>"""]
    h.append("".join(f'<th colspan="4">{html.escape(r)}</th>' for r in runs) + "</tr><tr>")
    h.append("<th>R²</th><th>slope</th><th>bias</th><th>RMSE</th>" * len(runs) + "</tr>")
    for key, name, unit, scale, digits in TRACERS:
        for i, (band, bname) in enumerate(BANDS):
            first = f'<td rowspan="{len(BANDS)}"><b>{name}</b><br><small>{unit}</small></td>' if i == 0 else ""
            cells = stat_cells({r: pm[r][key][band] for r in runs}, runs, scale, digits)
            h.append(f"<tr>{first}<td>{bname}</td>{cells}</tr>")
    h.append("</table>")

    # Monthly bias above ~100 hPa: the stratospheric drift.
    months = sorted(s["monthly_mean_fields"])
    h.append("<h2>Bias above ~100 hPa by month (monthly mean fields)</h2><table><tr><th>month</th>")
    h.append("".join(f'<th colspan="2">{html.escape(r)}</th>' for r in runs) + "</tr><tr><th></th>")
    h.append("<th>CO₂ ppm</th><th>SF₆ ppt</th>" * len(runs) + "</tr>")
    for mo in months:
        row = s["monthly_mean_fields"][mo]
        cells = "".join(f"<td>{row[r]['co2']['above_100hPa']['bias'] * 1e6:+.3f}</td>"
                        f"<td>{row[r]['sf6']['above_100hPa']['bias'] * 1e12:+.4f}</td>" for r in runs)
        h.append(f"<tr><td>{mo}</td>{cells}</tr>")
    h.append("</table>")

    if args.code:
        codes = dict(c.split("=", 1) for c in args.code)
        h.append("<h2>Code versions</h2><table><tr><th>run</th><th>met preprocessing and transport code</th></tr>")
        h.append("".join(f"<tr><td>{html.escape(r)}</td><td>{html.escape(codes.get(r, 'not recorded'))}</td></tr>"
                         for r in runs))
        h.append(f"<tr><td>reference</td><td>{html.escape(args.reference)}</td></tr>")
        h.append(f"<tr><td>comparison</td><td>catrine_compare_vs_geoschem.py and catrine_benchmark_page.py at "
                 f"{html.escape(script_revision())}</td></tr></table>")

    ratio = s["final_burden_ratio_to_gc"]
    h.append(f"<p>Final global burden / GEOS-Chem at {html.escape(str(t1))}: " + "; ".join(
        f"{html.escape(r)}: CO₂ {ratio[r]['co2']:.4f}, fossil {ratio[r]['fossil']:.5f}, "
        f"SF₆ {ratio[r]['sf6']:.4f}, Rn-222 {ratio[r]['rn222']:.4f}" for r in runs) +
        ". Dry-air mass / GEOS-Chem <code>Met_AD</code>: " +
        ", ".join(f"{html.escape(r)} {s['air_mass_ratio_to_gc'][r]:.4f}" for r in runs) +
        "; it sets the burden ratios of the long-lived tracers.</p>")

    h.append("<h2>Agreement through time</h2>")
    for f, caption in FIGURES:
        if f in have:
            h.append(f'<p>{caption}</p><img src="figures/{f}">')
    h.append("<h2>Mean column maps</h2>" + "".join(f'<img src="figures/{f}">' for f in MAPS if f in have))
    h.append("<h2>Mean zonal means on dry pressure</h2>" + "".join(f'<img src="figures/{f}">' for f in ZONAL if f in have))
    h.append('<p><small>Generated by <code>scripts/diagnostics/catrine_compare_vs_geoschem.py</code> and '
             '<code>scripts/diagnostics/catrine_benchmark_page.py</code> (AtmosTransport). '
             'Raw statistics: <a href="summary.json">summary.json</a>.'
             '</small></p></body></html>')
    with open(os.path.join(out, "index.html"), "w") as fh:
        fh.write("\n".join(h))
    print("wrote", os.path.join(out, "index.html"))


if __name__ == "__main__":
    main()
