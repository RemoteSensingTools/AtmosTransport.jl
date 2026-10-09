#!/usr/bin/env python3
"""Decompose monthly global NEE' of every TRENDY v14 S3 model into NPP, Rh and
fire terms, and score each model against the observed OCO-2 growth-rate anomaly.

NEE = Ra + Rh + fFire - GPP (positive = source). Stacks are the deseasonalised
anomalies of -(GPP-Ra)' ("NPP term"), Rh' and fFire'; GFED5 fire is overlaid as
a single observed reference identical in every panel.

Inputs (produced by the companion extraction scripts in the session scratchpad):
  trendy_all_models_global.json  {model: {gpp, ra, rh}}  Pg C/yr, monthly
  trendy_all_models_fire.json    {model: {ffire, ...}}   Pg C/yr, monthly
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

YEARS = (2015, 2024)
PGC_PER_PPM = 2.124
TRANSPORTED = {"JULES-ES", "ORCHIDEE", "CLM"}
GRESO_CSV = Path("/home/spandey/OCO_growth_rates/greso_minimal_noaa_portable_260624"
                 "/output_10sec_full_2025/greso_monthly_growth_rates.csv")
GFED_CSV = Path("/home/spandey/gfed5_monthly_global_fire_carbon_1997_2025"
                "/gfed5_global_monthly_fire_carbon_anomaly_1997_2025.csv")
INK, INK2, GRID = "#1f2933", "#52606d", "#d9e2ec"
C_NPP, C_RH, C_FIRE = "#4c3f9e", "#1fa97a", "#e8743b"


def months() -> list[tuple[int, int]]:
    return [(y, m) for y in range(YEARS[0], YEARS[1] + 1) for m in range(1, 13)]


def deseason(x: np.ndarray) -> np.ndarray:
    """Remove the calendar-month mean. Requires complete calendar years."""
    z = x.reshape(-1, 12)
    return (z - z.mean(axis=0)).ravel()


def read_csv(path: Path, date_col: str, value_col: str) -> dict[tuple[int, int], float]:
    rows = [ln.split(",") for ln in path.read_text().splitlines()
            if ln and not ln.startswith("#")]
    head = rows[0]
    di, vi = head.index(date_col), head.index(value_col)
    out = {}
    for r in rows[1:]:
        y, m, _ = r[di].split("-")
        out[(int(y), int(m))] = float(r[vi])
    return out


def series(table: dict[tuple[int, int], float], scale: float) -> np.ndarray:
    return np.array([table[k] * scale for k in months()])


def best_lag_corr(model: np.ndarray, obs: np.ndarray) -> tuple[float, int]:
    """Correlation with the observation allowed to lag the flux by 0 or 1 month."""
    best = (0.0, 0)
    for lag in (0, 1):
        a, b = (model[:-lag], obs[lag:]) if lag else (model, obs)
        r = float(np.corrcoef(a, b)[0, 1])
        if abs(r) > abs(best[0]):
            best = (r, lag)
    return best


def covariance_share(term: np.ndarray, total: np.ndarray) -> float:
    return 100.0 * float(np.dot(term, total) / np.dot(total, total))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True,
                    help="directory holding the two extraction JSON files")
    ap.add_argument("--outdir", type=Path, default=Path("plots/sif_gpp_iav"))
    args = ap.parse_args()

    comp = json.loads((args.scratch / "trendy_all_models_global.json").read_text())
    fire = json.loads((args.scratch / "trendy_all_models_fire.json").read_text())

    greso = deseason(series(read_csv(GRESO_CSV, "date", "growth_rate_ppm_per_month"),
                            PGC_PER_PPM * 12.0))
    gfed = deseason(series(read_csv(GFED_CSV, "date", "global_fire_carbon_PgC_month"),
                           12.0))

    stats = {}
    for model, g in comp.items():
        gpp, ra, rh = (np.array(g[k]) for k in ("gpp", "ra", "rh"))
        npp_term = deseason(-(gpp - ra))
        rh_term = deseason(rh)
        f = fire.get(model)
        fire_term = deseason(np.array(f["ffire"])) if f else None
        has_fire = fire_term is not None and np.ptp(fire_term) > 1e-6
        nee = npp_term + rh_term + (fire_term if has_fire else 0.0)
        nee_nofire = npp_term + rh_term
        r, lag = best_lag_corr(nee, greso)
        r_nf, _ = best_lag_corr(nee_nofire, greso)
        r_gfed, _ = best_lag_corr(nee_nofire + gfed, greso)
        stats[model] = dict(
            npp=npp_term, rh=rh_term, fire=fire_term if has_fire else np.zeros_like(nee),
            has_fire=has_fire, nee=nee, r=r, lag=lag, r_nofire=r_nf, r_gfed=r_gfed,
            gpp_mean=float(gpp.mean()),
            fire_mean=float(np.mean(f["ffire"])) if f else float("nan"),
            share_npp=covariance_share(npp_term, nee),
            share_rh=covariance_share(rh_term, nee),
            share_fire=covariance_share(fire_term, nee) if has_fire else 0.0,
            rms=float(np.sqrt(np.mean(nee ** 2))),
        )

    order = sorted(stats, key=lambda m: -abs(stats[m]["r"]))
    ens = np.mean([stats[m]["nee"] for m in order], axis=0)
    r_ens, lag_ens = best_lag_corr(ens, greso)

    print(f"{'model':12s} {'GPP':>6s} {'fire':>6s} {'NPP%':>6s} {'Rh%':>6s} {'fire%':>6s} "
          f"{'rms':>5s} {'r':>6s} {'lag':>3s} {'r-nofire':>8s} {'r+GFED':>7s}")
    for m in order:
        s = stats[m]
        print(f"{m:12s} {s['gpp_mean']:6.1f} {s['fire_mean']:6.2f} {s['share_npp']:6.1f} "
              f"{s['share_rh']:6.1f} {s['share_fire']:6.1f} {s['rms']:5.2f} {s['r']:+6.2f} "
              f"{s['lag']:3d} {s['r_nofire']:+8.2f} {s['r_gfed']:+7.2f}")
    print(f"{'ENSEMBLE':12s} {'':6s} {'':6s} {'':6s} {'':6s} {'':6s} "
          f"{np.sqrt(np.mean(ens ** 2)):5.2f} {r_ens:+6.2f} {lag_ens:3d}")
    print(f"GRESO rms {np.sqrt(np.mean(greso ** 2)):.2f} Pg C/yr; "
          f"GFED fire anomaly rms {np.sqrt(np.mean(gfed ** 2)):.2f} Pg C/yr; "
          f"corr(GFED, GRESO) {np.corrcoef(gfed, greso)[0, 1]:+.2f}")

    fire_swap_test(args.outdir, order, stats, greso, gfed)
    inter_model_agreement(args.outdir, order, stats)

    t = np.array([y + (m - 0.5) / 12 for y, m in months()])
    plot_stacks(args.outdir, order, stats, greso, gfed, t)
    plot_summary(args.outdir, order, stats, gfed, greso)


def align(model: np.ndarray, obs: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    return (model[:-lag], obs[lag:]) if lag else (model, obs)


def r2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def block_bootstrap_delta(a_own: np.ndarray, a_swap: np.ndarray, obs: np.ndarray,
                          block: int = 12, draws: int = 4000,
                          seed: int = 0) -> tuple[float, float, float]:
    """90% CI and P(improvement) for r2(swap) - r2(own) under a moving-block
    bootstrap, which keeps the ~annual autocorrelation of these series intact."""
    rng = np.random.default_rng(seed)
    n = len(obs)
    starts = np.arange(n - block + 1)
    nblk = n // block
    deltas = np.empty(draws)
    for d in range(draws):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, nblk)])
        deltas[d] = r2(a_swap[idx], obs[idx]) - r2(a_own[idx], obs[idx])
    lo, hi = np.percentile(deltas, [5, 95])
    return float(lo), float(hi), float(np.mean(deltas > 0))


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """corr(x, y) with z partialled out."""
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y), (x, z), (y, z)))
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def fire_swap_test(outdir: Path, order, stats, greso, gfed) -> None:
    """Does replacing each model's own fire flux with GFED5 raise R^2 vs GRESO?

    The observation lag is frozen at the value chosen for the own-fire NEE', so
    the three variants are scored on identical footing.
    """
    print("\nFire-swap test: R^2(NEE', GRESO'), observation lag frozen per model")
    print(f"{'model':12s} {'lag':>3s} {'R2 none':>7s} {'R2 own':>7s} {'R2 GFED':>7s} "
          f"{'dR2':>6s} {'90% CI':>15s} {'P(+)':>5s} {'beta':>5s} {'pcorr':>6s} {'a:solo->joint':>13s}")
    rows = []
    for m in order:
        s = stats[m]
        lag = s["lag"]
        nee_none = s["npp"] + s["rh"]
        own, obs = align(s["nee"], greso, lag)
        swap, _ = align(nee_none + gfed, greso, lag)
        none, _ = align(nee_none, greso, lag)
        g_al, _ = align(gfed, greso, lag)
        d_lo, d_hi, p_pos = block_bootstrap_delta(own, swap, obs)
        # least-squares weight GRESO' wants on the GFED term, given the fire-free NEE'
        design = np.column_stack([none, g_al, np.ones_like(none)])
        fit = np.linalg.lstsq(design, obs, rcond=None)[0]
        alpha, beta = float(fit[0]), float(fit[1])
        alpha_solo = float(np.dot(none, obs) / np.dot(none, none))
        pc = partial_corr(g_al, obs, own)
        row = dict(model=m, lag=lag, r2_none=r2(none, obs), r2_own=r2(own, obs),
                   r2_swap=r2(swap, obs), lo=d_lo, hi=d_hi, p=p_pos, beta=beta, pcorr=pc,
                   alpha=alpha, alpha_solo=alpha_solo, has_fire=s["has_fire"])
        rows.append(row)
        print(f"{m:12s} {lag:3d} {row['r2_none']:7.3f} {row['r2_own']:7.3f} "
              f"{row['r2_swap']:7.3f} {row['r2_swap'] - row['r2_own']:+6.3f} "
              f"[{d_lo:+.3f},{d_hi:+.3f}] {p_pos:5.2f} {beta:5.2f} {pc:+6.2f} "
              f"{alpha_solo:5.2f}->{alpha:5.2f}")
    n_up = sum(r["r2_swap"] > r["r2_own"] for r in rows)
    n_sig = sum(r["lo"] > 0 for r in rows)
    with_fire = [r for r in rows if r["has_fire"]]
    print(f"improves in {n_up}/{len(rows)} models ({n_sig} with a 90% CI clear of zero); "
          f"among the {len(with_fire)} that actually have fFire, "
          f"{sum(r['r2_swap'] > r['r2_own'] for r in with_fire)} improve")
    print(f"median dR2 {np.median([r['r2_swap'] - r['r2_own'] for r in rows]):+.3f}, "
          f"median GFED weight beta {np.median([r['beta'] for r in rows]):.2f} "
          f"(1.0 = GFED enters at face value)")
    plot_fire_swap(outdir, rows)


def running_mean(x: np.ndarray, window: int = 12) -> np.ndarray:
    k = np.ones(window) / window
    return np.convolve(x, k, mode="valid")


def agreement(members: list[np.ndarray]) -> dict:
    """Pairwise correlation and shared-variance statistics for one component."""
    m = np.array(members)
    r = np.corrcoef(m)
    off = r[np.triu_indices(len(m), k=1)]
    ens = m.mean(axis=0)
    return dict(matrix=r, cov=np.cov(m), mean_r=float(off.mean()),
                median_r=float(np.median(off)),
                frac_negative=float(np.mean(off < 0)),
                coherence=float(ens.var() / m.var(axis=1).mean()),
                rms=np.sqrt((m ** 2).mean(axis=1)))


def inter_model_agreement(outdir: Path, order, stats) -> None:
    """Do the models agree with each other more on the NPP term or on Rh?"""
    comps = {"-(GPP-Ra)'": [stats[m]["npp"] for m in order],
             "Rh'": [stats[m]["rh"] for m in order],
             "NEE'": [stats[m]["nee"] for m in order]}
    fire_models = [m for m in order if stats[m]["has_fire"]]
    comps["fFire'"] = [stats[m]["fire"] for m in fire_models]

    print("\nInter-model agreement (deseasonalised monthly anomalies, 2015-2024)")
    print(f"{'component':12s} {'n':>3s} {'mean r':>7s} {'median r':>8s} {'neg pairs':>9s} "
          f"{'shared var':>10s} {'rms lo/med/hi':>20s} {'mean r (12-mo)':>14s}")
    results = {}
    for name, members in comps.items():
        a = agreement(members)
        a12 = agreement([running_mean(x) for x in members])
        results[name] = a
        lo, med, hi = np.percentile(a["rms"], [0, 50, 100])
        print(f"{name:12s} {len(members):3d} {a['mean_r']:+7.2f} {a['median_r']:+8.2f} "
              f"{a['frac_negative']:8.0%} {a['coherence']:10.2f} "
              f"{lo:6.2f}/{med:.2f}/{hi:5.2f} {a12['mean_r']:+14.2f}")
    print("shared var = var(ensemble mean) / mean member variance; 1.0 = identical "
          "members, 1/n = independent members")

    labels = {k: (fire_models if k == "fFire'" else order) for k in results}
    plot_agreement(outdir, labels, results)
    write_agreement_html(outdir, labels, results)


def plot_agreement(outdir: Path, labels, results) -> None:
    keys = ["-(GPP-Ra)'", "Rh'", "fFire'", "NEE'"]
    covmax = max(np.abs(results[k]["cov"]).max() for k in keys)
    fig, axes = plt.subplots(2, 4, figsize=(20, 11.5))
    for col, key in enumerate(keys):
        a, names = results[key], labels[key]
        for row, (mat, vmax, cmap) in enumerate(((a["matrix"], 1.0, "RdBu_r"),
                                                 (a["cov"], covmax, "PuOr_r"))):
            ax = axes[row, col]
            im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap=cmap)
            ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6.5)
            ax.set_yticks(range(len(names)), names, fontsize=6.5)
            if row == 0:
                ax.set_title(f"{key}\nmean pairwise r = {a['mean_r']:+.2f},  "
                             f"shared var = {a['coherence']:.2f}", fontsize=11, color=INK)
            else:
                ax.set_title(f"{key} covariance", fontsize=11, color=INK)
            if col == 3:
                fig.colorbar(im, ax=axes[row, :], fraction=0.012, pad=0.01,
                             label="correlation" if row == 0 else "(Pg C/yr)$^2$")
    fig.suptitle("Inter-model agreement on monthly global anomalies, TRENDY v14 S3 "
                 f"{YEARS[0]}-{YEARS[1]} (models ordered by skill against GRESO); "
                 "top = correlation, bottom = covariance on a shared scale",
                 fontsize=13.5, color=INK)
    path = outdir / "trendy_intermodel_agreement.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"wrote {path}")


HTML_TEMPLATE = """<!DOCTYPE html>
<meta charset="utf-8"><title>TRENDY inter-model agreement</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
 body {font: 14px/1.45 -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       margin: 24px; color: #1f2933;}
 h1 {font-size: 19px; margin: 0 0 4px;}
 p.sub {color: #52606d; margin: 0 0 16px;}
 label {margin-right: 18px;} select {font-size: 14px; padding: 2px 6px;}
 #stat {margin: 12px 0 0; color: #52606d;}
</style>
<h1>TRENDY v14 S3 &mdash; inter-model agreement on monthly global anomalies</h1>
<p class="sub">__PERIOD__, deseasonalised. Models ordered by skill against the
GRESO growth-rate anomaly. Hover a cell for the pair.</p>
<label>component
 <select id="comp"></select></label>
<label>metric
 <select id="metric">
  <option value="matrix">correlation</option>
  <option value="cov">covariance (Pg C/yr)&sup2;</option>
 </select></label>
<div id="plot" style="width: 900px; height: 820px;"></div>
<p id="stat"></p>
<script>
const DATA = __DATA__;
const compSel = document.getElementById('comp');
Object.keys(DATA).forEach(k => compSel.add(new Option(k, k)));
function draw() {
  const key = compSel.value, metric = document.getElementById('metric').value;
  const d = DATA[key], z = d[metric], n = d.labels.length;
  const corr = metric === 'matrix';
  const lim = corr ? 1 : Math.max(...z.flat().map(Math.abs));
  const text = z.map((row, i) => row.map((v, j) =>
    `${d.labels[i]} &times; ${d.labels[j]}<br>r = ${d.matrix[i][j].toFixed(2)}` +
    `<br>cov = ${d.cov[i][j].toFixed(2)} (Pg C/yr)&sup2;`));
  Plotly.react('plot', [{
    z: z, x: d.labels, y: d.labels, type: 'heatmap',
    colorscale: corr ? 'RdBu' : 'PuOr', reversescale: true,
    zmin: -lim, zmax: lim, text: text, hoverinfo: 'text',
    colorbar: {title: corr ? 'r' : '(Pg C/yr)&sup2;', thickness: 14}
  }], {
    margin: {l: 110, r: 20, t: 30, b: 110},
    xaxis: {tickangle: -90, automargin: true, scaleanchor: 'y'},
    yaxis: {automargin: true, autorange: 'reversed'},
    title: {text: key, font: {size: 15}}
  }, {displaylogo: false, responsive: true});
  document.getElementById('stat').textContent =
    `${n} models  |  mean pairwise r ${d.mean_r.toFixed(2)}  |  ` +
    `median ${d.median_r.toFixed(2)}  |  ${(100 * d.frac_negative).toFixed(0)}% of ` +
    `pairs anti-correlated  |  shared variance ${d.coherence.toFixed(2)} ` +
    `(1 = identical members, 1/n = independent)`;
}
compSel.onchange = document.getElementById('metric').onchange = draw;
draw();
</script>
"""


def write_agreement_html(outdir: Path, labels, results) -> None:
    payload = {key: dict(labels=list(labels[key]),
                         matrix=np.round(a["matrix"], 4).tolist(),
                         cov=np.round(a["cov"], 4).tolist(),
                         mean_r=a["mean_r"], median_r=a["median_r"],
                         frac_negative=a["frac_negative"], coherence=a["coherence"])
               for key, a in results.items()}
    html = (HTML_TEMPLATE.replace("__PERIOD__", f"{YEARS[0]}-{YEARS[1]}")
            .replace("__DATA__", json.dumps(payload)))
    path = outdir / "trendy_intermodel_agreement.html"
    path.write_text(html)
    print(f"wrote {path}")


def plot_fire_swap(outdir: Path, rows) -> None:
    rows = sorted(rows, key=lambda r: r["r2_swap"])
    y = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8.5),
                                 gridspec_kw={"width_ratios": [1.55, 1]})
    for i, r in enumerate(rows):
        a1.plot([r["r2_own"], r["r2_swap"]], [i, i], color=GRID, lw=3, zorder=1,
                solid_capstyle="round")
    a1.scatter([r["r2_none"] for r in rows], y, s=26, color="none", edgecolor=INK2,
               lw=1.0, zorder=2, label="no fire term")
    a1.scatter([r["r2_own"] for r in rows], y, s=42, color=C_NPP, zorder=3,
               label="model's own fFire")
    a1.scatter([r["r2_swap"] for r in rows], y, s=42, color=C_FIRE, zorder=3,
               label="GFED5 fire swapped in")
    a1.set_yticks(y, [r["model"] + ("" if r["has_fire"] else " *") for r in rows],
                  fontsize=10)
    a1.set_xlabel("R$^2$ against the GRESO growth-rate anomaly")
    a1.set_title("Swapping GFED5 fire into each model's NEE'\n"
                 "(* = model has no fire term of its own, so GFED5 is added)",
                 fontsize=11, loc="left", color=INK)
    a1.legend(fontsize=9, loc="lower right")
    a1.grid(axis="x", color=GRID, lw=0.5)
    a1.set_axisbelow(True)

    d = np.array([r["r2_swap"] - r["r2_own"] for r in rows])
    err = np.array([[r["r2_swap"] - r["r2_own"] - r["lo"] for r in rows],
                    [r["hi"] - (r["r2_swap"] - r["r2_own"]) for r in rows]])
    colors = [C_FIRE if r["lo"] > 0 else GRID for r in rows]
    a2.barh(y, d, color=colors, xerr=err, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_yticks(y, [""] * len(rows))
    a2.set_xlabel("$\\Delta$R$^2$ (GFED5 swap - own fire)")
    a2.set_title("Change in explained variance\n"
                 "(bars: 90% moving-block bootstrap; orange = CI clear of zero)",
                 fontsize=11, loc="left", color=INK)
    a2.grid(axis="x", color=GRID, lw=0.5)
    a2.set_axisbelow(True)

    fig.suptitle("Does observed (GFED5) fire beat modelled fire in explaining the "
                 f"OCO-2 growth rate? TRENDY v14 S3, {YEARS[0]}-{YEARS[1]}",
                 fontsize=13.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = outdir / "trendy_fire_swap_r2.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def plot_stacks(outdir: Path, order, stats, greso, gfed, t) -> None:
    ncol, nrow = 4, int(np.ceil(len(order) / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    span = max(np.abs(np.concatenate([stats[m]["nee"] for m in order] + [greso])).max(), 1.0)
    w = 1 / 12
    for ax, model in zip(axes, order):
        s = stats[model]
        pos = np.zeros_like(t)
        neg = np.zeros_like(t)
        for term, color, label in ((s["npp"], C_NPP, "-(GPP-Ra)'"),
                                   (s["rh"], C_RH, "Rh'"),
                                   (s["fire"], C_FIRE, "fFire'")):
            base = np.where(term >= 0, pos, neg)
            ax.bar(t, term, bottom=base, width=w, color=color, linewidth=0, label=label)
            pos = pos + np.clip(term, 0, None)
            neg = neg + np.clip(term, None, 0)
        ax.plot(t, greso, color=INK, lw=1.4, label="GRESO growth-rate anomaly")
        ax.plot(t, gfed, color="#7b1fa2", lw=1.1, ls="--", label="GFED5 fire anomaly")
        ax.axhline(0, color=INK2, lw=0.6)
        tag = "" if s["has_fire"] else "  (no fFire)"
        ax.set_title(f"{model}{tag}   r = {s['r']:+.2f}", fontsize=10, color=INK)
        ax.set_ylim(-1.15 * span, 1.15 * span)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[len(order):]:
        ax.axis("off")
    spare = axes[len(order)] if len(order) < len(axes) else axes[0]
    spare.legend(handles, labels, fontsize=10, loc="center", frameon=False)
    for col in range(ncol):
        bottom = [i for i in range(col, len(order), ncol)]
        axes[bottom[-1]].set_xlabel("year")
        axes[bottom[-1]].tick_params(labelbottom=True)
    for i in range(0, len(order), ncol):
        axes[i].set_ylabel("Pg C/yr")
    fig.suptitle("TRENDY v14 S3: monthly global NEE' decomposition with fire, "
                 f"{YEARS[0]}-{YEARS[1]} (deseasonalised), sorted by agreement with GRESO",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = outdir / "trendy_allmodels_stackedbars_fire.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def plot_summary(outdir: Path, order, stats, gfed, greso) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 8.5))
    y = np.arange(len(order))[::-1]
    colors = ["#e8743b" if m in TRANSPORTED else "#2f7bd1" for m in order]
    a1.barh(y, [abs(stats[m]["r"]) for m in order], color=colors)
    a1.barh(y, [abs(stats[m]["r_nofire"]) for m in order], height=0.28,
            color="none", edgecolor=INK, lw=1.0, label="without fFire")
    a1.set_yticks(y, order, fontsize=10)
    a1.set_xlabel("|corr(NEE', GRESO')|, best of lag 0/1")
    a1.set_title("Agreement with the observed growth-rate anomaly\n"
                 "(orange = the 3 transported models; outline = fire term removed)",
                 fontsize=11, loc="left", color=INK)
    a1.legend(fontsize=9, loc="lower right")
    a1.grid(axis="x", color=GRID, lw=0.5)
    a1.set_axisbelow(True)

    left = np.zeros(len(order))
    for key, color, label in (("share_npp", C_NPP, "NPP-driven share"),
                              ("share_rh", C_RH, "Rh share"),
                              ("share_fire", C_FIRE, "fire share")):
        vals = np.array([stats[m][key] for m in order])
        a2.barh(y, vals, left=left, color=color, label=label)
        left = left + vals
    a2.axvline(100, color=INK2, ls=":", lw=1)
    a2.set_yticks(y, [""] * len(order))
    a2.set_xlabel("covariance share of NEE' variance (%)")
    a2.set_title("What drives NEE': -(GPP-Ra)' vs Rh' vs fFire'",
                 fontsize=11, loc="left", color=INK)
    a2.legend(fontsize=9, loc="lower right")
    a2.grid(axis="x", color=GRID, lw=0.5)
    a2.set_axisbelow(True)

    fig.suptitle(f"All {len(order)} TRENDY v14 S3 models - monthly global NEE' vs GRESO, "
                 f"{YEARS[0]}-{YEARS[1]} (deseasonalised)", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = outdir / "trendy_allmodels_summary_fire.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
