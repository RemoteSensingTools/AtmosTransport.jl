#!/usr/bin/env python3
"""Hovmoller and zonal growth-rate diagnostics for the C30 12-tracer TRENDY run
covering the full OCO-2 era (2014-09 to 2024-12).

Two figures:
  trendy_c30_xco2_hovmoller.png   NEE'-driven XCO2 anomaly per model in 40
                                  equal-area sin(lat) bands, with the observed
                                  OCO-2 anomaly on the same scale.
  trendy_c30_zonal_growth.png     monthly growth-rate anomaly per model on
                                  GRESO's ten 10-degree bands, against GRESO.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("clm", "jules_es", "orchidee")
NICE = {"clm": "CLM6.0", "jules_es": "JULES-ES", "orchidee": "ORCHIDEE"}
GRESO_BANDS_CSV = Path("/home/spandey/OCO_growth_rates/greso_minimal_noaa_portable_260624"
                       "/output_10sec_full_2025/greso_latitude_band_monthly_growth_rates.csv")
YEARS = (2015, 2024)
INK, INK2, GRID = "#1f2933", "#52606d", "#d9e2ec"
DIVERGING = "RdBu_r"


def monthly_mean(dates: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily series (time, ...) -> calendar-month means over YEARS."""
    keys = dates.astype("datetime64[M]")
    wanted = np.arange(f"{YEARS[0]}-01", f"{YEARS[1] + 1}-01", dtype="datetime64[M]")
    out = np.stack([x[keys == k].mean(axis=0) for k in wanted])
    return wanted, out


def backfit(x: np.ndarray, iters: int = 5) -> np.ndarray:
    """Remove a joint linear trend + mean seasonal cycle, NaN-tolerant.

    Fitting the two jointly (rather than trend-then-season) is required here:
    detrending first leaks the seasonal cycle into the trend whenever the
    seasonal amplitude dominates. Operates on the leading (month) axis.
    """
    n = x.shape[0]
    t = np.arange(n, dtype=float)
    t -= np.nanmean(t)
    season = np.zeros_like(x)
    trend = np.zeros_like(x)
    for _ in range(iters):
        resid = x - season
        good = np.isfinite(resid)
        num = np.nansum(np.where(good, resid * t[:, None], 0.0), axis=0)
        den = np.nansum(np.where(good, t[:, None] ** 2, 0.0), axis=0)
        slope = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        trend = t[:, None] * slope
        resid = x - trend
        month = np.arange(n) % 12
        season = np.zeros_like(x)
        for m in range(12):
            sel = month == m
            season[sel] = np.nanmean(resid[sel], axis=0)
    out = x - trend - season
    return out - np.nanmean(out, axis=0)


def smooth3(x: np.ndarray) -> np.ndarray:
    """Centred 3-month mean along the leading axis, NaN-tolerant.

    Month-to-month differencing of the tracer is far noisier than GRESO, which
    is a smoothed product; without this the two are not visually comparable.
    """
    pad = np.concatenate([x[:1], x, x[-1:]])
    stack = np.stack([pad[:-2], pad[1:-1], pad[2:]])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def read_greso_bands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [ln.split(",") for ln in GRESO_BANDS_CSV.read_text().splitlines() if ln]
    head = rows[0]
    di, ci, vi = (head.index(k) for k in
                  ("date", "latitude_band_center", "growth_rate_ppm_per_month"))
    centers = sorted({float(r[ci]) for r in rows[1:]})
    months = np.arange(f"{YEARS[0]}-01", f"{YEARS[1] + 1}-01", dtype="datetime64[M]")
    idx = {str(m): i for i, m in enumerate(months)}
    grid = np.full((len(months), len(centers)), np.nan)
    for r in rows[1:]:
        key = r[di][:7]
        if key in idx:
            grid[idx[key], centers.index(float(r[ci]))] = float(r[vi])
    return months, np.array(centers), grid


def hovmoller(ax, months, edges, field, vmax, cmap=DIVERGING):
    t = np.append(months, months[-1] + 1).astype("datetime64[D]").astype(float)
    return ax.pcolormesh(t, edges, field.T, cmap=cmap, vmin=-vmax, vmax=vmax,
                         shading="flat")


LAT_TICKS = (-90, -60, -30, -15, 0, 15, 30, 60, 90)


def lat_axis(ax):
    """Label the equal-area sin(lat) axis with real latitudes."""
    ax.set_yticks([np.sin(np.deg2rad(d)) for d in LAT_TICKS],
                  [f"{d}" for d in LAT_TICKS])
    ax.set_ylabel("latitude")


def shade_gaps(ax, months, edges, gap):
    """Grey out band-months OCO-2 never observed (polar night, 2017 outage)."""
    t = np.append(months, months[-1] + 1).astype("datetime64[D]").astype(float)
    ax.pcolormesh(t, edges, np.where(gap, 1.0, np.nan).T,
                  cmap=matplotlib.colors.ListedColormap(["#b9c2cc"]),
                  shading="flat", zorder=3)


def date_axis(ax, months):
    ticks = [m for m in months if int(str(m)[5:7]) == 1]
    ax.set_xticks([t.astype("datetime64[D]").astype(float) for t in ticks],
                  [str(t)[:4] for t in ticks])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots/sif_gpp_iav"))
    ap.add_argument("--no-fire", action="store_true",
                    help="ignore the GFED5 fire tracer even if it is present")
    ap.add_argument("--min-soundings", type=int, default=2000,
                    help="drop OCO-2 band-months below this count "
                         "(median coverage is ~65k for the equal-area bands)")
    args = ap.parse_args()

    z = np.load(args.scratch / "c30_trendy.npz", allow_pickle=True)
    tracers = list(z["tracers"])
    dates, edges = z["dates"], z["edges"]
    months, hov = monthly_mean(dates, z["hov"] * 1e6)          # ppm, (month, tracer, band)

    obs = np.load(args.scratch / "oco2_bands_2015_2024.npz", allow_pickle=True)
    # Drop band-months with too few soundings; otherwise the polar winter edge
    # is salt-and-pepper noise that dominates the colour scale.
    obs_eq = np.where(obs["eq_n"] >= args.min_soundings, obs["eq"], np.nan)
    obs_greso = np.where(obs["greso_n"] >= args.min_soundings, obs["greso"], np.nan)
    print(f"OCO-2 coverage after the {args.min_soundings}-sounding cut: "
          f"equal-area {np.isfinite(obs_eq).mean():.0%}, "
          f"GRESO bands {np.isfinite(obs_greso).mean():.0%}")

    raw_nee = {m: hov[:, tracers.index(f"co2_{m}_nee")] for m in MODELS}
    obs_anom = backfit(obs_eq)
    seen = np.isfinite(obs_anom)

    # The TRENDY NEE tracers carry Ra + Rh - GPP only; GFED5 fire is a separate
    # run that superimposes exactly, transport being linear in the tracer.
    fire_path = args.scratch / "c30_fire.npz"
    fire = None
    if fire_path.exists() and not args.no_fire:
        f = np.load(fire_path, allow_pickle=True)
        _, fire_eq = monthly_mean(f["dates"], f["hov_eq"] * 1e6)
        fire = fire_eq
        print(f"GFED5 fire tracer loaded, anomaly rms "
              f"{np.sqrt(np.mean(backfit(fire) ** 2)):.3f} ppm")

    def score(field):
        return np.corrcoef(field[seen], obs_anom[seen])[0, 1]

    print("XCO2 anomaly rms (ppm) and R with OCO-2, 40 equal-area bands")
    print(f"  {'model':9s} {'rms':>6s} {'R (NEE)':>9s} {'rms+fire':>9s} "
          f"{'R (NEE+fire)':>13s}")
    variants = {}
    for m in MODELS:
        plain = backfit(raw_nee[m])
        row = f"  {NICE[m]:9s} {np.sqrt(np.mean(plain ** 2)):6.3f} {score(plain):+9.2f}"
        variants[m] = plain
        if fire is not None:
            withfire = backfit(raw_nee[m] + fire)
            variants[m + "+fire"] = withfire
            row += (f" {np.sqrt(np.mean(withfire ** 2)):9.3f} "
                    f"{score(withfire):+13.2f}")
        print(row)
    print(f"  {'OCO-2':9s} {np.sqrt(np.nanmean(obs_anom ** 2)):6.3f}")

    plot_hovmoller(args.outdir, months, edges,
                   {m: variants[m] for m in MODELS}, obs_anom, "")
    if fire is not None:
        plot_hovmoller(args.outdir, months, edges,
                       {m: variants[m + "+fire"] for m in MODELS}, obs_anom,
                       "_fire", " NEE + GFED5 fire")
    if fire is not None:
        plot_fire_only(args.outdir, months, edges, fire, obs_anom, seen)
    zonal_growth(args.outdir, args.scratch, tracers, dates, obs_greso,
                 use_fire=fire is not None)


def plot_hovmoller(outdir, months, edges, nee, obs_anom, suffix,
                   label=" NEE tracer") -> None:
    # One colour scale for every panel, so model and observed amplitudes are
    # directly comparable rather than each panel being self-normalised.
    vmax = float(np.nanpercentile(np.abs(np.concatenate(
        [v.ravel() for v in nee.values()] + [obs_anom.ravel()])), 99))
    seen = np.isfinite(obs_anom)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, sharey=True)
    panels = [(NICE[m] + label, nee[m]) for m in MODELS]
    panels.append(("OCO-2 observed XCO2", obs_anom))
    for ax, (title, field) in zip(axes, panels):
        mesh = hovmoller(ax, months, edges, field, vmax)
        shade_gaps(ax, months, edges, ~seen)
        lat_axis(ax)
        rms = float(np.sqrt(np.nanmean(field ** 2)))
        # Scored only on cells OCO-2 actually saw, so the shaded gaps cannot
        # inflate or deflate the comparison.
        tag = "" if field is obs_anom else \
            f",  R with OCO-2 = {np.corrcoef(field[seen], obs_anom[seen])[0, 1]:+.2f}"
        ax.set_title(f"{title}   (rms {rms:.2f} ppm{tag})", fontsize=11, color=INK,
                     loc="left")
    fig.colorbar(mesh, ax=axes, pad=0.01, fraction=0.025,
                 label=f"ppm  (common scale +/-{vmax:.2f})")
    date_axis(axes[-1], months)
    axes[-1].set_xlabel("year")
    fig.suptitle("C30 TRENDY v14 S3 transported NEE vs observed XCO2 anomalies\n"
                 f"{YEARS[0]}-{YEARS[1]}, detrended + deseasonalised, equal-area bands "
                 "(tracers spun up from 2014-09; grey = no OCO-2 data)",
                 fontsize=12.5, color=INK)
    path = outdir / f"trendy_c30_xco2_hovmoller{suffix}.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove a per-band linear trend, keeping the seasonal cycle."""
    t = np.arange(x.shape[0], dtype=float)
    t -= t.mean()
    slope = (x * t[:, None]).sum(axis=0) / (t ** 2).sum()
    out = x - t[:, None] * slope
    return out - out.mean(axis=0)


def plot_fire_only(outdir, months, edges, fire, obs_anom, seen) -> None:
    """The GFED5 fire tracer with only the overall background removed.

    No deseasonalising and no per-band detrending: the top panel is the field
    minus its instantaneous global mean, so the seasonal march, the interannual
    events and the interhemispheric accumulation gradient all survive. The
    bottom panel shows exactly what was removed -- the well-mixed global
    background the tracer accumulates.
    """
    gmean = fire.mean(axis=1)                       # bands are equal-area
    dev = fire - gmean[:, None]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.1, 1]})

    vmax = float(np.percentile(np.abs(dev), 99.5))
    mesh = hovmoller(a1, months, edges, dev, vmax)
    lat_axis(a1)
    a1.set_title("fire XCO2 minus the global mean   "
                 f"(rms {np.sqrt(np.mean(dev ** 2)):.3f} ppm; seasonal cycle "
                 "and interannual events retained)", fontsize=11, color=INK,
                 loc="left")
    fig.colorbar(mesh, ax=a1, pad=0.01, fraction=0.03, label="ppm")

    tm = months.astype("datetime64[D]").astype(float)
    a2.plot(tm, gmean - gmean[0], color="#c2410c", lw=1.8)
    a2.set_title("the removed background: global-mean fire XCO2 above 2015-01",
                 fontsize=11, color=INK, loc="left")
    a2.set_ylabel("ppm")
    a2.set_xlabel("year")
    a2.grid(color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    pos = a2.get_position(); cb = 0.03 / (1 + 0.03)   # keep x-span aligned with a1
    date_axis(a2, months)

    fig.suptitle("GFED5 fire CO2 transported alone, C30, 2015-2024 "
                 "(equal-area bands; not deseasonalised, not detrended)",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = outdir / "gfed5_fire_c30_hovmoller.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def zonal_growth(outdir, scratch, tracers, dates, obs_greso, use_fire=False) -> None:
    zb = np.load(scratch / "c30_trendy_gresobands.npz", allow_pickle=True)
    months, band_hov = monthly_mean(dates, zb["hov"] * 1e6)
    gm, centers, greso = read_greso_bands()
    assert list(gm) == list(months), "GRESO months do not match the model months"

    fire_gr = 0.0
    if use_fire:
        f = np.load(scratch / "c30_fire.npz", allow_pickle=True)
        _, fire_gr = monthly_mean(f["dates"], f["hov_gr"] * 1e6)
    growth = {m: np.diff(band_hov[:, tracers.index(f"co2_{m}_nee")] + fire_gr, axis=0)
              for m in MODELS}
    obs_growth = np.diff(obs_greso, axis=0)
    gmid = months[1:]
    anom = {m: smooth3(backfit(growth[m])) for m in MODELS}
    anom["GRESO"] = smooth3(backfit(greso[1:]))
    anom["OCO-2 d(XCO2)"] = smooth3(backfit(obs_growth))

    kind = "NEE + GFED5 fire" if use_fire else "NEE"
    print(f"\nZonal growth-rate anomaly ({kind}), GRESO 10-degree bands, "
          "corr with GRESO'")
    header = "  band   " + "".join(f"{NICE[m]:>10s}" for m in MODELS) + f"{'dXCO2':>10s}"
    print(header)
    for b, c in enumerate(centers):
        cells = [f"{np.corrcoef(anom[m][:, b], anom['GRESO'][:, b])[0, 1]:+10.2f}"
                 for m in MODELS]
        ok = np.isfinite(anom["OCO-2 d(XCO2)"][:, b]) & np.isfinite(anom["GRESO"][:, b])
        cells.append(f"{np.corrcoef(anom['OCO-2 d(XCO2)'][ok, b], anom['GRESO'][ok, b])[0, 1]:+10.2f}")
        print(f"  {c:+5.0f}   " + "".join(cells))
    for m in MODELS:
        allb = np.corrcoef(anom[m].ravel(), anom["GRESO"].ravel())[0, 1]
        print(f"  all bands pooled {NICE[m]:10s} {allb:+.2f}")

    edges = zb["edges"]
    keys = [NICE[m] for m in MODELS] + ["GRESO"]
    fields = [anom[m] for m in MODELS] + [anom["GRESO"]]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([f.ravel() for f in fields])), 98))
    fig, axes = plt.subplots(len(keys), 1, figsize=(13, 11), sharex=True, sharey=True)
    for ax, key, field in zip(axes, keys, fields):
        mesh = hovmoller(ax, gmid, edges, field, vmax)
        ax.set_ylabel("latitude")
        rms = float(np.sqrt(np.nanmean(field ** 2)))
        tag = "" if key == "GRESO" else \
            f",  R with GRESO = {np.corrcoef(field.ravel(), anom['GRESO'].ravel())[0, 1]:+.2f}"
        ax.set_title(f"{key}   (rms {rms:.3f} ppm/month{tag})", fontsize=11, color=INK,
                     loc="left")
    fig.colorbar(mesh, ax=axes, pad=0.01, fraction=0.025,
                 label=f"ppm/month  (common scale +/-{vmax:.2f})")
    date_axis(axes[-1], gmid)
    axes[-1].set_xlabel("year")
    fig.suptitle(f"Zonal monthly growth-rate anomaly: transported TRENDY {kind} vs GRESO, "
                 f"{YEARS[0]}-{YEARS[1]} (detrended + deseasonalised, 3-month mean)",
                 fontsize=13.5, color=INK)
    path = outdir / ("trendy_c30_zonal_growth_fire.png" if use_fire
                     else "trendy_c30_zonal_growth.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
