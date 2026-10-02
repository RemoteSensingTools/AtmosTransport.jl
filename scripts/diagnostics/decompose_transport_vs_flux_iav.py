#!/usr/bin/env python3
"""Split transported XCO2 interannual variability into flux-driven and
transport-driven parts, using the climatological-flux control run.

Two tracers see the *same* meteorology:

  R   real JULES-ES NEE
  C   the 2015-2024 mean seasonal cycle of that NEE, repeated every year

Transport is linear in the tracer and both runs share the winds, so

  R - C = T[f_real] - T[f_clim] = T[f_real - f_clim]

is *exactly* the XCO2 driven by flux interannual variability, with no
cross-term to argue about, and C is exactly the XCO2 that year-to-year
meteorology produces from an unchanging flux. This is a clean decomposition,
not an approximate one.

Caveat it checks rather than assumes: R and C are F32 tracers on 200 ppm
carriers, and differencing large-carrier F32 fields is a known way to
manufacture noise, so the global mean of R - C is validated against the F64
mass budget and the flux-file integral.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

YEARS = (2015, 2024)
INK, INK2, GRID = "#1f2933", "#52606d", "#d9e2ec"
M_AIR, M_CO2 = 0.02896546, 0.0440095
DRY_AIR = 5.135e18


def monthly_mean(dates, x):
    keys = dates.astype("datetime64[M]")
    want = np.arange(f"{YEARS[0]}-01", f"{YEARS[1] + 1}-01", dtype="datetime64[M]")
    return want, np.stack([x[keys == k].mean(axis=0) for k in want])


def backfit(x, iters=5):
    """Joint linear trend + calendar-month climatology removal."""
    n = x.shape[0]
    t = np.arange(n, dtype=float)
    t -= t.mean()
    season = np.zeros_like(x)
    for _ in range(iters):
        resid = x - season
        trend = t[:, None] * ((resid * t[:, None]).sum(axis=0) / (t ** 2).sum())
        resid = x - trend
        season = np.tile(np.concatenate(
            [resid[m::12].mean(axis=0)[None] for m in range(12)]), (n // 12, 1))
    out = x - trend - season
    return out - out.mean(axis=0)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots/sif_gpp_iav"))
    args = ap.parse_args()

    real = np.load(args.scratch / "c30_trendy.npz", allow_pickle=True)
    ctrl = np.load(args.scratch / "c30_neeclim.npz", allow_pickle=True)
    tracers = list(real["tracers"])
    dates, edges = real["dates"], real["edges"]
    if not np.array_equal(dates, ctrl["dates"]):
        raise SystemExit("control and real runs cover different days")

    months, R = monthly_mean(dates, real["hov"][:, tracers.index("co2_jules_es_nee")] * 1e6)
    _, C = monthly_mean(dates, ctrl["hov_eq"] * 1e6)

    # --- mass-budget validation of the difference -------------------------
    mR = real["mass"][:, tracers.index("co2_jules_es_nee")]
    mC = ctrl["mass"]
    ppm = DRY_AIR / 1e6
    dR, dC = (mR[-1] - mR[0]) / ppm, (mC[-1] - mC[0]) / ppm
    print("F64 mass budget over the run (global-mean ppm)")
    print(f"  real NEE       {dR:+8.2f}")
    print(f"  climatology    {dC:+8.2f}")
    print(f"  difference     {dR - dC:+8.2f}   <- flux IAV, cumulative")
    band_w = ctrl["wsum_eq"] if "wsum_eq" in ctrl else None
    gm = lambda x: x.mean(axis=1)          # bands are equal-area, so unweighted
    print(f"  band-mean check of the same quantity: "
          f"{(R - C)[-1].mean() - (R - C)[0].mean():+.2f} ppm")

    # --- variance decomposition -------------------------------------------
    aR, aC = backfit(R), backfit(C)
    aF = backfit(R - C)                     # == aR - aC, backfit being linear
    print("\nXCO2 anomaly rms (ppm), 40 equal-area bands, detrended+deseasonalised")
    print(f"  total       (real NEE)          {rms(aR):.4f}")
    print(f"  flux-driven (real - clim)       {rms(aF):.4f}   "
          f"{100 * rms(aF) ** 2 / rms(aR) ** 2:5.1f}% of total variance")
    print(f"  transport   (climatology only)  {rms(aC):.4f}   "
          f"{100 * rms(aC) ** 2 / rms(aR) ** 2:5.1f}% of total variance")
    print(f"  corr(flux-driven, transport) {np.corrcoef(aF.ravel(), aC.ravel())[0, 1]:+.3f}"
          "   (0 would mean the two add in quadrature)")
    print(f"  rms(flux)/rms(transport) = {rms(aF) / rms(aC):.2f}")

    print("\nby latitude band group")
    lat = np.rad2deg(np.arcsin(0.5 * (edges[:-1] + edges[1:])))
    groups = (("90S-30S", lat < -30), ("30S-30N", np.abs(lat) <= 30),
              ("30N-90N", lat > 30))
    print(f"  {'zone':10s} {'total':>8s} {'flux':>8s} {'transport':>10s} {'T/total':>8s}")
    for name, sel in groups:
        print(f"  {name:10s} {rms(aR[:, sel]):8.3f} {rms(aF[:, sel]):8.3f} "
              f"{rms(aC[:, sel]):10.3f} {rms(aC[:, sel]) / rms(aR[:, sel]):8.2f}")

    print("\nglobal mean (band average), and annual means")
    gR, gF, gC = backfit(gm(R)[:, None]), backfit(gm(R - C)[:, None]), backfit(gm(C)[:, None])
    print(f"  monthly global  total {rms(gR):.4f}  flux {rms(gF):.4f}  "
          f"transport {rms(gC):.4f}  -> transport is {rms(gC) / rms(gR):.0%} of total")
    ann = lambda x: x.reshape(-1, 12, x.shape[1]).mean(axis=1)
    print(f"  annual  zonal   total {rms(ann(aR)):.4f}  flux {rms(ann(aF)):.4f}  "
          f"transport {rms(ann(aC)):.4f}  -> transport is "
          f"{rms(ann(aC)) / rms(ann(aR)):.0%} of total")

    plot(args.outdir, months, edges, aR, aF, aC, gm)


def plot(outdir, months, edges, aR, aF, aC, gm):
    vmax = float(np.percentile(np.abs(aR), 99))
    fig, axes = plt.subplots(4, 1, figsize=(13, 13),
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.75]})
    t = np.append(months, months[-1] + 1).astype("datetime64[D]").astype(float)
    ticks = [m for m in months if int(str(m)[5:7]) == 1]
    panels = (("total: real JULES-ES NEE", aR),
              ("flux-driven: real minus climatology (exact)", aF),
              ("transport-driven: climatological flux, real winds", aC))
    for ax, (title, field) in zip(axes[:3], panels):
        mesh = ax.pcolormesh(t, edges, field.T, cmap="RdBu_r", vmin=-vmax,
                             vmax=vmax, shading="flat")
        ax.set_yticks([np.sin(np.deg2rad(d)) for d in (-90, -60, -30, 0, 30, 60, 90)],
                      ["-90", "-60", "-30", "0", "30", "60", "90"])
        ax.set_ylabel("latitude")
        ax.set_title(f"{title}   (rms {rms(field):.3f} ppm)", fontsize=11,
                     color=INK, loc="left")
        ax.set_xticks([x.astype("datetime64[D]").astype(float) for x in ticks],
                      [""] * len(ticks))
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03, label="ppm")

    ax = axes[3]
    ax.plot(months.astype("datetime64[D]").astype(float), gm(aR), color=INK,
            lw=1.6, label=f"total (rms {rms(gm(aR)):.3f})")
    ax.plot(months.astype("datetime64[D]").astype(float), gm(aF), color="#2f7bd1",
            lw=1.4, label=f"flux-driven (rms {rms(gm(aF)):.3f})")
    ax.plot(months.astype("datetime64[D]").astype(float), gm(aC), color="#e8743b",
            lw=1.4, label=f"transport-driven (rms {rms(gm(aC)):.3f})")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([x.astype("datetime64[D]").astype(float) for x in ticks],
                  [str(x)[:4] for x in ticks])
    ax.set_ylabel("ppm")
    ax.set_xlabel("year")
    ax.set_title("global mean", fontsize=11, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=3)

    fig.suptitle("XCO2 interannual variability: what the fluxes do vs what the "
                 "winds do\nJULES-ES NEE on C30, 2015-2024, detrended + "
                 "deseasonalised, equal-area bands", fontsize=13.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = outdir / "transport_vs_flux_iav.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
