#!/usr/bin/env python3
"""Build a multi-year daily SIF-derived GPP CO2 flux driver on the GEOS C90 grid.

Converts the gap-filled daily TROPOMI S5P-PAL SIF product to daily-mean
atmospheric CO2 uptake using the *fixed* 2021 SIF-to-GPP calibration
(month x latitude-band robust regressions from the 2021 pilot, see
docs/memos/SIF_DERIVED_GPP_2021_ATBD.md).

Holding the calibration fixed in time is deliberate: every year is converted
with identical coefficients, so all interannual variability in the resulting
GPP comes from observed SIF rather than from the FLUXCOM-X training target.
FLUXCOM-X only covers 2014-2021, so a re-fit product could not span the SIF
record without a discontinuity in 2022.

Two variables are written:

  GPP_CO2_FLUX       full uptake flux (always <= 0)
  GPP_CO2_FLUX_ANOM  flux minus its smoothed day-of-year climatology
  GPP_CO2_FLUX_SEAS  the climatology itself, with each cell's annual mean
                     removed: an identical seasonal cycle every year, zero
                     annual mean everywhere. A tracer driven by this has no
                     flux interannual variability by construction, so any
                     interannual signal it develops comes purely from transport
                     (meteorology). Removing the annual mean costs nothing
                     physically -- it only strips the secular drawdown ramp --
                     and keeps the tracer small enough for good Float32
                     precision.
  RECO_FLAT_CO2_FLUX each cell's annual-mean GPP uptake, sign-flipped and
                     constant in time: a flat respiration proxy that exactly
                     balances the mean sink.
  GPP_CO2_FLUX_NET   GPP_CO2_FLUX + RECO_FLAT_CO2_FLUX, i.e. net flux with a
                     flat Reco. Annually balanced per cell, so a tracer driven
                     by it carries the full seasonal cycle and the full
                     interannual variability but no secular drawdown. By
                     linearity its response equals xco2_anom + xco2_clim; the
                     direct run exists for numerical precision and as an
                     end-to-end linearity check.

Because advection and diffusion are linear in the tracer, transporting
GPP_CO2_FLUX_ANOM yields exactly the XCO2 response to the departure from the
mean seasonal cycle, with no post-hoc detrending and no multi-hundred-ppm
drawdown. Note that this departure still contains the multi-year GPP trend, and
that 2018 lies outside the climatology fit window; see the memo section on the
trend-versus-IAV split before interpreting it as pure interannual variability.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import netCDF4 as nc
import numpy as np

R_EARTH = 6_371_000.0
EPOCH = dt.datetime(1970, 1, 1)
LAT_EDGES = np.array([-90, -60, -40, -20, 0, 20, 40, 60, 90])
# TROPOMI -> OCO-2 QF0 harmonization on the length-of-day-corrected scale
# (analysis_summary.json: tropomi_to_oco2_daylength_corrected).
HARM_A, HARM_B = 0.04077697321225716, 0.8778927542020224
# gC -> kgCO2 and per-day -> per-second. 44/12 matches the 2021 pilot exactly.
GC_TO_KGCO2 = (44.0 / 12.0) * 1e-3
SEC_PER_DAY = 86400.0

DEF_SIF = ("/kiwi-data/Data/satellite/TROPOMI/TROPOMI_SIF_S5P-PAL/regridded/interpolated/"
           "TROPOMI_sif_20180501_20251231_C90_daily_gt-2SIF743lt5_SIF743ERRORlt10_filled.nc")
DEF_NPZ = "/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/fluxcom_x_c90_2021.npz"
DEF_SUM = "/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/analysis_summary.json"


def c90_cell_area(corner_lats, corner_lons):
    def unit(la, lo):
        la, lo = np.deg2rad(la), np.deg2rad(lo)
        return np.stack((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), axis=-1)
    p00 = unit(corner_lats[:, :-1, :-1], corner_lons[:, :-1, :-1])
    p10 = unit(corner_lats[:, :-1, 1:], corner_lons[:, :-1, 1:])
    p11 = unit(corner_lats[:, 1:, 1:], corner_lons[:, 1:, 1:])
    p01 = unit(corner_lats[:, 1:, :-1], corner_lons[:, 1:, :-1])

    def tri(a, b, c):
        num = np.abs(np.sum(a * np.cross(b, c), axis=-1))
        den = 1 + np.sum(a * b, axis=-1) + np.sum(b * c, axis=-1) + np.sum(c * a, axis=-1)
        return 2 * np.arctan2(num, den)
    return (tri(p00, p10, p11) + tri(p00, p11, p01)) * R_EARTH ** 2


def coef_fields(lats_flat, summary_path):
    """(12, ncell) intercept/slope from the frozen month x lat-band calibration."""
    models = json.load(open(summary_path))["gpp_models"]
    glob = (models["global"]["intercept"], models["global"]["slope"])
    groups = {k: (v["intercept"], v["slope"]) for k, v in models["groups"].items()}
    band = np.digitize(lats_flat, LAT_EDGES[1:-1])
    b0 = np.empty((12, lats_flat.size))
    b1 = np.empty((12, lats_flat.size))
    n_fallback = 0
    for m in range(1, 13):
        for bi in range(len(LAT_EDGES) - 1):
            key = f"m{m:02d}_lat{LAT_EDGES[bi]:+03.0f}_{LAT_EDGES[bi+1]:+03.0f}"
            if key not in groups:
                n_fallback += 1
            i0, i1 = groups.get(key, glob)
            sel = band == bi
            b0[m - 1, sel] = i0
            b1[m - 1, sel] = i1
    if n_fallback:
        print(f"[warn] {n_fallback} calibration groups missing; used global fit")
    return b0, b1


def circular_smooth(a, window):
    """Circular running mean along axis 0 (length 366 day-of-year climatology)."""
    if window <= 1:
        return a
    n = a.shape[0]
    half = window // 2
    pad = np.concatenate([a[n - half:], a, a[:window - half - 1]], axis=0)
    out = np.empty_like(a)
    # centred running mean along axis 0 via cumulative sums
    cs = np.cumsum(np.vstack([np.zeros((1,) + a.shape[1:]), pad]), axis=0)
    for i in range(n):
        out[i] = (cs[i + window] - cs[i]) / window
    del cs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sif", default=DEF_SIF)
    ap.add_argument("--npz", default=DEF_NPZ, help="FLUXCOM-X C90 npz (vegetated land fraction)")
    ap.add_argument("--summary", default=DEF_SUM, help="2021 pilot analysis_summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clim-years", default="2019-2025",
                    help="inclusive year range used for the day-of-year climatology")
    ap.add_argument("--clim-smooth", type=int, default=15,
                    help="circular running-mean window (days) applied to the climatology")
    args = ap.parse_args()

    src = nc.Dataset(args.sif)
    lats = np.asarray(src.variables["lats"][:], float)
    lons = np.asarray(src.variables["lons"][:], float)
    shape = lats.shape                                   # (6, 90, 90)
    area = c90_cell_area(np.asarray(src.variables["corner_lats"][:], float),
                         np.asarray(src.variables["corner_lons"][:], float)).ravel()
    tdays = np.asarray(src.variables["time"][:], float)
    dates = [EPOCH + dt.timedelta(days=float(x)) for x in tdays]
    nt = len(dates)
    ncell = area.size

    sphere_err = abs(area.sum() / (4 * np.pi * R_EARTH ** 2) - 1.0)
    assert sphere_err < 1e-6, f"C90 areas do not close on the sphere: {sphere_err:.3e}"

    lf = np.load(args.npz)["land_fraction"].astype(np.float64).ravel()
    assert lf.size == ncell, f"land_fraction size {lf.size} != {ncell}"
    b0, b1 = coef_fields(lats.ravel(), args.summary)

    months = np.array([d.month for d in dates])
    # kg CO2 m-2 s-1 per *total* cell area; negative = uptake.
    flux = np.empty((nt, ncell), np.float32)
    gpp_pg = np.zeros(nt)
    cell_w = lf * area
    step = 366
    sif_var = src.variables["sif_743_corr_filled"]
    sif_fills = [float(sif_var.getncattr(k)) for k in ("_FillValue", "missing_value")
                 if k in sif_var.ncattrs() and np.isfinite(float(sif_var.getncattr(k)))]
    n_masked_land = 0
    for s in range(0, nt, step):
        e = min(nt, s + step)
        raw = sif_var[s:e]
        # Honour the mask explicitly: np.asarray() would drop it, and a *finite*
        # fill value (e.g. the netCDF default 9.97e36) would then survive
        # nan_to_num and produce absurd GPP with no assert tripping.
        if np.ma.isMaskedArray(raw):
            raw = raw.filled(np.nan)
        sif = np.asarray(raw, np.float64).reshape(e - s, -1)
        for fv in sif_fills:
            sif[sif == fv] = np.nan
        # SIF outside the retrieval range cannot be physical.
        sif[(sif < -5.0) | (sif > 10.0)] = np.nan
        n_masked_land = max(n_masked_land,
                            int(np.isnan(sif[:, lf > 0.05]).any(axis=0).sum()))
        sif_c = HARM_A + HARM_B * sif
        gpp = b0[months[s:e] - 1, :] + b1[months[s:e] - 1, :] * sif_c
        np.maximum(gpp, 0.0, out=gpp)
        gpp = np.nan_to_num(gpp, nan=0.0, posinf=0.0, neginf=0.0)
        gpp_pg[s:e] = (gpp * cell_w[None, :]).sum(axis=1) / 1e15
        flux[s:e] = (-gpp * lf[None, :] * GC_TO_KGCO2 / SEC_PER_DAY).astype(np.float32)
        del raw, sif, sif_c, gpp
    src.close()

    land_sel = lf > 0.05
    frac_cells = n_masked_land / max(1, int(land_sel.sum()))
    print(f"[gaps] {n_masked_land} of {int(land_sel.sum())} cells with lf>0.05 have "
          f"no usable SIF on at least one day ({100 * frac_cells:.2f}% of land cells); "
          "these are set to zero GPP. The mask is time-invariant, so it "
          "contributes no interannual variability.")

    # ---- day-of-year climatology on (month, day); Feb 29 folds onto Feb 28 ----
    y0, y1 = (int(v) for v in args.clim_years.split("-"))
    md = np.array([(d.month, d.day) for d in dates])
    md_key = np.array([(m * 100 + dd) if not (m == 2 and dd == 29) else 228
                       for m, dd in md])
    uniq = np.unique(md_key)
    idx_of = {k: i for i, k in enumerate(uniq)}
    years = np.array([d.year for d in dates])
    use = (years >= y0) & (years <= y1)
    print(f"[clim] {y0}-{y1}: {use.sum()} days, {len(uniq)} day-of-year slots")

    clim = np.zeros((len(uniq), ncell), np.float64)
    cnt = np.zeros(len(uniq))
    for i in np.where(use)[0]:
        j = idx_of[md_key[i]]
        clim[j] += flux[i]
        cnt[j] += 1
    assert cnt.min() > 0, "empty day-of-year slot in climatology"
    # Feb 28 carries 9 samples (Feb 29 of 2020 and 2024 fold onto it) while every
    # other slot carries 7. The unsmoothed climatology is exactly mass-neutral,
    # but circular_smooth preserves only the *unweighted* slot sum, so smoothing
    # moves ~0.01 Pg C between the weight-9 slot and its weight-7 neighbours.
    # That is the entire source of the small non-zero record-total reported below
    # for the fit years; it is not float32 rounding.
    clim /= cnt[:, None]
    print(f"[clim] samples per slot: min {int(cnt.min())} max {int(cnt.max())}")
    if args.clim_smooth > 1:
        clim = circular_smooth(clim, args.clim_smooth)
        print(f"[clim] circular running mean applied, window={args.clim_smooth} d")

    anom = np.empty_like(flux)
    for i in range(nt):
        anom[i] = flux[i] - clim[idx_of[md_key[i]]].astype(np.float32)

    # Identical-every-year seasonal forcing with zero annual mean per cell.
    # Weight each day-of-year slot equally (not by sample count) so that the
    # mean removed is the mean of the seasonal cycle itself.
    clim_annual_mean = clim.mean(axis=0)
    seas_by_slot = (clim - clim_annual_mean[None, :]).astype(np.float32)
    seas = np.empty_like(flux)
    for i in range(nt):
        seas[i] = seas_by_slot[idx_of[md_key[i]]]

    # Flat-Reco proxy and the annually balanced net flux. reco = -M is a
    # constant-in-time emission; net = F - M. Written as full time series so
    # the cs_native loader path is identical for every variable.
    reco_cell = (-clim_annual_mean).astype(np.float32)          # >= 0
    net = np.empty_like(flux)
    for i in range(nt):
        net[i] = flux[i] + reco_cell

    # ---- diagnostics -------------------------------------------------------
    # `flux` is already per *total* cell area, so global integrals use `area`.
    ppm_per_pgc = 1.0 / 2.124   # 1 ppm CO2 ~ 2.124 Pg C of dry-air burden

    def carbon_pgc(f):
        """Carbon removed from the atmosphere by flux block `f`, in Pg C."""
        return -float((np.asarray(f, np.float64) * area[None, :]).sum()) \
            * SEC_PER_DAY * (12.0 / 44.0) / 1e12

    print("\n Positive anomaly = carbon removed in EXCESS of climatology (strong")
    print(" uptake); its XCO2 effect has the OPPOSITE sign (XCO2 falls).")
    print("\n year   GPP (PgC)  flux-integral (PgC)   anom C removed (PgC)   as ppm of burden")
    for Y in sorted(set(years)):
        m = years == Y
        f_pg = carbon_pgc(flux[m])
        a_pg = carbon_pgc(anom[m])
        print(f" {Y}  {gpp_pg[m].sum():9.3f}  {f_pg:17.3f}  {a_pg:+21.3f}  "
              f"{a_pg * ppm_per_pgc:+16.4f}")
    tot = gpp_pg.sum()
    print(f"\n total GPP over record: {tot:.2f} PgC "
          f"(~{tot * ppm_per_pgc:.1f} ppm of drawdown for the full-flux tracer)")
    print(f" flux-integral cross-check: {carbon_pgc(flux):.2f} PgC "
          "(must match total GPP)")
    anom_pg = carbon_pgc(anom)
    print(f" record-total anomaly carbon: {anom_pg:+.3f} PgC "
          f"({anom_pg * ppm_per_pgc:+.4f} ppm) -- should be near zero")
    seas_pg_per_year = [carbon_pgc(seas[years == Y]) for Y in sorted(set(years))
                        if (years == Y).sum() >= 365]
    print(f"\n seasonal-climatology tracer forcing (GPP_CO2_FLUX_SEAS):")
    print(f"   per-complete-year carbon: "
          f"{np.array2string(np.array(seas_pg_per_year), precision=4)} PgC")
    print(f"   max |annual-mean flux| over cells: "
          f"{np.abs(seas.mean(axis=0)).max():.3e} kg m-2 s-1 "
          "(should be ~0: zero annual mean everywhere)")
    print(f"   seasonal amplitude (global p1..p99 of daily flux): "
          f"{np.percentile(seas, 1):.3e} .. {np.percentile(seas, 99):.3e}")
    assert not np.isnan(flux).any(), "NaN in full flux"
    assert not np.isnan(anom).any(), "NaN in anomaly flux"
    assert not np.isnan(seas).any(), "NaN in seasonal flux"
    assert not np.isnan(net).any(), "NaN in net flux"
    assert reco_cell.min() >= 0.0, "flat Reco must be non-negative"
    assert flux.max() <= 0.0, "full flux must be non-positive"
    reco_pg = float((reco_cell.astype(np.float64) * area).sum()) \
        * 86400.0 * 365.25 * (12.0 / 44.0) / 1e12
    print(f"\n flat-Reco emission: {reco_pg:.2f} Pg C/yr "
          "(must equal the climatology-mean GPP)")
    net_by_year = [carbon_pgc(net[years == Y]) for Y in sorted(set(years))
                   if (years == Y).sum() >= 365]
    print(f" net-flux per complete year (PgC, + = net uptake): "
          f"{np.array2string(np.array(net_by_year), precision=3)}")

    # ---- write -------------------------------------------------------------
    t0 = dates[0]
    hours = np.array([(d - t0).total_seconds() / 3600.0 for d in dates], float)
    nf, ny, nx = shape
    with nc.Dataset(args.out, "w", format="NETCDF4") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("nf", nf)
        ds.createDimension("Ydim", ny)
        ds.createDimension("Xdim", nx)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = f"hours since {t0.date().isoformat()} 00:00:00 UTC"
        tv.calendar = "proleptic_gregorian"
        tv.long_name = "Time (UTC), start of daily interval"
        tv[:] = hours
        ds.createVariable("lons", "f8", ("nf", "Ydim", "Xdim"))[:] = lons
        ds.createVariable("lats", "f8", ("nf", "Ydim", "Xdim"))[:] = lats
        ds.createVariable("land_fraction", "f4", ("nf", "Ydim", "Xdim"))[:] = \
            lf.reshape(shape).astype(np.float32)
        ds.createVariable("cell_area", "f8", ("nf", "Ydim", "Xdim"))[:] = area.reshape(shape)
        dims = ("time", "nf", "Ydim", "Xdim")
        for name, data, ln in (
            ("GPP_CO2_FLUX", flux,
             "daily-mean SIF-derived GPP atmospheric CO2 uptake per total grid-cell area"),
            ("GPP_CO2_FLUX_ANOM", anom,
             "GPP CO2 flux minus its smoothed day-of-year climatology"),
            ("GPP_CO2_FLUX_SEAS", seas,
             "day-of-year GPP climatology with each cell's annual mean removed; "
             "identical every year, zero annual mean (transport-only control)"),
            ("GPP_CO2_FLUX_NET", net,
             "net CO2 flux: GPP uptake plus a flat respiration equal to each "
             "cell's climatological annual-mean uptake; annually balanced"),
            ("RECO_FLAT_CO2_FLUX", np.broadcast_to(reco_cell, flux.shape),
             "flat respiration proxy: minus each cell's climatological "
             "annual-mean GPP flux; constant in time (stored as a full series "
             "because the cs_native loader requires a time axis)"),
        ):
            v = ds.createVariable(name, "f4", dims, zlib=True, complevel=1, shuffle=True,
                                  fill_value=np.float32(-999.0))
            v.units = "kg CO2 m-2 s-1"
            v.long_name = ln
            v.positive = "to_atmosphere"
            v[:] = data.reshape((nt,) + shape)
        ds.Conventions = "CF-1.8"
        ds.title = "Daily SIF-derived GPP CO2 flux on GEOS C90, fixed 2021 calibration"
        ds.sign_convention = "negative values are atmospheric CO2 uptake"
        ds.sif_source = args.sif
        ds.sif_variable = "sif_743_corr_filled"
        ds.sif_basis = "product-provided length-of-day corrected SIF at 740 nm"
        ds.harmonization = (f"SIF_OCO2scale = {HARM_A:.6f} + {HARM_B:.6f} * SIF_TROPOMI")
        ds.calibration = ("frozen 2021 month x latitude-band robust regressions from "
                          f"{args.summary}; GPP = max(0, b0 + b1 * SIF_OCO2scale)")
        ds.calibration_note = ("calibration is time-invariant so that all interannual "
                               "variability originates from observed SIF")
        ds.vegetated_land_fraction_source = args.npz
        ds.climatology_years = args.clim_years
        ds.climatology_smoothing_days = args.clim_smooth
        ds.annual_gpp_PgC = json.dumps(
            {int(Y): round(float(gpp_pg[years == Y].sum()), 4) for Y in sorted(set(years))})
        ds.history = f"created {dt.datetime.now(dt.timezone.utc).isoformat()} by " \
                     "scripts/preprocessing/build_sif_gpp_multiyear_c90_flux.py"
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
