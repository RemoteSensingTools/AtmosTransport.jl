#!/usr/bin/env python3
"""Concatenate and analyze the SIF-GPP interannual-variability XCO2 experiment.

Reads the daily C90 column-mean snapshots written by
config/runs/sif_gpp_iav_c90_2018_2025.toml, removes each tracer's carrier
offset, and produces:

  * a single XCO2 time-series NetCDF (ppm, both tracers),
  * global/hemispheric/zonal anomaly diagnostics,
  * a linearity check: (full - anom) must be the response to the climatological
    flux alone, i.e. it must carry essentially no interannual variability.

Sign convention: positive XCO2 anomaly means anomalously *weak* GPP uptake.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re

import netCDF4 as nc
import numpy as np

CARRIER_PPM = {"co2_gpp_anom": 100.0, "co2_gpp_full": 900.0}
DATE_RE = re.compile(r"(\d{8})\.nc$")


def collect(outdir):
    """Dated snapshots, sorted. Files still being written are skipped."""
    files = sorted(f for f in glob.glob(os.path.join(outdir, "*.nc")) if DATE_RE.search(f))
    if not files:
        raise SystemExit(f"no dated snapshots in {outdir}")
    dates, keep, bad = [], [], []
    for f in files:
        try:                                  # a snapshot mid-write raises here
            with nc.Dataset(f) as d:
                if not any(k.endswith("_column_mean") for k in d.variables):
                    raise OSError("no column-mean variable")
        except OSError as exc:
            bad.append((os.path.basename(f), str(exc)[:60]))
            continue
        s = DATE_RE.search(f).group(1)
        dates.append(dt.datetime(int(s[:4]), int(s[4:6]), int(s[6:8])))
        keep.append(f)
    if bad:
        print(f"[warn] skipped {len(bad)} unreadable/incomplete snapshot(s), "
              f"e.g. {bad[:2]}")
    order = np.argsort(dates)
    return [keep[i] for i in order], [dates[i] for i in order]


def read_field(var):
    """Read a netCDF variable honouring its mask and fill value.

    netCDF4 returns a MaskedArray; np.asarray() would silently drop the mask and
    let _FillValue (1e15 in these snapshots) into the sums as ~1e21 ppm.
    """
    a = var[:]
    if np.ma.isMaskedArray(a):
        a = a.filled(np.nan)
    a = np.asarray(a, np.float64)
    for attr in ("_FillValue", "missing_value"):
        if attr in var.ncattrs():
            fv = float(var.getncattr(attr))
            if np.isfinite(fv):
                a[a == fv] = np.nan
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/temp1/cfranken/sif_gpp_iav/output/sif_gpp_iav_c90")
    ap.add_argument("--flux", default="/temp1/cfranken/sif_gpp_iav/fluxes/"
                                     "sif_gpp_daily_co2flux_c90_2018_2025.nc")
    ap.add_argument("--out", default="/temp1/cfranken/sif_gpp_iav/"
                                     "sif_gpp_iav_c90_xco2_timeseries.nc")
    args = ap.parse_args()

    files, dates = collect(args.outdir)
    nt = len(files)
    print(f"[read] {nt} daily snapshots: {dates[0].date()} .. {dates[-1].date()}")

    # A silently skipped day would bias every annual mean and break the 1:1 map
    # onto the flux time axis used by the mass check below.
    gaps = [(dates[i].date(), dates[i + 1].date()) for i in range(nt - 1)
            if (dates[i + 1] - dates[i]).days != 1]
    span = (dates[-1] - dates[0]).days + 1
    if gaps:
        print(f"[warn] {len(gaps)} gap(s) in the daily sequence "
              f"({span - nt} day(s) missing), e.g. {gaps[:3]}")
    else:
        print(f"[check] daily sequence is complete and contiguous ({span} days)")

    fx = nc.Dataset(args.flux)
    area = np.asarray(fx.variables["cell_area"][:], np.float64)
    lats = np.asarray(fx.variables["lats"][:], np.float64)
    lons = np.asarray(fx.variables["lons"][:], np.float64)
    lf = np.asarray(fx.variables["land_fraction"][:], np.float64)
    fx.close()

    d0 = nc.Dataset(files[0])
    tracers = [t for t in CARRIER_PPM if f"{t}_column_mean" in d0.variables]
    shape = np.asarray(d0.variables[f"{tracers[0]}_column_mean"][:]).shape[-3:]
    d0.close()
    print(f"[read] tracers: {tracers}  cell shape {shape}")
    assert area.shape == shape, f"grid mismatch {area.shape} vs {shape}"
    if not tracers:
        raise SystemExit("no recognised tracer column-mean fields in the snapshots")
    # Ydim == Xdim == 90, so a shape check cannot catch a transpose or a panel
    # permutation. Compare the coordinates themselves.
    with nc.Dataset(files[0]) as _d:
        if "lats" in _d.variables:
            out_lats = np.asarray(_d.variables["lats"][:], np.float64)
            dl = np.nanmax(np.abs(out_lats - lats))
            assert dl < 1e-6, f"output/flux grid latitudes differ by up to {dl:.3e} deg"
            print(f"[check] output grid matches flux grid (max |dlat| = {dl:.1e} deg)")

    xco2 = {t: np.empty((nt,) + shape, np.float32) for t in tracers}
    missing = []
    for i, f in enumerate(files):
        d = nc.Dataset(f)
        for t in tracers:
            v = d.variables.get(f"{t}_column_mean")
            if v is None:
                missing.append((f, t))
                xco2[t][i] = np.nan
                continue
            a = read_field(v)
            if a.ndim == 4:          # (time, nf, Ydim, Xdim) -> take last frame
                a = a[-1]
            xco2[t][i] = (a * 1e6 - CARRIER_PPM[t]).astype(np.float32)
        d.close()
    if missing:
        print(f"[warn] {len(missing)} missing tracer fields, e.g. {missing[:3]}")

    w = area / area.sum()

    def gmean(a):
        """Area-weighted global mean per time step, NaN-safe."""
        a = a.astype(np.float64)
        good = np.isfinite(a)
        num = np.einsum("tfyx,fyx->t", np.where(good, a, 0.0), w)
        den = np.einsum("tfyx,fyx->t", good.astype(np.float64), w)
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

    for t in tracers:
        nbad = int(np.isnan(xco2[t]).sum())
        if nbad:
            print(f"[warn] {t}: {nbad} non-finite cell-days "
                  f"({100 * nbad / xco2[t].size:.4f}%); excluded from means")
    years = np.array([d.year for d in dates])
    ndays = {Y: int((years == Y).sum()) for Y in sorted(set(years))}

    print("\n annual-mean global XCO2 (ppm, carrier removed)")
    print("  years with <350 days are marked * (partial coverage)")
    gm = {t: gmean(xco2[t]) for t in tracers}
    print("  year  days " + "".join(f"{t:>18s}" for t in tracers))
    for Y in sorted(set(years)):
        m = years == Y
        flag = " " if ndays[Y] >= 350 else "*"
        print(f"  {Y}{flag} {ndays[Y]:4d} "
              + "".join(f"{np.nanmean(gm[t][m]):18.4f}" for t in tracers))

    if len(tracers) == 2:
        # Linearity check: full - anom is the climatological-flux response, which
        # must contain no interannual variability beyond its own trend.
        clim_resp = gm["co2_gpp_full"] - gm["co2_gpp_anom"]
        print("\n[linearity] (full - anom) = response to climatological flux")
        print("   annual means:", " ".join(
            f"{Y}:{clim_resp[years == Y].mean():.2f}" for Y in sorted(set(years))))
        fy = [Y for Y in sorted(set(years)) if ndays[Y] >= 350]
        consecutive = len(fy) > 2 and fy == list(range(fy[0], fy[-1] + 1))
        if consecutive:
            d1 = np.diff([np.nanmean(clim_resp[years == Y]) for Y in fy])
            print(f"   complete-year drawdown steps {fy[0]}-{fy[-1]} (ppm): "
                  f"{np.array2string(d1, precision=2)}")
            if abs(d1.mean()) > 1e-9:
                print(f"   std of steps / mean step = "
                      f"{d1.std() / abs(d1.mean()):.4f} (small => near-linear drawdown)")
        else:
            print("   (need >2 consecutive complete years for the step diagnostic)")

    # ---- mass-conservation cross-check against the flux driver -------------
    # The transported global mean must equal the well-mixed integral of the
    # anomaly flux, because transport conserves tracer mass. This catches a
    # misaligned flux time axis, a dropped forcing window, or a leak.
    FLUXVAR = {"co2_gpp_anom": "GPP_CO2_FLUX_ANOM", "co2_gpp_full": "GPP_CO2_FLUX"}
    with nc.Dataset(args.flux) as fd:
        ftimes = np.asarray(fd.variables["time"][:], float)
        funits = fd.variables["time"].units
        f0 = dt.datetime.fromisoformat(funits.split("since")[1].strip().split()[0])
        fdates = [f0 + dt.timedelta(hours=float(h)) for h in ftimes]
        fidx = {d: i for i, d in enumerate(fdates)}
        sel = [fidx[d] for d in dates if d in fidx]
        if len(sel) != nt or sel != sorted(sel):
            print("\n[mass check] skipped: snapshot dates do not map 1:1 onto the "
                  "flux time axis")
        else:
            print("\n[mass check] transported global mean vs well-mixed flux integral")
            print("   a large residual would mean a misaligned flux axis, a dropped")
            print("   forcing window, or limiter-induced non-conservation")
            for t in tracers:
                fv = fd.variables[FLUXVAR[t]]
                daily_pgc = np.empty(nt)
                for s0 in range(0, nt, 200):
                    s1 = min(nt, s0 + 200)
                    blk = fv[sel[s0]:sel[s1 - 1] + 1]
                    if np.ma.isMaskedArray(blk):
                        blk = blk.filled(0.0)
                    blk = np.asarray(blk, np.float64)
                    daily_pgc[s0:s1] = -(blk * area[None]).sum(axis=(1, 2, 3)) \
                        * 86400.0 * (12.0 / 44.0) / 1e12
                pred = -np.cumsum(daily_pgc) / 2.124        # ppm, well-mixed
                obs = gm[t]
                resid = obs - pred
                rng = float(np.nanmax(pred) - np.nanmin(pred))
                rms = float(np.sqrt(np.nanmean(resid ** 2)))
                print(f"   {t:14s} rms resid {rms:8.4f} ppm  max |resid| "
                      f"{np.nanmax(np.abs(resid)):8.4f} ppm  signal range "
                      f"{rng:8.3f} ppm  ({100 * rms / max(rng, 1e-9):.2f}%)")
                print(f"   {'':14s} final transported {obs[-1]:+10.4f} vs predicted "
                      f"{pred[-1]:+10.4f} ppm")

    if "co2_gpp_anom" not in tracers:
        print("\n[anomaly tracer] absent from the output; skipping its diagnostics")
        a = None
    else:
        a = xco2["co2_gpp_anom"]
        g = gm["co2_gpp_anom"]
        print(f"\n[anomaly tracer] global mean over record {np.nanmean(g):+.4f} ppm")
        print(f"   global-mean range {np.nanmin(g):+.3f} .. {np.nanmax(g):+.3f} ppm")
    if a is not None:
        def masked_mean(msk):
            ww = area * msk
            aa = a.astype(np.float64)
            good = np.isfinite(aa)
            num = np.einsum("tfyx,fyx->t", np.where(good, aa, 0.0), ww)
            den = np.einsum("tfyx,fyx->t", good.astype(np.float64), ww)
            return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

        for nm, msk in (("NH", lats > 0), ("SH", lats <= 0),
                        ("land", lf > 0.05)):
            ts = masked_mean(msk)
            print(f"   {nm:5s} annual means: " + " ".join(
                f"{Y}{'' if ndays[Y] >= 350 else '*'}:{np.nanmean(ts[years == Y]):+.3f}"
                for Y in sorted(set(years))))

    # ---- regional amplitude: is the signal observable? ---------------------
    # The global mean understates what a satellite sees over a source region.
    if a is not None:
        print("\n[regional amplitude] spread of ANNUAL-MEAN xco2_anom across cells")
        print("  year  global    p5     p95    max|dev|   area-wtd std   (ppm)")
        for Y in sorted(set(years)):
            m = years == Y
            fld = np.nanmean(a[m].astype(np.float64), axis=0)
            good = np.isfinite(fld)
            ww = (area * good) / (area * good).sum()
            mu = float((np.where(good, fld, 0.0) * ww).sum())
            dev = fld[good] - mu
            sd = float(np.sqrt((dev ** 2 * ww[good] / ww[good].sum()).sum()))
            p5, p95 = np.percentile(fld[good], [5, 95])
            flag = "" if ndays[Y] >= 350 else "*"
            print(f"  {Y}{flag} {mu:+7.3f} {p5:+7.3f} {p95:+7.3f} "
                  f"{np.abs(dev).max():+10.3f} {sd:14.3f}")
        print("  For scale: OCO-2 regional/monthly XCO2 uncertainty is roughly")
        print("  0.2-0.5 ppm, so annual-mean anomalies above ~0.5 ppm are in reach.")

    hours = np.array([(d - dates[0]).total_seconds() / 3600.0 for d in dates], float)
    nf, ny, nx = shape
    with nc.Dataset(args.out, "w", format="NETCDF4") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("nf", nf)
        ds.createDimension("Ydim", ny)
        ds.createDimension("Xdim", nx)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = f"hours since {dates[0].date().isoformat()} 00:00:00 UTC"
        tv.calendar = "proleptic_gregorian"
        tv[:] = hours
        ds.createVariable("lons", "f8", ("nf", "Ydim", "Xdim"))[:] = lons
        ds.createVariable("lats", "f8", ("nf", "Ydim", "Xdim"))[:] = lats
        ds.createVariable("cell_area", "f8", ("nf", "Ydim", "Xdim"))[:] = area
        ds.createVariable("land_fraction", "f4", ("nf", "Ydim", "Xdim"))[:] = lf
        dims = ("time", "nf", "Ydim", "Xdim")
        for t in tracers:
            v = ds.createVariable(f"xco2_{t.replace('co2_gpp_', '')}", "f4", dims,
                                  zlib=True, complevel=1, shuffle=True)
            v.units = "ppm"
            v.long_name = (f"column-mean dry-air CO2 from {t}, "
                           f"{CARRIER_PPM[t]:.0f} ppm carrier removed")
            v.sign_convention = "positive = anomalously weak GPP uptake"
            v[:] = xco2[t]
        ds.Conventions = "CF-1.8"
        ds.title = "SIF-driven GPP interannual-variability XCO2 experiment (C90)"
        ds.source_run = args.outdir
        ds.flux_driver = args.flux
        ds.history = (f"created {dt.datetime.now(dt.timezone.utc).isoformat()} by "
                      "scripts/diagnostics/analyze_sif_gpp_iav_xco2.py")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
