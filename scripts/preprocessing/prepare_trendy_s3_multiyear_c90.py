#!/usr/bin/env python3
"""Multi-year daily C90 GPP/TER/NEE flux drivers from TRENDYv14 S3.

Reuses the validated conservative-remap machinery of
prepare_trendy_s3_gpp_ter_c90.py (the 2021 hourly pilot) but produces one
daily-mean file per model spanning 2018-01-01 to 2024-12-31 (TRENDY v14 ends
2024-12), matching the SIF-GPP experiment's format: variables on the GEOS
native C90 cubed sphere, kg CO2 m-2 s-1 per total cell area, negative = uptake.

  GPP_CO2_FLUX   -gpp * 44/12            (<= 0)
  TER_CO2_FLUX   +(ra + rh) * 44/12      (>= 0, emission)
  NEE_CO2_FLUX   TER + GPP               (annually near-balanced net flux)

Daily means come from linear interpolation between monthly midpoints followed
by an exact per-calendar-month renormalization (leap-aware, continuous across
year boundaries, unlike the 2021 pilot's 365-day helper).
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_trendy_s3_gpp_ter_c90 as T

C90_GEOM = Path("/kiwi-data/Data/satellite/TROPOMI/TROPOMI_SIF_S5P-PAL/regridded/"
                "interpolated/TROPOMI_sif_20180501_20251231_C90_daily_"
                "gt-2SIF743lt5_SIF743ERRORlt10_filled.nc")


def month_centres(y0, m0, y1, m1):
    keys, centres = [], []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        keys.append((y, m))
        nd = calendar.monthrange(y, m)[1]
        centres.append(dt.datetime(y, m, 1) + dt.timedelta(days=nd / 2))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return keys, centres


def daily_from_monthly(monthly, keys, centres, day0, nday):
    """Linear interp between monthly midpoints + exact per-month renormalization."""
    origin = day0
    cx = np.array([(c - origin).total_seconds() / 86400.0 for c in centres])
    dx = np.arange(nday, dtype=np.float64) + 0.5
    flat = monthly.reshape(len(keys), -1)
    daily = np.empty((nday, flat.shape[1]), np.float64)
    for j0 in range(0, flat.shape[1], 12150):        # column blocks to bound memory
        j1 = min(flat.shape[1], j0 + 12150)
        for j in range(j0, j1):
            daily[:, j] = np.interp(dx, cx, flat[:, j])
    np.maximum(daily, 0.0, out=daily)
    dates = [day0 + dt.timedelta(days=int(d)) for d in range(nday)]
    lookup = {k: i for i, k in enumerate(keys)}
    max_err = 0.0
    for (y, m) in sorted({(d.year, d.month) for d in dates}):
        sel = np.array([d.year == y and d.month == m for d in dates])
        target = flat[lookup[(y, m)]]
        mean = daily[sel].mean(axis=0)
        scale = np.divide(target, mean, out=np.zeros_like(target), where=mean > 0)
        daily[sel] *= scale
        restored = daily[sel].mean(axis=0)
        act = target > 1e-20
        if act.any():
            max_err = max(max_err, float(np.abs(restored[act] / target[act] - 1).max()))
    return dates, daily, max_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["CLM", "JULES-ES", "ORCHIDEE"])
    ap.add_argument("--root", type=Path, default=T.DEFAULT_ROOT)
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, default=C90_GEOM,
                    help="NetCDF with lats/lons/corner_lats/corner_lons of the "
                         "target cubed-sphere grid (any Nc; e.g. a model output "
                         "file). Default: the C90 SIF driver grid.")
    ap.add_argument("--tag", default="c90",
                    help="grid tag used in the output filename")
    ap.add_argument("--staging-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "trendy_multiyear_staging")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(args.geometry) as ds:
        lat3 = np.asarray(ds["lats"][:], np.float64)
        lon3 = np.asarray(ds["lons"][:], np.float64)
        area3 = T.c90_cell_areas(np.asarray(ds["corner_lats"][:], np.float64),
                                 np.asarray(ds["corner_lons"][:], np.float64))
    area = area3.ravel()
    assert abs(area.sum() / (4 * np.pi * T.R_EARTH**2) - 1) < 1e-6
    print("building fine-grid overlap map (0.25 deg)", flush=True)
    fine = T.fine_grid_target_map(lat3, lon3, 0.25)

    # months with one-month padding each side; TRENDY ends 2024-12, so the
    # trailing edge relies on np.interp's clamp for the last ~15 days.
    mkeys, mcent = month_centres(args.start_year - 1, 12,
                                 min(args.end_year, 2024), 12)
    day0 = dt.datetime(args.start_year, 1, 1)
    nday = (dt.datetime(args.end_year + 1, 1, 1) - day0).days

    matrix_cache = {}
    for model in args.models:
        model_dir = args.root / model
        paths = T.model_inputs(model_dir)
        if paths is None:
            raise SystemExit(f"{model}: missing gpp/ra/rh")
        monthly = {}
        for kind in ("gpp", "ra", "rh"):
            print(f"[{model}] reading {kind}", flush=True)
            with T.materialized(paths[kind], args.staging_dir) as readable, \
                 netCDF4.Dataset(readable) as ds:
                var = T.flux_variable(ds, kind)
                shp = tuple(var.shape[-2:])
                lat = T.coordinate(ds, "lat", shp[0])
                lon = T.coordinate(ds, "lon", shp[1])
                gk = hashlib.sha1(lat.tobytes() + lon.tobytes()).hexdigest()
                if gk not in matrix_cache:
                    print(f"[{model}] conservative {shp} -> C90 weights", flush=True)
                    matrix_cache[gk] = T.conservative_matrix(
                        lat, lon, fine[2], fine[0], fine[1], fine[3], area.size)
                matrix, _ = matrix_cache[gk]
                if kind == "gpp":
                    lf, lf_note = T.read_land_fraction(model, model_dir, ds, shp,
                                                       args.staging_dir)
                src, units, negfrac, _ = T.read_months(ds, kind, mkeys)
            monthly[kind] = T.remap_months(matrix, src, lf, area)
            print(f"[{model}] {kind}: units {units}, negative fraction "
                  f"{negfrac:.4f}, land-fraction: {lf_note}", flush=True)

        dates, dgpp, e1 = daily_from_monthly(monthly["gpp"], mkeys, mcent, day0, nday)
        _, dra, e2 = daily_from_monthly(monthly["ra"], mkeys, mcent, day0, nday)
        _, drh, e3 = daily_from_monthly(monthly["rh"], mkeys, mcent, day0, nday)
        print(f"[{model}] monthly renormalization max error: gpp {e1:.2e}, "
              f"ra {e2:.2e}, rh {e3:.2e}", flush=True)
        gflux = (-dgpp * T.KG_CO2_PER_KG_C).astype(np.float32)
        raflux = (dra * T.KG_CO2_PER_KG_C).astype(np.float32)
        rhflux = (drh * T.KG_CO2_PER_KG_C).astype(np.float32)
        # TER built from the daily ra/rh so TER == RA + RH bitwise in the file.
        tflux = raflux + rhflux
        nflux = gflux + tflux
        yrs = np.array([d.year for d in dates])

        def pgc(f, m):
            return float((f[m].astype(np.float64) * area[None, :]).sum()) \
                * 86400.0 * (12.0 / 44.0) / 1e12

        print(f"[{model}]  year    GPP      TER      NEE   (PgC; NEE<0 = net uptake"
              " in atmospheric sign: value shown is carbon removed)")
        for Y in sorted(set(yrs)):
            m = yrs == Y
            print(f"[{model}]  {Y}  {-pgc(gflux, m):7.2f}  {pgc(tflux, m):7.2f}  "
                  f"{pgc(nflux, m):+7.2f}")
        assert not np.isnan(nflux).any()

        out = args.outdir / (f"TRENDYv14_S3_{model}_gpp_ter_nee_daily_co2flux_"
                             f"{args.tag}_{args.start_year}_{args.end_year}.nc")
        nf, ny, nx = lat3.shape
        with netCDF4.Dataset(out, "w", format="NETCDF4") as ds:
            ds.createDimension("time", nday)
            ds.createDimension("nf", nf)
            ds.createDimension("Ydim", ny)
            ds.createDimension("Xdim", nx)
            tv = ds.createVariable("time", "f8", ("time",))
            tv.units = f"hours since {day0.date().isoformat()} 00:00:00 UTC"
            tv.calendar = "proleptic_gregorian"
            tv[:] = [(d - day0).total_seconds() / 3600.0 for d in dates]
            ds.createVariable("lons", "f8", ("nf", "Ydim", "Xdim"))[:] = lon3
            ds.createVariable("lats", "f8", ("nf", "Ydim", "Xdim"))[:] = lat3
            ds.createVariable("cell_area", "f8", ("nf", "Ydim", "Xdim"))[:] = area3
            dims = ("time", "nf", "Ydim", "Xdim")
            for name, data, ln in (
                ("GPP_CO2_FLUX", gflux, "TRENDY S3 GPP as atmospheric CO2 uptake"),
                ("TER_CO2_FLUX", tflux, "TRENDY S3 ra+rh as atmospheric CO2 emission"),
                ("RA_CO2_FLUX", raflux, "TRENDY S3 autotrophic respiration emission"),
                ("RH_CO2_FLUX", rhflux, "TRENDY S3 heterotrophic respiration emission"),
                ("NEE_CO2_FLUX", nflux, "TER + GPP: net ecosystem exchange"),
            ):
                v = ds.createVariable(name, "f4", dims, zlib=True, complevel=1,
                                      shuffle=True, fill_value=np.float32(-999.0))
                v.units = "kg CO2 m-2 s-1"
                v.long_name = ln
                v.positive = "to_atmosphere"
                v[:] = data.reshape((nday,) + lat3.shape)
            ds.Conventions = "CF-1.8"
            ds.title = f"TRENDYv14 S3 {model} daily C90 CO2 fluxes"
            ds.source_model = model
            ds.method = ("conservative remap of monthly gpp/ra/rh to C90; daily by "
                         "monthly-midpoint interpolation with exact per-month "
                         "renormalization")
            ds.history = f"created {dt.datetime.now(dt.timezone.utc).isoformat()} by " \
                         "scripts/preprocessing/prepare_trendy_s3_multiyear_c90.py"
        print(f"[{model}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
