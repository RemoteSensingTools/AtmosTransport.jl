#!/usr/bin/env python3
"""Daily cubed-sphere GFED5 fire-CO2 flux driver.

Produces one transportable fire tracer that can be superimposed on the TRENDY
NEE tracers, which carry Ra + Rh - GPP only and no fire at all. Output matches
the TRENDY driver's format exactly (same grid, same units, same daily
construction), so the tracers add linearly:

  FIRE_CO2_FLUX   +C * 44/12   (>= 0, emission)   kg CO2 m-2 s-1

Source (spandey's snapshot, the series behind the Nature manuscript):
  1997-2022  GFED5.1 finalized, annual containers of 12 monthly grids, `C`
             in g C per month per grid cell (1 deg before 2001, 0.25 deg after)
  2023-2025  GFED5NRT reprocessed, one file per month, `EM` in g C per month
             summed over the 16 land-cover types

`EM` summed over `lct` reproduces the published monthly global total bitwise
(checked at 2024-07: 0.55272904300404746 Pg C both ways), so the two products
are used unmodified and simply concatenated.

Daily means come from linear interpolation between monthly midpoints followed
by an exact per-calendar-month renormalization -- identical to
prepare_trendy_s3_multiyear_c90.py, so fire and NEE receive the same temporal
treatment. GFED is monthly; the interpolation adds no information, it only
avoids month-boundary steps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_trendy_s3_gpp_ter_c90 as T
from prepare_trendy_s3_multiyear_c90 import daily_from_monthly, month_centres

GFED_ROOT = Path("/home/spandey/gfed5_monthly_global_fire_carbon_1997_2025"
                 "/gridded_monthly")
FINALIZED = GFED_ROOT / "GFED5.1_finalized/monthly_full_1997_2022"
NRT = GFED_ROOT / "GFED5NRT_reprocessed"
FINALIZED_LAST_YEAR = 2022
SECONDS_PER_DAY = 86400.0


def month_seconds(year: int, month: int) -> float:
    start = dt.datetime(year, month, 1)
    end = dt.datetime(year + 1, 1, 1) if month == 12 else dt.datetime(year, month + 1, 1)
    return (end - start).total_seconds()


def read_month(year: int, month: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (grams C in the month, lat, lon) on the native GFED grid."""
    if year <= FINALIZED_LAST_YEAR:
        path = FINALIZED / f"GFED5.1_monthly_{year}.nc"
        with netCDF4.Dataset(path) as ds:
            var = ds["C"]
            units = getattr(var, "units", "")
            if units != "g C per month":
                raise ValueError(f"{path}: unexpected units {units!r}")
            if var.shape[0] != 12:
                raise ValueError(f"{path}: expected 12 monthly slices")
            grams = np.ma.filled(var[month - 1], 0.0).astype(np.float64)
            lat = np.asarray(ds["lat"][:], np.float64)
            lon = np.asarray(ds["lon"][:], np.float64)
    else:
        path = NRT / f"{year}" / f"GFED5NRTeco_CMB_{year}-{month:02d}.nc"
        with netCDF4.Dataset(path) as ds:
            # EM is (time, lct, lat, lon) g C per month; sum the land-cover types.
            grams = np.ma.filled(ds["EM"][0], 0.0).astype(np.float64).sum(axis=0)
            lat = np.asarray(ds["lat"][:], np.float64)
            lon = np.asarray(ds["lon"][:], np.float64)
    return np.nan_to_num(grams), lat, lon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, required=True,
                    help="NetCDF with lats/lons/corner_lats/corner_lons of the "
                         "target cubed-sphere grid (any Nc)")
    ap.add_argument("--tag", default="c30", help="grid tag used in the filename")
    ap.add_argument("--staging-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "gfed5_staging")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(args.geometry) as ds:
        lat3 = np.asarray(ds["lats"][:], np.float64)
        lon3 = np.asarray(ds["lons"][:], np.float64)
        area3 = T.c90_cell_areas(np.asarray(ds["corner_lats"][:], np.float64),
                                 np.asarray(ds["corner_lons"][:], np.float64))
    area = area3.ravel()
    assert abs(area.sum() / (4 * np.pi * T.R_EARTH**2) - 1) < 1e-6
    print(f"target grid {lat3.shape}, building 0.25 deg overlap map", flush=True)
    fine = T.fine_grid_target_map(lat3, lon3, 0.25)

    mkeys, mcent = month_centres(args.start_year - 1, 12, args.end_year + 1, 1)
    day0 = dt.datetime(args.start_year, 1, 1)
    nday = (dt.datetime(args.end_year + 1, 1, 1) - day0).days

    matrix_cache: dict[tuple[int, int], object] = {}
    monthly = np.empty((len(mkeys), area.size))
    native_pgc = np.empty(len(mkeys))
    for i, (year, month) in enumerate(mkeys):
        grams, lat, lon = read_month(year, month)
        native_pgc[i] = grams.sum() / 1e15
        key = (lat.size, lon.size)
        if key not in matrix_cache:
            print(f"  conservative {grams.shape} -> target weights", flush=True)
            matrix_cache[key] = T.conservative_matrix(
                lat, lon, fine[2], fine[0], fine[1], fine[3], area.size)[0]
        # g C per cell per month -> kg C m-2 s-1 on the source grid
        src_area = T.regular_cell_areas(lat, lon)
        rate = grams * 1e-3 / src_area / month_seconds(year, month)
        monthly[i] = T.remap_months(matrix_cache[key], rate[None], np.ones_like(rate),
                                    area)[0]
        remapped = float((monthly[i] * area).sum()) * month_seconds(year, month) / 1e12
        if abs(remapped - native_pgc[i]) > 1e-6 * max(native_pgc[i], 1e-3):
            raise SystemExit(f"{year}-{month:02d}: remap lost mass, "
                             f"{native_pgc[i]:.6f} -> {remapped:.6f} Pg C")
        print(f"  {year}-{month:02d}  {native_pgc[i]:.4f} Pg C -> {remapped:.4f}",
              flush=True)

    dates, daily, err = daily_from_monthly(monthly, mkeys, mcent, day0, nday)
    print(f"monthly renormalization max error {err:.2e}", flush=True)
    flux = (daily * T.KG_CO2_PER_KG_C).astype(np.float32)
    assert np.isfinite(flux).all() and (flux >= 0).all()

    yrs = np.array([d.year for d in dates])
    lookup = {k: i for i, k in enumerate(mkeys)}
    print("  year   daily-file PgC   native GFED PgC")
    for Y in sorted(set(yrs)):
        sel = yrs == Y
        got = float((flux[sel].astype(np.float64) * area[None, :]).sum()) \
            * SECONDS_PER_DAY * (12.0 / 44.0) / 1e12
        want = sum(native_pgc[lookup[(Y, m)]] for m in range(1, 13))
        print(f"  {Y}   {got:10.3f}   {want:14.3f}   ({got / want - 1:+.2%})")

    out = args.outdir / (f"GFED5_fire_daily_co2flux_{args.tag}_"
                         f"{args.start_year}_{args.end_year}.nc")
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
        v = ds.createVariable("FIRE_CO2_FLUX", "f4", ("time", "nf", "Ydim", "Xdim"),
                              zlib=True, complevel=1, shuffle=True,
                              fill_value=np.float32(-999.0))
        v.units = "kg CO2 m-2 s-1"
        v.long_name = "GFED5 fire carbon as atmospheric CO2 emission"
        v.positive = "to_atmosphere"
        v[:] = flux.reshape((nday,) + lat3.shape)
        ds.Conventions = "CF-1.8"
        ds.title = f"GFED5 daily {args.tag} fire CO2 flux"
        ds.source = ("GFED5.1 finalized 1997-2022 (variable C) and GFED5NRT "
                     "reprocessed 2023-2025 (variable EM summed over lct)")
        ds.method = ("conservative remap of monthly g C per cell to the cubed "
                     "sphere; daily by monthly-midpoint interpolation with exact "
                     "per-month renormalization; C -> CO2 with 44/12 to match the "
                     "TRENDY drivers")
        ds.history = f"created {dt.datetime.now(dt.timezone.utc).isoformat()} by " \
                     "scripts/preprocessing/prepare_gfed5_fire_flux.py"
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
