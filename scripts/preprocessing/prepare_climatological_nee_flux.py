#!/usr/bin/env python3
"""Build a climatological-NEE flux driver: the same seasonal cycle every year.

Transporting this alongside the real NEE separates the two sources of XCO2
interannual variability. The climatological tracer sees identical fluxes in
every year, so every anomaly it develops comes from meteorology -- winds,
mixing, transport pathways. Differencing against the real-flux tracer then
attributes the rest to flux IAV.

The climatology is the 2015-2024 mean of each calendar month, expanded back to
daily by the same monthly-midpoint interpolation with exact per-month
renormalization used by the TRENDY and GFED drivers, so the only difference
from the real driver is the removal of year-to-year variation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_trendy_s3_multiyear_c90 import daily_from_monthly, month_centres

CLIM_YEARS = (2015, 2024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="daily cubed-sphere flux file to take the climatology from")
    ap.add_argument("--variable", default="NEE_CO2_FLUX")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2024)
    args = ap.parse_args()

    with netCDF4.Dataset(args.source) as ds:
        units = ds[args.variable].units
        origin = dt.datetime.strptime(
            ds["time"].units.split("since")[1].strip()[:10], "%Y-%m-%d")
        days = np.array([origin + dt.timedelta(hours=float(h)) for h in ds["time"][:]])
        lon3 = np.asarray(ds["lons"][:], np.float64)
        lat3 = np.asarray(ds["lats"][:], np.float64)
        area3 = np.asarray(ds["cell_area"][:], np.float64)
        shape = lat3.shape
        clim = np.zeros((12, area3.size))
        for month in range(1, 13):
            sel = np.where([CLIM_YEARS[0] <= d.year <= CLIM_YEARS[1]
                            and d.month == month for d in days])[0]
            if sel.size == 0:
                raise SystemExit(f"no {CLIM_YEARS} data for month {month}")
            # Read the month blocks year by year to keep the footprint small.
            total = np.zeros(area3.size)
            for year in range(CLIM_YEARS[0], CLIM_YEARS[1] + 1):
                idx = [i for i in sel if days[i].year == year]
                block = np.asarray(ds[args.variable][idx[0]:idx[-1] + 1], np.float64)
                total += block.reshape(len(idx), -1).mean(axis=0)
            clim[month - 1] = total / (CLIM_YEARS[1] - CLIM_YEARS[0] + 1)
            print(f"  month {month:2d}: {sel.size} days, global mean "
                  f"{(clim[month - 1] * area3.ravel()).sum() * 86400 * 365.25 / 1e12 * 12.011 / 44.0095:+7.2f} "
                  "Pg C/yr equivalent", flush=True)

    mkeys, mcent = month_centres(args.start_year - 1, 12, args.end_year + 1, 1)
    monthly = np.stack([clim[m - 1] for (_, m) in mkeys])   # same 12 fields, repeated
    day0 = dt.datetime(args.start_year, 1, 1)
    nday = (dt.datetime(args.end_year + 1, 1, 1) - day0).days
    # daily_from_monthly clips negatives, which would break a signed NEE field;
    # split into the emission and uptake halves and recombine.
    dates, pos, e1 = daily_from_monthly(np.maximum(monthly, 0.0), mkeys, mcent,
                                        day0, nday)
    _, neg, e2 = daily_from_monthly(np.maximum(-monthly, 0.0), mkeys, mcent,
                                    day0, nday)
    daily = (pos - neg).astype(np.float32)
    print(f"renormalization max error {max(e1, e2):.2e}")

    yrs = np.array([d.year for d in dates])
    for Y in sorted(set(yrs))[:3] + sorted(set(yrs))[-2:]:
        sel = yrs == Y
        pgc = float((daily[sel].astype(np.float64) * area3.ravel()[None, :]).sum()) \
            * 86400.0 * (12.011 / 44.0095) / 1e12
        print(f"  {Y} net {pgc:+7.3f} Pg C   (must be identical across years "
              "apart from leap days)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nf, ny, nx = shape
    with netCDF4.Dataset(args.out, "w", format="NETCDF4") as ds:
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
        v = ds.createVariable("NEE_CLIM_CO2_FLUX", "f4",
                              ("time", "nf", "Ydim", "Xdim"), zlib=True,
                              complevel=1, shuffle=True,
                              fill_value=np.float32(-999.0))
        v.units = units
        v.long_name = (f"{CLIM_YEARS[0]}-{CLIM_YEARS[1]} mean seasonal cycle of "
                       f"{args.variable}, repeated every year")
        v.positive = "to_atmosphere"
        v[:] = daily.reshape((nday,) + shape)
        ds.Conventions = "CF-1.8"
        ds.title = "Climatological NEE control flux"
        ds.source_file = str(args.source)
        ds.method = (f"calendar-month mean over {CLIM_YEARS[0]}-{CLIM_YEARS[1]}, "
                     "repeated each year, expanded to daily by monthly-midpoint "
                     "interpolation with exact per-month renormalization")
        ds.history = f"created {dt.datetime.now(dt.timezone.utc).isoformat()} by " \
                     "scripts/preprocessing/prepare_climatological_nee_flux.py"
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
