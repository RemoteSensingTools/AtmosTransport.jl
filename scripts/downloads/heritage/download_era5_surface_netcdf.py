#!/usr/bin/env python3
"""Download ERA5 single-level surface fields as a monthly NetCDF for the
N320 → C180 transport preprocessor's surface / PBL-diffusion payload.

The cubed-sphere N320 regrid path reads the surface fields from a *regular*
0.25° lat-lon NetCDF at

    <root>/sfc_an_native/era5_surface_YYYYMM.nc

(see `era5_surface_reader.jl`, which transparently unzips the CDS
`stepType-instant` + `stepType-accum` sub-NetCDFs). The CDS Modern API
splits a single hourly single-levels request into those two streams.

This pulls the FULL surface set the preprocessor and the TM5 boundary-layer
diffusion (`bldiff`) need:

  instant : blh (pblh), zust (ustar), 2t (t2m), 10u, 10v, sp, z, lsm, 2d
  accum   : sshf (sensible heat flux), slhf (LATENT heat flux)

`slhf` is the field that was missing from the first pull — TM5's `bldiff`
forms the surface virtual heat flux `wheatv = wheat(sshf) + c·θ·wqflx(slhf)`,
so the latent flux is required to enable the faithful TM5 non-local PBL
scheme (Holtslag-Boville).

Usage:
    python3 scripts/downloads/heritage/download_era5_surface_netcdf.py \
        --year 2021 --month 12 \
        --out ~/data/AtmosTransport/met/era5/N320/hourly/raw/sfc_an_native

CDS-only (the user has no MARS access); credentials read from ~/.cdsapirc.
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from pathlib import Path

import cdsapi

# Full surface set. The four the diffusion path strictly needs are
# boundary_layer_height, friction_velocity, 2m_temperature,
# surface_sensible_heat_flux + surface_latent_heat_flux; the rest round out
# the descriptor so the on-disk NetCDF is self-sufficient for any surface
# consumer (and fixes the earlier missing-variable gap).
SURFACE_VARIABLES = [
    "boundary_layer_height",
    "friction_velocity",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "geopotential",
    "land_sea_mask",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default="~/data/AtmosTransport/met/era5/N320/hourly/raw/sfc_an_native",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the target NetCDF already exists.",
    )
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"era5_surface_{args.year:04d}{args.month:02d}.nc"
    out_path = out_dir / stem

    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path} exists (pass --overwrite to refetch)")
        return 0

    ndays = calendar.monthrange(args.year, args.month)[1]
    days = [f"{d:02d}" for d in range(1, ndays + 1)]
    times = [f"{h:02d}:00" for h in range(24)]

    request = {
        "product_type": "reanalysis",
        "variable": SURFACE_VARIABLES,
        "year": f"{args.year:04d}",
        "month": f"{args.month:02d}",
        "day": days,
        "time": times,
        "data_format": "netcdf",
        "download_format": "zip",
    }

    # Stage to a temp name so a partial/interrupted download never looks like
    # a complete file to the preprocessor.
    tmp_path = out_path.with_suffix(".nc.tmp")
    print(f"[retrieve] reanalysis-era5-single-levels {args.year}-{args.month:02d} "
          f"({ndays} days × 24h, {len(SURFACE_VARIABLES)} vars) → {out_path}")
    c = cdsapi.Client()
    c.retrieve("reanalysis-era5-single-levels", request, str(tmp_path))
    os.replace(tmp_path, out_path)
    print(f"[done] {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
