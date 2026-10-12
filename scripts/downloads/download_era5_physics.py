#!/usr/bin/env python3
"""
Download ERA5 model-level physics fields for TM5-compatible convection and diffusion.

Downloads two groups of fields on all 137 model levels at hourly resolution:

  Convection (forecast fields, params 235009-235012):
    - 235009: Time-mean updraught mass flux (UDMF) [kg/m2/s]
    - 235010: Time-mean downdraught mass flux (DDMF) [kg/m2/s]
    - 235011: Time-mean updraught detrainment rate (UDRF) [kg/m3/s]
    - 235012: Time-mean downdraught detrainment rate (DDRF) [kg/m3/s]

  Thermodynamics (analysis fields, params 130 + 133):
    - 130: Temperature (T) [K]
    - 133: Specific humidity (Q) [kg/kg]

Parameter IDs confirmed by ECMWF (2026-03-31). Previous IDs (235071/235072,
214.162/215.162) do NOT exist in ERA5 experiment version 1.

Convection fields are time-mean forecasts from base times 06/18 UTC with
hourly steps 1-12, giving 24 values/day. Thermodynamic fields are hourly
analyses (00-23 UTC), also 24 values/day.

Output: one NetCDF per field group per day:
    era5_convection_YYYYMMDD.nc  — UDMF, DDMF, UDRF, DDRF
    era5_thermo_ml_YYYYMMDD.nc   — T, Q

Usage:
    python3 scripts/downloads/download_era5_physics.py \\
        --start 2021-12-01 --end 2021-12-31 \\
        --outdir ~/data/AtmosTransport/met/era5/physics

    # Only convection:
    python3 scripts/downloads/download_era5_physics.py \\
        --start 2021-12-01 --end 2021-12-07 \\
        --outdir ~/data/AtmosTransport/met/era5/physics \\
        --fields convection

    # Only thermodynamics (T + Q):
    python3 scripts/downloads/download_era5_physics.py \\
        --start 2021-12-01 --end 2021-12-07 \\
        --outdir ~/data/AtmosTransport/met/era5/physics \\
        --fields thermodynamics

Requirements:
    pip install cdsapi cfgrib eccodes xarray netcdf4
    (and ~/.cdsapirc configured for CDS access)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


# ── Parameter definitions ─────────────────────────────────────────────────

# Convection: forecast fields from 06/18 UTC, hourly steps 1-12
CONVECTION_PARAMS = {
    235009: "udmf",   # updraft mass flux
    235010: "ddmf",   # downdraft mass flux
    235011: "udrf",   # updraft detrainment rate
    235012: "ddrf",   # downdraft detrainment rate
}

# Thermodynamics: analysis fields, hourly 00-23 UTC
THERMO_PARAMS = {
    130: "t",   # temperature
    133: "q",   # specific humidity
}

# Canonical variable names for renaming after cfgrib conversion
PARAM_RENAME = {
    **{f"var{pid}": name for pid, name in CONVECTION_PARAMS.items()},
    **{f"var{pid}": name for pid, name in THERMO_PARAMS.items()},
}


# ── Helpers ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download ERA5 model-level physics fields (convection + thermodynamics)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Grid spacing in degrees (default: 0.5)")
    p.add_argument("--fields", default="all",
                   help="Which fields to download: all, convection, thermodynamics "
                        "(comma-separated, default: all)")
    p.add_argument("--keep-grib", action="store_true",
                   help="Keep GRIB files after NetCDF conversion")
    return p.parse_args()


def date_range(start, end_date):
    """Yield dates from start to end_date inclusive."""
    d = start
    while d <= end_date:
        yield d
        d += timedelta(days=1)


def grib_to_netcdf_by_param(grib_path, nc_path, param_ids, canonical_names):
    """Convert GRIB to NetCDF, extracting variables by paramId and renaming.

    Each paramId is opened separately via filter_by_keys to handle cases
    where cfgrib merges different parameters or assigns unexpected names.
    Variables are renamed to canonical names (udmf, ddmf, etc.).
    """
    import xarray as xr

    datasets = []
    for pid, cname in zip(param_ids, canonical_names):
        try:
            ds = xr.open_dataset(
                grib_path, engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"paramId": pid}})
            # Rename the data variable to canonical name
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1 and data_vars[0] != cname:
                ds = ds.rename({data_vars[0]: cname})
            datasets.append(ds)
        except Exception as e:
            print(f"    WARNING: Could not extract paramId {pid} ({cname}): {e}")

    if not datasets:
        raise RuntimeError(f"No variables extracted from {grib_path}")

    merged = xr.merge(datasets)
    # Compress with zlib level 4 (~4x reduction, fast)
    encoding = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
    merged.to_netcdf(nc_path, encoding=encoding)
    merged.close()
    for ds in datasets:
        ds.close()

    print(f"    Converted to NetCDF: {os.path.basename(nc_path)} "
          f"({', '.join(canonical_names)})")


# ── Download functions ────────────────────────────────────────────────────

def download_convection(client, date_str, resolution, outdir, keep_grib):
    """Download hourly convection fields (UDMF, DDMF, UDRF, DDRF) for one day."""
    nc_path = os.path.join(outdir, f"era5_convection_{date_str.replace('-', '')}.nc")
    grib_path = nc_path.replace(".nc", ".grib")

    if os.path.exists(nc_path):
        sz = os.path.getsize(nc_path) / 1e6
        print(f"  SKIP convection (exists, {sz:.1f} MB): {os.path.basename(nc_path)}")
        return

    if not os.path.exists(grib_path):
        print(f"  Downloading convection fields for {date_str}...")
        # Forecast fields from base times 06/18 UTC, hourly steps 1-12
        # step=1 from base 06 → mean over 06-07 UTC
        # step=12 from base 06 → mean over 17-18 UTC
        # step=1 from base 18 → mean over 18-19 UTC
        # step=12 from base 18 → mean over 05-06 UTC (+1 day)
        # Together: 24 hourly-averaged fields per day
        client.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "fc",
                "levtype": "ml",
                "levelist": "1/to/137",
                "param": "/".join(str(p) for p in CONVECTION_PARAMS),
                "date": date_str,
                "time": "06:00:00/18:00:00",
                "step": "/".join(str(s) for s in range(1, 13)),
                "grid": f"{resolution}/{resolution}",
            },
            grib_path,
        )
        sz = os.path.getsize(grib_path) / 1e6
        print(f"    GRIB: {os.path.basename(grib_path)} ({sz:.1f} MB)")

    grib_to_netcdf_by_param(
        grib_path, nc_path,
        list(CONVECTION_PARAMS.keys()),
        list(CONVECTION_PARAMS.values()))

    if not keep_grib and os.path.exists(grib_path):
        os.remove(grib_path)


def download_thermodynamics(client, date_str, resolution, outdir, keep_grib):
    """Download hourly T + Q on model levels for one day."""
    nc_path = os.path.join(outdir, f"era5_thermo_ml_{date_str.replace('-', '')}.nc")
    grib_path = nc_path.replace(".nc", ".grib")

    if os.path.exists(nc_path):
        sz = os.path.getsize(nc_path) / 1e6
        print(f"  SKIP thermodynamics (exists, {sz:.1f} MB): {os.path.basename(nc_path)}")
        return

    if not os.path.exists(grib_path):
        print(f"  Downloading T + Q for {date_str}...")
        # Analysis fields, hourly 00-23 UTC
        client.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": "1/to/137",
                "param": "/".join(str(p) for p in THERMO_PARAMS),
                "date": date_str,
                "time": "/".join(f"{h:02d}:00:00" for h in range(24)),
                "grid": f"{resolution}/{resolution}",
            },
            grib_path,
        )
        sz = os.path.getsize(grib_path) / 1e6
        print(f"    GRIB: {os.path.basename(grib_path)} ({sz:.1f} MB)")

    grib_to_netcdf_by_param(
        grib_path, nc_path,
        list(THERMO_PARAMS.keys()),
        list(THERMO_PARAMS.values()))

    if not keep_grib and os.path.exists(grib_path):
        os.remove(grib_path)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Parse --fields
    fields = set(args.fields.lower().replace(" ", "").split(","))
    if "all" in fields:
        fields = {"convection", "thermodynamics"}
    valid = {"convection", "thermodynamics"}
    unknown = fields - valid
    if unknown:
        print(f"ERROR: Unknown field groups: {unknown}. Valid: {', '.join(valid)}")
        sys.exit(1)

    do_conv = "convection" in fields
    do_thermo = "thermodynamics" in fields

    # Check dependencies
    for pkg in ["cdsapi", "cfgrib", "xarray"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"ERROR: {pkg} not installed. Run: pip install cdsapi cfgrib eccodes xarray netcdf4")
            sys.exit(1)

    import cdsapi

    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        print("ERROR: ~/.cdsapirc not found. Configure CDS API credentials.")
        sys.exit(1)

    client = cdsapi.Client()

    print("ERA5 Model-Level Physics Downloader")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Resolution: {args.resolution} deg")
    print(f"  Output:     {outdir}")
    print(f"  Fields:     {', '.join(sorted(fields))}")
    if do_conv:
        print(f"  Convection: params 235009/235010/235011/235012 (fc, hourly)")
    if do_thermo:
        print(f"  Thermo:     params 130/133 (an, hourly)")
    print()

    n_days = (end_date - start).days + 1
    for i, dt in enumerate(date_range(start, end_date)):
        date_str = dt.strftime("%Y-%m-%d")
        print(f"[{i+1}/{n_days}] {date_str}")

        if do_conv:
            download_convection(client, date_str, args.resolution, outdir,
                                args.keep_grib)
        if do_thermo:
            download_thermodynamics(client, date_str, args.resolution, outdir,
                                    args.keep_grib)

    print(f"\nDone! Downloaded {n_days} days to {outdir}")
    if do_conv:
        print("\nNext steps for convection:")
        print(f"  julia --project=. scripts/preprocessing/preprocess_era5_tm5_convection.jl \\")
        print(f"      <massflux.nc> {outdir}")
        print("  Then set [convection] type = \"tm5\" in your run config.")


if __name__ == "__main__":
    main()
