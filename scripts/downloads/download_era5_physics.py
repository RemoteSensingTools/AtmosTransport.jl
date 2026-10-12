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
import math
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


def expected_grib_inventory(date_str, param_ids, convection):
    date_id = int(date_str.replace("-", ""))
    times = (600, 1800) if convection else tuple(hour * 100 for hour in range(24))
    steps = range(1, 13) if convection else (0,)
    return {
        (param_id, date_id, time, step, level)
        for param_id in param_ids
        for time in times
        for step in steps
        for level in range(1, 138)
    }


def regular_ll_shape(resolution):
    """Return the global regular-lat/lon dimensions for a grid spacing."""
    ni = round(360.0 / resolution)
    nj_intervals = round(180.0 / resolution)
    if (not math.isclose(ni * resolution, 360.0, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(nj_intervals * resolution, 180.0,
                                rel_tol=0.0, abs_tol=1e-9)):
        raise ValueError(
            f"resolution {resolution} must divide both 360 and 180 degrees")
    return ni, nj_intervals + 1


def validate_grib(path, expected, label, resolution, convection):
    """Require the requested inventory and regular model-level representation."""
    from eccodes import (codes_get_double, codes_get_long, codes_get_string,
                         codes_grib_new_from_file, codes_release)

    seen = set()
    expected_ni, expected_nj = regular_ll_shape(resolution)
    expected_data_type = "fc" if convection else "an"
    expected_step_type = "avg" if convection else "instant"
    try:
        with open(path, "rb") as stream:
            while True:
                gid = codes_grib_new_from_file(stream)
                if gid is None:
                    break
                try:
                    key = (
                        codes_get_long(gid, "paramId"),
                        codes_get_long(gid, "dataDate"),
                        codes_get_long(gid, "dataTime"),
                        codes_get_long(gid, "step"),
                        codes_get_long(gid, "level"),
                    )
                    representation = {
                        "gridType": codes_get_string(gid, "gridType"),
                        "typeOfLevel": codes_get_string(gid, "typeOfLevel"),
                        "dataType": codes_get_string(gid, "dataType"),
                        "stepType": codes_get_string(gid, "stepType"),
                        "Ni": codes_get_long(gid, "Ni"),
                        "Nj": codes_get_long(gid, "Nj"),
                        "iDirectionIncrementInDegrees": codes_get_double(
                            gid, "iDirectionIncrementInDegrees"),
                        "jDirectionIncrementInDegrees": codes_get_double(
                            gid, "jDirectionIncrementInDegrees"),
                    }
                finally:
                    codes_release(gid)
                expected_representation = {
                    "gridType": "regular_ll",
                    "typeOfLevel": "hybrid",
                    "dataType": expected_data_type,
                    "stepType": expected_step_type,
                    "Ni": expected_ni,
                    "Nj": expected_nj,
                }
                for name, wanted in expected_representation.items():
                    if representation[name] != wanted:
                        raise ValueError(
                            f"message {key} has {name}={representation[name]!r}; "
                            f"expected {wanted!r}")
                for name in ("iDirectionIncrementInDegrees",
                             "jDirectionIncrementInDegrees"):
                    if not math.isclose(representation[name], resolution,
                                        rel_tol=0.0, abs_tol=1e-9):
                        raise ValueError(
                            f"message {key} has {name}={representation[name]!r}; "
                            f"expected {resolution!r}")
                if key in seen:
                    raise ValueError(f"duplicate GRIB message {key}")
                seen.add(key)
        if seen != expected:
            raise ValueError(
                f"inventory mismatch: {len(expected - seen)} missing and "
                f"{len(seen - expected)} unexpected messages "
                f"(read {len(seen)}, expected {len(expected)})")
        return True
    except Exception as exc:
        print(f"    INVALID {label}: {exc}")
        return False


def retrieve_grib_atomic(client, request, grib_path, expected, label,
                         resolution, convection):
    if (os.path.isfile(grib_path)
            and validate_grib(grib_path, expected, label,
                              resolution, convection)):
        return
    part_path = grib_path + ".part"
    if os.path.exists(part_path):
        os.remove(part_path)
    try:
        client.retrieve("reanalysis-era5-complete", request, part_path)
        if not validate_grib(part_path, expected, label,
                             resolution, convection):
            raise RuntimeError("downloaded GRIB failed inventory validation")
        os.replace(part_path, grib_path)
    except Exception:
        if os.path.exists(part_path):
            os.remove(part_path)
        raise


def validate_netcdf(path, canonical_names, convection, date_str, resolution):
    """Validate variables, times, levels, and the requested representation."""
    import numpy as np
    import xarray as xr

    ds = None
    try:
        ds = xr.open_dataset(path)
        missing = set(canonical_names) - set(ds.data_vars)
        if missing:
            raise ValueError(f"missing canonical variables: {sorted(missing)}")

        level_dim = next((name for name in ("hybrid", "model_level", "level")
                          if name in ds.dims), None)
        if level_dim is None or ds.sizes[level_dim] != 137:
            raise ValueError(f"expected one 137-level model dimension; sizes={dict(ds.sizes)}")
        if level_dim in ds.coords:
            levels = set(int(value) for value in np.asarray(ds[level_dim]).reshape(-1))
            if levels != set(range(1, 138)):
                raise ValueError("model levels are not exactly 1..137")

        expected_ni, expected_nj = regular_ll_shape(resolution)
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise ValueError("missing latitude/longitude coordinates")
        latitude = np.asarray(ds["latitude"]).reshape(-1)
        longitude = np.asarray(ds["longitude"]).reshape(-1)
        if latitude.size != expected_nj or longitude.size != expected_ni:
            raise ValueError(
                f"expected regular_ll dimensions latitude={expected_nj}, "
                f"longitude={expected_ni}; got latitude={latitude.size}, "
                f"longitude={longitude.size}")
        if (not np.allclose(np.sort(latitude),
                            np.linspace(-90.0, 90.0, expected_nj),
                            rtol=0.0, atol=1e-9)
                or not np.allclose(np.sort(np.mod(longitude, 360.0)),
                                   np.arange(expected_ni) * resolution,
                                   rtol=0.0, atol=1e-9)):
            raise ValueError(
                f"latitude/longitude coordinates do not describe the requested "
                f"global {resolution}-degree regular_ll grid")

        if "valid_time" not in ds.variables:
            raise ValueError("missing valid_time coordinate")
        valid_time = np.asarray(ds["valid_time"]).astype("datetime64[ns]").reshape(-1)
        start_hour = 7 if convection else 0
        expected_valid_time = (
            np.datetime64(date_str, "ns")
            + np.arange(start_hour, start_hour + 24).astype("timedelta64[h]"))
        if (valid_time.size != 24
                or not np.array_equal(np.sort(valid_time), expected_valid_time)):
            raise ValueError(
                f"valid times do not exactly match the 24 requested hours for "
                f"{date_str}")

        expected_time_dims = {"time", "step"} if convection else {"time"}
        if convection and (ds.sizes.get("time") != 2 or ds.sizes.get("step") != 12):
            raise ValueError(
                f"expected convection layout time=2, step=12; sizes={dict(ds.sizes)}")
        if not convection and ds.sizes.get("time") != 24:
            raise ValueError(f"expected thermodynamics time=24; sizes={dict(ds.sizes)}")
        expected_attrs = {
            "GRIB_gridType": "regular_ll",
            "GRIB_typeOfLevel": "hybrid",
            "GRIB_dataType": "fc" if convection else "an",
            "GRIB_stepType": "avg" if convection else "instant",
        }
        latitude_dim = ds["latitude"].dims[0]
        longitude_dim = ds["longitude"].dims[0]
        for name in canonical_names:
            dims = set(ds[name].dims)
            required_dims = expected_time_dims | {
                level_dim, latitude_dim, longitude_dim}
            if not required_dims.issubset(dims):
                raise ValueError(f"variable {name} has incomplete dimensions {ds[name].dims}")
            for attr, wanted in expected_attrs.items():
                actual = ds[name].attrs.get(attr)
                if actual != wanted:
                    raise ValueError(
                        f"variable {name} has {attr}={actual!r}; expected {wanted!r}")
        return True
    except Exception as exc:
        print(f"    INVALID NetCDF {os.path.basename(path)}: {exc}")
        return False
    finally:
        if ds is not None:
            ds.close()


def grib_to_netcdf_by_param(grib_path, nc_path, param_ids, canonical_names,
                            convection, date_str, resolution):
    """Convert GRIB to NetCDF, extracting variables by paramId and renaming.

    Each paramId is opened separately via filter_by_keys to handle cases
    where cfgrib merges different parameters or assigns unexpected names.
    Variables are renamed to canonical names (udmf, ddmf, etc.).
    """
    import xarray as xr

    datasets = []
    part_path = nc_path + ".part"
    if os.path.exists(part_path):
        os.remove(part_path)
    merged = None
    try:
        for pid, cname in zip(param_ids, canonical_names):
            ds = xr.open_dataset(
                grib_path, engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"paramId": pid}})
            data_vars = list(ds.data_vars)
            if len(data_vars) != 1:
                ds.close()
                raise RuntimeError(
                    f"paramId {pid} ({cname}) produced {len(data_vars)} data variables")
            if data_vars[0] != cname:
                ds = ds.rename({data_vars[0]: cname})
            datasets.append(ds)

        merged = xr.merge(datasets)
        if set(merged.data_vars) != set(canonical_names):
            raise RuntimeError(
                f"extracted variables {sorted(merged.data_vars)} do not match "
                f"requested {sorted(canonical_names)}")
        encoding = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
        merged.to_netcdf(part_path, encoding=encoding)
        if not validate_netcdf(part_path, canonical_names, convection,
                               date_str, resolution):
            raise RuntimeError("temporary NetCDF failed completeness validation")
        os.replace(part_path, nc_path)
    except Exception:
        if os.path.exists(part_path):
            os.remove(part_path)
        raise
    finally:
        if merged is not None:
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

    canonical_names = list(CONVECTION_PARAMS.values())
    if (os.path.isfile(nc_path)
            and validate_netcdf(nc_path, canonical_names, True,
                                date_str, resolution)):
        sz = os.path.getsize(nc_path) / 1e6
        print(f"  SKIP convection (validated, {sz:.1f} MB): {os.path.basename(nc_path)}")
        return

    print(f"  Downloading/validating convection fields for {date_str}...")
    request = {
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
    }
    expected = expected_grib_inventory(date_str, CONVECTION_PARAMS, True)
    retrieve_grib_atomic(client, request, grib_path, expected,
                         f"convection {date_str}", resolution, True)
    sz = os.path.getsize(grib_path) / 1e6
    print(f"    GRIB validated: {os.path.basename(grib_path)} ({sz:.1f} MB)")

    grib_to_netcdf_by_param(
        grib_path, nc_path,
        list(CONVECTION_PARAMS.keys()),
        canonical_names, True, date_str, resolution)

    if not keep_grib and os.path.exists(grib_path):
        os.remove(grib_path)


def download_thermodynamics(client, date_str, resolution, outdir, keep_grib):
    """Download hourly T + Q on model levels for one day."""
    nc_path = os.path.join(outdir, f"era5_thermo_ml_{date_str.replace('-', '')}.nc")
    grib_path = nc_path.replace(".nc", ".grib")

    canonical_names = list(THERMO_PARAMS.values())
    if (os.path.isfile(nc_path)
            and validate_netcdf(nc_path, canonical_names, False,
                                date_str, resolution)):
        sz = os.path.getsize(nc_path) / 1e6
        print(f"  SKIP thermodynamics (validated, {sz:.1f} MB): {os.path.basename(nc_path)}")
        return

    print(f"  Downloading/validating T + Q for {date_str}...")
    request = {
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
    }
    expected = expected_grib_inventory(date_str, THERMO_PARAMS, False)
    retrieve_grib_atomic(client, request, grib_path, expected,
                         f"thermodynamics {date_str}", resolution, False)
    sz = os.path.getsize(grib_path) / 1e6
    print(f"    GRIB validated: {os.path.basename(grib_path)} ({sz:.1f} MB)")

    grib_to_netcdf_by_param(
        grib_path, nc_path,
        list(THERMO_PARAMS.keys()),
        canonical_names, False, date_str, resolution)

    if not keep_grib and os.path.exists(grib_path):
        os.remove(grib_path)


def output_group_available(outdir, dates, resolution, convection):
    """Return whether every requested day has a valid output for one group."""
    if convection:
        prefix = "era5_convection_"
        canonical_names = list(CONVECTION_PARAMS.values())
    else:
        prefix = "era5_thermo_ml_"
        canonical_names = list(THERMO_PARAMS.values())
    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        path = os.path.join(outdir, f"{prefix}{dt.strftime('%Y%m%d')}.nc")
        if (not os.path.isfile(path)
                or not validate_netcdf(path, canonical_names, convection,
                                       date_str, resolution)):
            return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    if end_date < start:
        print("ERROR: --end must be on or after --start", file=sys.stderr)
        return 2
    if args.resolution <= 0:
        print("ERROR: --resolution must be positive", file=sys.stderr)
        return 2
    try:
        regular_ll_shape(args.resolution)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
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
    for pkg in ["cdsapi", "cfgrib", "eccodes", "xarray", "netCDF4"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"ERROR: {pkg} not installed. Run: pip install cdsapi cfgrib eccodes xarray netcdf4")
            return 1

    import cdsapi

    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        print("ERROR: ~/.cdsapirc not found. Configure CDS API credentials.")
        return 1

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
    failures = []
    for i, dt in enumerate(date_range(start, end_date)):
        date_str = dt.strftime("%Y-%m-%d")
        print(f"[{i+1}/{n_days}] {date_str}")

        if do_conv:
            try:
                download_convection(client, date_str, args.resolution, outdir,
                                    args.keep_grib)
            except Exception as exc:
                failures.append(f"{date_str} convection: {exc}")
                print(f"  ERROR convection {date_str}: {exc}", file=sys.stderr)
        if do_thermo:
            try:
                download_thermodynamics(client, date_str, args.resolution, outdir,
                                        args.keep_grib)
            except Exception as exc:
                failures.append(f"{date_str} thermodynamics: {exc}")
                print(f"  ERROR thermodynamics {date_str}: {exc}", file=sys.stderr)

    if failures:
        print(f"\nFAILED: {len(failures)} field-group download(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\nDone! Downloaded and validated {n_days} days to {outdir}")
    dates = list(date_range(start, end_date))
    convection_available = output_group_available(
        outdir, dates, args.resolution, True)
    thermodynamics_available = output_group_available(
        outdir, dates, args.resolution, False)
    if convection_available and thermodynamics_available:
        print("\nNext steps for convection:")
        print("  julia --project=. scripts/preprocessing/convert_era5_physics_nc_to_bin.jl \\")
        print(f"      --nc-dir {outdir} --bin-dir <BIN_DIR> \\")
        print(f"      --start {args.start} --end {args.end}")
        print("  The converter also needs the convection download for the day before --start.")
        print("  Then set [convection] type = \"tm5\" in your run config.")
    elif convection_available:
        print("\nThermodynamics are required before running the convection converter:")
        print("  python3 scripts/downloads/download_era5_physics.py \\")
        print(f"      --start {args.start} --end {args.end} \\")
        print(f"      --outdir {outdir} --resolution {args.resolution} \\")
        print("      --fields thermodynamics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
