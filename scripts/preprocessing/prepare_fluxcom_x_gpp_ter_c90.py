#!/usr/bin/env python3
"""Prepare hourly C90 atmospheric CO2 fluxes from FLUXCOM-X GPP and NEE.

FLUXCOM-X publishes GPP and NEE.  Ecosystem respiration is diagnosed as
TER = GPP + NEE, using the convention that GPP is a positive uptake magnitude
and NEE is positive from the ecosystem to the atmosphere.  Small unphysical
negative XGBoost predictions are clipped before temporal disaggregation.

The native 0.25-degree carbon totals are assigned to the nearest C90 cell
centre with exact source-cell area weights.  Dividing by spherical C90 cell
areas makes the remap globally conservative when the runtime later multiplies
the flux density by its C90 cell areas.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from scipy.spatial import cKDTree


R_EARTH = 6_371_000.0
KG_CO2_PER_G_C = 1.0e-3 * 44.0 / 12.0


def xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg))
    lon = np.deg2rad(np.asarray(lon_deg))
    c = np.cos(lat)
    return np.column_stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)))


def fine_to_c90_map(lat: np.ndarray, lon: np.ndarray,
                    target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    tree = cKDTree(xyz(target_lat.ravel(), target_lon.ravel()))
    mapping = np.empty(lat.size * lon.size, dtype=np.int32)
    for start in range(0, mapping.size, 200_000):
        stop = min(start + 200_000, mapping.size)
        flat = np.arange(start, stop)
        ii, jj = flat // lon.size, flat % lon.size
        mapping[start:stop] = tree.query(xyz(lat[ii], lon[jj]), k=1)[1]
    return mapping


def spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    numerator = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    denominator = 1.0 + np.einsum("...i,...i->...", a, b)
    denominator += np.einsum("...i,...i->...", b, c)
    denominator += np.einsum("...i,...i->...", c, a)
    return 2.0 * np.arctan2(numerator, denominator) * R_EARTH**2


def c90_cell_areas(corner_lat: np.ndarray, corner_lon: np.ndarray) -> np.ndarray:
    corners = xyz(corner_lat.ravel(), corner_lon.ravel()).reshape(corner_lat.shape + (3,))
    q00 = corners[:, :-1, :-1]
    q10 = corners[:, :-1, 1:]
    q11 = corners[:, 1:, 1:]
    q01 = corners[:, 1:, :-1]
    return spherical_triangle_area(q00, q10, q11) + spherical_triangle_area(q00, q11, q01)


def source_cell_areas(lat_bounds: np.ndarray, lon_bounds: np.ndarray) -> np.ndarray:
    lat_factor = np.abs(np.sin(np.deg2rad(lat_bounds[:, 1])) -
                        np.sin(np.deg2rad(lat_bounds[:, 0])))
    lon_width = np.abs(np.deg2rad(lon_bounds[:, 1] - lon_bounds[:, 0]))
    return R_EARTH**2 * lat_factor[:, None] * lon_width[None, :]


def aggregate_density(field: np.ndarray, valid_index: np.ndarray,
                      mapping: np.ndarray, mass_weight: np.ndarray,
                      target_area: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64).ravel()[valid_index]
    good = np.isfinite(values)
    total = np.bincount(mapping[good], weights=values[good] * mass_weight[good],
                        minlength=target_area.size)
    return total / target_area


def solar_weights(lat: np.ndarray, lon: np.ndarray, date: datetime) -> np.ndarray:
    """Simple positive cosine-zenith fallback, returned as (24, ncell)."""
    doy = date.timetuple().tm_yday
    decl = np.deg2rad(23.44 * np.sin(2.0 * np.pi * (284 + doy) / 365.0))
    lat_rad = np.deg2rad(lat)
    out = np.empty((24, lat.size), dtype=np.float64)
    for hour in range(24):
        local_hour = hour + lon / 15.0
        hour_angle = np.deg2rad(15.0 * (local_hour - 12.0))
        out[hour] = np.maximum(
            np.sin(lat_rad) * np.sin(decl) +
            np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle), 0.0)
    total = out.sum(axis=0)
    return np.divide(out, total, out=np.full_like(out, 1.0 / 24.0), where=total > 0)


def normalize_cycle(cycle: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    total = cycle.sum(axis=0)
    frac = np.divide(cycle, total, out=np.zeros_like(cycle), where=total > 0)
    frac[:, total <= 0] = fallback[:, total <= 0]
    return frac


def check_source_compatibility(gpp: Dataset, nee: Dataset) -> None:
    for name in ("lat", "lon", "lat_bnds", "lon_bnds", "time"):
        if not np.array_equal(gpp[name][:], nee[name][:]):
            raise ValueError(f"GPP and NEE {name} coordinates differ")
    if not np.array_equal(gpp["land_fraction"][:], nee["land_fraction"][:]):
        raise ValueError("GPP and NEE land_fraction fields differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True,
                        help="C90 NetCDF containing lons/lats and corner_lons/corner_lats")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    paths = {
        "gpp_daily": args.input_dir / "GPP_2021_025_daily.nc",
        "gpp_cycle": args.input_dir / "GPP_2021_025_monthlycycle.nc",
        "nee_daily": args.input_dir / "NEE_2021_025_daily.nc",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing FLUXCOM-X inputs: " + ", ".join(missing))

    with Dataset(args.geometry) as geometry:
        target_lon = np.asarray(geometry["lons"][:], dtype=np.float64)
        target_lat = np.asarray(geometry["lats"][:], dtype=np.float64)
        target_area_3d = c90_cell_areas(
            np.asarray(geometry["corner_lats"][:], dtype=np.float64),
            np.asarray(geometry["corner_lons"][:], dtype=np.float64),
        )
    if target_lat.shape != (6, 90, 90) or target_area_3d.shape != target_lat.shape:
        raise ValueError(f"expected C90 geometry, got centres {target_lat.shape} and areas {target_area_3d.shape}")
    sphere_area = 4.0 * np.pi * R_EARTH**2
    if not np.isclose(target_area_3d.sum(), sphere_area, rtol=2e-6):
        raise ValueError("C90 spherical cell areas do not close to 4*pi*R^2")
    target_area = target_area_3d.ravel()
    ncell = target_area.size

    with Dataset(paths["gpp_daily"]) as gpp, Dataset(paths["nee_daily"]) as nee:
        check_source_compatibility(gpp, nee)
        lat = np.asarray(gpp["lat"][:], dtype=np.float64)
        lon = np.asarray(gpp["lon"][:], dtype=np.float64)
        source_area = source_cell_areas(gpp["lat_bnds"][:], gpp["lon_bnds"][:])
        land_fraction = np.asarray(gpp["land_fraction"][:], dtype=np.float64)
        valid = np.isfinite(land_fraction) & (land_fraction > 0)
        valid_index = np.flatnonzero(valid.ravel())
        mass_weight = (source_area * np.nan_to_num(land_fraction)).ravel()[valid.ravel()]
        mapping_all = fine_to_c90_map(lat, lon, target_lat, target_lon)
        mapping = mapping_all[valid_index]

        times = num2date(gpp["time"][:], gpp["time"].units,
                        calendar=getattr(gpp["time"], "calendar", "standard"))
        dates = [datetime(t.year, t.month, t.day, tzinfo=timezone.utc) for t in times]
        if len(dates) != 365 or dates[0].date().isoformat() != "2021-01-01" or dates[-1].date().isoformat() != "2021-12-31":
            raise ValueError("daily files do not contain exactly calendar year 2021")

        daily_gpp = np.empty((365, ncell), dtype=np.float32)
        daily_ter = np.empty_like(daily_gpp)
        negative = {"gpp_daily": 0, "ter_daily": 0}
        samples = 0
        # The source is chunked 73 days at a time. Reading matching blocks
        # avoids decompressing the same large HDF5 chunk once per day.
        for block_start in range(0, 365, 73):
            block_stop = min(block_start + 73, 365)
            gpp_block = np.asarray(gpp["GPP"][block_start:block_stop], dtype=np.float32)
            nee_block = np.asarray(nee["NEE"][block_start:block_stop], dtype=np.float32)
            for local_day, day in enumerate(range(block_start, block_stop)):
                raw_gpp = np.asarray(gpp_block[local_day], dtype=np.float64)
                raw_nee = np.asarray(nee_block[local_day], dtype=np.float64)
                negative["gpp_daily"] += int(np.count_nonzero(raw_gpp[valid] < 0))
                physical_gpp = np.maximum(raw_gpp, 0.0)
                raw_ter = physical_gpp + raw_nee
                negative["ter_daily"] += int(np.count_nonzero(raw_ter[valid] < 0))
                samples += valid_index.size
                daily_gpp[day] = aggregate_density(
                    physical_gpp, valid_index, mapping, mass_weight, target_area)
                daily_ter[day] = aggregate_density(
                    np.maximum(raw_ter, 0.0), valid_index, mapping, mass_weight, target_area)
            print(f"daily remap {block_stop:3d}/365", flush=True)

    with Dataset(paths["gpp_cycle"]) as gpp:
        cycle_gpp = np.empty((12, 24, ncell), dtype=np.float32)
        negative.update({"gpp_cycle": 0})
        cycle_samples = 0
        # This source is chunked three months at a time for the leading axis.
        for block_start in range(0, 12, 3):
            block_stop = block_start + 3
            cycle_block = np.asarray(gpp["GPP"][block_start:block_stop], dtype=np.float32)
            for local_month, month in enumerate(range(block_start, block_stop)):
                for hour in range(24):
                    raw_gpp = np.asarray(cycle_block[local_month, hour], dtype=np.float64)
                    negative["gpp_cycle"] += int(np.count_nonzero(raw_gpp[valid] < 0))
                    physical_gpp = np.maximum(raw_gpp, 0.0)
                    cycle_samples += valid_index.size
                    cycle_gpp[month, hour] = aggregate_density(
                        physical_gpp, valid_index, mapping, mass_weight, target_area)
            print(f"diurnal remap {block_stop:2d}/12", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(args.output, "w", format="NETCDF4") as out:
        out.createDimension("time", 8760)
        out.createDimension("nf", 6)
        out.createDimension("Ydim", 90)
        out.createDimension("Xdim", 90)
        out.Conventions = "CF-1.8"
        out.title = "Hourly FLUXCOM-X X-BASE GPP and diagnosed TER atmospheric CO2 fluxes on C90"
        out.source = "FLUXCOM-X X-BASE 1.0 GPP and NEE, 0.25 degree daily and monthly diurnal products"
        out.derivation = "TER = max(0, max(0,GPP) + NEE); GPP uses its clipped monthly diurnal cycle; TER is a daily mean held constant for 24 UTC hours"
        out.remapping = "source-cell area and land-fraction weighted nearest-C90 assignment; globally mass conservative"
        out.sign_convention = "negative is atmospheric uptake; positive is emission to atmosphere"
        tvar = out.createVariable("time", "f8", ("time",))
        tvar.units = "hours since 2021-01-01 00:00:00 UTC"
        tvar.calendar = "proleptic_gregorian"
        tvar[:] = np.arange(8760, dtype=np.float64)
        out.createVariable("lons", "f8", ("nf", "Ydim", "Xdim"))[:] = target_lon
        out.createVariable("lats", "f8", ("nf", "Ydim", "Xdim"))[:] = target_lat
        out.createVariable("cell_area", "f8", ("nf", "Ydim", "Xdim"))[:] = target_area_3d
        gpp_var = out.createVariable(
            "GPP_CO2_FLUX", "f4", ("time", "nf", "Ydim", "Xdim"),
            zlib=True, complevel=4, shuffle=True, chunksizes=(24, 1, 90, 90))
        ter_var = out.createVariable(
            "TER_CO2_FLUX", "f4", ("time", "nf", "Ydim", "Xdim"),
            zlib=True, complevel=4, shuffle=True, chunksizes=(24, 1, 90, 90))
        for var, long_name in (
            (gpp_var, "gross primary production atmospheric CO2 uptake per total grid-cell area"),
            (ter_var, "diagnosed total ecosystem respiration CO2 emission per total grid-cell area"),
        ):
            var.units = "kg CO2 m-2 s-1"
            var.long_name = long_name
            var.positive = "to_atmosphere"

        flat_lat, flat_lon = target_lat.ravel(), target_lon.ravel()
        for day, date in enumerate(dates):
            fallback_gpp = solar_weights(flat_lat, flat_lon, date)
            gpp_fraction = normalize_cycle(cycle_gpp[date.month - 1], fallback_gpp)
            gpp_flux = -daily_gpp[day][None, :] * gpp_fraction * KG_CO2_PER_G_C / 3600.0
            ter_flux = np.broadcast_to(
                daily_ter[day][None, :] * KG_CO2_PER_G_C / 86400.0,
                (24, ncell),
            )
            sl = slice(day * 24, (day + 1) * 24)
            gpp_var[sl] = gpp_flux.reshape(24, 6, 90, 90).astype(np.float32)
            ter_var[sl] = ter_flux.reshape(24, 6, 90, 90).astype(np.float32)
            if day % 30 == 0 or day == 364:
                print(f"hourly write {day + 1:3d}/365", flush=True)

    annual_gpp_pg_c = float(np.sum(daily_gpp.astype(np.float64) * target_area) * 1.0e-15)
    annual_ter_pg_c = float(np.sum(daily_ter.astype(np.float64) * target_area) * 1.0e-15)
    summary = {
        "output": str(args.output),
        "records": 8760,
        "grid": "C90 GEOS-native, 6x90x90",
        "annual_gpp_pg_c": annual_gpp_pg_c,
        "annual_ter_pg_c": annual_ter_pg_c,
        "implied_nee_pg_c": annual_ter_pg_c - annual_gpp_pg_c,
        "source_negative_fraction": {
            "gpp_daily": negative["gpp_daily"] / samples,
            "ter_daily_before_clip": negative["ter_daily"] / samples,
            "gpp_monthly_diurnal": negative["gpp_cycle"] / cycle_samples,
        },
        "c90_area_relative_closure": float(target_area.sum() / sphere_area - 1.0),
    }
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
