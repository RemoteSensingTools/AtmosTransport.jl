#!/usr/bin/env python3
"""Prepare 2021 hourly C90 GPP and TER flux drivers from TRENDYv14 S3.

Each model's monthly GPP, autotrophic respiration (ra), and heterotrophic
respiration (rh) are conservatively remapped to GEOS-native C90.  Smooth daily
means are obtained by linear interpolation between monthly midpoints followed
by a cell-wise monthly renormalization, so every native monthly carbon total is
preserved.  GPP uses the established FLUXCOM-X monthly C90 diurnal fractions;
TER = ra + rh is held constant during each UTC day.

The script also understands the .nc.gz and .nc.tar.gz containers present in
the local TRENDY archive and stages one source variable at a time.
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import gzip
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


R_EARTH = 6_371_000.0
KG_CO2_PER_KG_C = 44.0 / 12.0
SECONDS_PER_DAY = 86_400.0
DEFAULT_ROOT = Path("/kiwi-data/Data/model/TRENDYv14/S3")
DEFAULT_GEOMETRY = Path(
    "/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/"
    "sif_gpp_weekly_diurnal_c90_2021.nc"
)
DEFAULT_OUTPUT = Path(
    "/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/"
    "TRENDYv14/S3/C90/2021"
)

# These model archives document grid-box fluxes per unit land area. Their
# static land fraction is therefore applied before conservative remapping.
# LPJ-GUESS explicitly documents its totals as per grid-cell area and is not
# included here.
LAND_FRACTION_FILES = {
    "CABLE-POP": "oceanCoverFrac",
    "CARDAMOM": "embedded_land_fraction",
    "CLASSIC": "land_fraction",
    "CLM": "oceanCoverFrac",
    "ELM-FATES": "oceancoverfrac",
    "IBIS": "oceanCoverFrac",
    "LPJml": "oceanCoverFrac",
    "ORCHIDEE": "oceanCoverFrac",
    "TEM": "oceanCoverFrac",
}


def xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg))
    lon = np.deg2rad(np.asarray(lon_deg))
    coslat = np.cos(lat)
    return np.column_stack((coslat * np.cos(lon), coslat * np.sin(lon), np.sin(lat)))


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


def solar_diurnal_fraction(lat: np.ndarray, lon: np.ndarray, date: datetime) -> np.ndarray:
    """Positive cosine-zenith GPP fallback, normalized over 24 UTC hours."""
    day_of_year = date.timetuple().tm_yday
    declination = np.deg2rad(
        23.44 * np.sin(2.0 * np.pi * (284 + day_of_year) / 365.0)
    )
    latitude = np.deg2rad(lat)
    weights = np.empty((24, lat.size), dtype=np.float64)
    for hour in range(24):
        local_hour = hour + lon / 15.0
        hour_angle = np.deg2rad(15.0 * (local_hour - 12.0))
        weights[hour] = np.maximum(
            np.sin(latitude) * np.sin(declination)
            + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle),
            0.0,
        )
    total = weights.sum(axis=0)
    return np.divide(
        weights, total,
        out=np.full_like(weights, 1.0 / 24.0), where=total > 0,
    )


def coordinate(ds: netCDF4.Dataset, axis: str, expected_size: int) -> np.ndarray:
    aliases = {"lat": ("lat", "latitude"), "lon": ("lon", "longitude")}[axis]
    standard = {"lat": "latitude", "lon": "longitude"}[axis]
    candidates = []
    for name, var in ds.variables.items():
        if var.ndim != 1 or var.size != expected_size:
            continue
        if name.lower() in aliases or getattr(var, "standard_name", "").lower() == standard:
            candidates.append(var)
    if not candidates:
        raise ValueError(f"cannot identify {axis} coordinate of length {expected_size}")
    return np.asarray(candidates[0][:], dtype=np.float64)


def regular_cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Spherical Voronoi areas for a regular latitude-longitude grid."""
    lat_order = np.argsort(lat)
    sorted_lat = lat[lat_order]
    lat_edge = np.empty(sorted_lat.size + 1)
    lat_edge[1:-1] = 0.5 * (sorted_lat[:-1] + sorted_lat[1:])
    lat_edge[0], lat_edge[-1] = -90.0, 90.0

    wrapped_lon = (lon + 180.0) % 360.0 - 180.0
    lon_order = np.argsort(wrapped_lon)
    sorted_lon = wrapped_lon[lon_order]
    extended = np.r_[sorted_lon[-1] - 360.0, sorted_lon, sorted_lon[0] + 360.0]
    lon_width = 0.5 * (extended[2:] - extended[:-2])
    sorted_area = (
        R_EARTH**2
        * np.abs(
            np.sin(np.deg2rad(lat_edge[1:])) - np.sin(np.deg2rad(lat_edge[:-1]))
        )[:, None]
        * np.deg2rad(lon_width)[None, :]
    )
    inverse_lat = np.argsort(lat_order)
    inverse_lon = np.argsort(lon_order)
    return sorted_area[inverse_lat][:, inverse_lon]


def fine_grid_target_map(target_lat: np.ndarray, target_lon: np.ndarray,
                         resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlat = int(round(180.0 / resolution))
    nlon = int(round(360.0 / resolution))
    fine_lat_edge = np.linspace(-90.0, 90.0, nlat + 1)
    fine_lon_edge = np.linspace(-180.0, 180.0, nlon + 1)
    fine_lat = 0.5 * (fine_lat_edge[:-1] + fine_lat_edge[1:])
    fine_lon = 0.5 * (fine_lon_edge[:-1] + fine_lon_edge[1:])
    fine_area_lat = (
        R_EARTH**2
        * np.abs(
            np.sin(np.deg2rad(fine_lat_edge[1:]))
            - np.sin(np.deg2rad(fine_lat_edge[:-1]))
        )
        * np.deg2rad(resolution)
    )
    tree = cKDTree(xyz(target_lat.ravel(), target_lon.ravel()))
    target = np.empty(nlat * nlon, dtype=np.int32)
    for start in range(0, target.size, 200_000):
        stop = min(start + 200_000, target.size)
        flat = np.arange(start, stop)
        ii, jj = flat // nlon, flat % nlon
        target[start:stop] = tree.query(xyz(fine_lat[ii], fine_lon[jj]), k=1)[1]
    fine_area = np.broadcast_to(fine_area_lat[:, None], (nlat, nlon)).ravel().copy()
    return fine_lat, fine_lon, target, fine_area


def conservative_matrix(lat: np.ndarray, lon: np.ndarray, target_index: np.ndarray,
                        fine_lat: np.ndarray, fine_lon: np.ndarray,
                        fine_area: np.ndarray, n_target: int) -> tuple[csr_matrix, np.ndarray]:
    """Approximate overlap at 0.25 degree and normalize every source column exactly."""
    lat_order = np.argsort(lat)
    sorted_lat = lat[lat_order]
    lat_mid = 0.5 * (sorted_lat[:-1] + sorted_lat[1:])
    fine_lat_sorted_index = np.searchsorted(lat_mid, fine_lat)
    fine_lat_source = lat_order[fine_lat_sorted_index]

    wrapped_lon = (lon + 180.0) % 360.0 - 180.0
    # Nearest longitude on a circle is robust to both 0..360 coordinates and
    # wrapped coordinate sequences such as 0..177.5,-180..-2.5.
    distance = np.abs(fine_lon[:, None] - wrapped_lon[None, :])
    distance = np.minimum(distance, 360.0 - distance)
    fine_lon_source = np.argmin(distance, axis=1)

    source_index = (
        fine_lat_source[:, None] * lon.size + fine_lon_source[None, :]
    ).ravel().astype(np.int32)
    source_area = regular_cell_areas(lat, lon).ravel()
    sampled_area = np.bincount(source_index, weights=fine_area, minlength=source_area.size)
    if np.any(sampled_area <= 0):
        raise ValueError("0.25-degree overlap sampling missed a source cell")
    weights = fine_area * (source_area / sampled_area)[source_index]
    matrix = csr_matrix(
        (weights, (target_index, source_index)),
        shape=(n_target, source_area.size),
    )
    column_area = np.asarray(matrix.sum(axis=0)).ravel()
    if not np.allclose(column_area, source_area, rtol=2e-12, atol=1e-3):
        raise ValueError("conservative remap columns do not close to source cell areas")
    return matrix, source_area


def container_stem(path: Path) -> str:
    name = path.name
    for suffix in (".nc.tar.gz", ".nc.gz", ".nc"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def find_flux_file(model_dir: Path, kind: str) -> Path | None:
    candidates = []
    for path in model_dir.rglob("*"):
        if not path.is_file() or not any(
            path.name.endswith(suffix) for suffix in (".nc", ".nc.gz", ".nc.tar.gz")
        ):
            continue
        stem = container_stem(path).lower()
        if "mean_annual" not in stem and stem.split("_")[-1] == kind.lower():
            candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"multiple {kind} files under {model_dir}: {candidates}")
    return candidates[0] if candidates else None


@contextlib.contextmanager
def materialized(path: Path, staging_dir: Path):
    """Yield a readable NetCDF path, expanding an archive only when needed."""
    if path.name.endswith(".nc"):
        yield path
        return
    staging_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:10]
    temporary = staging_dir / f"{container_stem(path)}_{digest}.nc"
    try:
        if path.name.endswith(".nc.gz") and not path.name.endswith(".nc.tar.gz"):
            with gzip.open(path, "rb") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        elif path.name.endswith(".nc.tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".nc")]
                if len(members) != 1:
                    raise ValueError(f"expected one NetCDF member in {path}, found {len(members)}")
                source = archive.extractfile(members[0])
                if source is None:
                    raise ValueError(f"cannot extract {members[0].name} from {path}")
                with source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        else:
            raise ValueError(f"unsupported input container: {path}")
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def month_keys(ds: netCDF4.Dataset) -> list[tuple[int, int]]:
    time = ds["time"]
    values = np.asarray(time[:], dtype=np.float64)
    units = getattr(time, "units", "")
    cal = getattr(time, "calendar", "standard")
    try:
        dates = netCDF4.num2date(
            values, units, calendar=cal, only_use_cftime_datetimes=True
        )
        return [(int(value.year), int(value.month)) for value in dates]
    except Exception:
        pass

    match = re.search(r"months\s+since\s+(\d{4})-(\d{1,2})", units, re.I)
    if match:
        origin = int(match.group(1)) * 12 + int(match.group(2)) - 1
        month_number = origin + np.floor(values + 1e-6).astype(int)
        return [(int(value // 12), int(value % 12 + 1)) for value in month_number]

    if "years since" in units.lower() or (
        not units and values.size and 1600.0 < values[0] < 2200.0
    ):
        years = np.floor(values).astype(int)
        months = np.floor((values - years) * 12.0 + 1e-5).astype(int) + 1
        months = np.clip(months, 1, 12)
        return list(zip(years.tolist(), months.tolist()))

    # CARDAMOM stores the useful epoch only in the auxiliary Time long_name.
    if "Time" in ds.variables:
        note = getattr(ds["Time"], "long_name", "")
        match = re.search(r"since\s+\d{1,2}/\d{1,2}/(\d{4})", note, re.I)
        if match:
            start_year = int(match.group(1))
            return [
                (start_year + index // 12, index % 12 + 1)
                for index in range(values.size)
            ]

    if values.size % 12 == 0:
        start_year = 2025 - values.size // 12
        return [(start_year + index // 12, index % 12 + 1) for index in range(values.size)]
    raise ValueError(f"cannot decode monthly time axis: units={units!r}, n={values.size}")


def annual_keys(ds: netCDF4.Dataset) -> list[int]:
    time = ds["time"]
    try:
        dates = netCDF4.num2date(
            time[:], time.units,
            calendar=getattr(time, "calendar", "standard"),
            only_use_cftime_datetimes=True,
        )
        return [int(value.year) for value in dates]
    except Exception:
        if time.size == 325:
            return list(range(1700, 2025))
        raise


def flux_variable(ds: netCDF4.Dataset, kind: str) -> netCDF4.Variable:
    for name, var in ds.variables.items():
        if name.lower() == kind.lower():
            return var
    raise KeyError(f"variable {kind!r} not found")


def validate_units(var: netCDF4.Variable) -> str:
    units = getattr(var, "units", "")
    normalized = (
        units.lower().replace(".", " ").replace("$", "")
        .replace("{", "").replace("}", "").replace("^", "")
    )
    if "kg" not in normalized or "m-2" not in normalized or "s-1" not in normalized:
        raise ValueError(f"unsupported units for {var.name}: {units!r}")
    return units


def read_months(ds: netCDF4.Dataset, kind: str,
                requested: list[tuple[int, int]]) -> tuple[np.ndarray, str, float, float]:
    var = flux_variable(ds, kind)
    units = validate_units(var)
    keys = month_keys(ds)
    lookup = {key: index for index, key in enumerate(keys)}
    missing = [key for key in requested if key not in lookup]
    if missing:
        raise ValueError(f"{kind} lacks requested months: {missing}")
    indices = [lookup[key] for key in requested]
    raw = np.ma.filled(var[indices], np.nan).astype(np.float64)
    finite = np.isfinite(raw)
    negative_fraction = float(np.count_nonzero(raw[finite] < 0) / max(1, np.count_nonzero(finite)))
    minimum = float(np.min(raw[finite])) if np.any(finite) else float("nan")
    return np.maximum(np.nan_to_num(raw, nan=0.0), 0.0), units, negative_fraction, minimum


def read_annual_as_months(ds: netCDF4.Dataset, kind: str,
                          requested: list[tuple[int, int]]) -> tuple[np.ndarray, str, float, float]:
    var = flux_variable(ds, kind)
    units = validate_units(var)
    lookup = {year: index for index, year in enumerate(annual_keys(ds))}
    years = sorted({year for year, _ in requested})
    if any(year not in lookup for year in years):
        raise ValueError(f"{kind} lacks one of requested years {years}")
    annual = {}
    negative_count = finite_count = 0
    minimum = np.inf
    for year in years:
        raw = np.ma.filled(var[lookup[year]], np.nan).astype(np.float64)
        finite = np.isfinite(raw)
        finite_count += int(np.count_nonzero(finite))
        negative_count += int(np.count_nonzero(raw[finite] < 0))
        if np.any(finite):
            minimum = min(minimum, float(np.min(raw[finite])))
        annual[year] = np.maximum(np.nan_to_num(raw, nan=0.0), 0.0)
    return (
        np.stack([annual[year] for year, _ in requested]),
        units,
        negative_count / max(1, finite_count),
        float(minimum),
    )


def find_fraction_file(model_dir: Path, token: str) -> Path | None:
    token = token.lower()
    candidates = [
        path for path in model_dir.rglob("*")
        if path.is_file() and token in container_stem(path).lower()
        and any(path.name.endswith(suffix) for suffix in (".nc", ".nc.gz", ".nc.tar.gz"))
    ]
    return sorted(candidates)[0] if candidates else None


def read_land_fraction(model: str, model_dir: Path, gpp_ds: netCDF4.Dataset,
                       source_shape: tuple[int, int], staging: Path) -> tuple[np.ndarray, str]:
    specification = LAND_FRACTION_FILES.get(model)
    if specification is None:
        return np.ones(source_shape, dtype=np.float64), "none; flux treated as per total grid-cell area"
    if specification == "embedded_land_fraction":
        raw = np.ma.filled(gpp_ds["land_fraction"][:], 0.0).astype(np.float64)
        return np.clip(raw, 0.0, 1.0), "embedded land_fraction"
    path = find_fraction_file(model_dir, specification)
    if path is None:
        raise FileNotFoundError(f"{model}: required land-fraction file containing {specification!r} not found")
    with materialized(path, staging) as readable, netCDF4.Dataset(readable) as ds:
        candidates = [
            var for name, var in ds.variables.items()
            if specification.lower() in name.lower() or (
                "ocean" in specification.lower() and "ocean" in name.lower()
            )
        ]
        if not candidates:
            candidates = [
                var for var in ds.variables.values()
                if var.ndim >= 2 and var.shape[-2:] == source_shape
            ]
        if not candidates:
            raise ValueError(f"{model}: cannot identify land-fraction variable in {path}")
        var = candidates[-1]
        # Some submissions incorrectly declare zero ocean fraction as a
        # missing_value, even though zero means fully land. Honor _FillValue
        # but deliberately ignore that invalid missing_value attribute.
        var.set_auto_mask(False)
        raw = np.asarray(var[:], dtype=np.float64)
        fill_value = getattr(var, "_FillValue", None)
        if fill_value is not None:
            raw[raw == float(fill_value)] = np.nan
        while raw.ndim > 2:
            raw = raw[-1]
        if raw.shape != source_shape:
            raise ValueError(f"{model}: fraction shape {raw.shape} != source shape {source_shape}")
        if "ocean" in specification.lower() or "ocean" in var.name.lower():
            raw = 1.0 - raw
            interpretation = f"1 - {path.name}:{var.name}"
        else:
            interpretation = f"{path.name}:{var.name}"
        return np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 1.0), interpretation


def remap_months(matrix: csr_matrix, source: np.ndarray, land_fraction: np.ndarray,
                 target_area: np.ndarray) -> np.ndarray:
    weighted = source.reshape(source.shape[0], -1) * land_fraction.ravel()[None, :]
    target_mass_rate = matrix @ weighted.T
    return (np.asarray(target_mass_rate).T / target_area[None, :]).astype(np.float64)


def daily_conservative(monthly_14: np.ndarray, year: int) -> tuple[list[datetime], np.ndarray, float]:
    """Interpolate monthly midpoint rates and renormalize each month exactly."""
    keys = [(year - 1, 12)] + [(year, month) for month in range(1, 13)] + [(year + 1, 1)]
    centres = []
    for key_year, month in keys:
        days = calendar.monthrange(key_year, month)[1]
        centres.append(datetime(key_year, month, 1, tzinfo=timezone.utc) + timedelta(days=days / 2))
    origin = datetime(year, 1, 1, tzinfo=timezone.utc)
    centre_day = np.array([(value - origin).total_seconds() / SECONDS_PER_DAY for value in centres])
    dates = [origin + timedelta(days=day) for day in range(365)]
    daily_point = np.arange(365, dtype=np.float64) + 0.5
    right = np.searchsorted(centre_day, daily_point, side="right")
    left = right - 1
    alpha = (daily_point - centre_day[left]) / (centre_day[right] - centre_day[left])
    daily = (
        monthly_14[left] * (1.0 - alpha[:, None])
        + monthly_14[right] * alpha[:, None]
    )
    daily = np.maximum(daily, 0.0)

    maximum_error = 0.0
    for month in range(1, 13):
        selection = np.array([date.month == month for date in dates])
        target = monthly_14[month]
        interpolated_mean = daily[selection].mean(axis=0)
        scale = np.divide(
            target, interpolated_mean,
            out=np.zeros_like(target), where=interpolated_mean > 0,
        )
        daily[selection] *= scale
        restored = daily[selection].mean(axis=0)
        active = target > 1e-20
        if np.any(active):
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(restored[active] / target[active] - 1.0))),
            )
    return dates, daily, maximum_error


def model_inputs(model_dir: Path) -> dict[str, Path] | None:
    paths = {kind: find_flux_file(model_dir, kind) for kind in ("gpp", "ra", "rh")}
    if paths["rh"] is None and model_dir.name == "LPJ-GUESS":
        paths["rh"] = find_flux_file(model_dir, "arh")
    return None if any(path is None for path in paths.values()) else paths  # type: ignore[return-value]


def process_model(model: str, paths: dict[str, Path], args: argparse.Namespace,
                  geometry: dict[str, np.ndarray], fine: tuple[np.ndarray, ...],
                  matrix_cache: dict[str, tuple[csr_matrix, np.ndarray]]) -> dict:
    requested = [(2020, 12)] + [(2021, month) for month in range(1, 13)] + [(2022, 1)]
    staging = args.staging_dir
    model_dir = args.root / model
    print(f"[{model}] staging/reading GPP", flush=True)
    with materialized(paths["gpp"], staging) as readable, netCDF4.Dataset(readable) as ds:
        gpp_var = flux_variable(ds, "gpp")
        source_shape = tuple(gpp_var.shape[-2:])
        lat = coordinate(ds, "lat", source_shape[0])
        lon = coordinate(ds, "lon", source_shape[1])
        grid_key = hashlib.sha1(lat.tobytes() + lon.tobytes()).hexdigest()
        if grid_key not in matrix_cache:
            print(f"[{model}] building conservative {source_shape} -> C90 weights", flush=True)
            matrix_cache[grid_key] = conservative_matrix(
                lat, lon, fine[2], fine[0], fine[1], fine[3], geometry["area"].size
            )
        matrix, source_area = matrix_cache[grid_key]
        land_fraction, land_note = read_land_fraction(
            model, model_dir, ds, source_shape, staging
        )
        gpp_source, gpp_units, gpp_negative, gpp_min = read_months(ds, "gpp", requested)
    gpp_monthly = remap_months(matrix, gpp_source, land_fraction, geometry["area"])

    components = {}
    for kind in ("ra", "rh"):
        actual_kind = "arh" if kind == "rh" and container_stem(paths[kind]).lower().endswith("_arh") else kind
        print(f"[{model}] staging/reading {actual_kind}", flush=True)
        with materialized(paths[kind], staging) as readable, netCDF4.Dataset(readable) as ds:
            var = flux_variable(ds, actual_kind)
            if tuple(var.shape[-2:]) != source_shape:
                raise ValueError(f"{model}: {actual_kind} grid differs from GPP")
            if actual_kind == "arh":
                source, units, negative, minimum = read_annual_as_months(ds, actual_kind, requested)
            else:
                source, units, negative, minimum = read_months(ds, actual_kind, requested)
        components[kind] = {
            "monthly": remap_months(matrix, source, land_fraction, geometry["area"]),
            "units": units,
            "negative_fraction": negative,
            "minimum": minimum,
            "source_variable": actual_kind,
        }
    ter_monthly = components["ra"]["monthly"] + components["rh"]["monthly"]

    dates, daily_gpp, gpp_month_error = daily_conservative(gpp_monthly, 2021)
    _, daily_ter, ter_month_error = daily_conservative(ter_monthly, 2021)
    output = args.output_dir / (
        f"TRENDYv14_S3_{model}_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021.nc"
    )
    summary_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{model}] writing {output.name}", flush=True)
    with netCDF4.Dataset(output, "w", format="NETCDF4") as out:
        out.createDimension("time", 8760)
        out.createDimension("nf", 6)
        out.createDimension("Ydim", 90)
        out.createDimension("Xdim", 90)
        out.Conventions = "CF-1.8"
        out.title = f"Hourly TRENDYv14 S3 {model} GPP and TER atmospheric CO2 fluxes on C90"
        out.source = f"TRENDYv14 S3 {model}: monthly gpp, ra, and {components['rh']['source_variable']}"
        out.derivation = (
            "TER = max(0,ra) + max(0,rh); monthly midpoint interpolation followed by "
            "cell-wise monthly renormalization; GPP uses FLUXCOM-X C90 monthly diurnal "
            "fractions; TER is constant for 24 UTC hours"
        )
        out.remapping = (
            "source-cell totals conservatively distributed to C90 using normalized "
            f"{args.overlap_resolution:g}-degree overlap sampling"
        )
        out.land_fraction_treatment = land_note
        out.sign_convention = "negative is atmospheric uptake; positive is emission to atmosphere"
        time = out.createVariable("time", "f8", ("time",))
        time.units = "hours since 2021-01-01 00:00:00 UTC"
        time.calendar = "proleptic_gregorian"
        time[:] = np.arange(8760, dtype=np.float64)
        out.createVariable("lons", "f8", ("nf", "Ydim", "Xdim"))[:] = geometry["lon3"]
        out.createVariable("lats", "f8", ("nf", "Ydim", "Xdim"))[:] = geometry["lat3"]
        out.createVariable("cell_area", "f8", ("nf", "Ydim", "Xdim"))[:] = geometry["area3"]
        variable_options = dict(
            zlib=True, complevel=args.deflate_level, shuffle=True,
            chunksizes=(24, 1, 90, 90),
        )
        gpp_out = out.createVariable(
            "GPP_CO2_FLUX", "f4", ("time", "nf", "Ydim", "Xdim"), **variable_options
        )
        ter_out = out.createVariable(
            "TER_CO2_FLUX", "f4", ("time", "nf", "Ydim", "Xdim"), **variable_options
        )
        for var, long_name in (
            (gpp_out, "gross primary production atmospheric CO2 uptake per total grid-cell area"),
            (ter_out, "total ecosystem respiration CO2 emission per total grid-cell area"),
        ):
            var.units = "kg CO2 m-2 s-1"
            var.long_name = long_name
            var.positive = "to_atmosphere"

        diurnal = geometry["diurnal"]
        for day, date in enumerate(dates):
            fraction = diurnal[date.month - 1]
            gpp_flux = -daily_gpp[day][None, :] * (24.0 * fraction) * KG_CO2_PER_KG_C
            ter_flux = np.broadcast_to(
                daily_ter[day][None, :] * KG_CO2_PER_KG_C, (24, geometry["area"].size)
            )
            selection = slice(day * 24, (day + 1) * 24)
            gpp_out[selection] = gpp_flux.reshape(24, 6, 90, 90).astype(np.float32)
            ter_out[selection] = ter_flux.reshape(24, 6, 90, 90).astype(np.float32)
            if day % 60 == 0 or day == 364:
                print(f"[{model}] hourly write {day + 1:3d}/365", flush=True)

    area = geometry["area"]
    seconds = np.array(
        [calendar.monthrange(2021, month)[1] * SECONDS_PER_DAY for month in range(1, 13)]
    )
    annual_gpp = float(np.sum(gpp_monthly[1:13] * seconds[:, None] * area) / 1e12)
    annual_ra = float(np.sum(components["ra"]["monthly"][1:13] * seconds[:, None] * area) / 1e12)
    annual_rh = float(np.sum(components["rh"]["monthly"][1:13] * seconds[:, None] * area) / 1e12)
    native_gpp = float(
        np.sum(gpp_source[1:13].reshape(12, -1) * land_fraction.ravel()[None, :]
               * seconds[:, None] * source_area[None, :]) / 1e12
    )
    summary = {
        "model": model,
        "year": 2021,
        "output": str(output),
        "records": 8760,
        "grid": "GEOS-native C90, 6x90x90",
        "inputs": {kind: str(path) for kind, path in paths.items()},
        "input_units": {"gpp": gpp_units, "ra": components["ra"]["units"],
                        "rh": components["rh"]["units"]},
        "rh_source_variable": components["rh"]["source_variable"],
        "land_fraction_treatment": land_note,
        "annual_totals_PgC": {
            "gpp": annual_gpp, "ra": annual_ra, "rh": annual_rh,
            "ter": annual_ra + annual_rh,
        },
        "native_to_c90_gpp_relative_error": (
            annual_gpp / native_gpp - 1.0 if native_gpp > 0 else None
        ),
        "max_cell_monthly_mean_relative_error": {
            "gpp": gpp_month_error, "ter": ter_month_error,
        },
        "source_negative_fraction": {
            "gpp": gpp_negative,
            "ra": components["ra"]["negative_fraction"],
            "rh": components["rh"]["negative_fraction"],
        },
        "source_minimum_kgC_m-2_s-1": {
            "gpp": gpp_min,
            "ra": components["ra"]["minimum"],
            "rh": components["rh"]["minimum"],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{model}] complete: GPP={annual_gpp:.3f}, TER={annual_ra + annual_rh:.3f} Pg C", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--models", nargs="*", help="model directory names; default is all usable")
    parser.add_argument("--overlap-resolution", type=float, default=0.25)
    parser.add_argument("--deflate-level", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse an existing NetCDF+JSON pair instead of regenerating it")
    args = parser.parse_args()
    if args.staging_dir is None:
        args.staging_dir = Path(tempfile.gettempdir()) / "trendy_v14_flux_staging"
    if not (0.0 < args.overlap_resolution <= 1.0):
        parser.error("--overlap-resolution must be in (0, 1]")

    with netCDF4.Dataset(args.geometry) as ds:
        lat3 = np.asarray(ds["lats"][:], dtype=np.float64)
        lon3 = np.asarray(ds["lons"][:], dtype=np.float64)
        area3 = c90_cell_areas(ds["corner_lats"][:], ds["corner_lons"][:])
        diurnal = np.ma.filled(ds["monthly_diurnal_fraction"][:], 0.0).astype(np.float64)
    if lat3.shape != (6, 90, 90) or diurnal.shape != (12, 24, 6, 90, 90):
        raise ValueError("geometry file is not the expected C90 pilot product")
    diurnal = diurnal.reshape(12, 24, -1)
    diurnal_sum = diurnal.sum(axis=1)
    supported = diurnal_sum > 0
    if not np.allclose(diurnal_sum[supported], 1.0, rtol=2e-6, atol=2e-6):
        raise ValueError("GPP diurnal fractions do not sum to one")
    for month in range(12):
        missing = ~supported[month]
        if np.any(missing):
            fallback = solar_diurnal_fraction(
                lat3.ravel(), lon3.ravel(),
                datetime(2021, month + 1, 15, tzinfo=timezone.utc),
            )
            diurnal[month][:, missing] = fallback[:, missing]
    geometry = {
        "lat3": lat3, "lon3": lon3, "area3": area3,
        "area": area3.ravel(), "diurnal": diurnal,
    }
    sphere_area = 4.0 * np.pi * R_EARTH**2
    if not np.isclose(area3.sum(), sphere_area, rtol=2e-6):
        raise ValueError("C90 areas do not close to the spherical Earth area")

    print("Building reusable fine-grid to C90 overlap map", flush=True)
    fine = fine_grid_target_map(lat3, lon3, args.overlap_resolution)
    requested_models = set(args.models) if args.models else None
    inventory = {}
    usable = []
    for model_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        if model_dir.name == "derived_c90_fluxes":
            continue
        if requested_models is not None and model_dir.name not in requested_models:
            continue
        paths = model_inputs(model_dir)
        if paths is None:
            inventory[model_dir.name] = {"status": "skipped", "reason": "missing gpp, ra, or rh/arh"}
        else:
            usable.append((model_dir.name, paths))
    if requested_models is not None:
        found = {name for name, _ in usable} | set(inventory)
        missing_names = requested_models - found
        if missing_names:
            raise ValueError(f"requested model directories not found: {sorted(missing_names)}")
    if not usable:
        raise ValueError("no usable TRENDY models found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_cache = {}
    for model, paths in usable:
        expected_output = args.output_dir / (
            f"TRENDYv14_S3_{model}_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021.nc"
        )
        expected_summary = expected_output.with_suffix(".json")
        if args.skip_existing and expected_output.is_file() and expected_summary.is_file():
            inventory[model] = {
                "status": "complete",
                "summary": json.loads(expected_summary.read_text()),
                "reused_existing": True,
            }
            print(f"[{model}] reusing existing validated output", flush=True)
            continue
        try:
            inventory[model] = {
                "status": "complete",
                "summary": process_model(model, paths, args, geometry, fine, matrix_cache),
            }
        except Exception as error:
            inventory[model] = {"status": "failed", "error": repr(error)}
            (args.output_dir / "TRENDYv14_S3_2021_manifest.json").write_text(
                json.dumps(inventory, indent=2) + "\n"
            )
            raise
        (args.output_dir / "TRENDYv14_S3_2021_manifest.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
    complete = sum(value["status"] == "complete" for value in inventory.values())
    print(f"Completed {complete}/{len(usable)} usable TRENDY models", flush=True)


if __name__ == "__main__":
    main()
