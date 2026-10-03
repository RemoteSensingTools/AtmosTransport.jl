#!/usr/bin/env python3
"""Split a C180 SIF-GPP CO2 flux file into major TransCom land-region tags.

This is a pragmatic pilot mask for tagged transport experiments. It follows the
standard major TransCom land-region names, excludes Antarctica and Greenland,
and relies on the input SIF-GPP flux being land-only/near-zero over oceans.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np


@dataclass(frozen=True)
class Region:
    key: str
    label: str
    code: int


REGIONS = [
    Region("na_boreal", "North American Boreal", 1),
    Region("na_temperate", "North American Temperate", 2),
    Region("south_america_tropical", "South American Tropical", 3),
    Region("south_america_temperate", "South American Temperate", 4),
    Region("northern_africa", "Northern Africa", 5),
    Region("southern_africa", "Southern Africa", 6),
    Region("eurasia_boreal", "Eurasian Boreal", 7),
    Region("eurasia_temperate", "Eurasian Temperate", 8),
    Region("tropical_asia", "Tropical Asia", 9),
    Region("australia", "Australia", 10),
    Region("europe", "Europe", 11),
]


def lon180(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def between_lon(lon, west, east):
    lon = lon180(lon)
    west = lon180(west)
    east = lon180(east)
    if west <= east:
        return (lon >= west) & (lon < east)
    return (lon >= west) | (lon < east)


def region_masks(lon_flat, lat_flat, shape):
    nf, ny, nx = shape
    lon = np.empty(shape, dtype=np.float64)
    lat = np.empty(shape, dtype=np.float64)
    cells_per_panel = ny * nx
    for p in range(nf):
        sl = slice(p * cells_per_panel, (p + 1) * cells_per_panel)
        # The forcing NetCDF is read as (nf, Ydim, Xdim) by Python, while
        # compact diagnostics flatten Julia panel matrices. C-order per panel
        # maps the diagnostic lon/lat vectors back onto the forcing layout.
        lon[p] = lon_flat[sl].reshape((ny, nx), order="C")
        lat[p] = lat_flat[sl].reshape((ny, nx), order="C")
    mask = {}

    greenland = (lat >= 58.0) & between_lon(lon, -75.0, -10.0)
    antarctica = lat <= -60.0
    excluded = greenland | antarctica

    mask["na_boreal"] = (
        (lat >= 49.0)
        & between_lon(lon, -170.0, -52.0)
        & ~greenland
    )
    mask["na_temperate"] = (
        (lat >= 15.0)
        & (lat < 49.0)
        & between_lon(lon, -170.0, -52.0)
    )
    mask["south_america_tropical"] = (
        (lat >= -20.0)
        & (lat < 15.0)
        & between_lon(lon, -82.0, -34.0)
    )
    mask["south_america_temperate"] = (
        (lat >= -60.0)
        & (lat < -20.0)
        & between_lon(lon, -82.0, -34.0)
    )
    mask["northern_africa"] = (
        (lat >= 0.0)
        & (lat < 37.0)
        & between_lon(lon, -20.0, 60.0)
    )
    mask["southern_africa"] = (
        (lat >= -35.0)
        & (lat < 0.0)
        & between_lon(lon, -20.0, 60.0)
    )
    mask["europe"] = (
        (lat >= 35.0)
        & (lat < 72.0)
        & between_lon(lon, -12.0, 40.0)
    )
    mask["eurasia_boreal"] = (
        (lat >= 55.0)
        & between_lon(lon, 40.0, 180.0)
    )
    mask["eurasia_temperate"] = (
        (lat >= 23.5)
        & (lat < 55.0)
        & between_lon(lon, 40.0, 180.0)
    )
    mask["tropical_asia"] = (
        (lat >= -12.0)
        & (lat < 23.5)
        & between_lon(lon, 60.0, 180.0)
    )
    mask["australia"] = (
        (lat >= -50.0)
        & (lat < -10.0)
        & between_lon(lon, 110.0, 180.0)
    )

    for key in list(mask):
        mask[key] &= ~excluded

    assigned = np.zeros(shape, dtype=bool)
    region_id = np.zeros(shape, dtype=np.int16)
    for region in REGIONS:
        current = mask[region.key] & ~assigned
        mask[region.key] = current
        assigned |= current
        region_id[current] = region.code

    return mask, region_id, excluded


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="C180 NetCDF with CO2_FLUX(time,nf,Ydim,Xdim)")
    p.add_argument("--geometry", required=True, help="Compact C180 XCO2 NetCDF with cs_lon/cs_lat")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    src_path = Path(args.input)
    geom_path = Path(args.geometry)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(geom_path) as geom:
        lon_flat = np.asarray(geom["cs_lon"][:], dtype=np.float64)
        lat_flat = np.asarray(geom["cs_lat"][:], dtype=np.float64)

    with nc.Dataset(src_path) as src:
        flux = src["CO2_FLUX"]
        time_len, nf, ny, nx = flux.shape
        shape = (nf, ny, nx)
        masks, region_id, excluded = region_masks(lon_flat, lat_flat, shape)

        with nc.Dataset(out_path, "w") as dst:
            for dim in ("time", "nf", "Ydim", "Xdim"):
                dst.createDimension(dim, len(src.dimensions[dim]))
            for name, var in src.variables.items():
                if name == "CO2_FLUX":
                    continue
                out = dst.createVariable(name, var.datatype, var.dimensions)
                out.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
                out[:] = var[:]

            rid = dst.createVariable("transcom_region_id", "i2", ("nf", "Ydim", "Xdim"))
            rid.long_name = "Major TransCom land-region id used for tagged SIF-GPP flux"
            rid.note = "0 is unassigned/ocean/excluded; Antarctica and Greenland are excluded"
            rid[:] = region_id

            ex = dst.createVariable("excluded_greenland_antarctica", "i1", ("nf", "Ydim", "Xdim"))
            ex.long_name = "Mask excluded from regional land tags"
            ex[:] = excluded.astype(np.int8)

            total = np.zeros((time_len, nf, ny, nx), dtype=np.float32)
            for region in REGIONS:
                var_name = f"CO2_FLUX_{region.key.upper()}"
                out = dst.createVariable(
                    var_name,
                    "f4",
                    ("time", "nf", "Ydim", "Xdim"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                )
                out.units = getattr(flux, "units", "kg CO2 m-2 s-1")
                out.long_name = f"SIF-GPP CO2 surface flux tagged to {region.label}"
                out.transcom_region = region.label
                out.transcom_region_code = region.code

                m = masks[region.key][None, :, :, :]
                # Keep chunks modest to avoid holding region*time temporary arrays
                # larger than the already-loaded month-sized source flux.
                tagged = np.asarray(flux[:], dtype=np.float32) * m
                out[:] = tagged
                total += tagged

            residual = dst.createVariable(
                "CO2_FLUX_UNASSIGNED",
                "f4",
                ("time", "nf", "Ydim", "Xdim"),
                zlib=True,
                complevel=4,
                shuffle=True,
            )
            residual.units = getattr(flux, "units", "kg CO2 m-2 s-1")
            residual.long_name = "Input SIF-GPP CO2 flux not assigned to a tagged TransCom land region"
            residual[:] = np.asarray(flux[:], dtype=np.float32) - total

            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
            dst.title = "C180 length-of-day SIF-GPP CO2 flux split into major TransCom land-region tags"
            dst.source_flux_file = str(src_path)
            dst.geometry_file = str(geom_path)
            dst.region_keys = ",".join(r.key for r in REGIONS)
            dst.region_labels = "|".join(r.label for r in REGIONS)

    print(out_path)


if __name__ == "__main__":
    main()
