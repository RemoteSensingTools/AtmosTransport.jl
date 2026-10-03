#!/usr/bin/env python3
"""Conservatively average an hourly native-C90 flux into daily means.

The output keeps one 00 UTC timestamp per day. Multiplying a daily mean by
86,400 seconds gives the same daily mass per unit area as integrating the 24
hourly input values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-variable", default="CO2_FLUX")
    parser.add_argument("--output-variable", default="GPP_CO2_FLUX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(args.input) as src:
        time_var = src["time"]
        units = time_var.units
        calendar = getattr(time_var, "calendar", "standard")
        time_values = np.asarray(time_var[:], dtype=np.float64)
        stamps = netCDF4.num2date(time_values, units, calendar=calendar)

        groups: list[list[int]] = []
        previous = None
        for index, stamp in enumerate(stamps):
            key = (stamp.year, stamp.month, stamp.day)
            if key != previous:
                groups.append([])
                previous = key
            groups[-1].append(index)
        if any(len(group) != 24 for group in groups):
            raise ValueError("input must contain exactly 24 hourly records per UTC day")

        source_flux = src[args.input_variable]
        if source_flux.dimensions != ("time", "nf", "Ydim", "Xdim"):
            raise ValueError(
                "flux dimensions must be (time,nf,Ydim,Xdim), got "
                f"{source_flux.dimensions}"
            )
        if source_flux.shape[1:] != (6, 90, 90):
            raise ValueError(f"input is not native C90: shape={source_flux.shape}")

        with netCDF4.Dataset(args.output, "w", format="NETCDF4") as dst:
            dst.createDimension("time", len(groups))
            for name in ("nf", "Ydim", "Xdim"):
                dst.createDimension(name, len(src.dimensions[name]))

            out_time = dst.createVariable("time", "f8", ("time",))
            out_time.setncatts(
                {name: time_var.getncattr(name) for name in time_var.ncattrs()}
            )
            out_time[:] = [time_values[group[0]] for group in groups]

            for name in ("lons", "lats"):
                if name in src.variables:
                    source = src[name]
                    target = dst.createVariable(name, source.dtype, source.dimensions)
                    target.setncatts(
                        {
                            attr: source.getncattr(attr)
                            for attr in source.ncattrs()
                            if attr != "_FillValue"
                        }
                    )
                    target[:] = source[:]

            output_flux = dst.createVariable(
                args.output_variable,
                "f4",
                ("time", "nf", "Ydim", "Xdim"),
                zlib=True,
                complevel=2,
                shuffle=True,
                fill_value=np.float32(-999.0),
            )
            output_flux.setncatts(
                {
                    attr: source_flux.getncattr(attr)
                    for attr in source_flux.ncattrs()
                    if attr != "_FillValue"
                }
            )
            output_flux.long_name = (
                "daily-mean SIF-derived GPP atmospheric CO2 uptake "
                "per total grid-cell area"
            )

            maximum = -np.inf
            for day, group in enumerate(groups):
                hourly = np.ma.filled(source_flux[group, :, :, :], np.nan)
                if not np.all(np.isfinite(hourly)):
                    raise ValueError(f"non-finite input flux on day index {day}")
                daily = np.mean(hourly, axis=0, dtype=np.float64)
                maximum = max(maximum, float(np.max(daily)))
                output_flux[day, :, :, :] = daily.astype(np.float32)

            if maximum > 1.0e-20:
                raise ValueError(
                    f"GPP atmospheric flux must not be positive; maximum={maximum}"
                )

            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))
            dst.title = "Daily-mean CO2 uptake from the 2021 SIF-GPP pilot"
            dst.temporal_aggregation = (
                "arithmetic mean of 24 hourly rates; daily integral conserved"
            )
            dst.source_file = str(args.input)

    print(f"Wrote {len(groups)} daily C90 fluxes to {args.output}")


if __name__ == "__main__":
    main()
