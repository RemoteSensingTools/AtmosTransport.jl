#!/usr/bin/env python3
"""Refine a panel-native cubed-sphere flux density by integer subdivision."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine a NetCDF (time,nf,Ydim,Xdim) cubed-sphere flux-density "
            "field without changing its areal density."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variable", default="CO2_FLUX")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--start", type=datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=datetime.fromisoformat, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.factor < 1:
        raise ValueError("--factor must be positive")
    if args.end <= args.start:
        raise ValueError("--end must be after --start")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(args.input) as src:
        flux_in = src[args.variable]
        expected_dims = ("time", "nf", "Ydim", "Xdim")
        if flux_in.dimensions != expected_dims:
            raise ValueError(
                f"{args.variable} dimensions are {flux_in.dimensions}, expected {expected_dims}"
            )
        time_in = src["time"]
        i0 = int(date2index(args.start, time_in, select="exact"))
        try:
            i1 = int(date2index(args.end, time_in, select="exact"))
        except ValueError:
            # Permit an exclusive bound exactly one source interval after the
            # final stamp (the usual complete-month case).
            vals = np.asarray(time_in[:], dtype=np.float64)
            if len(vals) < 2:
                raise
            unit_seconds = (args.end - args.start).total_seconds()
            expected_count = round(unit_seconds / ((vals[1] - vals[0]) * 3600.0))
            i1 = i0 + expected_count
            if i1 != len(vals):
                raise
        if not (0 <= i0 < i1 <= len(time_in)):
            raise ValueError(f"invalid time slice [{i0}:{i1}] for {len(time_in)} records")

        nf, ny, nx = flux_in.shape[1:]
        if nf != 6 or ny != nx:
            raise ValueError(f"expected six square panels, got {flux_in.shape[1:]}")
        ny_out, nx_out = ny * args.factor, nx * args.factor

        with Dataset(args.output, "w", format="NETCDF4") as dst:
            dst.createDimension("time", i1 - i0)
            dst.createDimension("nf", nf)
            dst.createDimension("Ydim", ny_out)
            dst.createDimension("Xdim", nx_out)

            time_out = dst.createVariable("time", time_in.dtype, ("time",))
            time_out.setncatts({k: time_in.getncattr(k) for k in time_in.ncattrs()})
            time_out[:] = time_in[i0:i1]

            fill = getattr(flux_in, "_FillValue", None)
            kwargs = dict(
                zlib=True,
                complevel=2,
                shuffle=True,
                chunksizes=(1, 1, ny_out, nx_out),
            )
            if fill is not None:
                kwargs["fill_value"] = fill
            flux_out = dst.createVariable(
                args.variable, flux_in.dtype, expected_dims, **kwargs
            )
            flux_out.setncatts(
                {
                    k: flux_in.getncattr(k)
                    for k in flux_in.ncattrs()
                    if k != "_FillValue"
                }
            )
            flux_out.long_name = (
                f"{getattr(flux_in, 'long_name', args.variable)}; "
                f"C{ny} density-preserving refinement to C{ny_out}"
            )

            for out_t, in_t in enumerate(range(i0, i1)):
                parent = np.asarray(flux_in[in_t, :, :, :])
                child = np.repeat(
                    np.repeat(parent, args.factor, axis=1), args.factor, axis=2
                )
                flux_out[out_t, :, :, :] = child

            for name in src.ncattrs():
                dst.setncattr(name, src.getncattr(name))
            dst.setncattr("source_file", str(args.input))
            dst.setncattr("source_grid", f"C{ny}")
            dst.setncattr("target_grid", f"C{ny_out}")
            dst.setncattr("grid_type", "cubed_sphere")
            dst.setncattr("panel_convention", "geos_native")
            dst.setncattr("refinement_factor", args.factor)
            dst.setncattr(
                "refinement_method",
                "each parent flux density copied to all child cells; no interpolation",
            )
            dst.setncattr("time_subset_start", args.start.isoformat())
            dst.setncattr("time_subset_end_exclusive", args.end.isoformat())

    print(
        f"Wrote {args.output}: {i1-i0} times, 6 panels, C{ny_out}, "
        f"variable={args.variable}"
    )


if __name__ == "__main__":
    main()
