#!/usr/bin/env python3
"""Zonal sums of the layered-tracer runs on a latitude-pressure grid.

    python3 examples/paper/layer_tracers/zonal_reduce.py --runs <dir with one folder per run> \
        --out <dir for the reduced files> [--names <run> ...] [--hours 0 24 168 336 504 672] [--jobs 8] \
        [--min-surface-pressure 950]

For every run and requested snapshot the tracer amount and the dry air mass of each cell are
distributed over fixed pressure bins in proportion to the overlap of the model layer with the
bin (the tracer is taken as uniform within a layer) and summed over latitude bins. Both grids
(72 and 117 layers) therefore end on the same latitude-pressure grid, and the sums keep the
global totals. Pressure is that of the dry air above a level, g times the dry air mass per area.

With `--min-surface-pressure` only columns whose surface pressure at the start of the run exceeds
the value (hPa) enter the sums. Over high terrain a release level given as a fraction of the surface
pressure lies at a lower pressure, so sums over all columns mix release pressures.

Output per run: `<out>/<run>.npz` with `hours`, `lat_edges`, `p_edges` (hPa), `air[t, p, lat]`
(kg) and one array `<tracer>[t, p, lat]` (mole fraction times kg of air), the global totals
`total_air[t]` and `total_<tracer>[t]` over all columns, and `kept_area_fraction[lat]`.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import netCDF4 as nc
import numpy as np

GRAVITY = 9.80665                      # m s-2
TRACERS = ("sfc", "s095", "s085", "s070", "s050", "s025", "s007")
LAT_EDGES = np.arange(-90.0, 90.1, 2.0)
P_EDGES = np.concatenate([[0, 5, 10, 20, 30, 40, 50, 60, 70, 85, 100], np.arange(125.0, 1051.0, 25.0)])


def reduce_run(args):
    run_dir, out_dir, hours, min_ps = args
    name = Path(run_dir).name
    with nc.Dataset(f"{run_dir}/output.nc") as d:
        time = np.asarray(d["time"][:])
        lat = np.asarray(d["lats"][:]).ravel()
        area = np.asarray(d["cell_area"][:], dtype=np.float64).ravel()
        lat_bin = np.clip(np.digitize(lat, LAT_EDGES) - 1, 0, len(LAT_EDGES) - 2)
        n_lat, n_p = len(LAT_EDGES) - 1, len(P_EDGES) - 1
        use = [h for h in hours if h in time]
        out = {t: np.zeros((len(use), n_p, n_lat)) for t in ("air",) + TRACERS}
        totals = {f"total_{t}": np.zeros(len(use)) for t in ("air",) + TRACERS}
        start = np.asarray(d["air_mass_per_area"][0], dtype=np.float64)
        keep = start.reshape(start.shape[0], -1).sum(axis=0) * GRAVITY / 100.0 > min_ps   # surface pressure, hPa
        kept_area = np.bincount(lat_bin[keep], weights=area[keep], minlength=n_lat) / np.bincount(lat_bin, weights=area, minlength=n_lat)
        lat_bin = lat_bin[keep]
        for it, h in enumerate(use):
            i = int(np.where(time == h)[0][0])
            per_area = np.asarray(d["air_mass_per_area"][i], dtype=np.float64)
            per_area = per_area.reshape(per_area.shape[0], -1)   # [lev, cell], kg m-2, top first
            p_bot = np.cumsum(per_area, axis=0) * GRAVITY / 100.0    # hPa
            p_top = p_bot - per_area * GRAVITY / 100.0
            thickness = np.maximum(p_bot - p_top, 1e-12)
            air = per_area * area
            totals["total_air"][it] = air.sum()
            amounts = {"air": air[:, keep]}
            for t in TRACERS:
                q = np.asarray(d[t][i], dtype=np.float64)
                amount = q.reshape(q.shape[0], -1) * air
                totals[f"total_{t}"][it] = amount.sum()
                amounts[t] = amount[:, keep]
            p_top, thickness = p_top[:, keep], thickness[:, keep]
            below = {t: np.zeros((len(P_EDGES), n_lat)) for t in amounts}
            for ie, edge in enumerate(P_EDGES):
                frac = np.clip((edge - p_top) / thickness, 0.0, 1.0)     # share of each layer above the edge
                for t, a in amounts.items():
                    below[t][ie] = np.bincount(lat_bin, weights=(frac * a).sum(axis=0), minlength=n_lat)
            for t in amounts:
                out[t][it] = np.diff(below[t], axis=0)
                out[t][it, -1] += np.bincount(lat_bin, weights=amounts[t].sum(axis=0), minlength=n_lat) - below[t][-1]
    np.savez_compressed(f"{out_dir}/{name}.npz", hours=np.array(use), lat_edges=LAT_EDGES, p_edges=P_EDGES,
                        kept_area_fraction=kept_area, min_surface_pressure=min_ps, **totals, **out)
    return name, len(use)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", help="run folders to reduce (default: all with an output.nc)")
    p.add_argument("--hours", nargs="+", type=int, default=[0, 24, 72, 168, 336, 504, 672])
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--min-surface-pressure", type=float, default=0.0, help="hPa; keep only columns above it")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    names = a.names or sorted(q.parent.name for q in Path(a.runs).glob("*/output.nc"))
    todo = [(f"{a.runs}/{n}", a.out, a.hours, a.min_surface_pressure) for n in names if not Path(f"{a.out}/{n}.npz").exists()]
    with Pool(a.jobs) as pool:
        for name, n in pool.imap_unordered(reduce_run, todo):
            print(f"{name}: {n} snapshots", flush=True)


if __name__ == "__main__":
    main()
