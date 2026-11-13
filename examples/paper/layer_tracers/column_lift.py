#!/usr/bin/env python3
"""Maps of how far each column has moved a layered tracer in the vertical.

    python3 examples/paper/layer_tracers/column_lift.py --runs <dir> --out <dir> --names <run> ... [--hours 24 72 168]

For every column and requested time: the mean sigma = p/ps of the tracer in the column, and the
share of the column's tracer above sigma = 0.5. Averaged onto a 2 degree latitude-longitude grid
(area weighted). Near the poles some 2 degree bins contain no cell centre of a C90 grid; they take
the value of the nearest filled bin of the same latitude row. In runs without advection the columns do not exchange tracer, so the maps show
the local effect of mixing and convection; with advection they also show where tracer converges.

Output per run: `<out>/<run>.npz` with `hours`, `lat_edges`, `lon_edges`, `surface_pressure[lat, lon]`
(hPa, at the start) and `mean_<tracer>[t, lat, lon]`, `upper_<tracer>[t, lat, lon]`.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import netCDF4 as nc
import numpy as np

GRAVITY = 9.80665
TRACERS = ("sfc", "s095", "s085", "s070", "s050", "s025", "s007")
LAT_EDGES = np.arange(-90.0, 90.1, 2.0)
LON_EDGES = np.arange(-180.0, 180.1, 2.0)


def to_grid(values, weights, index, n):
    total = np.bincount(index, weights=weights, minlength=n)
    return np.bincount(index, weights=values * weights, minlength=n) / np.where(total > 0, total, np.nan)


def fill_rows(grid):
    """Fill the empty bins of each latitude row from the nearest filled bin of that row (periodic)."""
    out = grid.copy()
    n = grid.shape[-1]
    lon = np.arange(n)
    for row in out.reshape(-1, n):
        ok = np.isfinite(row)
        if ok.all() or not ok.any():
            continue
        filled = lon[ok]
        distance = np.abs((lon[:, None] - filled[None, :] + n // 2) % n - n // 2)
        row[~ok] = row[filled[np.argmin(distance[~ok], axis=1)]]
    return out


def reduce_run(args):
    run_dir, out_dir, hours = args
    name = Path(run_dir).name
    with nc.Dataset(f"{run_dir}/output.nc") as d:
        time = np.asarray(d["time"][:])
        lat = np.asarray(d["lats"][:]).ravel()
        lon = (np.asarray(d["lons"][:]).ravel() + 180.0) % 360.0 - 180.0
        area = np.asarray(d["cell_area"][:], dtype=np.float64).ravel()
        n_lat, n_lon = len(LAT_EDGES) - 1, len(LON_EDGES) - 1
        index = (np.clip(np.digitize(lat, LAT_EDGES) - 1, 0, n_lat - 1) * n_lon
                 + np.clip(np.digitize(lon, LON_EDGES) - 1, 0, n_lon - 1))
        use = [h for h in hours if h in time]
        out = {f"{k}_{t}": np.zeros((len(use), n_lat, n_lon)) for k in ("mean", "upper") for t in TRACERS}
        start = np.asarray(d["air_mass_per_area"][0], dtype=np.float64)
        ps = to_grid(start.reshape(start.shape[0], -1).sum(axis=0) * GRAVITY / 100.0, area, index, n_lat * n_lon)
        for it, h in enumerate(use):
            i = int(np.where(time == h)[0][0])
            a = np.asarray(d["air_mass_per_area"][i], dtype=np.float64)
            a = a.reshape(a.shape[0], -1)
            bottom = np.cumsum(a, axis=0) / a.sum(axis=0)
            top = bottom - a / a.sum(axis=0)
            mid = 0.5 * (top + bottom)
            above = np.clip((0.5 - top) / (bottom - top), 0.0, 1.0)      # part of each layer above sigma 0.5
            for t in TRACERS:
                q = np.asarray(d[t][i], dtype=np.float64)
                m = q.reshape(q.shape[0], -1) * a
                column = m.sum(axis=0)
                safe = np.where(column > 0, column, np.nan)
                out[f"mean_{t}"][it] = fill_rows(to_grid((m * mid).sum(axis=0) / safe, area, index, n_lat * n_lon).reshape(n_lat, n_lon))
                out[f"upper_{t}"][it] = fill_rows(to_grid((m * above).sum(axis=0) / safe, area, index, n_lat * n_lon).reshape(n_lat, n_lon))
    np.savez_compressed(f"{out_dir}/{name}.npz", hours=np.array(use), lat_edges=LAT_EDGES, lon_edges=LON_EDGES,
                        surface_pressure=fill_rows(ps.reshape(n_lat, n_lon)), **out)
    return name, len(use)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--hours", nargs="+", type=int, default=[0, 24, 72, 168, 336, 672])
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    todo = []
    for n in a.names:
        roots = [r for r in a.runs if Path(f"{r}/{n}/output.nc").exists()]
        if not roots:
            print(f"{n}: no output.nc, skipped")
            continue
        source, target = Path(f"{roots[0]}/{n}/output.nc"), Path(f"{a.out}/{n}.npz")
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:      # also after a rerun
            todo.append((f"{roots[0]}/{n}", a.out, a.hours))
    with Pool(a.jobs) as pool:
        for name, n in pool.imap_unordered(reduce_run, todo):
            print(f"{name}: {n} snapshots", flush=True)


if __name__ == "__main__":
    main()
