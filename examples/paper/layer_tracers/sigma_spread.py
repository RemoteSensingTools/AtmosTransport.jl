#!/usr/bin/env python3
"""Vertical distribution of each layered tracer in sigma = p/ps at every output time.

    python3 examples/paper/layer_tracers/sigma_spread.py --runs <dir with one folder per run> \
        --out <dir for the reduced files> [--names <run> ...] [--jobs 8] [--min-surface-pressure 950]

For every run, snapshot and tracer the tracer amount is accumulated over sigma, the pressure of the
dry air above a level divided by the surface value of the column, with the tracer uniform inside
each model layer. The cumulative distribution is stored at `N_EDGES` equally spaced sigma values for
a set of regions (all latitudes, three latitude bands and the boxes of `BOXES`), so that any
percentile can be read off later: the 15.9 and 84.1 % points are what plus and minus one standard
deviation are for a Gaussian, the 2.3 and 97.7 % points what two are, without assuming a shape.

Sigma follows the terrain, and the tracers are released at fixed sigma, so all columns are used by
default. `--min-surface-pressure` keeps only columns whose surface pressure at the start of the run
exceeds the value (hPa), as zonal_reduce.py does for its pressure coordinate.

Output per run: `<out>/<run>.npz` with `hours`, `sigma_edges`, `regions` (names), `region_boxes`
(south, north, west, east) and one array `<tracer>[t, region, edge]`: the fraction of the region's
tracer above the sigma edge (0 at the top, 1 at the surface); `mean_<tracer>[t, region]` is the
mean sigma.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import netCDF4 as nc
import numpy as np

GRAVITY = 9.80665                      # m s-2
TRACERS = ("sfc", "s095", "s085", "s070", "s050", "s025", "s007")
BANDS = (("30°N–90°N", 30.0, 90.0), ("30°S–30°N", -30.0, 30.0), ("90°S–30°S", -90.0, -30.0))
# name: (south, north, west, east); west > east crosses the date line. The boxes do not overlap.
BOXES = {
    "Amazon": (-10, 0, -70, -55),
    "Congo basin": (-5, 5, 15, 30),
    "Sahel": (8, 16, -10, 30),
    "India": (15, 27, 73, 88),
    "Maritime Continent": (-10, 10, 100, 150),
    "West Pacific warm pool": (0, 10, 150, 175),
    "Pacific 48°N": (40, 55, 170, -140),
    "Pacific 26°N": (20, 32, -170, -140),
    "Pacific 8°N (ITCZ)": (4, 12, -160, -120),
    "Pacific equator": (-4, 2, -160, -110),
    "Pacific 10°S (SPCZ)": (-15, -5, 170, -160),
    "Pacific 26°S": (-32, -20, -150, -110),
    "Pacific 48°S": (-55, -40, -170, -110),
    "Southeast Pacific": (-30, -15, -100, -80),
    "North Atlantic storm track": (40, 58, -50, -15),
    "Southern Ocean (Indian sector)": (-60, -45, 40, 100),
    "Europe": (44, 56, 0, 25),
    "Siberia": (55, 70, 80, 130),
    "Great Plains": (33, 45, -105, -90),
}
REGIONS = ["all latitudes"] + [b[0] for b in BANDS] + list(BOXES)
N_EDGES = 401                          # sigma = 0, 0.0025, ..., 1


def cumulative(amount, top, bottom, region, n_regions):
    """Tracer above each sigma edge, per region, for amounts uniform between `top` and `bottom`.

    The amount of a layer above the edge e is amount * clip((e - top) / (bottom - top), 0, 1): a
    ramp that starts at `top` with slope amount / thickness and stops at `bottom`. The sum of ramps
    at the edges follows from two weighted counts of the ramp starts and ends, exactly.
    """
    edges = np.linspace(0.0, 1.0, N_EDGES)
    slope = amount / (bottom - top)
    out = np.zeros((n_regions, N_EDGES))
    for x, s in ((top, slope), (bottom, -slope)):
        # ramp starts at or before edge j belong to all edges j, j + 1, ...
        j = np.clip(np.ceil(x * (N_EDGES - 1) - 1e-12).astype(np.int64), 0, N_EDGES - 1)
        index = region * N_EDGES + j
        s0 = np.bincount(index, weights=s, minlength=n_regions * N_EDGES).reshape(n_regions, N_EDGES)
        s1 = np.bincount(index, weights=s * x, minlength=n_regions * N_EDGES).reshape(n_regions, N_EDGES)
        out += edges * np.cumsum(s0, axis=1) - np.cumsum(s1, axis=1)
    return out


def reduce_run(args):
    run_dir, out_dir, min_ps = args
    name = Path(run_dir).name
    with nc.Dataset(f"{run_dir}/output.nc") as d:
        hours = np.asarray(d["time"][:])
        lat = np.asarray(d["lats"][:]).ravel()
        area = np.asarray(d["cell_area"][:], dtype=np.float64).ravel()
        start = np.asarray(d["air_mass_per_area"][0], dtype=np.float64)
        keep = start.reshape(start.shape[0], -1).sum(axis=0) * GRAVITY / 100.0 > min_ps
        lon = (np.asarray(d["lons"][:]).ravel() + 180.0) % 360.0 - 180.0
        columns = np.flatnonzero(keep)
        # Each column belongs to one class: its latitude band and its box (or none). A region is
        # the union of classes, so one pass over the cells serves all regions.
        band = np.select([lat >= 30.0, lat >= -30.0], [0, 1], 2)
        box = np.zeros(lat.size, dtype=np.int64)
        for j, (south, north, west, east) in enumerate(BOXES.values(), start=1):
            inside = (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
            box[(lat >= south) & (lat <= north) & inside] = j
        n_box = len(BOXES) + 1
        n_class = len(BANDS) * n_box
        klass = (band * n_box + box)[columns]
        members = np.zeros((len(REGIONS), n_class))
        members[0] = 1.0
        for b in range(len(BANDS)):
            members[1 + b, b * n_box:(b + 1) * n_box] = 1.0
        for j in range(1, n_box):
            members[len(BANDS) + j, j::n_box] = 1.0
        out = {t: np.zeros((len(hours), len(REGIONS), N_EDGES), dtype=np.float32) for t in TRACERS}
        mean = {f"mean_{t}": np.zeros((len(hours), len(REGIONS))) for t in TRACERS}
        for i in range(len(hours)):
            per_area = np.asarray(d["air_mass_per_area"][i], dtype=np.float64)
            per_area = per_area.reshape(per_area.shape[0], -1)[:, columns]    # [lev, column], top first
            bottom = np.cumsum(per_area, axis=0)
            surface = bottom[-1]
            bottom /= surface
            bottom[-1] = 1.0
            top = np.vstack([np.zeros(len(columns)), bottom[:-1]])
            air = per_area * area[columns]
            cell_class = np.broadcast_to(klass, top.shape).ravel()
            tp, bt, mid = top.ravel(), bottom.ravel(), (0.5 * (top + bottom)).ravel()
            for t in TRACERS:
                q = np.asarray(d[t][i], dtype=np.float64)
                amount = (q.reshape(q.shape[0], -1)[:, columns] * air).ravel()
                by_region = members @ cumulative(amount, tp, bt, cell_class, n_class)
                total = by_region[:, -1]
                out[t][i] = by_region / total[:, None]
                first = members @ np.bincount(cell_class, weights=amount * mid, minlength=n_class)
                mean[f"mean_{t}"][i] = first / total
    boxes = [(-90, 90, -180, 180)] + [(b[1], b[2], -180, 180) for b in BANDS] + list(BOXES.values())
    np.savez_compressed(f"{out_dir}/{name}.npz", hours=hours, sigma_edges=np.linspace(0.0, 1.0, N_EDGES),
                        regions=np.array(REGIONS), min_surface_pressure=min_ps,
                        region_boxes=np.array(boxes, dtype=float), **out, **mean)
    return name, len(hours)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", help="run folders to reduce (default: all with an output.nc)")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--min-surface-pressure", type=float, default=0.0, help="hPa; keep only columns above it")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    names = a.names or sorted(q.parent.name for q in Path(a.runs).glob("*/output.nc"))
    def stale(n):      # no reduced file yet, or the run was repeated since
        source, target = Path(f"{a.runs}/{n}/output.nc"), Path(f"{a.out}/{n}.npz")
        return not target.exists() or target.stat().st_mtime < source.stat().st_mtime
    todo = [(f"{a.runs}/{n}", a.out, a.min_surface_pressure) for n in names if stale(n)]
    with Pool(a.jobs) as pool:
        for name, n in pool.imap_unordered(reduce_run, todo):
            print(f"{name}: {n} snapshots", flush=True)


if __name__ == "__main__":
    main()
