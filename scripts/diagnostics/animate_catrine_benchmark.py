#!/usr/bin/env python3
"""Daily column-mean animations of AtmosTransport CATRINE runs against GEOS-Chem.

Reads the same files and matches the same snapshots as
catrine_compare_vs_geoschem.py (its dataset and GEOS-Chem helpers are reused),
one frame per day at --hour UTC. Column means are dry-air-mass weighted from
the 3-D fields and sampled from the nearest C90 cell onto a 1° raster (all
datasets share the GEOS-Chem cube; the comparison script checks the nesting).
Writes two movies:

  <out>/catrine_benchmark_column_means.mp4
      rows = tracers, columns = GEOS-Chem | each dataset. CO2, fossil CO2 and
      SF6 are shown relative to the GEOS-Chem global mean of each frame, so the
      spatial pattern stays visible on top of the multi-year growth (a model
      whose burden drifts away from GEOS-Chem shifts as a whole); Rn-222 is
      absolute on a log scale.
  <out>/catrine_benchmark_differences.mp4
      rows = tracers, columns = dataset - GEOS-Chem.

Each movie also gets a PNG of its last frame.

  python3 scripts/diagnostics/animate_catrine_benchmark.py \\
      --dataset 'era5=/temp1/.../output/full/catrine_c90_*.nc' \\
      --extra 'era5=/temp1/.../output/full_fossil_dec2021/*.nc' --fossil era5=co2_fossil_from_dec2021 \\
      --dataset 'merra2=/temp1/.../output/merra2_full_f32/*.nc' --fossil merra2=co2_fossil_from_dec2021 \\
      --label era5='AT ERA5' --label merra2='AT MERRA-2' --out ~/www/catrine/benchmark
"""
import argparse, datetime as dt, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from netCDF4 import Dataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catrine_compare_vs_geoschem as cmp  # noqa: E402  (shared datasets and GEOS-Chem helpers)

NLON, NLAT = 360, 180
ROWS = [  # tracer key in cmp.TRACERS, title, relative to the GEOS-Chem global mean?
    ("co2", "CO₂", True),
    ("fossil", "Fossil CO₂", True),
    ("sf6", "SF₆", True),
    ("rn222", "Rn-222", False),
]


_NEAREST = {}


def raster_index(lons, lats):
    """Nearest cube cell for every pixel of the lat-lon raster (cached per process)."""
    if lons.shape not in _NEAREST:
        from scipy.spatial import cKDTree

        def xyz(lon, lat):
            lon, lat = np.radians(lon), np.radians(lat)
            return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
        plon, plat = np.meshgrid(np.arange(NLON) + 0.5 - 180, np.arange(NLAT) + 0.5 - 90)
        _, idx = cKDTree(xyz(lons.ravel(), lats.ravel())).query(xyz(plon.ravel(), plat.ravel()))
        _NEAREST[lons.shape] = idx.reshape(NLAT, NLON)
    return _NEAREST[lons.shape]


def column_mean(q, m):
    return np.nansum(q * m, axis=0) / np.nansum(m, axis=0)


def frame(args):
    """Column-mean rasters (rows × sources) and area-weighted global means for one time."""
    t, datasets, gc_dir = args
    with Dataset(cmp.gc_path(t, gc_dir)) as g:
        ad, area = cmp.as_float(g["Met_AD"][0]), cmp.as_float(g["Met_AREAM2"][0])
        lons, lats = cmp.as_float(g["lons"][:]), cmp.as_float(g["lats"][:])
        cols = [[column_mean(cmp.as_float(g[cmp.TRACERS[k][1]][0]), ad) * cmp.TRACERS[k][4] for k, *_ in ROWS]]
    for ds in datasets:
        f, i = ds.indexes[0][t]
        with Dataset(f) as d:
            m = cmp.as_float(d["air_mass_per_area"][i])
        row = []
        for k, *_ in ROWS:
            f, i = ds.indexes[ds.source[k]][t]
            with Dataset(f) as d:
                row.append(column_mean(cmp.as_float(d[ds.vars[k]][i]), m) * cmp.TRACERS[k][4])
        cols.append(row)
    w = area / area.sum()
    gmean = np.array([[np.nansum(c * w) for c in src] for src in cols])        # (sources, rows)
    idx = raster_index(lons, lats)
    maps = np.array([[c.ravel()[idx] for c in src] for src in cols], np.float32)  # (sources, rows, lat, lon)
    return t, maps, gmean


def symmetric_norm(values, pct):
    v = np.abs(values[np.isfinite(values)])
    lim = float(np.percentile(v, pct)) if v.size else 1.0
    return TwoSlopeNorm(0.0, -lim, lim)


def render(path, fig, update, nframes, fps):
    update(nframes - 1)
    fig.savefig(path.rsplit(".", 1)[0] + "_last.png", dpi=110)
    manim.FuncAnimation(fig, update, frames=nframes, blit=False).save(
        path, writer=manim.FFMpegWriter(fps=fps, bitrate=5000), dpi=110)
    plt.close(fig)
    print("wrote", path, flush=True)


def panel_grid(nrows, ncols, titles):
    proj, pc = ccrs.Robinson(), ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols + 1.2, 2.35 * nrows + 0.6), squeeze=False,
                             subplot_kw={"projection": proj})
    fig.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.02, wspace=0.04, hspace=0.12)
    for c, title in enumerate(titles):
        axes[0][c].set_title(title, fontsize=12, fontweight="bold")
    for ax in axes.flat:
        ax.set_global(); ax.coastlines(linewidth=0.25, color="0.2", alpha=0.6)
    return fig, axes, pc


def add_images(fig, axes, pc, first, norms, cmaps, row_labels):
    ims = [[ax.imshow(first[r][c], origin="lower", extent=[-180, 180, -90, 90], transform=pc,
                      cmap=cmaps[r], norm=norms[r]) for c, ax in enumerate(row)] for r, row in enumerate(axes)]
    for r, row in enumerate(axes):
        row[0].text(-0.05, 0.5, row_labels[r], fontsize=10, rotation=90, va="center", ha="center",
                    transform=row[0].transAxes)
        pos = row[-1].get_position()
        fig.colorbar(ims[r][-1], cax=fig.add_axes([pos.x1 + 0.008, pos.y0, 0.011, pos.height]))
    return ims


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gc-dir", default=cmp.GC_DIR)
    ap.add_argument("--dataset", action="append", required=True, help="label=glob of AtmosTransport files")
    ap.add_argument("--extra", action="append", default=[], help="label=glob of files with more tracers")
    ap.add_argument("--fossil", action="append", default=[], help="label=variable of the fossil tracer")
    ap.add_argument("--label", action="append", default=[], help="label=column title")
    ap.add_argument("--start", default="2021-12-02")
    ap.add_argument("--end", default="2024-01-01")
    ap.add_argument("--hour", type=int, default=0, help="UTC hour of the daily frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gc_dir = os.path.expanduser(args.gc_dir)
    out = os.path.expanduser(args.out); os.makedirs(out, exist_ok=True)
    fossil = dict(s.split("=", 1) for s in args.fossil)
    titles = dict(s.split("=", 1) for s in args.label)
    extras = defaultdict(list)
    for s in args.extra:
        lab, pat = s.split("=", 1); extras[lab].append(pat)
    datasets = []
    for spec in args.dataset:
        lab, pat = spec.split("=", 1)
        datasets.append(cmp.AtmosDataset(lab, pat, {"fossil": fossil[lab]} if lab in fossil else {},
                                         None, extras.get(lab, ())))
    day, end, times = dt.datetime.fromisoformat(args.start), dt.datetime.fromisoformat(args.end), []
    while day <= end:
        t = day.replace(hour=args.hour)
        if os.path.exists(cmp.gc_path(t, gc_dir)) and all(t in d.times for d in datasets):
            times.append(t)
        day += dt.timedelta(days=1)
    print(f"{len(times)} daily frames {times[0]} .. {times[-1]}", flush=True)

    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(frame, [(t, datasets, gc_dir) for t in times], chunksize=2))
    times = [r[0] for r in results]
    maps = np.stack([r[1] for r in results])      # (F, sources, rows, lat, lon)
    gmean = np.stack([r[2] for r in results])     # (F, sources, rows)
    np.savez_compressed(f"{out}/benchmark_frames.npz", times=np.array([str(t) for t in times]),
                        maps=maps, global_mean=gmean)
    sources = ["GEOS-Chem"] + [titles.get(d.label, d.label) for d in datasets]
    units = [cmp.TRACERS[k][3] for k, *_ in ROWS]

    # Column means, relative rows minus the GEOS-Chem global mean of each frame.
    shown = maps.copy()
    for r, (_, _, relative) in enumerate(ROWS):
        if relative:
            shown[:, :, r] -= gmean[:, 0, r][:, None, None, None]
    norms, cmaps = [], []
    for r, (_, _, relative) in enumerate(ROWS):
        if relative:
            norms.append(symmetric_norm(shown[:, :, r], 99.5)); cmaps.append("RdBu_r")
        else:
            v = shown[:, :, r][np.isfinite(shown[:, :, r]) & (shown[:, :, r] > 0)]
            norms.append(LogNorm(*np.percentile(v, [40, 99.5]), clip=True)); cmaps.append("magma_r")
    labels = [f"{title}\n{'− GEOS-Chem mean ' if rel else ''}({u})" for (_, title, rel), u in zip(ROWS, units)]
    fig, axes, pc = panel_grid(len(ROWS), len(sources), sources)
    axes = [list(axes[r]) for r in range(len(ROWS))]
    ims = add_images(fig, axes, pc, [[shown[0, c, r] for c in range(len(sources))] for r in range(len(ROWS))],
                     norms, cmaps, labels)
    sup = fig.suptitle("", fontsize=12)

    def update_means(f):
        for r in range(len(ROWS)):
            for c in range(len(sources)):
                ims[r][c].set_data(shown[f, c, r])
        g = gmean[f, 0]
        sup.set_text(f"Column-mean dry mole fraction · {times[f]:%Y-%m-%d %H}z · GEOS-Chem global means: "
                     f"CO₂ {g[0]:.1f} ppm, fossil {g[1]:.2f} ppm, SF₆ {g[2]:.2f} ppt")
    render(f"{out}/catrine_benchmark_column_means.mp4", fig, update_means, len(times), args.fps)

    # Differences: dataset - GEOS-Chem.
    diff = maps[:, 1:] - maps[:, :1]
    dnorms = [symmetric_norm(diff[:, :, r], 99.5) for r in range(len(ROWS))]
    fig, axes, pc = panel_grid(len(ROWS), len(datasets), [f"{s} − GEOS-Chem" for s in sources[1:]])
    axes = [list(axes[r]) for r in range(len(ROWS))]
    ims_d = add_images(fig, axes, pc, [[diff[0, c, r] for c in range(len(datasets))] for r in range(len(ROWS))],
                       dnorms, ["RdBu_r"] * len(ROWS), [f"{t}\n({u})" for (_, t, _), u in zip(ROWS, units)])
    sup_d = fig.suptitle("", fontsize=12)

    def update_diff(f):
        for r in range(len(ROWS)):
            for c in range(len(datasets)):
                ims_d[r][c].set_data(diff[f, c, r])
        sup_d.set_text(f"Column-mean difference to GEOS-Chem · {times[f]:%Y-%m-%d %H}z")
    render(f"{out}/catrine_benchmark_differences.mp4", fig, update_diff, len(times), args.fps)


if __name__ == "__main__":
    main()
