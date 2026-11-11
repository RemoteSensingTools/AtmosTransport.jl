#!/usr/bin/env python3
"""Where on the cubed sphere the error sits: cells near a panel edge against the panel interiors.

    python3 examples/paper/idealized_advection/panel_edge_split.py --workdir <work dir> --results <dir>

For every untagged cubed-sphere FP64 run and its smooth tracer at T, the squared error is summed
separately over the cells within 7.5 degrees of a panel edge (N/12 cells on C<N>, 31 % of the
sphere) and over the remaining cells, both normalized by the global norm of the exact field, so
that `l2_edge**2 + l2_interior**2 = l2**2`. `panel_edge_split.csv` lists the three norms per run
and the observed order of each from the next coarser resolution. Equal orders in both regions
mean that the panel edges do not limit the convergence.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import netCDF4 as nc

from analyze import FULL, SMOOTH, grid_family, grid_spacing, is_current, snapshot_index, tomllib

EDGE_BAND_FRACTION = 12      # band width = N / 12 cells = 7.5 degrees


def split_norms(run_dir, reference, tracer):
    with nc.Dataset(reference) as ref, nc.Dataset(run_dir / "output.nc") as out:
        period = float(ref["time"][FULL])
        it = snapshot_index(np.asarray(out["time"][:]), period, run_dir.name)
        q = np.asarray(out[tracer][it, -1], dtype=np.float64)            # (panel, y, x)
        r = np.asarray(ref[tracer][FULL], dtype=np.float64).reshape(q.shape)
        area = np.asarray(ref["cell_area"][:], dtype=np.float64).reshape(q.shape)
    n = q.shape[-1]
    i = np.arange(n)
    from_edge = np.minimum(i, n - 1 - i)
    near_edge = np.minimum.outer(from_edge, from_edge) < n // EDGE_BAND_FRACTION
    e2 = area * (q - r) ** 2
    total = (area * r**2).sum()
    return (math.sqrt(e2.sum() / total), math.sqrt(e2[:, near_edge].sum() / total),
            math.sqrt(e2[:, ~near_edge].sum() / total))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workdir", required=True)
    p.add_argument("--results", required=True)
    a = p.parse_args()
    workdir, results = Path(a.workdir), Path(a.results)
    rows = []
    for line in open(workdir / "runs" / "runs.txt"):
        if not line.strip():
            continue
        run_dir = Path(line.strip()).parent
        meta = tomllib.load(open(run_dir / "run.toml", "rb"))
        if grid_family(meta["grid"]) != "cs" or meta["precision"] != "Float64" or meta.get("tag", ""):
            continue
        if not is_current(run_dir):
            raise RuntimeError(f"{run_dir.name}: run not finished or stale; rerun run_matrix.sh run")
        reference = workdir / "inputs" / meta["case"] / "reference.nc"
        l2, edge, interior = split_norms(run_dir, reference, SMOOTH[meta["flow"]])
        rows.append(dict(grid=meta["grid"], flow=meta["flow"], scheme=meta["scheme"],
                         spacing_deg=grid_spacing(meta["grid"]), l2=l2, l2_edge=edge, l2_interior=interior))
    rows.sort(key=lambda r: (r["flow"], r["scheme"], -r["spacing_deg"]))
    for r in rows:
        coarser = [c for c in rows if (c["flow"], c["scheme"]) == (r["flow"], r["scheme"])
                   and c["spacing_deg"] > r["spacing_deg"]]
        c = min(coarser, key=lambda c: c["spacing_deg"]) if coarser else None
        for key in ("l2", "l2_edge", "l2_interior"):
            r[f"order_{key}"] = (math.log(c[key] / r[key]) / math.log(c["spacing_deg"] / r["spacing_deg"])
                                 if c else float("nan"))
    with open(results / "panel_edge_split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(rows)} rows to {results / 'panel_edge_split.csv'}")


if __name__ == "__main__":
    main()
