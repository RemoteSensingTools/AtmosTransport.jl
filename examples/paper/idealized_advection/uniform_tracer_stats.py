#!/usr/bin/env python3
"""How far a uniform tracer leaves its initial value, in absolute terms and in units of the rounding step.

    python3 examples/paper/idealized_advection/uniform_tracer_stats.py --workdir <work dir> --results <dir>

For the uniform tracer (mole fraction 1) of every untagged run, `uniform_tracer_stats.csv` lists

- `max_abs`: largest |chi - 1| of all output times up to T and both output levels (as in `norms.csv`);
- `rms`, `p99`: area-weighted root-mean-square and 99th percentile of |chi - 1| at T (bottom output level);
- `area_fraction_above_1e-4`: fraction of the sphere with |chi - 1| > 1e-4 at T;
- `mass_drift_max`: largest relative change of the global tracer mass (the model's budget);
- `*_in_eps`: the same numbers divided by the machine epsilon of the run's precision
  (2.2e-16 for Float64, 1.2e-7 for Float32);
- `*_fp32_over_fp64_in_eps` (Float32 rows): the run's value in its epsilon divided by that of its
  Float64 twin in its epsilon (empty where the Float64 value is zero).

The `*_in_eps` columns put the two precisions on one scale. The absolute deviations of a Float32
run and its Float64 twin differ by about the ratio of the two epsilons (5.4e8); the script prints
the distribution of the twin ratios. Maxima are those of the saved snapshots (not of every
internal step) and of the two output levels.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import netCDF4 as nc

from analyze import FULL, grid_family, is_current, snapshot_index, tomllib

EPS = {"Float32": float(np.finfo(np.float32).eps), "Float64": float(np.finfo(np.float64).eps)}
AREA_THRESHOLD = 1e-4


def run_stats(run_dir, reference):
    with nc.Dataset(reference) as ref, nc.Dataset(run_dir / "output.nc") as out:
        period = float(ref["time"][FULL])
        it = snapshot_index(np.asarray(out["time"][:]), period, run_dir.name)
        q = np.asarray(out["uniform"][: it + 1], dtype=np.float64)
        q = q.reshape(q.shape[0], q.shape[1], -1)                      # (time, level, cell)
        budget = np.asarray(out["uniform_total_mass"][: it + 1], dtype=np.float64)
        area = np.asarray(ref["cell_area"][:], dtype=np.float64).reshape(-1)
    d = np.abs(q[-1, -1] - 1)
    w = area / area.sum()
    order = np.argsort(d)
    p99 = d[order][min(np.searchsorted(np.cumsum(w[order]), 0.99), d.size - 1)]
    return dict(max_abs=np.abs(q - 1).max(), rms=math.sqrt((w * d**2).sum()), p99=p99,
                area_fraction=w[d > AREA_THRESHOLD].sum(), mass_drift_max=np.abs(budget / budget[0] - 1).max())


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
        if meta.get("tag", ""):
            continue
        if not is_current(run_dir):
            raise RuntimeError(f"{run_dir.name}: run not finished or stale; rerun run_matrix.sh run")
        s = run_stats(run_dir, workdir / "inputs" / meta["case"] / "reference.nc")
        eps = EPS[meta["precision"]]
        rows.append({"run_id": meta["run_id"], "family": grid_family(meta["grid"]), "grid": meta["grid"],
                     "flow": meta["flow"], "scheme": meta["scheme"], "precision": meta["precision"],
                     "max_abs": s["max_abs"], "rms": s["rms"], "p99": s["p99"],
                     "area_fraction_above_1e-4": s["area_fraction"], "mass_drift_max": s["mass_drift_max"],
                     "max_abs_in_eps": s["max_abs"] / eps, "rms_in_eps": s["rms"] / eps,
                     "mass_drift_max_in_eps": s["mass_drift_max"] / eps})
    by_id = {r["run_id"]: r for r in rows}
    ratio_keys = ("max_abs", "rms", "mass_drift_max")
    for r in rows:
        twin = by_id.get(r["run_id"][: -len("_f32")] + "_f64") if r["precision"] == "Float32" else None
        for key in ratio_keys:
            both = twin is not None and twin[f"{key}_in_eps"] > 0
            r[f"{key}_fp32_over_fp64_in_eps"] = r[f"{key}_in_eps"] / twin[f"{key}_in_eps"] if both else ""
    for key in ratio_keys:
        v = np.array([r[f"{key}_fp32_over_fp64_in_eps"] for r in rows if r[f"{key}_fp32_over_fp64_in_eps"] != ""])
        print(f"{key}, FP32 / FP64 in eps over {v.size} twins: median {np.median(v):.2f}, "
              f"quartiles {np.quantile(v, 0.25):.2f}-{np.quantile(v, 0.75):.2f}, range {v.min():.2f}-{v.max():.2f}")
    with open(results / "uniform_tracer_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(rows)} rows to {results / 'uniform_tracer_stats.csv'}")


if __name__ == "__main__":
    main()
