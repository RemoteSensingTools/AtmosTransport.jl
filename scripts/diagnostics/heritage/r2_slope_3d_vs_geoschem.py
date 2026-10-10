#!/usr/bin/env python3
"""Full-3D R^2 + regression slope of our run vs GEOS-Chem, per tracer, per
timestep, over the first N days. The pre-campaign validation gate: R^2 -> 1 and
slope -> 1 means our 3D distribution tracks GC cell-for-cell.

For each of the 4 tracers we regress OUR field on GC's over EVERY grid cell of
the full 3D field (lev x nf x Y x X) at each matched time:
    our_cell ~ a + b * gc_cell    ->    slope b, intercept a, R^2 = corr^2
Both are mol/mol dry on the native C180 cube; our output is TOA-first
(lev[0]=model top), GC is surface-first, so GC's vertical axis is flipped to
align cell-for-cell before flattening.

  python3 scripts/diagnostics/heritage/r2_slope_3d_vs_geoschem.py <our_4tracer.nc> [ndays]
"""
import sys, datetime as dt
import numpy as np
from netCDF4 import Dataset
import os

OUR = sys.argv[1]
NDAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SKIP  = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # skip spin-up day(s); correlate from day SKIP+1 on
GCDIR = "/home/cfranken/data/AtmosTransport/catrine-geoschem-runs"
T0 = dt.datetime(2021, 12, 1, 0, 0)

# (our var, GC var) per tracer; GC is surface-first, ours TOA-first.
TRACERS = [
    ("co2_natural", "SpeciesConcVV_CO2"),
    ("co2_fossil",  "SpeciesConcVV_FossilCO2"),
    ("sf6",         "SpeciesConcVV_SF6"),
    ("rn222",       "SpeciesConcVV_Rn222"),
]

def gc_path(stamp):  # stamp = datetime
    return f"{GCDIR}/GEOSChem.CATRINE_inst.{stamp:%Y%m%d}_{stamp:%H%M}z.nc4"

def our_time_axis(ds):
    """Return our output stamps as datetimes. Assumes start = T0; cadence from
    the time dim count over the run length (hourly or 3-hourly)."""
    nt = ds.variables[TRACERS[0][0]].shape[0]
    # infer cadence from a 'time' var if present, else assume hourly
    if "time" in ds.variables:
        tv = np.asarray(ds.variables["time"][:], dtype=float)
        # CF 'minutes since' or 'hours since' — fall back to even spacing
        units = getattr(ds.variables["time"], "units", "")
        per = 60.0 if "hour" in units else (1.0 if "minute" in units else None)
        if per is not None:
            return [T0 + dt.timedelta(minutes=float(tv[i]*per)) for i in range(nt)]
    # fallback: assume hourly from T0
    return [T0 + dt.timedelta(hours=i) for i in range(nt)]

def r2_slope(our, gc):
    """R^2 and slope of (our ~ a + b*gc) over all finite cells."""
    x = gc.ravel().astype(np.float64); y = our.ravel().astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10 or np.std(x) == 0:
        return np.nan, np.nan, x.size
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x-xm)**2); sxy = np.sum((x-xm)*(y-ym))
    b = sxy/sxx                                  # slope
    r = sxy/np.sqrt(sxx*np.sum((y-ym)**2))       # corr
    return r*r, b, x.size

def main():
    ods = Dataset(OUR, "r")
    stamps = our_time_axis(ods)
    start = T0 + dt.timedelta(days=SKIP)
    end   = T0 + dt.timedelta(days=NDAYS)
    # keep stamps in [day SKIP+1 .. NDAYS] with a GC file (skip the spin-up day)
    pairs = [(i, s) for i, s in enumerate(stamps)
             if start <= s < end and os.path.exists(gc_path(s))]
    if not pairs:
        print(f"no matched our/GC timesteps in days {SKIP+1}..{NDAYS}"); return

    print(f"our={os.path.basename(OUR)}  days {SKIP+1}..{NDAYS} (spin-up day(s) skipped)  matched timesteps={len(pairs)}")
    summ = {our_v: {"3d": ([], []), "col": ([], [])} for our_v, _ in TRACERS}
    for our_v, gc_v in TRACERS:
        colvar = f"{our_v}_column_mean"      # our precomputed dry-dp-weighted column mean
        have_col = colvar in ods.variables
        print(f"\n=== {our_v}  vs  {gc_v}    (full 3D | column-mean) ===")
        print(f"  {'UTC':<16}{'R2_3d':>9}{'slope':>8}  | {'R2_col':>9}{'slope':>8}")
        for i, s in pairs:
            our3d = np.asarray(ods.variables[our_v][i])          # (lev,nf,Y,X) TOA-first
            try:
                with Dataset(gc_path(s), "r") as g:
                    gc_nat = np.asarray(g.variables[gc_v][0])    # (lev,nf,Y,X) surface-first
                    ad     = np.asarray(g.variables["Met_AD"][0])# dry air mass per box (weight)
            except Exception as e:
                print(f"  {s:%m-%d %H:%M}  (GC read failed: {e})"); continue
            gc3d = gc_nat[::-1]                                   # -> TOA-first to align with ours
            # Full-3D R^2 only when the vertical grids match (geosit L72 == GC L72).
            # ERA5 is L137 != GC L72, so 3D is N/A there; the column-mean (a
            # vertical integral, grid-agnostic on the shared C180 horizontal) is
            # the valid metric and is computed regardless.
            if gc3d.shape == our3d.shape:
                r2, slope, _ = r2_slope(our3d, gc3d)
                summ[our_v]["3d"][0].append(r2); summ[our_v]["3d"][1].append(slope)
                c3 = f"{r2:>9.4f}{slope:>8.3f}"
            else:
                c3 = f"{'(L%d vs L%d)' % (our3d.shape[0], gc3d.shape[0]):>17}"
            # column-mean: GC = sum_k(VMR*AD)/sum_k(AD) [order-free vertical sum], ours = precomputed
            if have_col:
                gc_col = np.sum(gc_nat * ad, axis=0) / np.sum(ad, axis=0)   # (nf,Y,X)
                our_col = np.asarray(ods.variables[colvar][i])             # (nf,Y,X)
                r2c, slc, _ = r2_slope(our_col, gc_col)
                summ[our_v]["col"][0].append(r2c); summ[our_v]["col"][1].append(slc)
                print(f"  {s:%Y-%m-%d %H:%M} {c3}  | {r2c:>9.4f}{slc:>8.3f}")
            else:
                print(f"  {s:%Y-%m-%d %H:%M} {c3}  |   (no {colvar})")
    ods.close()

    print(f"\n=== SUMMARY (mean over day {SKIP+1}..{NDAYS} timesteps, spin-up skipped) ===")
    print(f"  {'tracer':<14}{'R2_3d':>9}{'slope_3d':>10}  | {'R2_col':>9}{'slope_col':>11}")
    for our_v, _ in TRACERS:
        d = summ[our_v]
        r3 = f"{np.nanmean(d['3d'][0]):>9.4f}{np.nanmean(d['3d'][1]):>10.4f}" if d['3d'][0] else f"{'-':>9}{'-':>10}"
        rc = f"{np.nanmean(d['col'][0]):>9.4f}{np.nanmean(d['col'][1]):>11.4f}" if d['col'][0] else f"{'-':>9}{'-':>11}"
        print(f"  {our_v:<14}{r3}  | {rc}")
    print("\nColumn-mean = dry-air-mass-weighted vertical average (the satellite-observable). GO")
    print("when R2~1 & slope~1; the column-mean integrates out PBL-level mixing differences.")

if __name__ == "__main__":
    main()
