#!/usr/bin/env python3
"""True global mass balance of the GEOS-Chem CATRINE run, from its own output.

The GEOS-Chem twin of catrine_true_mass_balance.jl. For every 3-hourly output
time t and tracer, the residual

    r(t) = B(t) - B(t0) - integral of E dt            (CO2, fossil CO2, SF6)
    r(t) = B(t) - Bhat(t),  Bhat' = E - lambda Bhat    (Rn-222)

is the mass GEOS-Chem gained (r > 0) or lost (r < 0) beyond its own emissions.
Everything comes from the CATRINE_inst files: B = sum(SpeciesConcVV * Met_AD)
in dry-air-equivalent storage units (kg), and E = sum(Emis* * Met_AREAM2)
(kg species / s) converted with GEOS-Chem's molar masses. The emission fields are
instantaneous: each is taken as the rate over the 3 h before its output time
(the emission step that ends there). Over months this convention only shifts
the cumulative source by one 3-hour slice.

  python3 scripts/diagnostics/catrine_true_mass_balance_gc.py \\
      --gc-dir ~/data/AtmosTransport/catrine-geoschem-runs/C90 --out gc.csv
"""
import argparse, datetime as dt, math, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from netCDF4 import Dataset

AIR_MW = 28.9644e-3                        # GEOS-Chem AIRMW (kg/mol)
# tracer: (concentration, emission, molar mass in kg/mol as in GEOS-Chem's species database)
SPECIES = {"co2_natural": ("SpeciesConcVV_CO2", "EmisCO2_Total", 44.01e-3),
           "co2_fossil": ("SpeciesConcVV_FossilCO2", "Emis_FossilCO2_Total", 44.01e-3),
           "sf6": ("SpeciesConcVV_SF6", "EmisSF6", 146.06e-3),
           "rn222": ("SpeciesConcVV_Rn222", "EmisRn_Soil", 222.0e-3)}
HALF_LIFE = {"rn222": 3.8235 * 86400.0}


def gc_path(gc_dir, t):
    name = f"GEOSChem.CATRINE_inst.{t:%Y%m%d}_{t:%H%M}z.nc4"
    nested = os.path.join(gc_dir, f"{t:%Y}", f"{t:%m}", name)
    return nested if os.path.exists(nested) else os.path.join(gc_dir, name)


def one(args):
    """Burdens (storage kg), emission rates (storage kg/s) and dry-air mass of one file."""
    gc_dir, t = args
    try:
        with Dataset(gc_path(gc_dir, t)) as d:
            ad = np.ma.filled(d["Met_AD"][0], 0.0).astype(float)
            area = np.ma.filled(d["Met_AREAM2"][0], 0.0).astype(float)
            out = {"air": float(ad.sum())}
            for n, (conc, emis, mw) in SPECIES.items():
                out[n] = float((np.ma.filled(d[conc][0], 0.0).astype(float) * ad).sum())
                out["E_" + n] = float((np.ma.filled(d[emis][0], 0.0).astype(float) * area).sum()) * AIR_MW / mw
        return t, out
    except (OSError, KeyError, IndexError) as e:
        print(f"unreadable {t}: {e}")
        return t, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gc-dir", default=os.path.expanduser("~/data/AtmosTransport/catrine-geoschem-runs/C90"))
    ap.add_argument("--start", default="2021-12-01T03:00")
    ap.add_argument("--end", default="2024-01-01T00:00")
    ap.add_argument("--origin", default="2021-12-01T00:00", help="time origin of the csv hours (the run start)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0, t1 = dt.datetime.fromisoformat(a.start), dt.datetime.fromisoformat(a.end)
    origin = dt.datetime.fromisoformat(a.origin)
    times = [t0 + dt.timedelta(hours=3 * k) for k in range(int((t1 - t0).total_seconds() // 10800) + 1)]
    with ProcessPoolExecutor(a.workers) as pool:
        res = dict(pool.map(one, [(a.gc_dir, t) for t in times], chunksize=16))
    bad = [t for t in times if res[t] is None]
    if bad:
        print(f"{len(bad)} unreadable files (rate taken from the next readable one; burden skipped):",
              ", ".join(str(t) for t in bad[:10]))
    ok = [t for t in times if res[t] is not None]

    def rate(t, n):                                     # emission rate over the 3 h before t
        later = [u for u in ok if u >= t]                # an unreadable file: the next readable rate,
        u = later[0] if later else ok[-1]                # or the last one at the end of the series
        return res[u]["E_" + n]

    rows, summary = [], {}
    state = {n: (0.0, res[ok[0]][n]) for n in SPECIES}  # (cumulative source, Bhat), from the first readable time
    for k, t in enumerate(times):
        if t > ok[0]:
            for n in SPECIES:
                c, ref = state[n]
                r, l = rate(t, n), (math.log(2) / HALF_LIFE[n] if n in HALF_LIFE else 0.0)
                c += r * 10800
                ref = ref * math.exp(-l * 10800) + r * (1 - math.exp(-l * 10800)) / l if l > 0 else ref + r * 10800
                state[n] = (c, ref)
        if res[t] is None:
            continue
        row = [(t - origin).total_seconds() / 3600, res[t]["air"]]
        for n in SPECIES:
            c, ref = state[n]
            row += [res[t][n], c, res[t][n] - ref]
            summary[n] = (res[t][n] - ref, c, res[t]["air"])
        rows.append(row)
    with open(a.out, "w") as fh:
        fh.write("hours,dry_air_kg," + ",".join(f"{n}_burden,{n}_source_cum,{n}_residual" for n in SPECIES) + "\n")
        for row in rows:
            fh.write(",".join(repr(x) for x in row) + "\n")
    print(f"GEOS-Chem true mass balance {ok[0]} .. {ok[-1]} ({len(ok)} times) -> {a.out}")
    for n, (r, c, air) in summary.items():
        print(f"  {n:12s} r = {r:+.4e} kg-storage   r/sum E = {r / c if c else float('nan'):+.3e}   r/air = {r / air:+.4e} mol/mol")


if __name__ == "__main__":
    main()
