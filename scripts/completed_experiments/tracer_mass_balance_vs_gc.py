#!/usr/bin/env python3
"""Tracer global mass-balance vs GEOS-Chem, per timestep. Decouples MASS
CONSERVATION (global burden) from SPATIAL agreement (R^2): if the global burden
ratio our/GC ~ 1 (same emission, mass conserved) while the map R^2 is low, the
disagreement is pure spatial redistribution, not a mass leak/source.

Global tracer burden (mass-proportional) = sum over all cells of VMR * air_mass:
  ours: sum(VMR_3d * air_mass_3d) ;  GC: sum(SpeciesConcVV * Met_AD).
Also reports our global dry-air-mass drift (continuity / mass closure).

  python3 scripts/completed_experiments/tracer_mass_balance_vs_gc.py <our_4tracer.nc> [ndays]
"""
import sys, datetime as dt, os
import numpy as np
from netCDF4 import Dataset
OUR = sys.argv[1]; NDAYS = int(sys.argv[2]) if len(sys.argv)>2 else 2
GCDIR = "/home/cfranken/data/AtmosTransport/catrine-geoschem-runs"
T0 = dt.datetime(2021,12,1)
TR = [("co2_natural","SpeciesConcVV_CO2"),("co2_fossil","SpeciesConcVV_FossilCO2"),
      ("sf6","SpeciesConcVV_SF6"),("rn222","SpeciesConcVV_Rn222")]
def gc_path(s): return f"{GCDIR}/GEOSChem.CATRINE_inst.{s:%Y%m%d}_{s:%H%M}z.nc4"

o = Dataset(OUR,"r")
nt = o.variables["co2_natural"].shape[0]
air = o.variables["air_mass"]                          # (t,lev,nf,Y,X) dry air mass
def _our_stamps(ds, n):
    # Read the actual output time axis (output may be hourly OR 3-hourly); do
    # NOT assume index==hour. Origin is the run start T0.
    if "time" in ds.variables:
        tv = np.asarray(ds.variables["time"][:], dtype=float)
        u = getattr(ds.variables["time"], "units", "").lower()
        per = 3600.0 if "hour" in u else 60.0 if "minute" in u else 1.0 if "second" in u else None
        if per is not None:
            return [T0 + dt.timedelta(seconds=float(tv[i]*per)) for i in range(n)]
    return [T0 + dt.timedelta(hours=h) for h in range(n)]   # fallback: assume hourly
stamps = _our_stamps(o, nt)
print("=== global tracer burden  our/GC ratio  (mass-proportional sum VMR*airmass) ===")
print(f"  {'UTC':<16}{'co2_nat':>10}{'co2_fos':>10}{'sf6':>10}{'rn222':>10}{'airmass(our,kg)':>18}")
for i,s in enumerate(stamps):
    if s >= T0+dt.timedelta(days=NDAYS) or not os.path.exists(gc_path(s)): continue
    am = np.asarray(air[i]); our_airmass = float(np.sum(am))
    row=[]
    try:                                              # skip unreadable GC files (e.g. corrupt 20211221_1200z)
        with Dataset(gc_path(s)) as g:
            ad = np.asarray(g.variables["Met_AD"][0])
            for ov,gv in TR:
                ours = float(np.sum(np.asarray(o.variables[ov][i]) * am))
                gc   = float(np.sum(np.asarray(g.variables[gv][0]) * ad))
                row.append(ours/gc if gc!=0 else np.nan)
    except Exception as e:
        print(f"  {s:%Y-%m-%d %H:%M}  (GC read failed, skipped: {e})"); continue
    print(f"  {s:%Y-%m-%d %H:%M}"+"".join(f"{r:>10.4f}" for r in row)+f"{our_airmass:>18.6e}")
o.close()
print("\nratio ~1.000 => global mass conserved & emission matches GC (low map-R^2 = pure")
print("spatial redistribution); ratio !=1 => a mass-balance / emission problem.")
