#!/usr/bin/env python3
"""Route-1 verdict: does deriving fluxes from MERRA-2 winds (+ Cameron-Smith
pressure-fix) remove the SH-UTLS tracer fingering that native GEOS MFXC carries?

Compares the ABSOLUTE grid-scale roughness (Laplacian RMS, tracer units) of
co2_natural over the SH (lat<-30) across the UTLS band, at the last output time,
for three MATCHED Dec-11 adv-only runs (same catrine_co2 IC + lmdz_co2 flux +
PPM, no diffusion/convection) — the only difference is the met binary:
  GEOS   = native cubed-sphere MFXC          (fingers)
  MERRA2 = lat-lon winds -> flux -> pjc_pfix  (the Route-1 fix under test)
  ERA5   = spectral winds -> flux -> balance  (the clean target)

MERRA2 and GEOS are both L72 -> identical levels (exact comparison). ERA5 is
L137 -> TOA-fraction aligned (approximate). Route 1 works if MERRA2/GEOS << 1
and approaches ERA5/GEOS at the UTLS.
"""
import numpy as np
from netCDF4 import Dataset

OUT = "/home/cfranken/data/AtmosTransport/output/route1_dec11"
RUNS = [
    ("GEOS",   f"{OUT}/geos_c180_advonly_co2nat_dec11.nc"),
    ("MERRA2", f"{OUT}/merra2_c180_advonly_co2nat_dec11.nc"),
    ("ERA5",   f"{OUT}/era5_c180_advonly_co2nat_dec11.nc"),
]
FRACS = (0.48, 0.52, 0.56, 0.60, 0.64, 0.68)   # ~ 50..360 hPa on L72/L137

def sh_stats(field2d, lat):
    laps = []; vals = []
    for p in range(6):
        f = field2d[p].astype(np.float64); m = lat[p] < -30.0
        lap = f[1:-1,1:-1] - 0.25*(f[1:-1,2:]+f[1:-1,:-2]+f[2:,1:-1]+f[:-2,1:-1])
        laps.append(lap[m[1:-1,1:-1]]); vals.append(f[m])
    laps = np.concatenate(laps); vals = np.concatenate(vals)
    return np.sqrt(np.mean(laps**2)), np.std(vals), vals.mean()

res = {}
for label, path in RUNS:
    try:
        ds = Dataset(path, "r")
    except Exception as e:
        print(f"{label}: OPEN FAILED ({e})"); continue
    lat = ds.variables["lats"][:]
    co2 = ds.variables["co2_natural"]
    nt, nz = co2.shape[0], co2.shape[1]; tlast = nt-1
    print(f"\n=== {label}  (Nz={nz}, time={nt}, t={tlast}) ===")
    print("  frac lev   |Lap|RMS      SHstd       rel(/std)   mean(SH)")
    res[label] = {}
    for frac in FRACS:
        k = int(round(frac*(nz-1)))
        rms, sd, mn = sh_stats(np.asarray(co2[tlast, k, :, :, :]), lat)
        res[label][frac] = (rms, rms/sd if sd > 0 else np.nan)
        print(f"  {frac:.2f} {k:3d}   {rms:.3e}   {sd:.3e}   {rms/sd:8.4f}   {mn:.4e}")
    ds.close()

print("\n=== HEADLINE: SH-UTLS grid-noise vs GEOS native (absolute |Lap|RMS) ===")
print("  frac    MERRA2/GEOS    ERA5/GEOS    (want MERRA2<1, near ERA5)")
for frac in FRACS:
    g = res.get("GEOS", {}).get(frac)
    m = res.get("MERRA2", {}).get(frac)
    e = res.get("ERA5", {}).get(frac)
    if g and m and e and g[0] > 0:
        print(f"  {frac:.2f}     {m[0]/g[0]:.3f}          {e[0]/g[0]:.3f}")
print("\n(MERRA2/GEOS << 1 ⇒ wind-derived fluxes remove the fingering; the residual"
      "\n gap to ERA5/GEOS is the gridpoint-vs-spectral wind-smoothness difference.)")
