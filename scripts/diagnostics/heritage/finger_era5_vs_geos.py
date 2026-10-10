#!/usr/bin/env python3
"""Quantify SH-UTLS tracer 'fingering': ABSOLUTE grid-scale roughness (Laplacian
RMS, in tracer units) of co2_natural over the SH (lat<-30) across the UTLS band,
at the last output time. ERA5 (user: no fingering) vs GEOS (fingers). The
ABSOLUTE Laplacian (not normalized by std) is the amplitude that actually shows
as visible fingering — the earlier normalized metric was misleading.
"""
import numpy as np
from netCDF4 import Dataset

RUNS = [
    ("GEOS adv-only",  "/home/cfranken/data/AtmosTransport/output/tropopause_iso/catrine_geosit_c180_ppm_advonly_co2nat_dec1-5.nc"),
    ("GEOS fullphys",  "/home/cfranken/data/AtmosTransport/output/catrine_geosit_c180_fullL72_ppm_cmfmc_dec2021_combined.nc"),
    ("ERA5 fullphys",  "/home/cfranken/data/AtmosTransport/output/catrine_era5_n320_ppm_dec2021_combined.nc"),
]

def sh_stats(field2d, lat, Nc):
    # field2d: (nf,Y,X). Vectorized 5-pt Laplacian over interior; SH mask lat<-30.
    laps = []; vals = []
    for p in range(6):
        f = field2d[p].astype(np.float64); m = lat[p] < -30.0
        lap = f[1:-1,1:-1] - 0.25*(f[1:-1,2:]+f[1:-1,:-2]+f[2:,1:-1]+f[:-2,1:-1])
        ms = m[1:-1,1:-1]
        laps.append(lap[ms]); vals.append(f[m])
    laps = np.concatenate(laps); vals = np.concatenate(vals)
    return np.sqrt(np.mean(laps**2)), np.std(vals), vals.mean()

# collect per-frac absolute Laplacian RMS for each run, to print a ratio
res = {}
for label, path in RUNS:
    try:
        ds = Dataset(path, "r")
    except Exception as e:
        print(f"{label}: OPEN FAILED ({e})"); continue
    lat = ds.variables["lats"][:]
    co2 = ds.variables["co2_natural"]
    nt, nz = co2.shape[0], co2.shape[1]; Nc = co2.shape[-1]; tlast = nt-1
    print(f"\n=== {label}  (Nz={nz}, time={nt}, t={tlast}) ===")
    print("  TOAfrac lev   |Lap|RMS      SHstd       |Lap|RMS/std   mean(SH)")
    res[label] = {}
    for frac in (0.48, 0.52, 0.56, 0.60, 0.64, 0.68):
        k = int(round(frac*(nz-1)))
        f = np.asarray(co2[tlast, k, :, :, :])
        rms, sd, mn = sh_stats(f, lat, Nc)
        res[label][frac] = rms
        print(f"   {frac:.2f}  {k:3d}   {rms:.3e}   {sd:.3e}   {rms/sd:8.4f}      {mn:.4e}")
    ds.close()

print("\n=== HEADLINE: GEOS/ERA5 absolute grid-noise ratio at SH-UTLS ===")
g_adv, g_full, e_full = "GEOS adv-only", "GEOS fullphys", "ERA5 fullphys"
if all(r in res for r in (g_full, e_full)):
    for frac in (0.48, 0.52, 0.56, 0.60):
        gv, ev = res[g_full].get(frac), res[e_full].get(frac)
        av = res.get(g_adv, {}).get(frac)
        if gv and ev:
            extra = f"  (GEOS adv-only/ERA5 {av/ev:.1f}x)" if av else ""
            print(f"  frac {frac:.2f}:  GEOSfull/ERA5 = {gv/ev:.1f}x{extra}")
print("\n(Absolute Laplacian RMS = visible grid-noise amplitude. GEOS>>ERA5 ⇒ "
      "fingering confirmed; ERA5 level is the fix target.)")
