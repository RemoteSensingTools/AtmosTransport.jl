#!/usr/bin/env python3
"""How much closer to GEOS-Chem? Compare our adv-only co2_natural runs (started
from the shared Dec-1 IC = GEOS-Chem Dec-1 0000z state) to the GEOS-Chem
reference SpeciesConcVV_CO2 at the SH-UTLS over Dec 2-6, for the three met
sources. The fingering inflates BOTH the RMS difference to GEOS-Chem and the
grid-scale noise of the (ours - GEOS-Chem) difference; MERRA-2 should track
GEOS-Chem far better than native GEOS-IT MFXC.

Both fields are native C180 cubed-sphere (no regrid). GEOS-Chem is surface-first
(lev=1=surface); our output is TOA-first -- so GEOS-Chem lev is flipped. Levels
are compared by TOA-fraction so L72 (GEOS/MERRA2) and L137 (ERA5) align. At the
SH-UTLS surface fluxes have not arrived, so co2_natural ~ total CO2 there (both
= the transported Dec-1 IC), making the field difference well-posed.
"""
import numpy as np
from netCDF4 import Dataset

OUT = "/home/cfranken/data/AtmosTransport/output/route1_dec1-5"
GC  = "/home/cfranken/data/AtmosTransport/catrine-geoschem-runs"
RUNS = [
    ("GEOS",   "/home/cfranken/data/AtmosTransport/output/tropopause_iso/catrine_geosit_c180_ppm_advonly_co2nat_dec1-5.nc"),
    ("MERRA2", f"{OUT}/merra2_c180_advonly_co2nat_dec1-5.nc"),
    ("ERA5",   f"{OUT}/era5_c180_advonly_co2nat_dec1-5.nc"),
]
DAYS = ["20211202", "20211203", "20211204", "20211205", "20211206"]  # 00z compare points
FRACS = (0.52, 0.56, 0.60)   # ~ 85, 139, 230 hPa (the UTLS finger band)

def sh_mask(lat): return lat < -30.0

def lev_at(co2, frac):
    nz = co2.shape[0]
    return co2[int(round(frac*(nz-1)))]

def gc_field(date):
    p = f"{GC}/GEOSChem.CATRINE_inst.{date}_0000z.nc4"
    ds = Dataset(p, "r")
    c = np.asarray(ds.variables["SpeciesConcVV_CO2"][0])  # (lev,nf,Y,X) surface-first
    lat = np.asarray(ds.variables["lats"][:])
    ds.close()
    return c[::-1], lat  # flip -> TOA-first

def sh_diff_stats(ours, gc, lat, frac):
    fo = lev_at(ours, frac); fg = lev_at(gc, frac)
    laps = []; diffs = []; gcvals = []
    for p in range(6):
        d = (fo[p].astype(np.float64) - fg[p].astype(np.float64))
        m = sh_mask(lat[p])
        lap = d[1:-1,1:-1] - 0.25*(d[1:-1,2:]+d[1:-1,:-2]+d[2:,1:-1]+d[:-2,1:-1])
        laps.append(lap[m[1:-1,1:-1]]); diffs.append(d[m]); gcvals.append(fg[p][m])
    diffs = np.concatenate(diffs); laps = np.concatenate(laps); gcvals = np.concatenate(gcvals)
    return np.sqrt(np.mean(diffs**2)), np.sqrt(np.mean(laps**2)), np.std(gcvals)

# open our runs once
data = {}
for label, path in RUNS:
    try:
        ds = Dataset(path, "r")
        data[label] = (ds, np.asarray(ds.variables["lats"][:]), ds.variables["co2_natural"])
    except Exception as e:
        print(f"{label}: OPEN FAILED ({e})")

for frac in FRACS:
    print(f"\n=== SH-UTLS frac={frac:.2f} : RMS(ours - GEOS-Chem) [ppm] | grid-noise of diff [ppm] ===")
    print(f"  {'date':>10}   {'GEOS':>16} {'MERRA2':>16} {'ERA5':>16}")
    for di, date in enumerate(DAYS):
        gc, gclat = gc_field(date)
        tidx = 8*(di+1)            # 24h*(di+1), snapshots every 3h
        row_rms = {}; row_gn = {}
        for label in ("GEOS","MERRA2","ERA5"):
            if label not in data: continue
            ds, lat, co2 = data[label]
            if tidx >= co2.shape[0]: continue
            ours = np.asarray(co2[tidx])  # (lev,nf,Y,X) TOA-first
            rms, gn, gcstd = sh_diff_stats(ours, gc, lat, frac)
            row_rms[label] = rms*1e6; row_gn[label] = gn*1e6   # VMR -> ppm
        def fmt(lbl): return f"{row_rms.get(lbl,float('nan')):6.3f}|{row_gn.get(lbl,float('nan')):6.3f}".rjust(16)
        print(f"  {date[:8]:>10}   {fmt('GEOS')} {fmt('MERRA2')} {fmt('ERA5')}")
    print("  (lower RMS = closer to GEOS-Chem; lower grid-noise = less fingering in the diff)")
print("\nHeadline: MERRA2 RMS/grid-noise vs GEOS ratio at the SH-UTLS = how much closer Route 1 gets us to GEOS-Chem.")
