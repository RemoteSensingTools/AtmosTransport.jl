#!/usr/bin/env python3
"""Column-averaged (dry-air-mass-weighted) mixing-ratio MAPS, ours vs GEOS-Chem,
for the 4 CATRINE tracers at a few day-2 timesteps — to see the spatial pattern
behind the R^2/slope, and a tracer-mass-conservation check (total burden vs
integrated emission). Cube cells are binned to a lat-lon grid for plotting.

  python3 scripts/completed_experiments/plot_column_mean_maps.py <our_4tracer.nc> [HH HH ...]
"""
import sys, datetime as dt, os
import numpy as np
from netCDF4 import Dataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUR = sys.argv[1]
HOURS = [int(h) for h in sys.argv[2:]] or [6, 12, 18]     # day-2 UTC hours
GCDIR = "/home/cfranken/data/AtmosTransport/catrine-geoschem-runs"
OUTDIR = os.path.expanduser("~/www/catrine"); os.makedirs(OUTDIR, exist_ok=True)
T0 = dt.datetime(2021, 12, 1)
TRACERS = [("co2_natural","SpeciesConcVV_CO2",1e6,"ppm"),
           ("co2_fossil","SpeciesConcVV_FossilCO2",1e6,"ppm"),
           ("sf6","SpeciesConcVV_SF6",1e12,"ppt"),
           ("rn222","SpeciesConcVV_Rn222",1e21,"1e-21")]
NLON, NLAT = 180, 90                                       # 2-deg bin grid for maps
def gc_path(s): return f"{GCDIR}/GEOSChem.CATRINE_inst.{s:%Y%m%d}_{s:%H%M}z.nc4"

def bin_latlon(lons, lats, vals):
    lo = ((lons + 180) % 360); la = lats
    H,_,_ = np.histogram2d(la.ravel(), lo.ravel(), bins=[NLAT,NLON],
                           range=[[-90,90],[0,360]], weights=vals.ravel())
    C,_,_ = np.histogram2d(la.ravel(), lo.ravel(), bins=[NLAT,NLON], range=[[-90,90],[0,360]])
    with np.errstate(invalid="ignore"): return H/np.where(C>0,C,np.nan)

def our_time_index(ds, stamp):
    # Map a UTC stamp to the output index via the actual time axis (output may
    # be hourly OR 3-hourly); do NOT assume index==hour. Origin is T0.
    nt = ds.variables["co2_natural"].shape[0]
    if "time" in ds.variables:
        tv = np.asarray(ds.variables["time"][:], dtype=float)
        u = getattr(ds.variables["time"], "units", "").lower()
        per = 3600.0 if "hour" in u else 60.0 if "minute" in u else 1.0 if "second" in u else None
        if per is not None:
            want = (stamp - T0).total_seconds()
            i = int(np.argmin(np.abs(tv*per - want)))
            return i if abs(tv[i]*per - want) < 1.0 else None
    h = int((stamp - T0).total_seconds()//3600)          # fallback: assume hourly
    return h if 0 <= h < nt else None

def main():
    o = Dataset(OUR,"r")
    lons = np.asarray(o.variables["lons"][:]); lats = np.asarray(o.variables["lats"][:])  # (nf,Y,X)
    stamps = [dt.datetime(2021,12,2,h) for h in HOURS]
    for ov, gv, sc, unit in TRACERS:
        colv = f"{ov}_column_mean"
        if colv not in o.variables: print("no", colv); continue
        fig, axes = plt.subplots(len(stamps), 3, figsize=(13, 3.2*len(stamps)), squeeze=False)
        for r, s in enumerate(stamps):
            ti = our_time_index(o, s)
            if ti is None or not os.path.exists(gc_path(s)):
                for c in range(3): axes[r][c].set_visible(False); continue
            our_col = np.asarray(o.variables[colv][ti]) * sc                      # (nf,Y,X)
            try:                                                                  # skip unreadable GC files (corrupt 20211221_1200z)
                with Dataset(gc_path(s)) as g:
                    vmr = np.asarray(g.variables[gv][0]); ad = np.asarray(g.variables["Met_AD"][0])
            except Exception:
                for c in range(3): axes[r][c].set_visible(False)
                continue
            gc_col = (np.sum(vmr*ad,axis=0)/np.sum(ad,axis=0)) * sc               # (nf,Y,X)
            mo = bin_latlon(lons,lats,our_col); mg = bin_latlon(lons,lats,gc_col)
            vmin, vmax = np.nanpercentile(np.concatenate([mo.ravel(),mg.ravel()]),[2,98])
            dmax = np.nanpercentile(np.abs(mo-mg),98)
            for c,(M,ttl,cmap,vlim) in enumerate([
                    (mo,"OURS","viridis",(vmin,vmax)),(mg,"GEOS-Chem","viridis",(vmin,vmax)),
                    (mo-mg,"OURS - GC","RdBu_r",(-dmax,dmax))]):
                ax=axes[r][c]; im=ax.imshow(M,origin="lower",extent=[-180,180,-90,90],
                    aspect="auto",cmap=cmap,vmin=vlim[0],vmax=vlim[1])
                if r==0: ax.set_title(ttl,fontsize=11)
                if c==0: ax.set_ylabel(f"{s:%m-%d %H}z",fontsize=9)
                ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im,ax=ax,shrink=0.8,pad=0.01)
        fig.suptitle(f"{ov} column-mean ({unit}) — Dec 2 — ours vs GEOS-Chem", fontsize=13)
        p=f"{OUTDIR}/colmap_{ov}.png"; fig.savefig(p,dpi=95,bbox_inches="tight"); plt.close(fig)
        print("wrote", p)
    o.close()

if __name__=="__main__": main()
