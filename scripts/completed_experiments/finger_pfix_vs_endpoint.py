#!/usr/bin/env python3
"""Does pfix_corrected reduce the SH-UTLS tracer fingering vs the endpoint
closure? Same period (Dec-11), same IC, same advection — only the cm closure
differs. ABSOLUTE grid-scale Laplacian RMS of co2_natural over SH (lat<-30),
UTLS band, last output time."""
import numpy as np
from netCDF4 import Dataset

RUNS = [
    ("endpoint (Path A)", "/home/cfranken/data/AtmosTransport/output/pfix_test/advonly_endpoint_dec11.nc"),
    ("pfix_corrected",    "/home/cfranken/data/AtmosTransport/output/pfix_test/advonly_pfixcorrected_dec11.nc"),
]

def sh_stats(f, lat, Nc):
    laps=[]; vals=[]
    for p in range(6):
        a=f[p].astype(np.float64); m=lat[p]<-30.0
        lap=a[1:-1,1:-1]-0.25*(a[1:-1,2:]+a[1:-1,:-2]+a[2:,1:-1]+a[:-2,1:-1])
        laps.append(lap[m[1:-1,1:-1]]); vals.append(a[m])
    laps=np.concatenate(laps); vals=np.concatenate(vals)
    return np.sqrt(np.mean(laps**2)), np.std(vals)

res={}
for label,path in RUNS:
    try: ds=Dataset(path,"r")
    except Exception as e: print(f"{label}: OPEN FAILED ({e})"); continue
    lat=ds.variables["lats"][:]; co2=ds.variables["co2_natural"]
    nt,nz=co2.shape[0],co2.shape[1]; Nc=co2.shape[-1]; t=nt-1
    print(f"\n=== {label} (Nz={nz} t={t}) ===")
    print("  TOAfrac lev   |Lap|RMS     SHstd      |Lap|/std")
    res[label]={}
    for frac in (0.48,0.52,0.56,0.60,0.64):
        k=int(round(frac*(nz-1))); f=np.asarray(co2[t,k,:,:,:])
        rms,sd=sh_stats(f,lat,Nc); res[label][frac]=(rms,sd)
        print(f"   {frac:.2f}  {k:3d}   {rms:.3e}   {sd:.3e}   {rms/sd:7.4f}")
    ds.close()

a,b="endpoint (Path A)","pfix_corrected"
if a in res and b in res:
    print("\n=== pfix_corrected / endpoint (grid-noise ratio; <1 = fingering reduced) ===")
    for frac in (0.48,0.52,0.56,0.60,0.64):
        ra=res[a][frac][0]; rb=res[b][frac][0]
        na=res[a][frac][0]/res[a][frac][1]; nb=res[b][frac][0]/res[b][frac][1]
        print(f"  frac {frac:.2f}:  |Lap| ratio = {rb/ra:.3f}   normalized {na:.4f}->{nb:.4f}")
