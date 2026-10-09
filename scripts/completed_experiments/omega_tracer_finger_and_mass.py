#!/usr/bin/env python3
"""OMEGA-consistent closure: TRACER-level fingering (deliverable 4) + tracer
mass conservation (deliverable 5c), vs DIRTY (native GEOS) and CLEAN (MERRA-2).

Deliverable 4 — SH-UTLS (lat<-30, 80-300 hPa) grid-scale Laplacian roughness of
the advected co2_natural at matched times. Same metric as finger_route1_dec11.py:
  |Lap|RMS = sqrt(mean( (f - 0.25*sum 4-neighbours)^2 )) over SH cells,
  rel = |Lap|RMS / std(SH field).  Lower = less fingering.

Deliverable 5c — global atmospheric CO2 burden Σ(co2_vmr * air_mass) per output
time. Advection conserves tracer mass, so the burden change over the run must
equal the integrated surface emission. The emission is identical across the
three runs (same IC + lmdz_co2 flux), so the OMEGA burden GAIN must match DIRTY's
to roundoff — a clean closure proof that the OMEGA cm does not create/destroy
tracer mass. We report absolute burdens + the OMEGA-vs-DIRTY gain agreement.

L72 hybrid coefficients give pmid(k) at a reference PS for the 80-300 hPa band.

Usage:
  python3 scripts/completed_experiments/omega_tracer_finger_and_mass.py
"""
import numpy as np
from netCDF4 import Dataset
import os, re

RUNS = [
    ("OMEGA", os.path.expanduser("~/data/AtmosTransport/output/route1_dec1-5/omega_c180_advonly_co2nat_dec1-3.nc")),
    ("DIRTY", os.path.expanduser("~/data/AtmosTransport/output/tropopause_iso/catrine_geosit_c180_ppm_advonly_co2nat_dec1-5.nc")),
    ("CLEAN", os.path.expanduser("~/data/AtmosTransport/output/route1_dec1-5/merra2_c180_advonly_co2nat_dec1-5.nc")),
]
COEF = os.path.expanduser("~/code/gitHub/AtmosTransportModel/config/geos_L72_coefficients.toml")
UTLS_LO, UTLS_HI = 80.0, 300.0      # hPa
SH_LAT = -30.0
PS_REF = 1.0e5                       # Pa

def _grab_array(text, name):
    # match `name = [ ... ]` (multi-line) and parse the float list
    m = re.search(rf"(?m)^\s*{name}\s*=\s*\[(.*?)\]", text, re.S)
    if not m:
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
    return np.asarray([float(x) for x in nums], float)

def load_pmid_hpa():
    with open(COEF) as f:
        text = f.read()
    A = _grab_array(text, "a")
    if A is None: A = _grab_array(text, "A")
    B = _grab_array(text, "b")
    if B is None: B = _grab_array(text, "B")
    if A is None or B is None:
        return None
    nz = len(A) - 1
    pmid = np.array([0.5*((A[k]+A[k+1]) + (B[k]+B[k+1])*PS_REF)/100.0 for k in range(nz)])
    return pmid

def sh_stats(field2d, lat):
    """field2d: (6,Nc,Nc). Return (|Lap|RMS, std, mean) over SH interior cells."""
    laps = []; vals = []
    for p in range(6):
        f = field2d[p].astype(np.float64); m = lat[p] < SH_LAT
        lap = f[1:-1,1:-1] - 0.25*(f[1:-1,2:]+f[1:-1,:-2]+f[2:,1:-1]+f[:-2,1:-1])
        laps.append(lap[m[1:-1,1:-1]]); vals.append(f[m])
    laps = np.concatenate(laps); vals = np.concatenate(vals)
    return np.sqrt(np.mean(laps**2)), np.std(vals), vals.mean()

def main():
    pmid = load_pmid_hpa()
    print("pmid available:", pmid is not None)
    data = {}
    for label, path in RUNS:
        try:
            ds = Dataset(path, "r")
        except Exception as e:
            print(f"{label}: OPEN FAILED ({e})"); continue
        nz = ds.variables["co2_natural"].shape[1]
        if pmid is not None and len(pmid) == nz:
            utls_k = [k for k in range(nz) if UTLS_LO <= pmid[k] <= UTLS_HI]
        else:
            # fallback: fractional band ~80-300 hPa on a TOA-first L72
            utls_k = [k for k in range(nz) if 0.46 <= k/(nz-1) <= 0.70]
        data[label] = dict(ds=ds, nz=nz, utls_k=utls_k,
                           lat=ds.variables["lats"][:],
                           nt=ds.variables["co2_natural"].shape[0])
        print(f"{label}: Nz={nz} nt={data[label]['nt']} UTLS levels={len(utls_k)} "
              f"k={utls_k[0]}..{utls_k[-1]}")

    # ---- deliverable 4: SH-UTLS tracer fingering at matched times -------------
    # Dec 1-3 OMEGA has fewer times than Dec 1-5 DIRTY/CLEAN; match by output
    # index from t0 (all are 3-hourly from Dec-1 00Z, same IC), capped to OMEGA's
    # length, and report at the last common time + a UTLS-mean over all common t.
    common_nt = min(d["nt"] for d in data.values())
    tlast = common_nt - 1
    print(f"\n=== DELIVERABLE 4: SH-UTLS tracer fingering ===")
    print(f"matched times: 0..{tlast} (3-hourly from Dec-1 00Z); headline at t={tlast}")
    band_rms = {}; band_rel = {}
    for label, d in data.items():
        ds = d["ds"]; co2 = ds.variables["co2_natural"]; lat = d["lat"]
        # UTLS-band |Lap|RMS averaged over the band, at t=tlast.
        rmss = []; rels = []
        for k in d["utls_k"]:
            rms, sd, mn = sh_stats(np.asarray(co2[tlast, k, :, :, :]), lat)
            rmss.append(rms); rels.append(rms/sd if sd > 0 else np.nan)
        band_rms[label] = np.mean(rmss); band_rel[label] = np.nanmean(rels)
        print(f"  {label}:  band-mean |Lap|RMS = {band_rms[label]:.4e}   "
              f"rel(/std) = {band_rel[label]:.4f}")

    if "DIRTY" in band_rms and band_rms["DIRTY"] > 0:
        print("\n  RATIO vs DIRTY (native GEOS) at SH-UTLS, t=tlast:")
        for label in ("OMEGA", "CLEAN"):
            if label in band_rms:
                print(f"    {label}/DIRTY  |Lap|RMS = {band_rms[label]/band_rms['DIRTY']:.3f}"
                      f"    rel-ratio = {band_rel[label]/band_rel['DIRTY']:.3f}")
        print("  (OMEGA/DIRTY << 1 and ≈ CLEAN/DIRTY ⇒ closure cures fingering at the tracer level)")

    # per-level table at tlast
    print(f"\n  per-level |Lap|RMS at t={tlast} (SH, lat<-30):")
    hdr = "  p(hPa) " + "".join(f"{lab:>13}" for lab,_ in RUNS) + "   OMEGA/DIRTY"
    print(hdr)
    ref = "DIRTY"
    for ki, k in enumerate(data.get("DIRTY", data[list(data)[0]])["utls_k"]):
        row_vals = {}
        for label, d in data.items():
            if k < d["nz"]:
                rms, sd, mn = sh_stats(np.asarray(d["ds"].variables["co2_natural"][tlast, k, :, :, :]), d["lat"])
                row_vals[label] = rms
        pk = pmid[k] if pmid is not None and k < len(pmid) else np.nan
        cells = "".join(f"{row_vals.get(lab, np.nan):13.3e}" for lab,_ in RUNS)
        rr = (row_vals.get("OMEGA", np.nan)/row_vals[ref]) if (ref in row_vals and row_vals[ref] > 0) else np.nan
        print(f"  {pk:6.1f} {cells}   {rr:.3f}")

    # ---- deliverable 5c: global tracer (CO2) burden conservation -------------
    print(f"\n=== DELIVERABLE 5c: global CO2 burden Σ(co2_vmr·air_mass) ===")
    burdens = {}
    for label, d in data.items():
        ds = d["ds"]
        if "air_mass" not in ds.variables:
            print(f"  {label}: no air_mass var, skipping"); continue
        co2 = ds.variables["co2_natural"]; am = ds.variables["air_mass"]
        # burden(t) = Σ_cells co2_vmr(t) * air_mass(t)  (vmr·kg; relative drift is
        # what matters). air_mass shape matches co2 (t,lev,nf,Y,X) or static.
        t0 = 0; tl = d["nt"] - 1
        def burden(t):
            c = np.asarray(co2[t, :, :, :, :], np.float64)
            a = np.asarray(am[t, :, :, :, :], np.float64) if am.ndim == 5 else np.asarray(am[:], np.float64)
            return np.sum(c * a)
        b0 = burden(t0); bl = burden(tl)
        burdens[label] = (b0, bl)
        print(f"  {label}:  burden(t0) = {b0:.9e}   burden(t_end={tl}) = {bl:.9e}   "
              f"Δ = {bl-b0:.6e}  (rel {(bl-b0)/b0:.3e})")

    if "OMEGA" in burdens and "DIRTY" in burdens:
        # both share IC + emission; compare GAIN over the common window length.
        tl = common_nt - 1
        def gain(label):
            d = data[label]; ds = d["ds"]; co2 = ds.variables["co2_natural"]; am = ds.variables["air_mass"]
            def burden(t):
                c = np.asarray(co2[t, :, :, :, :], np.float64)
                a = np.asarray(am[t, :, :, :, :], np.float64) if am.ndim == 5 else np.asarray(am[:], np.float64)
                return np.sum(c * a)
            return burden(tl) - burden(0)
        go = gain("OMEGA"); gd = gain("DIRTY")
        print(f"\n  CONSERVATION CROSS-CHECK (same IC+emission ⇒ equal burden gain):")
        print(f"    OMEGA gain(0..{tl}) = {go:.6e}")
        print(f"    DIRTY gain(0..{tl}) = {gd:.6e}")
        print(f"    |OMEGA-DIRTY|/DIRTY = {abs(go-gd)/abs(gd):.3e}   "
              f"(≈0 ⇒ OMEGA cm conserves tracer mass like DIRTY)")

    for d in data.values():
        d["ds"].close()

main()
