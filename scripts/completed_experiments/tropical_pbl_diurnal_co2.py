#!/usr/bin/env python3
"""Tropical PBL diurnal CO2 cycle: NEW time-varying vs OLD constant vs GEOS-Chem.

Compares the tropical-mean (area-weighted, |lat|<TROP_DEG) SURFACE co2_natural
diurnal cycle on Dec 2 2021 (day 2, post 1-day spin-up) for:
  (a) our NEW time-varying surface-flux run,
  (b) our OLD constant (monthly-mean) surface-flux run,
  (c) the GEOS-Chem CATRINE reference (3-hourly inst).

All three are C180 native cubed-sphere (nf=6, 180x180, lev=72). Our model is
TOA-first (lev=1 = TOA) so surface = lev[-1]; GEOS-Chem is surface-first
(lev=1 = surface) so surface = lev[0]. Cells correspond 1:1 on the shared
gnomonic grid, so per-cell `lats` weighting is consistent across all three.

Outputs a 3-curve plot + prints amplitude (ppm peak-to-peak) and phase
(UTC hour of max) for each curve.
"""
import os
import sys
import glob
import numpy as np
import netCDF4 as nc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TROP_DEG = 20.0  # |lat| < TROP_DEG defines the tropics
HOME = os.path.expanduser("~")
OUTDIR = os.path.join(HOME, "data/AtmosTransport/output/diurnal_test_dec1-2")
GCDIR = os.path.join(HOME, "data/AtmosTransport/catrine-geoschem-runs")
WWW = os.path.join(HOME, "www/catrine")
PNG = os.path.join(WWW, "omega_tropical_PBL_diurnal_co2nat.png")

NEW_NC = os.path.join(OUTDIR, "omega_fullphys_co2nat_NEW_timevarying_dec1-2.nc")
OLD_NC = os.path.join(OUTDIR, "omega_fullphys_co2nat_OLD_constant_dec1-2.nc")

# Day-2 diagnostic window (hours since run start 2021-12-01 00:00).
DAY2_START_H = 24.0
DAY2_END_H = 48.0


def tropical_surface_mean_ours(path, varname="co2_natural"):
    """Return (hours_since_start, tropmean_ppm[, ]) for our run's surface level.

    Our output: var dims (time, lev, nf, Ydim, Xdim) or (Xdim,Ydim,nf,lev,time)?
    We detect dims via the NetCDF variable dimension names.
    Surface = last lev index (TOA-first). Tropical area-weighted by cell_area
    restricted to |lats| < TROP_DEG.
    """
    ds = nc.Dataset(path)
    var = ds.variables[varname]
    dimnames = var.dimensions
    # Build an index map
    arr = var[:]
    # Move to canonical (time, lev, nf, Ydim, Xdim)
    order = {}
    for canon in ("time", "lev", "nf", "Ydim", "Xdim"):
        if canon in dimnames:
            order[canon] = dimnames.index(canon)
        else:
            raise RuntimeError(f"{path}: var {varname} missing dim {canon}; has {dimnames}")
    arr = np.transpose(arr, (order["time"], order["lev"], order["nf"], order["Ydim"], order["Xdim"]))
    nt, nlev, nf, ny, nx = arr.shape

    # NetCDF stores lats/cell_area with dims (nf, Ydim, Xdim) as seen by Python
    # (Julia column-major declaration order is reversed on read), matching the
    # surf array's trailing (nf, Ydim, Xdim) axes — no transpose needed.
    lats = np.asarray(ds.variables["lats"][:])
    area = np.asarray(ds.variables["cell_area"][:])
    assert lats.shape == (nf, ny, nx), f"lats shape {lats.shape} != {(nf, ny, nx)}"

    tvar = ds.variables["time"]
    tvals = np.asarray(tvar[:], dtype=float)
    tunits = getattr(tvar, "units", "")
    # convert to hours since start
    hours = _to_hours(tvals, tunits)

    surf = arr[:, -1, :, :, :]  # TOA-first -> surface is last lev. (time, nf, Ydim, Xdim)
    units = getattr(var, "units", "")
    scale = _vmr_scale(units)

    mask = np.abs(lats) < TROP_DEG
    w = (area * mask).astype(np.float64)
    wsum = w.sum()
    out = np.empty(nt)
    for it in range(nt):
        out[it] = float((surf[it] * w).sum() / wsum) * scale
    ds.close()
    return hours, out, units


def tropical_surface_mean_gc():
    """Return (utc_hours, tropmean_ppm) for GEOS-Chem Dec 2 (3-hourly inst)."""
    files = sorted(glob.glob(os.path.join(GCDIR, "GEOSChem.CATRINE_inst.20211202_*z.nc4")))
    if not files:
        raise RuntimeError("no GC Dec-2 files found")
    hours, vals = [], []
    lats = area = None
    for f in files:
        ds = nc.Dataset(f)
        co2 = ds.variables["SpeciesConcVV_CO2"][0]  # (lev, nf, Ydim, Xdim), surface-first
        surf = co2[0]  # lev=1 = surface
        if lats is None:
            lats = np.asarray(ds.variables["lats"][:])  # (nf, Ydim, Xdim)
            area = np.asarray(ds.variables["Met_AREAM2"][0])  # (nf, Ydim, Xdim)
            mask = np.abs(lats) < TROP_DEG
            w = (area * mask).astype(np.float64)
            wsum = w.sum()
        m = float((surf * w).sum() / wsum) * 1e6  # mol/mol -> ppm
        # parse UTC hour from filename ..._20211202_HHMMz.nc4
        base = os.path.basename(f)
        hhmm = base.split("_")[-1].replace("z.nc4", "")
        utc = int(hhmm[:2]) + int(hhmm[2:4]) / 60.0
        hours.append(utc)
        vals.append(m)
        ds.close()
    order = np.argsort(hours)
    return np.array(hours)[order], np.array(vals)[order]


def gc_point_series(lat0, lon0, label):
    """GC surface co2 at the cube cell nearest (lat0, lon0), Dec 2 vs UTC."""
    files = sorted(glob.glob(os.path.join(GCDIR, "GEOSChem.CATRINE_inst.20211202_*z.nc4")))
    hours, vals = [], []
    sel = None
    for f in files:
        ds = nc.Dataset(f)
        if sel is None:
            lats = np.asarray(ds.variables["lats"][:])
            lons = np.asarray(ds.variables["lons"][:]) % 360.0
            d = (lats - lat0) ** 2 + (((lons - (lon0 % 360) + 180) % 360 - 180)) ** 2
            sel = np.unravel_index(np.argmin(d), lats.shape)
        co2 = ds.variables["SpeciesConcVV_CO2"][0]
        surf = co2[0]
        vals.append(float(surf[sel]) * 1e6)
        base = os.path.basename(f)
        hhmm = base.split("_")[-1].replace("z.nc4", "")
        hours.append(int(hhmm[:2]) + int(hhmm[2:4]) / 60.0)
        ds.close()
    order = np.argsort(hours)
    return np.array(hours)[order], np.array(vals)[order], sel


def ours_point_series(path, sel_lat, sel_lon, varname="co2_natural"):
    ds = nc.Dataset(path)
    var = ds.variables[varname]
    dimnames = var.dimensions
    arr = var[:]
    order = {c: dimnames.index(c) for c in ("time", "lev", "nf", "Ydim", "Xdim")}
    arr = np.transpose(arr, (order["time"], order["lev"], order["nf"], order["Ydim"], order["Xdim"]))
    lats = np.asarray(ds.variables["lats"][:])  # (nf, Ydim, Xdim)
    lons = np.asarray(ds.variables["lons"][:]) % 360.0
    d = (lats - sel_lat) ** 2 + (((lons - (sel_lon % 360) + 180) % 360 - 180)) ** 2
    sel = np.unravel_index(np.argmin(d), lats.shape)
    surf = arr[:, -1, :, :, :]
    scale = _vmr_scale(getattr(var, "units", ""))
    tvals = np.asarray(ds.variables["time"][:], dtype=float)
    hours = _to_hours(tvals, getattr(ds.variables["time"], "units", ""))
    series = np.array([float(surf[it][sel]) * scale for it in range(surf.shape[0])])
    ds.close()
    return hours, series


def _vmr_scale(units):
    u = units.lower().replace(" ", "")
    if "ppm" in u:
        return 1.0
    if "mol/mol" in u or "molmol-1" in u or "molmol" in u or u in ("1", "vmr", ""):
        return 1e6
    return 1e6  # assume dry VMR fraction -> ppm


def _to_hours(tvals, units):
    u = units.lower()
    if "second" in u:
        return tvals / 3600.0
    if "minute" in u:
        return tvals / 60.0
    if "hour" in u:
        return tvals
    if "day" in u:
        return tvals * 24.0
    # fallback: assume already hours
    return tvals


def amp_phase(hours, vals):
    """Peak-to-peak amplitude (ppm) and UTC hour of the maximum."""
    if len(vals) == 0:
        return float("nan"), float("nan")
    amp = float(np.nanmax(vals) - np.nanmin(vals))
    phase = float(hours[int(np.nanargmax(vals))] % 24)
    return amp, phase


def main():
    # --- tropical area-weighted surface means ---
    h_new, v_new, u_new = tropical_surface_mean_ours(NEW_NC)
    h_old, v_old, u_old = tropical_surface_mean_ours(OLD_NC)
    h_gc, v_gc = tropical_surface_mean_gc()

    # restrict our runs to day-2 window and convert hours-since-start -> UTC hour
    def day2(h, v):
        m = (h >= DAY2_START_H - 1e-6) & (h <= DAY2_END_H + 1e-6)
        return (h[m] % 24), v[m]

    hu_new, vd_new = day2(h_new, v_new)
    hu_old, vd_old = day2(h_old, v_old)

    print(f"units: NEW={u_new!r} OLD={u_old!r}")
    a_new, p_new = amp_phase(hu_new, vd_new)
    a_old, p_old = amp_phase(hu_old, vd_old)
    a_gc, p_gc = amp_phase(h_gc, v_gc)

    print("\n=== TROPICAL (|lat|<%.0f) area-weighted SURFACE co2_natural diurnal (Dec 2) ===" % TROP_DEG)
    print(f"  NEW time-varying : amp={a_new:7.3f} ppm   phase(UTC max)={p_new:5.1f}h   mean={np.nanmean(vd_new):8.3f}")
    print(f"  OLD constant     : amp={a_old:7.3f} ppm   phase(UTC max)={p_old:5.1f}h   mean={np.nanmean(vd_old):8.3f}")
    print(f"  GEOS-Chem        : amp={a_gc:7.3f} ppm   phase(UTC max)={p_gc:5.1f}h   mean={np.nanmean(v_gc):8.3f}")

    # --- plot ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    ax[0].plot(hu_new, vd_new, "-o", color="tab:red", lw=2, label=f"NEW time-varying (amp={a_new:.2f})")
    ax[0].plot(hu_old, vd_old, "-s", color="tab:gray", lw=2, label=f"OLD constant (amp={a_old:.2f})")
    ax[0].plot(h_gc, v_gc, "-^", color="tab:blue", lw=2.5, label=f"GEOS-Chem (amp={a_gc:.2f})")
    ax[0].set_xlabel("UTC hour (Dec 2 2021)")
    ax[0].set_ylabel("surface co2_natural (ppm, dry VMR)")
    ax[0].set_title(f"Tropical (|lat|<{TROP_DEG:.0f}) area-wt surface CO2 diurnal")
    ax[0].legend(loc="best", fontsize=9)
    ax[0].grid(alpha=0.3)
    ax[0].set_xlim(0, 24)
    ax[0].set_xticks(range(0, 25, 3))

    # detrended (remove per-curve mean) to compare amplitude+phase cleanly
    def dt(v):
        return v - np.nanmean(v)

    ax[1].plot(hu_new, dt(vd_new), "-o", color="tab:red", lw=2, label="NEW time-varying")
    ax[1].plot(hu_old, dt(vd_old), "-s", color="tab:gray", lw=2, label="OLD constant")
    ax[1].plot(h_gc, dt(v_gc), "-^", color="tab:blue", lw=2.5, label="GEOS-Chem")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("UTC hour (Dec 2 2021)")
    ax[1].set_ylabel("surface co2_natural anomaly (ppm)")
    ax[1].set_title("Mean-removed (phase + amplitude)")
    ax[1].legend(loc="best", fontsize=9)
    ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, 24)
    ax[1].set_xticks(range(0, 25, 3))

    fig.suptitle("Tropical PBL diurnal CO2(natural): NEW time-varying vs OLD constant vs GEOS-Chem (omega full-physics)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(WWW, exist_ok=True)
    fig.savefig(PNG, dpi=130)
    print(f"\nsaved plot -> {PNG}")

    # --- optional: a strong-biosphere tropical land point (central Africa, Congo) ---
    try:
        lat0, lon0 = 0.0, 20.0  # Congo basin
        hg, vg, sel = gc_point_series(lat0, lon0, "Congo")
        ag, pg = amp_phase(hg, vg)
        hn, vn = ours_point_series(NEW_NC, lat0, lon0)
        ho, vo = ours_point_series(OLD_NC, lat0, lon0)
        mn = (hn >= DAY2_START_H - 1e-6) & (hn <= DAY2_END_H + 1e-6)
        mo = (ho >= DAY2_START_H - 1e-6) & (ho <= DAY2_END_H + 1e-6)
        an, pn = amp_phase(hn[mn] % 24, vn[mn])
        ao, po = amp_phase(ho[mo] % 24, vo[mo])
        print("\n=== Congo basin land point (0N,20E) surface co2_natural diurnal (Dec 2) ===")
        print(f"  NEW : amp={an:7.3f} ppm  phase={pn:5.1f}h")
        print(f"  OLD : amp={ao:7.3f} ppm  phase={po:5.1f}h")
        print(f"  GC  : amp={ag:7.3f} ppm  phase={pg:5.1f}h")
    except Exception as e:
        print("point-series step skipped:", e)


if __name__ == "__main__":
    main()
