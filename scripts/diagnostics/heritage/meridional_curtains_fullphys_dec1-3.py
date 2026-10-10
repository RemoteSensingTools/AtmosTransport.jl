#!/usr/bin/env python3
"""FULL-PHYSICS meridional (latitude-pressure) curtain animation of co2_natural
at three tropical source longitudes — central Amazonia (60°W), central Africa
(20°E), SE Asia (110°E) — over Dec 1-3 2021, comparing
  GEOS-Chem │ GEOS-native(full) │ OMEGA-consistent(full) │ ERA5(full).

This is the full-physics counterpart of `meridional_curtains_dec1-5.py`
(retired in the script cleanup; recover from 7c515038). The
three model rows now run advection + convection + diffusion (apples-to-apples
with GEOS-Chem, which includes everything). MERRA-2 is DROPPED here: its met
binary is flux-only ([:m,:am,:bm,:cm,:ps,:dm], no convection/diffusion fields)
so it cannot do full physics without a binary regen.

All three model rows are native C180 cubed-sphere (ERA5 is L137, the GEOS rows
L72); a meridian is extracted by nearest-cell (3D-sphere KDTree, built once) so
no full regrid is needed. Runs start from the shared Dec-1 IC so they're
state-aligned with GEOS-Chem.

  python3 scripts/diagnostics/heritage/meridional_curtains_fullphys_dec1-3.py [stride]
"""
import sys, datetime as dt
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim

OUT = "/home/cfranken/data/AtmosTransport/output/route1_dec1-5"
GCDIR = "/home/cfranken/data/AtmosTransport/catrine-geoschem-runs"
OUTGIF = ("/home/cfranken/www/catrine/route1_meridional_curtains/"
          "meridional_curtains_omega_FULLPHYS_amazonia_africa_seasia.gif")

# GEOS native L72 nominal layer TOP-edge pressures [hPa] (GEOS FP File Spec, App. B)
_EDGE72 = [0.0100,0.0200,0.0327,0.0476,0.0660,0.0893,0.1197,0.1595,0.2113,0.2785,
    0.3650,0.4758,0.6168,0.7951,1.0194,1.3005,1.6508,2.0850,2.6202,3.2764,4.0766,
    5.0468,6.2168,7.6198,9.2929,11.2769,13.6434,16.4571,19.7916,23.7304,28.3678,
    33.8100,40.1754,47.6439,56.3879,66.6034,78.5123,92.3657,108.663,127.837,150.393,
    176.930,208.152,244.875,288.083,337.500,375.000,412.500,450.000,487.500,525.000,
    562.500,600.000,637.500,675.000,700.000,725.000,750.000,775.000,800.000,820.000,
    835.000,850.000,865.000,880.000,895.000,910.000,925.000,940.000,955.000,970.000,985.000]

REGIONS = [("Amazonia 60°W", -60.0), ("Africa 20°E", 20.0), ("SE Asia 110°E", 110.0)]
RUNS = [  # (label, path, var, surface_first)
    ("GEOS-Chem",            None,                                          "SpeciesConcVV_CO2", True),
    ("GEOS-native (full)",   f"{OUT}/geosnative_fullphys_co2nat_dec1-3.nc", "co2_natural", False),
    ("OMEGA-consistent (full)", f"{OUT}/omega_fullphys_co2nat_dec1-3.nc",   "co2_natural", False),
    ("ERA5 (full)",          f"{OUT}/era5_fullphys_co2nat_dec1-3.nc",       "co2_natural", False),  # L137
]
L137 = "/home/cfranken/code/gitHub/AtmosTransportModel/config/era5_L137_coefficients.toml"
LATG = np.linspace(-88.0, 88.0, 119)          # meridian sample latitudes
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
T0 = dt.datetime(2021, 12, 1, 0, 0)

_edge = np.asarray(_EDGE72 + [1000.0])               # append nominal surface
PMID = 0.5 * (_edge[:-1] + _edge[1:])                # 72 layer-center pressures [hPa]
NZ = len(PMID)

def _parse_toml_array(path, key):                    # no tomllib on py3.9 -> regex
    import re
    txt = open(path).read()
    m = re.search(rf'^\s*{key}\s*=\s*\[(.*?)\]', txt, re.S | re.M)
    return np.array([float(x) for x in
                     re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', m.group(1))])
_a137 = _parse_toml_array(L137, "a"); _b137 = _parse_toml_array(L137, "b")
assert len(_a137) == 138 and len(_b137) == 138, (len(_a137), len(_b137))
_ph = _a137 + _b137 * 1.01325e5                      # half-level p [Pa], nominal ps
PMID137 = 0.5 * (_ph[:-1] + _ph[1:]) / 100.0         # 137 layer-center pressures [hPa]

def pmid_for(nz):
    return PMID137 if nz == 137 else PMID

def sphere_xyz(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])

def build_meridian_idx(lons, lat_cells):
    """nearest flat CS-cell index for each (region_lon, LATG) point."""
    tree = cKDTree(sphere_xyz(lons.ravel(), lat_cells.ravel()))
    idx = {}
    for name, L in REGIONS:
        _, ii = tree.query(sphere_xyz(np.full_like(LATG, L), LATG))
        idx[name] = ii
    return idx

def gc_path(t):
    d = T0 + dt.timedelta(hours=3*t)
    return f"{GCDIR}/GEOSChem.CATRINE_inst.{d:%Y%m%d}_{d:%H%M}z.nc4"

# --- preload meridian curtains for every frame -----------------------------
print("Preloading meridian curtains (FULL PHYSICS)...")
src_idx = {}
ours = Dataset(RUNS[1][1], "r")
# Cap frames to the shortest our-run so every row is present in every frame;
# all runs are 3-hourly and start from the Dec-1 IC.
nt = min(Dataset(p, "r").variables["co2_natural"].shape[0] for _, p, *_ in RUNS if p)
our_lons = np.asarray(ours.variables["lons"][:]); our_lats = np.asarray(ours.variables["lats"][:])
src_idx["ours"] = build_meridian_idx(our_lons, our_lats)
ours.close()
import os
_gf = next((gc_path(t) for t in range(nt) if os.path.exists(gc_path(t))), None)
if _gf is None:
    raise FileNotFoundError("no GEOS-Chem CATRINE_inst files found under " + GCDIR)
with Dataset(_gf, "r") as g:
    g_lons = np.asarray(g.variables["lons"][:]); g_lats = np.asarray(g.variables["lats"][:])
src_idx["gc"] = build_meridian_idx(g_lons, g_lats)

frames = list(range(0, nt, STRIDE))
data = {lab: {name: [] for name, _ in REGIONS} for lab, *_ in RUNS}
our_ds = {lab: Dataset(p, "r") for lab, p, *_ in RUNS if p}
for t in frames:
    for lab, path, var, sfc in RUNS:
        if path is None:  # GEOS-Chem: one file per time
            try:
                with Dataset(gc_path(t), "r") as g:
                    fld = np.asarray(g.variables[var][0])            # (lev,nf,Y,X) surface-first
            except Exception:
                fld = None
            idx = src_idx["gc"]
        else:
            fld = np.asarray(our_ds[lab].variables[var][t])         # (lev,nf,Y,X) top-first
            idx = src_idx["ours"]
        for name, _ in REGIONS:
            if fld is None:
                cur = np.full((NZ, len(LATG)), np.nan)
            else:
                f2 = fld[::-1] if sfc else fld                      # -> top-first
                flat = f2.reshape(f2.shape[0], -1)                  # (lev, N)
                cur = flat[:, idx[name]] * 1e6                      # VMR -> ppm, (lev, lat)
            data[lab][name].append(cur)
for d in our_ds.values(): d.close()

allvals = np.concatenate([np.asarray(data[lab][nm]).ravel()
                          for lab, *_ in RUNS for nm, _ in REGIONS])
allvals = allvals[np.isfinite(allvals)]
vmin, vmax = np.percentile(allvals, [2, 98])
print(f"frames={len(frames)}  colorrange=[{vmin:.1f}, {vmax:.1f}] ppm")

# --- animate: rows=runs, cols=regions --------------------------------------
nrow, ncol = len(RUNS), len(REGIONS)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.0*nrow), squeeze=False)
ims = {}
for r, (lab, *_ ) in enumerate(RUNS):
    pm = pmid_for(np.asarray(data[lab][REGIONS[0][0]][0]).shape[0])
    for c, (name, _) in enumerate(REGIONS):
        ax = axes[r][c]
        im = ax.pcolormesh(LATG, pm, data[lab][name][0], cmap="viridis",
                           vmin=vmin, vmax=vmax, shading="nearest")
        ims[(r, c)] = im
        ax.set_ylim(1013, 0)                          # linear pressure, surface -> 0 hPa top
        if r == 0: ax.set_title(name, fontsize=11)
        if c == 0: ax.set_ylabel(f"{lab}\np (hPa)", fontsize=9)
        if r == nrow-1: ax.set_xlabel("latitude")
        ax.tick_params(labelsize=8)
fig.colorbar(ims[(0, ncol-1)], ax=axes, label="CO₂ (ppm)", shrink=0.6, pad=0.01)
sup = fig.suptitle("", fontsize=13)

def update(fi):
    t = frames[fi]
    for r, (lab, *_ ) in enumerate(RUNS):
        for c, (name, _) in enumerate(REGIONS):
            ims[(r, c)].set_array(data[lab][name][fi].ravel())
    sup.set_text("co2_natural meridional curtains (FULL PHYSICS) — "
                 f"{(T0+dt.timedelta(hours=3*t)):%Y-%m-%d %H:%M}Z")
    return list(ims.values()) + [sup]

import os
os.makedirs(os.path.dirname(OUTGIF), exist_ok=True)
ani = anim.FuncAnimation(fig, update, frames=len(frames), blit=False)
ani.save(OUTGIF, writer=anim.PillowWriter(fps=4), dpi=90)
print("wrote", OUTGIF)
