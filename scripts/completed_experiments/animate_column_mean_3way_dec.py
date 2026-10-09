#!/usr/bin/env python3
"""4x3 column-mean animation over the full December campaign: rows = the 4
tracers (natural CO2, fossil CO2, SF6, Rn222), columns = GeoChem | AtmosTransport
GEOS-IT-omega | AtmosTransport ERA5. One frame per 3-hourly GeoChem snapshot.

Each panel shows the dry-air-mass-weighted COLUMN-MEAN mixing ratio binned from
the C180 cube to a lat-lon raster, per-row shared colorbar. Per-tracer scaling:
  co2_natural : linear viridis, stretched to the upper end (~425 ppm)
  co2_fossil  : SEMILOG (white->red), ~0..8 ppm
  sf6         : linear viridis
  rn222       : SEMILOG (white->red)

Budget readout at the top-left of each panel (referenced to the FIRST frame):
  EVERY column shows dm vs |F = its own emission-conservation check:
    dm = mass(t) - mass(0)            (burden change from the IC)
    |F = integral of that run's own surface flux   (cumulative emission)
  |F is the COMMON forcing inventory on ALL THREE columns — GC and both AT runs
  are driven by the same CAMS biospheric / EDGAR / gridfed fluxes, so each
  column's dm-vs-|F is a like-for-like conservation check against the IDENTICAL
  emission. (GC's own EmisCO2_Total diagnostic integrates ~0.4% below the lmdz
  file for co2_natural — an instantaneous-snapshot vs stepwise-integral
  artefact, same inventory — kept in the printed budget line, not the panels.)
  The |F series:
    sf6 / co2_fossil / rn222 : constant global rate (run-log conservative-regrid
                               src_total) integrated as rate*t.
    co2_natural              : time-varying lmdz CAMS (stepwise), the global rate
                               series read from the SAME file the model uses,
                               anchored to the model's logged first-slice rate.
For a conserving, non-decaying tracer dm == |F to that run's conservation floor
(F32 LinRood fillz=false: ~exact; PPM: ~0.04-0.4% by tracer sharpness). For
Rn222 the dm < |F gap is the radioactive-decay sink. AT dm prefers the exact
F64 `<tracer>_total_mass` variable when the run wrote it (true mass balance);
GC dm is always the field integral (GC files carry no such variable).

ALL masses/column-means come from the 3D fields and the dry layer air mass
(= dry-dP * area / g; our `air_mass`, GC's `Met_AD`) — never the precomputed
{tracer}_column_mass_per_area (which had a constant-factor issue).

  python3 scripts/completed_experiments/animate_column_mean_3way_dec.py [out.mp4]
"""
import sys, os, datetime as dt
import numpy as np
from netCDF4 import Dataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.colors import Normalize, LogNorm
import cartopy.feature as cfeature
import cartopy.crs as ccrs

OUTDIR = os.path.expanduser("~/www/catrine"); os.makedirs(OUTDIR, exist_ok=True)
OUTMP4 = sys.argv[1] if len(sys.argv) > 1 else f"{OUTDIR}/column_mean_3way_dec_4tracer.mp4"
GCDIR  = "/home/cfranken/data/AtmosTransport/catrine-geoschem-runs"
# Override the two AtmosTransport inputs without editing the script:
#   ANIM_GEOSIT=/path/a.nc ANIM_ERA5=/path/b.nc animate_... out.gif
GEOSIT = os.environ.get("ANIM_GEOSIT",
    "/home/cfranken/data/AtmosTransport/output/campaign_winter2021/geosit_omega_4tracer_dec2021_feb2022.nc")
ERA5   = os.environ.get("ANIM_ERA5",
    "/home/cfranken/data/AtmosTransport/output/campaign_winter2021/era5_4tracer_dec2021_feb2022.nc")
T0 = dt.datetime(2021, 12, 1)
NLON, NLAT = 180, 90
M_AIR = 28.9644
# our, gc_vmr, gc_emis, M_tracer, mapscale, unit, cmap, norm, (lo,hi) or ("pct",lo,hi)
TRACERS = [
    ("co2_natural", "SpeciesConcVV_CO2",       "EmisCO2_Total",        44.01, 1e6,  "ppm",   "viridis", "lin", (412.0, 424.0)),
    ("co2_fossil",  "SpeciesConcVV_FossilCO2", "Emis_FossilCO2_Total", 44.01, 1e6,  "ppm",   "Reds",    "log", (0.02, 8.0)),
    ("sf6",         "SpeciesConcVV_SF6",       "EmisSF6",             146.06, 1e12, "ppt",   "viridis", "lin", ("pct", 2, 98)),
    ("rn222",       "SpeciesConcVV_Rn222",     "EmisRn_Soil",         222.0,  1e21, "1e-21", "Reds",    "log", ("pct", 40, 99.5)),
]
SOURCES = ["GEOS-Chem", "AT GEOS-IT", "AT ERA5"]

# --- our own emission flux for the AT columns (identical inventory for both AT
# runs; only the transport met differs). Constant global rates [kg species/s]
# from the run-log conservative-regrid src_total. ---
LMDZ = os.path.expanduser("~/data/AtmosTransport/catrine/Emissions/LMDZ_fluxes/"
                          "z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc")
AT_CONST_RATE = {           # kg species / s (constant over Dec)
    "co2_fossil": 1.229399e6,
    "rn222":      4.443261e-7,
    "sf6":        3.201820e-1,
}
CO2NAT_ANCHOR = 7.283045e6  # kgCO2/s, model's logged first-slice lmdz rate

def build_our_rates(frames, name_to_idx):
    """Our global emission rate [kg species/s] per (frame, tracer) for the AT runs."""
    nf = len(frames)
    our = np.zeros((nf, len(TRACERS)))
    for nm, rate in AT_CONST_RATE.items():
        if nm in name_to_idx:
            our[:, name_to_idx[nm]] = rate
    if "co2_natural" in name_to_idx and os.path.exists(LMDZ):
        with Dataset(LMDZ) as fz:                                  # flux_apos kgC/m2/s
            flux = np.asarray(fz.variables["flux_apos"][:], dtype=np.float64)  # (248,180,360)
            area = np.asarray(fz.variables["area"][:], dtype=np.float64)       # (180,360)
        sl = np.array([np.sum(flux[t] * area) for t in range(flux.shape[0])])  # kgC/s
        rate_series = CO2NAT_ANCHOR * (sl / sl[0])                 # kgCO2/s, stepwise per 3-h slice
        ci = name_to_idx["co2_natural"]
        for f, s in enumerate(frames):
            k = int(round((s - T0).total_seconds() / 3600.0 / 3.0))
            our[f, ci] = rate_series[min(max(k, 0), len(rate_series) - 1)]
    elif "co2_natural" in name_to_idx:
        print(f"WARNING: lmdz file not found ({LMDZ}); co2_natural |F left at 0")
    return our

def gc_path(s): return f"{GCDIR}/GEOSChem.CATRINE_inst.{s:%Y%m%d}_{s:%H%M}z.nc4"

def fmt_mass(kg):
    s = "-" if kg < 0 else ""
    a = abs(kg)
    for div, unit in [(1e12,"Pg"), (1e9,"Tg"), (1e6,"Gg"), (1e3,"t"), (1.0,"kg"), (1e-3,"g")]:
        if a >= div: return f"{s}{a/div:.3f} {unit}"
    return f"{kg:.3g} kg"

def bin_latlon(lons, lats, vals):
    lo = (lons + 180) % 360
    H,_,_ = np.histogram2d(lats.ravel(), lo.ravel(), bins=[NLAT,NLON],
                           range=[[-90,90],[0,360]], weights=vals.ravel())
    C,_,_ = np.histogram2d(lats.ravel(), lo.ravel(), bins=[NLAT,NLON], range=[[-90,90],[0,360]])
    with np.errstate(invalid="ignore"): return H/np.where(C>0, C, np.nan)

_COAST = None
def _coast_segments():
    global _COAST
    if _COAST is None:
        _COAST = []
        for geom in cfeature.COASTLINE.geometries():
            for g in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]):
                x, y = g.xy
                _COAST.append((np.asarray(x), np.asarray(y)))
    return _COAST

def draw_coast(ax):
    for x, y in _coast_segments():
        ax.plot(x, y, color="0.2", lw=0.25, alpha=0.6)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)

def col_and_mass(vmr3d, drymass3d, mr):
    num = np.sum(vmr3d * drymass3d, axis=0)
    den = np.sum(drymass3d, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"): col = num / den
    return col, float(np.sum(num, dtype=np.float64)) * mr

def build_norm(normtype, spec, allv):
    v = allv[np.isfinite(allv)]
    if isinstance(spec[0], str) and spec[0] == "pct":
        if normtype == "log": v = v[v > 0]
        lo, hi = np.nanpercentile(v, [spec[1], spec[2]])
    else:
        lo, hi = spec
    if normtype == "log":
        return LogNorm(vmin=max(lo, 1e-9), vmax=hi, clip=True)
    return Normalize(vmin=lo, vmax=hi)

def main():
    go = Dataset(GEOSIT); er = Dataset(ERA5)
    g_lons = np.asarray(go.variables["lons"][:]); g_lats = np.asarray(go.variables["lats"][:])
    e_lons = np.asarray(er.variables["lons"][:]); e_lats = np.asarray(er.variables["lats"][:])
    nt = go.variables["co2_natural"].shape[0]
    M_ratio = [M/M_AIR for (_,_,_,M,_,_,_,_,_) in TRACERS]

    def our_idx(stamp):
        k = int(round((stamp - T0).total_seconds() / 3600.0 / 3.0))
        return k if 0 <= k < nt else None

    stamps = [T0 + dt.timedelta(hours=h) for h in range(3, 31*24, 3)]
    _MX = int(os.environ.get("ANIM_MAXFRAMES", "0"))
    gc_area = None
    frames, maps, mass, erate = [], [], [], []        # erate[f][ti] = global emission rate [kg/s]
    print(f"scanning {len(stamps)} candidate 3-hourly stamps (reading 3D fields) ...")
    for s in stamps:
        gi = our_idx(s)
        if gi is None or not os.path.exists(gc_path(s)): continue
        try:
            with Dataset(gc_path(s)) as g:
                if gc_area is None: gc_area = np.asarray(g.variables["Met_AREAM2"][0])
                ad = np.asarray(g.variables["Met_AD"][0])
                gc = [col_and_mass(np.asarray(g.variables[gv][0]), ad, mr)
                      for (ov,gv,ge,M,sc,un,cm,nm,vl),mr in zip(TRACERS, M_ratio)]
                er_rate = [float(np.sum(np.asarray(g.variables[ge][0]) * gc_area))
                           for (ov,gv,ge,M,sc,un,cm,nm,vl) in TRACERS]
        except Exception as e:
            print(f"  skip {s:%m-%d %H:%M} (GC read failed: {e})"); continue
        go_am = np.asarray(go.variables["air_mass"][gi]); er_am = np.asarray(er.variables["air_mass"][gi])
        row_maps, row_mass = [], []
        for ti,(ov,gv,ge,M,sc,un,cm,nm,vl) in enumerate(TRACERS):
            mr = M_ratio[ti]
            gc_col, gc_m = gc[ti]
            o_col, o_m = col_and_mass(np.asarray(go.variables[ov][gi]), go_am, mr)
            e_col, e_m = col_and_mass(np.asarray(er.variables[ov][gi]), er_am, mr)
            # TRUE mass balance: prefer the exact F64 `<tracer>_total_mass`
            # written at capture (authoritative; the F32 spatial-field
            # integral is reconstruction-polluted for reference-state
            # tracers). GC files have no such variable -> field integral.
            tmv = f"{ov}_total_mass"
            if tmv in go.variables: o_m = float(go.variables[tmv][gi]) * mr
            if tmv in er.variables: e_m = float(er.variables[tmv][gi]) * mr
            row_maps.append([bin_latlon(g_lons,g_lats, gc_col*sc),
                             bin_latlon(g_lons,g_lats, o_col*sc),
                             bin_latlon(e_lons,e_lats, e_col*sc)])
            row_mass.append([gc_m, o_m, e_m])
        frames.append(s); maps.append(row_maps); mass.append(row_mass); erate.append(er_rate)
        if len(frames) % 24 == 0: print(f"  ... {len(frames)} frames @ {s:%m-%d %H:%M}")
        if _MX and len(frames) >= _MX: print(f"  QC limit {_MX} reached"); break
    go.close(); er.close()
    nframes = len(frames)
    print(f"built {nframes} frames")
    if nframes == 0: print("no frames"); return

    # dm[f][ti][src] = mass since frame 0
    # cumF[f][ti]    = GC emission integral (for GC column budget check only)
    mass = np.asarray(mass); erate = np.asarray(erate)            # (F, ntr, 3) ; (F, ntr)
    dm = mass - mass[0:1, :, :]                                   # dm[:,:,0]=GC, [1]=GEOS, [2]=ERA5
    dts = np.array([1.0] + [(frames[f]-frames[f-1]).total_seconds() for f in range(1, nframes)])
    cumF_gc = np.zeros((nframes, len(TRACERS)))
    for f in range(1, nframes):
        cumF_gc[f] = cumF_gc[f-1] + erate[f] * dts[f]            # GC emission integral, GC column

    # OUR own emission integral for the AT columns (same accumulation as GC,
    # referenced to frame 0; identical for both AT runs).
    name_to_idx = {t[0]: i for i, t in enumerate(TRACERS)}
    our_rate = build_our_rates(frames, name_to_idx)             # (F, ntr) kg species/s
    cumF_at = np.zeros((nframes, len(TRACERS)))
    for f in range(1, nframes):
        cumF_at[f] = cumF_at[f-1] + our_rate[f] * dts[f]
    for ti,(ov,*_ ) in enumerate(TRACERS):                      # final-frame budget sanity
        print(f"  {ov:11s}: dm_GEOS={fmt_mass(dm[-1][ti][1])}  dm_ERA5={fmt_mass(dm[-1][ti][2])}  "
              f"|F_AT={fmt_mass(cumF_at[-1][ti])}  dm_GC={fmt_mass(dm[-1][ti][0])}  |F_GC={fmt_mass(cumF_gc[-1][ti])}")

    norms, cmaps = [], []
    for ti,(ov,gv,ge,M,sc,un,cmn,nm,vl) in enumerate(TRACERS):
        allv = np.concatenate([np.asarray(maps[f][ti]).ravel() for f in range(nframes)])
        norms.append(build_norm(nm, vl, allv))
        cmap = plt.get_cmap(cmn).copy(); cmap.set_bad("white"); cmaps.append(cmap)

    # Robinson projection: a global pseudocylindrical view that shows the full
    # globe (incl. high latitudes) without the equirectangular pole-stretching.
    # The lat-lon raster is drawn with a PlateCarree transform; cartopy warps it.
    proj = ccrs.Robinson(central_longitude=0)
    pc   = ccrs.PlateCarree()
    fig, axes = plt.subplots(len(TRACERS), 3, figsize=(13.5, 9.4), squeeze=False,
                             subplot_kw={"projection": proj})
    fig.subplots_adjust(left=0.08, right=0.91, top=0.93, bottom=0.02, wspace=0.05, hspace=0.10)
    ims = [[None]*3 for _ in TRACERS]; txt = [[None]*3 for _ in TRACERS]
    for ti,(ov,gv,ge,M,sc,un,cmn,nm,vl) in enumerate(TRACERS):
        for c in range(3):
            ax = axes[ti][c]
            ax.set_global()
            ims[ti][c] = ax.imshow(maps[0][ti][c], origin="lower", extent=[-180,180,-90,90],
                                   transform=pc, cmap=cmaps[ti], norm=norms[ti])
            txt[ti][c] = ax.text(0.015, 0.97, "", transform=ax.transAxes, fontsize=7,
                                 color="black", va="top", ha="left", zorder=5,
                                 bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))
            ax.coastlines(linewidth=0.25, color="0.2", alpha=0.6)
            if ti == 0: ax.set_title(SOURCES[c], fontsize=12, fontweight="bold")
            if c == 0: ax.text(-0.04, 0.5, f"{ov}\n({un})", fontsize=10, rotation=90,
                               va="center", ha="center", transform=ax.transAxes)
        cax = fig.add_axes([0.915, axes[ti][2].get_position().y0,
                            0.012, axes[ti][2].get_position().height])
        fig.colorbar(ims[ti][2], cax=cax)
    sup = fig.suptitle("", fontsize=11.5, y=0.985)

    def update(f):
        for ti in range(len(TRACERS)):
            for c in range(3):
                ims[ti][c].set_data(maps[f][ti][c])
                # |F is the COMMON forcing inventory on every panel (both AT
                # runs and GC are driven by the same CAMS biospheric / EDGAR /
                # gridfed fluxes), so each column's dm-vs-|F is a like-for-like
                # conservation check against the identical emission. (GC's own
                # EmisCO2_Total diagnostic integrates ~0.4% below the lmdz file
                # — an instantaneous-snapshot vs stepwise-integral artefact, NOT
                # a different inventory; printed in the budget line for the
                # record but not shown per-panel.)
                txt[ti][c].set_text(
                    f"dm = {fmt_mass(dm[f][ti][c if c else 0])}\n|F = {fmt_mass(cumF_at[f][ti])}")
        # Column identities live on the per-axis titles (GeoChem / AT GEOS-IT /
        # AT ERA5), so the suptitle stays short enough to fit the figure width:
        # date + the dm/|F legend only.
        sup.set_text(f"Column-mean mixing ratio   ·   {frames[f]:%Y-%m-%d %H:%M}z"
                     f"      dm = burden change from IC      "
                     f"|F = ∫ common surface flux")
        return []

    _qc = nframes - 1                                            # QC on the last frame (accumulated dm/|F)
    update(_qc); fig.savefig(OUTMP4.rsplit(".",1)[0]+"_qc.png", dpi=110)
    print("wrote QC frame ->", OUTMP4.rsplit(".",1)[0]+"_qc.png")
    print(f"rendering {nframes} frames -> {OUTMP4}")
    anim = manim.FuncAnimation(fig, update, frames=nframes, blit=False)
    try:
        anim.save(OUTMP4, writer=manim.FFMpegWriter(fps=8, bitrate=4000), dpi=110)
        print("wrote", OUTMP4)
    except Exception as e:
        gif = OUTMP4.rsplit(".",1)[0] + ".gif"
        print(f"ffmpeg failed ({e}); writing gif {gif}")
        anim.save(gif, writer=manim.PillowWriter(fps=8), dpi=90); print("wrote", gif)

if __name__ == "__main__":
    main()
