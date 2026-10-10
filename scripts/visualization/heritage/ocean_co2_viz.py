#!/usr/bin/env python3
"""
Visualize the ECCO-Darwin ocean-CO2 December run:

  (1) A combined 2-panel MP4:
        left  = daily ocean air-sea CO2 flux (ECCO-Darwin input, held per day)
        right = ocean XCO2 anomaly (co2_ocean column mean), 3-hourly
      NOTE: the run was driven by the MONTHLY-MEAN flux; the left panel shows the
      daily INPUT flux for context on its sub-monthly variability.

  (2) A 3-panel static PNG: 2-week-mean (Dec 18-31) XCO2 spatial anomaly for
      natural / ocean / anthropogenic(fossil) tracers (each minus its global mean).

Inputs:
  --means   column_means_3hourly.nc  (cs_lon, cs_lat, cs_area,
                                      co2_*[cell,time], time_hours)
  --flux    eccodarwin_ocean_co2_perday_05deg.nc  (co2_flux_ocean[time,lat,lon])
  --outdir  output directory
"""
import argparse, os
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import cartopy.crs as ccrs
from datetime import datetime, timedelta

RUN_START = datetime(2021, 12, 1)
KG_S_TO_MOL_YR = 3.15576e7 / 0.0440095   # kgCO2/m2/s -> mol CO2/m2/yr


def latlon_grid(dlon=0.5, dlat=0.5):
    nlon, nlat = int(round(360 / dlon)), int(round(180 / dlat))
    lon = -180 + (np.arange(nlon) + 0.5) * dlon
    lat = -90 + (np.arange(nlat) + 0.5) * dlat
    return lon, lat


def unit_vec(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def build_cs_to_latlon(cs_lon, cs_lat, tlon, tlat):
    """Nearest-neighbour index mapping each target lat-lon cell to a CS cell."""
    tree = cKDTree(unit_vec(cs_lon, cs_lat))
    LO, LA = np.meshgrid(tlon, tlat)                 # (nlat, nlon)
    _, idx = tree.query(unit_vec(LO.ravel(), LA.ravel()))
    return idx.reshape(LA.shape)                      # (nlat, nlon)


def add_map(ax):
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.set_global()


def make_video(means, flux, outdir, dlon=0.5, dlat=0.5, fps=8, ocean_offset=0.0):
    ds = Dataset(means)
    cs_lon = np.array(ds["cs_lon"][:]); cs_lat = np.array(ds["cs_lat"][:])
    oc = np.array(ds["co2_ocean"][:]) - ocean_offset  # (time, cell) ppm; subtract IC background
    th = np.array(ds["time_hours"][:]); ds.close()
    nt = len(th)

    fd = Dataset(flux)
    flon = np.array(fd["lon"][:]); flat = np.array(fd["lat"][:])
    fday = np.array(fd["co2_flux_ocean"][:]) * KG_S_TO_MOL_YR   # (31, lat, lon) mol/m2/yr
    fd.close()

    tlon, tlat = latlon_grid(dlon, dlat)
    idx = build_cs_to_latlon(cs_lon, cs_lat, tlon, tlat)

    vflux = np.percentile(np.abs(fday), 99.5)
    voc = np.percentile(np.abs(oc), 99.0)
    print(f"video: {nt} frames; flux |v|<={vflux:.2f} mol/m2/yr; xco2 |v|<={voc:.3f} ppm")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 4.2),
                                   subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06, wspace=0.08)
    add_map(axl); add_map(axr)
    ext = [-180, 180, -90, 90]
    im_l = axl.imshow(fday[0], extent=ext, origin="lower", transform=ccrs.PlateCarree(),
                      cmap="RdBu_r", vmin=-vflux, vmax=vflux)
    im_r = axr.imshow(oc[0][idx], extent=ext, origin="lower", transform=ccrs.PlateCarree(),
                      cmap="RdBu_r", vmin=-voc, vmax=voc)
    axl.set_title("ECCO-Darwin daily air-sea CO$_2$ flux (input)", fontsize=11)
    axr.set_title("Ocean XCO$_2$ anomaly (model)", fontsize=11)
    plt.colorbar(im_l, ax=axl, orientation="horizontal", pad=0.03, shrink=0.8,
                 label="mol CO$_2$ m$^{-2}$ yr$^{-1}$ (+ = to atmosphere)")
    plt.colorbar(im_r, ax=axr, orientation="horizontal", pad=0.03, shrink=0.8,
                 label="ppm (dry-column anomaly)")
    sup = fig.suptitle("", fontsize=13)

    def upd(t):
        day = min(int(th[t] // 24), fday.shape[0] - 1)
        im_l.set_data(fday[day])
        im_r.set_data(oc[t][idx])
        dt = RUN_START + timedelta(hours=float(th[t]))
        sup.set_text(f"Ocean CO$_2$: {dt:%b %d %H:%M}  (flux = climatological Dec, day {day+1})")
        return im_l, im_r, sup

    ani = manim.FuncAnimation(fig, upd, frames=nt, blit=False)
    out = os.path.join(outdir, "ocean_co2_flux_and_xco2.mp4")
    ani.save(out, writer=manim.FFMpegWriter(fps=fps, bitrate=4000))
    plt.close(fig)
    print("wrote", out)


def make_static(means, outdir, dlon=0.5, dlat=0.5, win_start_h=17 * 24, ocean_offset=0.0):
    ds = Dataset(means)
    cs_lon = np.array(ds["cs_lon"][:]); cs_lat = np.array(ds["cs_lat"][:])
    if "cs_area" not in ds.variables:
        ds.close()
        raise ValueError("means file lacks cs_area; regenerate it with extract_cs_column_means.jl")
    cs_area = np.array(ds["cs_area"][:])
    th = np.array(ds["time_hours"][:])
    sel = th >= win_start_h                            # last two weeks (Dec 18-31)
    fields = {n: np.array(ds[n][:])[sel, :].mean(0) for n in ("co2_natural", "co2_ocean", "co2_fossil")}
    fields["co2_ocean"] = fields["co2_ocean"] - ocean_offset   # subtract IC background -> anomaly
    ds.close()
    dt0 = RUN_START + timedelta(hours=float(win_start_h))
    dt1 = RUN_START + timedelta(hours=float(th[sel].max()))
    win_label = f"{dt0:%b %d}-{dt1:%b %d}"
    print(f"static: window {win_label} ({int(sel.sum())} frames)")

    tlon, tlat = latlon_grid(dlon, dlat)
    idx = build_cs_to_latlon(cs_lon, cs_lat, tlon, tlat)
    ext = [-180, 180, -90, 90]

    titles = {"co2_natural": "Natural (biosphere, LMDZ/CAMS)",
              "co2_ocean": "Ocean (ECCO-Darwin)",
              "co2_fossil": "Anthropogenic (fossil, GridFED)"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 11),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.suptitle(f"2-week-mean XCO$_2$ spatial anomaly ({win_label}), by source", fontsize=13)
    for ax, name in zip(axes, ("co2_natural", "co2_ocean", "co2_fossil")):
        gmean = np.sum(fields[name] * cs_area) / np.sum(cs_area)
        anom = fields[name][idx] - gmean               # spatial anomaly (minus global mean)
        v = np.percentile(np.abs(anom), 99)
        add_map(ax)
        im = ax.imshow(anom, extent=ext, origin="lower", transform=ccrs.PlateCarree(),
                       cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{titles[name]}   (mean {gmean:+.2f} ppm)", fontsize=11)
        plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.85,
                     label="ppm anomaly")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.94, bottom=0.02, hspace=0.15)
    out = os.path.join(outdir, "xco2_2week_mean_by_source.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--means", required=True)
    ap.add_argument("--flux", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--ocean-offset-ppm", type=float, default=0.0,
                    help="legacy files only: subtract an unremoved ocean carrier (normally 0; current extractor removes it)")
    ap.add_argument("--win-start-h", type=float, default=17 * 24,
                    help="start hour of the 2-week-average window (default 408 = Dec 18)")
    ap.add_argument("--skip-video", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    make_static(a.means, a.outdir, ocean_offset=a.ocean_offset_ppm, win_start_h=a.win_start_h)
    if not a.skip_video:
        make_video(a.means, a.flux, a.outdir, fps=a.fps, ocean_offset=a.ocean_offset_ppm)
