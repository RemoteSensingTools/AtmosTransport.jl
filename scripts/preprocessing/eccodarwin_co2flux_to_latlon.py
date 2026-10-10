#!/usr/bin/env python3
"""
Convert ECCO-Darwin v5 daily air-sea CO2 flux (native LLC270 "compact" binary)
into a regular lat-lon NetCDF that AtmosTransport's surface-flux loader can read
and conservatively regrid onto the C180 cubed sphere at runtime.

Source: NASA NAS  https://data.nas.nasa.gov/ecco/llc_270/ecco_darwin_v5/output/daily/CO2_flux/
Grid:   NASA NAS  https://data.nas.nasa.gov/ecco/llc_270/grid/   (XC, YC, RAC, Depth)

Native field `fluxCO2` is mmol CO2 m^-2 s^-1, POSITIVE = INTO the ocean
(verified empirically: N. Atlantic subpolar = + (sink), eq. Pacific = - (source);
area-weighted annualized net ~ -3 PgC/yr ocean uptake). We flip the sign to the
model's emission convention (POSITIVE = flux INTO the atmosphere) and convert
mmol CO2 -> kg CO2 (x M_CO2 = 44.0095 g/mol):

    surface_flux [kgCO2/m2/s] = native[mmol CO2/m2/s] * (-1) * 44.0095e-6 kg/mmol

The 31 daily files of one December are averaged into a single monthly-mean field
(driven like the static GridFED fossil flux). The LLC270 cells are an unstructured
point cloud (XC/YC give each cell's lon/lat); we bin them conservatively onto a
regular lat-lon grid:  cell_value = sum(flux_i * area_i) / A_latlon_cell, so the
global integral is preserved exactly (empty / land cells = 0).

Usage:
    python3 eccodarwin_co2flux_to_latlon.py \
        --flux-dir  ~/data/AtmosTransport/catrine/Emissions/ECCO_Darwin/dec2008 \
        --grid-dir  ~/data/AtmosTransport/catrine/Emissions/ECCO_Darwin/grid \
        --out       ~/data/AtmosTransport/catrine/Emissions/ECCO_Darwin/eccodarwin_ocean_co2_december_05deg.nc \
        --dlon 0.5 --dlat 0.5
"""
import argparse, glob, os, sys
import numpy as np
from netCDF4 import Dataset

NX, NY = 270, 3510          # LLC270 compact layout
N = NX * NY                 # 947_700 cells
R = 6.371e6                 # Earth radius (m), matches the model's _lonlat_cell_areas_m2
M_CO2_KG_PER_MMOL = 44.0095e-6   # kg CO2 per mmol CO2
MISSING = -999.0


def read_compact(path):
    """Read a LLC270 'compact' big-endian float32 record (947700 values)."""
    a = np.fromfile(path, dtype=">f4", count=N)
    if a.size != N:
        sys.exit(f"ERROR: {path} has {a.size} values, expected {N}")
    return a.astype(np.float64)


def latlon_cell_areas(lat_centers, dlon_deg, dlat_deg):
    """Geographic area (m^2) of each regular lat band (same formula as the model)."""
    dlon = np.deg2rad(dlon_deg)
    half = np.deg2rad(dlat_deg) / 2.0
    phi = np.deg2rad(lat_centers)
    band = R * R * dlon * np.abs(np.sin(phi + half) - np.sin(phi - half))
    return band  # per (lat) -> broadcast over lon


def bin_flux_density(flux_atm, XC, YC, RAC, ocean, dlon, dlat):
    """Conservatively bin a per-cell flux (kgCO2/m2/s, +into atm) point cloud
    onto a regular lat-lon grid: cell density = sum(flux*area)/A_latlon, so the
    global integral is preserved. Returns (density (nlat,nlon), lon_c, lat_c, A)."""
    nlon = int(round(360.0 / dlon)); nlat = int(round(180.0 / dlat))
    lon_c = -180.0 + (np.arange(nlon) + 0.5) * dlon
    lat_c = -90.0 + (np.arange(nlat) + 0.5) * dlat
    lon, lat, area = XC[ocean], YC[ocean], RAC[ocean]
    fa = flux_atm[ocean] * area
    ix = np.clip(((lon + 180.0) / dlon).astype(int), 0, nlon - 1)
    iy = np.clip(((lat + 90.0) / dlat).astype(int), 0, nlat - 1)
    sum_fa = np.bincount(iy * nlon + ix, weights=fa, minlength=nlon * nlat).reshape(nlat, nlon)
    A = latlon_cell_areas(lat_c, dlon, dlat)
    return sum_fa / A[:, None], lon_c, lat_c, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux-dir", required=True)
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dlon", type=float, default=0.5)
    ap.add_argument("--dlat", type=float, default=0.5)
    ap.add_argument("--per-day-out", default="",
                    help="also write a (time, lat, lon) NetCDF of the per-day flux (for viz)")
    args = ap.parse_args()

    flux_dir = os.path.expanduser(args.flux_dir)
    grid_dir = os.path.expanduser(args.grid_dir)
    out = os.path.expanduser(args.out)

    XC = read_compact(os.path.join(grid_dir, "XC.data"))     # lon [-180,180]
    YC = read_compact(os.path.join(grid_dir, "YC.data"))     # lat [-90,90]
    RAC = read_compact(os.path.join(grid_dir, "RAC.data"))   # cell area (m^2)
    DEP = read_compact(os.path.join(grid_dir, "Depth.data")) # ocean depth (m)

    files = sorted(glob.glob(os.path.join(flux_dir, "CO2_flux.*.data")))
    if not files:
        sys.exit(f"ERROR: no CO2_flux.*.data in {flux_dir}")
    print(f"Averaging {len(files)} daily files -> monthly mean")

    acc = np.zeros(N); ndays = 0
    for f in files:
        fl = read_compact(f)
        fl[fl == MISSING] = 0.0
        acc += fl; ndays += 1
    native = acc / ndays                       # monthly-mean native flux (mmol/m2/s, +into ocean)

    ocean = (DEP > 0) & np.isfinite(native) & (np.abs(native) < 1.0)
    # model surface flux: +into atmosphere, kg CO2 / m^2 / s
    flux_atm = -native * M_CO2_KG_PER_MMOL
    flux_atm[~ocean] = 0.0

    # --- diagnostics on the native point cloud (ground truth for conservation) ---
    area = RAC[ocean]
    native_mean = np.sum(native[ocean] * area) / np.sum(area)
    global_kg_s = np.sum(flux_atm[ocean] * area)            # net kg CO2/s into atmosphere
    print(f"  ocean cells           : {int(ocean.sum())}")
    print(f"  native area-wtd mean  : {native_mean:+.4e} mmol CO2/m2/s (+into ocean)")
    print(f"  net to-atm flux       : {global_kg_s:+.4e} kgCO2/s "
          f"= {global_kg_s*3.15576e7/1e12:+.3f} PgCO2/yr "
          f"({global_kg_s*3.15576e7/1e12*12.0107/44.0095:+.3f} PgC/yr)")

    # --- conservative binning to a regular lat-lon grid ---
    dlon, dlat = args.dlon, args.dlat
    density, lon_centers, lat_centers, A = bin_flux_density(flux_atm, XC, YC, RAC, ocean, dlon, dlat)
    nlon, nlat = len(lon_centers), len(lat_centers)

    # conservation check after binning
    A2d = np.broadcast_to(A[:, None], density.shape)
    binned_total = np.sum(density * A2d)
    print(f"  binned net to-atm flux: {binned_total:+.4e} kgCO2/s "
          f"(rel err vs point cloud: {abs(binned_total-global_kg_s)/abs(global_kg_s):.2e})")
    print(f"  grid: {nlon} lon x {nlat} lat @ {dlon}x{dlat} deg")

    # --- write NetCDF ---
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds = Dataset(out, "w", format="NETCDF4")
    ds.createDimension("lon", nlon)
    ds.createDimension("lat", nlat)
    vlon = ds.createVariable("lon", "f8", ("lon",)); vlon.units = "degrees_east"; vlon[:] = lon_centers
    vlat = ds.createVariable("lat", "f8", ("lat",)); vlat.units = "degrees_north"; vlat[:] = lat_centers
    # CF (lat, lon) order. NCDatasets reverses C-order on read, so the model's
    # surface-flux loader sees raw[i_lon, j_lat] as it expects.
    v = ds.createVariable("co2_flux_ocean", "f4", ("lat", "lon"), zlib=True)
    v.units = "kgCO2/m2/s"
    v.long_name = "Air-sea CO2 flux into the atmosphere (ECCO-Darwin v5, monthly mean)"
    v.sign_convention = "positive = flux into the atmosphere (ocean outgassing)"
    v[:, :] = density.astype(np.float32)
    ds.source = ("ECCO-Darwin v5 daily fluxCO2 (LLC270), NASA NAS; native mmol CO2/m2/s "
                 "(+into ocean) sign-flipped and x44.0095e-6 -> kgCO2/m2/s (+into atmosphere)")
    ds.n_daily_files = ndays
    ds.close()
    print(f"wrote {out}")

    # --- optional per-day (time, lat, lon) NetCDF for visualization ---
    if args.per_day_out:
        pdo = os.path.expanduser(args.per_day_out)
        os.makedirs(os.path.dirname(pdo), exist_ok=True)
        stack = np.zeros((len(files), nlat, nlon), dtype=np.float32)
        for k, f in enumerate(files):
            fl = read_compact(f); fl[fl == MISSING] = 0.0
            ocean_k = (DEP > 0) & np.isfinite(fl) & (np.abs(fl) < 1.0)
            fa = -fl * M_CO2_KG_PER_MMOL; fa[~ocean_k] = 0.0
            dens_k, _, _, _ = bin_flux_density(fa, XC, YC, RAC, ocean_k, dlon, dlat)
            stack[k] = dens_k.astype(np.float32)
        ds = Dataset(pdo, "w", format="NETCDF4")
        ds.createDimension("time", len(files)); ds.createDimension("lat", nlat); ds.createDimension("lon", nlon)
        ds.createVariable("lon", "f8", ("lon",))[:] = lon_centers
        ds.createVariable("lat", "f8", ("lat",))[:] = lat_centers
        ds.createVariable("day", "i4", ("time",))[:] = np.arange(1, len(files) + 1)
        v = ds.createVariable("co2_flux_ocean", "f4", ("time", "lat", "lon"), zlib=True)
        v.units = "kgCO2/m2/s"; v.sign_convention = "positive = into the atmosphere"
        v[:, :, :] = stack
        ds.close()
        print(f"wrote {pdo} ({len(files)} daily slices)")


if __name__ == "__main__":
    main()
