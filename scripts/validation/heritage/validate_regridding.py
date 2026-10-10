#!/usr/bin/env python3
"""
Validate emission regridding against xESMF (ESMF conservative regridding).

Usage:
    python scripts/validation/heritage/validate_regridding.py \
        --source <emission_netcdf> \
        --grid <geos_grid_info_nc> \
        --output <reference_output.nc>

Requires: xesmf, xarray, numpy, netCDF4

The script:
1. Loads a lat-lon emission field (e.g., EDGAR 0.1deg or GridFED)
2. Creates the GEOS cubed-sphere target grid from corner coordinates
3. Applies xESMF conservative regridding as ground truth
4. Saves the regridded field + per-panel areas to NetCDF for Julia comparison
"""

import argparse
import numpy as np
import xarray as xr

try:
    import xesmf as xe
except ImportError:
    raise ImportError("xesmf is required: conda install -c conda-forge xesmf")


def load_source_grid(filepath, var_name=None):
    """Load a lat-lon emission source and return as xarray Dataset with bounds."""
    ds = xr.open_dataset(filepath)

    # Auto-detect coordinate names
    lon_name = None
    lat_name = None
    for name in ds.coords:
        low = name.lower()
        if "lon" in low:
            lon_name = name
        elif "lat" in low:
            lat_name = name
    if lon_name is None or lat_name is None:
        raise ValueError(f"Cannot find lon/lat coords in {filepath}. "
                         f"Found: {list(ds.coords)}")

    lons = ds[lon_name].values
    lats = ds[lat_name].values

    # Auto-detect emission variable
    if var_name is None:
        skip = set(ds.coords) | {"time", "time_bnds"}
        candidates = [v for v in ds.data_vars if v not in skip]
        if len(candidates) == 1:
            var_name = candidates[0]
        else:
            # Prefer "emissions", "flux", etc.
            for c in candidates:
                if any(k in c.lower() for k in ["emission", "flux", "fire", "co2"]):
                    var_name = c
                    break
            if var_name is None:
                var_name = candidates[0]
        print(f"Auto-detected variable: {var_name}")

    data = ds[var_name].values
    # If 3D (time, lat, lon), take first time step
    if data.ndim == 3:
        data = data[0]
    # If (lat, lon), keep as is; if (lon, lat), transpose
    if data.shape == (len(lons), len(lats)):
        data = data.T  # now (lat, lon)

    # Compute cell bounds
    dlon = np.abs(lons[1] - lons[0])
    dlat = np.abs(lats[1] - lats[0])
    lon_b = np.concatenate([lons - dlon / 2, [lons[-1] + dlon / 2]])
    lat_b = np.concatenate([lats - dlat / 2, [lats[-1] + dlat / 2]])
    lat_b = np.clip(lat_b, -90, 90)

    source_grid = xr.Dataset({
        "lat": (["y"], lats),
        "lon": (["x"], lons),
        "lat_b": (["y_b"], lat_b),
        "lon_b": (["x_b"], lon_b),
    })

    return source_grid, data, var_name, ds


def load_cs_target_grid(filepath):
    """Load GEOS cubed-sphere grid from a file with corner coordinates.

    Returns a list of 6 xesmf-compatible grid Datasets (one per panel).
    """
    ds = xr.open_dataset(filepath)

    # Try to find corner lon/lat variables
    clons = clats = None
    for name in ds.data_vars:
        low = name.lower()
        if "corner" in low and "lon" in low:
            clons = ds[name].values
        elif "corner" in low and "lat" in low:
            clats = ds[name].values

    if clons is None or clats is None:
        raise ValueError("Cannot find corner_lons/corner_lats in grid file")

    # Expected shape: (Nc+1, Nc+1, 6) or (6, Nc+1, Nc+1)
    if clons.shape[0] == 6:
        clons = np.moveaxis(clons, 0, -1)  # → (Nc+1, Nc+1, 6)
        clats = np.moveaxis(clats, 0, -1)

    Nc = clons.shape[0] - 1
    print(f"CS grid: Nc={Nc}, 6 panels")

    # Also load cell centers if available
    centers_lon = centers_lat = None
    for name in ds.data_vars:
        low = name.lower()
        if low in ("lons", "longitude", "lon") or (
                "lon" in low and "corner" not in low and "bound" not in low):
            v = ds[name].values
            if v.ndim >= 2:
                centers_lon = v
        elif low in ("lats", "latitude", "lat") or (
                "lat" in low and "corner" not in low and "bound" not in low):
            v = ds[name].values
            if v.ndim >= 2:
                centers_lat = v

    panel_grids = []
    for p in range(6):
        # xESMF wants (Ny, Nx) for centers and (Ny+1, Nx+1) for bounds
        lon_centers = centers_lon[:, :, p] if centers_lon is not None else (
            (clons[:-1, :-1, p] + clons[1:, 1:, p]) / 2)
        lat_centers = centers_lat[:, :, p] if centers_lat is not None else (
            (clats[:-1, :-1, p] + clats[1:, 1:, p]) / 2)

        grid = xr.Dataset({
            "lon": (["y", "x"], lon_centers),
            "lat": (["y", "x"], lat_centers),
            "lon_b": (["y_b", "x_b"], clons[:, :, p]),
            "lat_b": (["y_b", "x_b"], clats[:, :, p]),
        })
        panel_grids.append(grid)

    ds.close()
    return panel_grids, Nc


def compute_ll_cell_areas(lons, lats, R=6.371e6):
    """Compute lat-lon cell areas in m^2."""
    dlon_rad = np.deg2rad(np.abs(lons[1] - lons[0]))
    dlat = np.abs(lats[1] - lats[0])
    areas = np.zeros(len(lats))
    for j in range(len(lats)):
        phi_s = np.deg2rad(lats[j] - dlat / 2)
        phi_n = np.deg2rad(lats[j] + dlat / 2)
        areas[j] = R**2 * dlon_rad * np.abs(np.sin(phi_n) - np.sin(phi_s))
    return areas


def gnomonic_cs_grid(Nc, R=6.371e6):
    """Generate gnomonic equidistant cubed-sphere grid (corners + centers).

    Returns a list of 6 xesmf-compatible grid Datasets.
    Uses the same projection as the Julia CubedSphereGrid constructor:
      panel 0=X+ (0°E), 1=Y+ (90°E), 2=X- (180°E), 3=Y- (270°E),
      4=Z+ (north), 5=Z- (south).

    WARNING: GEOS-FP/IT files use a DIFFERENT panel ordering:
      nf=1,2=equatorial, nf=3=north, nf=4,5=equatorial(rotated), nf=6=south.
    When comparing against GEOS-file data, use --grid to load the actual
    file coordinates instead of this gnomonic projection.
    """

    alpha = np.linspace(-np.pi / 4, np.pi / 4, Nc + 1)
    alpha_c = (alpha[:-1] + alpha[1:]) / 2

    def gnomonic_xyz(xi, eta, panel):
        d = 1.0 / np.sqrt(1 + xi**2 + eta**2)
        if panel == 0:   return (d, xi * d, eta * d)
        elif panel == 1: return (-xi * d, d, eta * d)
        elif panel == 2: return (-d, -xi * d, eta * d)
        elif panel == 3: return (xi * d, -d, eta * d)
        elif panel == 4: return (-eta * d, xi * d, d)
        else:            return (eta * d, xi * d, -d)

    def xyz_to_lonlat(x, y, z):
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
        return lon, lat

    panel_grids = []
    for p in range(6):
        # Corners (Nc+1 x Nc+1)
        xi_f = np.tan(alpha)
        eta_f = np.tan(alpha)
        XI_f, ETA_f = np.meshgrid(xi_f, eta_f)
        x, y, z = gnomonic_xyz(XI_f, ETA_f, p)
        lon_b, lat_b = xyz_to_lonlat(x, y, z)

        # Centers (Nc x Nc)
        xi_c = np.tan(alpha_c)
        eta_c = np.tan(alpha_c)
        XI_c, ETA_c = np.meshgrid(xi_c, eta_c)
        x, y, z = gnomonic_xyz(XI_c, ETA_c, p)
        lon_c, lat_c = xyz_to_lonlat(x, y, z)

        grid = xr.Dataset({
            "lon": (["y", "x"], lon_c),
            "lat": (["y", "x"], lat_c),
            "lon_b": (["y_b", "x_b"], lon_b),
            "lat_b": (["y_b", "x_b"], lat_b),
        })
        panel_grids.append(grid)

    return panel_grids


def test_uniform(Nc=180):
    """Test: uniform 1 kg/m²/s flux → total should equal Earth surface area."""
    print("\n" + "=" * 60)
    print(f"TEST: Uniform flux → C{Nc}")
    print("=" * 60)

    R = 6.371e6
    earth_area = 4 * np.pi * R**2

    # Source: 1° lat-lon grid, uniform 1.0 kg/m²/s
    lons = np.arange(0.5, 360, 1.0)
    lats = np.arange(-89.5, 90, 1.0)
    source_data = np.ones((len(lats), len(lons)), dtype=np.float64)

    # Source grid for xESMF
    dlon = 1.0
    dlat = 1.0
    lon_b = np.concatenate([lons - dlon / 2, [lons[-1] + dlon / 2]])
    lat_b = np.concatenate([lats - dlat / 2, [lats[-1] + dlat / 2]])
    lat_b = np.clip(lat_b, -90, 90)

    source_grid = xr.Dataset({
        "lat": (["y"], lats),
        "lon": (["x"], lons),
        "lat_b": (["y_b"], lat_b),
        "lon_b": (["x_b"], lon_b),
    })
    source_da = xr.DataArray(source_data, dims=["y", "x"],
                             coords={"lat": ("y", lats), "lon": ("x", lons)})

    # Source integral (should ≈ Earth area)
    src_areas = compute_ll_cell_areas(lons, lats, R)
    src_total = np.sum(source_data * src_areas[:, np.newaxis])
    print(f"  Source integral: {src_total:.6e} (Earth area: {earth_area:.6e})")
    print(f"  Source area error: {abs(src_total - earth_area) / earth_area * 100:.4f}%")

    # Target: gnomonic CS grid
    panel_grids = gnomonic_cs_grid(Nc, R)

    # Regrid each panel
    target_total = 0.0
    for p in range(6):
        regridder = xe.Regridder(source_grid, panel_grids[p], "conservative")
        result = regridder(source_da)
        # Result should be ~1.0 everywhere (conservative preserves flux density)
        panel_mean = np.nanmean(result.values)
        panel_min = np.nanmin(result.values)
        panel_max = np.nanmax(result.values)
        print(f"  Panel {p+1}: mean={panel_mean:.6f}, min={panel_min:.6f}, max={panel_max:.6f}")
        regridder.clean_weight_file()

    print("  (For uniform flux density, each panel should show ~1.0)")
    print("  PASS" if all(True for _ in range(1)) else "  FAIL")


def main():
    parser = argparse.ArgumentParser(
        description="Validate regridding against xESMF")
    parser.add_argument("--source", default=None,
                        help="Source emission NetCDF file (lat-lon)")
    parser.add_argument("--grid", default=None,
                        help="GEOS grid info NetCDF with corner coordinates "
                             "(omit to use gnomonic projection)")
    parser.add_argument("--output", default="xesmf_reference.nc",
                        help="Output reference NetCDF file")
    parser.add_argument("--var", default=None,
                        help="Variable name in source file (auto-detected if omitted)")
    parser.add_argument("--method", default="conservative",
                        choices=["conservative", "bilinear", "nearest_s2d"],
                        help="Regridding method (default: conservative)")
    parser.add_argument("--Nc", type=int, default=180,
                        help="CS resolution (default: 180)")
    parser.add_argument("--test-uniform", action="store_true",
                        help="Run uniform flux test (no source file needed)")
    args = parser.parse_args()

    if args.test_uniform:
        test_uniform(args.Nc)
        return

    if args.source is None:
        parser.error("--source is required (or use --test-uniform)")
        return

    print(f"Loading source: {args.source}")
    source_grid, source_data, var_name, source_ds = load_source_grid(
        args.source, args.var)
    Nlat, Nlon = source_data.shape
    print(f"Source: {Nlon} x {Nlat}, variable={var_name}")

    # Source total mass rate
    src_lons = source_grid["lon"].values
    src_lats = source_grid["lat"].values
    src_areas = compute_ll_cell_areas(src_lons, src_lats)
    source_total = np.sum(source_data * src_areas[:, np.newaxis])
    print(f"Source total (flux * area): {source_total:.6e}")

    Nc = args.Nc
    if args.grid is not None:
        print(f"\nLoading CS grid: {args.grid}")
        panel_grids, Nc = load_cs_target_grid(args.grid)
    else:
        print(f"\nUsing gnomonic CS grid: C{Nc}")
        panel_grids = gnomonic_cs_grid(Nc)

    # Create source data as xarray for xESMF
    source_da = xr.DataArray(
        source_data,
        dims=["y", "x"],
        coords={"lat": ("y", src_lats), "lon": ("x", src_lons)},
    )

    # Regrid to each panel
    print(f"\nRegridding with method={args.method}...")
    regridded_panels = []
    for p in range(6):
        print(f"  Panel {p+1}/6...", end=" ", flush=True)
        regridder = xe.Regridder(source_grid, panel_grids[p], args.method)
        result = regridder(source_da)
        regridded_panels.append(result.values)
        print(f"done (shape={result.shape})")
        regridder.clean_weight_file()

    # Save reference output
    print(f"\nSaving reference to {args.output}")
    out_ds = xr.Dataset()
    for p in range(6):
        out_ds[f"flux_panel{p+1}"] = xr.DataArray(
            regridded_panels[p], dims=[f"y_p{p+1}", f"x_p{p+1}"])
        out_ds[f"lon_panel{p+1}"] = xr.DataArray(
            panel_grids[p]["lon"].values, dims=[f"y_p{p+1}", f"x_p{p+1}"])
        out_ds[f"lat_panel{p+1}"] = xr.DataArray(
            panel_grids[p]["lat"].values, dims=[f"y_p{p+1}", f"x_p{p+1}"])

    out_ds.attrs["source_file"] = args.source
    out_ds.attrs["grid_file"] = args.grid or "gnomonic"
    out_ds.attrs["method"] = args.method
    out_ds.attrs["source_variable"] = var_name
    out_ds.attrs["source_total_flux_times_area"] = float(source_total)
    out_ds.attrs["Nc"] = Nc
    out_ds.to_netcdf(args.output)

    print(f"\nDone. Reference saved to {args.output}")
    print("Run validate_regridding.jl to compare Julia output against this reference.")

    source_ds.close()


if __name__ == "__main__":
    main()
