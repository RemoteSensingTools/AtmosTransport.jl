# Heritage downloads

Scripts in `heritage/` are not part of a maintained workflow. They are kept
for reference, as the method or evidence of a finished study, until they are
trimmed. They may hard-code the paths, dates and data of that study, and they
are not tested; check imports and inputs before running one. Retired scripts
are listed under "Retired scripts" in [`scripts/README.md`](../../README.md).

| Script | Purpose | Why it is here |
|---|---|---|
| `download_era5_surface_netcdf.py` | Downloads one month of ERA5 hourly single-level surface fields from CDS (blh, zust, 2t, 10u/10v, sp, z, lsm, 2d, sshf, slhf) as `sfc_an_native/era5_surface_YYYYMM.nc` for the N320 to C180 surface/bldiff payload | Only CDS route that writes the monthly NetCDF layout `src/Preprocessing/era5_surface_reader.jl` still accepts. ARCO (`config/downloads/era5_arco.toml`) is the default ERA5 surface route |
