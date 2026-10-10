# Heritage validation

Scripts in `heritage/` are not part of a maintained workflow. They are kept
for reference, as the method or evidence of a finished study, until they are
trimmed. They may hard-code the paths, dates and data of that study, and they
are not tested; check imports and inputs before running one. Retired scripts
are listed under "Retired scripts" in [`scripts/README.md`](../../README.md).

| Script | Purpose | Why it is here |
|---|---|---|
| `assess_cs_merge_groups.jl` | Evaluates candidate vertical layer-merge groups for a CS binary (CFL ratios, vector-spread and turnover proxies) | Design tool of the May 2026 full-L72 study; layer merging is now set in preprocessing configs |
| `campaign5d_plots.jl` | Plots CATRINE 5-day campaign deltas (overview, F32-F64, CPU-GPU) with the Visualization API | Figures of the finished April 2026 5-day campaign; its data and `config/runs/catrine5d/` configs exist |
| `campaign5d_summary.jl` | Collates the CATRINE 5-day run matrix into CSVs (mass drift, CPU-GPU, F32-F64, LL-CS) | Summary collator of the same campaign; uses only NCDatasets |
| `compare_geosit_reference.jl` | Compares an ERA5-on-GEOS-native CS binary with GEOS-IT C180 (metadata, PS correlation, column wind vectors) | Geometry, sign and flux-scale validator from the April 2026 ERA5-to-GEOS-native work. Defaults to a personal `raw_catrine` path |
| `diagnose_cs_courant.jl` | Per-window, per-level Courant-style ratios (x/y/z/CMFMC/TM5) of CS binaries to CSV | `inspect_transport_binary.jl` does not report CFL; from the layer-merge study |
| `validate_geos_native_lonlat.jl` | Checks `CubedSphereMesh` GEOS-native cell centers and corners against the lons/lats of a raw GEOS-IT C180 file | Independent-reference panel-convention check; no test compares against a real GEOS file (candidate for `test/real_data/`) |
| `validate_regridding.py` | Produces an xESMF conservative-regridding reference of a lat-lon emission field on the GEOS CS grid | External cross-check method for ConservativeRegridding; nothing consumes its output. Needs xesmf/xarray |
| `write_geosit_agreement_netcdf.jl` | Writes layer-by-layer agreement diagnostics between an ERA5-on-GEOS-native L72 binary and GEOS-IT C180 to NetCDF | Stricter companion of `compare_geosit_reference.jl`; hard-codes the GEOS-IT root |
