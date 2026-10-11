# Porting the Python surface-flux preprocessing to Julia: design (2026-10-10)

**Status.** This is the agreed design. The owner approved the full port on
2026-10-10 (Section 0). Step P0 (probes, fresh Python references, a file
comparator) has run, and its corrections are applied in place: every statement
that P0 changed is marked **P0:**. The measurements, with their commands and
data paths, are in `2026-10-10_python_flux_port_p0_facts.md`.

**Base.** Commit `71a08395` ("Carbon fluxes converted by M(CO2)/M(C), not
44/12"). Code paths are relative to the repository root. The synthesis this
memo is written from is `/temp1/cfranken/flux_port_parity/design_final.md`.
Parity data (Python references, probes, comparator reports) live under
`/temp1/cfranken/flux_port_parity/`.

**Origin.** Three candidate designs were scored by a judge: LIBRARY (47 of 60),
PARITY (46) and RUNTIME (40). This design uses:
- LIBRARY as the base: one module owns the file conventions, a typed builder,
  the TRENDY quirks as a data catalog, exact conservative remapping;
- PARITY's validation: fresh Python references, legacy emulators that live only
  in the validation harness, a generic file comparator, and never overwriting
  existing files;
- RUNTIME's runtime pieces: public emission integrals, the span-independence
  rule, the `species` key, and the GPU-memory analysis that became R1.

---

## 0. Owner decisions (2026-10-10)

| Decision | What it means for the port |
|---|---|
| **Full port** | Every core routine moves to Julia: the 12 flux preprocessors, the 2 campaign runners and the numeric diagnostics. Only download and plotting scripts stay in Python. |
| **New versioned paths; the owner switches live configs** | Regenerated drivers are written to new versioned paths. No existing file is overwritten. The owner repoints each live run config in P11, one campaign at a time. Finished campaigns keep their legacy files. |
| **`species` replaces `molar_mass_kg_mol`** | `surface_flux.species = "co2"` takes the molar mass from `Parameters` (`SPECIES_MOLAR_MASS`). The old key is removed outright in P2c; its 236 occurrences in `config/runs/` are migrated. |
| **R1 after P6** | After P6, the runtime keeps each `cs_native` flux series on the host and uploads only the slices of the current met window to the device (Section 5). |

The other defaults of the design and the open questions are in Section 6.

---

## 1. Scope

Each Python file gets one of three dispositions:
- **Port**: replaced by Julia. The Python file is deleted in the commit where its
  replacement passes validation, and listed in a new "Retired scripts" table in
  `scripts/README.md` with the last commit that contains it (so
  `git show <commit>:<path>` recovers it).
- **Split**: the numbers move to Julia; a plotting-only Python script reads the
  Julia NetCDF or JSON.
- **Retire**: the script moves to the `heritage/` folder for its purpose, with a
  row in that folder's `README.md`.

Before every deletion, run `grep -rn "<module>" scripts test docs config`.

### 1a. Flux preprocessors (12): all become configs of `Preprocessing.SurfaceFluxes`

| # | Script (`scripts/preprocessing/`) | Replacement (`config/surface_fluxes/…`) |
|---|---|---|
| 1 | `prepare_trendy_s3_gpp_ter_c90.py` (2021 hourly; also the shared module `T`) | `trendy_v14_s3_hourly_gpp_ter_c90_2021.toml`. TER is the sum Ra + Rh formed before the time construction (pilot semantics). GPP gets `DiurnalDisaggregation(FileDiurnalFractions(…; fallback = CosineZenith(MidMonthDay())))`. The pieces of `T` become library types plus `catalogs/trendy_v14_s3.toml`. Deleted in P8, after its last importer is gone. |
| 2 | `prepare_trendy_s3_multiyear_c90.py` | `trendy_v14_s3_daily_gpp_ra_rh_c90_2014_2024.toml`: an ensemble over the catalog, any Nc through `[target]`. LPJ-GUESS uses `AnnualAsMonthly` on its annual `arh` file. **P0:** the Python script has never built LPJ-GUESS (it stops with `KeyError 'rh'` on `LPJ-GUESS_S3_arh.nc`), so this is a new capability with no multi-year Python reference. |
| 3 | `prepare_trendy_s3_npp_rh.py` | `trendy_v14_s3_daily_npp_rh_c30_2014_2024.toml`, C90 through `[target]`. The manifest keys `models.<m>.status` and `.output` are kept. |
| 4 | `prepare_gfed5_fire_flux.py` | `gfed5_fire_daily_{c90,c30}_2014_2024.toml`: a `SourceSequence` (finalized `C` until 2022, then NRT `EM` summed over `lct`) in per-cell units. |
| 5 | `prepare_climatological_nee_flux.py` | `jules_es_nee_climatology_daily_c30_2014_2024.toml`: `CubedSphereNetCDFSource` → `MonthlyClimatology(2015:2024)` → `RepeatAnnualCycle` → `MidMonthInterpolation(SignedParts())`. |
| 6 | `prepare_fluxcom_x_gpp_ter_c90.py` | `fluxcom_x_hourly_gpp_ter_c90_2021.toml`: TER = max(0, max(0, GPP) + NEE) as a `DerivedSource` on the source grid; `CycleDiurnalShape` with fallback `CosineZenith(ActualDay())`. |
| 7 | `hourly_c90_flux_to_daily.py` | `sif_gpp_daily_c90_2021.toml`: `DailyMeanOfHourly`, with `expect_sign = "non_positive"` on the output. |
| 8 | `eccodarwin_co2flux_to_latlon.py` | `eccodarwin_ocean_co2_dec2008_05deg.toml` (`LLCCompactSource` + `PointBinning` → lat-lon writer; the same 0.5° product), plus a `_perday` variant for the heritage visualisation. |
| 9 | `build_sif_gpp_multiyear_c90_flux.py` | `sif_gpp_iav_daily_c90_2018_2025.toml`: `SIFCalibratedGPPSource`, `DayOfYearClimatology(CircularRunningMean(15))`, `CellTimeMean`, five outputs. |
| 10 | `make_transcom_sif_flux_tags.py` | `sif_gpp_transcom_boxes_c180_dec2021.toml`: `LatLonBoxRegions` from `regions/transcom_land_boxes_pilot.toml`, `RegionVariablesInOneFile("CO2_FLUX_UNASSIGNED")`. |
| 11 | `prepare_transcom_c30.py` | `trendy_gfed5_transcom11_c30_2014_2024.toml`: `RasterRegions` (exact overlap fractions, nearest-labelled fallback) with `FilePerRegion`, plus the weights file and the partition audit. |
| 12 | `refine_cs_surface_flux.py` | `sif_gpp_hourly_c180_dec2021.toml`: CS source on C90, target C180; `build_regridder` returns `NestedCubedSphereRegrid{Prolongation}`. |

All twelve are **Port**. Also absorbed:
- `scripts/preprocessing/catrine_protocol_cs_fluxes.jl` becomes
  `catrine_protocol_c90_2021_2023.toml` (P10). This removes the last callers of
  private `InitialConditionIO` functions.
- `config/emissions/*.toml` is read by nothing and is removed in P3.

### 1b. Campaign runners (2): one generic Julia controller

| Script | Disposition | Replacement |
|---|---|---|
| `scripts/run_trendy_npp_rh_campaign.py` | Port and merge (C1) | `scripts/campaigns/run_campaign.jl` + `config/campaigns/trendy_allmodels_npp_rh_c30.toml`. The 23-model C90 rerun needs a controller. |
| `scripts/run_transcom_campaign.py` | Port and merge; deleted in P7 with `prepare_transcom_c30.py`, which it imports | `config/campaigns/trendy_transcom_c30.toml` for the same controller. `summarize()` becomes `Analysis.band_means`. Tolerances are not loosened. |

### 1c. Numeric and mixed diagnostics (9)

| Script (`scripts/diagnostics/`) | Disposition | Replacement |
|---|---|---|
| `check_trendy_npp_rh_run.py` | Port (merge) | Daily-series mode of `tracer_budget_closure.jl`, built on `Analysis.budget_closure`: `<tracer>_total_mass`, tracers and carriers from `run.toml`, carrier-ULP tolerance, JSON report, non-zero exit on failure. |
| `catrine_true_mass_balance_gc.py` | Port (merge) | `catrine_true_mass_balance.jl --model geoschem` through `Analysis.GEOSChemCATRINEArchive`. Same CSV schema; GEOS-Chem molar masses from `GEOSCHEM_SPECIES_MOLAR_MASS`. |
| `float32_conservation_attribution.py` | Port | `float32_conservation_attribution.jl write-configs\|budget`: configs built as Dicts, `validate_config`, `TOML.print`; tag and file prefix are arguments. |
| `compare_float32_float64_runs.py` | Split | `.jl` writes JSON and a NetCDF (daily series, last-day difference field). Plot: `scripts/visualization/plot_float32_vs_float64.py`. |
| `catrine_compare_vs_geoschem.py` | Split | `.jl` writes `summary.json` (same keys), `timeseries.nc` and `monthly_fields_c90.nc`. Plot: `scripts/visualization/plot_catrine_benchmark.py`. `catrine_benchmark_page.py` and `animate_catrine_benchmark.py` move to `scripts/visualization/` and read NetCDF; per-frame extraction moves to Julia. |
| `compare_sif_gpp_transport_control.py` | Split | `.jl` computes the RMS and correlation numbers and writes JSON and band arrays (NetCDF). Plot: `scripts/visualization/plot_sif_flux_vs_transport.py`. `plot_xco2_hovmoller.py` reads the NetCDF instead of importing `load_daily`. |
| `rank_trendy_hovmoller.py` | Split | `.jl` does extraction, decomposition, scores, CSV/JSON and `REPORT.md`. Plot: `scripts/visualization/plot_trendy_ranking.py`. The wait/lock logic is dropped. |
| `analyze_sif_gpp_iav_xco2.py` | Port | `.jl`: bitwise NetCDF, plus an exact `total_mass` closure that replaces the 2.124 PgC/ppm check. |
| `catrine_station_timeseries.py` | Retire → `scripts/diagnostics/heritage/` | Broken against the current GEOS-Chem layout and referenced nowhere. The runtime `[output.observations]` sampler covers future runs. |

### 1d. Python tests (2)

| Test | Disposition | Replacement |
|---|---|---|
| `test/diagnostics/test_transcom_partition.py` | Port (P7) | `test/core/test_surface_flux_regions.jl` |
| `test/diagnostics/test_trendy_hovmoller.py` | Port (B3) | `test/core/test_analysis_decomposition.jl`; `test/diagnostics/` is removed once empty |

Python left afterwards: plotting and presentation (`scripts/visualization/*.py`,
the moved page and animation scripts, `plot_sif_gpp_*`,
`plot_xco2_hovmoller.py`) and download scripts.

---

## 2. Architecture

### 2.1 Modules and load order

New modules are marked `*`:

```
Architectures → SectionTimer → ConfigChecks → LogLevels → *AtomicFiles → Quantities → Parameters
→ Grids → State → Output → MetDrivers → Operators → Tape → Adjoints → Regridding → *GriddedIO
→ Preprocessing (+ *Preprocessing.SurfaceFluxes) → Visualization → Models → *Analysis → Downloads
```

| Module | Path | Imports | Contract |
|---|---|---|---|
| `AtomicFiles` | `src/AtomicFiles.jl` | Base only | Publish a file only when it is complete. The temp file is a sibling, created with `O_EXCL` and mode `0o666 & ~umask` (never `mktemp`); renamed on success, removed on error. |
| `GriddedIO` | `src/GriddedIO/` | Parameters, Grids, Output, AtomicFiles, NCDatasets, Dates | The single owner of the conventions of native gridded input files: coordinate discovery, fill policy, lat-lon orientation, source lat-lon meshes, time decoding, staging of compressed inputs, flux-unit grammar, and the `cs_native` and lat-lon flux schemas (reader and writer). Used by initial conditions, the runtime flux loaders and the builder. |
| `Preprocessing.SurfaceFluxes` | `src/Preprocessing/surface_fluxes/` | GriddedIO, Regridding, Quantities, Grids, Parameters, Output (`cell_locator`), MetDrivers (mesh from a binary), ConfigChecks, LogLevels, JSON3, SHA, TOML, FileWatching | The typed surface-flux builder. Never imports `Models`. Top-level export: `build_surface_fluxes`. |
| `Analysis` | `src/Analysis/` | Models, GriddedIO, Operators.SurfaceFlux, Visualization | Shared numerics for post-run diagnostics and the campaign controller. |

How this respects the load order:
- The conventions sit below both consumers, so the builder never needs `Models`.
- The builder's readback check calls `GriddedIO.read_cs_native_flux`, which
  after P2 is the runtime's own reader.
- `Models.InitialConditionIO` keeps only runtime concerns: the storage scale,
  the tracer → species mapping and the temporal scheme.
- `CLAUDE.md` (code map) and `docs/src/concepts/architecture.md` are updated
  with the new order.

### 2.2 De-duplication map

These are moves; the old definitions are deleted.

| Today | New home |
|---|---|
| `_normalize_units_string`, `_KG_PER_M2_S`, `_per_second_flux_scale`, the GridFED/EDGAR unit branches (`surface_flux.jl`) | `GriddedIO.parse_flux_units`, `mass_rate_factor` |
| `_days_in_month`, `_is_leap_year` | `Dates.daysinmonth`, `Dates.isleapyear` |
| `_ic_find_coord` (`InitialConditionIO.jl`) | `GriddedIO.find_coordinate`, with a size check. **P0:** by variable name only; never by `units`, `long_name` or `standard_name` (Section 2.4). |
| The two lat-flip / lon-roll copies (static and time-varying loaders) | `GriddedIO.latlon_orientation` / `orient` |
| `_build_source_latlon_mesh` (`cubed_sphere.jl`), `_lonlat_cell_areas_m2` | `GriddedIO.source_latlon_mesh` on the new `Grids.LatLonMesh(λᶠ, φᶠ; radius)`; areas from `cell_area(::LatLonMesh)` |
| `_surface_flux_times_seconds`, `_parse_cf_time_origin` | `GriddedIO.decode_time`, `seconds_since` (adds noleap/365_day, `months since`, decimal years) |
| File half of `_build_native_timevarying_cs_surface_flux_source`; `_check_native_flux_cell_area` | `GriddedIO.read_cs_native_flux`, `check_cell_area` |
| `_regrid_cache_dir` (Models) | `Regridding.default_regrid_cache_dir()` (runtime and builder share weights) |
| `_density_correct!` (`cs_transport_helpers.jl`) | `Regridding.apply_regridder!(dst, r, src, ::QuantityKind)` |
| Both CS coarsening kernel sets: Set A in `transport_binary/cubed_sphere_coarsen.jl`, Set B in `transport_binary/geos_cs_resolution.jl` | One set in `Regridding/cs_nesting.jl`. **P0:** merged on Set A semantics (Float64 accumulation, one rounding), plus a 3-D area-weighted variant that accumulates the same way. The sum and face-sum kernels of the two sets are bitwise identical; the Float32 area-weighted means are not (Section 6, open question 3). |
| `_write_transport_binary_atomically`, `_open_streaming_staging`, the temp+mv in `_write_spectral_day_cache`, `mktemp` in `save_regridder` | `AtomicFiles` |
| CS geometry block of `_define_cs_geometry!`; `_define_provenance_attributes!` | `Output.define_cs_horizontal_geometry!(ds, mesh)`, `Output.define_provenance_attributes!(ds)` (public; two owners) |
| `DrivenRunner._surface_source_total_rate` (`hasproperty`); private `_flux_temporal_segments` used by two scripts | `Operators.SurfaceFlux.total_emission_rate` / `integrated_emission` |
| `write_series` in `catrine_protocol_cs_fluxes.jl` | `GriddedIO.open_cs_native_writer` |
| `molar_mass_kg_mol = 0.0440095` × 236 in configs; `_KNOWN_TRACER_MOLAR_MASS_KG_MOL` used by a script | `surface_flux.species = "co2"` → `SPECIES_MOLAR_MASS` (P2c) |

### 2.3 Foundation changes (P1–P2)

```julia
# src/AtomicFiles.jl
atomic_publish(f, path::AbstractString) -> String
    # f(tmp) writes tmp = "$(path).tmp-$(getpid())-$(rand(UInt32))" (reserved with O_EXCL, 0o666 & ~umask);
    # mv(tmp, path; force = true) on return; rm(tmp) and rethrow on error.
open_atomic_staging(path) -> (tmp::String, io::IOStream)   # moved from streaming_writer.jl
commit_atomic_staging(tmp, path); abandon_atomic_staging(tmp)

# src/Grids/LatLonMesh.jl
LatLonMesh(λᶠ::AbstractVector, φᶠ::AbstractVector; FT = Float64, radius)
    # Δλ uniform (rtol 1e-10), span ≤ 360°; the first face may be any value (P0: not only −180 or 0).
    # φᶠ strictly monotone in [−90, 90] (P0: descending files exist).
    # Non-uniform φᶠ (pole-centred, truncated, Gaussian): field Δφ = NaN; dx, dy and _face_length_x
    # throw ArgumentError, so such a mesh cannot enter transport code silently.

# src/Regridding/quantity_regrid.jl
apply_regridder!(dst, r, src, kind::QuantityKind)   # Intensive: as today; Extensive: ÷ r.src_areas, regrid, × r.dst_areas
regrid_conservation_report(r, src, dst, kind) -> (; src_total, dst_total, rel_error)
default_regrid_cache_dir() -> String                # ATMOSTR_REGRID_CACHE_DIR or ~/.cache/AtmosTransport/cr_regridding

# src/Regridding/cs_nesting.jl
meshes_nested(a::CubedSphereMesh, b::CubedSphereMesh) -> Bool    # same definition, convention, offset, radius; integer Nc ratio
abstract type AbstractNestingDirection end; struct Restriction <: … end; struct Prolongation <: … end
struct NestedCubedSphereRegrid{D<:AbstractNestingDirection}; ratio::Int; src_areas::Vector{Float64}; dst_areas::Vector{Float64}; end
    # build_regridder: identity → nested → CR.jl. Restriction = area-weighted block sum; prolongation splits by area fraction.
coarsen_sum!(dst, src, r); coarsen_area_weighted!(dst, src, area, r)
    # the one copy of the block kernels, Set A semantics (P0): Float64 accumulators, one rounding to FT;
    # 2-D and 3-D; Float64 areas accepted with Float32 data; Int ratio (Val{R} gave no measurable speed-up)

# src/Output/netcdf_schema.jl
define_cs_horizontal_geometry!(ds, mesh::CubedSphereMesh)  # Xdim/Ydim/nf, lons, lats, corners, cell_area, cubed_sphere mapping (semi_major_axis = mesh.radius)
define_provenance_attributes!(ds) -> (creation_date, git_commit, git_dirty)

# src/Operators/SurfaceFlux/integrals.jl
total_emission_rate(src::AbstractSurfaceFluxSource, t) -> Float64       # storage kg s⁻¹ summed over cells at t (s since run start)
integrated_emission(src::AbstractSurfaceFluxSource, t0, t1, dt) -> Float64
    # Σ over the run's steps of length dt of _flux_temporal_segments: exactly what the operator applies.
    # Replaces _surface_source_total_rate (DrivenRunner logging) and the private calls in the two scripts.
```

**Source-grid cases the explicit-face mesh path must accept (P0).** Measured
on the 23 usable TRENDY v14 S3 models:

| Case | Models | Detail |
|---|---|---|
| First longitude face not ±180 or 0 (a centre sits on 0° or ±180°, so the dateline falls inside a cell) | CLASSIC −180.5, CLM −0.625, CLM-FATES and ELM-FATES −1.25, ELM −180.625, JSBACH −180.9375 | |
| Wrapped longitude centres | CLM-FATES | 0 … 177.5, then −180 … −2.5 (non-monotonic in the file) |
| Longitude 0 … 360 | CLM, ELM-FATES, ISAM | |
| Descending latitude | VISIT, VISIT-UT | 89.75 … −89.75 |
| Pole-centred latitude (outer faces at ±90) | CLM, ELM (192 rows), iMAPLE (181 rows of 1°) | |
| Truncated latitude, north pole only | CLM-FATES, ELM-FATES | southernmost centre −59.684°; physical face −60.6316° |
| Gaussian latitude | JSBACH (T63, 96 rows) | spacing varies by 0.0156°; any `dy`-based path must throw |

Within each model, gpp, ra and rh have bitwise-identical coordinates, so one
source mesh per model suffices. GFED5 finalized 2001–2022 and NRT 2023–2025 share
one 0.25° grid; 1997–2000 needs a second (1°) regridder.

**Constants** are added to `src/Parameters/PhysicalConstants.jl` in the commit
of their first user, each with a docstring and a reference:
- `EARTH_OBLIQUITY_DEG = 23.44` (Cooper 1969), in P8;
- `RN222_HALF_LIFE = 3.8235 * 86400.0` s, in B2; the literal `330350.4` in
  `RuntimePhysicsRecipe.jl` switches to it;
- `GEOSCHEM_SPECIES_MOLAR_MASS = (co2 = 44.01e-3, sf6 = 146.06e-3, rn222 = 222.0e-3)`
  (GEOS-Chem species database), in B2.

No numeric constant that could diverge is written as a literal in the builder.

### 2.4 `GriddedIO`: the file conventions

```julia
# coordinates.jl
abstract type AbstractAxis end; struct LongitudeAxis <: AbstractAxis end; struct LatitudeAxis <: AbstractAxis end; struct TimeAxis <: AbstractAxis end
find_coordinate(ds, axis::AbstractAxis; length = nothing) -> String
    # by variable name only (lat/latitude, lon/longitude, time) plus the size check (P0)
struct LatLonOrientation; flip_lat::Bool; lon_shift::Int; end
latlon_orientation(lon, lat) -> LatLonOrientation        # ascending latitude; longitude rolled to start at the first value ≥ 0
orient(A::AbstractArray, o::LatLonOrientation) -> Array  # acts on the two leading dims of any field
abstract type AbstractFillPolicy end; struct FillValueOnly <: … end; struct MaskedAsZero <: … end
read_raw(ds, var, policy::AbstractFillPolicy, idx...) -> Array{Float64}   # IBIS land fraction: FillValueOnly (its missing_value = 0 is wrong)

# source_meshes.jl
source_latlon_mesh(lon, lat; radius) -> LatLonMesh{Float64}
    # regular cell-centred axes: the snapping rule below; faces from integers
    # regular axes with a centre on 0° or ±180°: faces at centre ± Δ/2 (first face not ±180 or 0)
    # irregular latitude: faces at midpoints of adjacent centres; an outer face is ±90 only when the
    #   outer centre sits on the pole (CLM, ELM, iMAPLE), else centre ∓ half the adjacent spacing
    #   (CLM-FATES and ELM-FATES south: −60.6316°). JSBACH (Gaussian): midpoint rule (Section 6, question 2).
    # *_bnds variables are never used to build faces (P0).

# time_axes.jl
struct TimeInterval; start::DateTime; stop::DateTime; end        # [start, stop)
struct MonthKey; year::Int; month::Int; end
abstract type AbstractTimeDecoder end
struct CFDecoder <: … end; struct MonthsSince <: … end; struct DecimalYears <: … end
struct AuxiliaryTimeVariable <: …; variable::String; origin::Date; unit::Dates.Period; end   # CARDAMOM
struct AnnualFromFirstYear <: …; first_year::Int; end         # P0: not needed by any TRENDY v14 file (see below)
decode_time(ds, name, decoders::Tuple) -> Vector{DateTime}
    # classifies the axis by the units string of the raw variable (ds[name].var) BEFORE any CFTime call (P0);
    # no length guessing; time bounds are not relied on
month_keys(times) -> Vector{MonthKey}; interval(k::MonthKey) -> TimeInterval
seconds_since(times, reference::DateTime) -> Vector{Float64}

# staging.jl
stage_input(path; staging_dir::AbstractString) -> String
    # .nc passthrough; .nc.gz via `gzip -dc`, single-member .tar.gz via `tar -xzOf` (Sys.which checked);
    # cached by sha1(path, size, mtime); published with atomic_publish. No /tmp default.

# flux_units.jl
abstract type AbstractSubstance end; struct Carbon <: … end; struct Species <: …; name::Symbol; end
abstract type AbstractAreaBasis end; struct PerSquareMetre <: … end; struct PerCell <: … end
struct FluxUnits{S<:AbstractSubstance, A<:AbstractAreaBasis}; kg_per_unit::Float64; molar::Bool; per_seconds::Function; substance::S; area::A; end
parse_flux_units(s; substance = nothing) -> FluxUnits
mass_rate_factor(u::FluxUnits, species::Symbol, iv::TimeInterval) -> Float64
    # → kg `species` s⁻¹ (per m² or per cell). C → CO2 = SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS;
    # mol → kg via SPECIES_MOLAR_MASS; month length from iv; year = DAYS_PER_JULIAN_YEAR
canonical_units_string(species::Symbol) -> String       # "kg CO2 m-2 s-1"

# cs_native_flux.jl: the runtime contract, defined once
const CS_NATIVE_FLUX_DIMS = ("Xdim", "Ydim", "nf", "time")
open_cs_native_writer(path, mesh::CubedSphereMesh, intervals::Vector{TimeInterval}, variables;
                      species::Symbol, chunk_time::Int, deflate = 1, attributes = Dict()) -> CSNativeFluxWriter
write_slices!(w, name, k::UnitRange{Int}, density::AbstractArray{Float32,4}) -> w      # asserts all finite; no _FillValue
publish!(w) -> String   # time, time_bnds, mesh geometry, cell_area = mesh.cell_areas, provenance, config text + SHA-1; atomic
read_cs_native_flux(path, variable; mesh, slices = :) -> (; density::Array{Float32,4}, times::Vector{DateTime}, units::FluxUnits)
check_cell_area(ds, mesh; file) -> nothing   # element-wise max|a_file/a_mesh − 1|: > 1e-3 error, > 1e-9 warn;
                                             # semi_major_axis, when present, must equal mesh.radius (rtol 1e-12)
# latlon_flux.jl
write_latlon_flux(path, mesh::LatLonMesh, intervals, variables; species, attributes)   # CF (lat, lon[, time]), Float32, for kind = "file"
```

**Coordinates (P0).** Coordinates are found by variable name only. The
attributes cannot be trusted: FLUXCOM-X labels its `lon` variable
`units = degrees_north`, `long_name = latitude`, and GFED5 NRT `lat`/`lon` carry
no attributes at all. FLUXCOM-X coordinates are Float64 (not Float32) and carry
round-off of at most 2.84e-14° (1.1e-13 cells); GFED5 finalized coordinates are
Float32 and exact.

**Snapping rule for regular cell-centred axes (P0).** Measured for GFED5 and
FLUXCOM-X. For an axis with n stored centres x_k, k = 0 … n − 1:
1. res = 180/n (latitude) or 360/n (longitude), in Float64. The origin x₀ is −90
   for ascending latitude, +90 for descending latitude, −180 for longitude, or 0
   for a longitude axis whose minimum is ≥ 0. The direction is s = ±1.
2. k̂ = round((x_k − x₀)/(s·res) − ½). Require k̂ = k and
   |(x_k − x₀)/(s·res) − ½ − k̂| ≤ 1e-6; otherwise throw. (Measured residuals: 0
   for GFED5, ≤ 1.1e-13 for FLUXCOM-X; the bound still catches a half-cell or
   wrong-resolution error.)
3. Faces are x₀ + s·j·res for j = 0 … n, built from integers; the outer faces
   are exactly ±90 and −180/+180.

Faces are never taken from `lat_bnds`/`lon_bnds` (FLUXCOM-X `lon_bnds` run from
−179.99999999999997 to 180.00000000000003; ELM's pole-row `lat_bnds` reach
±90.471°) and never from midpoints of the stored centres.

**Fill values (P0).** VISIT, VISIT-UT and iMAPLE mask only through
`missing_value = -99999`; DLEM has no fill attribute; ELM-FATES and LPJwsl use
`_FillValue = NaN`. A reader that honours only `_FillValue` would let −99999
through for the first three, so the flux fill policy honours `missing_value`
too. No flux variable uses `scale_factor`, `add_offset` or `valid_*`.

**Time decoding (P0).** NCDatasets 0.14.15 / CFTime 0.2.10 does not reject
`months since`: it decodes silently with a fixed month of 30.436849895833 days
whatever the calendar. On the TRENDY files this gives 3201 of 3900 wrong month
keys for CABLE-POP (365_day), 2436 for LPJml (365_day) and 296 for iMAPLE (no
calendar); ISAM and VISIT come out right only by luck. Axes without units
(CARDAMOM, ELM-FATES) come back as raw numbers, and VISIT-UT gives only a
warning. Therefore `decode_time` reads the raw values and units first and never
hands a `months since` axis to CFTime:

| Units of the raw axis | Decoder | Rule | Models |
|---|---|---|---|
| `days since …`, `hours since …` | `CFDecoder` (CFTime) | calendars noleap, 365_day, 360_day, gregorian, proleptic_gregorian, or missing | CLASSIC, CLM, CLM-FATES, DLEM, ED, ELM, IBIS, JSBACH, JULES-ES, LPJ-GUESS (gpp, ra and annual `arh`), LPJwsl, LPX-Bern, ORCHIDEE, SDGVM (360_day), TEM |
| `months since …` | `MonthsSince` | month key = origin month + ⌊v + 10⁻⁶⌋; only year and month of the origin are read (CABLE-POP's origin is `1700-1-16 12:00:00`) | CABLE-POP, ISAM, LPJml, VISIT, iMAPLE |
| none, or unparseable (`years since AD 0-Jan-1st (double)`) | `DecimalYears` (named in the catalog) | month position by round-to-nearest (k for month-start stamps, k + ½ for mid-month), residual ≤ 10⁻⁶ month, else throw. A plain floor fails: ELM-FATES has 1296 values 9.1e-13 month below the integer, and floor gives 2592 bad steps. | ELM-FATES (month start), VISIT-UT (mid-month) |
| none (`time` is an index) | `AuxiliaryTimeVariable` (named in the catalog) | epoch from the `Time` variable ("days since 01/01/2003") | CARDAMOM |

LPJ-GUESS `arh` (`days since 1700-01-01`, noleap, 325 annual stamps on 1 July)
decodes with `CFDecoder` to 1700 … 2024, so `AnnualFromFirstYear` is not needed
for it. Only CLM, CLM-FATES, ED and ELM carry time bounds, so no decoder relies
on bounds. Every axis must end at the catalog's `last_month` (2024-12).

**Units (P0).** The strings in use that must parse include `kg m-2 s-1`,
`kg  m-2 s-1` (ISAM, two spaces), `kg.m-2.s-1`, `kg m$^{-2}$ s$^{-1}$`,
`kg m-2 s-1, per unit land area occupied by the PFT` (SDGVM `ra` only),
`kgC m-2 s-1`, `gC m-2 d-1`, `g C per month` (per cell), `mmol m-2 s-1`,
`kgCO2/month/m2` and `Tonnes`. Molar units without a species throw.

**Staging (P0).** SDGVM and VISIT are `.nc.gz`, iMAPLE is `.nc.tar.gz` with
exactly one member per archive; all decompress to `NETCDF3_CLASSIC`.
Decompression takes 1.5 s (SDGVM, 389 MB), 8.1–8.6 s (VISIT, 2.05 GB) and
4.7–4.9 s (iMAPLE, 1.02 GB) per file.

**Chunking (P0).** Daily `cs_native` files use chunks `(Nc, Nc, 1, 1)`
(deflate 1, shuffle). On C90 with the runtime's panel-by-panel read pattern
this layout is fastest or tied for windows of up to 31 slices, and within
3–19 % of the best layout for spans of a year or more. `(Nc, Nc, 6, 1)` is
4.9–5.8× slower for long spans and is not used. Hourly files use
`(Nc, Nc, 1, 24)` (deflate 1, shuffle, one output day per write; P0 hourly
benchmark on 8760 C90 slices of the 2021 CLM hourly driver). A variable held
constant within the UTC day (TER) is 10.4× smaller than with `(Nc, Nc, 1, 1)`
(55.2 against 574.9 MB) and reads about 2× faster for month and year spans;
diurnal GPP is 4.6 % smaller and 9–11 % faster for those spans. The cost: a
one-day run reads 25 slices (`_needed_slice_window` keeps the first stamp at or
after the run end), which crosses a chunk boundary, so GPP takes 25.7 instead of
15.7 ms per variable for that window.

After P2, `Models.InitialConditionIO` adds only three things to the reader:
1. It multiplies by `scale × DRY_AIR_MOLAR_MASS / M_species` and by the
   runtime `mesh.cell_areas`.
2. It wraps the result in `TimeVaryingSurfaceFluxSource`.
3. It chooses the temporal scheme.

### 2.5 `Preprocessing.SurfaceFluxes`: types and contracts

**Canonical internal representation.**
- Every flux field is a `Vector{Float64}` of the mass rate per cell in kg s⁻¹
  of a declared substance (carbon or a species).
- Values are magnitudes in the source's sign convention.
- Cells are in the regridder's flat order: lat-lon with `i` fastest; on the
  cubed sphere `(p−1)Nc² + i + (j−1)Nc`, which reshapes to `(Nc, Nc, 6)`.

Why: conservation is a plain sum; per-cell inventories (GFED, LLC points) are
extensive by nature; the extensive remap is exact whatever polygon areas CR.jl
uses; writers divide by the target `mesh.cell_areas`, the same areas the
runtime multiplies by.

| Role | Abstract type | Concrete types needed by the 12 ports |
|---|---|---|
| Flux source | `AbstractFluxSource` | `LatLonNetCDFSource{D}`, `CubedSphereNetCDFSource`, `LLCCompactSource`, `SIFCalibratedGPPSource`, `SourceSequence`, `DerivedSource` (linear combination on one native grid, optional clip) |
| File discovery | `AbstractFileDiscovery` | `SingleFile`, `FilePattern("{YYYY}", "{MM}")`, `SuffixSearch(dir, suffix; exclude)` |
| Native transform | `AbstractNativeTransform` | `ClipNegative` (records the negative fraction and minimum), `LandFractionWeight(file, var; complement, fill, slice = :last)`, `SumOverDimension(:lct, 16)`, `ZeroBefore(date)`, `ScaleToAnnualTotals(table)` |
| Spatial operator | `AbstractSpatialOperator` | `ConservativeRemap()` (identity, nested or CR.jl via `build_regridder`), `PointBinning()` (`Output.cell_locator`/`locate`, adding in input order), `NoRemap()` (asserts `meshes_equivalent`) |
| Temporal stage | `AbstractTemporalStage` | `AnnualAsMonthly`, `MidMonthInterpolation{P}`, `MonthlyClimatology(years)`, `RepeatAnnualCycle`, `DayOfYearClimatology{S}(years, smoothing)`, `CellTimeMean`, `DailyMeanOfHourly`, `DiurnalDisaggregation{S}`, `PeriodMean`, `PassThrough` |
| Positivity | `AbstractPositivity` | `NonNegative` (clip, then renormalise), `SignedParts` (positive and negative parts separately) |
| Smoothing | `AbstractSmoothing` | `NoSmoothing`, `CircularRunningMean(days)` (cumulative-sum differencing) |
| Diurnal shape | `AbstractDiurnalShape` | `CosineZenith{D}` with `D = MidMonthDay \| ActualDay`; `FileDiurnalFractions{F}(file, var; fallback::F)`; `CycleDiurnalShape{F}(component; fallback::F)`; `Flat` |
| Output rule | `AbstractFluxOutput` | `Combination(terms)`: Float64, rounded once. `Float32Sum(items)`: each item is a written output or a component term rounded to Float32, summed left to right in Float32. |
| Region set | `AbstractRegionSet` | `RasterRegions(file, var, codes; fallback = NearestLabelled())`, `LatLonBoxRegions(table)` (ordered, half-open, exclusions) |
| Partition layout | `AbstractPartitionLayout` | `RegionVariablesInOneFile(residual_name)`, `FilePerRegion(path_template)` |
| Assembly | (none) | `FluxComponent(source, transforms, spatial, temporal::Tuple)`, `FluxDriver(target, period, components::NamedTuple, outputs, partition, path)`, `FluxEnsemble(catalog, members, template)` |

The target is a mesh: a `CubedSphereMesh` from `[target].binary` (definition,
convention and radius from its header) or an explicit `[target].grid` that
names its radius. The writer is chosen by dispatch on the mesh type. Small
internal `AbstractCadence` types (`Hourly`, `Daily`, `Monthly`, `Annual`,
`Static`, `MonthOfYear`, `DayOfYear`) let the parser check each source → stages
→ `[period].cadence` chain.

```julia
# sources
source_grid(s) -> LatLonMesh | CubedSphereMesh | PointCloud     # PointCloud(lon, lat, area)
source_intervals(s) -> Vector{TimeInterval}                     # ascending; checked against the catalog's last_month
flux_substance(s) -> AbstractSubstance
read_rate!(dst::Vector{Float64}, s, k::Int, stats) -> dst       # interval k as kg substance s⁻¹ per native cell, finite
describe(s) -> Dict{String,Any}                                 # paths, size, mtime, units, decoders used

apply!(x, t::AbstractNativeTransform, grid, iv, stats) -> x
prepare(op::AbstractSpatialOperator, grid, target, kind::QuantityKind) -> PreparedSpatial
remap!(dst, p::PreparedSpatial, src) -> dst                     # extensive: Σdst = Σsrc to 1e-13

input_cadence(st), output_cadence(st)
required_inputs(st, out::TimeInterval, available::Vector{TimeInterval}) -> UnitRange{Int}
needs_full_record(st)::Bool; accumulate!(acc, st, x, iv); finalize!(acc, st)
evaluate!(out, st, inputs, iv::TimeInterval, target) -> out
hourly_fractions!(f::Matrix{Float64}, shape::AbstractDiurnalShape, day::Date, target) -> f   # (ncell, 24); rows sum to 1

region_weights(r::AbstractRegionSet, target; cache_dir) -> RegionWeights   # (nregion × ncell); raster column sums = 1, boxes one-hot or 0
partition!(regional::Vector{Vector{Float32}}, x::Vector{Float32}, w) -> regional   # Float32(Float64(x)·w_k); residual by Float32 left fold

plan_surface_fluxes(cfg::AbstractDict) -> FluxBuildPlan         # parse + validate + resolve inputs (metadata only)
build_surface_fluxes(cfg::AbstractDict; members = nothing, skip_existing = false, check_only = false) -> Vector{FluxBuildResult}
run!(d::FluxDriver) -> FluxAudit
```

**Engine** (`engine.jl`) streams the output period in calendar-month blocks:
1. Resolve the target mesh and prepare the spatial operators. Weights go to the
   shared disk cache; the cache needs no lock, because two builders publish
   identical bytes atomically.
2. Load the auxiliary fields (land fraction, diurnal fractions) and remap them
   once, on the intensive path.
3. If any stage has `needs_full_record` (climatologies), stream the source once
   in a reduction pass.
4. Open the writer. For each output month, evaluate the components in dependency
   order (`DerivedSource` and derived stages; a cycle is a config error). A
   rolling cache holds the native intervals needed; `MidMonthInterpolation`
   needs months m−1, m and m+1.
   - **Span-independence rule:** neighbouring source months are always read when
     the product has them. Flat extrapolation happens only at the product's real
     first and last months, never at the edges of `[period]`.
5. Form the outputs: apply `Combination`/`Float32Sum`, the species factor, divide
   by `mesh.cell_areas`, round to Float32, then `partition!` if configured.
   Write one output day per `write_slices!` call. Per-cell loops use
   `Threads.@threads`; NetCDF I/O stays on the main task; all sums are Float64
   in a fixed order.
6. `publish!`. Readback: every slice through `read_cs_native_flux`;
   Σ density × `mesh.cell_areas` must equal the audit total to 1e-6 relative.
7. Write the audit JSON atomically, then rebuild `manifest.json` from the audit
   files under a `FileWatching.Pidfile` lock.

**Audit** (`audit.jl`): one schema replaces the five Python ones. It records
status, member, config SHA-1 and `source_revision()`; `describe` of every
input, with the units and decoders used; per source interval, the native and
remapped totals and their relative error; `ClipNegative` statistics; the
largest month-renormalisation error; per output, totals per month and year
(Pg C and Pg species, using a named `KG_PER_PG = 1e12`); and
`carbon_to_co2_factor`. It holds no storage-basis numbers: the carrier
drawdown is a property of the run and moves to `Analysis.carrier_drawdown`.

### 2.6 How the products are composed

| Product | Source → transforms | Spatial | Temporal stages | Outputs |
|---|---|---|---|---|
| TRENDY daily (2) | catalog `LatLonNetCDFSource` → `ClipNegative`, `LandFractionWeight` (LPJ-GUESS `rh` adds `AnnualAsMonthly`; new, no multi-year Python reference) | `ConservativeRemap` | `MidMonthInterpolation(NonNegative())` per component | GPP = −gpp; RA; RH; TER = `Float32Sum(RA, RH)`; NEE = `Float32Sum(GPP, TER)` |
| NPP/Rh (3) | as above | same | same | NPP = `Combination(ra = 1, gpp = −1)`; RH |
| 2021 hourly (1) | as above; `ter = DerivedSource(ra + rh)` | same | `MidMonthInterpolation(NonNegative())` → `DiurnalDisaggregation(FileDiurnalFractions(…; fallback = CosineZenith(MidMonthDay())))` for GPP; flat for TER | GPP, TER |
| GFED5 (4) | `SourceSequence` (finalized `C`; NRT `EM` + `SumOverDimension(:lct)`), units `g C month-1 cell-1` | `ConservativeRemap` (extensive) | `MidMonthInterpolation(NonNegative())` | FIRE |
| NEE climatology (5) | `CubedSphereNetCDFSource` | `NoRemap` | `MonthlyClimatology(2015:2024)` → `RepeatAnnualCycle` → `MidMonthInterpolation(SignedParts())` | NEE_CLIM |
| FLUXCOM-X (6) | daily GPP (clip), `DerivedSource(gpp + nee; clip)`, `LandFractionWeight(… MaskedAsZero)` | `ConservativeRemap` | GPP: `DiurnalDisaggregation(CycleDiurnalShape(:gpp_cycle; fallback = CosineZenith(ActualDay())))`; TER flat | GPP, TER |
| Hourly → daily (7) | `CubedSphereNetCDFSource` | `NoRemap` | `DailyMeanOfHourly` (24 complete UTC hours required) | GPP (`expect_sign`) |
| ECCO (8) | `LLCCompactSource` (big-endian, `.meta` checked, `Depth > 0`, \|x\| < 1, −999 → 0), units `mmol m-2 s-1` CO2 | `PointBinning` onto a 0.5° `LatLonMesh` (radius `"earth"`) | `PeriodMean` | `co2_flux_ocean` = −ocean |
| SIF IAV (9) | `SIFCalibratedGPPSource` (harmonisation and month × band fits from the JSON; land fraction from the converted NetCDF; band edges from TOML) | `NoRemap` (nested or CR.jl for other Nc) | `gpp_clim = DayOfYearClimatology(2019:2025, CircularRunningMean(15))`; `gpp_mean = CellTimeMean(gpp_clim)` | FLUX; ANOM = `Float32Sum(FLUX, +Float32(gpp_clim))`; SEAS = `Combination(gpp_clim − gpp_mean)·(−1)`; RECO = `Combination(gpp_mean)`; NET = `Float32Sum(FLUX, RECO)` |
| TransCom C180 boxes (10) | CS source | `NoRemap` | `PassThrough` | partition: `LatLonBoxRegions`, `RegionVariablesInOneFile("CO2_FLUX_UNASSIGNED")` |
| TransCom C30 raster (11) | parent CS files (validation) or the TRENDY/GFED drivers directly (production) | `NoRemap` | `PassThrough` with a period subset | partition: `RasterRegions`, `FilePerRegion("tc{region:02d}/{member}.nc")` |
| Refine C90 → C180 (12) | CS source | `ConservativeRemap` → `NestedCubedSphereRegrid{Prolongation}` | `PassThrough`, period subset | CO2_FLUX |

Units: sources declare `FluxUnits` (catalog `units`, checked against the file's
`file_units`). `read_rate!` applies `mass_rate_factor` and multiplies by the
source cell areas when the basis is `PerSquareMetre`. Outputs name a species
(`[output] species = "co2"`). Carbon is converted by
`SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`, and the factor is written into
the file.

### 2.7 Config (TOML) and command line

- Configs live in `config/surface_fluxes/<product>.toml`, catalogs in
  `config/surface_fluxes/catalogs/`, region sets in
  `config/surface_fluxes/regions/`, with a `README.md`.
- Tables are typed by `kind` through the descriptor pattern of
  `Output/observations/observation_sources.jl`: a `_KINDS` NamedTuple, then
  `config_keys(T)` + `ConfigChecks.check_known_keys`, then `_parse(T, …)`.
- An editor schema `schemas/surface_flux_build.schema.json` is kept in agreement
  with the parser by a test.
- Radii are named (`"earth"` = `EARTH_RADIUS`, `"ifs"` = `IFS_EARTH_RADIUS`);
  `[target].binary` is preferred.
- `[period].end` is the inclusive last day, as `[input].end_date` in run configs.

```toml
# config/surface_fluxes/trendy_v14_s3_daily_gpp_ra_rh_c90_2014_2024.toml
[target]
binary = "/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour_v3/era5_n320_transport_20140901_float32.bin"

[period]
start = 2014-01-01
end = 2024-12-31
cadence = "daily"

[ensemble]
catalog = "config/surface_fluxes/catalogs/trendy_v14_s3.toml"
members = "all"                                   # or ["CLM", "JULES-ES", "ORCHIDEE"]

[components.gpp]
source = { kind = "catalog", component = "gpp" }
transforms = [{ kind = "clip_negative" }, { kind = "land_fraction" }]
spatial = { kind = "conservative" }
temporal = [{ kind = "mid_month_interpolation", positivity = "non_negative" }]
# [components.ra] and [components.rh] are the same with component = "ra" / "rh"

[outputs.GPP_CO2_FLUX]
kind = "combination"
terms = { gpp = -1.0 }
long_name = "gross primary production (uptake)"
expect_sign = "non_positive"
[outputs.RA_CO2_FLUX]
kind = "combination"
terms = { ra = 1.0 }
[outputs.RH_CO2_FLUX]
kind = "combination"
terms = { rh = 1.0 }
[outputs.TER_CO2_FLUX]
kind = "float32_sum"
items = ["RA_CO2_FLUX", "RH_CO2_FLUX"]           # TER == RA + RH bit for bit in the file
[outputs.NEE_CO2_FLUX]
kind = "float32_sum"
items = ["GPP_CO2_FLUX", "TER_CO2_FLUX"]

[output]
path = "/temp1/cfranken/fluxes_julia_v1/trendy_c90/TRENDYv14_S3_{member}_gpp_ra_rh_ter_nee_daily_co2flux_c{Nc}_2014_2024.nc"
species = "co2"
title = "TRENDYv14 S3 {member} daily C{Nc} CO2 fluxes"

[audit]
manifest = "/temp1/cfranken/fluxes_julia_v1/trendy_c90/manifest.json"

[staging]
dir = "/temp1/cfranken/trendy_staging"           # required when any input is compressed
```

```toml
# config/surface_fluxes/catalogs/trendy_v14_s3.toml (excerpt): data, not code
[catalog]
root = "/kiwi-data/Data/model/TRENDYv14/S3"
units = "kgC m-2 s-1"
file_units = "kg m-2 s-1"                        # every file must parse to this (carbon implied)
last_month = 2024-12                             # every decoded axis must end here
discovery = { kind = "suffix_search", directory = "{root}/{member}", exclude = ["mean_annual", "derived_c90_fluxes"] }
time = ["cf", "months_since", "decimal_years"]   # chosen by the units string of the raw axis, never by trial (P0)
excluded = { "ISBA-CTRIP" = "no usable S3 fluxes", "OCN" = "no usable S3 fluxes" }   # the only allowed exclusions
[members.CLM.land_fraction]
file = "CLM6.0_S3_oceanCoverFrac.nc"
variable = "oceanCoverFrac"
complement = true
[members.IBIS.land_fraction]
file = "IBIS_S3_oceanCoverFrac.nc"
variable = "oceanCoverFrac"
complement = true
fill = "fillvalue_only"
[members.CARDAMOM]
land_fraction = { variable = "land_fraction", in_component = "gpp" }
time = [{ kind = "auxiliary_variable", variable = "Time", origin = 2003-01-01, unit = "days" }]
[members."LPJ-GUESS".rh]
suffix = "arh"
temporal_prefix = [{ kind = "annual_as_monthly" }]
# P0: the arh axis is `days since 1700-01-01` (noleap, annual stamps on 1 July) and decodes with "cf"
```

Short snippets for other products:

```toml
# trendy_v14_s3_hourly_gpp_ter_c90_2021.toml (pilot semantics)
[components.ter]
source = { kind = "derived", terms = { ra = 1.0, rh = 1.0 } }   # summed before the time construction
[components.gpp]
temporal = [{ kind = "mid_month_interpolation", positivity = "non_negative" },
            { kind = "diurnal", shape = { kind = "file_fractions",
              file = "/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/sif_gpp_weekly_diurnal_c90_2021.nc",
              variable = "monthly_diurnal_fraction", fallback = { kind = "cosine_zenith", day = "mid_month" } } }]

# gfed5_fire_daily_c90_2014_2024.toml
[components.fire.source]
kind = "sequence"
parts = [
  { until = 2023-01-01, kind = "latlon_netcdf", files = "{gfed}/GFED5.1_finalized/monthly_full_1997_2022/GFED5.1_monthly_{YYYY}.nc",
    variable = "C", units = "gC month-1 cell-1", file_units = "g C per month" },
  { from = 2023-01-01, kind = "latlon_netcdf", files = "{gfed}/GFED5NRT_reprocessed/{YYYY}/GFED5NRTeco_CMB_{YYYY}-{MM}.nc",
    variable = "EM", sum_over = { dimension = "lct", size = 16 }, units = "gC month-1 cell-1" } ]

# trendy_gfed5_transcom11_c30_2014_2024.toml
[partition]
regions = "config/surface_fluxes/regions/transcom23_land.toml"   # raster, codes 1..11, nearest_labelled fallback
layout = "file_per_region"
path = "/temp1/cfranken/fluxes_julia_v1/transcom_c30/tc{region:02d}/{member}.nc"
```

**Command line.**
`julia --project=. scripts/preprocessing/build_surface_fluxes.jl <config.toml> [--member NAME ...] [--skip-existing] [--check]`
- A strict hand-written `_parse_cli` and the logger, as in
  `preprocess_transport_binary.jl`.
- `--check` runs `plan_surface_fluxes` and prints inputs, decoders, the cadence
  chain, output paths and missing files. It reads metadata only.
- `--skip-existing` reuses a member only when both the NetCDF and an audit with
  `status = "complete"` and the same config SHA-1 exist.
- Members run in parallel as separate processes, because netcdf-c is not
  thread-safe.

### 2.8 `Analysis`, diagnostic command lines and the campaign controller

```julia
daily_snapshot_series(paths, vars; origin = nothing) -> (; times::Vector{DateTime}, values::Dict)
    # drops duplicated boundary frames; origin from CF units (after B0) or explicit
budget_closure(run_cfg; output_glob, tolerance = BudgetTolerance(strict = 1e-4, operational = 1e-3)) -> BudgetReport
    # expected = Σ integrated_emission over the run (the runtime's own sources, areas, storage scale);
    # actual = Δ <tracer>_total_mass; carrier ULP term eps(Float32(carrier)) × air mass
carrier_drawdown(run_cfg) -> Dict{Symbol, Float64}   # max cumulative drawdown in VMR, using the binary's dry-air mass
EqualAreaBands(n); band_means(field, lat, area, bands)   # NaN cells excluded from numerator and denominator
doy_linear_detrend(x, dates)                              # Feb 29 folded onto Feb 28
seasonal_trend_decomposition(series, seen; min_months = 24, min_years_per_month = 5)   # minimum norm via pinv
skill_metrics(x, y) -> (; r, rmse, bias, amplitude_ratio, n)   # population std
weighted_fit_stats(x, y, w) -> (; r2, slope, bias, rmse, ref_mean, ref_std)
column_band_weights(m, edges); logp_interpolate(p, p_mid, q; outside = NaN)
GEOSChemCATRINEArchive(dir); gc_path(a, t::DateTime)   # flat or YYYY/MM; unreadable-file bookkeeping; serialised reads
```

`scripts/campaigns/run_campaign.jl <campaign.toml> --run-root <dir> --julia <path> --project <path>` (C1):
- reads a campaign TOML with the template run config, smoke span, an explicit
  `[[batch]]` list, budget tolerances and the expected flux manifest;
- builds configs as Dicts, runs `validate_config`, writes them with `TOML.print`;
- locks with a `FileWatching.Pidfile` and writes `status.json` (unchanged
  schema) atomically;
- runs one Julia child process per phase and fails early when
  `CUDA_VISIBLE_DEVICES` is unset;
- calls `Analysis.carrier_drawdown` in preflight and `Analysis.budget_closure`
  after the smoke and full runs.

---

## 3. Numerics and conventions: what changes relative to the Python outputs

| Item | Python | Julia port | Effect on outputs | Why |
|---|---|---|---|---|
| Lat-lon → CS remap | 0.25° samples assigned to the nearest C90 centre (whole 0.25° cells for GFED and FLUXCOM) | Exact spherical overlap (CR.jl) on the extensive path (equations below) | Global, monthly and annual totals unchanged (≤ 1e-13). Per-cell changes sit at coasts and source/target edges; patchy fire edge cells can move by tens of %. Regional totals ~1e-3. | Exact geometry, no sampling resolution. Σ qₜAₜ^mesh = Σ mₛ even though CR.jl's polygon areas differ from band areas by up to 1.27e-5. |
| Source faces | Voronoi midpoints, outer faces always ±90 | Same rule, except an outer face is ±90 only for pole-centred centres; the FATES southern face is −60.6316° (P0) | Identical for all full-latitude grids (~1e-12). FATES: identical, because its southern row is fill in all 3888 months (P0). | Physical geometry. The runtime loader is also fixed for pole-centred grids (CLM, ELM and, P0, iMAPLE), where today's uniform mesh misplaces rows by up to 0.47° (CLM) and 0.497° (iMAPLE, arithmetic). |
| Regular source grids | faces from Voronoi midpoints of the stored centres | snapping rule, faces from integers (Section 2.4) | Faces differ from Python's by 0 (GFED5) and ≤ 2.84e-14° (FLUXCOM-X) (P0) | No dependence on coordinate round-off or bounds variables |
| Target areas and coordinates | Float32-era geometry files | `mesh.cell_areas`, centres and corners of the binary's mesh (Float64 Girard) | Per-cell density up to 1.75e-4 (C90 polar panels) and 2.7e-5 (C30; P0 measured 2.668e-5); centres move by up to 2.3e-4° | The runtime multiplies by these areas |
| Radius | Literal 6.371e6 | From the binary header (6.371e6 for the current C90 and C30) | None today | `build_target_geometry`'s hard-coded `IFS_EARTH_RADIUS` is never used; flagged as a separate Preprocessing bug |
| C → CO2 | 44/12 in every file on disk | `SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`, recorded as `carbon_to_co2_factor` | kg CO2 × 0.999324844; carbon identical (P0: fresh Python outputs equal the old ones × this ratio within 1 Float32 ulp, except FLUXCOM-X, whose script converts in float32: its old file maps to the new one through Float32(1e-3·K_new)/Float32(1e-3·44/12) = 0.9993248757811442, within 2 ulp for GPP and 3 ulp for TER) | Commit 71a08395 |
| M_air, dry-air mass in budgets and drawdown | 0.02896546; literal 5.1353e18 | `DRY_AIR_MOLAR_MASS`; the binary's dry mass; the runtime storage scale through `integrated_emission` | −3.66e-5 on expected storage changes; drawdown ~1e-4 | One source of constants |
| Monthly → daily | `np.interp` at day centres, flat beyond the end midpoints, clip, per-month rescale | Same formula and order, streamed per month; flat only at the product's real ends | Bitwise or 1 ulp wherever the month list is the same | Span independence |
| 2021 TER | Interpolate the sum Ra + Rh | Same (`DerivedSource`) | None | One change at a time (Section 6) |
| Cosine-zenith fallback | Start of the UTC hour; 15th of the month (pilot), actual day (FLUXCOM); FLUXCOM in float32 | `CosineZenith{MidMonthDay \| ActualDay}`, start of hour, Float64; obliquity `EARTH_OBLIQUITY_DEG` | FLUXCOM ≤ 1–2 ulp; pilot bitwise | Parity; documented equation |
| SIF land fraction | npz from the FLUXCOM pilot | The same field, converted once to NetCDF (sha256 of the npz recorded) | None (P0: the file's `land_fraction` equals the npz bitwise) | The frozen calibration was fitted against it |
| SIF bands | File centre latitudes | Mesh centre latitudes | Cells within 2.3e-4° of ±20, ±40, ±60 or 0 may change band (counted) | The mesh is the runtime truth |
| Rounding | Mixed per script | Float64 throughout; `Combination` rounds once; `Float32Sum` reproduces every Float32 identity (TER, NEE, NET, ANOM) | Identities and SIF ANOM bitwise | One explicit rule per output |
| Time | `"hours since … UTC"`, calendar sometimes missing | Float64 `"hours since <period start> 00:00:00"`, `calendar = "proleptic_gregorian"`, `time_bnds`, stamps at interval start (00 UTC daily, HH:00 hourly) | Decoded times identical; refine's raw values are re-encoded from its own start | Matches the runtime `StepwiseFlux` and its numeric origin parser |
| Sign | Baked in, `positive = 'to_atmosphere'` | Magnitudes internally; output coefficients set the sign; attribute `sign_convention = "positive into the atmosphere"` | None | Positivity applies to magnitudes |
| Fill values | `_FillValue = -999` on most files | None; non-finite values are an error before writing | None | The runtime rejects missing values |
| Chunking | Library default `(1005,1,23,23)` or ad hoc | Daily: `(Nc, Nc, 1, 1)`, deflate 1, shuffle (P0 benchmark). Hourly: `(Nc, Nc, 1, 24)`, deflate 1, shuffle (P0 hourly benchmark; Section 2.4) | Performance only | The runtime reads `[:, :, p, kr]` per panel; one output day per write |
| Atomic writes | Mostly direct | `atomic_publish` for NetCDF, audit, manifest, status | No partial or 0600 files | Persistence rule |
| Input calendars | cftime, regex fallbacks, length guessing | Months keyed by `MonthKey`; the decoder chosen by the units string before any CFTime call (P0); month lengths in the Gregorian output calendar; checked against `last_month` | None where Python decoded correctly (P0: Python's month keys are contiguous and end in 2024-12 for all 23 models) | No silent guessing |

### 3.1 Equations

These equations are documented in `docs/src/preprocessing/surface_fluxes.md`
(P3).

**Extensive remap.** For a source cell s with rate density qₛ, land fraction
Lₛ and band area Aₛ^band, and CR.jl weights Wₜₛ with CR.jl source areas Aₛ^CR:

$$m_s = q_s\,L_s\,A_s^{\rm band}, \qquad m_t = \sum_s \frac{W_{ts}}{A_s^{\rm CR}}\, m_s, \qquad q_t = \frac{m_t}{A_t^{\rm mesh}}$$

so that $\sum_t q_t A_t^{\rm mesh} = \sum_s m_s$ whatever polygon areas CR.jl uses.

**Monthly → daily (`MidMonthInterpolation`).** Month j starts at day sⱼ, has nⱼ
days and the monthly mean rate qⱼ. Its midpoint and the day centres are

$$c_j = s_j + \frac{n_j}{2}, \qquad x_d = d + \frac12 .$$

Between two midpoints the rate is interpolated linearly,

$$\tilde q_d = q_j + (q_{j+1} - q_j)\,\frac{x_d - c_j}{c_{j+1} - c_j}, \qquad c_j \le x_d < c_{j+1},$$

and held flat beyond the first and last midpoints **of the product** (never of
`[period]`). Positivity: `NonNegative` sets $\tilde q_d \leftarrow \max(\tilde q_d, 0)$;
`SignedParts` constructs the positive and negative parts separately. Each
calendar month m is then rescaled to its monthly value Qₘ:

$$q_d = \tilde q_d\;\frac{Q_m}{\frac{1}{n_m}\sum_{d' \in m} \tilde q_{d'}} \qquad (q_d = 0 \text{ where the denominator is } 0),$$

so every calendar-month mean equals Qₘ.

**Diurnal disaggregation.** The hourly rate is

$$r_h = 24\, f_h\, r_{\rm day}, \qquad \sum_{h=0}^{23} f_h = 1 .$$

The cosine-zenith fallback is

$$f_h \propto \max\big(0,\ \sin\varphi\sin\delta + \cos\varphi\cos\delta\cos H_h\big),$$

$$\delta = \varepsilon\,\sin\!\Big(\frac{2\pi\,(284 + n)}{365}\Big), \qquad H_h = 15^\circ\,\big(h + \lambda/15^\circ - 12\big),$$

with ε = `EARTH_OBLIQUITY_DEG` (Cooper 1969), n the day of year (the 15th of the
month for `MidMonthDay`, the actual day for `ActualDay`), h the start of the UTC
hour and λ the longitude. Where all 24 weights are zero, fₕ = 1/24.

**Daily mean of hourly.** $r_{\rm day} = \frac{1}{24}\sum_{h=0}^{23} r_h$, from 24
complete UTC hours.

**Climatologies.** The monthly climatology is the equal-weight mean over the
years Y of the per-year monthly means, $\overline{Q}_m = \frac{1}{|Y|}\sum_{y \in Y} Q_{y,m}$.
The day-of-year climatology folds 29 February onto 28 February and smooths with
a circular running mean of 15 days, computed by cumulative-sum differencing.
For SIF IAV, `gpp_mean` is the per-cell time mean of `gpp_clim`.

**Output rounding.** `Combination` forms $y = \mathrm{fl}_{32}\big(\sum_i c_i x_i\big)$
in Float64 and rounds once. `Float32Sum` adds Float32 values left to right in
Float32, $y = \big(\mathrm{fl}_{32}(a_1) \oplus \mathrm{fl}_{32}(a_2)\big) \oplus \cdots$.
So TER = RA ⊕ RH, NEE = GPP ⊕ TER, NET = FLUX ⊕ RECO and ANOM = FLUX ⊕ fl₃₂(gpp_clim)
hold bit for bit in the files. **P0:** these identities must be checked as
Float32 sums, never by subtraction: in the SIF IAV file the Float32 difference
NET − FLUX − RECO is non-zero for 2,625,150 of 136,177,200 values, while
NET = FLUX ⊕ RECO holds everywhere.

**Partition.** $x_k = \mathrm{fl}_{32}\big(\mathrm{Float64}(x)\, w_k\big)$. Raster
weights: $\mathrm{raw}_{kc} = \sum_s W_{cs}\,\mathbf{1}(\mathrm{label}_s = k)$ and
$w_{kc} = \mathrm{raw}_{kc} / \sum_k \mathrm{raw}_{kc}$. Cells with no labelled
overlap take the nearest labelled pixel (great circle; ties → lowest index).

**Units.** The C → CO2 factor is K = M(CO2)/M(C) =
`SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS` (3.6641910962724906 at
`71a08395`, against 44/12 = 3.6666666666666665 before). Month lengths come
from the interval; a year is `DAYS_PER_JULIAN_YEAR` days where a unit is per
year.

---

## 4. Validation plan

### 4.1 Methods

| Level | What | Where |
|---|---|---|
| L0 | Synthetic unit tests (C4/C8, 10° lat-lon) in CI | `test/core`, `test/regridding` |
| S0 | Fresh Python references captured at `71a08395` (after the constant change, so no 44/12 confound) on reduced inputs, before any deletion, with a manifest of command, versions, sha256 and wall time. **P0:** reruns are compared by data, not by file hash: a GFED5 rerun gives identical data in every variable but a different sha256, because only the `history` attribute differs. The manifest's output hashes are a fixity check only. The commands name script paths at `71a08395`; once P4+ deletes the scripts, reproduce them from a worktree checked out at `71a08395`. | `/temp1/cfranken/flux_port_parity/python_ref_71a08395/<driver>/` (P0) |
| L1 | Algorithmic parity. The Julia library plus harness-only emulators plugged in by dispatch: `LegacyGeometryTarget` (file centres, two-triangle areas) and `LegacyNearestCentreRemap` (0.25° samples, nearest file centre among the 3×3 neighbours of the containing cell, Voronoi source columns). Target ≤ 1 Float32 ulp, ≥ 99.9 % bitwise. | `scripts/validation/flux_port_parity/` (emulators deleted in P11) |
| T | Temporal isolation: per-cell monthly means recovered from the Python daily file go through `MidMonthInterpolation` and are compared with that file; expect ≤ 2 Float32 ulp. Isolates leap years, flat ends, month boundaries and `SignedParts` from the remap. | harness |
| L2 | Production: exact remap and mesh areas. Totals per month and year, per-cell relative-difference quantiles (p50/p99/max; interior, coastal and polar-panel cells), 60 sin-latitude band totals, TransCom-11 region totals. **P0:** implemented except the TransCom-11 region totals (Section 5), with a reference scale for files written with 44/12, cell-by-cell comparison of (Xdim, Ydim, nf) fields such as `land_fraction`, and coastal masks for C90 and C30 (none exists at C180). | `scripts/validation/compare_cs_flux_files.jl` (kept long term); runner `scripts/validation/flux_port_parity/run_parity.jl` |
| R | Span independence: building `[2016-03-01, 2016-06-30]` equals those days of the 2014–2024 build bitwise | `test_surface_flux_temporal.jl` and real data |
| X | Runtime smoke: a 3-day `cs_native` run per live config; the logged source rate (`total_emission_rate`) equals Σ density × mesh area to 1e-6 | P4–P11 |

**S0 flat-end rule (P0).** The Python multi-year script accepts only whole
years, so the S0 TRENDY references span 2015–2016 and were built from the months
2014-12 … 2016-12. Their values are held flat after the 2016-12 midpoint, so
in December 2016 they differ from a 2014–2024 build (per-cell values by up to
many orders of magnitude); from January 2015 to November 2016 they equal the
existing 2014–2024 files × K_new/K_old within 1 ulp. L1 and T builds compared
against them must therefore be given the same month list (2014-12 … 2016-12),
or leave out the second half of December 2016.

The measured P0 report is `2026-10-10_python_flux_port_p0_facts.md`. Each later
port's measured report goes into a dated `docs/memos/…_python_flux_port_parity.md`
(facts only) and into its PR.

### 4.2 Flux drivers

| Driver | Reference | L1 / T | L2 metrics and expected agreement |
|---|---|---|---|
| NEE climatology | `/temp1/cfranken/sif_gpp_iav/fluxes/trendy_c30/JULES-ES_NEEclim_daily_co2flux_c30_2014_2024.nc`, built from the old JULES-ES C30 source | Run on the old source: bitwise or 1 ulp | Times 0 … 96408 identical; month means = climatology to 1e-12 on the Float64 values before storage. **P0:** the Float32 file can only show its rounding: every cell-month mean lies within its Float32 rounding bound of the climatology (at most 0.49 Float32 ulp, 2.8e-15 absolute), and global month totals agree to 6.2e-9. Compare kg CO2, which does not depend on K: −3.402094951918086e13 kg CO2 in every non-leap year (all identical) and −3.4010306993331016e13 in leap years. (The earlier "−9.285 Pg C" is this kg CO2 divided by K_new; the file was written with 44/12, and the carbon it was built from is −9.278440777958417 Pg C per non-leap year.) |
| Hourly → daily | `/kiwi-data/Data/satellite/FLUXCOM-X/X-BASE/2021/pilot/sif_gpp_daily_co2flux_c90_2021.nc` | Bitwise or 1 ulp | Times 0 … 8736; Σ daily·86400 = Σ hourly·3600 to 1e-7; max ≤ 0. P0: the S0 rerun is bitwise identical to this file. |
| Refine | `…/pilot/sif_gpp_hourly_co2flux_c180_dec2021.nc` | ≥ 99.99 % bitwise, ≤ 1 ulp | Decoded times 2021-12-01T00 … 12-31T23; per-hour Σρ·A₁₈₀ = Σρ·A₉₀ to 1e-12 (GMAO grids nest to 6.3e-13). P0: every C180 record is a 2×2 copy of C90 parent records 8016 … 8759, bitwise; per-hour Σρ·A₁₈₀ = Σρ·A₉₀ to 2.45e-14 (GMAO meshes, radius 6.371e6 m). |
| TRENDY multi-year | `/temp1/cfranken/sif_gpp_iav/fluxes/trendy_v2/`, `trendy_c30/`, `logs/prep_trendy_v2.log`; S0 C30 2015–2016 references for one member per source-grid type: CABLE-POP (1°), CLM (pole-centred), CLM-FATES (partial), JSBACH (Gaussian), ORCHIDEE (0.5°), CARDAMOM (auxiliary time), IBIS (fill) | L1 ≤ 1 ulp; T ≤ 2 ulp (same month list as S0) | Audit (Float64) annual Pg C ≤ 1e-9 per member and year (e.g. CLM 2021: 148.06 / 136.65 / −11.41); file-derived ≤ 1e-7. Renormalisation ≤ 1e-14. TER and NEE identities bitwise. kg CO2 vs on-disk × 0.999324844. Zonal bands ~1e-3. Flux in the FATES southern row reported. **P0:** no LPJ-GUESS multi-year reference exists (the Python script fails on `arh`); LPJ-GUESS parity uses the NPP/Rh S0 run or the 2021 hourly campaign. Use `trendy_v2/` for the daily construction: the 2018–2024 `trendy/` build has different TER and NEE daily values on every day of 2018–2024 while its month totals agree to ≤ 3.3e-10. |
| NPP/Rh | `/temp1/cfranken/trendy_allmodels_c30_20260908/fluxes/` (23 files, JSON, manifest); S0 run for LPJ-GUESS and CLM | L1 ≤ 1 ulp | `annual_PgC` ≤ 1e-9; remap ≤ 1e-13; daily-monthly ≤ 1e-14; manifest keys identical; RH ≥ 0; RH bitwise equal to the multi-year C30 RH (shared code); NPP = RA + GPP of the multi-year build within 1.6e-7 of the peak |
| 2021 hourly | `/kiwi-data/Data/groupMembers/cfranken/AtmosTransport/fluxes/TRENDYv14/S3/C90/2021/`; S0 run for CLM | L1 (CLM) ≤ 1 ulp | GPP 148.05793351340287 and TER 136.64600885387657 Pg C ≤ 1e-9; 8760 records; TER constant within each UTC day; 24-h mean GPP = daily value; daily GPP = multi-year 2021 GPP within 1.7e-7 |
| GFED5 | `/temp1/cfranken/sif_gpp_iav/fluxes/gfed5/GFED5_fire_daily_co2flux_{c90,c30}_2014_2024.nc`, logs; S0 C30 2014 | L1 ≤ 1 ulp | Native monthly totals ≤ 1e-12 (same grams); annual 2014–2024 (3.263174 … 4.028272 Pg C, P0 comparator) ≤ 1e-9; month means = remapped monthly mass to 1e-12; regional ~1e-3; per-cell statistics; X with `gfed5_fire_c30_2014_2024` |
| FLUXCOM-X | `…/2021/derived/fluxcom_x_gpp_diurnal_ter_daily_hourly_co2flux_c90_2021.nc` + JSON. **P0:** this file predates `71a08395`, whose change altered FLUXCOM-X output values (`KG_CO2_PER_G_C`). S0 reference: `python_ref_71a08395/fluxcom_x/` (full 2021; JSON identical apart from `output`). The script converts in float32 (NumPy 1.x keeps a float32 array times a Python float in float32): GPP = −(g·f)·K₃₂/3600 and TER = d·K₃₂/86400, every step rounded to Float32, K₃₂ = Float32(1e-3·K). The old file maps to S0 through K₃₂ new/old = 0.9993248757811442 (2 ulp GPP, 3 ulp TER, year totals ≤ 1.6e-11), not through K_new/K_old (3 ulp, year totals 3.137e-8). | L1 bitwise or 1 ulp only if the emulator reproduces that float32 order; Float64 conversion moves values by up to about 3 ulp | Negative fractions exact (0.0913147913102717, 0.01563289665267889, 0.2328722232422215); annual GPP 128.738 and TER 123.609 Pg C ≤ 1e-7; Σ₂₄ₕ GPP·3600 = −daily·K₃₂ (**P0:** holds in aggregate with K₃₂, 2.4e-10 for GPP and 1.8e-11 for TER; −2.6e-8 with the Float64 K; TER = Float32(d·K₃₂)/86400 bitwise in every cell-day); signs; X with the `_smoke` config |
| SIF IAV | `/temp1/cfranken/sif_gpp_iav/fluxes/sif_gpp_daily_co2flux_c90_2018_2025_v3.nc`; S0 full run `python_ref_71a08395/sif_iav/` (P0, all 2802 days) | L1 (file centres): all five variables bitwise | vs v3: flux = v3 × 0.999324844 within 1–2 ulp on cells with unchanged band (band switches counted). **P0 (S0 vs v3 × K_new/K_old):** GPP_CO2_FLUX and RECO_FLAT_CO2_FLUX within 1 ulp for every value, year totals ≤ 4.1e-10. ANOM, NET and SEAS are differences of near-equal values: per-value ulps reach 687,854,529 (ANOM), NET slice totals 7.4e-4 and SEAS year totals 1.12 (zero annual mean). Gate them in Float32 ulps of the largest operand (measured ≤ 2.58 ANOM, 2.64 NET, 1.30 SEAS) or on month totals, not per value. In v3: anomaly record total −1.92701 Pg C (the S0 log prints −1.927); sphere closure of the file `cell_area` 4.1756e-11. NET = FLUX ⊕ RECO as a Float32 sum, bitwise; `annual_gpp_PgC` to 4 decimals (sum 995.778 Pg C); 16,343 cells with land fraction > 0.05; RECO 128.96182257359436 Pg C per 365.25-day year, as the script's `reco_pg` computes it (128.87355301673358 per 365 days); `land_fraction` bitwise. |
| ECCO-Darwin | `~/data/AtmosTransport/catrine/Emissions/ECCO_Darwin/eccodarwin_ocean_co2_december_05deg.nc` (+ per-day file). P0: still valid as a reference (only printed diagnostics changed at `71a08395`). | Bitwise or 1 ulp (same Float64 add order). **P0:** a Julia re-evaluation of the script's formulas with constants from `Parameters` reproduces both files bitwise, so bitwise is the target. | −4.1093e5 kg CO2/s; 168,566 non-zero cells; per-day mean = monthly within 2e-15 absolute (**P0:** max 1.52e-15, from Float32 storage of each day); X with `_smoke_oceanflux_dec1-2.toml` |
| TransCom C180 boxes | `…/pilot/sif_gpp_hourly_co2flux_c180_dec2021_transcom_tags.nc` | With the geometry-file centres: region ids and all 12 variables bitwise | With mesh centres: ≤ 8 edge cells differ; Σ regions + UNASSIGNED == flux bitwise every hour; residual ≈ 0.9 %. **P0:** in the existing file Σ regions + UNASSIGNED == flux as a Float32 sum in 744/744 hours, no cell has two non-zero regions, UNASSIGNED share 0.877 % (hourly 0.025–3.06 %). Region ids not yet checked (needs the P7 partition). |
| TransCom C30 raster | `/temp1/cfranken/trendy_transcom_c30_20260908/transcom_weights_c30.nc`, `partition_validation.json`, `fluxes/tc01/` | Nearest-centre emulation: weights bitwise, fallback identical, distance ≤ 1e-9 | Exact overlap: max\|Δw\|, per-region area totals (~1e-3); Σ_k w = 1 ≤ 5e-16; closure < 2e-7; run on the old parent files so that only the weights differ |

### 4.3 Diagnostics and controller

| Port | Reference | Expected |
|---|---|---|
| Budget closure (`check_trendy`) | `/temp1/cfranken/trendy_allmodels_c30_20260908/{validation,smoke_validation}.json` | With the legacy constants injected by the harness: expected and actual ≤ 1e-12; TEM NPP fails at 1.2714e-4; 45/46 pass; smoke max 1.96e-4. The runtime-consistent result is reported next to it. |
| GEOS-Chem mass balance | `/temp1/cfranken/catrine_protocol/true_mass_balance/geoschem.csv` + `.log` | Hours identical; burdens and sources ≤ 1e-12; residuals ~1e-13 × burden; the four summary lines to every printed digit |
| Float32 attribution | `~/data/AtmosTransport/output/precision_attribution/out` | Every printed digit (e.g. full_final F32−F64 −2.322e-09, −2.003e-08, −3.747e-09, −1.805e-08); 16 configs pass `validate_config` |
| F32 vs F64 | `compare_f32_f64_dec2021/default/float32_vs_float64.json`, then the 761-day JSON | 16 numbers ≤ 1e-12 |
| CATRINE vs GEOS-Chem | `/temp1/cfranken/catrine_protocol/compare_c90_full` (Dec 2021 first) | Monthly and time series ≤ 1e-10; month-0 fields ≤ 1 ulp; corner offset exactly 0.00054931640625 |
| SIF transport control | Memo values (0.644/0.244, ratio 0.38; 0.627/0.012; r = −0.002) + a fresh Python run | ≤ 1e-10; band arrays element by element |
| TRENDY ranking | `/temp1/cfranken/trendy_allmodels_c30_20260908/intercomparison/` | Monthly fields ≤ 1e-13; arrays and CSV ≤ 1e-10; identical ranks; CABLE-POP r = 0.76797 |
| SIF IAV analysis | `/temp1/cfranken/sif_gpp_iav/sif_gpp_iav_c90_xco2_timeseries.nc` | NetCDF bitwise; tables equal a fresh Python run; the exact closure beats the 0.25 % / 0.17 % residuals |
| Controller | `…/20260908/{run,smoke}.toml`; `trendy_transcom_c30_20260908/batches/tc01/*.toml` | Parsed Dicts equal apart from intended keys (`species`); phase sequence with fake child commands |

### 4.4 New unit tests

`test/core/*` is discovered automatically; `test/regridding/*` is added to
`test/regridding/runtests.jl`. Only `test/Project.toml` dependencies are used.
Fixtures are written into `mktempdir()`.

| File | Asserts |
|---|---|
| `test/core/test_atomic_files.jl` | A failure mid-write leaves neither target nor temp file; mode = 0o666 & ~umask; the regridder cache file is not 0600; the streaming writer is unchanged |
| `test/core/test_latlon_explicit_faces.jl` | Pole-centred faces (CLM-like, iMAPLE-like); FATES face −60.6316°; first lon faces −180.5 / −0.625 / −1.25 / −180.625 / −180.9375; the wrapped CLM-FATES sequence; descending latitude; Gaussian latitude, with `dy` throwing; area sum 4πR² (full grids); uniform grids reproduce today's mesh bit for bit |
| `test/regridding/test_quantity_regrid.jl` | Extensive Σ to 1e-13; `regrid_2d_to_cs_panels!` bit-identical before and after the move |
| `test/regridding/test_nested_cs_regrid.jl` | C4 ↔ C8 restriction and prolongation conserve to 1e-13; equal CR.jl to 1e-12; `build_regridder` selects the nested type. Merged coarsening kernels (P0): bitwise equal to Set A for Float32 and Float64; bitwise equal to Set B for sums, face sums and Float64 means; Float32 means against Set B differ (measured ≤ 5 ulp on positive data) and are not asserted bitwise |
| `test/core/test_surface_flux_integrals.jl` | `integrated_emission` = CPU stepped application for every scheme (1e-14); `total_emission_rate` replaces the log value |
| `test/core/test_gridded_io_coordinates.jl` | `find_coordinate` by name only, including a FLUXCOM-X-like `lon` labelled `degrees_north`/`latitude` and GFED5-NRT-like coordinates without attributes; the snapping rule (exact, round-off ≤ 1e-13, and a half-cell offset that must throw); faces never from `*_bnds`; orientation; `missing_value`-only fill (VISIT-like) and `FillValueOnly` vs `MaskedAsZero` |
| `test/core/test_gridded_io_time_axes.jl` | noleap, 365_day and 360_day; `months since` never routed through CFTime (a CABLE-POP-like 365_day axis and an iMAPLE-like Int16 axis with no calendar); decimal years with round-to-nearest and negative residuals (ELM-FATES-like); auxiliary origin; `" UTC"` suffix; `last_month` mismatch error |
| `test/core/test_flux_units.jl` | Every unit string in use parses, including `kg  m-2 s-1` and `kg m-2 s-1, per unit land area occupied by the PFT`; factors equal `PhysicalConstants` expressions; molar units without a species rejected |
| `test/core/test_cs_native_flux_schema.jl` | Writer → `read_cs_native_flux` round trip; no fill attribute; chunks `(Nc, Nc, 1, 1)` for daily files; area gate (1.75e-4 single cell → warning, 2e-3 → error, `semi_major_axis` mismatch → error) |
| `test/core/test_surface_flux_temporal.jl` | `MidMonthInterpolation` = a loop reference of the `np.interp` algorithm, bitwise in Float64; month means to 1e-14 including leap years; flat ends only at product ends (R); `SignedParts`; `DailyMeanOfHourly`; equal-year climatology; Feb-29 fold; cumulative-sum smoothing = direct window sum |
| `test/core/test_surface_flux_diurnal.jl` | Fractions sum to 1; zero at night; 1/24 in polar night; daily integral preserved; file fallback cells |
| `test/core/test_surface_flux_spatial.jl` | Extensive remap conservation; `PointBinning` on CS and lat-lon in input order; `DerivedSource` clip; `LandFractionWeight` |
| `test/core/test_surface_flux_regions.jl` | The three Python cases (coast 0.75/0.25, fallback, closure < 2e-7 with signs kept, negative weights rejected); C4 raster column sums to 5e-16; box priority, half-open intervals, longitude wrap; exact residual |
| `test/core/test_surface_flux_driver.jl` | Synthetic monthly lat-lon NetCDF + TOML → build → `Models.build_surface_flux_source` on the file gives the analytic kg/s per slice (1e-6); `Float32Sum` identities checked as sums; audit and manifest; `--skip-existing` honours the config hash; unknown keys and cadence-chain errors raised at parse time |
| `test/core/test_surface_flux_build_schema.jl` | `schemas/surface_flux_build.schema.json` agrees with the parser key tables |
| `test/core/test_llc_compact_source.jl`, `test_sif_calibrated_gpp.jl` | Synthetic big-endian compact + `.meta`; band and month fits; Feb-29 slot |
| `test/core/test_analysis_decomposition.jl`, `test_analysis_bands_stats.jl` | The four Hovmöller cases (polar gap through `pinv`, carrier invariance, linearity, metrics); band means; weighted stats |
| `test/core/test_campaign_controller.jl` | Smoke derivation; status transitions; lock; failure path, using `true`/`false` child commands |
| `test/core/test_compare_cs_flux_files.jl` and `test_flux_port_parity_harness.jl` (both written in P0), `test_snapshot_time_units.jl` | Comparator script (precedent `test_ll_to_cs_regrid_script.jl`); parity runner `run_parity.jl`; CF snapshot time (B0) |
| `test/real_data/test_surface_flux_reference_outputs.jl` | Opt-in `--real-data`; skips without data. NEEclim bitwise, one refine day, one hourly→daily day, GFED5 C30 2014 totals, against `test/real_data/surface_flux_reference_numbers.toml` (written in P0) |

Gates that stay green on every commit:
- the golden harness (`test/golden/run_goldens.jl`): bit-identical through
  P1–P2, except the pole-centred-faces commit, which needs owner sign-off if any
  golden moves (P0: neither coarsening path is reached by any golden case, so
  the kernel merge moves no golden);
- `test_native_cs_surface_flux.jl`, `test_timevarying_surface_flux.jl`,
  `test_persistence_hardening.jl`, `test_config_schema_agreement.jl`,
  `test_public_api_surface.jl`, `test_public_docstrings.jl`, Aqua, JET;
- `ATMOSTR_DOCS_BUILD_ONLY=true julia docs/build.jl`.

---

## 5. Phasing

Every step starts with a probe (shape, dtype, one physical number) and ends
with tests green, docs updated, `codex review --uncommitted` and one commit per
bullet. It deletes its Python files in the same commit and adds them to the
"Retired scripts" table. Effort is in developer-days.

**Prerequisite (not counted):** the constants refactor (`refactor/constants`,
72 uncommitted files at design time) is committed or merged before P1.

| Step | Content | Python removed | Effort | Depends on |
|---|---|---|---|---|
| P0 | **Done:** probes of the daily and hourly chunk layouts, all TRENDY time axes and grids (including SDGVM, VISIT and iMAPLE), GFED5/FLUXCOM-X coordinate snapping, the FATES southern row, and the equivalence of the two coarsening kernel sets; S0 capture (16 runs, including full FLUXCOM-X 2021 and SIF IAV 2018–2025); reference-numbers TOML with per-member sources and the §4.2 items verified; `compare_cs_flux_files.jl` (reference scale, (Xdim, Ydim, nf) fields, C90 and C30 coastal masks) and its test; the harness skeleton `scripts/validation/flux_port_parity/` (`run_parity.jl`, README, S0 example) and its test; this memo and the facts memo. **Deferred, with the step that needs it:** TransCom-11 region totals and integer-stored fields such as region ids in the comparator (P7, when the regions exist); TransCom C180 region-id check (P7 partition); the X-level runner kind (P4, first smoke); a per-operand ulp gate for SIF ANOM/NET/SEAS (P9); FLUXCOM-X per-cell 24-hour GPP identity (P8, on the port's own Float32 intermediate); NEEclim month means to 1e-12 (P4, on Float64 values before storage). No C180 coastal mask exists. | — | 1.5 | — |
| P1 | (a) `AtomicFiles` + migration of 4 writers (fixes the 0600 `save_regridder` files). (b) `LatLonMesh(λᶠ, φᶠ)` with the P0 face cases. (c) Regridding: quantity kinds, `cs_nesting.jl` (kernels moved and merged on Set A semantics; owner decision on the Float32 GEOS change, Section 6), `default_regrid_cache_dir`. (d) `Output.define_cs_horizontal_geometry!`, `define_provenance_attributes!`. (e) `Operators.SurfaceFlux` integrals (DrivenRunner and both scripts switched). `src/Regridding/README.md`, `docs/src/api/*`. | — | 2.5 | P0 |
| P2 | (a) `GriddedIO` + runtime and IC migration as pure moves, plus the noleap/units grammar and the units-first time classification; goldens bit-identical; `src/GriddedIO/README.md`, `src/Models/initial_conditions/README.md`, architecture docs, `CLAUDE.md`. (b) Pole-centred faces in the runtime loader (CLM, ELM, iMAPLE; own commit). (c) `surface_flux.species` replaces `molar_mass_kg_mol` outright; 236 config occurrences migrated; run schema; `toml_schema.md`. | — | 3 | P1 |
| P3 | `SurfaceFluxes` core: types, config parsing, engine (two passes), writer, readback, audit and manifest, CLI, `MidMonthInterpolation`/`PassThrough`/`DailyMeanOfHourly`, lat-lon and CS NetCDF sources, conservative, nested and no-remap. `docs/src/preprocessing/surface_fluxes.md` (equations), `docs/src/config/surface_flux_build.md`, `docs/make.jl`, schema, top-level export + `public_api.md`. `config/emissions/` removed. | — | 4 | P2 |
| P4 | CS-file drivers: refine, hourly → daily, NEEclim; SIF quickstart updated | `refine_cs_surface_flux.py`, `hourly_c90_flux_to_daily.py`, `prepare_climatological_nee_flux.py` | 1.5 | P3 |
| P5 | GFED5 (`SourceSequence`, `SumOverDimension`, per-cell units) | `prepare_gfed5_fire_flux.py` | 1 | P3 |
| P6 | TRENDY catalog, `stage_input`, decoder chain, clip statistics, `AnnualAsMonthly`; daily multi-year and NPP/Rh ensembles; L1 emulators run here; rerun-guide commands updated. **After this step all 23 C90 drivers can be built in Julia.** | `prepare_trendy_s3_multiyear_c90.py`, `prepare_trendy_s3_npp_rh.py` | 4 | P3 |
| C1 | `scripts/campaigns/run_campaign.jl` + `config/campaigns/trendy_allmodels_npp_rh_c30.toml` | `run_trendy_npp_rh_campaign.py` | 2 | B2 |
| P7 | Regions (raster and boxes), partition layouts, weights file, partition audit; TransCom C30 and C180 configs; `config/campaigns/trendy_transcom_c30.toml` | `prepare_transcom_c30.py`, `run_transcom_campaign.py`, `make_transcom_sif_flux_tags.py`, `test/diagnostics/test_transcom_partition.py` | 2 | P3, C1 |
| P8 | Diurnal shapes, `EARTH_OBLIQUITY_DEG`, FLUXCOM-X, 2021 hourly | `prepare_fluxcom_x_gpp_ter_c90.py`, `prepare_trendy_s3_gpp_ter_c90.py` (last importer gone after P5–P7) | 2 | P6, P7 |
| P9 | `SIFCalibratedGPPSource`, day-of-year climatology, `CellTimeMean`, one-time npz → NetCDF conversion; `LLCCompactSource`, `PointBinning`, lat-lon writer; ECCO configs | `build_sif_gpp_multiyear_c90_flux.py`, `eccodarwin_co2flux_to_latlon.py` | 2.5 | P3 |
| P10 | CATRINE protocol as a config (`ZeroBefore`, `RepeatAnnualCycle`, `ScaleToAnnualTotals`) | `catrine_protocol_cs_fluxes.jl` | 1 | P3 |
| P11 | Production switch (owner-decided): regenerate live drivers (members in parallel) into new versioned paths; the owner repoints `config/runs/*` and their headers one campaign at a time; the runtime area gate becomes element-wise; X smokes; harness emulators deleted (comparator kept); sweep of `scripts/README.md`, the rerun guide, the SIF quickstart, the IAV memo, ATBDs and CHANGELOG | — | 2 (+ data wall clock) | P4–P10 |
| B0 | Snapshot writer emits CF `hours since <start>`; in-repo readers switched; `output_schema.md` | — | 0.5 | — |
| B1 | `Analysis` foundations; Hovmöller test port | `catrine_station_timeseries.py` → heritage | 2 | P1 |
| B2 | Budget closure CLI; GEOS-Chem mass balance (`GEOSCHEM_SPECIES_MOLAR_MASS`, `RN222_HALF_LIFE`); Float32 attribution (`docs/src/theory/float32_conservation.md`); F32/F64 split | `check_trendy_npp_rh_run.py`, `catrine_true_mass_balance_gc.py`, `float32_conservation_attribution.py`, `compare_float32_float64_runs.py` (→ plot script) | 3 | B1, P1 |
| B3 | SIF IAV analysis; SIF control and TRENDY ranking splits; OCO-2 band cache converted to NetCDF | `analyze_sif_gpp_iav_xco2.py`, `compare_sif_gpp_transport_control.py`, `rank_trendy_hovmoller.py` (→ plot scripts), `test/diagnostics/test_trendy_hovmoller.py` | 3 | B1 |
| B4 | CATRINE vs GEOS-Chem split (Distributed over times); page and animation rewired | `catrine_compare_vs_geoschem.py` (→ plot script) | 4 | B1 |
| R1 | **Owner-decided, after P6** (outside the port count): the runtime keeps each `cs_native` series on the host and uploads only the slices of the current met window to the device | — | about 3 | P6 |

**Totals.** Flux track (P0–P11) 27 d; diagnostics (B0–B4) 12.5 d; controller
(C1) 2 d; about 41.5 developer-days, plus R1 (about 3 d).

**Parallelism.** P0 → P1 → P2 → P3 run in series. After P3, P4, P5, P6, P9 and
P10 run in parallel; P7 waits for C1 and can validate on the old parent files;
P8 needs P6. The diagnostics track starts after P1; B2 needs the P1 integrals.

**Critical path for the 23-model C90 rerun drivers:** P0 → P1 → P2 → P3 → P6
(+ P5 in parallel), about 15 days. With three workers the whole port takes about
4 weeks.

**R1 in detail.** `TimeVaryingSurfaceFluxSource`'s `Adapt` hook today moves the
whole run-span series to the device: about 1.5 GB per tracer for the C90
2014–2024 TRENDY config, about 68 GB in Float32 for 93 C90 tracers. R1 keeps the
file format and removes that device footprint. File chunks `(Nc, Nc, 1, 1)`
make the per-window reads cheap (P0: about 0.8 ms per 6-panel daily read,
whether the file is reopened or kept open). Hourly `(Nc, Nc, 1, 24)` files
are not slower when kept open (P0: mean 0.66 ms per hourly slice for GPP
against 0.74 ms with `(Nc, Nc, 1, 1)`; median 0.14 ms, one 13 ms read on the
first slice of each chunk). RUNTIME's device-side monthly
reconstruction (2n + 2 slices) stays available as a later option, because the
temporal stages live in a module `Models` can import.

---

## 6. Defaults and open questions

### 6.1 Defaults

| # | Item | Default | Status |
|---|---|---|---|
| 1 | Per-cell values change in every regenerated driver (exact overlap, mesh areas, M(CO2)/M(C)) | Accept; new versioned paths, never overwrite; owner switches each live config in P11; finished campaigns keep legacy files | Owner decision, 2026-10-10 |
| 2 | Runtime cell-area gate | Element-wise: error above 1e-3 or on a `semi_major_axis` mismatch; warn above 1e-9. Legacy Float32-era files (1.75e-4) still run, with a warning; new files are exact | Design default |
| 3 | 2021 hourly TER semantics | Keep the pilot's sum-before-interpolation, so the finished 46-tracer run stays reproducible. Per-component interpolation (consistent with the multi-year drivers; up to 4.3 % per cell, monthly totals equal) is a one-line config change in a later, separate commit | Design default |
| 4 | Cosine-zenith fallback time | Keep the start of the UTC hour (pilot convention), documented; the hour centre would shift fallback cells by 30 minutes | Design default |
| 5 | C180 TransCom boxes | Keep the pilot boxes as a named region set; new work uses the TransCom raster | Design default |
| 6 | Compressed TRENDY inputs and npz files | `stage_input` shells out to `gzip`/`tar` (no new dependency). The SIF land fraction and the OCO-2 band cache are converted once to NetCDF with NPZ.jl in a throwaway environment, recording sha256; NPZ.jl never enters `Project.toml` | Design default |
| 7 | Output and config contracts | `species` replaces `molar_mass_kg_mol` outright (P2c); CF snapshot time units (B0) | `species`: owner decision, 2026-10-10. B0: design default |
| 8 | GPU memory for the 23-model C90 rerun | The port itself does not change it. R1 after P6. If the rerun cannot wait, run the Python multi-year script at HEAD without LPJ-GUESS into a scratch path and regenerate later | R1: owner decision, 2026-10-10 |
| 9 | Process risk | The constants refactor must land before P1 | Prerequisite |

### 6.2 Open questions for the owner

**1. CLASSIC longitude registration.** CLASSIC's longitudes are whole degrees
(−180 … 179) while its latitudes are half degrees (−89.5 … 89.5), and the file
carries `node_offset = 1`. A land-mask cross-correlation against the FLUXCOM-X
0.25° land indicator (|lat| < 60°) peaks at a −0.25° shift (r = 0.98298), then
−0.5° (0.98250), 0 (0.97829) and +0.5° (0.95736); the CABLE-POP control peaks
at 0, so the method is unbiased. The data fit best about 0.25–0.5° west of
"integer = centre". Python and this design treat the integers as centres
(first face −180.5°). *Proposed default:* keep the Python registration for
parity until the owner or the data provider decides.

**2. JSBACH Gaussian latitudes: midpoint faces or Gauss-weight faces.** JSBACH
latitudes are T63 Gaussian (equal to Gauss–Legendre to 1.4e-14°). Python's
Voronoi-midpoint faces differ from the Gauss-weight faces
(sin φ_face = −1 + Σw) by up to 0.065°, and the polar row is 5.8 % larger under
the midpoint rule. With weight faces the 2021 totals would change by
+4.42e-5 relative (GPP 140.82282 → 140.82904 Pg C). The midpoint numbers
reproduce the Python audit to 2e-16. *Proposed default:* keep the midpoint rule
for parity; switching is a later, separate decision.

**3. The Set A merge changes Float32 GEOS block-coarsened outputs.** Merging the
two coarsening kernel sets on Set A semantics (correctly rounded Float32 means)
changes the area-weighted intensive fields of Float32 GEOS-native
block-coarsened builds: `config/preprocessing/geosit_c90_dec2021_catrine_f32_fullL72.toml`
(C180 → C90), `geosit_c45_dec2021_catrine_f32_fullL72.toml` (C180 → C45) and
`geosfp_c720_native_to_cs180.toml` (C720 → C180); fields `pblh`, `ustar`,
`hflux`, `t2m`, vdiff `u/v/t/qv`, `cmfmc` and `dtrain`, before vertical
remapping. On positive fields the change is 1–5 Float32 ulp in about half the
cells. Signed fields (vdiff `u`, `v`, `hflux`) can move by more: on
near-cancelling signed test data, Set B was off by up to 6172 ulp from the
correctly rounded value. Extensive fields, Float64 builds and same-Nc builds
are unchanged, and no golden case reaches either coarsening path.
*Proposed default:* merge on Set A (it is correctly rounded in every tested
cell and accepts Float64 areas with Float32 data, which the flux path needs).

---

## 7. Repository facts the design relies on

Checked at design time (2026-10-10) on `refactor/constants` at `71a08395`:
- The streaming-binary 0600 fix is committed (`streaming_writer.jl:_open_streaming_staging`,
  `O_EXCL` with mode 0o666; PR #23 `fc873bfc`). `Regridding/weights_io.jl:save_regridder`
  still uses `mktemp`, so published regridder cache files are 0600.
- The CS coarsening kernels are duplicated in two Preprocessing files (Set A and
  Set B above).
- `Models` already does `using ..Preprocessing` (`Models.jl:19`,
  `InitialConditionIO.jl:61`), so a builder inside Preprocessing respects the
  load order. There is no `CSHelpers` submodule; the helpers sit flat in
  `cs_transport_helpers.jl`.
- `DrivenRunner._surface_source_total_rate` branches on `hasproperty`;
  `tracer_budget_closure.jl` and `catrine_true_mass_balance.jl` call the private
  `_flux_temporal_segments` and `_KNOWN_TRACER_MOLAR_MASS_KG_MOL`.
- `TimeVaryingSurfaceFluxSource`'s `Adapt` hook moves the whole run-span series
  to the device (header of `config/runs/trendy_v14_s3_gpp_ra_rh_gfed5_c90_conv_2014_2024.toml`:
  1.5 GB per tracer, about 42 GB for 13 tracers).
- `molar_mass_kg_mol = 0.0440095` appears 236 times in `config/runs/`.
- Snapshot time units are a bare `"hours"` (`Output/netcdf_schema.jl:106`).
- `config/emissions/*.toml` is read by nothing.
- `DAYS_PER_JULIAN_YEAR` exists; there is no obliquity constant; the Rn-222
  half-life is the literal `330350.4` (`Models/RuntimePhysicsRecipe.jl:398`).
- `test/core/*.jl` is discovered automatically; `test/regridding/` needs an
  `include` in its `runtests.jl`. The editor-schema precedent is
  `schemas/atmos_transport_run.schema.json` + `test/core/test_config_schema_agreement.jl`.
