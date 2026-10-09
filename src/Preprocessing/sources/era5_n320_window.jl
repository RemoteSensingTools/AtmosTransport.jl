# ERA5 N320 GRIB reader: per-window spectral and grid-point field synthesis on the N320 source mesh.
# Split from era5.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ===========================================================================
# Per-window field synthesis on the N320 source mesh.
#
# Each daily N320 `core` GRIB carries spectral T/VO/D/LNSP and reduced-Gaussian
# Q across 24 hours × 137 hybrid levels in a single file. For one window the
# reader produces:
#
#   - U, V on N320 cell centers (vod2uv on spectral coefficients, then
#     ring-by-ring synthesis via `spectral_to_reduced_scalar!`)
#   - T on N320 cell centers (direct spectral synthesis)
#   - PS on N320 cell centers (synthesise LNSP, exponentiate)
#   - Q on N320 cell centers (direct reduced_gg gridpoint reorder; the GRIB
#     stores rings north→south but the mesh stores them south→north)
#
# Layer-mass derivation and the convection reader live in
# era5_n320_mass_convection.jl; the regrid to a cubed-sphere target in
# era5_n320_to_cs.jl.
# ===========================================================================

const ERA5_NATIVE_LEVEL_COUNT = 137

"""
    discover_era5_n320_source_grid(core_path; FT=Float64) -> ReducedGaussianTargetGeometry

Build the N320 source-grid descriptor from the first `gridType=reduced_gg`
message in `core_path` (the `q` field is always present). The resulting mesh
ring order is south→north — the GRIB stores rings north→south and the
`read_era5_reduced_gaussian_*` helpers flip into the project convention.
"""
function discover_era5_n320_source_grid(core_path::AbstractString;
                                         FT::Type{<:AbstractFloat} = Float64)
    geom = read_era5_reduced_gaussian_geometry(core_path; FT = FT)
    mesh = ReducedGaussianMesh(geom.latitudes, geom.nlon_per_ring;
                                FT = FT, radius = FT(IFS_EARTH_RADIUS))
    lons_by_ring = [FT.(ring_longitudes(mesh, j)) for j in 1:nrings(mesh)]
    return ReducedGaussianTargetGeometry{FT, typeof(mesh)}(
        mesh,
        String(core_path),
        geom.gaussian_number,
        copy(geom.nlon_per_ring),
        copy(geom.latitudes),
        lons_by_ring,
    )
end

"""
    discover_era5_spectral_truncation(core_path) -> Int

Read the first spectral (`gridType=sh`) message in `core_path` and return its
triangular truncation `J = K = M`. ERA5 native model-level analyses are T639
in the current archive; the helper avoids hard-coding that value in case a
future archive convention shifts.
"""
function discover_era5_spectral_truncation(core_path::AbstractString)
    truncation = 0
    GribFile(core_path) do gf
        for msg in gf
            String(msg["gridType"]) == "sh" || continue
            truncation = Int(msg["J"])
            break
        end
    end
    truncation > 0 ||
        error("No spectral (gridType=sh) message found in $core_path")
    return truncation
end

# ---------------------------------------------------------------------------
# Workspace + output field structs.
#
# The workspace owns the level-indexed spectral cubes for one hour and the
# Legendre / FFT synthesis caches. The output fields struct owns the
# gridpoint result arrays. The two are decoupled so callers can keep the
# workspace alive across windows (cheap) and reuse a single output buffer
# across multiple consumers.
# ---------------------------------------------------------------------------

"""
    ERA5N320SpectralWorkspace{FT, G}

Per-window workspace for ERA5 N320 spectral synthesis. Owns:

  - the spectral coefficient cubes for T, VO, D, LNSP for the current hour
    (sized `(T+1) × (T+1) × Nz` in `ComplexF64`),
  - per-level scratch matrices `u_spec` / `v_spec` reused inside the
    synthesis loop,
  - a `ReducedSpectralThreadCache` with Legendre column buffer plus FFT and
    real ring buffers sized to every unique ring length in the source mesh,
  - a `read_buf` scratch used by `read_spectral_coeffs!` for the raw ecCodes
    `codes_get_double_array` payload,
  - completion bookkeeping (`have_t` / `have_vo` / `have_d` / `have_lnsp`).

The cubes dominate memory: at T = 639 and Nz = 137 each cube is ≈ 0.9 GB.
Workspaces are intended to be allocated once per day-handle and reused across
the 24 hours.

Spectral buffers (`vo_spec`, `d_spec`, `t_spec`, `lnsp_spec`, `u_spec`,
`v_spec`, `synth_cache.P_buf`) are unconditionally `ComplexF64` / `Float64`
regardless of `FT` — `read_spectral_coeffs!` and `vod2uv!` only operate at
that precision. `FT` only controls the eltype of the downstream gridpoint
fields written via [`read_era5_n320_window_fields!`](@ref).
"""
struct ERA5N320SpectralWorkspace{FT <: AbstractFloat,
                                  G <: ReducedGaussianTargetGeometry{FT},
                                  S <: ReducedSpectralSynthesis}
    source_grid  :: G
    T            :: Int
    Nz           :: Int
    # Spectral VO, D and T of every level. After a window is read, `vo_spec`
    # and `d_spec` hold U·cos φ and V·cos φ instead (`vod2uv!` in place).
    vo_spec      :: Array{ComplexF64, 3}
    d_spec       :: Array{ComplexF64, 3}
    t_spec       :: Array{ComplexF64, 3}
    lnsp_spec    :: Matrix{ComplexF64}
    # Batched synthesis of all levels (Legendre tables, FFT plans, buffers);
    # may be shared by workspaces whose reads never overlap.
    synthesis    :: S
    # Per-thread caches: `u_spec`/`v_spec` hold the `vod2uv!` output of one
    # level; the single-field path (`lnsp`) uses the Legendre/FFT buffers.
    # Length = `Threads.maxthreadid()` at allocation.
    synth_caches :: Vector{ReducedSpectralThreadCache}
    read_buf     :: Vector{Float64}
    have_t       :: BitVector
    have_vo      :: BitVector
    have_d       :: BitVector
    have_q       :: BitVector
    have_lnsp    :: Base.RefValue{Bool}
    lnsp_grid    :: Vector{Float64}
end

"""
    allocate_era5_n320_spectral_workspace(source_grid, T, Nz)

Allocate a fresh workspace sized to `source_grid` (build via
[`discover_era5_n320_source_grid`](@ref)), spectral truncation `T`
(from [`discover_era5_spectral_truncation`](@ref)) and vertical level count
`Nz` (defaults to 137 in callers, but the workspace itself stays generic).
"""
function allocate_era5_n320_spectral_workspace(source_grid::ReducedGaussianTargetGeometry{FT},
                                                T::Integer, Nz::Integer;
                                                synthesis = ReducedSpectralSynthesis(source_grid, T, Nz)) where FT
    T  >= 1 || throw(ArgumentError("T must be ≥ 1, got $T"))
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    T_int  = Int(T)
    Nz_int = Int(Nz)

    mesh = source_grid.mesh
    nc = T_int + 1

    buffer_lengths = sort!(unique(vcat(collect(mesh.nlon_per_ring),
                                       collect(mesh.boundary_counts))))
    # One cache + synthesis-scratch per thread so the level loop can run
    # `Threads.@threads` with no shared mutable state. Each cache holds its own
    # `u_spec`/`v_spec` (vod2uv output) and per-ring FFT/real buffers.
    # Size by `maxthreadid()`, NOT `nthreads()`: `julia -tN` also spins up an
    # interactive-pool thread, so `threadid()` inside the loop can exceed the
    # default-pool count `nthreads()` (it returned 17 with `-t16`). `nthreads()+1`
    # is the tight bound for the usual single interactive thread, but
    # `maxthreadid()` is robust to any interactive-pool size and the extra
    # caches (~18 MB each at T639) are negligible here.
    n_cells = ncells(mesh)
    n_caches = Threads.maxthreadid()
    synth_caches = [ReducedSpectralThreadCache(
        zeros(Float64, nc, nc),
        Dict(n => zeros(ComplexF64, n) for n in buffer_lengths),
        Dict(n => zeros(Float64, n)    for n in buffer_lengths),
        zeros(ComplexF64, nc, nc),
        zeros(ComplexF64, nc, nc),
    ) for _ in 1:n_caches]

    (synthesis.grid.mesh.nlon_per_ring == mesh.nlon_per_ring && synthesis.grid.lats == source_grid.lats &&
     synthesis.T == T_int && synthesis.Nf == Nz_int) ||
        throw(ArgumentError("synthesis was built for another grid, truncation or level count"))
    return ERA5N320SpectralWorkspace{FT, typeof(source_grid), typeof(synthesis)}(
        source_grid, T_int, Nz_int,
        zeros(ComplexF64, nc, nc, Nz_int),     # vo_spec
        zeros(ComplexF64, nc, nc, Nz_int),     # d_spec
        zeros(ComplexF64, nc, nc, Nz_int),     # t_spec
        zeros(ComplexF64, nc, nc),             # lnsp_spec
        synthesis,
        synth_caches,
        Float64[],                             # read_buf — grows in read_spectral_coeffs!
        falses(Nz_int), falses(Nz_int), falses(Nz_int), falses(Nz_int),
        Ref(false),
        zeros(Float64, n_cells),               # lnsp_grid — scratch for PS synthesis
    )
end

"""
    ERA5N320WindowFields{FT}

Per-window output container. `u` / `v` (m/s, geographic frame), `t` (K),
`qv` (kg/kg specific humidity) are `(n_cells, Nz)`; `ps` (Pa) is `(n_cells,)`.
Dry-basis layer mass derivation, regridding, and convection conversion are
downstream of this struct.
"""
struct ERA5N320WindowFields{FT <: AbstractFloat}
    u  :: Matrix{FT}
    v  :: Matrix{FT}
    t  :: Matrix{FT}
    qv :: Matrix{FT}
    ps :: Vector{FT}
end

function allocate_era5_n320_window_fields(source_grid::ReducedGaussianTargetGeometry{FT},
                                            Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    nc = ncells(source_grid.mesh)
    Nz_int = Int(Nz)
    return ERA5N320WindowFields{FT}(
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc),
    )
end

# ---------------------------------------------------------------------------
# reduced_gg → mesh ring reorder.
#
# ERA5 stores rings north→south (jScansPositively=0). The
# `read_era5_reduced_gaussian_*` helpers flip ring order so the mesh is
# south→north. Cells within each ring are already west→east; only the ring
# axis needs reversing. This helper is small enough to bake here without
# pulling in MetDrivers, and it asserts the per-ring count match so a
# miscounted `pl` array fails loudly instead of silently aliasing rings.
# ---------------------------------------------------------------------------

# `_reorder_grib_reduced_gg_to_mesh!` assumes ECMWF's layout: rings north to
# south, points west to east from 0°. The ring lengths are symmetric about the
# equator, so a mirrored file would otherwise pass its ring-count check.
function _check_grid_point_layout(msg)
    λ₀ = msg["longitudeOfFirstGridPointInDegrees"]
    (λ₀ == 0 && msg["iScansNegatively"] == 0 && msg["jScansPositively"] == 0) ||
        error("ERA5 reduced Gaussian field $(msg["shortName"]): first point at $(λ₀)°, " *
              "iScansNegatively=$(msg["iScansNegatively"]), jScansPositively=$(msg["jScansPositively"]); " *
              "expected 0°, 0, 0")
end

"""
    _reorder_grib_reduced_gg_to_mesh!(mesh_vals, native_vals, native_nlon, mesh) -> mesh_vals

Put grid-point values `native_vals` (GRIB native ring order, north→south)
on the mesh cells `mesh_vals` (`ReducedGaussianMesh` order, south→north).
Asserts the per-ring counts in `native_nlon` match `mesh.nlon_per_ring` (after
reversal); only the ring axis flips.

Within a ring the GRIB points sit at longitudes `(i − 1) Δλ` (first point
at 0°, checked by `_check_grid_point_layout`), while the mesh cells, the
regridder and the spectral synthesis use cell centres at `(i − ½) Δλ`. Each
cell therefore takes the mean of its two bounding grid points,
`(q_i + q_{i+1}) / 2` (cyclic): linear interpolation to the cell centre. Copying the points
directly would shift the field half a cell east — 0.14° at the equator, 10°
in the 18-point polar rings of N320.
"""
function _reorder_grib_reduced_gg_to_mesh!(mesh_vals::AbstractVector,
                                            native_vals::AbstractVector,
                                            native_nlon::AbstractVector{<:Integer},
                                            mesh::ReducedGaussianMesh)
    nrings_native = length(native_nlon)
    nrings_native == nrings(mesh) ||
        throw(DimensionMismatch("native_nlon has $(nrings_native) rings; mesh has $(nrings(mesh))"))
    length(mesh_vals) == length(native_vals) == sum(native_nlon) ||
        throw(DimensionMismatch("native_vals/mesh_vals/native_nlon length mismatch"))

    native_offset = 1
    @inbounds for j_native in 1:nrings_native
        j_mesh = nrings_native - j_native + 1
        n = Int(native_nlon[j_native])
        n == mesh.nlon_per_ring[j_mesh] ||
            throw(DimensionMismatch("native ring $j_native nlon=$n vs mesh ring $j_mesh nlon=$(mesh.nlon_per_ring[j_mesh])"))
        mesh_start = mesh.ring_offsets[j_mesh]
        ring = @view native_vals[native_offset:(native_offset + n - 1)]
        for i in 1:n                           # grid points (i-1)Δλ, iΔλ → cell centre (i-½)Δλ
            mesh_vals[mesh_start + i - 1] = (ring[i] + ring[i == n ? 1 : i + 1]) / 2
        end
        native_offset += n
    end
    return mesh_vals
end

# ---------------------------------------------------------------------------
# Window read + synthesis.
# ---------------------------------------------------------------------------

"""
    read_era5_n320_window_fields!(fields, workspace, handles, date, hour) -> fields

Fill `fields` with one window's worth of N320 source-grid fields for
`(date, hour)`. Performs one forward pass over `handles.core_path`, decoding
every message whose `(dataDate, dataTime)` matches into the workspace's
level-indexed spectral cubes (for `gridType=sh`) or directly into the
output `qv` array (for `gridType=reduced_gg`). After the pass the function
synthesizes spectral T per level, applies `vod2uv!` per level and synthesizes
U/V. PS comes from LNSP synthesis, or — when `settings.arco_surface_pressure`
is set — from bilinear interpolation of the ARCO single_level `sp` netCDF
(`_fill_ps_from_arco_sp!`). Errors loudly if any required level/field is
absent so a partial download is immediately visible.

The function does not allocate beyond `read_buf` resizing inside
`read_spectral_coeffs!`. Reusing `(fields, workspace)` across the 24 windows
of a day is the intended call pattern.
"""
function read_era5_n320_window_fields!(fields::ERA5N320WindowFields{FT},
                                        workspace::ERA5N320SpectralWorkspace{FT},
                                        handles::ERA5GRIBDayHandles,
                                        date::Date,
                                        hour::Integer) where FT
    0 <= hour <= 23 || throw(ArgumentError("hour must be in 0..23, got $hour"))

    fill!(workspace.have_t,  false)
    fill!(workspace.have_vo, false)
    fill!(workspace.have_d,  false)
    fill!(workspace.have_q,  false)
    workspace.have_lnsp[] = false

    mesh = workspace.source_grid.mesh
    T  = workspace.T
    Nz = workspace.Nz
    nc = ncells(mesh)
    length(fields.ps) == nc ||
        throw(DimensionMismatch("fields.ps length $(length(fields.ps)) != n_cells $nc"))
    size(fields.qv) == (nc, Nz) ||
        throw(DimensionMismatch("fields.qv size $(size(fields.qv)) != ($nc, $Nz)"))

    _prof = get(ENV, "ERA5_N320_PROFILE", "") == "1"
    _t_io = time()
    for msg in _core_messages(handles, date, hour)     # only this date and hour
        grid_type  = String(msg["gridType"])
        short_name = String(msg["shortName"])
        level      = Int(msg["level"])

        if grid_type == "sh"
            if short_name == "t"
                _read_into_level_slot!(workspace.t_spec, msg, workspace.read_buf, level, Nz)
                workspace.have_t[level] = true
            elseif short_name == "vo"
                _read_into_level_slot!(workspace.vo_spec, msg, workspace.read_buf, level, Nz)
                workspace.have_vo[level] = true
            elseif short_name == "d"
                _read_into_level_slot!(workspace.d_spec, msg, workspace.read_buf, level, Nz)
                workspace.have_d[level] = true
            elseif short_name == "lnsp"
                read_spectral_coeffs!(workspace.lnsp_spec, msg, workspace.read_buf)
                workspace.have_lnsp[] = true
            end
        elseif grid_type == "reduced_gg" && short_name == "q"
            1 <= level <= Nz ||
                error("Q level $level outside [1, $Nz] for date=$date hour=$hour")
            _check_grid_point_layout(msg)
            vals = msg["values"]
            pl   = msg["pl"]
            _reorder_grib_reduced_gg_to_mesh!(
                view(fields.qv, :, level), vals, pl, mesh)
            workspace.have_q[level] = true
        end
    end

    # Completeness gates — fail with the missing fields named so logs are
    # debuggable. Q has its own gate because `fields` is reused across
    # windows and a stale Q slice from the previous read would otherwise
    # silently corrupt dry-mass + the regridded Q output.
    (handles.settings.arco_surface_pressure || workspace.have_lnsp[]) ||
        error("ERA5 N320 read: LNSP missing for $(date) hour $(hour)")
    all(workspace.have_t) ||
        error("ERA5 N320 read: T missing for $(date) hour $(hour) at levels $(findall(!, workspace.have_t))")
    all(workspace.have_vo) ||
        error("ERA5 N320 read: VO missing for $(date) hour $(hour) at levels $(findall(!, workspace.have_vo))")
    all(workspace.have_d) ||
        error("ERA5 N320 read: D missing for $(date) hour $(hour) at levels $(findall(!, workspace.have_d))")
    all(workspace.have_q) ||
        error("ERA5 N320 read: Q missing for $(date) hour $(hour) at levels $(findall(!, workspace.have_q))")

    _prof && (_t_io = time() - _t_io)

    # Spectral → gridpoint synthesis. `vod2uv!` turns VO, D of each level into
    # ECMWF's pseudo-winds U·cos φ, V·cos φ (in place; levels are independent),
    # then each field is synthesised for all levels at once and the winds are
    # divided by cos φ per ring.
    grid = workspace.source_grid
    caches = workspace.synth_caches

    _t_synth = time()
    Threads.@threads :static for k in 1:Nz
        cache = caches[Threads.threadid()]   # threadid() ≤ maxthreadid() == length(caches)
        vo_lvl = view(workspace.vo_spec, :, :, k)
        d_lvl  = view(workspace.d_spec,  :, :, k)
        vod2uv!(cache.u_spec, cache.v_spec, vo_lvl, d_lvl, T)
        copyto!(vo_lvl, cache.u_spec)
        copyto!(d_lvl, cache.v_spec)
    end
    synthesize_reduced!(fields.u, workspace.vo_spec, workspace.synthesis)
    synthesize_reduced!(fields.v, workspace.d_spec,  workspace.synthesis)
    synthesize_reduced!(fields.t, workspace.t_spec,  workspace.synthesis)
    Threads.@threads :static for k in 1:Nz
        _divide_by_cos_lat_per_ring!(view(fields.u, :, k), mesh)
        _divide_by_cos_lat_per_ring!(view(fields.v, :, k), mesh)
    end

    # LNSP → PS = exp(LNSP). Single synthesis after the threaded loop; the
    # `@threads` barrier guarantees every level iteration has finished, so any
    # cache (here `caches[1]`) is free to reuse. The `lnsp_grid` buffer is Float64 already, so it
    # serves as both column and scratch (the in-method copy becomes a self-copy
    # of ~500 KB and is amortised across the synthesis kernel cost).
    if handles.settings.arco_surface_pressure
        # No spectral LNSP in ARCO core: interpolate the 0.25° ARCO `sp` onto
        # the N320 cell centers. The global-mean dry-mass pin downstream
        # (era5_n320_regrid.jl) absorbs any residual mean bias.
        _fill_ps_from_arco_sp!(fields.ps, workspace.source_grid,
                                handles.arco_sp_path::String, date, Int(hour))
    else
        _synthesize_into_column!(workspace.lnsp_grid, workspace.lnsp_spec, T,
                                  grid, caches[1], workspace.lnsp_grid)
        @inbounds for c in 1:nc
            fields.ps[c] = exp(workspace.lnsp_grid[c])
        end
    end

    if _prof
        @info @sprintf("      [prof] read_window: io+decode %.1fs  synthesis(%d lvls ×3) %.1fs",
                       _t_io, Nz, time() - _t_synth)
    end
    return fields
end

# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------

"""
    _fill_ps_from_arco_sp!(ps, source_grid, nc_path, date, hour) -> ps

Populate `ps` (Pa; N320 cell order south→north) from the ARCO single_level
surface-pressure netCDF at `nc_path` (`sp[longitude, latitude, time]`, 0.25°
regular lat-lon, longitude ascending 0→359.75, latitude descending 90→-90),
by bilinear interpolation to the reduced-Gaussian cell centers of `source_grid`.
Used when the ARCO `core` GRIB omits spectral `lnsp`. The slice's decoded
time coordinate must equal `DateTime(date) + Hour(hour)` — a shifted or
mis-assembled time axis fails loudly instead of silently offsetting PS.
"""
function _fill_ps_from_arco_sp!(ps::AbstractVector, source_grid,
                                 nc_path::AbstractString, date::Date, hour::Int)
    lon, lat, sp = NCDataset(nc_path, "r") do ds
        slice_time = ds["time"][hour + 1]
        expected = DateTime(date) + Hour(hour)
        slice_time == expected ||
            error("ARCO sp time axis mismatch in $nc_path: slot $(hour + 1) " *
                  "decodes to $slice_time, expected $expected")
        (Array{Float64}(ds["longitude"][:]),
         Array{Float64}(ds["latitude"][:]),
         Array{Float64}(ds["sp"][:, :, hour + 1]))   # (nlon, nlat)
    end
    nlon = length(lon); nlat = length(lat)
    nlon >= 2 && nlat >= 2 ||
        error("ARCO sp grid too small: nlon=$nlon nlat=$nlat in $nc_path")
    size(sp) == (nlon, nlat) ||
        throw(DimensionMismatch("ARCO sp slice $(size(sp)) != ($nlon, $nlat)"))

    mesh   = source_grid.mesh
    dlon   = lon[2] - lon[1]          # +0.25 (ascending, periodic)
    dlat   = lat[2] - lat[1]          # -0.25 (descending N→S)
    lon0   = lon[1]
    lat0   = lat[1]
    # The periodic-longitude wrap below assumes a [0,360) origin (ARCO's layout).
    # A shifted origin (e.g. -180) would misindex silently, so assert it.
    (dlon > 0 && -1e-3 <= lon0 <= 1.0) ||
        error("ARCO sp longitude axis not [0,360)-ascending (lon0=$lon0, dlon=$dlon) in $nc_path")

    @inbounds for j in 1:nrings(mesh)
        latj = Float64(mesh.latitudes[j])
        fy   = (latj - lat0) / dlat            # fractional row on descending axis
        jy   = clamp(floor(Int, fy), 0, nlat - 2)
        wy   = clamp(fy - jy, 0.0, 1.0)
        j0   = jy + 1; j1 = j0 + 1
        lons_j = source_grid.lons_by_ring[j]
        off    = mesh.ring_offsets[j]
        for (i, lonc) in enumerate(lons_j)
            fx  = (Float64(lonc) - lon0) / dlon
            ix  = floor(Int, fx)
            wx  = fx - ix
            i0  = mod(ix,     nlon) + 1        # periodic longitude wrap
            i1  = mod(ix + 1, nlon) + 1
            v0  = sp[i0, j0] + wx * (sp[i1, j0] - sp[i0, j0])
            v1  = sp[i0, j1] + wx * (sp[i1, j1] - sp[i0, j1])
            ps[off + i - 1] = v0 + wy * (v1 - v0)
        end
    end
    return ps
end

"""Decode `msg` spectral coefficients straight into level slot `level` of
`cube`. Asserts `1 ≤ level ≤ Nz` so a stray off-archive level fails loudly
instead of silently aliasing a neighbouring slot."""
function _read_into_level_slot!(cube::Array{ComplexF64, 3}, msg,
                                 read_buf::Vector{Float64},
                                 level::Int, Nz::Int)
    1 <= level <= Nz ||
        error("spectral level $level outside [1, $Nz]")
    read_spectral_coeffs!(view(cube, :, :, level), msg, read_buf)
    return cube
end

"""Divide an `n_cells`-laid-out field by `cos(latitude)` of its ring. Used to
recover physical `U`, `V` from the ECMWF `U·cos(φ)` / `V·cos(φ)` form produced
by `vod2uv!`. The polar rings of N320 are at ±89.78° (`cos(φ) ≈ 0.004`); the
guard clamps any non-positive value to `eps(Float64)` so floating-point noise
near `cos(90°)` never crashes a 13-hour preprocess run. Synthesis noise at
the poles is bounded by the spectral truncation and tolerated downstream
(Poisson balance, VDIFF payload regrid)."""
function _divide_by_cos_lat_per_ring!(field::AbstractVector,
                                       mesh::ReducedGaussianMesh)
    @inbounds for j in 1:nrings(mesh)
        cos_lat = max(cosd(Float64(mesh.latitudes[j])), eps(Float64))
        inv_cos = 1.0 / cos_lat
        ring_start = mesh.ring_offsets[j]
        ring_end   = mesh.ring_offsets[j + 1] - 1
        @views field[ring_start:ring_end] .*= inv_cos
    end
    return field
end
