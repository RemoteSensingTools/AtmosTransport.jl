# ERA5 N320 → cubed sphere: conservative regrid of the window fields and the per-window pipeline.
# Split from era5.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ===========================================================================
# Conservative regrid from N320 source mesh to a C180
# cubed-sphere target. Intensive scalars (PS, T, Q, U, V) use the
# `ConservativeRegridding` weights cached on disk; dry-mass derivation on
# the C180 target stays a downstream concern (re-derived from regridded
# PS + Q in the breakpoint-F glue) so the regridder operates on a small
# fixed set of fields.
# ===========================================================================

"""
    ERA5C180RegridFields{FT}

Per-window output container for fields regridded onto the C180 cubed-sphere
target. Holds:

  - `ps` — 2D `(Nc, Nc)` matrix per panel (Pa, moist surface pressure),
  - `u`, `v`, `t`, `qv` — 3D `(Nc, Nc, Nz)` arrays per panel (U/V in
    geographic east/north frame; rotation to panel-local axes happens in
    the breakpoint-F glue where the panel basis is known).

Mass fields (m_dry, delp_dry, ps_dry) are *not* regridded directly; they
are re-derived on the C180 mesh from the regridded PS + Q so that
`Σ_k DELP_dry == PS_dry` to roundoff on the target side as well.
"""
struct ERA5C180RegridFields{FT <: AbstractFloat}
    ps :: NTuple{6, Matrix{FT}}
    u  :: NTuple{6, Array{FT, 3}}
    v  :: NTuple{6, Array{FT, 3}}
    t  :: NTuple{6, Array{FT, 3}}
    qv :: NTuple{6, Array{FT, 3}}
end

function allocate_era5_c180_regrid_fields(target_grid::CubedSphereTargetGeometry{FT},
                                            Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nc = target_grid.mesh.Nc
    Nz_int = Int(Nz)
    return ERA5C180RegridFields{FT}(
        ntuple(_ -> zeros(FT, Nc, Nc), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
    )
end

"""
    ERA5C180TM5ConvectionFields{FT}

TM5 convection fields derived on the C180 cubed-sphere target after the raw
ECMWF diagnostics have been conservatively mapped there.
Each field is a 6-tuple of `(Nc, Nc, Nz)` panel arrays, layer-centered,
kg / m² / s.
"""
struct ERA5C180TM5ConvectionFields{FT <: AbstractFloat}
    entu :: NTuple{6, Array{FT, 3}}
    detu :: NTuple{6, Array{FT, 3}}
    entd :: NTuple{6, Array{FT, 3}}
    detd :: NTuple{6, Array{FT, 3}}
end

"""Raw ECMWF convection diagnostics after conservative N320 → C180 mapping.
Conversion to TM5 entrainment/detrainment deliberately happens only after this
step, matching `tmm_Read_Convec_EC` in the vendored TM5 procedure."""
struct ERA5C180RawConvectionFields{FT <: AbstractFloat}
    udmf :: NTuple{6, Array{FT, 3}}
    ddmf :: NTuple{6, Array{FT, 3}}
    udrf :: NTuple{6, Array{FT, 3}}
    ddrf :: NTuple{6, Array{FT, 3}}
end

function allocate_era5_c180_raw_convection_fields(
        target_grid::CubedSphereTargetGeometry{FT}, Nz::Integer) where FT
    Nc, Nz_int = target_grid.mesh.Nc, Int(Nz)
    panels() = ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6)
    return ERA5C180RawConvectionFields{FT}(panels(), panels(), panels(), panels())
end

function allocate_era5_c180_tm5_convection_fields(target_grid::CubedSphereTargetGeometry{FT},
                                                    Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nc = target_grid.mesh.Nc
    Nz_int = Int(Nz)
    return ERA5C180TM5ConvectionFields{FT}(
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz_int), 6),
    )
end

"""
    ERA5C180RegridWorkspace{FT, R}

Owns the conservative regridder + flat scratch buffers used by
[`regrid_n320_to_c180!`](@ref). Allocated once per (source_grid,
target_grid) pair and reused across every window. `coverage` is the fraction
of each target cell covered by the source mesh (see `_regrid_intensive!`).
"""
struct ERA5C180RegridWorkspace{FT <: AbstractFloat, R}
    regridder    :: R
    coverage     :: Vector{Float64}
    src_flat_2d  :: Vector{Float64}
    src_flat_3d  :: Matrix{Float64}
    dst_flat_2d  :: Vector{Float64}
    dst_flat_3d  :: Matrix{Float64}
end

"""
    allocate_era5_c180_regrid_workspace(source_grid, target_grid, Nz; cache_dir=nothing)

Build (or load from `cache_dir`) the N320 → C180 conservative regridder and
allocate the flat scratch buffers. N320 spectral synthesis remains on the
physical IFS sphere, but conservative weights for these intensive fields use
an otherwise identical source mesh on the target sphere: spherical
intersection libraries require one manifold, and a common radius leaves
intensive interpolation weights unchanged. The regridder's `intersections`
matrix is the expensive piece — on first run it is built and serialised to
JLD2 under `cache_dir`, and subsequent runs load it in milliseconds.
"""
function _n320_regrid_mesh(source::ReducedGaussianMesh, target_radius::Real)
    source.radius == target_radius && return source
    return ReducedGaussianMesh(source.latitudes, source.nlon_per_ring;
                               FT = eltype(source), radius = target_radius)
end

function allocate_era5_c180_regrid_workspace(source_grid::ReducedGaussianTargetGeometry{FT},
                                                target_grid::CubedSphereTargetGeometry{FT},
                                                Nz::Integer;
                                                cache_dir::Union{Nothing, AbstractString} = nothing) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nz_int = Int(Nz)
    regrid_source = _n320_regrid_mesh(source_grid.mesh, target_grid.mesh.radius)
    regridder = build_regridder(regrid_source, target_grid.mesh;
                                 normalize = false,
                                 cache_dir = cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    n_src == ncells(source_grid.mesh) ||
        throw(DimensionMismatch("regridder src_areas length $n_src ≠ N320 cells $(ncells(source_grid.mesh))"))
    n_dst == ncells(target_grid.mesh) ||
        throw(DimensionMismatch("regridder dst_areas length $n_dst ≠ target cells $(ncells(target_grid.mesh))"))
    coverage = _n320_regrid_coverage(regridder, "cubed-sphere")
    return ERA5C180RegridWorkspace{FT, typeof(regridder)}(
        regridder,
        coverage,
        zeros(Float64, n_src),               # src_flat_2d
        zeros(Float64, n_src, Nz_int),       # src_flat_3d
        zeros(Float64, n_dst),               # dst_flat_2d
        zeros(Float64, n_dst, Nz_int),       # dst_flat_3d
    )
end

"""
    regrid_n320_to_c180!(c180_fields, n320_window, workspace, target_grid,
                         wind = ScalarWindRegrid()) -> c180_fields

Conservatively regrid PS (2D) and U, V, T, Q (3D) from the N320 source mesh
to the C180 cubed-sphere target. PS, T and Q are intensive scalars; U and V go
through `wind`, as two scalars (`ScalarWindRegrid`, the default) or as a vector
(`CartesianWindRegrid`). The flat scratch buffers in `workspace` hold the
intermediate Float64 arrays.
"""
function regrid_n320_to_c180!(c180_fields::ERA5C180RegridFields{FT},
                                n320_window::ERA5N320WindowFields,
                                workspace::ERA5C180RegridWorkspace{FT},
                                target_grid::CubedSphereTargetGeometry{FT},
                                wind::AbstractWindRegrid = ScalarWindRegrid()) where FT
    Nc = target_grid.mesh.Nc
    Nz = size(n320_window.u, 2)
    size(workspace.src_flat_3d, 2) == Nz ||
        throw(DimensionMismatch("workspace Nz $(size(workspace.src_flat_3d, 2)) ≠ window Nz $Nz"))
    size(workspace.dst_flat_3d, 2) == Nz ||
        throw(DimensionMismatch("workspace dst Nz $(size(workspace.dst_flat_3d, 2)) ≠ $Nz"))

    # PS — 2D intensive field.
    _regrid_intensive!(workspace.dst_flat_2d, workspace.src_flat_2d,
                       workspace.regridder, workspace.coverage, n320_window.ps)
    unpack_flat_to_panels_2d!(c180_fields.ps, workspace.dst_flat_2d, Nc)

    # 3D intensive fields — all share the same workspace scratch. The winds
    # go through `wind` (as two scalars, or as a vector).
    function regrid3d!(dst_panels, src_field)
        _regrid_intensive!(workspace.dst_flat_3d, workspace.src_flat_3d,
                           workspace.regridder, workspace.coverage, src_field)
        return unpack_flat_to_panels_3d!(dst_panels, workspace.dst_flat_3d, Nc, Nz)
    end
    regrid_winds!(wind, c180_fields.u, c180_fields.v, regrid3d!, n320_window.u, n320_window.v)
    regrid3d!(c180_fields.t, n320_window.t)
    regrid3d!(c180_fields.qv, n320_window.qv)

    return c180_fields
end

function regrid_n320_raw_convection_to_c180!(
        conv_c180::ERA5C180RawConvectionFields{FT},
        conv_n320::ERA5N320ConvectionFields{FT},
        workspace::ERA5C180RegridWorkspace{FT},
        target_grid::CubedSphereTargetGeometry{FT}) where FT
    Nc = target_grid.mesh.Nc
    Nz = size(conv_n320.udmf, 2)
    for (src_field, dst_panels) in (
            (conv_n320.udmf, conv_c180.udmf),
            (conv_n320.ddmf, conv_c180.ddmf),
            (conv_n320.udrf, conv_c180.udrf),
            (conv_n320.ddrf, conv_c180.ddrf))
        _regrid_intensive!(workspace.dst_flat_3d, workspace.src_flat_3d,
                           workspace.regridder, workspace.coverage, src_field)
        unpack_flat_to_panels_3d!(dst_panels, workspace.dst_flat_3d, Nc, Nz)
    end
    return conv_c180
end

# Fraction of each target cell covered by the N320 source mesh: the regridded
# constant 1 (see `_regrid_intensive!`).
function _n320_regrid_coverage(regridder, target_label::AbstractString)
    coverage = apply_regridder!(zeros(length(regridder.dst_areas)), regridder,
                                ones(length(regridder.src_areas)))
    all(c -> 0.9 < c < 1.1, coverage) ||
        error("N320 → $target_label regridder covers target cells by $(extrema(coverage)) " *
              "(expected ≈ 1); rebuild the regridder cache")
    return coverage
end

"""
    ERA5HumidityRemap

Specific humidity of one ERA5 N320 day conservatively remapped to another
horizontal mesh, for converting convective fluxes that live on that mesh to
dry air. Open with [`open_era5_humidity_remap`](@ref), fill with
[`remap_era5_humidity!`](@ref), release with `close`.
"""
struct ERA5HumidityRemap{H <: ERA5GRIBDayHandles, M <: ReducedGaussianMesh, R}
    handles     :: H
    date        :: Date
    source_mesh :: M
    regridder   :: R
    coverage    :: Vector{Float64}
    q_source    :: Matrix{Float64}     # (N320 cells, Nz)
    src_scratch :: Matrix{Float64}
    q_target    :: Matrix{Float64}     # (target cells, Nz)
end

"""
    open_era5_humidity_remap(root_dir, date, target_mesh; Nz=ERA5_NATIVE_LEVEL_COUNT,
                             cache_dir=nothing) -> ERA5HumidityRemap

Prepare to read the ERA5 N320 specific humidity of `date` from the native
GRIB archive under `root_dir` (`ml_an_native_core/era5_core_YYYYMMDD.grib`)
and remap it conservatively, with the coverage normalisation of
`_regrid_intensive!`, to `target_mesh`. The regridder weights are cached
in `cache_dir`.
"""
function open_era5_humidity_remap(root_dir::AbstractString, date::Date, target_mesh;
                                  Nz::Integer = ERA5_NATIVE_LEVEL_COUNT,
                                  cache_dir::Union{Nothing, AbstractString} = nothing)
    handles = open_era5_day(ERA5N320Settings(root_dir = String(root_dir)), date;
                            next_day_handle = false)
    try
        source_mesh = discover_era5_n320_source_grid(handles.core_path; FT = Float64).mesh
        regrid_source = _n320_regrid_mesh(source_mesh, target_mesh.radius)
        regridder = build_regridder(regrid_source, target_mesh; normalize = false,
                                    cache_dir = cache_dir)
        coverage = _n320_regrid_coverage(regridder, string(nameof(typeof(target_mesh))))
        n_src, n_dst = length(regridder.src_areas), length(regridder.dst_areas)
        return ERA5HumidityRemap(handles, date, source_mesh, regridder, coverage,
                                 zeros(n_src, Nz), zeros(n_src, Nz), zeros(n_dst, Nz))
    catch
        close_era5_day!(handles)
        rethrow()
    end
end

"""
    remap_era5_humidity!(remap::ERA5HumidityRemap, hour) -> q_target

Read the specific humidity of `hour` (0–23 UTC) on the N320 mesh
([`read_era5_n320_humidity!`](@ref)) and remap it to the target mesh.
Returns `remap.q_target`, `(target cells, Nz)` with top-first levels.
"""
function remap_era5_humidity!(remap::ERA5HumidityRemap, hour::Integer)
    read_era5_n320_humidity!(remap.q_source, remap.handles, remap.source_mesh,
                             remap.date, hour)
    return _regrid_intensive!(remap.q_target, remap.src_scratch, remap.regridder,
                              remap.coverage, remap.q_source)
end

Base.close(remap::ERA5HumidityRemap) = close_era5_day!(remap.handles)

"""
    derive_c180_tm5_convection!(tm5_fields, conv_fields, window_fields, vc, scratches;
                                basis, stats=nothing) -> tm5_fields

TM5 entrainment and detrainment on every target column from the raw ECMWF
diagnostics already mapped to the target (`conv_fields`), following TM5's
`tmm_Read_Convec_EC`: UDMF/DDMF level `k` becomes interface `k` (the top of
layer `k`) and the ground interface `Nz + 1` is zero, layer thickness from the
target `T`, `q` and `ps` (`dz_hydrostatic_virtual!`), the fluxes put on the
air-mass `basis` of the binary with [`ecmwf_convection_to_basis!`](@ref)
(`DryBasis()` multiplies them by `1 − q`), then [`ec2tm_from_rates!`](@ref).
"""
function derive_c180_tm5_convection!(
        tm5_fields::ERA5C180TM5ConvectionFields{FT},
        conv_fields::ERA5C180RawConvectionFields{FT},
        window_fields::ERA5C180RegridFields{FT},
        vc::HybridSigmaPressure,
        scratches::Vector{TM5ConvectionColumnScratch{FT}};
        basis::AbstractMassBasis,
        stats = nothing) where FT
    thread_stats = stats === nothing ? nothing :
        [TM5CleanupStats() for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for p in 1:6
        tid = Threads.threadid()
        scratch = scratches[tid]
        local_stats = thread_stats === nothing ? nothing : thread_stats[tid]
        Nz = size(window_fields.t[p], 3)
        @inbounds for j in axes(window_fields.t[p], 2), i in axes(window_fields.t[p], 1)
            # GRIB level k (1 = model top) is the flux through the top of
            # layer k, interface k; none crosses the ground, interface Nz + 1
            # (TM5 tmm.F90: `udmf_gg(:,0:nlev-1) = buf; udmf_gg(:,nlev) = 0`).
            for k in 1:Nz
                scratch.udmf_col[k] = conv_fields.udmf[p][i, j, k]
                scratch.ddmf_col[k] = conv_fields.ddmf[p][i, j, k]
                scratch.udrf_col[k] = conv_fields.udrf[p][i, j, k]
                scratch.ddrf_col[k] = conv_fields.ddrf[p][i, j, k]
                scratch.t_col[k] = window_fields.t[p][i, j, k]
                scratch.q_col[k] = window_fields.qv[p][i, j, k]
            end
            scratch.udmf_col[Nz + 1] = zero(FT)
            scratch.ddmf_col[Nz + 1] = zero(FT)
            dz_hydrostatic_virtual!(scratch.dz_col, scratch.t_col, scratch.q_col,
                                     window_fields.ps[p][i, j], vc.A, vc.B, Nz)
            ecmwf_convection_to_basis!(basis, scratch.udmf_col, scratch.ddmf_col,
                                       scratch.udrf_col, scratch.ddrf_col, scratch.q_col)
            ec2tm_from_rates!(scratch.entu_col, scratch.detu_col,
                              scratch.entd_col, scratch.detd_col,
                              scratch.udmf_col, scratch.ddmf_col,
                              scratch.udrf_col, scratch.ddrf_col,
                              scratch.dz_col, Nz; stats = local_stats)
            for k in 1:Nz
                tm5_fields.entu[p][i, j, k] = scratch.entu_col[k]
                tm5_fields.detu[p][i, j, k] = scratch.detu_col[k]
                tm5_fields.entd[p][i, j, k] = scratch.entd_col[k]
                tm5_fields.detd[p][i, j, k] = scratch.detd_col[k]
            end
        end
    end
    if stats !== nothing
        for name in propertynames(stats)
            getproperty(stats, name)[] +=
                sum(getproperty(s, name)[] for s in thread_stats)
        end
    end
    return tm5_fields
end

# ---------------------------------------------------------------------------
# Internal regrid helpers.
# ---------------------------------------------------------------------------

"""
    _regrid_intensive!(dst_flat, src_scratch, regridder, coverage, src) -> dst_flat

Conservative regrid of an intensive field, `src` of size `n_src` or
`(n_src, Nz)`, into `dst_flat`: the area-weighted mean over the part of each
target cell that the source mesh covers. `src_scratch` holds the Float64 copy
that `apply_regridder!`'s sparse matmul needs, without per-window allocation.

`apply_regridder!` divides by the full target-cell area. The reduced Gaussian
source cells, however, have great-circle (chord) edges along their latitude
bounds, so neighbouring rings with different longitude counts leave thin
slivers uncovered: a constant field regrids to 0.9967 poleward of 89° on C90.
Dividing by `coverage`, the regridded constant 1, removes this bias.
"""
function _regrid_intensive!(dst_flat::AbstractVecOrMat{Float64},
                            src_scratch::AbstractVecOrMat{Float64},
                            regridder, coverage::AbstractVector{Float64},
                            src::AbstractVecOrMat)
    size(src_scratch) == size(src) ||
        throw(DimensionMismatch("src_scratch size $(size(src_scratch)) ≠ source $(size(src))"))
    copyto!(src_scratch, src)
    apply_regridder!(dst_flat, regridder, src_scratch)
    dst_flat ./= coverage
    return dst_flat
end

# ---------------------------------------------------------------------------
# Internal helpers carried over from breakpoint B.
# ---------------------------------------------------------------------------

"""Synthesize one 2D spectral field onto an `n_cells`-laid-out output column.
`spectral_to_reduced_scalar!` always writes `Float64`, so the caller supplies
a Float64 `scratch` of the same length as `column` and the result is then
copied into `column` (which may be Float32 or Float64). The copy is one pass
of `n_cells` and is <<1% of the synthesis cost. Callers reuse a single
workspace-owned `scratch` across every level to keep the hot path
allocation-free."""
function _synthesize_into_column!(column::AbstractVector,
                                   spec::AbstractMatrix{ComplexF64},
                                   T::Int,
                                   grid::ReducedGaussianTargetGeometry,
                                   cache::ReducedSpectralThreadCache,
                                   scratch::AbstractVector{Float64})
    length(column) == length(scratch) == ncells(grid.mesh) ||
        throw(DimensionMismatch("column/scratch length mismatch with mesh ($(ncells(grid.mesh)))"))
    spectral_to_reduced_scalar!(scratch, spec, T, grid, cache)
    @inbounds for c in eachindex(column)
        column[c] = scratch[c]
    end
    return column
end

# ===========================================================================
# Per-window end-to-end pipeline.
#
# Bundles the breakpoint B (spectral synthesis + reduced_gg reader), C
# (dry-mass), D (regrid to C180), and E (convection) workspaces into a
# single per-window driver. Produces the regridded C180 scalar fields
# plus per-cell dry-mass and convection forecast fields on the N320
# source mesh that downstream consumers (binary writer, diagnostic
# notebooks) can pick up.
#
# Mass-flux reconstruction, panel-local wind rotation, Poisson balance,
# and v4 binary writing are *not* part of this commit. They build on the
# existing `cubed_sphere_regrid.jl` LL→CS pipeline and are the natural
# follow-on once the per-window scalar surface is stable.
# ===========================================================================

"""
    ERA5N320ToC180Pipeline{FT, RW, CSGrid, SrcGrid, W}

All-in-one container for the per-day ERA5 N320 → C180 preprocessing
workspace. Holds the source-grid descriptor, the hybrid coordinate, the
shared per-cell area vector, every read/derive/regrid workspace from
breakpoints B-D, the convection workspace from E, and the per-window
output fields on both the source mesh and the C180 target.

One pipeline allocated per day-handle, reused across the 24 hourly windows.
"""
struct ERA5N320ToC180Pipeline{FT <: AbstractFloat,
                               RW <: ERA5C180RegridWorkspace{FT},
                               CSGrid <: CubedSphereTargetGeometry{FT},
                               SrcGrid <: ReducedGaussianTargetGeometry{FT},
                               W <: AbstractWindRegrid}
    source_grid        :: SrcGrid
    target_grid        :: CSGrid
    vc                 :: HybridSigmaPressure
    cell_areas         :: Vector{Float64}
    spectral_ws        :: ERA5N320SpectralWorkspace{FT}
    regrid_ws          :: RW
    window_fields      :: ERA5N320WindowFields{FT}
    dry_fields         :: ERA5N320DryMassFields{FT}
    convection_fields  :: Union{Nothing, ERA5N320ConvectionFields{FT}}
    # Raw diagnostics are mapped first; nonlinear ec2tm conversion and closure
    # are then evaluated independently on every target column.
    convection_c180_fields :: Union{Nothing, ERA5C180RawConvectionFields{FT}}
    tm5_derive_scratches :: Union{Nothing, Vector{TM5ConvectionColumnScratch{FT}}}
    tm5_c180_fields    :: Union{Nothing, ERA5C180TM5ConvectionFields{FT}}
    c180_fields        :: ERA5C180RegridFields{FT}
    wind               :: W      # how U/V are regridded (`ScalarWindRegrid` or `CartesianWindRegrid`)
end

"""
    allocate_era5_n320_to_c180_pipeline(handles, target_grid;
                                        Nz=ERA5_NATIVE_LEVEL_COUNT,
                                        cache_dir=nothing,
                                        include_convection=true)

Build the full per-window pipeline for the ERA5 source described by
`handles` (resolved via [`open_era5_day`](@ref)) and the C-tier target
`target_grid`. Discovers the source mesh and spectral truncation from
the day's core GRIB, loads the hybrid coordinate file declared in the
settings, builds (or JLD2-loads from `cache_dir`) the conservative
regridder, and allocates every B/C/D/E workspace.

Convection workspace allocation is gated on `include_convection` so the
caller can opt out for a scalar-only smoke.
"""
function allocate_era5_n320_to_c180_pipeline(handles::ERA5GRIBDayHandles,
                                                target_grid::CubedSphereTargetGeometry{FT};
                                                Nz::Integer = ERA5_NATIVE_LEVEL_COUNT,
                                                cache_dir::Union{Nothing, AbstractString} = nothing,
                                                include_convection::Bool = true,
                                                wind_regrid::Union{Symbol, AbstractWindRegrid} = :cartesian,
                                                synthesis::Union{Nothing, ReducedSpectralSynthesis} = nothing) where FT
    Nz_int = Int(Nz)
    Nz_int >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))

    source_grid = discover_era5_n320_source_grid(handles.core_path; FT = FT)
    T_trunc     = discover_era5_spectral_truncation(handles.core_path)
    vc          = load_hybrid_coefficients(handles.settings.coefficients_file)
    length(vc.A) == length(vc.B) == Nz_int + 1 ||
        throw(DimensionMismatch("hybrid A/B length $(length(vc.A))/$(length(vc.B)) ≠ Nz+1 = $(Nz_int + 1); " *
                                "check `settings.coefficients_file` vs the requested Nz"))

    cell_areas    = n320_cell_areas(source_grid)
    synthesis === nothing && (synthesis = ReducedSpectralSynthesis(source_grid, T_trunc, Nz_int))
    spectral_ws   = allocate_era5_n320_spectral_workspace(source_grid, T_trunc, Nz_int; synthesis)
    regrid_ws     = allocate_era5_c180_regrid_workspace(source_grid, target_grid, Nz_int;
                                                          cache_dir = cache_dir)
    window_fields = allocate_era5_n320_window_fields(source_grid, Nz_int)
    dry_fields    = allocate_era5_n320_dry_mass_fields(source_grid, Nz_int)
    convection_fields = include_convection ?
        allocate_era5_n320_convection_fields(source_grid, Nz_int) : nothing
    convection_c180_fields = include_convection ?
        allocate_era5_c180_raw_convection_fields(target_grid, Nz_int) : nothing
    tm5_derive_scratches = include_convection ?
        [allocate_tm5_convection_column_scratch(FT, Nz_int)
         for _ in 1:Threads.maxthreadid()] : nothing
    tm5_c180_fields = include_convection ?
        allocate_era5_c180_tm5_convection_fields(target_grid, Nz_int) : nothing
    c180_fields   = allocate_era5_c180_regrid_fields(target_grid, Nz_int)
    wind          = build_wind_regrid(wind_regrid, source_grid.mesh, target_grid.mesh, Nz_int, FT)

    return ERA5N320ToC180Pipeline{FT, typeof(regrid_ws), typeof(target_grid), typeof(source_grid),
                                  typeof(wind)}(
        source_grid, target_grid, vc, cell_areas,
        spectral_ws, regrid_ws,
        window_fields, dry_fields, convection_fields,
        convection_c180_fields, tm5_derive_scratches, tm5_c180_fields,
        c180_fields, wind)
end

"""
    process_era5_n320_window!(pipeline, handles, date, hour) -> pipeline

Drive the per-window pipeline for `(date, hour)`:

  1. Synthesise PS / U / V / T / Q on the N320 source mesh (breakpoint B).
  2. Derive dry-air mass + DELP_dry + PS_dry on the source mesh (breakpoint C).
  3. Conservatively regrid PS / U / V / T / Q to the C-tier target.
  4. Optionally read and regrid raw UDMF / DDMF / UDRF / DDRF, put them on
     the dry basis, and perform the nonlinear TM5 conversion and closure on
     the target columns.

After the call, `pipeline.window_fields`, `pipeline.dry_fields`,
`pipeline.convection_fields`, and `pipeline.c180_fields` carry the
window's data on their respective grids.
"""
function process_era5_n320_window!(pipeline::ERA5N320ToC180Pipeline,
                                     handles::ERA5GRIBDayHandles,
                                     date::Date,
                                     hour::Integer)
    _prof = env_flag("ERA5_N320_PROFILE"; true_values = ("1",), case_sensitive = true)
    _t = time()
    read_era5_n320_window_fields!(pipeline.window_fields, pipeline.spectral_ws,
                                    handles, date, hour)
    _t_read = time() - _t; _t = time()
    derive_n320_dry_mass!(pipeline.dry_fields, pipeline.window_fields,
                           pipeline.vc, pipeline.cell_areas)
    _t_drymass = time() - _t; _t_conv = 0.0
    if pipeline.convection_fields !== nothing
        _t = time()
        read_era5_n320_convection_window!(pipeline.convection_fields, handles,
                                            pipeline.source_grid.mesh, date, hour)
        _t_conv += time() - _t
    end
    _t = time()
    regrid_n320_to_c180!(pipeline.c180_fields, pipeline.window_fields,
                           pipeline.regrid_ws, pipeline.target_grid, pipeline.wind)
    _t_regrid = time() - _t
    if pipeline.convection_fields !== nothing
        _t = time()
        regrid_n320_raw_convection_to_c180!(pipeline.convection_c180_fields,
                                             pipeline.convection_fields,
                                             pipeline.regrid_ws,
                                             pipeline.target_grid)
        # This pipeline is dry by construction (step 2), and so are its
        # convective fluxes; the writer accepts only `mass_basis = :dry`.
        derive_c180_tm5_convection!(pipeline.tm5_c180_fields,
                                     pipeline.convection_c180_fields,
                                     pipeline.c180_fields,
                                     pipeline.vc,
                                     pipeline.tm5_derive_scratches;
                                     basis = DryBasis())
        _t_conv += time() - _t
    end
    if _prof
        @info @sprintf("    [prof] window phases: read+synth %.1fs  drymass %.1fs  conv %.1fs  regrid→c180 %.1fs",
                       _t_read, _t_drymass, _t_conv, _t_regrid)
    end
    return pipeline
end
