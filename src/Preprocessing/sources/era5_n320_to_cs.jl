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
allocate the flat scratch buffers. The regridder's `intersections` matrix is
the expensive piece — on first run it is built and serialised to JLD2 under
`cache_dir`, and subsequent runs load it in milliseconds.
"""
function allocate_era5_c180_regrid_workspace(source_grid::ReducedGaussianTargetGeometry{FT},
                                                target_grid::CubedSphereTargetGeometry{FT},
                                                Nz::Integer;
                                                cache_dir::Union{Nothing, AbstractString} = nothing) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    Nz_int = Int(Nz)
    regridder = build_regridder(source_grid.mesh, target_grid.mesh;
                                 normalize = false,
                                 cache_dir = cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    n_src == ncells(source_grid.mesh) ||
        throw(DimensionMismatch("regridder src_areas length $n_src ≠ N320 cells $(ncells(source_grid.mesh))"))
    n_dst == ncells(target_grid.mesh) ||
        throw(DimensionMismatch("regridder dst_areas length $n_dst ≠ C180 cells $(ncells(target_grid.mesh))"))
    coverage = apply_regridder!(zeros(n_dst), regridder, ones(n_src))   # regridded constant 1
    all(c -> 0.9 < c < 1.1, coverage) ||
        error("N320 → cubed-sphere regridder covers target cells by $(extrema(coverage)) " *
              "(expected ≈ 1); rebuild the regridder cache")
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
    _unpack_flat_to_cs_panels_2d!(c180_fields.ps, workspace.dst_flat_2d, Nc)

    # 3D intensive fields — all share the same workspace scratch. The winds
    # go through `wind` (as two scalars, or as a vector).
    function regrid3d!(dst_panels, src_field)
        _regrid_intensive!(workspace.dst_flat_3d, workspace.src_flat_3d,
                           workspace.regridder, workspace.coverage, src_field)
        return _unpack_flat_to_cs_panels_3d!(dst_panels, workspace.dst_flat_3d, Nc, Nz)
    end
    _regrid_winds!(wind, c180_fields.u, c180_fields.v, regrid3d!, n320_window.u, n320_window.v)
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
        _unpack_flat_to_cs_panels_3d!(dst_panels, workspace.dst_flat_3d, Nc, Nz)
    end
    return conv_c180
end

function derive_c180_tm5_convection!(
        tm5_fields::ERA5C180TM5ConvectionFields{FT},
        conv_fields::ERA5C180RawConvectionFields{FT},
        window_fields::ERA5C180RegridFields{FT},
        vc::HybridSigmaPressure,
        scratches::Vector{TM5ConvectionColumnScratch{FT}};
        stats = nothing) where FT
    thread_stats = stats === nothing ? nothing :
        [TM5CleanupStats() for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for p in 1:6
        tid = Threads.threadid()
        scratch = scratches[tid]
        local_stats = thread_stats === nothing ? nothing : thread_stats[tid]
        Nz = size(window_fields.t[p], 3)
        @inbounds for j in axes(window_fields.t[p], 2), i in axes(window_fields.t[p], 1)
            scratch.udmf_col[1] = zero(FT)
            scratch.ddmf_col[1] = zero(FT)
            for k in 1:Nz
                scratch.udmf_col[k + 1] = conv_fields.udmf[p][i, j, k]
                scratch.ddmf_col[k + 1] = conv_fields.ddmf[p][i, j, k]
                scratch.udrf_col[k] = conv_fields.udrf[p][i, j, k]
                scratch.ddrf_col[k] = conv_fields.ddrf[p][i, j, k]
                scratch.t_col[k] = window_fields.t[p][i, j, k]
                scratch.q_col[k] = window_fields.qv[p][i, j, k]
            end
            dz_hydrostatic_virtual!(scratch.dz_col, scratch.t_col, scratch.q_col,
                                     window_fields.ps[p][i, j], vc.A, vc.B, Nz)
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

"""Unpack a flat `n_dst = 6 × Nc²` vector into 6 panels of `(Nc, Nc)`. The
CS mesh enumerates cells panel-major in column-major Julia order within each
panel: `flat_index = (p-1)·Nc² + (j-1)·Nc + i`, and
`panel[i, j] = flat[offset + (j-1)·Nc + i]`."""
function _unpack_flat_to_cs_panels_2d!(panels::NTuple{6, Matrix{FT}},
                                        flat::AbstractVector{Float64},
                                        Nc::Int) where FT
    length(flat) == 6 * Nc * Nc ||
        throw(DimensionMismatch("flat length $(length(flat)) ≠ 6×$Nc² = $(6*Nc*Nc)"))
    @inbounds for p in 1:6
        panel = panels[p]
        offset = (p - 1) * Nc * Nc
        for j in 1:Nc, i in 1:Nc
            panel[i, j] = FT(flat[offset + (j - 1) * Nc + i])
        end
    end
    return panels
end

"""Unpack a flat `(n_dst, Nz)` matrix into 6 panels of `(Nc, Nc, Nz)`."""
function _unpack_flat_to_cs_panels_3d!(panels::NTuple{6, Array{FT, 3}},
                                        flat::AbstractMatrix{Float64},
                                        Nc::Int, Nz::Int) where FT
    size(flat, 1) == 6 * Nc * Nc ||
        throw(DimensionMismatch("flat rows $(size(flat, 1)) ≠ 6×$Nc² = $(6*Nc*Nc)"))
    size(flat, 2) == Nz ||
        throw(DimensionMismatch("flat cols $(size(flat, 2)) ≠ Nz=$Nz"))
    @inbounds for p in 1:6
        panel = panels[p]
        offset = (p - 1) * Nc * Nc
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            panel[i, j, k] = FT(flat[offset + (j - 1) * Nc + i, k])
        end
    end
    return panels
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
    spectral_to_reduced_scalar!(scratch, spec, T, grid, cache; centered = true)
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
                                                wind_regrid::Union{Symbol, AbstractWindRegrid} = :scalar,
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
    wind          = _wind_regrid(wind_regrid, source_grid.mesh, target_grid.mesh, Nz_int, FT)

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
  4. Optionally read and regrid raw UDMF / DDMF / UDRF / DDRF, then perform
     the nonlinear TM5 conversion and closure on the target columns.

After the call, `pipeline.window_fields`, `pipeline.dry_fields`,
`pipeline.convection_fields`, and `pipeline.c180_fields` carry the
window's data on their respective grids.
"""
function process_era5_n320_window!(pipeline::ERA5N320ToC180Pipeline,
                                     handles::ERA5GRIBDayHandles,
                                     date::Date,
                                     hour::Integer)
    _prof = get(ENV, "ERA5_N320_PROFILE", "") == "1"
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
        derive_c180_tm5_convection!(pipeline.tm5_c180_fields,
                                     pipeline.convection_c180_fields,
                                     pipeline.c180_fields,
                                     pipeline.vc,
                                     pipeline.tm5_derive_scratches)
        _t_conv += time() - _t
    end
    if _prof
        @info @sprintf("    [prof] window phases: read+synth %.1fs  drymass %.1fs  conv %.1fs  regrid→c180 %.1fs",
                       _t_read, _t_drymass, _t_conv, _t_regrid)
    end
    return pipeline
end
