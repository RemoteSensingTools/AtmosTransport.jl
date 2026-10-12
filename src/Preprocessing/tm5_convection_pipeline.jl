# ---------------------------------------------------------------------------
# TM5 convection pipeline wiring.
#
# Bridges the column math, physics BIN reader, and grid-level loop
# + native→merged remap to the `process_day` preprocessor.  Shared
# across LL / RG / CS targets.
#
# Per-hour flow on the ERA5-native horizontal grid (Nlon_src, Nlat_src,
# typically 720×361 matching the physics BIN):
#
#   native-L137 UDMF/DDMF/UDRF/DDRF + T + Q + PS (from physics BIN)
#       ↓  tm5_native_fields_for_hour! (× (1 − q) onto the binary's basis, ec2tm)
#   native-L137 entu/detu/entd/detd (kg/m²/s)
#       ↓  merge_tm5_field_3d!
#   merged-Nz entu/detu/entd/detd on the source (ERA5) grid
#       ↓  ConservativeRegridding (per-topology) — NOT bilinear, see
#          feedback_conservative_regrid_for_mass_fluxes.md
#   merged-Nz entu/detu/entd/detd on the target grid (LL / RG / CS)
#
# The regrid step is topology-specific and happens inline in each
# process_day variant; the PIPELINE helper here only produces the
# source-grid merged fields.
# ---------------------------------------------------------------------------

"""
    TM5PreprocessingWorkspace{FT, R, B, MB}

Per-day scratch for the TM5 convection preprocessor.  `basis::MB`
(`DryBasis()` or `MoistBasis()`) is the air-mass basis of the binary being
written; the ECMWF fluxes are put on it before ec2tm.  Holds the
per-column vectors reused across all columns of an hour, plus the
4×(Nlon_src, Nlat_src, Nz_native) native-vertical buffers and the
4×(Nlon_src, Nlat_src, Nz) merged-vertical buffers on the ERA5
source grid.  `regridder` is the conservative regridder from the
ERA5 LL source mesh to the target mesh, or `nothing` when source
and target are the same LL grid (identity fast-path).

`physics_bufs::B` holds optional FT-typed conversion buffers for
the six physics inputs (`udmf, ddmf, udrf, ddrf, t, q`).  When the
workspace FT matches the physics BIN eltype (Float32 today), `B ===
Nothing` and the kernel reads the BIN views directly with zero
copy.  When FT differs (e.g. the F64 path), `B === NTuple{6,
Array{FT, 3}}` and `compute_tm5_merged_hour_on_source!` upcasts
hour views into them in-place — alloc once per workspace, reused
across all 24 hours per day.
"""
struct TM5PreprocessingWorkspace{FT, R, B, MB <: AbstractMassBasis}
    basis           :: MB
    col_scratch     :: NTuple{11, Vector{FT}}
    entu_native     :: Array{FT, 3}
    detu_native     :: Array{FT, 3}
    entd_native     :: Array{FT, 3}
    detd_native     :: Array{FT, 3}
    entu_merged_src :: Array{FT, 3}
    detu_merged_src :: Array{FT, 3}
    entd_merged_src :: Array{FT, 3}
    detd_merged_src :: Array{FT, 3}
    physics_bufs    :: B
    regridder       :: R
    target_nlon     :: Int
    target_nlat     :: Int
end

"""
    allocate_tm5_workspace(Nlon_src, Nlat_src, Nz_native, Nz, FT;
                           basis, regridder=nothing, target_nlon=Nlon_src,
                           target_nlat=Nlat_src) -> TM5PreprocessingWorkspace

Allocate the TM5 preprocessing workspace for a binary on the air-mass
`basis` (`DryBasis()` or `MoistBasis()`, required).  Pass `regridder=nothing`
for identity (source and target grids match).  Otherwise pass a
`ConservativeRegridding.Regridder` built via
`build_regridder(source_mesh, target_mesh)` from
`src/Regridding/`.
"""
function allocate_tm5_workspace(Nlon_src::Integer, Nlat_src::Integer,
                                Nz_native::Integer, Nz::Integer,
                                ::Type{FT};
                                basis::AbstractMassBasis,
                                regridder = nothing,
                                target_nlon::Integer = Nlon_src,
                                target_nlat::Integer = Nlat_src,
                                physics_eltype::Type{<:AbstractFloat} = Float32,
                                ) where {FT <: AbstractFloat}
    col_scratch = (
        Vector{FT}(undef, Nz_native + 1),    # udmf_col
        Vector{FT}(undef, Nz_native + 1),    # ddmf_col
        Vector{FT}(undef, Nz_native),        # udrf_col
        Vector{FT}(undef, Nz_native),        # ddrf_col
        Vector{FT}(undef, Nz_native),        # t_col
        Vector{FT}(undef, Nz_native),        # q_col
        Vector{FT}(undef, Nz_native),        # dz_col
        Vector{FT}(undef, Nz_native),        # entu_col
        Vector{FT}(undef, Nz_native),        # detu_col
        Vector{FT}(undef, Nz_native),        # entd_col
        Vector{FT}(undef, Nz_native),        # detd_col
    )
    physics_bufs = if physics_eltype === FT
        nothing
    else
        # Allocate once, reuse across all 24 hours per day.
        ntuple(_ -> Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native), 6)
    end
    return TM5PreprocessingWorkspace{FT, typeof(regridder), typeof(physics_bufs), typeof(basis)}(
        basis,
        col_scratch,
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        physics_bufs,
        regridder,
        Int(target_nlon),
        Int(target_nlat),
    )
end

"""
    compute_tm5_merged_hour_on_source!(ws, reader, hour, ps_hour, ak_full, bk_full,
                                         Nz_native, merge_map; stats)

Phase 1 of the per-hour TM5 step.  Reads native-L137 ERA5 physics
fields for `hour` from the mmap-backed `reader`, puts them on `ws.basis`
and runs the TM5 math column-by-column into `ws.*_native`, then collapses to merged Nz
into `ws.*_merged_src`.  All work is on the ERA5-native horizontal
grid (720×361 for standard physics BINs).

This path requires source and target grids to match
`(Nx, Ny) == (720, 361)`.  PS comes from the caller (typically the
preprocessor's `transform.sp` after spectral synthesis) because the
physics BIN does not carry PS.

`stats::Union{Nothing, NamedTuple}` is the TM5CleanupStats bundle;
when non-nothing, counters accumulate across all columns of all
hours processed.

Shape guards:

- `reader` fields shape `(Nlon_src, Nlat_src, Nz_native, Nt)` —
  must match `ws.entu_native`'s leading dims.
- `ps_hour` shape `(Nlon_src, Nlat_src)`.
- `merge_map` length `Nz_native`.
"""
function compute_tm5_merged_hour_on_source!(
        ws::TM5PreprocessingWorkspace{FT},
        reader,
        hour::Integer,
        ps_hour::AbstractArray{<:Real, 2},
        ak_full::AbstractVector,
        bk_full::AbstractVector,
        Nz_native::Integer,
        merge_map::AbstractVector{<:Integer};
        stats = nothing) where {FT <: AbstractFloat}

    udmf_4d = get_era5_physics_field(reader, :udmf)
    ddmf_4d = get_era5_physics_field(reader, :ddmf)
    udrf_4d = get_era5_physics_field(reader, :udrf_rate)
    ddrf_4d = get_era5_physics_field(reader, :ddrf_rate)
    t_4d    = get_era5_physics_field(reader, :t)
    q_4d    = get_era5_physics_field(reader, :q)

    # Slice to 3D per-hour view (zero-copy — Julia reshape-view).
    udmf_h = @view udmf_4d[:, :, :, hour]
    ddmf_h = @view ddmf_4d[:, :, :, hour]
    udrf_h = @view udrf_4d[:, :, :, hour]
    ddrf_h = @view ddrf_4d[:, :, :, hour]
    t_h    = @view t_4d[:, :, :, hour]
    q_h    = @view q_4d[:, :, :, hour]

    size(udmf_h, 1) == size(ws.entu_native, 1) || error(
        "Physics BIN Nlon=$(size(udmf_h,1)) ≠ workspace Nlon=$(size(ws.entu_native,1))")
    size(udmf_h, 2) == size(ws.entu_native, 2) || error(
        "Physics BIN Nlat=$(size(udmf_h,2)) ≠ workspace Nlat=$(size(ws.entu_native,2))")
    size(ps_hour)   == (size(udmf_h, 1), size(udmf_h, 2)) || error(
        "ps_hour shape $(size(ps_hour)) ≠ expected $(size(udmf_h,1)), $(size(udmf_h,2)))")

    # Match workspace FT (Float32 or Float64) against physics BIN eltype
    # (Float32 today). When they differ, upcast hour views into the
    # workspace's pre-allocated `physics_bufs` (allocated once at
    # workspace construction). When they match, pass views through with
    # zero copy.
    FT_reader = eltype(udmf_h)
    udmf_in, ddmf_in, udrf_in, ddrf_in, t_in, q_in =
        if FT_reader === FT
            ws.physics_bufs === nothing || error(
                "Workspace FT=$FT matches physics BIN eltype $FT_reader but " *
                "conversion buffers were allocated; rebuild the workspace " *
                "with `physics_eltype=$FT_reader`.")
            (udmf_h, ddmf_h, udrf_h, ddrf_h, t_h, q_h)
        else
            ws.physics_bufs === nothing && error(
                "Workspace FT=$FT differs from physics BIN eltype $FT_reader " *
                "but no conversion buffers were allocated; rebuild the " *
                "workspace with `physics_eltype=$FT_reader`.")
            bufs = ws.physics_bufs
            for (dst, src) in zip(bufs, (udmf_h, ddmf_h, udrf_h, ddrf_h, t_h, q_h))
                @. dst = FT(src)
            end
            bufs
        end

    # The Commit-3 grid helper expects FT-typed ps_hour; accept any
    # real and convert here so callers can pass Float64 transform.sp.
    ps_h_ft = eltype(ps_hour) === FT ? ps_hour : FT.(ps_hour)

    tm5_native_fields_for_hour!(
        ws.entu_native, ws.detu_native, ws.entd_native, ws.detd_native,
        udmf_in, ddmf_in, udrf_in, ddrf_in, t_in, q_in, ps_h_ft,
        ak_full, bk_full, Nz_native;
        basis = ws.basis, stats = stats, scratch = ws.col_scratch,
    )

    merge_tm5_field_3d!(ws.entu_merged_src, ws.entu_native, merge_map)
    merge_tm5_field_3d!(ws.detu_merged_src, ws.detu_native, merge_map)
    merge_tm5_field_3d!(ws.entd_merged_src, ws.entd_native, merge_map)
    merge_tm5_field_3d!(ws.detd_merged_src, ws.detd_native, merge_map)
    return nothing
end

"""
    log_tm5_cleanup_stats(stats, date_str)

Pretty-print the per-day TM5 cleanup counters produced by
`TM5CleanupStats()`.  Zero-valued counters are omitted to keep the
output compact; a fully-clean day yields a single line.
"""
function log_tm5_cleanup_stats(stats, date_str::AbstractString)
    n_cols = stats.columns_processed[]
    n_cols == 0 && return nothing

    nonzero = NamedTuple{
        (:no_updraft, :no_downdraft,
         :levels_udmf_clipped, :levels_ddmf_clipped,
         :levels_udrf_clipped, :levels_ddrf_clipped,
         :levels_entu_neg, :levels_detu_neg,
         :levels_entd_neg, :levels_detd_neg)
    }((
        stats.no_updraft[], stats.no_downdraft[],
        stats.levels_udmf_clipped[], stats.levels_ddmf_clipped[],
        stats.levels_udrf_clipped[], stats.levels_ddrf_clipped[],
        stats.levels_entu_neg[], stats.levels_detu_neg[],
        stats.levels_entd_neg[], stats.levels_detd_neg[],
    ))

    any_nonzero = any(v -> v > 0, values(nonzero))
    if !any_nonzero
        @info "  TM5 convection $date_str: $n_cols columns, all clean"
        return nothing
    end

    summary_parts = String["$n_cols cols"]
    for (k, v) in pairs(nonzero)
        v > 0 && push!(summary_parts, "$k=$v")
    end
    @info "  TM5 convection $date_str: $(join(summary_parts, ", "))"
    return nothing
end

"""
    tm5_copy_or_regrid_ll!(dst_3d, ws_field, ws)

LL phase-2 helper: copy (identity, when shapes match) or regrid
(conservative) a single source-grid merged TM5 field into the
per-window target array `dst_3d` of shape `(Nx, Ny, Nz)`.
"""
function tm5_copy_or_regrid_ll!(dst_3d::AbstractArray{FT, 3},
                                 ws_field::AbstractArray{FT, 3},
                                 ws::TM5PreprocessingWorkspace{FT}) where FT
    if ws.regridder === nothing
        size(dst_3d) == size(ws_field) ||
            error("TM5 LL identity path requires matching shapes, got " *
                  "dst=$(size(dst_3d)) vs src=$(size(ws_field))")
        copyto!(dst_3d, ws_field)
    else
        size(dst_3d, 1) == ws.target_nlon && size(dst_3d, 2) == ws.target_nlat ||
            error("TM5 LL regrid target shape mismatch: dst=$(size(dst_3d)), " *
                  "expected ($(ws.target_nlon), $(ws.target_nlat), $(size(ws_field, 3)))")
        size(dst_3d, 3) == size(ws_field, 3) ||
            error("TM5 LL regrid vertical mismatch: dst Nz=$(size(dst_3d, 3)) " *
                  "vs src Nz=$(size(ws_field, 3))")
        src_flat = reshape(ws_field, size(ws_field, 1) * size(ws_field, 2), size(ws_field, 3))
        dst_flat = reshape(dst_3d, size(dst_3d, 1) * size(dst_3d, 2), size(dst_3d, 3))
        apply_regridder!(dst_flat, ws.regridder, src_flat)
    end
    return dst_3d
end
