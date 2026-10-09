# ERA5 N320 GRIB reader: dry-air mass (N320 and C180) and convection forecast fields, TM5 column scratch.
# Split from era5.jl (refactor phase 4); included by Preprocessing.jl in this order.

# ===========================================================================
# Hybrid pressure → dry-air mass on the N320 source grid.
#
# Mirrors the GEOS endpoint dry-mass derivation (`endpoint_dry_mass!`) so the
# dry-basis runtime contract is nominally bit-identical across source paths:
#
#   ΔA[k] = A[k+1] − A[k]   (Pa)
#   ΔB[k] = B[k+1] − B[k]   (dimensionless)
#   DELP_full[k] = ΔA[k] + ΔB[k] · PS_total
#   DELP_dry[k]  = (1 − Q[k]) · DELP_full[k]
#   PS_dry       = Σ_k DELP_dry[k]
#   m_dry[k]     = DELP_dry[k] · cell_area / g
#
# All arithmetic runs in Float64 internally and is downcast to FT only on
# write. Vertical merge (e.g. `MergeAbovePressure` for parity with the GEOS
# L72 cap) is delegated to the existing `apply_vertical!` plumbing in the
# breakpoint-F glue.
# ===========================================================================

"""
    ERA5N320DryMassFields{FT}

Per-window dry-basis output container. `m_dry` is dry-air mass per cell per
layer (kg), `delp_dry` is dry pressure thickness per cell per layer (Pa),
`ps_dry` is the column-integrated dry surface pressure (Pa). `ps_dry_acc` is
a Float64 accumulator that backs `ps_dry` so Σ_k DELP_dry stays accurate
even when FT = Float32 (137-layer summation with ~10 hPa values per layer
would otherwise lose ~100 Pa to single-precision rounding).
"""
struct ERA5N320DryMassFields{FT <: AbstractFloat}
    m_dry      :: Matrix{FT}     # (n_cells, Nz)
    delp_dry   :: Matrix{FT}     # (n_cells, Nz)
    ps_dry     :: Vector{FT}     # (n_cells,)
    ps_dry_acc :: Vector{Float64} # Float64 accumulator scratch
end

function allocate_era5_n320_dry_mass_fields(source_grid::ReducedGaussianTargetGeometry{FT},
                                              Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    nc = ncells(source_grid.mesh)
    Nz_int = Int(Nz)
    return ERA5N320DryMassFields{FT}(
        zeros(FT, nc, Nz_int),
        zeros(FT, nc, Nz_int),
        zeros(FT, nc),
        zeros(Float64, nc),
    )
end

"""
    derive_c180_dry_mass!(m_dry, delp_dry, ps_dry, ps_dry_acc,
                           ps_panels, qv_panels, vc, cell_areas; grav=STANDARD_GRAVITY) -> nothing

Cubed-sphere variant of [`derive_n320_dry_mass!`](@ref). Builds the dry-air
layer mass, dry pressure thickness, and dry surface pressure for each of
the 6 C-tier panels from the regridded moist PS + Q on C180. Same formula
as the GEOS endpoint dry-mass derivation so `Σ_k DELP_dry = PS_dry` to
roundoff on the target mesh too.

Inputs and outputs are `NTuple{6, ...}` panel tuples; `ps_dry_acc` is a
panel-tuple Float64 accumulator (so a Float32 output preserves the
multi-level summation precision). `cell_areas[i, j]` is shared across all
6 panels (the CS mesh is isotropic per panel).
"""
function derive_c180_dry_mass!(m_dry::NTuple{6, AbstractArray{<:Real, 3}},
                                 delp_dry::NTuple{6, AbstractArray{<:Real, 3}},
                                 ps_dry::NTuple{6, AbstractMatrix{<:Real}},
                                 ps_dry_acc::NTuple{6, Matrix{Float64}},
                                 ps_panels::NTuple{6, AbstractMatrix{<:Real}},
                                 qv_panels::NTuple{6, AbstractArray{<:Real, 3}},
                                 vc::HybridSigmaPressure,
                                 cell_areas::AbstractMatrix{<:Real};
                                 grav::Real = STANDARD_GRAVITY)
    Nc, _, Nz = size(m_dry[1])
    length(vc.A) == length(vc.B) == Nz + 1 ||
        throw(DimensionMismatch("hybrid A/B length $(length(vc.A))/$(length(vc.B)) ≠ Nz+1 = $(Nz + 1)"))
    size(cell_areas) == (Nc, Nc) ||
        throw(DimensionMismatch("cell_areas $(size(cell_areas)) ≠ (Nc=$Nc, Nc)"))

    A = vc.A
    B = vc.B
    inv_g = 1.0 / Float64(grav)

    @inbounds for p in 1:6
        fill!(ps_dry_acc[p], 0.0)
        # k outer / i,j inner — column-major over (Nc, Nc, Nz).
        for k in 1:Nz
            dA = Float64(A[k + 1]) - Float64(A[k])
            dB = Float64(B[k + 1]) - Float64(B[k])
            for j in 1:Nc, i in 1:Nc
                ps_total   = Float64(ps_panels[p][i, j])
                area       = Float64(cell_areas[i, j])
                delp_full  = dA + dB * ps_total
                delp_dry_k = (1.0 - Float64(qv_panels[p][i, j, k])) * delp_full
                FT = eltype(delp_dry[p])
                delp_dry[p][i, j, k] = FT(delp_dry_k)
                m_dry[p][i, j, k]    = FT(delp_dry_k * area * inv_g)
                ps_dry_acc[p][i, j] += delp_dry_k
            end
        end
        FT = eltype(ps_dry[p])
        for j in 1:Nc, i in 1:Nc
            ps_dry[p][i, j] = FT(ps_dry_acc[p][i, j])
        end
    end
    return nothing
end

"""
    n320_cell_areas(source_grid) -> Vector{Float64}

Materialise per-cell areas (m²) for the N320 source mesh. Always returns
`Vector{Float64}` regardless of `source_grid`'s element type — downstream
dry-mass arithmetic runs in Float64 for precision and the per-cell mesh
quadrature is itself Float64. Cached by callers that derive dry mass for
many windows of the same day.
"""
function n320_cell_areas(source_grid::ReducedGaussianTargetGeometry)
    mesh = source_grid.mesh
    return [Float64(cell_area(mesh, c)) for c in 1:ncells(mesh)]
end

"""
    derive_n320_dry_mass!(dry, window, vc, cell_areas; grav=STANDARD_GRAVITY) -> dry

Reconstruct dry-air mass per layer, dry pressure thickness, and dry surface
pressure from a populated `window::ERA5N320WindowFields` (moist PS, Q) using
the hybrid coordinate `vc` (length `Nz+1` A and B arrays in Pa and 1
respectively, top-down) and the per-cell areas (m²). Matches the GEOS
`endpoint_dry_mass!` formula so the runtime sees a nominally bit-identical
dry-basis contract regardless of source.

Asserts shapes and `length(vc.A) == length(vc.B) == Nz + 1` so a coefficient
table that does not match the workspace `Nz` fails immediately with a clear
DimensionMismatch.
"""
function derive_n320_dry_mass!(dry::ERA5N320DryMassFields{FT},
                                window::ERA5N320WindowFields,
                                vc::HybridSigmaPressure,
                                cell_areas::AbstractVector{<:Real};
                                grav::Real = STANDARD_GRAVITY) where FT
    nc, Nz = size(dry.m_dry)
    size(window.qv) == (nc, Nz) ||
        throw(DimensionMismatch("window.qv $(size(window.qv)) ≠ ($nc, $Nz)"))
    length(window.ps) == nc ||
        throw(DimensionMismatch("window.ps length $(length(window.ps)) ≠ $nc"))
    length(cell_areas) == nc ||
        throw(DimensionMismatch("cell_areas length $(length(cell_areas)) ≠ $nc"))
    length(dry.ps_dry_acc) == nc ||
        throw(DimensionMismatch("ps_dry_acc length $(length(dry.ps_dry_acc)) ≠ $nc"))
    length(vc.A) == length(vc.B) == Nz + 1 ||
        throw(DimensionMismatch("hybrid A/B length $(length(vc.A))/$(length(vc.B)) ≠ Nz+1 = $(Nz + 1)"))

    A = vc.A
    B = vc.B
    inv_g = 1.0 / Float64(grav)
    fill!(dry.ps_dry_acc, 0.0)

    # k-outer / c-inner traversal: the `(n_cells, Nz)` Float arrays are
    # column-major, so the contiguous axis is `c`. This keeps every write
    # to `delp_dry[c, k]`, `m_dry[c, k]` and every read from `qv[c, k]` on
    # a unit stride. dA, dB are level-only and hoist out of the cell loop.
    # `ps_dry_acc` accumulates in Float64 to keep precision when FT=Float32.
    @inbounds for k in 1:Nz
        dA = Float64(A[k + 1]) - Float64(A[k])
        dB = Float64(B[k + 1]) - Float64(B[k])
        for c in 1:nc
            ps_total   = Float64(window.ps[c])
            area       = Float64(cell_areas[c])
            delp_full  = dA + dB * ps_total
            delp_dry_k = (1.0 - Float64(window.qv[c, k])) * delp_full
            dry.delp_dry[c, k] = FT(delp_dry_k)
            dry.m_dry[c, k]    = FT(delp_dry_k * area * inv_g)
            dry.ps_dry_acc[c] += delp_dry_k
        end
    end
    @inbounds for c in 1:nc
        dry.ps_dry[c] = FT(dry.ps_dry_acc[c])
    end
    return dry
end

# ===========================================================================
# Convection forecast fields on the N320 source grid.
#
# The ERA5 convection product is a forecast bundle: model-level UDMF, DDMF,
# UDRF, DDRF (param ids 235009-235012, GRIB shortNames `avg_umf`, `avg_dmf`,
# `avg_udr`, `avg_ddr`) archived twice daily from 06 UTC and 18 UTC bases,
# each carrying hourly time-mean values for steps 1..12. One UTC day of
# windowed transport-binary output is covered by:
#
#   hour 0..5  → previous-day `era5_convection_(D-1).grib`, 18 UTC base,
#                stepRange "$(h+6)-$(h+7)"
#   hour 6..17 → today's `era5_convection_D.grib`,           06 UTC base,
#                stepRange "$(h-6)-$(h-5)"
#   hour 18..23 → today's `era5_convection_D.grib`,          18 UTC base,
#                 stepRange "$(h-18)-$(h-17)"
#
# All four fields live on the N320 reduced_gg mesh — same layout as Q. The
# reader reuses `_reorder_grib_reduced_gg_to_mesh!` to flip the ring axis
# from the GRIB's native N→S to the mesh's S→N convention.
# ===========================================================================

"""
    ERA5N320ConvectionFields{FT}

Per-window convection forecast fields on the N320 source mesh. All four
fields are `(n_cells, Nz)`:

  - `udmf` — updraft convective mass flux (kg m⁻² s⁻¹)
  - `ddmf` — downdraft convective mass flux (kg m⁻² s⁻¹)
  - `udrf` — updraft detrainment rate (kg m⁻³ s⁻¹)
  - `ddrf` — downdraft detrainment rate (kg m⁻³ s⁻¹)

Downstream conversion to GEOS-style CMFMC + DTRAIN (or TM5-style entu/entd)
happens in the breakpoint-F glue once both source and target geometries
are known.
"""
struct ERA5N320ConvectionFields{FT <: AbstractFloat}
    udmf :: Matrix{FT}
    ddmf :: Matrix{FT}
    udrf :: Matrix{FT}
    ddrf :: Matrix{FT}
end

function allocate_era5_n320_convection_fields(source_grid::ReducedGaussianTargetGeometry{FT},
                                                Nz::Integer) where FT
    Nz >= 1 || throw(ArgumentError("Nz must be ≥ 1, got $Nz"))
    nc = ncells(source_grid.mesh)
    return ERA5N320ConvectionFields{FT}(
        zeros(FT, nc, Int(Nz)),
        zeros(FT, nc, Int(Nz)),
        zeros(FT, nc, Int(Nz)),
        zeros(FT, nc, Int(Nz)),
    )
end

"""
    era5_convection_hour_address(hour) -> (use_prev_day, data_time_hhmm, step_range)

Map a UTC hour `h ∈ 0..23` to the GRIB header tuple that addresses the
matching ECMWF convection forecast sample (see `read_era5_n320_convection_window!`
docstring for the cycle layout). Returns `(::Bool, ::Int, ::String)`.
"""
function era5_convection_hour_address(hour::Integer)
    0 <= hour <= 23 || throw(ArgumentError("hour must be in 0..23, got $hour"))
    h = Int(hour)
    if h < 6
        return (true,  1800, "$(h + 6)-$(h + 7)")
    elseif h < 18
        return (false,  600, "$(h - 6)-$(h - 5)")
    else
        return (false, 1800, "$(h - 18)-$(h - 17)")
    end
end

# Maps ERA5 GRIB shortName → field slot in `ERA5N320ConvectionFields`.
const _ERA5_CONVECTION_SHORT_NAMES = (
    avg_umf = :udmf,
    avg_dmf = :ddmf,
    avg_udr = :udrf,
    avg_ddr = :ddrf,
)

"""
    read_era5_n320_convection_window!(fields, handles, mesh, date, hour) -> fields

Fill `fields` with one hour's worth of N320 convection forecast fields for
`(date, hour)`. The reader picks `handles.convection_path` or
`handles.prev_convection_path` based on the hour-address mapping and
forward-iterates that file, matching messages whose `(dataTime, stepRange,
shortName)` triple falls within the requested sample. Each matching message
is decoded directly into the appropriate `(n_cells, Nz)` output slot via
`_reorder_grib_reduced_gg_to_mesh!`.

Completeness gates name the missing field in any error so a corrupt or
partial download is immediately visible. All four fields × 137 levels must
be present for the call to succeed.
"""
function read_era5_n320_convection_window!(fields::ERA5N320ConvectionFields{FT},
                                            handles::ERA5GRIBDayHandles,
                                            mesh::ReducedGaussianMesh,
                                            date::Date,
                                            hour::Integer) where FT
    handles.convection_path !== nothing ||
        error("ERA5 convection read requested but settings.include_convection=false")

    use_prev, data_time, step_range = era5_convection_hour_address(hour)
    path = if use_prev
        handles.prev_convection_path !== nothing ||
            error("ERA5 convection hour $hour of $(date) needs the previous-day file " *
                  "($(date - Day(1)) era5_convection*.grib) which is not on disk")
        handles.prev_convection_path
    else
        handles.convection_path
    end

    nc, Nz = size(fields.udmf)
    nc == ncells(mesh) ||
        throw(DimensionMismatch("fields.udmf rows $nc ≠ ncells(mesh) $(ncells(mesh))"))

    # NamedTuple of (slot_symbol → output Matrix) and (slot_symbol → completion
    # BitVector). The shared `field_slot` symbol indexes both, so adding a
    # fifth convection field is a one-liner here and in
    # `_ERA5_CONVECTION_SHORT_NAMES`.
    field_matrices = (udmf = fields.udmf, ddmf = fields.ddmf,
                       udrf = fields.udrf, ddrf = fields.ddrf)
    have = (udmf = falses(Nz), ddmf = falses(Nz),
            udrf = falses(Nz), ddrf = falses(Nz))

    GribFile(path) do gf
        for msg in gf
            Int(msg["dataTime"]) == data_time || continue
            String(msg["stepRange"]) == step_range || continue
            short_name = String(msg["shortName"])
            field_slot = get(_ERA5_CONVECTION_SHORT_NAMES, Symbol(short_name), nothing)
            field_slot === nothing && continue

            level = Int(msg["level"])
            1 <= level <= Nz || continue

            _check_grid_point_layout(msg)
            vals = msg["values"]
            pl   = msg["pl"]
            _reorder_grib_reduced_gg_to_mesh!(
                view(getproperty(field_matrices, field_slot), :, level),
                vals, pl, mesh)
            getproperty(have, field_slot)[level] = true
        end
    end

    for name in propertynames(have)
        bits = getproperty(have, name)
        all(bits) || error("ERA5 convection read: $(uppercase(string(name))) missing for " *
                            "$(date) hour $hour at levels $(findall(!, bits))")
    end

    return fields
end

"""
    TM5ConvectionColumnScratch{FT}

Per-column storage for converting ECMWF convection diagnostics to TM5
entrainment and detrainment on the target grid. One instance is allocated per
Julia thread and reused across panels, cells, and windows.
"""
struct TM5ConvectionColumnScratch{FT <: AbstractFloat}
    udmf_col :: Vector{FT}    # Nz+1 half-level
    ddmf_col :: Vector{FT}    # Nz+1
    udrf_col :: Vector{FT}    # Nz
    ddrf_col :: Vector{FT}    # Nz
    t_col    :: Vector{FT}    # Nz
    q_col    :: Vector{FT}    # Nz
    dz_col   :: Vector{FT}    # Nz
    entu_col :: Vector{FT}    # Nz
    detu_col :: Vector{FT}
    entd_col :: Vector{FT}
    detd_col :: Vector{FT}
end

function allocate_tm5_convection_column_scratch(::Type{FT}, Nz::Integer) where FT
    Nz_int = Int(Nz)
    return TM5ConvectionColumnScratch{FT}(
        Vector{FT}(undef, Nz_int + 1),  # udmf_col half-level
        Vector{FT}(undef, Nz_int + 1),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
        Vector{FT}(undef, Nz_int),
    )
end
