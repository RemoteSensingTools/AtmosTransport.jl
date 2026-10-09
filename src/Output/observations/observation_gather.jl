# ---------------------------------------------------------------------------
# Device gather of observation columns and host reconstruction of pressure,
# height, and dry mixing ratio.
#
# The gather is the only device work in observation sampling. It copies the
# air mass and the selected tracer storage of the requested columns into
# compact (level, observation) buffers; every derived quantity is computed on
# the host in Float64 with the same conventions as `mixing_ratio_field`
# (NaN where the carrier mass is not positive).
# ---------------------------------------------------------------------------

# One work-item per (level, observation). Pure gather with no arithmetic on
# the values, so Metal Float32 and CUDA Float64 share one path. `columns`
# holds Int32 linear column indices into the parent slab and is promoted to
# Int before multiplying so large slabs cannot overflow. `offset` places this
# launch's observations inside the shared output buffers. Bounds are checked
# on the host before the launch (`_check_gather_inputs`).
@kernel function _gather_columns!(out_air, out_tracers, @Const(air), @Const(raw),
                                  @Const(columns), @Const(slots), ncolumn::Int, nlevel::Int,
                                  offset::Int)
    k, m = @index(Global, NTuple)
    n = m + offset
    @inbounds begin
        base = Int(columns[n]) + (k - 1) * ncolumn
        out_air[k, n] = air[base]
        for t in eachindex(slots)
            out_tracers[k, t, n] = raw[base + (Int(slots[t]) - 1) * ncolumn * nlevel]
        end
    end
end

# Host twin with the identical index arithmetic; also the CPU-backend path.
function _gather_columns_host!(out_air, out_tracers, air, raw, columns, slots,
                               ncolumn::Int, nlevel::Int, offset::Int, count::Int)
    @inbounds for m in 1:count, k in 1:nlevel
        n = m + offset
        base = Int(columns[n]) + (k - 1) * ncolumn
        out_air[k, n] = air[base]
        for t in eachindex(slots)
            out_tracers[k, t, n] = raw[base + (Int(slots[t]) - 1) * ncolumn * nlevel]
        end
    end
    return nothing
end

"""
    ObservationGatherBuffers(reference, nlevel, slots; capacity = 1024)

Grow-only scratch for [`gather_columns!`](@ref). `reference` is any state
array; device buffers are allocated with `similar(reference, …)` so they
live on the same backend. `slots` are the tracer storage indices to gather
(see `tracer_index`); an empty `slots` gathers a single field (see
[`gather_field!`](@ref)). Host mirrors `air_host (nlevel, capacity)` and
`tracers_host (nlevel, nslots, capacity)` keep the state element type and
are valid in columns `1:n` after a gather of `n` requests.
"""
mutable struct ObservationGatherBuffers{FT, DI <: AbstractVector{Int32},
                                        DA <: AbstractMatrix{FT}, DT <: AbstractArray{FT, 3}}
    nlevel::Int
    slots::DI
    max_slot::Int
    capacity::Int
    columns::DI
    air::DA
    tracers::DT
    air_host::Matrix{FT}
    tracers_host::Array{FT, 3}
end

function ObservationGatherBuffers(reference::AbstractArray{FT}, nlevel::Integer,
                                  slots::AbstractVector{<:Integer};
                                  capacity::Integer = 1024) where {FT}
    nlevel >= 1 || throw(ArgumentError("nlevel must be positive; got $(nlevel)"))
    all(>(0), slots) || throw(ArgumentError("tracer slots must be positive; got $(slots)"))
    cap = max(Int(capacity), 1)
    dev_slots = similar(reference, Int32, length(slots))
    copyto!(dev_slots, Int32.(slots))
    columns = similar(reference, Int32, cap)
    air = similar(reference, FT, (Int(nlevel), cap))
    tracers = similar(reference, FT, (Int(nlevel), length(slots), cap))
    return ObservationGatherBuffers{FT, typeof(columns), typeof(air), typeof(tracers)}(
        Int(nlevel), dev_slots, isempty(slots) ? 0 : Int(maximum(slots)), cap, columns, air, tracers,
        Matrix{FT}(undef, nlevel, cap), Array{FT, 3}(undef, nlevel, length(slots), cap))
end

"Number of tracer slots a buffer gathers (0 for a field buffer)."
nslots(buf::ObservationGatherBuffers) = size(buf.tracers, 2)

function _ensure_capacity!(buf::ObservationGatherBuffers{FT}, n::Integer) where {FT}
    n <= buf.capacity && return buf
    cap = nextpow(2, Int(n))
    buf.columns = similar(buf.columns, cap)
    buf.air = similar(buf.air, (buf.nlevel, cap))
    buf.tracers = similar(buf.tracers, (buf.nlevel, nslots(buf), cap))
    buf.air_host = Matrix{FT}(undef, buf.nlevel, cap)
    buf.tracers_host = Array{FT, 3}(undef, buf.nlevel, nslots(buf), cap)
    buf.capacity = cap
    return buf
end

# Host-side validation that makes the `@inbounds` gather safe: the slab must
# share the buffer backend and level count, the tracer parent must carry the
# slab's horizontal layout with enough slots, and every column index must
# address the slab.
function _check_gather_inputs(buf::ObservationGatherBuffers, air, raw, columns::AbstractVector{Int32},
                              range::UnitRange{Int})
    get_backend(air) == get_backend(buf.air) || throw(ArgumentError(
        "state arrays live on $(get_backend(air)) but the gather buffers on $(get_backend(buf.air))"))
    nlevel = size(air, ndims(air))
    nlevel == buf.nlevel || throw(DimensionMismatch(
        "state has $(nlevel) levels but the gather buffers were built for $(buf.nlevel)"))
    ncolumn = length(air) ÷ nlevel
    if nslots(buf) > 0
        ndims(raw) == ndims(air) + 1 && size(raw)[1:ndims(air)] == size(air) || throw(DimensionMismatch(
            "tracer storage $(size(raw)) does not extend the air-mass layout $(size(air))"))
        buf.max_slot <= size(raw, ndims(raw)) || throw(ArgumentError(
            "gather buffers address tracer slot $(buf.max_slot) but the state holds $(size(raw, ndims(raw)))"))
    end
    for i in range
        c = columns[i]
        1 <= c <= ncolumn || throw(BoundsError(air, (Int(c), 1)))
    end
    return ncolumn, nlevel
end

function _launch_gather!(buf::ObservationGatherBuffers, air, raw, columns::AbstractVector{Int32},
                         range::UnitRange{Int})
    isempty(range) && return nothing
    ncolumn, nlevel = _check_gather_inputs(buf, air, raw, columns, range)
    offset = first(range) - 1
    count = length(range)
    backend = get_backend(air)
    if backend isa KA_CPU
        _gather_columns_host!(buf.air, buf.tracers, air, raw, buf.columns, buf.slots,
                              ncolumn, nlevel, offset, count)
    else
        _gather_columns!(backend)(buf.air, buf.tracers, air, raw, buf.columns, buf.slots,
                                  ncolumn, nlevel, offset; ndrange = (nlevel, count))
    end
    return nothing
end

function _download!(buf::ObservationGatherBuffers, n::Int)
    synchronize(get_backend(buf.air))
    copyto!(buf.air_host, 1, buf.air, 1, buf.nlevel * n)
    copyto!(buf.tracers_host, 1, buf.tracers, 1, buf.nlevel * nslots(buf) * n)
    return nothing
end

function _upload_columns!(buf::ObservationGatherBuffers, columns::AbstractVector{Int32})
    n = length(columns)
    _ensure_capacity!(buf, n)
    # GPU backends only implement `copyto!` from a plain `Array` source.
    n == 0 || copyto!(buf.columns, 1, convert(Vector{Int32}, columns), 1, n)
    return n
end

# Ranges must tile 1:n in order, one per slab (empty ranges allowed).
function _check_slab_ranges(ranges, nslabs::Int, n::Int)
    length(ranges) == nslabs || throw(ArgumentError(
        "expected $(nslabs) panel ranges, got $(length(ranges))"))
    next = 1
    for (p, range) in enumerate(ranges)
        isempty(range) && continue
        first(range) == next || throw(ArgumentError(
            "panel range $(range) for panel $(p) must start at $(next): ranges must tile 1:$(n) in order"))
        next = last(range) + 1
    end
    next == n + 1 || throw(ArgumentError("panel ranges cover 1:$(next - 1) but $(n) columns were requested"))
    return nothing
end

# Shared driver: one launch per slab over its contiguous run of columns.
function _gather_slabs!(buf::ObservationGatherBuffers, airs::Tuple, raws::Tuple,
                        columns::AbstractVector{Int32}, ranges)
    n = _upload_columns!(buf, columns)
    n == 0 && return 0
    _check_slab_ranges(ranges, length(airs), n)
    for (air, raw, range) in zip(airs, raws, ranges)
        _launch_gather!(buf, air, raw, columns, UnitRange{Int}(range))
    end
    _download!(buf, n)
    return n
end

"""
    gather_columns!(buf, state, columns, panel_ranges = nothing) -> n

Gather the air mass and the buffered tracer slots of `columns` (Int32 linear
column indices, see `CellLocation.column`) from `state` into
`buf.air_host[:, 1:n]` and `buf.tracers_host[:, :, 1:n]`. Lat-lon and
reduced-Gaussian states ignore `panel_ranges`; cubed-sphere states require
`columns` grouped by panel with `panel_ranges[p]` giving panel `p`'s run so
that the six ranges tile `1:n` in order.
"""
gather_columns!(buf::ObservationGatherBuffers, state::CellState, columns::AbstractVector{Int32},
                panel_ranges = nothing) =
    _gather_slabs!(buf, (state.air_mass,), (state.tracers_raw,), columns, (1:length(columns),))

gather_columns!(buf::ObservationGatherBuffers, state::CubedSphereState, columns::AbstractVector{Int32},
                panel_ranges) =
    _gather_slabs!(buf, state.air_mass, state.tracers_raw, columns, panel_ranges)

"""
    gather_field!(buf, field, columns, panel_ranges = nothing) -> n

Gather one per-layer field with the air-mass layout (an array, or a 6-tuple of
panel arrays) into `buf.air_host[:, 1:n]`; `buf` must have been built with no
tracer slots. Used for optional temperature fields.
"""
function gather_field!(buf::ObservationGatherBuffers, field::AbstractArray,
                       columns::AbstractVector{Int32}, panel_ranges = nothing)
    nslots(buf) == 0 || throw(ArgumentError("gather_field! needs buffers without tracer slots"))
    return _gather_slabs!(buf, (field,), (field,), columns, (1:length(columns),))
end

function gather_field!(buf::ObservationGatherBuffers, field::NTuple{6, <:AbstractArray},
                       columns::AbstractVector{Int32}, panel_ranges)
    nslots(buf) == 0 || throw(ArgumentError("gather_field! needs buffers without tracer slots"))
    return _gather_slabs!(buf, field, field, columns, panel_ranges)
end

# -- host reconstruction ----------------------------------------------------

"""
    interface_pressures!(p_half, air_column, area, g, p_top)

Interface pressures (Pa) from a column of air mass per cell (kg), top down:
`p_half[1] = p_top`, `p_half[k+1] = p_half[k] + g·m[k]/area`. On a dry-basis
state this is the dry partial pressure.
"""
function interface_pressures!(p_half::AbstractVector{Float64}, air_column::AbstractVector,
                              area::Real, g::Real, p_top::Real)
    nlevel = length(air_column)
    length(p_half) == nlevel + 1 || throw(DimensionMismatch(
        "p_half needs $(nlevel + 1) entries for $(nlevel) layers; got $(length(p_half))"))
    g_over_a = Float64(g) / Float64(area)
    p = Float64(p_top)
    @inbounds p_half[1] = p
    @inbounds for k in 1:nlevel
        p += Float64(air_column[k]) * g_over_a
        p_half[k + 1] = p
    end
    return p_half
end

"How the layer temperature for hypsometric heights is obtained."
abstract type AbstractLayerTemperature end
"One temperature for every layer (fallback when the binary carries none)."
struct ConstantLayerTemperature <: AbstractLayerTemperature
    kelvin::Float64
end
"""
2 m temperature with a constant lapse rate (K m⁻¹) above the surface, floored
at the standard-atmosphere tropopause temperature so tall columns stay physical.
"""
struct SurfaceLapseTemperature <: AbstractLayerTemperature
    surface_kelvin::Float64
    lapse_rate::Float64
end
const STANDARD_LAPSE_RATE = 0.0065          # K m⁻¹
const STANDARD_TROPOPAUSE_KELVIN = 216.65   # US standard atmosphere
SurfaceLapseTemperature(surface_kelvin::Real) = SurfaceLapseTemperature(surface_kelvin, STANDARD_LAPSE_RATE)
"Per-layer temperatures, `kelvin[k]` with k = 1 at the top."
struct ProfileLayerTemperature{V <: AbstractVector} <: AbstractLayerTemperature
    kelvin::V
end

_check_layer_temperature(::AbstractLayerTemperature, nlevel::Int) = nothing
function _check_layer_temperature(t::ProfileLayerTemperature, nlevel::Int)
    length(t.kelvin) == nlevel || throw(DimensionMismatch(
        "temperature profile has $(length(t.kelvin)) layers; the column has $(nlevel)"))
    return nothing
end

layer_temperature(t::ConstantLayerTemperature, k, z_bottom) = t.kelvin
layer_temperature(t::SurfaceLapseTemperature, k, z_bottom) =
    max(t.surface_kelvin - t.lapse_rate * z_bottom, STANDARD_TROPOPAUSE_KELVIN)
layer_temperature(t::ProfileLayerTemperature, k, z_bottom) = Float64(t.kelvin[k])
height_method_label(::ConstantLayerTemperature) = :constant
height_method_label(::SurfaceLapseTemperature) = :surface_lapse
height_method_label(::ProfileLayerTemperature) = :profile
# Integer codes stored in the site files; `HEIGHT_METHOD_CODES` is the single table.
const HEIGHT_METHOD_CODES = (
    (ConstantLayerTemperature, Int8(0), "constant temperature"),
    (SurfaceLapseTemperature, Int8(1), "2 m temperature with lapse rate"),
    (ProfileLayerTemperature, Int8(2), "layer temperature profile"))
height_method_code(::ConstantLayerTemperature) = HEIGHT_METHOD_CODES[1][2]
height_method_code(::SurfaceLapseTemperature) = HEIGHT_METHOD_CODES[2][2]
height_method_code(::ProfileLayerTemperature) = HEIGHT_METHOD_CODES[3][2]
"Code written for a site outside its time range (no heights computed)."
const HEIGHT_METHOD_NOT_SAMPLED = Int8(-1)
height_method_codes_description() =
    join(("$(code) = $(text)" for (_, code, text) in HEIGHT_METHOD_CODES), ", ") *
    ", $(HEIGHT_METHOD_NOT_SAMPLED) = not sampled (site outside its time range)"

"""
Integer `interp_flag` codes of point-event rows: blended between two window
ends, sampled one-sided at the later end (window length changed between
binaries), or taken from the nearest window end (`nearest_window` mode).
"""
const INTERP_FLAG_CODES = (bracketed = Int8(0), one_sided = Int8(1), nearest_window = Int8(2))
interp_flag(::NearestWindowSampling) = INTERP_FLAG_CODES.nearest_window
interp_flag(::LinearWindowInterpolation, bracketed::Bool) =
    bracketed ? INTERP_FLAG_CODES.bracketed : INTERP_FLAG_CODES.one_sided


"""
    layer_heights_agl!(z_half, p_half, temperature, g; R_dry = R_DRY_AIR)

Hypsometric interface heights above ground (m) from interface pressures,
`z_half[end] = 0` at the surface, accumulated upward with the layer
temperature from `temperature` evaluated at the layer's bottom height. A
non-positive or non-monotone pressure gives an infinite height from that
layer upward.
"""
function layer_heights_agl!(z_half::AbstractVector{Float64}, p_half::AbstractVector{Float64},
                            temperature::AbstractLayerTemperature, g::Real;
                            R_dry::Float64 = R_DRY_AIR)
    nlevel = length(p_half) - 1
    length(z_half) == nlevel + 1 || throw(DimensionMismatch(
        "z_half needs $(nlevel + 1) entries; got $(length(z_half))"))
    _check_layer_temperature(temperature, nlevel)
    z = 0.0
    @inbounds z_half[nlevel + 1] = z
    @inbounds for k in nlevel:-1:1
        if isfinite(z) && p_half[k] > 0 && p_half[k + 1] >= p_half[k]
            T = layer_temperature(temperature, k, z)
            z += R_dry * T / Float64(g) * log(p_half[k + 1] / p_half[k])
        else
            z = Inf
        end
        z_half[k] = z
    end
    return z_half
end

"""
    intake_layer_index(z_half, intake_height) -> Int

Layer `k` with `z_half[k+1] <= h < z_half[k]` (k = 1 top). A `NaN` or
non-positive height selects the lowest layer; a height above the column top
selects layer 1.
"""
function intake_layer_index(z_half::AbstractVector, intake_height::Real)
    nlevel = length(z_half) - 1
    (isnan(intake_height) || intake_height <= 0) && return nlevel
    @inbounds for k in nlevel:-1:1
        intake_height < z_half[k] && return k
    end
    return 1
end

"Air-mass-weighted column mean mixing ratio; NaN when the column holds no air."
function column_mean_vmr(air_column::AbstractVector, tracer_column::AbstractVector)
    den = sum(Float64, air_column)
    return den > 0 ? sum(Float64, tracer_column) / den : NaN
end

"Per-layer mixing ratio `tracer/air`, NaN where the carrier mass is not positive."
function mixing_ratio_profile!(out::AbstractVector{Float64}, air_column::AbstractVector,
                               tracer_column::AbstractVector)
    @inbounds for k in eachindex(out)
        m = Float64(air_column[k])
        out[k] = m > 0 ? Float64(tracer_column[k]) / m : NaN
    end
    return out
end
