"""
    SelectedSnapshotFrame

CPU-resident output containing only requested levels and column reductions.
`nlevel` preserves the original vertical coordinate. `levels` maps stored layers
back to model levels; column sums are computed before selecting layers. Tracer
storage is VMR × air mass, with no molecular-weight conversion.
"""
struct SelectedSnapshotFrame{A,C} <: AbstractSnapshotFrame
    time_hours::Float64
    air_mass::A
    tracers::Dict{Symbol,A}
    mass_basis::Symbol
    tracer_total_mass::Dict{Symbol,Float64}
    nlevel::Int
    levels::Vector{Int}
    column_air::Union{Nothing,C}
    column_tracers::Dict{Symbol,C}
end

# One thread per horizontal cell; sequential vertical accumulation gives the
# same summation order as the CPU diagnostic. No device scalar indexing.
@kernel function _snapshot_column_sum!(out, values, ncolumn, nlevel)
    c = @index(Global, Linear)
    total = zero(eltype(out))
    @inbounds for k in 1:nlevel
        total += values[c + (k - 1) * ncolumn]
    end
    @inbounds out[c] = total
end

function _backend_column_sum(values::AbstractArray)
    backend = get_backend(values)
    backend isa KA_CPU && return _column_sum(values)
    return _backend_column_sum(values, _total_accumulator_type(backend))
end

function _backend_column_sum(values::AbstractArray, ::Type{Float64})
    horizontal = size(values)[1:end-1]
    out = similar(values, Float64, horizontal)
    _snapshot_column_sum!(get_backend(values), 256)(out, values, length(out),
        size(values, ndims(values)); ndrange=length(out))
    synchronize(get_backend(values))
    return Array(out)
end

# Backends without Float64 (Metal) still need the CPU diagnostic's precision.
# Carry each column's sum across bounded slabs in the original vertical order;
# reducing a slab in Float32 first would lose signed cancellation residuals.
function _backend_column_sum(values::AbstractArray, ::Type{Float32})
    axis = ndims(values)
    out = zeros(Float64, size(values)[1:end-1])
    totals = vec(out)
    for first in 1:16:size(values, axis)
        last = min(first + 15, size(values, axis))
        slab = Array(selectdim(values, axis, first:last))
        columns = reshape(slab, length(out), last - first + 1)
        @inbounds for k in axes(columns, 2), c in eachindex(totals)
            totals[c] += Float64(columns[c, k])
        end
    end
    return out
end
_backend_column_sum(values::NTuple{6}) = map(_backend_column_sum, values)

_output_interior(values, ::Union{LatLonMesh,ReducedGaussianMesh}, halo) = values
function _output_interior(values::NTuple{6}, mesh::CubedSphereMesh, halo)
    h = Int(halo)
    h >= 0 || throw(ArgumentError("halo_width must be non-negative"))
    r = h + 1:h + mesh.Nc
    return map(values) do panel
        size(panel, 1) >= mesh.Nc + 2h && size(panel, 2) >= mesh.Nc + 2h ||
            throw(DimensionMismatch("panel cannot provide requested interior"))
        @view panel[r, r, :]
    end
end

function _capture_levels(values::AbstractArray, levels)
    shape = (size(values)[1:end-1]..., length(levels))
    isempty(levels) && return Array{eltype(values)}(undef, shape)
    if levels == collect(1:size(values, ndims(values)))
        return Array(values)
    elseif get_backend(values) isa KA_CPU
        return Array(selectdim(values, ndims(values), levels))
    end
    # A host Vector of indices inside a CuArray view is not GPU-safe. Copy
    # selected layers using scalar (isbits) indices into compact device storage.
    packed = similar(values, eltype(values), shape)
    for (dest, source) in enumerate(levels)
        copyto!(selectdim(packed, ndims(packed), dest),
                selectdim(values, ndims(values), source))
    end
    return Array(packed)
end

_capture_levels(values::NTuple{6}, levels) = map(a -> _capture_levels(a, levels), values)

function _capture_selected_snapshot(model, fields::OutputFieldSpec;
                                    time_hours=0, halo_width=0)
    mesh = model.grid.horizontal
    names = _select_tracer_keys(collect(tracer_names(model.state)), fields)
    source_air = _output_interior(model.state.air_mass, mesh, halo_width)
    first_air = source_air isa Tuple ? first(source_air) : source_air
    Nz = size(first_air, ndims(first_air))
    levels = (fields.air_mass || fields.air_mass_per_area) ?
        _layer_indices(fields.air_mass_layers, fields, Nz) : Int[]
    for name in names
        append!(levels, _layer_indices(tracer_fields(fields, name).layers, fields, Nz))
    end
    sort!(unique!(levels))
    air = _capture_levels(source_air, levels)
    # Keep one common layer mapping for air and selected tracers. Column-only
    # output has zero stored layers even when Nz is large.
    tracers = Dict{Symbol,typeof(air)}()
    needs_column_air = fields.column_air_mass_per_area ||
        any(name -> tracer_fields(fields, name).column_mean, names)
    # C is determined without doing an unrequested reduction.
    C = source_air isa Tuple ? NTuple{6,Array{Float64,2}} : Array{Float64,ndims(first_air)-1}
    column_air = needs_column_air ? _float64_columns(_backend_column_sum(source_air)) : nothing
    columns = Dict{Symbol,C}()
    totals = Dict{Symbol,Float64}()
    for name in names
        source = _output_interior(get_tracer(model.state, name), mesh, halo_width)
        tracers[name] = _capture_levels(source, levels)
        totals[name] = _backend_tracer_total(source)
        tf = tracer_fields(fields, name)
        if tf.column_mean || tf.column_mass_per_area
            columns[name] = _float64_columns(_backend_column_sum(source))
        end
    end
    return SelectedSnapshotFrame{typeof(air),C}(Float64(time_hours), air, tracers,
        _basis_symbol(mass_basis(model.state)), totals, Nz, levels, column_air, columns)
end

_float64_columns(x::AbstractArray) = convert(Array{Float64,ndims(x)}, x)
_float64_columns(x::Tuple) = map(_float64_columns, x)
_nlevel(frame::SelectedSnapshotFrame, mesh) = frame.nlevel

function _stored_indices(frame::SelectedSnapshotFrame, levels)
    idx = [searchsortedfirst(frame.levels, k) for k in levels]
    all(i -> i <= length(frame.levels), idx) && frame.levels[idx] == levels ||
        throw(ArgumentError("writer requested levels that were not captured"))
    return idx
end
_air_layers(frame::SnapshotFrame, levels) = _select_levels(frame.air_mass, levels)
_air_layers(frame::SelectedSnapshotFrame, levels) =
    _select_levels(frame.air_mass, _stored_indices(frame, levels))
_tracer_layers(frame::SnapshotFrame, name, levels) = _select_levels(frame.tracers[name], levels)
_tracer_layers(frame::SelectedSnapshotFrame, name, levels) =
    _select_levels(frame.tracers[name], _stored_indices(frame, levels))
_frame_vmr(frame, name, levels) =
    mixing_ratio_field(_air_layers(frame, levels), _tracer_layers(frame, name, levels))
_frame_column_air(frame::SnapshotFrame) = _column_sum(frame.air_mass)
function _frame_column_air(frame::SelectedSnapshotFrame)
    frame.column_air === nothing && throw(ArgumentError("column air mass was not captured"))
    return frame.column_air
end
_frame_column_tracer(frame::SnapshotFrame, name) = _column_sum(frame.tracers[name])
_frame_column_tracer(frame::SelectedSnapshotFrame, name) = frame.column_tracers[name]
_column_ratio(num::AbstractArray, den::AbstractArray) = ifelse.(den .> 0, num ./ den, NaN)
_column_ratio(num::Tuple, den::Tuple) = map(_column_ratio, num, den)
_frame_column_mean(frame, name) =
    _column_ratio(_frame_column_tracer(frame, name), _frame_column_air(frame))

function _check_frame_shapes(frame::SelectedSnapshotFrame, mesh)
    _check_frame_shapes(SnapshotFrame(frame.time_hours, frame.air_mass,
                                     frame.tracers, frame.mass_basis;
                                     tracer_total_mass=frame.tracer_total_mass), mesh)
    return nothing
end
