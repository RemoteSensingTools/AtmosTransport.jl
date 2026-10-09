abstract type AbstractSnapshotFrame end

"""
    SnapshotFrame

CPU-resident snapshot of one model state at one output time.

`air_mass` and each entry in `tracers` keep the topology-native storage shape:

- LL: `(lon, lat, lev)`
- RG: `(cell, lev)`
- CS: `NTuple{6, Array{T, 3}}` with panel interiors `(Xdim, Ydim, lev)`

Tracer arrays store the model's conservative `χ × carrier-air-mass` quantity,
not mixing ratio or physical kg species. Derived VMR and column diagnostics are
computed by [`write_snapshot_netcdf`](@ref). `tracer_total_mass` stores a
compensated Float64 sum of that conservative quantity for every tracer. It is
kept separately so Float32 visualization payloads cannot degrade global signed
mass diagnostics through cancellation or output conversion.

Snapshot storage must be CPU-resident. Use [`capture_snapshot`](@ref) for model
state on any backend; it strips cubed-sphere halos and copies device arrays to
the host before constructing the frame.
"""
struct SnapshotFrame{A} <: AbstractSnapshotFrame
    time_hours::Float64
    air_mass::A
    tracers::Dict{Symbol, A}
    mass_basis::Symbol
    tracer_total_mass::Dict{Symbol, Float64}
end

function _require_cpu_snapshot_storage(field::AbstractArray, label)
    get_backend(field) isa KA_CPU || throw(ArgumentError(
        "$(label) must be CPU-resident snapshot storage; use capture_snapshot " *
        "for model state or copy the array with Array(...)"))
    return nothing
end

function _require_cpu_snapshot_storage(panels::NTuple{6, <:AbstractArray}, label)
    for (panel, field) in enumerate(panels)
        _require_cpu_snapshot_storage(field, "$(label) panel $(panel)")
    end
    return nothing
end

function _snapshot_tracer_totals(tracers)
    return Dict(name => _backend_tracer_total(tracer)
                for (name, tracer) in tracers)
end

function _validated_tracer_totals(tracers, provided)
    tracer_keys = Set(keys(tracers))
    total_keys = Set(keys(provided))
    total_keys == tracer_keys || throw(ArgumentError(
        "snapshot tracer-total keys $(sort!(collect(total_keys))) do not match " *
        "tracer keys $(sort!(collect(tracer_keys)))"))
    totals = Dict{Symbol, Float64}()
    for name in keys(tracers)
        value = Float64(provided[name])
        isfinite(value) || throw(ArgumentError(
            "snapshot tracer total for $(name) is not finite: $(value)"))
        totals[name] = value
    end
    return totals
end

function SnapshotFrame(time_hours::Real,
                       air_mass::A,
                       tracers::AbstractDict{Symbol, <:Any},
                       mass_basis::Symbol;
                       tracer_total_mass=nothing) where A
    _require_cpu_snapshot_storage(air_mass, "snapshot air_mass")
    typed_tracers = Dict{Symbol, A}()
    for (name, tracer) in tracers
        tracer isa A || throw(ArgumentError(
            "snapshot tracer $(name) has type $(typeof(tracer)); expected $(A) to match air_mass"))
        _require_cpu_snapshot_storage(tracer, "snapshot tracer $(name)")
        typed_tracers[name] = tracer
    end
    totals = tracer_total_mass === nothing ?
             _snapshot_tracer_totals(typed_tracers) :
             _validated_tracer_totals(typed_tracers, tracer_total_mass)
    return SnapshotFrame{A}(Float64(time_hours), air_mass, typed_tracers,
                            mass_basis, totals)
end

"""
    SnapshotWriteOptions(; float_type=Float32, deflate_level=0, shuffle=true)

Options controlling NetCDF snapshot output.

- `float_type` is the on-disk type for heavy diagnostic variables.
- `deflate_level` is the NetCDF compression level for heavy payload variables:
  `0` disables compression, `1` is fast/light, and `9` is maximum compression.
- `shuffle` enables the NetCDF shuffle filter when compression is active.
"""
struct SnapshotWriteOptions{FT <: AbstractFloat}
    float_type::Type{FT}
    deflate_level::Int
    shuffle::Bool
end

function SnapshotWriteOptions(; float_type::DataType=Float32,
                              deflate_level::Integer=0,
                              shuffle::Bool=true)
    float_type in (Float32, Float64) ||
        throw(ArgumentError("float_type must be Float32 or Float64 for NetCDF output, got $(float_type)"))
    0 <= deflate_level <= 9 ||
        throw(ArgumentError("deflate_level must be in 0:9, got $(deflate_level)"))
    if float_type === Float32
        return SnapshotWriteOptions{Float32}(Float32, Int(deflate_level), shuffle)
    else
        return SnapshotWriteOptions{Float64}(Float64, Int(deflate_level), shuffle)
    end
end

_basis_symbol(::DryBasis) = :dry
_basis_symbol(::MoistBasis) = :moist

function _copy_cpu_array(a)
    return Array(a)
end

function _strip_cs_halo(panel, mesh::CubedSphereMesh, halo_width::Integer)
    Hp = Int(halo_width)
    Nc = mesh.Nc
    Hp >= 0 || throw(ArgumentError("halo_width must be non-negative, got $(halo_width)"))
    size(panel, 1) >= Nc + 2Hp && size(panel, 2) >= Nc + 2Hp ||
        throw(DimensionMismatch("CS panel size $(size(panel)) cannot provide C$(Nc) interior with halo_width=$(Hp)"))
    return Array(panel[Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :])
end

_extract_for_output(a, ::LatLonMesh; halo_width::Integer=0) = _copy_cpu_array(a)
_extract_for_output(a, ::ReducedGaussianMesh; halo_width::Integer=0) = _copy_cpu_array(a)

function _extract_for_output(a::NTuple{6, <:Any}, mesh::CubedSphereMesh;
                             halo_width::Integer=0)
    return ntuple(p -> _strip_cs_halo(a[p], mesh, halo_width), 6)
end

"""
    capture_snapshot(model; time_hours=0, halo_width=0, fields=nothing)

Capture air-mass and conservative tracer-storage fields from a `TransportModel`.
Without `fields`, return a full `SnapshotFrame`. Passing an `OutputFieldSpec`
returns a `SelectedSnapshotFrame` containing only requested layers and column
diagnostics, plus compensated Float64 totals for the selected tracers.

The result is CPU-resident and topology-native. For cubed-sphere states,
`halo_width` strips panel halos before writing. GPU-backed arrays are copied to
host memory by `Array(...)`.
"""
function capture_snapshot(model; time_hours::Real=0, halo_width::Integer=0, fields=nothing)
    fields === nothing || return _capture_selected_snapshot(model, fields; time_hours, halo_width)
    mesh = model.grid.horizontal
    air = _extract_for_output(model.state.air_mass, mesh; halo_width=halo_width)
    names = tracer_names(model.state)
    tracers = Dict{Symbol, typeof(air)}()
    for name in names
        tracers[name] = _extract_for_output(get_tracer(model.state, name), mesh;
                                            halo_width=halo_width)
    end
    return SnapshotFrame(Float64(time_hours), air, tracers,
                         _basis_symbol(mass_basis(model.state)))
end

function _frame_tracer_names(frame::AbstractSnapshotFrame)
    return sort!(collect(keys(frame.tracers)))
end

function _check_same_keys(frames)
    keys0 = _frame_tracer_names(first(frames))
    for (idx, frame) in enumerate(frames)
        keys_i = _frame_tracer_names(frame)
        keys_i == keys0 || throw(ArgumentError(
            "snapshot frame $(idx) has tracer keys $(keys_i), expected $(keys0)"))
        total_keys = sort!(collect(keys(frame.tracer_total_mass)))
        total_keys == keys0 || throw(ArgumentError(
            "snapshot frame $(idx) has tracer-total keys $(total_keys), expected $(keys0)"))
    end
    return keys0
end

function _check_mass_basis(frames, mass_basis_sym::Symbol)
    for (idx, frame) in enumerate(frames)
        frame.mass_basis == mass_basis_sym || throw(ArgumentError(
            "snapshot frame $(idx) has mass_basis=$(frame.mass_basis), " *
            "writer requested $(mass_basis_sym)"))
    end
    return nothing
end

function _check_frame_shapes(frames, mesh)
    _check_frame_shapes(first(frames), mesh)
    shape0 = _shape_signature(first(frames).air_mass)
    for (idx, frame) in enumerate(frames)
        _shape_signature(frame.air_mass) == shape0 || throw(DimensionMismatch(
            "snapshot frame $(idx) air_mass shape changed from $(shape0) to $(_shape_signature(frame.air_mass))"))
        for name in keys(first(frames).tracers)
            _shape_signature(frame.tracers[name]) == shape0 || throw(DimensionMismatch(
                "snapshot frame $(idx) tracer $(name) shape $(_shape_signature(frame.tracers[name])) " *
                "does not match air_mass shape $(shape0)"))
        end
    end
    return nothing
end

_shape_signature(a::AbstractArray) = size(a)
_shape_signature(a::NTuple{6, <:AbstractArray}) = ntuple(p -> size(a[p]), 6)

function _check_frame_shapes(frame::SnapshotFrame, mesh::LatLonMesh)
    size(frame.air_mass, 1) == nx(mesh) &&
        size(frame.air_mass, 2) == ny(mesh) ||
        throw(DimensionMismatch("LL air_mass shape $(size(frame.air_mass)) does not match mesh $(nx(mesh))×$(ny(mesh))"))
    return nothing
end

function _check_frame_shapes(frame::SnapshotFrame, mesh::ReducedGaussianMesh)
    size(frame.air_mass, 1) == ncells(mesh) ||
        throw(DimensionMismatch("RG air_mass shape $(size(frame.air_mass)) does not match $(ncells(mesh)) cells"))
    return nothing
end

function _check_frame_shapes(frame::SnapshotFrame, mesh::CubedSphereMesh)
    for p in 1:6
        size(frame.air_mass[p], 1) == mesh.Nc &&
            size(frame.air_mass[p], 2) == mesh.Nc ||
            throw(DimensionMismatch("CS air_mass panel $(p) shape $(size(frame.air_mass[p])) does not match C$(mesh.Nc) interior"))
    end
    return nothing
end
