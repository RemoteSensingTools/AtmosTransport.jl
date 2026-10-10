function _var_attrib(; units::AbstractString,
                     long_name::AbstractString,
                     coordinates::Union{Nothing, AbstractString}=nothing,
                     grid_mapping::Union{Nothing, AbstractString}=nothing,
                     cell_methods::Union{Nothing, AbstractString}=nothing,
                     extra=Dict{String, Any}())
    attrs = Dict{String, Any}("units" => units, "long_name" => long_name)
    coordinates === nothing || (attrs["coordinates"] = coordinates)
    grid_mapping === nothing || (attrs["grid_mapping"] = grid_mapping)
    cell_methods === nothing || (attrs["cell_methods"] = cell_methods)
    for (k, v) in extra
        attrs[k] = v
    end
    return attrs
end

_tracer_units(mass_basis_sym::Symbol) =
    mass_basis_sym === :dry ? "mol mol-1 dry" : "mol mol-1"

function _payload_deflate_level(options::SnapshotWriteOptions)
    return options.deflate_level == 0 ? nothing : options.deflate_level
end

function _payload_shuffle(options::SnapshotWriteOptions)
    return options.deflate_level > 0 && options.shuffle
end

"""
    PAYLOAD_FILL_VALUE

Sentinel for masked / invalid payload cells in NetCDF snapshots. Matches the
GEOS-Chem convention (`Met_AD._FillValue == 1.0e15`) so Panoply / ncview / IDV
all mask the same out-of-range cells with the same value. Float32 outputs
truncate to `Float32(1e15)`, which is well below `floatmax(Float32) ≈ 3.4e38`
and safely outside any physical mass or mixing-ratio range.
"""
const PAYLOAD_FILL_VALUE = 1.0e15

@inline _payload_fill_value(::Type{Float32}) = Float32(PAYLOAD_FILL_VALUE)
@inline _payload_fill_value(::Type{Float64}) = Float64(PAYLOAD_FILL_VALUE)

function _def_payload_var(ds, name::AbstractString, ::Type{T}, dims;
                          attrib, options::SnapshotWriteOptions{T}, reuse::Bool=false) where
        {T <: AbstractFloat}
    # Append calls reuse the schema defined by the first snapshot.
    if haskey(ds, name)
        reuse || throw(ArgumentError("duplicate output variable name: $(name)"))
        return ds[name]
    end
    # Inject the FillValue sentinel as both `_FillValue` (CF) and
    # `missing_value` (GEOS-Chem / GrADS / older readers) so every tool
    # masks invalid cells correctly. `defVar(..., fillvalue=...)` writes
    # the storage default at file-creation time so uninitialized cells
    # are masked even if the writer never reaches them.
    fv = _payload_fill_value(T)
    attrs = copy(attrib)
    attrs["missing_value"] = fv
    return defVar(ds, name, T, dims;
                  attrib = attrs,
                  fillvalue = fv,
                  deflatelevel = _payload_deflate_level(options),
                  shuffle = _payload_shuffle(options))
end

# Cubed-sphere layer fields are stacked as (Xdim, Ydim, nf, lev) straight into
# a reused buffer of the output type `T`, reading only the requested stored
# layers. Values match converting to Float64 first and then to `T`, since that
# intermediate is exact for Float32/Float64 input.
_cs_layer_buffer!(buffers::Dict{Int, Array{T, 4}}, Nc::Int, nlayer::Int) where T =
    get!(() -> Array{T}(undef, Nc, Nc, 6, nlayer), buffers, nlayer)

function _check_cs_interior(buf, panels::NTuple{6})
    all(q -> size(q, 1) == size(buf, 1) && size(q, 2) == size(buf, 2), panels) ||
        throw(DimensionMismatch("snapshot panels $(size(panels[1])) are not the " *
                                "$(size(buf, 1)) × $(size(buf, 2)) panel interior"))
    return nothing
end

function _cs_stack_layers!(buf::Array{T, 4}, panels::NTuple{6}, positions) where T
    _check_cs_interior(buf, panels)
    @inbounds for (kk, k) in enumerate(positions), p in 1:6, j in axes(buf, 2), i in axes(buf, 1)
        buf[i, j, p, kk] = T(panels[p][i, j, k])
    end
    return buf
end

# Layer mass per area, as `layer_mass_per_area`.
function _cs_stack_mass_per_area!(buf::Array{T, 4}, panels::NTuple{6}, positions,
                                  mesh::CubedSphereMesh) where T
    _check_cs_interior(buf, panels)
    @inbounds for (kk, k) in enumerate(positions), p in 1:6, j in axes(buf, 2), i in axes(buf, 1)
        buf[i, j, p, kk] = T(Float64(panels[p][i, j, k]) / Float64(cell_area(mesh, i, j)))
    end
    return buf
end

# Mixing ratio, as `mixing_ratio_field`: Float64 quotient, NaN where the air
# mass is not positive, rounded once to `T`.
function _cs_stack_vmr!(buf::Array{T, 4}, air::NTuple{6}, tracer::NTuple{6}, positions) where T
    _check_cs_interior(buf, air)
    _check_cs_interior(buf, tracer)
    @inbounds for (kk, k) in enumerate(positions), p in 1:6, j in axes(buf, 2), i in axes(buf, 1)
        m = Float64(air[p][i, j, k])
        buf[i, j, p, kk] = T(m > 0 ? Float64(tracer[p][i, j, k]) / m : NaN)
    end
    return buf
end

function _cs_stack2(panels::NTuple{6, <:AbstractArray})
    Nc1, Nc2 = size(panels[1])
    out = Array{Float64}(undef, Nc1, Nc2, 6)
    @inbounds for p in 1:6
        out[:, :, p] = Float64.(panels[p])
    end
    return out
end

function _rg_rasterize(native::AbstractVector, nn_map::AbstractMatrix{Int})
    out = Array{Float64}(undef, size(nn_map))
    @inbounds for idx in eachindex(nn_map)
        out[idx] = Float64(native[nn_map[idx]])
    end
    return out
end

function _select_tracer_keys(all_keys::AbstractVector{Symbol}, fields::OutputFieldSpec)
    requested = fields.tracers
    requested === nothing && return all_keys
    available = Set(all_keys)
    for name in requested
        name in available || throw(ArgumentError(
            "[output.fields].tracers requested $(name), but snapshot carries $(all_keys)"))
    end
    return copy(requested)
end

function _validate_selected_levels(fields::OutputFieldSpec, Nz::Integer)
    levels = fields.selected_levels
    isempty(levels) &&
        throw(ArgumentError("[output.fields].levels must be non-empty when layers = \"selected\""))
    last(levels) <= Nz || throw(ArgumentError(
        "[output.fields].levels contains $(last(levels)), but snapshots have only $(Nz) levels"))
    return levels
end

_layer_indices(::FullLayerSelection, fields::OutputFieldSpec, Nz::Integer) = collect(1:Int(Nz))
_layer_indices(::NoLayerSelection, fields::OutputFieldSpec, Nz::Integer) = Int[]
_layer_indices(::SelectedLayerSelection, fields::OutputFieldSpec, Nz::Integer) =
    _validate_selected_levels(fields, Nz)

function _ensure_selected_lev!(ds, fields::OutputFieldSpec, Nz::Integer)
    levels = _validate_selected_levels(fields, Nz)
    if !haskey(ds.dim, "lev_selected")
        defDim(ds, "lev_selected", length(levels))
        v = defVar(ds, "lev_selected", Float64, ("lev_selected",),
                   attrib = Dict("units" => "1",
                                 "long_name" => "selected model level",
                                 "standard_name" => "model_level_number",
                                 "axis" => "Z",
                                 "positive" => "down",
                                 "selection" => "configured [output.fields].levels"))
        v[:] = Float64.(levels)
    elseif size(ds["lev_selected"], 1) != length(levels)
        throw(ArgumentError("all selected-layer outputs must use the same [output.fields].levels"))
    end
    return levels
end

function _layer_dims(base_dims::Tuple, ::FullLayerSelection)
    return (base_dims..., "lev", "time")
end

function _layer_dims(base_dims::Tuple, ::SelectedLayerSelection)
    return (base_dims..., "lev_selected", "time")
end

_select_levels(a::AbstractArray{<:Any, 3}, idx::AbstractVector{Int}) =
    a[:, :, idx]

_select_levels(a::AbstractArray{<:Any, 2}, idx::AbstractVector{Int}) =
    a[:, idx]

function _select_levels(a::AbstractArray, ::AbstractVector{Int})
    throw(ArgumentError("selected layer output expects 2D or 3D arrays, got ndims=$(ndims(a))"))
end

_select_levels(a::NTuple{6, <:AbstractArray}, idx::AbstractVector{Int}) =
    ntuple(p -> _select_levels(a[p], idx), 6)

function _fields_string(fields::OutputFieldSpec, tracer_keys)
    tracer_label = fields.tracers === nothing ? "all" : join(String.(fields.tracers), ",")
    written = join(String.(tracer_keys), ",")
    return "tracers=$(tracer_label); tracer_layers=$(layer_selection_label(fields.default_tracer.layers)); " *
           "levels=$(fields.selected_levels); column_mean=$(fields.default_tracer.column_mean); " *
           "column_mass_per_area=$(fields.default_tracer.column_mass_per_area); " *
           "air_mass_layers=$(layer_selection_label(fields.air_mass_layers)); air_mass=$(fields.air_mass); " *
           "air_mass_per_area=$(fields.air_mass_per_area); " *
           "column_air_mass_per_area=$(fields.column_air_mass_per_area); " *
           "written_tracers=$(written)"
end

function _validate_snapshot_inputs(frames, mesh, mass_basis_sym::Symbol)
    isempty(frames) && throw(ArgumentError("write_snapshot_netcdf requires at least one SnapshotFrame"))
    _check_same_keys(frames)
    _check_mass_basis(frames, mass_basis_sym)
    _check_frame_shapes(frames, mesh)
    return nothing
end

function _write_tracer_total_mass!(ds, frames, tracer_keys, mass_basis_sym::Symbol; time_offset::Integer=0)
    for name in tracer_keys
        label = String(name)
        variable_name = "$(label)_total_mass"
        variable = time_offset > 0 ? ds[variable_name] : defVar(
            ds, "$(label)_total_mass", Float64, ("time",),
            attrib = _var_attrib(
                units = "kg",
                long_name = "global total conservative $(label) tracer storage",
                extra = Dict(
                    "description" =>
                        "Compensated Float64 sum captured before any spatial output precision conversion",
                    "storage_contract" => "mixing_ratio_times_carrier_air_mass",
                    "mass_basis" => String(mass_basis_sym),
                ),
            ),
        )
        variable[time_offset+1:time_offset+length(frames)] = [frame.tracer_total_mass[name] for frame in frames]
    end
    return nothing
end

# Thread-safe HDF5 builds use thread-local error handlers. NetCDF initializes the
# main thread quietly, but a first `nc_create` on a `Threads.@spawn` worker can
# print an alarming H5Fis_accessible stack while probing a path that does not
# exist yet. Silence only that expected constructor probe, then restore the
# worker's handler before defining variables or writing payloads so genuine
# HDF5/NetCDF failures remain visible and continue to throw normally.
function _create_netcdf_dataset(path::AbstractString)
    old_func = Ref{Ptr{Cvoid}}(C_NULL)
    old_data = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall((:H5Eget_auto2, HDF5_jll.libhdf5), Cint,
                   (Int64, Ref{Ptr{Cvoid}}, Ref{Ptr{Cvoid}}),
                   0, old_func, old_data)
    status < 0 && return NCDataset(path, "c")

    ccall((:H5Eset_auto2, HDF5_jll.libhdf5), Cint,
          (Int64, Ptr{Cvoid}, Ptr{Cvoid}),
          0, C_NULL, C_NULL)
    try
        return NCDataset(path, "c")
    finally
        ccall((:H5Eset_auto2, HDF5_jll.libhdf5), Cint,
              (Int64, Ptr{Cvoid}, Ptr{Cvoid}),
              0, old_func[], old_data[])
    end
end

"""
    write_snapshot_netcdf(path, frames, grid; mass_basis=:dry, options=SnapshotWriteOptions(),
                          fields=output_field_spec())

Write topology-aware runtime snapshots to NetCDF.

The output contains full per-level VMR fields, stored air mass, layer
mass-per-area diagnostics, column air mass per area, and tracer column means.
Reduced-Gaussian files also carry a legacy lon/lat raster view for quick plots.
Cubed-sphere files carry panel lon/lat coordinates and a `cubed_sphere`
grid-mapping variable modeled after GEOS-Chem diagnostics.
"""
function write_snapshot_netcdf(path::AbstractString,
                               frames::AbstractVector{<:AbstractSnapshotFrame},
                               grid::AtmosGrid;
                               mass_basis::Symbol=:dry,
                               options::SnapshotWriteOptions=SnapshotWriteOptions(),
                               fields::OutputFieldSpec=output_field_spec())
    expanded = expand_data_path(String(path))
    mesh = grid.horizontal
    _validate_snapshot_inputs(frames, mesh, mass_basis)
    Nz = _nlevel(first(frames), mesh)
    times = [frame.time_hours for frame in frames]
    tracer_keys = _select_tracer_keys(_check_same_keys(frames), fields)

    _ensure_parent_dir(expanded)
    # netcdf-c is not thread-safe; daily writes run on a background task.
    with_netcdf_lock() do
        ds = _create_netcdf_dataset(expanded)
        try
            _define_common_attributes!(ds, mesh, frames, mass_basis; options = options)
            ds.attrib["output_fields"] = _fields_string(fields, tracer_keys)
            geometry = _define_geometry!(ds, mesh, Nz, times)
            _write_tracer_total_mass!(ds, frames, tracer_keys, mass_basis)
            _write_snapshot_payload!(ds, mesh, frames, tracer_keys, geometry,
                                     mass_basis, options, fields)
        finally
            close(ds)
        end
    end
    @info @sprintf("Saved snapshots: %s (%d frame(s), %s, mass_basis=%s)",
                   expanded, length(frames), summary(mesh), mass_basis)
    return expanded
end

function _write_snapshot_payload!(ds, mesh::LatLonMesh, frames, tracer_keys,
                                  geometry, mass_basis_sym::Symbol,
                                  options::SnapshotWriteOptions,
                                  fields::OutputFieldSpec; time_offset::Integer=0)
    T = options.float_type
    Nz = _nlevel(first(frames), mesh)
    air_idx = _layer_indices(fields.air_mass_layers, fields, Nz)
    air_dims = isempty(air_idx) ? () : _layer_dims(("lon", "lat"), fields.air_mass_layers)
    fields.air_mass_layers isa SelectedLayerSelection && _ensure_selected_lev!(ds, fields, Nz)

    air = fields.air_mass && !isempty(air_idx) ?
          _def_payload_var(ds, "air_mass", T, air_dims,
                           attrib = _var_attrib(units = "kg",
                                                long_name = "stored air mass",
                                                coordinates = "lon lat"),
                           options = options, reuse = time_offset > 0) : nothing
    air_area = fields.air_mass_per_area && !isempty(air_idx) ?
               _def_payload_var(ds, "air_mass_per_area", T, air_dims,
                                attrib = _var_attrib(units = "kg m-2",
                                                     long_name = "stored layer air mass per area",
                                                     coordinates = "lon lat"),
                                options = options, reuse = time_offset > 0) : nothing
    col_air = fields.column_air_mass_per_area ?
              _def_payload_var(ds, "column_air_mass_per_area", T, ("lon", "lat", "time"),
                               attrib = _var_attrib(units = "kg m-2",
                                                    long_name = "column air mass per area",
                                                    coordinates = "lon lat",
                                                    cell_methods = "lev: sum"),
                               options = options, reuse = time_offset > 0) : nothing

    tracer_vars = Dict{Symbol, Any}()
    tracer_cm_vars = Dict{Symbol, Any}()
    tracer_col_vars = Dict{Symbol, Any}()
    tracer_layer_idx = Dict{Symbol, Vector{Int}}()
    for name in tracer_keys
        s = String(name)
        tf = tracer_fields(fields, name)
        idx = _layer_indices(tf.layers, fields, Nz)
        tracer_layer_idx[name] = idx
        tf.layers isa SelectedLayerSelection && _ensure_selected_lev!(ds, fields, Nz)
        if !isempty(idx)
            tracer_vars[name] = _def_payload_var(ds, s, T, _layer_dims(("lon", "lat"), tf.layers),
                                                 attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                      long_name = "per-layer $(s) mixing ratio",
                                                                      coordinates = "lon lat"),
                                                 options = options, reuse = time_offset > 0)
        end
        if tf.column_mean
            tracer_cm_vars[name] = _def_payload_var(ds, "$(s)_column_mean", T, ("lon", "lat", "time"),
                                                    attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                         long_name = "air-mass-weighted column-mean $(s) mixing ratio",
                                                                         coordinates = "lon lat",
                                                                         cell_methods = "lev: mean"),
                                                    options = options, reuse = time_offset > 0)
        end
        if tf.column_mass_per_area
            tracer_col_vars[name] = _def_payload_var(ds, "$(s)_column_mass_per_area", T, ("lon", "lat", "time"),
                                                     attrib = _var_attrib(units = "kg m-2",
                                                                          long_name = "column model tracer mass per area for $(s)",
                                                                          coordinates = "lon lat",
                                                                          cell_methods = "lev: sum",
                                                                          extra = Dict("description" =>
                                                                              "Sum of model tracer mass divided by horizontal cell area; no molecular-weight conversion is applied.")),
                                                     options = options, reuse = time_offset > 0)
        end
    end

    for (index, frame) in enumerate(frames)
        t = time_offset + index
        air === nothing || (air[:, :, :, t] = T.(_air_layers(frame, air_idx)))
        air_area === nothing || (air_area[:, :, :, t] =
            T.(layer_mass_per_area(_air_layers(frame, air_idx), mesh)))
        col_air === nothing || (col_air[:, :, t] = T.(_horizontal_per_area(_frame_column_air(frame), mesh)))
        for name in tracer_keys
            if haskey(tracer_vars, name)
                tracer_vars[name][:, :, :, t] =
                    T.(_frame_vmr(frame, name, tracer_layer_idx[name]))
            end
            haskey(tracer_cm_vars, name) &&
                (tracer_cm_vars[name][:, :, t] =
                    T.(_frame_column_mean(frame, name)))
            haskey(tracer_col_vars, name) &&
                (tracer_col_vars[name][:, :, t] =
                    T.(_horizontal_per_area(_frame_column_tracer(frame, name), mesh)))
        end
    end
    return nothing
end

function _write_snapshot_payload!(ds, mesh::ReducedGaussianMesh, frames, tracer_keys,
                                  geometry, mass_basis_sym::Symbol,
                                  options::SnapshotWriteOptions,
                                  fields::OutputFieldSpec; time_offset::Integer=0)
    T = options.float_type
    Nz = _nlevel(first(frames), mesh)
    air_idx = _layer_indices(fields.air_mass_layers, fields, Nz)
    air_dims = isempty(air_idx) ? () : _layer_dims(("cell",), fields.air_mass_layers)
    fields.air_mass_layers isa SelectedLayerSelection && _ensure_selected_lev!(ds, fields, Nz)

    air = fields.air_mass && !isempty(air_idx) ?
          _def_payload_var(ds, "air_mass", T, air_dims,
                           attrib = _var_attrib(units = "kg",
                                                long_name = "stored air mass",
                                                coordinates = "cell_lon cell_lat"),
                           options = options, reuse = time_offset > 0) : nothing
    air_area = fields.air_mass_per_area && !isempty(air_idx) ?
               _def_payload_var(ds, "air_mass_per_area", T, air_dims,
                                attrib = _var_attrib(units = "kg m-2",
                                                     long_name = "stored layer air mass per area",
                                                     coordinates = "cell_lon cell_lat"),
                                options = options, reuse = time_offset > 0) : nothing
    col_air = fields.column_air_mass_per_area ?
              _def_payload_var(ds, "column_air_mass_per_area", T, ("cell", "time"),
                               attrib = _var_attrib(units = "kg m-2",
                                                    long_name = "column air mass per area",
                                                    coordinates = "cell_lon cell_lat",
                                                    cell_methods = "lev: sum"),
                               options = options, reuse = time_offset > 0) : nothing

    tracer_vars = Dict{Symbol, Any}()
    tracer_cm_native_vars = Dict{Symbol, Any}()
    tracer_cm_raster_vars = Dict{Symbol, Any}()
    tracer_col_vars = Dict{Symbol, Any}()
    tracer_layer_idx = Dict{Symbol, Vector{Int}}()
    for name in tracer_keys
        s = String(name)
        tf = tracer_fields(fields, name)
        idx = _layer_indices(tf.layers, fields, Nz)
        tracer_layer_idx[name] = idx
        tf.layers isa SelectedLayerSelection && _ensure_selected_lev!(ds, fields, Nz)
        if !isempty(idx)
            tracer_vars[name] = _def_payload_var(ds, s, T, _layer_dims(("cell",), tf.layers),
                                                 attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                      long_name = "native per-layer $(s) mixing ratio",
                                                                      coordinates = "cell_lon cell_lat"),
                                                 options = options, reuse = time_offset > 0)
        end
        if tf.column_mean
            tracer_cm_native_vars[name] = _def_payload_var(ds, "$(s)_column_mean_native", T, ("cell", "time"),
                                                           attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                                long_name = "native air-mass-weighted column-mean $(s) mixing ratio",
                                                                                coordinates = "cell_lon cell_lat",
                                                                                cell_methods = "lev: mean"),
                                                           options = options, reuse = time_offset > 0)
            tracer_cm_raster_vars[name] = _def_payload_var(ds, "$(s)_column_mean", T, ("lon", "lat", "time"),
                                                           attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                                long_name = "diagnostic lon-lat raster column-mean $(s) mixing ratio",
                                                                                coordinates = "lon lat",
                                                                                cell_methods = "lev: mean",
                                                                                extra = Dict("regridding" => ds.attrib["regridding"])),
                                                           options = options, reuse = time_offset > 0)
        end
        if tf.column_mass_per_area
            tracer_col_vars[name] = _def_payload_var(ds, "$(s)_column_mass_per_area", T, ("cell", "time"),
                                                     attrib = _var_attrib(units = "kg m-2",
                                                                          long_name = "native column model tracer mass per area for $(s)",
                                                                          coordinates = "cell_lon cell_lat",
                                                                          cell_methods = "lev: sum",
                                                                          extra = Dict("description" =>
                                                                              "Sum of model tracer mass divided by horizontal cell area; no molecular-weight conversion is applied.")),
                                                     options = options, reuse = time_offset > 0)
        end
    end

    nn_map = geometry.nn_map
    for (index, frame) in enumerate(frames)
        t = time_offset + index
        air === nothing || (air[:, :, t] = T.(_air_layers(frame, air_idx)))
        air_area === nothing || (air_area[:, :, t] =
            T.(layer_mass_per_area(_air_layers(frame, air_idx), mesh)))
        col_air === nothing || (col_air[:, t] = T.(_horizontal_per_area(_frame_column_air(frame), mesh)))
        for name in tracer_keys
            if haskey(tracer_vars, name)
                tracer_vars[name][:, :, t] =
                    T.(_frame_vmr(frame, name, tracer_layer_idx[name]))
            end
            if haskey(tracer_cm_native_vars, name)
                cm = _frame_column_mean(frame, name)
                tracer_cm_native_vars[name][:, t] = T.(cm)
                tracer_cm_raster_vars[name][:, :, t] = T.(_rg_rasterize(cm, nn_map))
            end
            haskey(tracer_col_vars, name) &&
                (tracer_col_vars[name][:, t] =
                    T.(_horizontal_per_area(_frame_column_tracer(frame, name), mesh)))
        end
    end
    return nothing
end

function _write_snapshot_payload!(ds, mesh::CubedSphereMesh, frames, tracer_keys,
                                  geometry, mass_basis_sym::Symbol,
                                  options::SnapshotWriteOptions,
                                  fields::OutputFieldSpec; time_offset::Integer=0)
    T = options.float_type
    dims4 = ("Xdim", "Ydim", "nf", "time")
    coord = "lons lats"
    Nz = _nlevel(first(frames), mesh)
    air_idx = _layer_indices(fields.air_mass_layers, fields, Nz)
    air_dims = isempty(air_idx) ? () : _layer_dims(("Xdim", "Ydim", "nf"), fields.air_mass_layers)
    fields.air_mass_layers isa SelectedLayerSelection && _ensure_selected_lev!(ds, fields, Nz)

    air = fields.air_mass && !isempty(air_idx) ?
          _def_payload_var(ds, "air_mass", T, air_dims,
                           attrib = _var_attrib(units = "kg",
                                                long_name = "stored air mass",
                                                coordinates = coord,
                                                grid_mapping = "cubed_sphere"),
                           options = options, reuse = time_offset > 0) : nothing
    air_area = fields.air_mass_per_area && !isempty(air_idx) ?
               _def_payload_var(ds, "air_mass_per_area", T, air_dims,
                                attrib = _var_attrib(units = "kg m-2",
                                                     long_name = "stored layer air mass per area",
                                                     coordinates = coord,
                                                     grid_mapping = "cubed_sphere"),
                                options = options, reuse = time_offset > 0) : nothing
    col_air = fields.column_air_mass_per_area ?
              _def_payload_var(ds, "column_air_mass_per_area", T, dims4,
                               attrib = _var_attrib(units = "kg m-2",
                                                    long_name = "column air mass per area",
                                                    coordinates = coord,
                                                    grid_mapping = "cubed_sphere",
                                                    cell_methods = "lev: sum"),
                               options = options, reuse = time_offset > 0) : nothing

    tracer_vars = Dict{Symbol, Any}()
    tracer_cm_vars = Dict{Symbol, Any}()
    tracer_col_vars = Dict{Symbol, Any}()
    tracer_layer_idx = Dict{Symbol, Vector{Int}}()
    for name in tracer_keys
        s = String(name)
        tf = tracer_fields(fields, name)
        idx = _layer_indices(tf.layers, fields, Nz)
        tracer_layer_idx[name] = idx
        tf.layers isa SelectedLayerSelection && _ensure_selected_lev!(ds, fields, Nz)
        if !isempty(idx)
            tracer_vars[name] = _def_payload_var(ds, s, T,
                                                 _layer_dims(("Xdim", "Ydim", "nf"), tf.layers),
                                                 attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                      long_name = "per-layer $(s) mixing ratio",
                                                                      coordinates = coord,
                                                                      grid_mapping = "cubed_sphere"),
                                                 options = options, reuse = time_offset > 0)
        end
        if tf.column_mean
            tracer_cm_vars[name] = _def_payload_var(ds, "$(s)_column_mean", T, dims4,
                                                    attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                         long_name = "air-mass-weighted column-mean $(s) mixing ratio",
                                                                         coordinates = coord,
                                                                         grid_mapping = "cubed_sphere",
                                                                         cell_methods = "lev: mean"),
                                                    options = options, reuse = time_offset > 0)
        end
        if tf.column_mass_per_area
            tracer_col_vars[name] = _def_payload_var(ds, "$(s)_column_mass_per_area", T, dims4,
                                                     attrib = _var_attrib(units = "kg m-2",
                                                                          long_name = "column model tracer mass per area for $(s)",
                                                                          coordinates = coord,
                                                                          grid_mapping = "cubed_sphere",
                                                                          cell_methods = "lev: sum",
                                                                          extra = Dict("description" =>
                                                                              "Sum of model tracer mass divided by horizontal cell area; no molecular-weight conversion is applied.")),
                                                     options = options, reuse = time_offset > 0)
        end
    end

    buffers = Dict{Int, Array{T, 4}}()
    for (index, frame) in enumerate(frames)
        t = time_offset + index
        # Selected capture stores the configured air-output layers only when an
        # air output is enabled (tracer layers are always stored, for the mixing
        # ratios), so the air positions are looked up only in that case.
        if air !== nothing || air_area !== nothing
            air_pos = _layer_positions(frame, air_idx)
            air === nothing || (air[:, :, :, :, t] = _cs_stack_layers!(
                _cs_layer_buffer!(buffers, mesh.Nc, length(air_idx)), frame.air_mass, air_pos))
            air_area === nothing || (air_area[:, :, :, :, t] = _cs_stack_mass_per_area!(
                _cs_layer_buffer!(buffers, mesh.Nc, length(air_idx)), frame.air_mass, air_pos, mesh))
        end
        col_air === nothing || (col_air[:, :, :, t] =
            T.(_cs_stack2(_horizontal_per_area(_frame_column_air(frame), mesh))))
        for name in tracer_keys
            if haskey(tracer_vars, name)
                idx = tracer_layer_idx[name]
                tracer_vars[name][:, :, :, :, t] = _cs_stack_vmr!(
                    _cs_layer_buffer!(buffers, mesh.Nc, length(idx)), frame.air_mass,
                    frame.tracers[name], _layer_positions(frame, idx))
            end
            haskey(tracer_cm_vars, name) &&
                (tracer_cm_vars[name][:, :, :, t] =
                    T.(_cs_stack2(_frame_column_mean(frame, name))))
            haskey(tracer_col_vars, name) &&
                (tracer_col_vars[name][:, :, :, t] =
                    T.(_cs_stack2(_horizontal_per_area(_frame_column_tracer(frame, name), mesh))))
        end
    end
    return nothing
end
