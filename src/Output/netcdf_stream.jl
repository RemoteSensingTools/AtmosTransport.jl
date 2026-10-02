"""
    NetCDFSnapshotStream(path, grid; mass_basis=:dry, options, fields)

Internal single-file runtime sink. Each `append_snapshot!` writes and flushes
one record, retaining only a file handle, schema metadata and the RG raster map.
The file is created on the first append and uses an unlimited time dimension.
A stream owns its output path, must be used serially, and must be closed with
`close`. Reopening an existing run is not supported. Invalid frames are rejected before opening the file. After an
I/O failure the stream cannot be reused, and the file may contain a partial
last record; `completed_snapshots` records the last successful append.
"""
mutable struct NetCDFSnapshotStream{G,O,F}
    path::String
    grid::G
    mass_basis::Symbol
    options::O
    fields::F
    count::Int
    last_time::Float64
    nlevel::Int
    tracer_keys::Vector{Symbol}
    shape::Any
    geometry::Any
    dataset::Union{Nothing,NCDataset}
    closed::Bool
    failed::Bool
end

function NetCDFSnapshotStream(path::AbstractString, grid;
                              mass_basis::Symbol=:dry,
                              options::SnapshotWriteOptions=SnapshotWriteOptions(),
                              fields::OutputFieldSpec=output_field_spec())
    return NetCDFSnapshotStream(expand_data_path(String(path)), grid, mass_basis,
                                options, deepcopy(fields), 0, -Inf, 0, Symbol[],
                                nothing, nothing, nothing, false, false)
end

function Base.close(stream::NetCDFSnapshotStream)
    stream.closed && return nothing
    try
        dataset = stream.dataset
        dataset === nothing || with_netcdf_lock(() -> close(dataset))
    finally
        stream.dataset = nothing
        stream.closed = true
    end
    return nothing
end

function _validate_stream_frame(stream, frame)
    stream.failed && throw(ArgumentError("cannot append after a NetCDF write failure"))
    stream.closed && throw(ArgumentError("cannot append to a closed NetCDF stream"))
    mesh = stream.grid.horizontal
    _validate_snapshot_inputs([frame], mesh, stream.mass_basis)
    isfinite(frame.time_hours) && frame.time_hours > stream.last_time ||
        throw(ArgumentError("snapshot times must be finite and strictly increasing"))
    keys = _frame_tracer_names(frame)
    Nz = _nlevel(frame, mesh)
    if stream.count > 0
        keys == stream.tracer_keys || throw(ArgumentError("snapshot tracer keys changed"))
        Nz == stream.nlevel || throw(DimensionMismatch("snapshot vertical size changed"))
        _shape_signature(frame.air_mass) == stream.shape ||
            throw(DimensionMismatch("snapshot stored-layer shape changed"))
    end
    fields = stream.fields
    names = _select_tracer_keys(keys, fields)
    levels = (fields.air_mass || fields.air_mass_per_area) ?
             _layer_indices(fields.air_mass_layers, fields, Nz) : Int[]
    for name in names
        append!(levels, _layer_indices(tracer_fields(fields, name).layers, fields, Nz))
    end
    if frame isa SelectedSnapshotFrame
        _stored_indices(frame, sort!(unique!(levels)))
        needs_air = fields.column_air_mass_per_area ||
                    any(name -> tracer_fields(fields, name).column_mean, names)
        needs_air && frame.column_air === nothing &&
            throw(ArgumentError("column air mass was not captured"))
        for name in names
            tf = tracer_fields(fields, name)
            (tf.column_mean || tf.column_mass_per_area) &&
                !haskey(frame.column_tracers, name) &&
                throw(ArgumentError("column tracer $(name) was not captured"))
        end
    end
    return keys, names, Nz
end

function append_snapshot!(stream::NetCDFSnapshotStream, frame::AbstractSnapshotFrame)
    keys, names, Nz = _validate_stream_frame(stream, frame)
    mesh = stream.grid.horizontal
    first_record = stream.count == 0
    lock(_NETCDF_IO_LOCK)   # shared with observation appends and the daily write task
    try
        if first_record
            _ensure_parent_dir(stream.path)
            stream.dataset = _create_netcdf_dataset(stream.path)
        end
        ds = stream.dataset
        if first_record
            _define_common_attributes!(ds, mesh, [frame], stream.mass_basis;
                                       options=stream.options)
            ds.attrib["output_fields"] = _fields_string(stream.fields, names)
            ds.attrib["completed_snapshots"] = 0
            stream.geometry = _define_geometry!(ds, mesh, Nz, nothing)
        end
        _write_tracer_total_mass!(ds, [frame], names, stream.mass_basis; time_offset=stream.count)
        _write_snapshot_payload!(ds, mesh, [frame], names, stream.geometry,
                                 stream.mass_basis, stream.options, stream.fields;
                                 time_offset=stream.count)
        ds["time"][stream.count + 1] = frame.time_hours
        # Flush payloads before publishing the completed-record count.
        NCDatasets.sync(ds)
        ds.attrib["completed_snapshots"] = stream.count + 1
        ds.attrib["history"] = replace(String(ds.attrib["history"]),
            r"with \d+ frame\(s\)$" => "with $(stream.count + 1) frame(s)")
        NCDatasets.sync(ds)
    catch
        stream.failed = true
        try
            close(stream)
        catch
            # Preserve the original write error when cleanup also fails.
        end
        rethrow()
    finally
        unlock(_NETCDF_IO_LOCK)
    end
    stream.count += 1
    stream.last_time = frame.time_hours
    stream.nlevel = Nz
    stream.tracer_keys = keys
    stream.shape = _shape_signature(frame.air_mass)
    return stream.path
end
