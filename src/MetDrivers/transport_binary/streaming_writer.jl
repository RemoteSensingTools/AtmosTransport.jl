# Incremental structured and reduced-Gaussian transport-binary writer.

"""
    StreamingTransportBinaryWriter{FT}

Handle for incrementally writing transport-binary windows to disk without
holding all windows in memory.  Created by [`open_streaming_transport_binary`](@ref),
each window is written via [`write_streaming_window!`](@ref), and the file is
finalised by [`close_streaming_transport_binary!`](@ref).

Memory footprint: one `elems_per_window`-length pack buffer (`Vector{FT}`)
plus the open `IOStream`.  All other per-window data is owned by the caller.
"""
mutable struct StreamingTransportBinaryWriter{FT}
    io::IOStream
    path::String
    staging_path::String
    header::Dict{String, Any}
    payload_sections::Vector{Symbol}
    elems_per_window::Int
    header_bytes::Int
    expected_windows::Int
    written_windows::Int
    pack_buffer::Vector{FT}
    section_shapes::Vector{Vector{Vector{Int}}}
end

function _open_streaming_staging(path::AbstractString)
    parent = dirname(abspath(path))
    isdir(parent) || throw(ArgumentError(
        "streaming binary parent directory does not exist: $(parent)"))
    return mktemp(parent; cleanup=false)
end

function _streaming_section_shapes(window, payload_sections::Vector{Symbol})
    return [[collect(size(_transport_window_field(window, section)))]
            for section in payload_sections]
end

function _validate_streaming_window(writer::StreamingTransportBinaryWriter, window)
    base_sections = :am in writer.payload_sections ?
        Symbol[:m, :am, :bm, :cm, :ps] :
        Symbol[:m, :hflux, :cm, :ps]
    actual_sections = _transport_push_optional_sections!(base_sections, window)
    actual_sections == writer.payload_sections || throw(ArgumentError(
        "transport-binary window payload sections $(actual_sections) do not match " *
        "the writer contract $(writer.payload_sections)"))

    for (index, section) in pairs(writer.payload_sections)
        field = _transport_window_field(window, section)
        actual = collect(size(field))
        expected = only(writer.section_shapes[index])
        actual == expected || throw(DimensionMismatch(
            "transport-binary section $(section) has shape $(Tuple(actual)); expected $(Tuple(expected))"))
    end
    return nothing
end

"""
    open_streaming_transport_binary(path, grid::AtmosGrid{<:ReducedGaussianMesh},
                                    nwindow, sample_window; kwargs...)

Open a transport binary file for streaming (per-window) writes on a
reduced-Gaussian grid.

`sample_window` is a NamedTuple with the same keys as the windows that will
be written (e.g. `(m=..., hflux=..., cm=..., ps=...)`).  Its arrays must
have the correct sizes but their *values* are ignored — it is only used to
determine `payload_sections` and to validate dimensions.

Returns a [`StreamingTransportBinaryWriter`](@ref).
"""
function open_streaming_transport_binary(
        path::AbstractString,
        grid::AtmosGrid{<:ReducedGaussianMesh},
        nwindow::Int,
        sample_window;
        FT::Type{<:AbstractFloat} = floattype(grid),
        header_bytes::Int = 131072,
        dt_met_seconds::Real = 3600.0,
        half_dt_seconds::Real = dt_met_seconds / 2,
        steps_per_window::Integer = 2,
        source_flux_sampling::Symbol,
        air_mass_sampling::Symbol = :window_start_endpoint,
        flux_sampling::Symbol = :window_start_endpoint,
        flux_kind::Symbol = :substep_mass_amount,
        humidity_sampling::Symbol = :auto,
        delta_semantics::Symbol = :auto,
        mass_basis::Symbol = :moist,
        extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}())

    flux_kind === :substep_mass_amount || throw(ArgumentError(
        "reduced-Gaussian transport binaries require flux_kind=:substep_mass_amount"))

    mesh = grid.horizontal
    vc   = grid.vertical
    ncell   = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel  = nlevels(grid)
    basis_sym = _transport_basis_symbol(mass_basis)
    payload_sections = _transport_payload_sections(grid, sample_window)

    _transport_validate_reduced_window(sample_window, ncell, nface_h, nlevel, basis_sym)

    elems_per_window = sum(_transport_faceindexed_section_elements(ncell, nface_h, nlevel, s)
                           for s in payload_sections)

    # Build header — identical to the non-streaming write_transport_binary.
    header = _transport_common_header("reduced_gaussian", "FaceIndexed",
                                      ncell, nface_h, nlevel, nwindow, vc,
                                      payload_sections, elems_per_window;
                                      FT=FT,
                                      planet_radius=mesh.radius,
                                      header_bytes=header_bytes,
                                      dt_met_seconds=dt_met_seconds,
                                      half_dt_seconds=half_dt_seconds,
                                      steps_per_window=steps_per_window,
                                      mass_basis=basis_sym,
                                      source_flux_sampling=source_flux_sampling,
                                      air_mass_sampling=air_mass_sampling,
                                      flux_sampling=flux_sampling,
                                      flux_kind=flux_kind,
                                      humidity_sampling=humidity_sampling,
                                      delta_semantics=delta_semantics)
    merge!(header, Dict{String, Any}(
        "nlat"          => nrings(mesh),
        "latitudes"     => Float64.(mesh.latitudes),
        "nlon_per_ring" => mesh.nlon_per_ring,
        "n_dhflux" => (:dhflux in payload_sections) ?
            _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dhflux) : 0,
        "n_dcm" => (:dcm in payload_sections) ?
            _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dcm) : 0,
        "n_dm" => (:dm in payload_sections) ?
            _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dm) : 0,
    ))
    isempty(extra_header) || _merge_transport_extra_header!(header, extra_header)

    validate_transport_contract!(header)
    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad > 0 || error("transport binary header must leave room for a null terminator within header_bytes=$(header_bytes)")

    pack_buffer = Vector{FT}(undef, elems_per_window)
    section_shapes = _streaming_section_shapes(sample_window, payload_sections)
    staging_path, io = _open_streaming_staging(path)
    try
        write(io, header_json)
        write(io, zeros(UInt8, pad))
    catch
        close(io)
        rm(staging_path; force=true)
        rethrow()
    end

    return StreamingTransportBinaryWriter{FT}(
        io, String(path), staging_path, header, payload_sections, elems_per_window,
        header_bytes, nwindow, 0, pack_buffer, section_shapes)
end

function set_transport_header_steps_per_window_schedule!(
        header::AbstractDict,
        schedule::AbstractVector{<:Integer})
    steps = Int.(collect(schedule))
    isempty(steps) &&
        throw(ArgumentError("steps_per_window schedule must not be empty"))
    all(>=(1), steps) ||
        throw(ArgumentError("steps_per_window schedule must contain only positive integers; got $(steps)"))
    header["steps_per_window_by_window"] = steps
    header["steps_per_window"] = maximum(steps)
    header["time_step_schedule"] =
        _has_variable_step_schedule(steps) ? "per_window" : "constant"
    flux_kind = Symbol(replace(lowercase(String(get(header, "flux_kind",
        "substep_mass_amount"))), '-' => '_', ' ' => '_'))
    if flux_kind === :full_window_mass_amount
        header["poisson_balance_target_scale_by_window"] = fill(1.0, length(steps))
        header["poisson_balance_target_semantics"] = "forward_window_mass_difference"
        header["poisson_balance_target_scale"] = 1.0
    else
        header["poisson_balance_target_scale_by_window"] =
            [1.0 / (2 * s) for s in steps]
        header["poisson_balance_target_semantics"] =
            _has_variable_step_schedule(steps) ?
            "forward_window_mass_difference / (2 * steps_per_window_by_window[win])" :
            "forward_window_mass_difference / (2 * steps_per_window)"
        header["poisson_balance_target_scale"] = 1.0 / (2 * maximum(steps))
    end
    validate_transport_contract!(header)
    return header
end

function set_streaming_steps_per_window_schedule!(
        writer::StreamingTransportBinaryWriter,
        schedule::AbstractVector{<:Integer})
    length(schedule) == writer.expected_windows ||
        throw(ArgumentError("steps_per_window schedule length $(length(schedule)) " *
                            "does not match expected windows $(writer.expected_windows)"))
    set_transport_header_steps_per_window_schedule!(writer.header, schedule)
    return writer
end

function _rewrite_streaming_header!(writer::StreamingTransportBinaryWriter)
    validate_transport_contract!(writer.header)
    header_json = JSON3.write(writer.header)
    pad = writer.header_bytes - ncodeunits(header_json)
    pad > 0 || error("transport binary header must leave room for a null terminator within header_bytes=$(writer.header_bytes)")
    seek(writer.io, 0)
    write(writer.io, header_json)
    write(writer.io, zeros(UInt8, pad))
    flush(writer.io)
    return nothing
end

"""
    write_streaming_window!(writer, window)

Pack and write a single window to the streaming transport binary.
Windows must be written in order (1, 2, …, nwindow).
"""
function write_streaming_window!(writer::StreamingTransportBinaryWriter{FT},
                                  window) where FT
    writer.written_windows >= writer.expected_windows &&
        error("Already wrote $(writer.written_windows)/$(writer.expected_windows) windows")
    _validate_streaming_window(writer, window)
    _transport_pack_window!(writer.pack_buffer, 0, window, writer.payload_sections)
    write(writer.io, writer.pack_buffer)
    writer.written_windows += 1
    return nothing
end

"""
    close_streaming_transport_binary!(writer) -> String

Flush and close the streaming transport binary. Returns the file path. An
incomplete stream is closed and rejected without publishing a final header.
"""
function close_streaming_transport_binary!(writer::StreamingTransportBinaryWriter)
    if writer.written_windows != writer.expected_windows
        isopen(writer.io) && close(writer.io)
        rm(writer.staging_path; force = true)
        throw(ArgumentError(
            "Streaming binary is incomplete: expected $(writer.expected_windows) windows, " *
            "wrote $(writer.written_windows); refusing to finalise $(writer.path)"))
    end
    try
        _rewrite_streaming_header!(writer)
    catch
        isopen(writer.io) && close(writer.io)
        rm(writer.staging_path; force = true)
        rethrow()
    end
    isopen(writer.io) && close(writer.io)
    try
        mv(writer.staging_path, writer.path; force=true)
    catch
        rm(writer.staging_path; force=true)
        rethrow()
    end
    return writer.path
end
