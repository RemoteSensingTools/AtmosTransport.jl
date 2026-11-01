# Writer adapters.

# These wrappers put the existing topology-specific binary writers behind the
# typed `AbstractBinaryWriter{G, FT, Basis}` contract without changing the byte
# layout or current production call sites.

mutable struct LatLonBinaryWriter{FT,
                                  Basis<:AbstractMassBasis,
                                  IO_t<:IO,
                                  Settings,
                                  Merged,
                                  Last} <: AbstractBinaryWriter{LatLonTargetGeometry, FT, Basis}
    io::IO_t
    path::String
    final_path::String
    settings::Settings
    merged::Merged
    last_hour_next::Last
    header::Dict{String, Any}
    header_bytes::Int
    fidelity::FluxFidelityAccumulator
    bytes_written::Int64
    closed::Bool
    promoted::Bool
end

function LatLonBinaryWriter(path::AbstractString,
                            header_json::AbstractString,
                            settings,
                            merged,
                            last_hour_next,
                            ::Type{FT},
                            _basis::Basis;
                            final_path::AbstractString=path) where {FT, Basis<:AbstractMassBasis}
    encoded_header = codeunits(header_json)
    length(encoded_header) < HEADER_SIZE ||
        throw(ArgumentError("header JSON exceeds fixed header size ($HEADER_SIZE bytes)"))

    parsed = JSON3.read(String(header_json))
    header = Dict{String, Any}(String(k) => v for (k, v) in pairs(parsed))
    io = open(path, "w")
    write_padded_header!(io, header_json, HEADER_SIZE)
    return LatLonBinaryWriter{FT, Basis, typeof(io), typeof(settings), typeof(merged), typeof(last_hour_next)}(
        io,
        String(path),
        String(final_path),
        settings,
        merged,
        last_hour_next,
        header,
        HEADER_SIZE,
        FluxFidelityAccumulator(),
        HEADER_SIZE,
        false,
        false,
    )
end

function write_window!(writer::LatLonBinaryWriter{FT},
                       ready::ReadyWindow{LatLonTargetGeometry, FT}) where {FT}
    writer.closed && throw(ArgumentError("cannot write to a closed LatLonBinaryWriter"))
    verify_flux_fidelity!(writer.fidelity, ready.fidelity, ready.index)
    storage = ready.payload.storage
    bytes = write_window!(writer.io,
                          get(ready.payload,:storage_index,ready.index),
                          storage,
                          writer.settings,
                          writer.merged,
                          writer.last_hour_next)
    writer.bytes_written += bytes
    return bytes
end

mutable struct ReducedGaussianBinaryWriter{FT,
                                           Basis<:AbstractMassBasis,
                                           W} <: AbstractBinaryWriter{ReducedGaussianTargetGeometry, FT, Basis}
    inner::W
    path::String
    final_path::String
    closed::Bool
    promoted::Bool
    fidelity::FluxFidelityAccumulator
end

function ReducedGaussianBinaryWriter(inner::StreamingTransportBinaryWriter{FT},
                                     _basis::Basis;
                                     final_path::AbstractString=inner.path) where {FT, Basis<:AbstractMassBasis}
    return ReducedGaussianBinaryWriter{FT, Basis, typeof(inner)}(
        inner,
        String(inner.path),
        String(final_path),
        false,
        false,
        FluxFidelityAccumulator(),
    )
end

function write_window!(writer::ReducedGaussianBinaryWriter{FT},
                       ready::ReadyWindow{ReducedGaussianTargetGeometry, FT}) where {FT}
    writer.closed && throw(ArgumentError("cannot write to a closed ReducedGaussianBinaryWriter"))
    verify_flux_fidelity!(writer.fidelity, ready.fidelity, ready.index)
    return write_streaming_window!(writer.inner, ready.payload)
end

mutable struct CubedSphereBinaryWriter{FT,
                                       Basis<:AbstractMassBasis,
                                       W} <: AbstractBinaryWriter{CubedSphereTargetGeometry, FT, Basis}
    inner::W
    path::String
    final_path::String
    Nc::Int
    npanel::Int
    closed::Bool
    promoted::Bool
    fidelity::FluxFidelityAccumulator
end

function CubedSphereBinaryWriter(inner::StreamingTransportBinaryWriter{FT},
                                 _basis::Basis;
                                 Nc::Integer,
                                 npanel::Integer,
                                 final_path::AbstractString=inner.path) where {FT, Basis<:AbstractMassBasis}
    return CubedSphereBinaryWriter{FT, Basis, typeof(inner)}(
        inner,
        String(inner.path),
        String(final_path),
        Int(Nc),
        Int(npanel),
        false,
        false,
        FluxFidelityAccumulator(),
    )
end

function write_window!(writer::CubedSphereBinaryWriter{FT},
                       ready::ReadyWindow{CubedSphereTargetGeometry, FT}) where {FT}
    writer.closed && throw(ArgumentError("cannot write to a closed CubedSphereBinaryWriter"))
    verify_flux_fidelity!(writer.fidelity, ready.fidelity, ready.index)
    return write_streaming_cs_window!(writer.inner, ready.payload, writer.Nc, writer.npanel)
end

# Closing patches the certificate into the header; a failed aggregate gate
# (`flux_fidelity_header!` throws) must not leave the IO open. The LL writer's
# file is its own staging path (removed by `quarantine_streaming_binary!`).
function close_streaming_binary!(writer::LatLonBinaryWriter)
    writer.closed && return writer.path
    writer.closed = true
    try
        flux_fidelity_header!(writer.header, writer.fidelity)
        seek(writer.io, 0)
        write_padded_header!(writer.io, writer.header, writer.header_bytes)
        flush(writer.io)
    finally
        close(writer.io)
    end
    return writer.path
end

# The RG/CS adapters wrap a streaming writer with its own atomic staging file
# (`inner.staging_path`, published onto `inner.path` = the adapter's staging
# path). When the certificate fails, that inner staging file is abandoned and
# its IO closed here: quarantine removes only the adapter's path.
function _close_wrapped_streaming_writer!(writer)
    writer.closed && return writer.path
    writer.closed = true
    try
        flux_fidelity_header!(writer.inner.header, writer.fidelity)
    catch
        abandon_streaming_transport_binary!(writer.inner)
        rethrow()
    end
    close_streaming_transport_binary!(writer.inner)
    return writer.path
end

close_streaming_binary!(writer::ReducedGaussianBinaryWriter) =
    _close_wrapped_streaming_writer!(writer)
close_streaming_binary!(writer::CubedSphereBinaryWriter) =
    _close_wrapped_streaming_writer!(writer)

function promote_streaming_binary!(writer::AbstractBinaryWriter)
    writer.promoted && return writer.final_path
    close_streaming_binary!(writer)
    if writer.path != writer.final_path
        commit_atomic_staging(writer.path, writer.final_path)
    end
    writer.promoted = true
    return writer.final_path
end

function quarantine_streaming_binary!(writer::AbstractBinaryWriter)
    writer.promoted && return writer.path
    try
        close_streaming_binary!(writer)
    finally
        abandon_atomic_staging(writer.path)
    end
    return writer.path
end

function validate_staged_size(path::AbstractString, expected_bytes::Integer)
    isfile(path) || throw(ArgumentError(
        "staged transport binary is missing: $(path)"))
    actual_bytes = filesize(path)
    actual_bytes == expected_bytes || throw(ArgumentError(
        "staged transport binary size mismatch: expected $(expected_bytes) bytes, " *
        "found $(actual_bytes) at $(path)"))
    return path
end



function validate_staged_binary!(writer::Union{ReducedGaussianBinaryWriter, CubedSphereBinaryWriter})
    expected = writer.inner.header_bytes +
               writer.inner.expected_windows * writer.inner.elems_per_window *
               sizeof(eltype(writer.inner.pack_buffer))
    return validate_staged_size(writer.path, expected)
end
