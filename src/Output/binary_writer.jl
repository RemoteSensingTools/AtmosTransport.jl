# Binary snapshot writer — cubed-sphere only.
#
# Drop-in alternative to `write_snapshot_netcdf` that skips the HDF5 / NetCDF
# library entirely. Writes raw Float32 panel data to a single file with a
# self-describing JSON header. Designed for offline conversion to NetCDF via
# `scripts/postprocess/binary_to_netcdf.jl` so downstream tools stay
# unchanged.
#
# File layout:
#
#   [ 8 bytes  ] magic:        "ATMSNAP1" (ASCII, no null terminator)
#   [ 8 bytes  ] header_size:  UInt64, byte length of the JSON header
#   [ N bytes  ] JSON header:  UTF-8, schema described below
#   [ padding  ] zero bytes to align payload to 4096-byte page boundary
#   [ payload  ] frames × fields × panels × (Nc × Nc × Nz) Float32, column-major
#
# JSON header schema (CS-only, v1):
#
#   { "format": "ATMSNAP1", "grid_type": "cubed_sphere", "version": 1,
#     "mass_basis": "dry" | "moist",
#     "float_dtype": "Float32",
#     "grid": { "Nc": int, "Nz": int, "n_panels": 6,
#               "definition": str, "panel_convention": str,
#               "coordinate_law": str, "center_law": str,
#               "planet_radius_m": float },     # absent in older files: EARTH_RADIUS
#     "fields": [ "air_mass", tracer_name_1, ... ],   # air_mass is always first
#     "times_hours": [ float, ... ],
#     "tracer_total_mass": { tracer_name_1: [float64, ...], ... },
#     "n_frames": int, "payload_offset": int, "payload_bytes": int }
#
# Tracer storage is mass (not mixing ratio) — same contract as `SnapshotFrame`.
# The offline converter derives VMR and column diagnostics on the host using
# the same `Output` module helpers as the live NetCDF writer would.

using Mmap

const _ATMSNAP_MAGIC = "ATMSNAP1"
const _ATMSNAP_HEADER_ALIGN = 4096
const _ATMSNAP_VERSION = 1

"""
    write_snapshot_binary(path, frames, grid; mass_basis=:dry, options=SnapshotWriteOptions())

Write CS snapshots to a self-describing binary file. Skips NetCDF library
overhead entirely; pair with `scripts/postprocess/binary_to_netcdf.jl` for the
offline conversion. Throws if `grid.horizontal` is not a `CubedSphereMesh`.
"""
function write_snapshot_binary(path::AbstractString,
                                frames::AbstractVector{<:AbstractSnapshotFrame},
                                grid::AtmosGrid;
                                mass_basis::Symbol = :dry,
                                options::SnapshotWriteOptions = SnapshotWriteOptions())
    all(f -> f isa SnapshotFrame, frames) || throw(ArgumentError("binary output requires full snapshots"))
    mesh = grid.horizontal
    mesh isa CubedSphereMesh || throw(ArgumentError(
        "write_snapshot_binary currently supports cubed-sphere grids only; got $(typeof(mesh))"))
    isempty(frames) && throw(ArgumentError(
        "write_snapshot_binary requires at least one SnapshotFrame"))
    _check_same_keys(frames)
    _check_mass_basis(frames, mass_basis)
    _check_frame_shapes(frames, mesh)
    options.float_type === Float32 || throw(ArgumentError(
        "binary writer only supports Float32 on-disk dtype; got $(options.float_type)"))

    expanded = expand_data_path(String(path))
    _ensure_parent_dir(expanded)
    isfile(expanded) && rm(expanded)

    tracer_keys = _frame_tracer_names(first(frames))
    field_names = String["air_mass"; String.(tracer_keys)]
    n_fields = length(field_names)
    Nc, Nz = mesh.Nc, _nlevel(first(frames), mesh)
    n_frames = length(frames)
    n_panels = 6
    panel_floats = Nc * Nc * Nz
    payload_bytes = n_frames * n_fields * n_panels * panel_floats * sizeof(Float32)

    header = Dict{String, Any}(
        "format" => _ATMSNAP_MAGIC,
        "version" => _ATMSNAP_VERSION,
        "grid_type" => "cubed_sphere",
        "mass_basis" => String(mass_basis),
        "float_dtype" => "Float32",
        "grid" => Dict{String, Any}(
            "Nc" => Nc,
            "Nz" => Nz,
            "n_panels" => n_panels,
            "definition" => String(cs_definition_tag(cs_definition(mesh))),
            "panel_convention" => _panel_convention_tag(mesh),
            "coordinate_law" => String(coordinate_law_tag(coordinate_law(mesh))),
            "center_law" => String(center_law_tag(center_law(mesh))),
            "planet_radius_m" => Float64(mesh.radius),
        ),
        "fields" => field_names,
        "times_hours" => [frame.time_hours for frame in frames],
        "tracer_total_mass" => Dict(
            String(name) => [frame.tracer_total_mass[name] for frame in frames]
            for name in tracer_keys),
        "tracer_total_mass_dtype" => "Float64",
        "tracer_total_mass_storage" => "mixing_ratio_times_carrier_air_mass",
        "n_frames" => n_frames,
        "payload_bytes" => payload_bytes,
    )
    header_bytes = Vector{UInt8}(JSON3.write(header))
    fixed_prefix = length(_ATMSNAP_MAGIC) + sizeof(UInt64)   # 8 + 8 = 16
    raw_header_end = fixed_prefix + length(header_bytes)
    payload_offset = cld(raw_header_end, _ATMSNAP_HEADER_ALIGN) * _ATMSNAP_HEADER_ALIGN
    header["payload_offset"] = payload_offset
    # Re-serialize with payload_offset filled in. Re-pad header so the
    # alignment is stable.
    header_bytes = Vector{UInt8}(JSON3.write(header))
    raw_header_end = fixed_prefix + length(header_bytes)
    payload_offset >= raw_header_end || throw(ErrorException(
        "binary writer header outgrew its pre-computed payload offset; bump _ATMSNAP_HEADER_ALIGN"))

    total_size = payload_offset + payload_bytes
    open(expanded, "w+") do io
        # Pre-allocate file to full size (sparse on most filesystems, then
        # written through mmap below). Lets the OS lay out contiguous extents.
        truncate(io, total_size)
        # Header block.
        write(io, Vector{UInt8}(_ATMSNAP_MAGIC))
        write(io, UInt64(length(header_bytes)))
        write(io, header_bytes)
        # Pad to payload_offset.
        zero_pad = payload_offset - raw_header_end
        zero_pad > 0 && write(io, zeros(UInt8, zero_pad))
        flush(io)

        # Memory-map the payload region and stream frames into it. The OS
        # handles dirty-page flushing asynchronously; closing the file does
        # an msync but does not block on the full payload write.
        payload = Mmap.mmap(io, Vector{Float32}, payload_bytes ÷ sizeof(Float32),
                            payload_offset)
        cursor = 1
        @inbounds for frame in frames
            cursor = _write_cs_frame_into_payload!(payload, cursor, frame, tracer_keys,
                                                    Nc, Nz)
        end
        Mmap.sync!(payload)
    end
    @info @sprintf("Saved binary snapshots: %s (%d frame(s), %s, mass_basis=%s, %.2f GiB payload)",
                   expanded, n_frames, summary(mesh), mass_basis,
                   payload_bytes / 2.0^30)
    return expanded
end

@inline function _panel_convention_tag(mesh::CubedSphereMesh)
    conv = mesh.convention
    conv isa GnomonicPanelConvention && return "gnomonic"
    conv isa GEOSNativePanelConvention && return "geos_native"
    return String(Symbol(typeof(conv).name.name))
end

@inline function _write_cs_frame_into_payload!(payload::Vector{Float32}, cursor::Int,
                                                frame::SnapshotFrame,
                                                tracer_keys::Vector{Symbol},
                                                Nc::Int, Nz::Int)
    cursor = _write_cs_field_into_payload!(payload, cursor, frame.air_mass, Nc, Nz)
    for name in tracer_keys
        cursor = _write_cs_field_into_payload!(payload, cursor, frame.tracers[name], Nc, Nz)
    end
    return cursor
end

@inline function _write_cs_field_into_payload!(payload::Vector{Float32}, cursor::Int,
                                                field::NTuple{6, <:AbstractArray},
                                                Nc::Int, Nz::Int)
    panel_floats = Nc * Nc * Nz
    @inbounds for p in 1:6
        panel = field[p]
        size(panel, 1) == Nc && size(panel, 2) == Nc && size(panel, 3) == Nz ||
            throw(DimensionMismatch(
                "binary writer expected panel shape ($(Nc), $(Nc), $(Nz)); got $(size(panel))"))
        flat = reshape(panel, panel_floats)
        copyto!(payload, cursor, flat, 1, panel_floats)
        cursor += panel_floats
    end
    return cursor
end
