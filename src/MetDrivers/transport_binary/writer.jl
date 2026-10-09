# Eager transport-binary writer and payload packers.

@inline function _transport_copy_flat!(dest::Vector{FT}, offset::Int, src) where FT
    src_lin = vec(src)
    n = length(src_lin)
    @inbounds for idx in 1:n
        dest[offset + idx] = convert(FT, src_lin[idx])
    end
    return offset + n
end

function _transport_pack_window!(dest::Vector{FT},
                                 window_offset::Int,
                                 window,
                                 payload_sections::Vector{Symbol}) where FT
    offset = window_offset
    for section in payload_sections
        offset = _transport_copy_flat!(dest, offset, _transport_window_field(window, section))
    end
    return nothing
end

function _transport_pack_payload(windows::AbstractVector,
                                 payload_sections::Vector{Symbol},
                                 elems_per_window::Int,
                                 ::Type{FT};
                                 threaded::Bool = Threads.nthreads() > 1) where FT
    nwindows = length(windows)
    payload = Vector{FT}(undef, nwindows * elems_per_window)

    if threaded && nwindows > 1
        Threads.@threads for win in eachindex(windows)
            window_offset = (win - 1) * elems_per_window
            _transport_pack_window!(payload, window_offset, windows[win], payload_sections)
        end
    else
        @inbounds for win in eachindex(windows)
            window_offset = (win - 1) * elems_per_window
            _transport_pack_window!(payload, window_offset, windows[win], payload_sections)
        end
    end

    return payload
end

function _write_transport_payload!(io::IO,
                                   windows::AbstractVector,
                                   payload_sections::Vector{Symbol},
                                   elems_per_window::Int,
                                   ::Type{FT};
                                   threaded::Bool = Threads.nthreads() > 1) where FT
    payload = _transport_pack_payload(windows, payload_sections, elems_per_window, FT; threaded=threaded)
    write(io, payload)
    return nothing
end

function _write_transport_binary_atomically(write_file::Function, path::AbstractString)
    staging = String(path) * ".tmp"
    rm(staging; force=true)
    try
        open(staging, "w") do io
            write_file(io)
        end
        mv(staging, path; force=true)
    catch
        rm(staging; force=true)
        rethrow()
    end
    return path
end

"""
    write_transport_binary(path, grid, windows; kwargs...) -> path

Write a canonical v4 lat-lon or reduced-Gaussian transport binary from an
AtmosGrid and an ordered vector of forcing windows. Window payloads must match
the grid topology and declared mass basis. The file is written atomically and
replaces path only after the header and all payloads succeed.
"""
function write_transport_binary(path::AbstractString,
                                grid::AtmosGrid{<:LatLonMesh},
                                windows::AbstractVector;
                                FT::Type{<:AbstractFloat} = floattype(grid),
                                header_bytes::Int = 16384,
                                dt_met_seconds::Real = 3600.0,
                                half_dt_seconds::Real = dt_met_seconds / 2,
                                steps_per_window::Integer = 2,
                                source_flux_sampling::Symbol,
                                air_mass_sampling::Symbol = :window_start_endpoint,
                                flux_sampling::Symbol = :window_start_endpoint,
                                flux_kind::Symbol = :substep_mass_amount,
                                humidity_sampling::Symbol = :auto,
                                delta_semantics::Symbol = :auto,
                                mass_basis::Symbol = :dry,
                                extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}(),
                                threaded::Bool = Threads.nthreads() > 1)
    flux_kind === :substep_mass_amount || throw(ArgumentError(
        "lat-lon transport binaries require flux_kind=:substep_mass_amount"))
    isempty(windows) && throw(ArgumentError("write_transport_binary requires at least one window"))

    mesh = grid.horizontal
    vc = grid.vertical
    Nx = nx(mesh)
    Ny = ny(mesh)
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)
    basis_sym = _transport_basis_symbol(mass_basis)
    payload_sections = _transport_payload_sections(grid, first(windows))

    for window in windows
        _transport_payload_sections(grid, window) == payload_sections ||
            throw(ArgumentError("all transport-binary windows must carry the same payload sections"))
        _transport_validate_structured_window(window, Nx, Ny, nlevel, basis_sym)
    end

    elems_per_window = sum(_transport_structured_section_elements(Nx, Ny, ncell, nlevel, section)
                           for section in payload_sections)

    header = _transport_common_header("latlon", "StructuredDirectional",
                                      ncell, nface_h, nlevel, length(windows), vc,
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
        "Nx" => Nx,
        "Ny" => Ny,
        "lons" => Float64.(mesh.λᶜ),
        "lats" => Float64.(mesh.φᶜ),
        "longitude_interval" => Float64[mesh.λᶠ[1], mesh.λᶠ[end]],
        "latitude_interval" => Float64[mesh.φᶠ[1], mesh.φᶠ[end]],
        "grid_convention" => "south_to_north_periodic_longitude",
        "n_m" => ncell * nlevel,
        "n_am" => (Nx + 1) * Ny * nlevel,
        "n_bm" => Nx * (Ny + 1) * nlevel,
        "n_cm" => ncell * (nlevel + 1),
        "n_ps" => ncell,
        "n_dam" => (:dam in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dam) : 0,
        "n_dbm" => (:dbm in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dbm) : 0,
        "n_dcm" => (:dcm in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dcm) : 0,
        "n_dm" => (:dm in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dm) : 0,
    ))
    isempty(extra_header) || _merge_transport_extra_header!(header, extra_header)

    validate_transport_contract!(header)
    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad > 0 || error("transport binary header must leave room for a null terminator within header_bytes=$(header_bytes)")

    _write_transport_binary_atomically(path) do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))
        _write_transport_payload!(io, windows, payload_sections, elems_per_window, FT; threaded=threaded)
    end

    return path
end

function write_transport_binary(path::AbstractString,
                                grid::AtmosGrid{<:ReducedGaussianMesh},
                                windows::AbstractVector;
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
                                mass_basis::Symbol = :dry,
                                extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}(),
                                threaded::Bool = Threads.nthreads() > 1)
    flux_kind === :substep_mass_amount || throw(ArgumentError(
        "reduced-Gaussian transport binaries require flux_kind=:substep_mass_amount"))
    isempty(windows) && throw(ArgumentError("write_transport_binary requires at least one window"))

    mesh = grid.horizontal
    vc = grid.vertical
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)
    basis_sym = _transport_basis_symbol(mass_basis)
    payload_sections = _transport_payload_sections(grid, first(windows))

    for window in windows
        _transport_payload_sections(grid, window) == payload_sections ||
            throw(ArgumentError("all transport-binary windows must carry the same payload sections"))
        _transport_validate_reduced_window(window, ncell, nface_h, nlevel, basis_sym)
    end

    elems_per_window = sum(_transport_faceindexed_section_elements(ncell, nface_h, nlevel, section)
                           for section in payload_sections)

    header = _transport_common_header("reduced_gaussian", "FaceIndexed",
                                      ncell, nface_h, nlevel, length(windows), vc,
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
        "nlat" => nrings(mesh),
        "latitudes" => Float64.(mesh.latitudes),
        "nlon_per_ring" => mesh.nlon_per_ring,
        "n_dhflux" => (:dhflux in payload_sections) ? _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dhflux) : 0,
        "n_dcm" => (:dcm in payload_sections) ? _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dcm) : 0,
        "n_dm" => (:dm in payload_sections) ? _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dm) : 0,
    ))
    isempty(extra_header) || _merge_transport_extra_header!(header, extra_header)

    validate_transport_contract!(header)
    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad > 0 || error("transport binary header must leave room for a null terminator within header_bytes=$(header_bytes)")

    _write_transport_binary_atomically(path) do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))
        _write_transport_payload!(io, windows, payload_sections, elems_per_window, FT; threaded=threaded)
    end

    return path
end
