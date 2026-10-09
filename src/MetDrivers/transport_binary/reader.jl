# Shared version-4 reader, accessors, and structured/face-indexed loaders.

"""
    TransportBinaryReader{FT, DiskFT, G}

Memory-mapped reader for the current transport-binary format.

`FT` is the element type produced by window loaders, `DiskFT` is the mmap
element type, and `G` is the concrete geometry metadata type. Payloads remain
in their native topology: structured arrays for lat-lon, face-indexed arrays
for reduced Gaussian, and six panel arrays for cubed sphere.
"""
struct TransportBinaryReader{FT, DiskFT, G <: AbstractTransportBinaryGeometry}
    data   :: Vector{DiskFT}
    io     :: IOStream
    header :: TransportBinaryHeader{G}
    path   :: String
end

_transport_load_float_type(::TransportBinaryReader{FT}) where FT = FT

function Base.summary(r::TransportBinaryReader{FT, DiskFT}) where {FT, DiskFT}
    return string(
        "TransportBinaryReader{", FT, "←", DiskFT, "}(",
        basename(r.path), ", ", grid_type(r), "/", horizontal_topology(r), ", ",
        r.header.nwindow, " windows)"
    )
end

function Base.show(io::IO, r::TransportBinaryReader)
    h = r.header
    print(io, summary(r), "\n",
          "├── path:          ", r.path, "\n",
          "├── geometry:      ", _transport_geometry_summary(h), "\n",
          "├── storage:       ", h.on_disk_float_type, " on disk, load as ",
              _transport_load_float_type(r), "\n",
          "├── basis:         ", h.mass_basis, "\n",
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=",
              _steps_per_window_summary(h.steps_per_window, h.steps_per_window_by_window), "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", _transport_qv_summary(h), "\n",
          "├── semantics:     ", _transport_semantics_summary(h), "\n",
          "├── poisson:       ", isnan(h.poisson_balance_target_scale) ? "unspecified" :
                               string("scale=", h.poisson_balance_target_scale, ", ", h.poisson_balance_target_semantics), "\n",
          "└── windows:       ", h.nwindow)
end

window_count(r::TransportBinaryReader) = r.header.nwindow
steps_per_window(r::TransportBinaryReader) = r.header.steps_per_window
steps_per_window(r::TransportBinaryReader, win::Integer) =
    r.header.steps_per_window_by_window[Int(win)]
steps_per_window_schedule(r::TransportBinaryReader) =
    copy(r.header.steps_per_window_by_window)
mass_basis(r::TransportBinaryReader) = r.header.mass_basis
binary_geometry(r::TransportBinaryReader) = binary_geometry(r.header)
grid_type(r::TransportBinaryReader) = grid_type(r.header)
horizontal_topology(r::TransportBinaryReader) = horizontal_topology(r.header)
source_flux_sampling(r::TransportBinaryReader) = r.header.source_flux_sampling
air_mass_sampling(r::TransportBinaryReader) = r.header.air_mass_sampling
flux_sampling(r::TransportBinaryReader) = r.header.flux_sampling
flux_kind(r::TransportBinaryReader) = r.header.flux_kind
humidity_sampling(r::TransportBinaryReader) = r.header.humidity_sampling
delta_semantics(r::TransportBinaryReader) = r.header.delta_semantics
A_ifc(r::TransportBinaryReader) = r.header.A_ifc
B_ifc(r::TransportBinaryReader) = r.header.B_ifc
"""Return `true` when both specific-humidity endpoint sections are present."""
has_qv_endpoints(r::TransportBinaryReader) =
    :qv_start in r.header.payload_sections && :qv_end in r.header.payload_sections
has_flux_delta(r::TransportBinaryReader) = any(section in (:dam, :dbm, :dcm, :dm, :dhflux) for section in r.header.payload_sections)

"""
    has_tm5_convection(r::TransportBinaryReader) -> Bool

`true` if the binary carries all four TM5 convection sections
(`entu`, `detu`, `entd`, `detd`) — the contract enforced by the
preprocessor when `tm5_convection = true`. Used by the
`TransportBinaryDriver` to decide whether to populate
`ConvectionForcing.tm5_fields` on loaded windows.
"""
has_tm5_convection(r::TransportBinaryReader) =
    all(s in r.header.payload_sections for s in (:entu, :detu, :entd, :detd))
has_cmfmc(r::TransportBinaryReader) = :cmfmc in r.header.payload_sections
has_surface(r::TransportBinaryReader) =
    all(s in r.header.payload_sections for s in _PBL_SURFACE_PAYLOAD_SECTIONS)
has_vdiff_fields(r::TransportBinaryReader) =
    all(s in r.header.payload_sections for s in _GCHP_VDIFF_PAYLOAD_SECTIONS)
has_pbl_eflux(r::TransportBinaryReader) = :pbl_eflux in r.header.payload_sections
has_cmfmc_cloud_base(r::TransportBinaryReader) = :cmfmc_cloud_base in r.header.payload_sections

# ---------------------------------------------------------------------------
# Reader construction. Capability summaries and `inspect_binary` live in
# `inspect.jl`.
# ---------------------------------------------------------------------------

function TransportBinaryReader(bin_path::String; FT::Type{<:AbstractFloat} = Float32)
    io = open(bin_path, "r")
    try
        raw = _read_transport_header_json(io; source = "transport binary $(bin_path)")

        # Validate the self-describing transport-binary contract before mmap.
        # `String(::Vector{UInt8})` may take ownership of and empty the vector;
        # keep `raw` intact for the typed header parser below.
        hdr_obj = JSON3.read(String(copy(raw)))
        hdr_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr_obj))
        validate_transport_contract!(hdr_dict)
        length(raw) < Int(hdr_dict["header_bytes"]) || throw(ArgumentError(
            "transport binary JSON header is not null-terminated before header_bytes"))

        header = _parse_transport_header(raw)
        DiskFT = _transport_disk_float_type(header.on_disk_float_type)
        total_elems = header.n_geometry_elems + header.elems_per_window * header.nwindow
        expected_bytes = header.header_bytes + total_elems * sizeof(DiskFT)
        actual_bytes = filesize(bin_path)
        actual_bytes == expected_bytes || throw(ArgumentError(
            "transport binary size mismatch for $(bin_path): expected $(expected_bytes) bytes " *
            "from the header, found $(actual_bytes)"))
        data = Mmap.mmap(io, Vector{DiskFT}, total_elems, header.header_bytes)
        G = typeof(header.geometry)
        return TransportBinaryReader{FT, DiskFT, G}(data, io, header, bin_path)
    catch
        isopen(io) && close(io)
        rethrow()
    end
end

Base.close(r::TransportBinaryReader) = close(r.io)

@inline _transport_window_offset(r::TransportBinaryReader, win::Int) =
    r.header.n_geometry_elems + (win - 1) * r.header.elems_per_window

function _transport_interval_from_centers(centers::Vector{Float64}, fallback_Δ::Float64)
    isempty(centers) && error("Cannot reconstruct interval from empty center array")
    Δ = length(centers) > 1 ? centers[2] - centers[1] : fallback_Δ
    return (centers[1] - Δ / 2, centers[end] + Δ / 2)
end

function load_grid(reader::TransportBinaryReader{<:Any, <:Any, LatLonBinaryGeometry};
                   FT::Type{<:AbstractFloat} = Float64, arch = CPU())
    h = reader.header
    g = h.geometry
    vc = HybridSigmaPressure(FT.(h.A_ifc), FT.(h.B_ifc))
    longitude = length(g.longitude_interval) == 2 ?
        (FT(g.longitude_interval[1]), FT(g.longitude_interval[2])) :
        let interval = _transport_interval_from_centers(g.longitudes, 360.0 / g.Nx)
            (FT(interval[1]), FT(interval[2]))
        end
    latitude = length(g.latitude_interval) == 2 ?
        (FT(g.latitude_interval[1]), FT(g.latitude_interval[2])) :
        let interval = _transport_interval_from_centers(g.latitudes, 180.0 / g.Ny)
            (FT(interval[1]), FT(interval[2]))
        end
    mesh = LatLonMesh(; FT=FT, size=(g.Nx, g.Ny), longitude=longitude, latitude=latitude,
                      radius=FT(h.planet_radius_m))
    return AtmosGrid(mesh, vc, arch; FT=FT, radius=h.planet_radius_m)
end

function load_grid(reader::TransportBinaryReader{<:Any, <:Any, ReducedGaussianBinaryGeometry};
                   FT::Type{<:AbstractFloat} = Float64, arch = CPU())
    h = reader.header
    g = h.geometry
    vc = HybridSigmaPressure(FT.(h.A_ifc), FT.(h.B_ifc))
    mesh = ReducedGaussianMesh(g.latitudes, g.nlon_per_ring; FT=FT, radius=FT(h.planet_radius_m))
    return AtmosGrid(mesh, vc, arch; FT=FT, radius=h.planet_radius_m)
end

_transport_allocate_mass(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

_transport_allocate_ps(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny) :
        Array{FT}(undef, reader.header.ncell)

_transport_allocate_cm(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny, reader.header.nlevel + 1) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel + 1)

_transport_allocate_am(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.geometry.Nx + 1, reader.header.geometry.Ny, reader.header.nlevel)

_transport_allocate_bm(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny + 1, reader.header.nlevel)

_transport_allocate_hflux(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.nface_h, reader.header.nlevel)

_transport_allocate_qv(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

_transport_allocate_dam(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.geometry.Nx + 1, reader.header.geometry.Ny, reader.header.nlevel)

_transport_allocate_dbm(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny + 1, reader.header.nlevel)

_transport_allocate_dhflux(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.nface_h, reader.header.nlevel)

_transport_allocate_dm(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

_transport_allocate_dcm(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny, reader.header.nlevel + 1) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel + 1)

_transport_allocate_surface_field(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny) :
        Array{FT}(undef, reader.header.ncell)

# TM5 convection fields — all layer-center, shape matches `m`.
_transport_allocate_tm5_field(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.geometry.Nx, reader.header.geometry.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

function _transport_make_fluxes(::Val{:dry}, am, bm, cm)
    return StructuredFaceFluxState{DryBasis}(am, bm, cm)
end

function _transport_make_fluxes(::Val{:moist}, am, bm, cm)
    return StructuredFaceFluxState{MoistBasis}(am, bm, cm)
end

function _transport_make_fluxes(::Val{:dry}, hflux, cm)
    return FaceIndexedFluxState{DryBasis}(hflux, cm)
end

function _transport_make_fluxes(::Val{:moist}, hflux, cm)
    return FaceIndexedFluxState{MoistBasis}(hflux, cm)
end

"""
    load_window!(reader, win; buffers...) -> (air_mass, surface_pressure, fluxes)

Load the required transport state for one lat-lon or reduced-Gaussian window.
Optional buffer keywords reuse caller-owned storage; omitted buffers are
allocated with the reader's output element type.
"""
function load_window!(reader::TransportBinaryReader{FT}, win::Int;
                      m = nothing,
                      ps = nothing,
                      hflux = nothing,
                      am = nothing,
                      bm = nothing,
                      cm = nothing) where FT
    h = reader.header
    m = isnothing(m) ? _transport_allocate_mass(reader) : m
    ps = isnothing(ps) ? _transport_allocate_ps(reader) : ps
    cm = isnothing(cm) ? _transport_allocate_cm(reader) : cm

    if _transport_is_structured(h)
        am = isnothing(am) ? _transport_allocate_am(reader) : am
        bm = isnothing(bm) ? _transport_allocate_bm(reader) : bm
        o = _transport_window_offset(reader, win)
        saw_m = saw_am = saw_bm = saw_cm = saw_ps = false

        for section in h.payload_sections
            n = _transport_section_elements(h, section)
            if section === :m
                copyto!(m, 1, reader.data, o + 1, n)
                saw_m = true
            elseif section === :am
                copyto!(am, 1, reader.data, o + 1, n)
                saw_am = true
            elseif section === :bm
                copyto!(bm, 1, reader.data, o + 1, n)
                saw_bm = true
            elseif section === :cm
                copyto!(cm, 1, reader.data, o + 1, n)
                saw_cm = true
            elseif section === :ps
                copyto!(ps, 1, reader.data, o + 1, n)
                saw_ps = true
            end
            o += n
        end

        saw_m || error("Transport binary payload is missing required section `m`")
        saw_am || error("Transport binary payload is missing required section `am`")
        saw_bm || error("Transport binary payload is missing required section `bm`")
        saw_cm || error("Transport binary payload is missing required section `cm`")
        saw_ps || error("Transport binary payload is missing required section `ps`")

        fluxes = _transport_make_fluxes(Val(h.mass_basis), am, bm, cm)
        return m, ps, fluxes
    elseif _transport_is_faceindexed(h)
        hflux = isnothing(hflux) ? _transport_allocate_hflux(reader) : hflux
        o = _transport_window_offset(reader, win)
        saw_m = saw_hflux = saw_cm = saw_ps = false

        for section in h.payload_sections
            n = _transport_section_elements(h, section)
            if section === :m
                copyto!(m, 1, reader.data, o + 1, n)
                saw_m = true
            elseif section === :hflux
                copyto!(hflux, 1, reader.data, o + 1, n)
                saw_hflux = true
            elseif section === :cm
                copyto!(cm, 1, reader.data, o + 1, n)
                saw_cm = true
            elseif section === :ps
                copyto!(ps, 1, reader.data, o + 1, n)
                saw_ps = true
            end
            o += n
        end

        saw_m || error("Transport binary payload is missing required section `m`")
        saw_hflux || error("Transport binary payload is missing required section `hflux`")
        saw_cm || error("Transport binary payload is missing required section `cm`")
        saw_ps || error("Transport binary payload is missing required section `ps`")

        fluxes = _transport_make_fluxes(Val(h.mass_basis), hflux, cm)
        return m, ps, fluxes
    else
        throw(ArgumentError("Unsupported transport binary grid/topology combination: " *
                            "$(grid_type(h)) / $(horizontal_topology(h))"))
    end
end

"""
    load_flux_delta_window!(reader, win; buffers...) -> NamedTuple | nothing

Load forward endpoint deltas for air mass and the topology's face fluxes.
Returns `nothing` when the binary carries no delta sections.
"""
function load_flux_delta_window!(reader::TransportBinaryReader{FT}, win::Int;
                                 dam = nothing,
                                 dbm = nothing,
                                 dhflux = nothing,
                                 dcm = nothing,
                                 dm = nothing) where FT
    has_flux_delta(reader) || return nothing
    h = reader.header

    dam = isnothing(dam) && _transport_is_structured(h) ? _transport_allocate_dam(reader) : dam
    dbm = isnothing(dbm) && _transport_is_structured(h) ? _transport_allocate_dbm(reader) : dbm
    dhflux = isnothing(dhflux) && _transport_is_faceindexed(h) ? _transport_allocate_dhflux(reader) : dhflux
    dcm = isnothing(dcm) ? _transport_allocate_dcm(reader) : dcm
    dm = isnothing(dm) ? _transport_allocate_dm(reader) : dm

    o = _transport_window_offset(reader, win)
    found_any = false
    found_dam = found_dbm = found_dhflux = found_dcm = found_dm = false
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :dam
            copyto!(dam, 1, reader.data, o + 1, n)
            found_dam = true
            found_any = true
        elseif section === :dbm
            copyto!(dbm, 1, reader.data, o + 1, n)
            found_dbm = true
            found_any = true
        elseif section === :dhflux
            copyto!(dhflux, 1, reader.data, o + 1, n)
            found_dhflux = true
            found_any = true
        elseif section === :dcm
            copyto!(dcm, 1, reader.data, o + 1, n)
            found_dcm = true
            found_any = true
        elseif section === :dm
            copyto!(dm, 1, reader.data, o + 1, n)
            found_dm = true
            found_any = true
        end
        o += n
    end

    found_any || return nothing

    result = NamedTuple()
    if found_dam
        result = merge(result, (; dam))
    end
    if found_dbm
        result = merge(result, (; dbm))
    end
    if found_dhflux
        result = merge(result, (; dhflux))
    end
    if found_dcm
        result = merge(result, (; dcm))
    end
    if found_dm
        result = merge(result, (; dm))
    end
    return result
end

"""
    load_tm5_convection_window!(reader, win; entu=..., detu=..., entd=..., detd=...) -> NamedTuple | nothing

Load the four TM5 convection layer-center fields for window `win`.
Returns `(; entu, detu, entd, detd)` when the binary carries all
four sections, or `nothing` when no TM5 data is present. Allocates
only if the caller doesn't provide pre-allocated buffers.

All fields share the same shape as `m`: `(Nx, Ny, Nz)` for
structured or `(ncells, Nz)` for face-indexed binaries. Orientation
is as written by the preprocessor (AtmosTransport: k=1=TOA,
k=Nz=surface); no runtime reorientation happens here — the kernel
reads them directly.

Invariant: if ANY of the four sections is present in the header,
ALL four must be present. This mirrors the
`ConvectionForcing.tm5_fields` NamedTuple contract — partial
payload is not a valid convection forcing.
"""
function load_tm5_convection_window!(reader::TransportBinaryReader{FT}, win::Int;
                                      entu = nothing,
                                      detu = nothing,
                                      entd = nothing,
                                      detd = nothing) where FT
    has_tm5_convection(reader) || return nothing
    h = reader.header

    entu = isnothing(entu) ? _transport_allocate_tm5_field(reader) : entu
    detu = isnothing(detu) ? _transport_allocate_tm5_field(reader) : detu
    entd = isnothing(entd) ? _transport_allocate_tm5_field(reader) : entd
    detd = isnothing(detd) ? _transport_allocate_tm5_field(reader) : detd

    o = _transport_window_offset(reader, win)
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :entu
            copyto!(entu, 1, reader.data, o + 1, n)
        elseif section === :detu
            copyto!(detu, 1, reader.data, o + 1, n)
        elseif section === :entd
            copyto!(entd, 1, reader.data, o + 1, n)
        elseif section === :detd
            copyto!(detd, 1, reader.data, o + 1, n)
        end
        o += n
    end
    return (; entu, detu, entd, detd)
end

"""
    load_surface_window!(reader, win; pblh=..., ustar=..., hflux=..., t2m=...) -> PBLSurfaceForcing | nothing

Load raw PBL surface fields for window `win`. Returns `nothing` when
the binary lacks surface sections. A binary carrying only a subset of
`pblh`, `ustar`, `pbl_hflux`, and `t2m` is rejected because the runtime
PBL closure needs the complete raw surface payload. The on-disk heat-flux
section is `pbl_hflux`; callers still receive it as `PBLSurfaceForcing.hflux`.
"""
function load_surface_window!(reader::TransportBinaryReader{FT}, win::Int;
                              pblh = nothing,
                              ustar = nothing,
                              hflux = nothing,
                              t2m = nothing) where FT
    surface_present = has_surface(reader)
    surface_partial = any(s in reader.header.payload_sections
                          for s in _PBL_SURFACE_PAYLOAD_SECTIONS) && !surface_present
    if surface_partial
        legacy_hflux = :hflux in reader.header.payload_sections &&
                       !(:pbl_hflux in reader.header.payload_sections)
        msg = "transport binary has a partial PBL surface payload; expected all " *
              "of pblh, ustar, pbl_hflux, t2m"
        legacy_hflux && (msg *= "\n  This binary appears to be pre-2026-05-01 " *
                                "(commit 66bbce3): the on-disk PBL sensible-heat " *
                                "section is `:hflux` rather than the renamed " *
                                "`:pbl_hflux`. Regenerate via " *
                                "scripts/preprocessing/preprocess_transport_binary.jl.")
        throw(ArgumentError(msg))
    end
    surface_present || return nothing

    pblh = isnothing(pblh) ? _transport_allocate_surface_field(reader) : pblh
    ustar = isnothing(ustar) ? _transport_allocate_surface_field(reader) : ustar
    hflux = isnothing(hflux) ? _transport_allocate_surface_field(reader) : hflux
    t2m = isnothing(t2m) ? _transport_allocate_surface_field(reader) : t2m

    o = _transport_window_offset(reader, win)
    found_pblh = found_ustar = found_hflux = found_t2m = false
    for section in reader.header.payload_sections
        n = _transport_section_elements(reader.header, section)
        if section === :pblh
            copyto!(pblh, 1, reader.data, o + 1, n)
            found_pblh = true
        elseif section === :ustar
            copyto!(ustar, 1, reader.data, o + 1, n)
            found_ustar = true
        elseif section === :pbl_hflux
            copyto!(hflux, 1, reader.data, o + 1, n)
            found_hflux = true
        elseif section === :t2m
            copyto!(t2m, 1, reader.data, o + 1, n)
            found_t2m = true
        end
        o += n
    end

    (found_pblh && found_ustar && found_hflux && found_t2m) ||
        throw(ArgumentError(
            "transport binary has an incomplete PBL surface payload in window $(win)"))
    return PBLSurfaceForcing(pblh, ustar, hflux, t2m)
end

"""
    load_qv_pair_window!(reader, win; qv_start=nothing, qv_end=nothing)

Load the specific-humidity fields at the start and end of window `win`.
Returns `(; qv_start, qv_end)` or `nothing` when humidity endpoints are absent.
Caller-provided arrays are filled in place.
"""
function load_qv_pair_window!(reader::TransportBinaryReader{FT}, win::Int;
                              qv_start = nothing,
                              qv_end = nothing) where FT
    h = reader.header
    has_qv_endpoints(reader) || return nothing
    qv_start = isnothing(qv_start) ? _transport_allocate_qv(reader) : qv_start
    qv_end = isnothing(qv_end) ? _transport_allocate_qv(reader) : qv_end

    o = _transport_window_offset(reader, win)
    found_start = false
    found_end = false
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :qv_start
            copyto!(qv_start, 1, reader.data, o + 1, n)
            found_start = true
        elseif section === :qv_end
            copyto!(qv_end, 1, reader.data, o + 1, n)
            found_end = true
        end
        o += n
    end

    return found_start && found_end ? (; qv_start, qv_end) : nothing
end
