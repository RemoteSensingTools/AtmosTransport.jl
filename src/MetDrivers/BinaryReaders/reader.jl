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
          "├── geometry:      ", transport_geometry_summary(h), "\n",
          "├── storage:       ", h.on_disk_float_type, " on disk, load as ",
              _transport_load_float_type(r), "\n",
          "├── basis:         ", h.mass_basis, ", ps=",
              transport_surface_pressure_basis_label(h.surface_pressure_basis), "\n",
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=",
              steps_per_window_summary(h.steps_per_window, h.steps_per_window_by_window), "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", transport_qv_summary(h), "\n",
          "├── semantics:     ", transport_semantics_summary(h), "\n",
          "├── poisson:       ", string("scale=", h.poisson_balance_target_scale, ", ", h.poisson_balance_target_semantics), "\n",
          "└── windows:       ", h.nwindow)
end

window_count(r::TransportBinaryReader) = r.header.nwindow
steps_per_window(r::TransportBinaryReader) = r.header.steps_per_window
steps_per_window(r::TransportBinaryReader, win::Integer) =
    r.header.steps_per_window_by_window[Int(win)]
steps_per_window_schedule(r::TransportBinaryReader) =
    copy(r.header.steps_per_window_by_window)
"""Return the transport binary header mass basis as `:dry` or `:moist`."""
binary_mass_basis(r::TransportBinaryReader) = r.header.mass_basis
surface_pressure_basis(r::TransportBinaryReader) = r.header.surface_pressure_basis
binary_geometry(r::TransportBinaryReader) = binary_geometry(r.header)
grid_type(r::TransportBinaryReader) = grid_type(r.header)
horizontal_topology(r::TransportBinaryReader) = horizontal_topology(r.header)
source_flux_sampling(r::TransportBinaryReader) = r.header.source_flux_sampling
air_mass_sampling(r::TransportBinaryReader) = r.header.air_mass_sampling
flux_sampling(r::TransportBinaryReader) = r.header.flux_sampling
flux_kind(r::TransportBinaryReader) = r.header.flux_kind
humidity_sampling(r::TransportBinaryReader) = r.header.humidity_sampling
delta_semantics(r::TransportBinaryReader) = r.header.delta_semantics
"""Return `true` when both specific-humidity endpoint sections are present."""
has_qv_endpoints(r::TransportBinaryReader) =
    :qv_start in r.header.payload_sections && :qv_end in r.header.payload_sections
has_flux_delta(r::TransportBinaryReader) = any(section in FLUX_DELTA_PAYLOAD_SECTIONS for section in r.header.payload_sections)

"""
    has_tm5_convection(r::TransportBinaryReader) -> Bool

`true` if the binary carries all four TM5 convection sections
(`entu`, `detu`, `entd`, `detd`) — the contract enforced by the
preprocessor when `tm5_convection = true`. Used by the
`TransportBinaryDriver` to decide whether to populate
`ConvectionForcing.tm5_fields` on loaded windows.
"""
has_tm5_convection(r::TransportBinaryReader) =
    all(s in r.header.payload_sections for s in TM5_CONVECTION_PAYLOAD_SECTIONS)
"""
    tm5_convection_mass_basis(r::TransportBinaryReader) -> Union{AbstractMassBasis, Nothing}

Typed air-mass basis of the TM5 convection sections (`DryBasis()` or
`MoistBasis()`) parsed from the header key
[`TM5_CONVECTION_MASS_BASIS_KEY`](@ref); `nothing` when the binary has no TM5
sections or does not record the basis (written before the key existed, with
moist fluxes).
"""
tm5_convection_mass_basis(r::TransportBinaryReader) =
    r.header.tm5_convection_mass_basis

"""
    _check_tm5_convection_basis(reader) -> nothing

Check on opening that TM5 convection sections are on the binary's air-mass
basis. A recorded mismatch is an error. A legacy dry binary without the
record carries moist fluxes and warns until the corrected production archive
has been regenerated.
"""
function _check_tm5_convection_basis(reader::TransportBinaryReader)
    has_tm5_convection(reader) || return nothing
    recorded = tm5_convection_mass_basis(reader)
    binary_basis = binary_mass_basis(reader)
    if recorded === nothing
        # Planned production-regeneration switch: after the ERA5 TM5 archive
        # is regenerated with issue #19 fixed, replace this warning by an error.
        binary_basis === :dry && @warn(
            "$(basename(reader.path)) does not record `$(TM5_CONVECTION_MASS_BASIS_KEY)`: " *
            "it was written before TM5 convection fluxes were converted to dry air " *
            "(issue #19). Its moist fluxes act on dry air mass, so convective rates " *
            "are too large by 1/(1-q): about 0.7 % (updraft) and 1.1 % (downdraft) " *
            "globally, up to about 2 % in humid tropical columns. Regenerate the " *
            "binary to remove the bias. (Reported once per session.)",
            maxlog = 1)
    elseif typeof(recorded) !== typeof(mass_basis(binary_basis))
        recorded_label = basis_symbol(recorded)
        throw(ArgumentError(
            "$(basename(reader.path)): TM5 convection fluxes are on a $(recorded_label) basis " *
            "(`$(TM5_CONVECTION_MASS_BASIS_KEY)`) but the air mass is $(binary_basis) " *
            "(`mass_basis`); TM5Convection needs both on the same basis. Regenerate the binary."))
    end
    return nothing
end

"""
    _check_tm5_convection_provenance(reader) -> nothing

Reject, on opening, TM5 convection sections built by a conversion with a known
index error. Only binaries with all four TM5 sections are checked. Sections
attached from the TM5-converted CATRINE archive (`convection_attachment`,
converted by TM5's own `ECconv_to_TMconv`) are accepted first; the others are
rejected by these rules, in order:

1. Lat-lon binaries of the ERA5 spectral preprocessor (header key
   `spectral_half_dt_seconds`): their sections came from the retired lat-lon
   physics-file route (issue #55, which also used the issue-#54 indexing).
2. Lat-lon → cubed-sphere regrids (`preprocessor = "regrid_ll_binary_to_cs"`)
   without [`TM5_CONVECTION_SOURCE_KEY`](@ref): written before the regrid
   carried its source's record, from binaries of that route.
3. A recorded [`TM5_EC2TM_CONVENTION_KEY`](@ref) other than
   [`TM5_EC2TM_CONVENTION`](@ref).
4. ERA5-derived sections (`TM5_CONVECTION_SOURCE_KEY` starting with `ec2tm`)
   without the convention record: built before the issue-#54 fix, with ERA5
   UDMF/DDMF placed one interface too low and the cloud-top layer dropped.

Sections without any recorded source are accepted.
"""
function _check_tm5_convection_provenance(reader::TransportBinaryReader)
    has_tm5_convection(reader) || return nothing
    raw = reader.header.raw_header
    # TM5's own ECconv_to_TMconv converted the attached CATRINE archive.
    haskey(raw, "convection_attachment") && return nothing
    name = basename(reader.path)
    retired = "TM5 convection sections from the lat-lon ERA5 physics-file route, which " *
              "was removed: it placed the fields 180° off in longitude and one hour early " *
              "(issue #55) and used the ERA5 → TM5 indexing of issue #54. Regenerate the " *
              "binary without TM5 sections, or build TM5 convection with the ERA5 N320 → " *
              "cubed-sphere preprocessor (include_convection = true) or " *
              "scripts/preprocessing/attach_catrine_tm5_convection_cs.jl."
    if grid_type(reader.header) === :latlon && haskey(raw, "spectral_half_dt_seconds")
        throw(ArgumentError("$name carries $retired"))
    end
    if get(raw, "preprocessor", nothing) == "regrid_ll_binary_to_cs" &&
       !haskey(raw, TM5_CONVECTION_SOURCE_KEY)
        throw(ArgumentError("$name is a lat-lon → cubed-sphere regrid that carries $retired"))
    end
    convention = get(raw, TM5_EC2TM_CONVENTION_KEY, nothing)
    if convention !== nothing
        String(convention) == TM5_EC2TM_CONVENTION || throw(ArgumentError(
            "$name: TM5 convection sections record `$(TM5_EC2TM_CONVENTION_KEY) = " *
            "$(repr(convention))`, but this version reads only " *
            "$(repr(TM5_EC2TM_CONVENTION)). Regenerate the binary with the current preprocessor."))
        return nothing
    end
    if startswith(String(get(raw, TM5_CONVECTION_SOURCE_KEY, "")), "ec2tm")
        throw(ArgumentError(
            "$name: its TM5 convection sections were converted from ERA5 before the fix of " *
            "issue #54 (no `$(TM5_EC2TM_CONVENTION_KEY)` record): ERA5 UDMF/DDMF were placed " *
            "one interface too low and the cloud-top layer was dropped, so convection lifts " *
            "surface air too little. Regenerate the binary with the current preprocessor."))
    end
    return nothing
end

has_cmfmc(r::TransportBinaryReader) = :cmfmc in r.header.payload_sections
"Whether the binary carries the explicit DTRAIN payload (reader or capability summary)."
has_dtrain(r::TransportBinaryReader) = :dtrain in r.header.payload_sections
has_dtrain(caps) = :dtrain in caps.payload_sections
"Whether the binary carries the exact precomputed Dkg payload."
has_dkg(r::TransportBinaryReader) = :dkg in r.header.payload_sections
has_surface(r::TransportBinaryReader) =
    all(s in r.header.payload_sections for s in PBL_SURFACE_PAYLOAD_SECTIONS)
has_vdiff_fields(r::TransportBinaryReader) =
    all(s in r.header.payload_sections for s in GCHP_VDIFF_PAYLOAD_SECTIONS)
has_pbl_eflux(r::TransportBinaryReader) = :pbl_eflux in r.header.payload_sections
has_cmfmc_cloud_base(r::TransportBinaryReader) = :cmfmc_cloud_base in r.header.payload_sections

# ---------------------------------------------------------------------------
# Reader construction. Capability summaries and `inspect_binary` live in
# `inspect.jl`.
# ---------------------------------------------------------------------------

function TransportBinaryReader(bin_path::String; FT::Type{<:AbstractFloat} = Float32)
    io = open(bin_path, "r")
    try
        raw = read_transport_header_json(io; source = "transport binary $(bin_path)")

        # Validate the self-describing transport-binary contract before mmap.
        # `String(::Vector{UInt8})` may take ownership of and empty the vector;
        # keep `raw` intact for the typed header parser below.
        hdr_obj = JSON3.read(String(copy(raw)))
        hdr_dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr_obj))
        validate_transport_contract!(hdr_dict)
        length(raw) < Int(hdr_dict["header_bytes"]) || throw(ArgumentError(
            "transport binary JSON header is not null-terminated before header_bytes"))

        header = parse_transport_header(raw)
        DiskFT = transport_disk_float_type(header.on_disk_float_type)
        total_elems = header.n_geometry_elems + header.elems_per_window * header.nwindow
        expected_bytes = header.header_bytes + total_elems * sizeof(DiskFT)
        actual_bytes = filesize(bin_path)
        actual_bytes == expected_bytes || throw(ArgumentError(
            "transport binary size mismatch for $(bin_path): expected $(expected_bytes) bytes " *
            "from the header, found $(actual_bytes)"))
        data = Mmap.mmap(io, Vector{DiskFT}, total_elems, header.header_bytes)
        G = typeof(header.geometry)
        reader = TransportBinaryReader{FT, DiskFT, G}(data, io, header, bin_path)
        if !haskey(header.raw_header, "surface_pressure_basis")
            inferred = infer_surface_pressure_basis(reader)
            @warn "$(basename(bin_path)) has no `surface_pressure_basis` header key; " *
                  "inferred `$(transport_surface_pressure_basis_label(inferred))` once from " *
                  "window-1 surface pressure and air mass. Regenerate the binary before the " *
                  "legacy fallback becomes an error." maxlog=1
            header = parse_transport_header(raw; legacy_surface_pressure_basis=inferred)
            reader = TransportBinaryReader{FT, DiskFT, G}(data, io, header, bin_path)
        end
        _check_tm5_convection_provenance(reader)
        _check_tm5_convection_basis(reader)
        return reader
    catch
        isopen(io) && close(io)
        rethrow()
    end
end

"""Infer legacy `ps` basis from stored column masses using Float64 column sums.

The reader calls this once when a binary lacks `surface_pressure_basis`.
"""
function infer_surface_pressure_basis(reader::TransportBinaryReader)
    binary_mass_basis(reader) === :moist && return TotalSurfacePressure()
    grid = _legacy_surface_pressure_grid(reader, binary_geometry(reader))
    air_mass = section_view(reader, 1, :m)
    surface_pressure = section_view(reader, 1, :ps)
    return infer_surface_pressure_basis(
        grid, air_mass, surface_pressure, binary_geometry(reader))
end

_legacy_surface_pressure_grid(reader::TransportBinaryReader,
                              ::Union{LatLonBinaryGeometry,
                                      ReducedGaussianBinaryGeometry}) =
    load_grid(reader; FT=Float64)
_legacy_surface_pressure_grid(reader::TransportBinaryReader,
                              ::CubedSphereBinaryGeometry) =
    load_grid(reader; FT=Float64, Hp=0)

function infer_surface_pressure_basis(grid, air_mass, surface_pressure,
                                               ::LatLonBinaryGeometry)
    mesh = grid.horizontal
    g = Float64(gravity(grid))
    Nx, Ny = mesh.Nx, mesh.Ny
    ncell = Nx * Ny
    Nz = nlevels(grid)
    worst = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        c = i + (j - 1) * Nx
        m_column = sum(k -> Float64(air_mass[c + (k - 1) * ncell]), 1:Nz)
        p_dry = m_column * g / Float64(cell_area(mesh, i, j))
        ps = Float64(surface_pressure[c])
        worst = max(worst, abs(p_dry - ps) / max(abs(ps), 1.0))
    end
    return worst <= SURFACE_PRESSURE_BASIS_TOLERANCE ? DrySurfacePressure() : TotalSurfacePressure()
end

function infer_surface_pressure_basis(grid, air_mass, surface_pressure,
                                               ::ReducedGaussianBinaryGeometry)
    mesh = grid.horizontal
    g = Float64(gravity(grid))
    ncell = ncells(mesh)
    Nz = nlevels(grid)
    worst = 0.0
    @inbounds for c in 1:ncell
        m_column = sum(k -> Float64(air_mass[c + (k - 1) * ncell]), 1:Nz)
        p_dry = m_column * g / Float64(cell_area(mesh, c))
        ps = Float64(surface_pressure[c])
        worst = max(worst, abs(p_dry - ps) / max(abs(ps), 1.0))
    end
    return worst <= SURFACE_PRESSURE_BASIS_TOLERANCE ? DrySurfacePressure() : TotalSurfacePressure()
end

function infer_surface_pressure_basis(grid, air_mass, surface_pressure,
                                               geometry::CubedSphereBinaryGeometry)
    mesh = grid.horizontal
    g = Float64(gravity(grid))
    panel_cells = geometry.Nc * geometry.Nc
    Nz = nlevels(grid)
    worst = 0.0
    @inbounds for p in 1:geometry.npanel, j in 1:geometry.Nc, i in 1:geometry.Nc
        c = i + (j - 1) * geometry.Nc
        m_base = (p - 1) * panel_cells * Nz
        m_column = sum(k -> Float64(air_mass[m_base + c + (k - 1) * panel_cells]),
                       1:Nz)
        p_dry = m_column * g / Float64(cell_area(mesh, i, j))
        ps = Float64(surface_pressure[(p - 1) * panel_cells + c])
        worst = max(worst, abs(p_dry - ps) / max(abs(ps), 1.0))
    end
    return worst <= SURFACE_PRESSURE_BASIS_TOLERANCE ? DrySurfacePressure() : TotalSurfacePressure()
end

Base.close(r::TransportBinaryReader) = close(r.io)

@inline _transport_window_offset(r::TransportBinaryReader, win::Int) =
    r.header.n_geometry_elems + (win - 1) * r.header.elems_per_window

"""
    section_view(reader, win, section)

Return a zero-copy view of one payload `section` in window `win`. This is the
section-level reader primitive for metadata probes that do not need a complete
transport window (and must not allocate all of its flux and physics arrays).
"""
function section_view(reader::TransportBinaryReader,
                                 win::Int, wanted::Symbol)
    1 <= win <= reader.header.nwindow || throw(BoundsError(reader, win))
    o = _transport_window_offset(reader, win)
    for section in reader.header.payload_sections
        n = section_elements(reader.header, section)
        section === wanted && return @view reader.data[(o + 1):(o + n)]
        o += n
    end
    throw(ArgumentError("transport binary payload is missing section `$(wanted)`"))
end

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

_transport_allocate(reader::TransportBinaryReader{FT}, section::Symbol) where FT =
    Array{FT}(undef, section_shape(reader.header, section))

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

# Walk sections once; copies retain the disk-to-load conversion of copyto!.
function _load_sections!(buffers::NamedTuple, reader::TransportBinaryReader, win::Int)
    o = _transport_window_offset(reader, win)
    for section in reader.header.payload_sections
        n = section_elements(reader.header, section)
        if haskey(buffers, section)
            copyto!(getproperty(buffers, section), 1, reader.data, o + 1, n)
        end
        o += n
    end
    return nothing
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
    m = isnothing(m) ? _transport_allocate(reader, :m) : m
    ps = isnothing(ps) ? _transport_allocate(reader, :ps) : ps
    cm = isnothing(cm) ? _transport_allocate(reader, :cm) : cm

    if transport_is_structured(h)
        am = isnothing(am) ? _transport_allocate(reader, :am) : am
        bm = isnothing(bm) ? _transport_allocate(reader, :bm) : bm
        _load_sections!((; m, am, bm, cm, ps), reader, win)

        fluxes = _transport_make_fluxes(Val(h.mass_basis), am, bm, cm)
        return m, ps, fluxes
    elseif transport_is_faceindexed(h)
        hflux = isnothing(hflux) ? _transport_allocate(reader, :hflux) : hflux
        _load_sections!((; m, hflux, cm, ps), reader, win)

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

    dam = isnothing(dam) && transport_is_structured(h) ? _transport_allocate(reader, :dam) : dam
    dbm = isnothing(dbm) && transport_is_structured(h) ? _transport_allocate(reader, :dbm) : dbm
    dhflux = isnothing(dhflux) && transport_is_faceindexed(h) ? _transport_allocate(reader, :dhflux) : dhflux
    dcm = isnothing(dcm) ? _transport_allocate(reader, :dcm) : dcm
    dm = isnothing(dm) ? _transport_allocate(reader, :dm) : dm

    _load_sections!((; dam, dbm, dhflux, dcm, dm), reader, win)
    found_dam = :dam in h.payload_sections
    found_dbm = :dbm in h.payload_sections
    found_dhflux = :dhflux in h.payload_sections
    found_dcm = :dcm in h.payload_sections
    found_dm = :dm in h.payload_sections

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

    entu = isnothing(entu) ? _transport_allocate(reader, :entu) : entu
    detu = isnothing(detu) ? _transport_allocate(reader, :detu) : detu
    entd = isnothing(entd) ? _transport_allocate(reader, :entd) : entd
    detd = isnothing(detd) ? _transport_allocate(reader, :detd) : detd

    _load_sections!((; entu, detu, entd, detd), reader, win)
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
    has_surface(reader) || return nothing

    pblh = isnothing(pblh) ? _transport_allocate(reader, :pblh) : pblh
    ustar = isnothing(ustar) ? _transport_allocate(reader, :ustar) : ustar
    hflux = isnothing(hflux) ? _transport_allocate(reader, :pbl_hflux) : hflux
    t2m = isnothing(t2m) ? _transport_allocate(reader, :t2m) : t2m

    _load_sections!((; pblh, ustar, pbl_hflux=hflux, t2m), reader, win)
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
    qv_start = isnothing(qv_start) ? _transport_allocate(reader, :qv_start) : qv_start
    qv_end = isnothing(qv_end) ? _transport_allocate(reader, :qv_end) : qv_end

    _load_sections!((; qv_start, qv_end), reader, win)
    return (; qv_start, qv_end)
end
