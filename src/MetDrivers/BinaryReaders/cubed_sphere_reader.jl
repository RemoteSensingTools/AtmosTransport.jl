# Cubed-sphere specializations for `TransportBinaryReader`. Version-4 files
# share header parsing, validation, mmap ownership, accessors, and inspection
# with the other geometries; only panel layout and mesh reconstruction differ.



_transport_allocate(reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry},
                    section::Symbol) where {FT, DiskFT} =
    ntuple(_ -> Array{FT}(undef, section_shape(reader.header, section)), reader.header.geometry.npanel)

"""
    load_window!(reader, win) -> NamedTuple

Load window `win` from a cubed-sphere transport binary. Returns NTuples of
per-panel arrays plus optional `cmfmc` / `dtrain` payloads when they are
present in the binary.
"""
function load_window!(reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry},
                      win::Int) where {FT, DiskFT}
    h = reader.header
    g = h.geometry
    Nc, Nz, np = g.Nc, h.nlevel, g.npanel

    # Compute window offset in elements
    win_offset = _transport_window_offset(reader, win)

    panels_m  = _transport_allocate(reader, :m)
    panels_ps = _transport_allocate(reader, :ps)
    panels_am = _transport_allocate(reader, :am)
    panels_bm = _transport_allocate(reader, :bm)
    panels_cm = _transport_allocate(reader, :cm)
    surface_present = all(s in h.payload_sections for s in PBL_SURFACE_PAYLOAD_SECTIONS)
    panels_pblh  = surface_present ? _transport_allocate(reader, :pblh) : nothing
    panels_ustar = surface_present ? _transport_allocate(reader, :ustar) : nothing
    panels_hflux = surface_present ? _transport_allocate(reader, :pbl_hflux) : nothing
    panels_t2m   = surface_present ? _transport_allocate(reader, :t2m) : nothing
    panels_cmfmc = :cmfmc in h.payload_sections ? _transport_allocate(reader, :cmfmc) : nothing
    panels_dtrain = has_dtrain(reader) ? _transport_allocate(reader, :dtrain) : nothing
    vdiff_present = all(s in h.payload_sections for s in GCHP_VDIFF_PAYLOAD_SECTIONS)
    panels_vdiff_u  = vdiff_present ? _transport_allocate(reader, :vdiff_u) : nothing
    panels_vdiff_v  = vdiff_present ? _transport_allocate(reader, :vdiff_v) : nothing
    panels_vdiff_t  = vdiff_present ? _transport_allocate(reader, :vdiff_t) : nothing
    panels_vdiff_qv = vdiff_present ? _transport_allocate(reader, :vdiff_qv) : nothing

    dkg_present = has_dkg(reader)
    panels_dkg = dkg_present ? _transport_allocate(reader, :dkg) : nothing
    optional_2d = Dict(s => _transport_allocate(reader, s)
                       for s in CS_OPTIONAL_2D_SECTIONS if s in h.payload_sections)

    # TM5 convection fields — all four must be present together or
    # all four absent, as validated by the reader constructor.
    tm5_present = all(s in h.payload_sections for s in TM5_CONVECTION_PAYLOAD_SECTIONS)
    panels_entu = tm5_present ? _transport_allocate(reader, :entu) : nothing
    panels_detu = tm5_present ? _transport_allocate(reader, :detu) : nothing
    panels_entd = tm5_present ? _transport_allocate(reader, :entd) : nothing
    panels_detd = tm5_present ? _transport_allocate(reader, :detd) : nothing

    destinations = (m=panels_m, am=panels_am, bm=panels_bm, cm=panels_cm,
                    ps=panels_ps, pblh=panels_pblh, ustar=panels_ustar,
                    pbl_hflux=panels_hflux, t2m=panels_t2m,
                    cmfmc=panels_cmfmc, dtrain=panels_dtrain,
                    entu=panels_entu, detu=panels_detu, entd=panels_entd, detd=panels_detd,
                    vdiff_u=panels_vdiff_u, vdiff_v=panels_vdiff_v,
                    vdiff_t=panels_vdiff_t, vdiff_qv=panels_vdiff_qv, dkg=panels_dkg)
    o = win_offset
    for section in h.payload_sections
        panels = get(destinations, section, get(optional_2d, section, nothing))
        n = prod(section_shape(h, section))
        if panels !== nothing
            for p in 1:np
                copyto!(panels[p], 1, reader.data, o + (p - 1) * n + 1, n)
            end
        end
        o += section_elements(h, section)
    end

    # TM5 fields are returned as a NamedTuple when present, to match
    # the runtime `ConvectionForcing.tm5_fields` contract. Absent
    # means the binary doesn't carry TM5 convection data.
    tm5_fields = tm5_present ?
        (entu = panels_entu, detu = panels_detu,
         entd = panels_entd, detd = panels_detd) :
        nothing
    surface = surface_present ?
        PBLSurfaceForcing(panels_pblh, panels_ustar, panels_hflux, panels_t2m,
                          get(optional_2d, :pbl_eflux, nothing)) :
        nothing
    vdiff = vdiff_present ?
        (u = panels_vdiff_u, v = panels_vdiff_v,
         t = panels_vdiff_t, qv = panels_vdiff_qv) :
        nothing

    return (
        m = panels_m,
        ps = panels_ps,
        am = panels_am,
        bm = panels_bm,
        cm = panels_cm,
        surface = surface,
        cmfmc = panels_cmfmc,
        dtrain = panels_dtrain,
        tm5_fields = tm5_fields,
        vdiff = vdiff,
        dkg = dkg_present ? panels_dkg : nothing,
        cmfmc_cloud_base = get(optional_2d, :cmfmc_cloud_base, nothing),
    )
end

"""
    load_vdiff_qv_window!(dest, reader, win) -> dest

Copy the GCHP VDIFF specific humidity `:vdiff_qv` of window `win` (the
window-start humidity, interior panels `(Nc, Nc, Nz)`, k = 1 at the top) into
the host panels `dest` without loading the rest of the window. Observation
sampling reads it as the humidity at the end of the previous window.
"""
function load_vdiff_qv_window!(dest::NTuple{N, Array{<:AbstractFloat, 3}},
                               reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
                               win::Int) where {N}
    h = reader.header
    g = h.geometry
    N == g.npanel || throw(DimensionMismatch("$(N) humidity panels for $(g.npanel) binary panels"))
    all(p -> size(p) == (g.Nc, g.Nc, h.nlevel), dest) || throw(DimensionMismatch(
        "humidity panels must be $((g.Nc, g.Nc, h.nlevel)); got $(size.(dest))"))
    source = section_view(reader, win, :vdiff_qv)
    n = g.Nc * g.Nc * h.nlevel
    for p in 1:g.npanel
        copyto!(dest[p], 1, source, (p - 1) * n + 1, n)
    end
    return dest
end

function load_flux_delta_window!(
    reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry},
    win::Int; dm = nothing,
) where {FT, DiskFT}
    h = reader.header
    g = h.geometry
    :dm in h.payload_sections || return nothing

    dm = isnothing(dm) ?
        _transport_allocate(reader, :dm) : dm
    o = _transport_window_offset(reader, win)

    for section in h.payload_sections
        if section === :dm
            for p in 1:g.npanel
                n = g.Nc * g.Nc * h.nlevel
                copyto!(dm[p], 1, reader.data, o + 1, n)
                o += n
            end
            return (; dm)
        end
        o += section_elements(h, section)
    end

    return nothing
end

"""
    load_surface_window!(reader, win) -> PBLSurfaceForcing | nothing

Load the raw PBL surface payload for one CS window. This is a convenience
wrapper over `load_window!`; callers that already need the advection fields
should use `load_window!(reader, win).surface` to avoid a second payload read.
"""
load_surface_window!(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
    win::Int; kwargs...,
) = load_window!(reader, win).surface

"""
    mesh_convention(reader) -> AbstractCubedSpherePanelConvention

Return the panel-numbering convention declared in the binary header.

Returns `GnomonicPanelConvention()` for ERA5-CS binaries and
`GEOSNativePanelConvention()` for GEOS-FP/IT binaries tagged with
`panel_convention="geos_native"`. Callers should pass the result directly to
`CubedSphereMesh(; convention=mesh_convention(reader))` to guarantee that the
halo exchange uses the correct edge-to-edge connectivity table.
"""
function mesh_convention(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
)
    return parse_panel_convention(reader.header.geometry.panel_convention)
end

"""
    mesh_definition(reader) -> CubedSphereDefinition

Return the full cubed-sphere geometry definition declared in the binary
header. Binaries record `cs_coordinate_law`, `cs_center_law`,
`panel_convention`, and `longitude_offset_deg` explicitly.
"""
function mesh_definition(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
)
    g = reader.header.geometry
    return CubedSphereDefinition(parse_coordinate_law(g.coordinate_law),
                                 parse_center_law(g.center_law),
                                 mesh_convention(reader);
                                 longitude_offset_deg = g.longitude_offset_deg,
                                 tag = g.definition)
end

function load_grid(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry};
    FT::Type{<:AbstractFloat} = Float64,
    arch = CPU(),
    Hp::Int = 1,
)
    h = reader.header
    g = h.geometry
    vertical = HybridSigmaPressure(FT.(h.A_ifc), FT.(h.B_ifc))
    mesh = CubedSphereMesh(; FT, Nc=g.Nc, Hp, definition=mesh_definition(reader),
                           radius=FT(h.planet_radius_m))
    return AtmosGrid(mesh, vertical, arch; FT, radius=h.planet_radius_m)
end

export load_surface_window!, mesh_convention, mesh_definition
