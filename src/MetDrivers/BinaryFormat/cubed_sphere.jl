# Cubed-sphere panel packing and streaming writer.

# The latent heat flux travels with the other PBL surface fields.
@inline _cs_window_has_pbl_eflux(window) =
    _transport_window_has_surface(window) && haskey(window, :surface) &&
    hasproperty(window.surface, :eflux) && window.surface.eflux !== nothing

"""
    _cs_window_section(window, section) -> NTuple of panels

Return the per-panel arrays that a CS window NamedTuple stores for payload
`section`, resolving PBL surface fields (`window.surface` or top-level
fields), TM5 convection (`window.tm5_fields`), and GCHP VDIFF
(`window.vdiff`) sub-tables.
"""
@inline function _cs_window_section(window, section::Symbol)
    section === :pbl_eflux && return window.surface.eflux
    if _is_pbl_surface_payload_section(section)
        if haskey(window, :surface) && window.surface !== nothing
            return getfield(window.surface, _pbl_surface_field_name(section))
        end
        field_name = _pbl_surface_field_name(section)
        if field_name === :hflux && haskey(window, :pbl_hflux)
            return getfield(window, :pbl_hflux)
        end
        return getfield(window, field_name)
    end
    if section === :entu || section === :detu ||
       section === :entd || section === :detd
        haskey(window, :tm5_fields) && window.tm5_fields !== nothing ||
            error("CS streaming window is missing `tm5_fields` required by section $(section)")
        return getfield(window.tm5_fields, section)
    end
    if _is_gchp_vdiff_payload_section(section)
        haskey(window, :vdiff) && window.vdiff !== nothing ||
            error("CS streaming window is missing `vdiff` required by section $(section)")
        return getfield(window.vdiff, _gchp_vdiff_field_name(section))
    end
    return getfield(window, section)
end

"""
    _pack_cs_window!(dest, offset, window, payload_sections, Nc, npanel)

Pack a CS window (with NTuple-of-panels fields) into a flat buffer.
Each section's panels are stored sequentially: [P1][P2]...[P6].
"""
function _pack_cs_window!(dest::Vector{FT}, offset::Int,
                           window, payload_sections::Vector{Symbol},
                           Nc::Int, npanel::Int) where FT
    o = offset
    for section in payload_sections
        panels = _cs_window_section(window, section)
        o = pack_panels!(dest, o, panels)
    end
    return nothing
end

function _validate_streaming_cs_window(writer::StreamingTransportBinaryWriter,
                                       window, Nc::Int, npanel::Int)
    expected_Nc = Int(writer.header["Nc"])
    expected_npanel = Int(writer.header["npanel"])
    Nc == expected_Nc || throw(DimensionMismatch(
        "CS write requested Nc=$(Nc); writer expects Nc=$(expected_Nc)"))
    npanel == expected_npanel || throw(DimensionMismatch(
        "CS write requested npanel=$(npanel); writer expects npanel=$(expected_npanel)"))

    haskey(window, :qv) && throw(ArgumentError(
        "CS transport-binary windows do not support section qv"))
    for section in (:qv_start, :qv_end, :dam, :dbm, :dhflux, :dcm)
        _transport_window_has_value(window, section) && throw(ArgumentError(
            "CS transport-binary windows do not support section $(section)"))
    end
    actual_sections = Symbol[:m, :am, :bm, :cm, :ps]
    _transport_window_has_value(window, :dm) && push!(actual_sections, :dm)
    _transport_window_has_surface(window) &&
        append!(actual_sections, PBL_SURFACE_PAYLOAD_SECTIONS)
    _transport_window_has_value(window, :cmfmc) && push!(actual_sections, :cmfmc)
    _transport_window_has_value(window, :dtrain) && push!(actual_sections, :dtrain)
    if haskey(window, :tm5_fields) && window.tm5_fields !== nothing
        append!(actual_sections, TM5_CONVECTION_PAYLOAD_SECTIONS)
    end
    _transport_window_has_vdiff_fields(window) &&
        append!(actual_sections, GCHP_VDIFF_PAYLOAD_SECTIONS)
    _transport_window_has_value(window, :dkg) && push!(actual_sections, :dkg)
    _cs_window_has_pbl_eflux(window) && push!(actual_sections, :pbl_eflux)
    _transport_window_has_value(window, :cmfmc_cloud_base) && push!(actual_sections, :cmfmc_cloud_base)
    actual_sections == writer.payload_sections || throw(ArgumentError(
        "CS transport-binary window payload sections $(actual_sections) do not match " *
        "the writer contract $(writer.payload_sections)"))

    for (index, section) in pairs(writer.payload_sections)
        panels = _cs_window_section(window, section)
        length(panels) == npanel || throw(DimensionMismatch(
            "CS section $(section) has $(length(panels)) panels; expected $(npanel)"))
        for panel in 1:npanel
            actual = collect(size(panels[panel]))
            expected = writer.section_shapes[index][panel]
            actual == expected || throw(DimensionMismatch(
                "CS section $(section) panel $(panel) has shape $(Tuple(actual)); expected $(Tuple(expected))"))
        end
    end
    return nothing
end


"""
    open_streaming_cs_transport_binary(path, Nc, npanel, nlevel, nwindow, vc;
                                       kwargs...) -> StreamingTransportBinaryWriter

Open a canonical cubed-sphere binary for streaming window writes (format
version 4, or 5 when the typed `flux_construction` keyword records a
flux-construction identity). `vc` is a `HybridSigmaPressure` vertical
coordinate; the payload is per-panel structured arrays with
`StructuredDirectional` topology. The header records the panel
convention (`"gnomonic"` or `"geos_native"`), the geometry law and
`planet_radius` (the radius of the mesh on which the air masses were computed),
so readers reconstruct the same mesh. Each subsequent window must match the
registered panel shapes and payload sections exactly.
The typed `surface_pressure_basis` keyword is required and independently
declares whether the stored `ps` panels are total or dry pressure.
"""
function open_streaming_cs_transport_binary(
        path::AbstractString,
        Nc::Int,
        npanel::Int,
        nlevel::Int,
        nwindow::Int,
        vc;
        FT::Type{<:AbstractFloat} = Float64,
        header_bytes::Int = 131072,
        dt_met_seconds::Real = DEFAULT_MET_INTERVAL_SECONDS,
        half_dt_seconds::Real = dt_met_seconds / 2,
        steps_per_window::Integer = 4,
        source_flux_sampling::Symbol = :window_start_endpoint,
        air_mass_sampling::Symbol = :window_start_endpoint,
        flux_sampling::Symbol = :window_constant,
        flux_kind::Symbol = :substep_mass_amount,
        include_flux_delta::Bool = false,
        mass_basis::Symbol,
        surface_pressure_basis::Union{TotalSurfacePressure, DrySurfacePressure},
        tm5_convection_mass_basis::Union{_DefaultTM5ConvectionMassBasis,
                                         Nothing, AbstractMassBasis} =
            _DEFAULT_TM5_CONVECTION_MASS_BASIS,
        include_cmfmc::Bool = false,
        include_dtrain::Bool = false,
        include_surface::Bool = false,
        include_tm5conv::Bool = false,
        include_gchp_vdiff::Bool = false,
        include_precomputed_dkg::Bool = false,
        include_pbl_eflux::Bool = false,
        include_cmfmc_cloud_base::Bool = false,
        panel_convention = "gnomonic",
        cs_definition = nothing,
        cs_coordinate_law = nothing,
        cs_center_law = nothing,
        longitude_offset_deg = nothing,
        planet_radius::Real,
        flux_construction::AbstractFluxConstructionHeader = NoFluxConstructionHeader(),
        extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}())
    include_dtrain && !include_cmfmc &&
        throw(ArgumentError("CS transport binaries cannot include dtrain without cmfmc"))
    include_cmfmc_cloud_base && !include_cmfmc &&
        throw(ArgumentError("CS transport binaries cannot include cmfmc_cloud_base without cmfmc"))
    include_pbl_eflux && !include_surface &&
        throw(ArgumentError("CS transport binaries cannot include pbl_eflux without the PBL surface sections"))
    include_precomputed_dkg && mass_basis !== :dry && throw(ArgumentError(
        "exact TM5 `:dkg` is defined on the runtime dry-air mass basis; " *
        "include_precomputed_dkg=true requires mass_basis=:dry"))

    ncell = npanel * Nc * Nc
    nface_h = npanel * 2 * Nc * (Nc + 1)
    payload_sections = Symbol[:m, :am, :bm, :cm, :ps]
    include_flux_delta && push!(payload_sections, :dm)
    if include_surface
        append!(payload_sections, PBL_SURFACE_PAYLOAD_SECTIONS)
    end
    include_cmfmc && push!(payload_sections, :cmfmc)
    include_dtrain && push!(payload_sections, :dtrain)
    if include_tm5conv
        append!(payload_sections, TM5_CONVECTION_PAYLOAD_SECTIONS)
    end
    include_gchp_vdiff && append!(payload_sections, GCHP_VDIFF_PAYLOAD_SECTIONS)
    include_precomputed_dkg && push!(payload_sections, :dkg)
    include_pbl_eflux && push!(payload_sections, :pbl_eflux)
    include_cmfmc_cloud_base && push!(payload_sections, :cmfmc_cloud_base)

    elems_per_window = sum(_cs_section_elements(Nc, npanel, nlevel, s)
                           for s in payload_sections)

    header = _transport_common_header("cubed_sphere", "StructuredDirectional",
                                      ncell, nface_h, nlevel, nwindow, vc,
                                      payload_sections, elems_per_window;
                                      FT=FT,
                                      planet_radius=planet_radius,
                                      header_bytes=header_bytes,
                                      dt_met_seconds=dt_met_seconds,
                                      half_dt_seconds=half_dt_seconds,
                                      steps_per_window=steps_per_window,
                                      mass_basis=mass_basis,
                                      surface_pressure_basis,
                                      source_flux_sampling=source_flux_sampling,
                                      air_mass_sampling=air_mass_sampling,
                                      flux_sampling=flux_sampling,
                                      flux_kind=flux_kind,
                                      humidity_sampling=:none,
                                      delta_semantics=include_flux_delta ?
                                          :forward_window_endpoint_difference : :none)

    panel_convention_norm = panel_convention_tag(parse_panel_convention(panel_convention))
    default_geometry = cs_geometry_tags(default_cs_definition(parse_panel_convention(panel_convention_norm)))

    revision = source_revision()
    target = poisson_balance_target(flux_kind, fill(steps_per_window, nwindow); steps_per_window)
    merge!(header, Dict{String, Any}(
        # --- Runtime CS contract keys (single source of truth) ---
        # Emitted HERE, at the one choke point every CS writer funnels through,
        # as DEFAULTS that writers override via `extra_header`. This makes it
        # structurally impossible for a CS writer to omit them (the recurring
        # N320-vs-GEOS drift). `runtime_substep_contract` is a format invariant:
        # every streaming CS binary carries the per-window substep schedule, so
        # the runtime MUST apply convection/chemistry once per met window, not
        # once per advection substep. Asserted below by
        # `validate_cs_writer_contract!`. See the 2026-05-31 contract audit.
        "runtime_substep_contract" => "binary_schedule",
        "preprocessor_contract" => "streaming_cs_v4",
        "git_commit" => revision.commit,                # code that wrote the binary
        "git_dirty" => revision.dirty != "clean",       # Bool, as in the lat-lon header
        "adaptive_substeps" => false,
        "Nc" => Nc,
        "npanel" => npanel,
        "panel_convention" => panel_convention_norm,
        "cs_definition" => something(cs_definition, default_geometry.cs_definition),
        "cs_coordinate_law" => something(cs_coordinate_law, default_geometry.cs_coordinate_law),
        "cs_center_law" => something(cs_center_law, default_geometry.cs_center_law),
        "longitude_offset_deg" => something(longitude_offset_deg, default_geometry.longitude_offset_deg),
        "Hp" => 0,
        "poisson_balance_method" => "global_cg_graph_laplacian",
        "poisson_balance_target_scale" => target.scale,
        "poisson_balance_target_semantics" => target.semantics,
        "poisson_balance_target_scale_by_window" => target.by_window,
        "n_dm" => include_flux_delta ? _cs_section_elements(Nc, npanel, nlevel, :dm) : 0,
        "include_cmfmc" => include_cmfmc,
        "include_dtrain" => include_dtrain,
        "include_surface" => include_surface,
        "surface_payload" => include_surface ? "pbl_raw_v2" : "none",
        "include_tm5conv" => include_tm5conv,
        "include_gchp_vdiff" => include_gchp_vdiff,
        "gchp_vdiff_payload" => include_gchp_vdiff ? "u_v_t_qv_layer_center_v1" : "none",
        "include_precomputed_dkg" => include_precomputed_dkg,
        "precomputed_dkg_payload" => include_precomputed_dkg ? "tm5_bldiff_interface_dkg_dry_v1" : "none",
        "n_dkg" => include_precomputed_dkg ? _cs_section_elements(Nc, npanel, nlevel, :dkg) : 0,
        "n_pblh" => include_surface ? _cs_section_elements(Nc, npanel, nlevel, :pblh) : 0,
        "n_ustar" => include_surface ? _cs_section_elements(Nc, npanel, nlevel, :ustar) : 0,
        "n_pbl_hflux" => include_surface ? _cs_section_elements(Nc, npanel, nlevel, :pbl_hflux) : 0,
        "n_t2m" => include_surface ? _cs_section_elements(Nc, npanel, nlevel, :t2m) : 0,
        "n_cmfmc" => include_cmfmc ? _cs_section_elements(Nc, npanel, nlevel, :cmfmc) : 0,
        "n_dtrain" => include_dtrain ? _cs_section_elements(Nc, npanel, nlevel, :dtrain) : 0,
        "n_entu" => include_tm5conv ? _cs_section_elements(Nc, npanel, nlevel, :entu) : 0,
        "n_detu" => include_tm5conv ? _cs_section_elements(Nc, npanel, nlevel, :detu) : 0,
        "n_entd" => include_tm5conv ? _cs_section_elements(Nc, npanel, nlevel, :entd) : 0,
        "n_detd" => include_tm5conv ? _cs_section_elements(Nc, npanel, nlevel, :detd) : 0,
        "n_vdiff_u" => include_gchp_vdiff ? _cs_section_elements(Nc, npanel, nlevel, :vdiff_u) : 0,
        "n_vdiff_v" => include_gchp_vdiff ? _cs_section_elements(Nc, npanel, nlevel, :vdiff_v) : 0,
        "n_vdiff_t" => include_gchp_vdiff ? _cs_section_elements(Nc, npanel, nlevel, :vdiff_t) : 0,
        "n_vdiff_qv" => include_gchp_vdiff ? _cs_section_elements(Nc, npanel, nlevel, :vdiff_qv) : 0,
        "n_pbl_eflux" => include_pbl_eflux ? _cs_section_elements(Nc, npanel, nlevel, :pbl_eflux) : 0,
        "n_cmfmc_cloud_base" => include_cmfmc_cloud_base ?
            _cs_section_elements(Nc, npanel, nlevel, :cmfmc_cloud_base) : 0,
        "cmfmc_cloud_base_payload" => include_cmfmc_cloud_base ? "top_down_layer_index_v1" : "none",
    ))
    _set_tm5_convection_mass_basis!(header, payload_sections, mass_basis,
                                    tm5_convection_mass_basis)
    _set_flux_construction!(header, flux_construction)
    isempty(extra_header) || _merge_transport_extra_header!(header, extra_header)
    header["panel_convention"] = panel_convention_tag(parse_panel_convention(header["panel_convention"]))

    validate_cs_writer_contract!(header)
    validate_transport_contract!(header)

    pack_buffer = Vector{FT}(undef, elems_per_window)
    section_shapes = [[copy(collect(_cs_section_shape(Nc, nlevel, section)))
                       for _ in 1:npanel] for section in payload_sections]
    staging_path, io = open_atomic_staging(path)
    try
        write_padded_header!(io, header, header_bytes)
    catch
        close(io)
        abandon_atomic_staging(staging_path)
        rethrow()
    end

    return StreamingTransportBinaryWriter{FT}(
        io, String(path), staging_path, header, payload_sections, elems_per_window,
        header_bytes, nwindow, 0, pack_buffer, section_shapes)
end

"""
    write_streaming_cs_window!(writer, window, Nc, npanel)

Pack and write a single CS window to the streaming transport binary.
`window` is a NamedTuple with NTuple-of-panels fields `:m`, `:am`, `:bm`, `:cm`, `:ps`.
"""
function write_streaming_cs_window!(writer::StreamingTransportBinaryWriter{FT},
                                     window, Nc::Int, npanel::Int) where FT
    writer.written_windows >= writer.expected_windows &&
        error("Already wrote $(writer.written_windows)/$(writer.expected_windows) windows")
    _validate_streaming_cs_window(writer, window, Nc, npanel)
    _pack_cs_window!(writer.pack_buffer, 0, window, writer.payload_sections, Nc, npanel)
    write(writer.io, writer.pack_buffer)
    writer.written_windows += 1
    return nothing
end
