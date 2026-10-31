_supports_pbl_diffusion(::TransportBinaryReader) = false
_supports_pbl_diffusion(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = has_surface(reader)

_supports_gchp_nonlocal_vdiff(reader::TransportBinaryReader) =
    _supports_gchp_vdiff(reader) && has_pbl_eflux(reader)

_supports_gchp_vdiff(::TransportBinaryReader) = false
_supports_gchp_vdiff(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = has_surface(reader) && has_vdiff_fields(reader)

function binary_dry_mass_target(reader::TransportBinaryReader)
    raw = reader.header.raw_header
    label = basename(reader.path)
    enabled = dry_mass_pin_enabled(raw, label)
    recorded = get(raw, "mass_fix_target_ps_dry_pa", nothing)
    if enabled === true && recorded !== nothing
        recorded isa Real && !(recorded isa Bool) || throw(ArgumentError(
            "$(label) records a non-numeric mass_fix_target_ps_dry_pa"))
        target = Float64(recorded)
        isfinite(target) && target > 0.0 || throw(ArgumentError(
            "$(label) has invalid mass_fix_target_ps_dry_pa=$(recorded)"))
        return (ps_dry_pa = target, source = :header, pinned = true)
    end

    m = section_view(reader, 1, :m)
    dry_mass = if binary_mass_basis(reader) === :dry
        sum(Float64, m)
    elseif has_qv_endpoints(reader)
        qv = section_view(reader, 1, :qv_start)
        length(qv) == length(m) || throw(DimensionMismatch(
            "window-1 qv_start length $(length(qv)) does not match mass length $(length(m))"))
        mapreduce(pair -> Float64(first(pair)) * (1.0 - Float64(last(pair))),
                  +, zip(m, qv); init = 0.0)
    else
        return (ps_dry_pa = nothing, source = :unclassifiable,
                pinned = enabled === false ? false : enabled)
    end
    area = 4pi * reader.header.planet_radius_m^2
    target = dry_mass * STANDARD_GRAVITY / area
    isfinite(target) && target > 0.0 || throw(ArgumentError(
        "could not infer a positive legacy dry-mass target from window 1 of $(reader.path)"))
    source = enabled === false ? :unpinned_window1 : :legacy_window1
    pin_state = enabled === true ? true : enabled === false ? false : nothing
    return (ps_dry_pa = target, source, pinned = pin_state)
end

const _TM5_CONVECTION_PROVENANCE_KEYS = (TM5_CONVECTION_SOURCE_KEY, TM5_EC2TM_CONVENTION_KEY,
                                         "convection_attachment")

function _tm5_convection_provenance(reader::TransportBinaryReader)
    has_tm5_convection(reader) || return Pair{String, String}[]
    raw = reader.header.raw_header
    return [key => string(raw[key]) for key in _TM5_CONVECTION_PROVENANCE_KEYS if haskey(raw, key)]
end

function binary_capabilities(reader::TransportBinaryReader)
    hdr = reader.header
    raw = hdr.raw_header
    required_advection = required_advection_sections(hdr.geometry)
    dry_target = binary_dry_mass_target(reader)
    return (
        advection        = all(s in hdr.payload_sections for s in required_advection),
        replay_gate      = has_flux_delta(reader),
        tm5_convection   = has_tm5_convection(reader),
        tm5_convection_mass_basis = tm5_convection_mass_basis(reader),
        tm5_convection_provenance = _tm5_convection_provenance(reader),
        cmfmc_convection = has_cmfmc(reader),
        pbl_diffusion    = _supports_pbl_diffusion(reader),
        gchp_vdiff       = _supports_gchp_vdiff(reader),
        gchp_nonlocal_vdiff = _supports_gchp_nonlocal_vdiff(reader),
        cmfmc_cloud_base = has_cmfmc_cloud_base(reader),
        surface_pressure = :ps in hdr.payload_sections,
        humidity         = has_qv_endpoints(reader),
        mass_basis       = hdr.mass_basis,
        surface_pressure_basis = surface_pressure_basis(reader),
        grid_type        = grid_type(hdr),
        format_version   = hdr.format_version,
        flux_construction = hdr.flux_construction,
        nlevel           = hdr.nlevel,
        nwindow          = hdr.nwindow,
        window_seconds   = hdr.dt_met_seconds,
        steps_per_window = hdr.steps_per_window,
        variable_step_schedule = has_variable_step_schedule(hdr.steps_per_window_by_window),
        flux_kind = flux_kind(reader),
        preprocessor_contract = get(raw, "preprocessor_contract", nothing),
        vertical_Nz_output = get(raw, "vertical_Nz_output", nothing),
        adaptive_substeps = get(raw, "adaptive_substeps", nothing),
        dry_mass_target_ps_pa = dry_target.ps_dry_pa,
        dry_mass_target_source = dry_target.source,
        dry_mass_pinned = dry_target.pinned,
        dry_mass_target_is_current = dry_target.ps_dry_pa !== nothing &&
            isapprox(dry_target.ps_dry_pa, GLOBAL_MEAN_DRY_SURFACE_PRESSURE;
                     atol = 0.1, rtol = 1e-7),
        # false only for binaries written with the write-time replay gate off
        write_replay_check = get(raw, "write_replay_check", true) !== false,
        payload_sections = hdr.payload_sections,
    )
end

function inspect_binary(path::AbstractString; io::IO = stdout)
    isfile(path) || throw(ArgumentError("binary not found: $(path)"))
    reader = TransportBinaryReader(String(path); FT = Float64)
    try
        caps = binary_capabilities(reader)
        println(io, reader)
        println(io)
        print_capability_rows(io, caps)
        return caps
    finally
        close(reader)
    end
end
