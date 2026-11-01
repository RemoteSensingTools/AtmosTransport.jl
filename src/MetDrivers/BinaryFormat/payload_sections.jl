# One section -> shape-kind table shared by writers, readers, and allocation.
# Topology allow-lists remain separate: equal shapes do not imply equal contracts.
const _TRANSPORT_SECTION_SHAPES = (
    tm5_t=:cell, tm5_q=:cell, tm5_lsm=:surface, tm5_sr=:surface, tm5_fsr=:surface,
    tm5_u10=:surface, tm5_v10=:surface, tm5_ewss=:surface, tm5_nsss=:surface,
    tm5_sshf=:surface, tm5_slhf=:surface, tm5_oro=:surface,
    m=:cell, dm=:cell, qv_start=:cell, qv_end=:cell,
    am=:xface, dam=:xface, bm=:yface, dbm=:yface,
    hflux=:hface, dhflux=:hface, cm=:interface, dcm=:interface,
    ps=:surface, pblh=:surface, ustar=:surface, pbl_hflux=:surface,
    t2m=:surface, pbl_eflux=:surface, cmfmc_cloud_base=:surface,
    entu=:cell, detu=:cell, entd=:cell, detd=:cell,
    vdiff_u=:cell, vdiff_v=:cell, vdiff_t=:cell, vdiff_qv=:cell,
    dkg=:cell, dtrain=:cell, cmfmc=:interface,
)
const _STRUCTURED_SECTIONS = (TM5_PBL_PAYLOAD_SECTIONS...,:m, :dm, :qv_start, :qv_end, :am, :dam,
    :bm, :dbm, :cm, :dcm, :ps, :pblh, :ustar, :pbl_hflux, :t2m,
    :entu, :detu, :entd, :detd, :vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv)
const _FACEINDEXED_SECTIONS = (TM5_PBL_PAYLOAD_SECTIONS...,:m, :dm, :qv_start, :qv_end, :hflux,
    :dhflux, :cm, :dcm, :ps, :pblh, :ustar, :pbl_hflux, :t2m,
    :entu, :detu, :entd, :detd, :vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv)
const _CS_SECTIONS = (:m, :dm, :am, :dam, :bm, :dbm, :cm, :dcm, :ps,
    :pblh, :ustar, :pbl_hflux, :t2m, :pbl_eflux, :cmfmc_cloud_base,
    :entu, :detu, :entd, :detd, :vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv,
    :dkg, :dtrain, :cmfmc)

@inline function _structured_section_shape(Nx::Int, Ny::Int, nlevel::Int, section::Symbol)
    section in _STRUCTURED_SECTIONS || error("Unsupported structured payload section: $section")
    kind = getproperty(_TRANSPORT_SECTION_SHAPES, section)
    kind === :xface && return (Nx + 1, Ny, nlevel)
    kind === :yface && return (Nx, Ny + 1, nlevel)
    kind === :interface && return (Nx, Ny, nlevel + 1)
    kind === :surface && return (Nx, Ny)
    return (Nx, Ny, nlevel)
end

@inline function _faceindexed_section_shape(ncell::Int, nface_h::Int, nlevel::Int, section::Symbol)
    section in _FACEINDEXED_SECTIONS || error("Unsupported face-indexed payload section: $section")
    kind = getproperty(_TRANSPORT_SECTION_SHAPES, section)
    kind === :hface && return (nface_h, nlevel)
    kind === :interface && return (ncell, nlevel + 1)
    kind === :surface && return (ncell,)
    return (ncell, nlevel)
end

@inline function _cs_section_shape(Nc::Int, nlevel::Int, section::Symbol)
    section in _CS_SECTIONS || error("Unsupported CS section: $section")
    kind = getproperty(_TRANSPORT_SECTION_SHAPES, section)
    kind === :xface && return (Nc + 1, Nc, nlevel)
    kind === :yface && return (Nc, Nc + 1, nlevel)
    kind === :interface && return (Nc, Nc, nlevel + 1)
    kind === :surface && return (Nc, Nc)
    return (Nc, Nc, nlevel)
end

"""Lat-lon section element count before constructing a complete binary header."""
section_elements(::Type{LatLonBinaryGeometry}, Nx::Int, Ny::Int, nlevel::Int, section::Symbol) =
    prod(_structured_section_shape(Nx, Ny, nlevel, section))

"""Cubed-sphere section element count before constructing a complete header."""
section_elements(::Type{CubedSphereBinaryGeometry}, Nc::Int, npanel::Int, nlevel::Int, section::Symbol) =
    npanel * prod(_cs_section_shape(Nc, nlevel, section))

"""Array dimensions of a payload section; cubed-sphere dimensions describe one panel."""
section_shape(h::TransportBinaryHeader{LatLonBinaryGeometry}, section::Symbol) =
    _structured_section_shape(h.geometry.Nx, h.geometry.Ny, h.nlevel, section)
section_shape(h::TransportBinaryHeader{ReducedGaussianBinaryGeometry}, section::Symbol) =
    _faceindexed_section_shape(h.ncell, h.nface_h, h.nlevel, section)
section_shape(h::TransportBinaryHeader{CubedSphereBinaryGeometry}, section::Symbol) =
    _cs_section_shape(h.geometry.Nc, h.nlevel, section)

"""Number of stored elements in a section, including every cubed-sphere panel."""
section_elements(h::TransportBinaryHeader, section::Symbol) = prod(section_shape(h, section))
section_elements(h::TransportBinaryHeader{CubedSphereBinaryGeometry}, section::Symbol) =
    h.geometry.npanel * prod(section_shape(h, section))

_transport_structured_section_elements(Nx::Int, Ny::Int, ncell::Int, nlevel::Int, section::Symbol) =
    prod(_structured_section_shape(Nx, Ny, nlevel, section))
_transport_faceindexed_section_elements(ncell::Int, nface_h::Int, nlevel::Int, section::Symbol) =
    prod(_faceindexed_section_shape(ncell, nface_h, nlevel, section))
_cs_section_elements(Nc::Int, npanel::Int, nlevel::Int, section::Symbol) =
    section_elements(CubedSphereBinaryGeometry, Nc, npanel, nlevel, section)

const _FIELD_PRODUCER_HINT = Dict{Symbol, String}(
    :m => "Regenerate the transport binary with the unified preprocessor; `m` is a required air-mass section.",
    :state => "Provide `window.state.air_mass` when using the in-memory writer API.",
    :ps => "Regenerate with surface pressure enabled; `ps` is a required runtime surface-pressure section.",
    :hflux => "Regenerate with horizontal mass fluxes enabled; LL/RG writers provide `am`/`bm`, RG writers may provide `hflux`.",
    :fluxes => "Provide `window.fluxes` when using the in-memory writer API.",
    :am => "Regenerate with x-direction mass fluxes enabled; `am` is required for lat-lon transport.",
    :bm => "Regenerate with y-direction mass fluxes enabled; `bm` is required for lat-lon transport.",
    :cm => "Regenerate with vertical continuity closure enabled so `cm` is written.",
    :dam => "Regenerate with replay-gate x-flux deltas enabled.",
    :dbm => "Regenerate with replay-gate y-flux deltas enabled.",
    :dhflux => "Regenerate with replay-gate flux deltas enabled for reduced-Gaussian horizontal fluxes.",
    :dcm => "Regenerate with replay-gate vertical-flux deltas enabled.",
    :dm => "Regenerate with replay-gate air-mass deltas enabled.",
    :tm5_fields => "Set `[preprocessing] include_convection = true` / TM5-enabled preprocessing when writing TM5 convection sections.",
    :vdiff => "Set `[preprocessing] include_vdiff_fields = true` when writing GCHP VDIFF sections.",
)

function _missing_window_field_error(names::Symbol...)
    rendered = join(("`$(name)`" for name in names), " or ")
    hints = String[]
    for name in names
        hint = get(_FIELD_PRODUCER_HINT, name, "")
        isempty(hint) || push!(hints, hint)
    end
    suffix = isempty(hints) ? "" : " " * join(unique(hints), " ")
    return error("transport-binary window is missing $(rendered)." * suffix)
end

_transport_window_mass(window) =
    haskey(window, :m) ? window.m :
    haskey(window, :state) ? window.state.air_mass :
    _missing_window_field_error(:m, :state)

_transport_window_ps(window) =
    haskey(window, :ps) ? window.ps :
    _missing_window_field_error(:ps)

_transport_window_hflux(window) =
    haskey(window, :hflux) ? window.hflux :
    haskey(window, :fluxes) ? window.fluxes.horizontal_flux :
    _missing_window_field_error(:hflux, :fluxes)

_transport_window_am(window) =
    haskey(window, :am) ? window.am :
    haskey(window, :fluxes) ? window.fluxes.am :
    _missing_window_field_error(:am, :fluxes)

_transport_window_bm(window) =
    haskey(window, :bm) ? window.bm :
    haskey(window, :fluxes) ? window.fluxes.bm :
    _missing_window_field_error(:bm, :fluxes)

_transport_window_cm(window) =
    haskey(window, :cm) ? window.cm :
    haskey(window, :fluxes) ? window.fluxes.cm :
    _missing_window_field_error(:cm, :fluxes)

_transport_window_dam(window) = haskey(window, :dam) ? window.dam : _missing_window_field_error(:dam)
_transport_window_dbm(window) = haskey(window, :dbm) ? window.dbm : _missing_window_field_error(:dbm)
_transport_window_dhflux(window) = haskey(window, :dhflux) ? window.dhflux : _missing_window_field_error(:dhflux)
_transport_window_dcm(window) = haskey(window, :dcm) ? window.dcm : _missing_window_field_error(:dcm)
_transport_window_dm(window) = haskey(window, :dm) ? window.dm : _missing_window_field_error(:dm)

@inline function _transport_window_has_surface(window)
    if haskey(window, :surface)
        return window.surface !== nothing
    end
    pblh_present = haskey(window, :pblh)
    ustar_present = haskey(window, :ustar)
    hflux_present = haskey(window, :pbl_hflux) || haskey(window, :hflux)
    t2m_present = haskey(window, :t2m)
    pbl_marker_present = pblh_present || ustar_present ||
                         haskey(window, :pbl_hflux) || t2m_present
    if pbl_marker_present && !(pblh_present && ustar_present && hflux_present && t2m_present)
        throw(ArgumentError(
            "transport-binary surface payload is partial; provide all of " *
            "`pblh`, `ustar`, `pbl_hflux`, and `t2m`, or none of them."))
    end
    return pbl_marker_present
end

@inline function _transport_window_surface_field(window, name::Symbol)
    if haskey(window, :surface) && window.surface !== nothing
        return getfield(window.surface, name)
    elseif name === :hflux && haskey(window, :pbl_hflux)
        return getfield(window, :pbl_hflux)
    elseif haskey(window, name)
        return getfield(window, name)
    else
        error("transport-binary window is missing PBL surface field `$(name)`. " *
              "Set `[preprocessing] include_surface = true` when producing the binary.")
    end
end

# TM5 convection payload writers read a NamedTuple of four
# layer-center fields from the preprocessor window.  The
# preprocessor supplies `window.tm5_fields.entu`, `.detu`, `.entd`,
# `.detd`.  Errors loudly if the writer requested a TM5 section
# but the window didn't include `tm5_fields`.
@inline function _transport_window_tm5_field(window, name::Symbol)
    haskey(window, :tm5_fields) ||
        _missing_window_field_error(:tm5_fields)
    nt = window.tm5_fields
    hasproperty(nt, name) ||
        error("transport-binary window.tm5_fields is missing field `$(name)`")
    return getproperty(nt, name)
end

@inline function _transport_window_has_vdiff_fields(window)
    if haskey(window, :vdiff)
        return window.vdiff !== nothing
    end
    present = map(name -> haskey(window, Symbol(:vdiff_, name)), _GCHP_VDIFF_FIELD_NAMES)
    if any(present) && !all(present)
        throw(ArgumentError(
            "transport-binary GCHP VDIFF payload is partial; provide all of " *
            "`vdiff_u`, `vdiff_v`, `vdiff_t`, and `vdiff_qv`, or none of them."))
    end
    return any(present)
end

@inline function _transport_window_vdiff_field(window, name::Symbol)
    if haskey(window, :vdiff) && window.vdiff !== nothing
        return getfield(window.vdiff, name)
    end
    key = Symbol(:vdiff_, name)
    haskey(window, key) ||
        error("transport-binary window is missing GCHP VDIFF field `$(key)`. " *
              get(_FIELD_PRODUCER_HINT, :vdiff, ""))
    return getfield(window, key)
end

function _transport_window_field(window, section::Symbol)
    if section in TM5_PBL_PAYLOAD_SECTIONS
        return getproperty(window.tm5_pbl, TM5_PBL_FIELD_NAMES[findfirst(==(section), TM5_PBL_PAYLOAD_SECTIONS)])
    elseif section === :m
        return _transport_window_mass(window)
    elseif section === :am
        return _transport_window_am(window)
    elseif section === :bm
        return _transport_window_bm(window)
    elseif section === :hflux
        return _transport_window_hflux(window)
    elseif section === :cm
        return _transport_window_cm(window)
    elseif section === :dam
        return _transport_window_dam(window)
    elseif section === :dbm
        return _transport_window_dbm(window)
    elseif section === :dhflux
        return _transport_window_dhflux(window)
    elseif section === :dcm
        return _transport_window_dcm(window)
    elseif section === :dm
        return _transport_window_dm(window)
    elseif section === :ps
        return _transport_window_ps(window)
    elseif section === :qv_start
        return window.qv_start
    elseif section === :qv_end
        return window.qv_end
    elseif section === :pblh
        return _transport_window_surface_field(window, :pblh)
    elseif section === :ustar
        return _transport_window_surface_field(window, :ustar)
    elseif section === :pbl_hflux
        return _transport_window_surface_field(window, :hflux)
    elseif section === :t2m
        return _transport_window_surface_field(window, :t2m)
    # TM5 convection fields.
    elseif section === :entu
        return _transport_window_tm5_field(window, :entu)
    elseif section === :detu
        return _transport_window_tm5_field(window, :detu)
    elseif section === :entd
        return _transport_window_tm5_field(window, :entd)
    elseif section === :detd
        return _transport_window_tm5_field(window, :detd)
    elseif _is_gchp_vdiff_payload_section(section)
        return _transport_window_vdiff_field(window, _gchp_vdiff_field_name(section))
    else
        error("Unsupported transport-binary section: $section")
    end
end

@inline _transport_window_has_value(window, name::Symbol) =
    haskey(window, name) && getfield(window, name) !== nothing

function _transport_push_optional_sections!(sections::Vector{Symbol}, window)
    haskey(window, :qv) && throw(ArgumentError(
        "single-field qv is not part of the maintained version-4/5 contract; " *
        "provide qv_start and qv_end"))
    _transport_window_has_value(window, :qv_start) && push!(sections, :qv_start)
    _transport_window_has_value(window, :qv_end) && push!(sections, :qv_end)
    _transport_window_has_value(window, :dam) && push!(sections, :dam)
    _transport_window_has_value(window, :dbm) && push!(sections, :dbm)
    _transport_window_has_value(window, :dhflux) && push!(sections, :dhflux)
    _transport_window_has_value(window, :dcm) && push!(sections, :dcm)
    _transport_window_has_value(window, :dm) && push!(sections, :dm)
    if _transport_window_has_surface(window)
        push!(sections, :pblh)
        push!(sections, :ustar)
        push!(sections, :pbl_hflux)
        push!(sections, :t2m)
    end
    # TM5 convection adds a NamedTuple of four
    # layer-center fields.  Writer emits these when the preprocessor
    # window provides `tm5_fields`; reader populates
    # `ConvectionForcing.tm5_fields` from the corresponding binary
    # sections.
    if haskey(window, :tm5_fields) && window.tm5_fields !== nothing
        push!(sections, :entu)
        push!(sections, :detu)
        push!(sections, :entd)
        push!(sections, :detd)
    end
    if _transport_window_has_vdiff_fields(window)
        append!(sections, GCHP_VDIFF_PAYLOAD_SECTIONS)
    end
    if haskey(window, :tm5_pbl) && window.tm5_pbl !== nothing
        append!(sections, TM5_PBL_PAYLOAD_SECTIONS)
    end
    return sections
end

function _transport_payload_sections(::AtmosGrid{<:LatLonMesh}, window)
    return _transport_push_optional_sections!(Symbol[:m, :am, :bm, :cm, :ps], window)
end

function _transport_payload_sections(::AtmosGrid{<:ReducedGaussianMesh}, window)
    return _transport_push_optional_sections!(Symbol[:m, :hflux, :cm, :ps], window)
end

function _transport_validate_basis(window, basis_sym::Symbol)
    if haskey(window, :state)
        basis_symbol(mass_basis(window.state)) == basis_sym ||
            throw(ArgumentError("window state basis does not match requested transport binary basis $(basis_sym)"))
    end
    if haskey(window, :fluxes)
        basis_symbol(mass_basis(window.fluxes)) == basis_sym ||
            throw(ArgumentError("window flux basis does not match requested transport binary basis $(basis_sym)"))
    end
    return nothing
end

function _transport_validate_humidity_endpoints(window, expected)
    if _transport_window_has_value(window, :qv_start)
        size(window.qv_start) == expected ||
            throw(DimensionMismatch("window qv_start has size $(size(window.qv_start)), expected $(expected)"))
    end
    if _transport_window_has_value(window, :qv_end)
        size(window.qv_end) == expected ||
            throw(DimensionMismatch("window qv_end has size $(size(window.qv_end)), expected $(expected)"))
    end
    return nothing
end

function _transport_validate_optional_surface(window, expected)
    _transport_window_has_surface(window) || return nothing
    for name in _PBL_SURFACE_FIELD_NAMES
        field = _transport_window_surface_field(window, name)
        size(field) == expected ||
            throw(DimensionMismatch("window $(name) has size $(size(field)), expected $(expected)"))
    end
    return nothing
end

function _transport_validate_optional_vdiff(window, expected)
    _transport_window_has_vdiff_fields(window) || return nothing
    for name in _GCHP_VDIFF_FIELD_NAMES
        field = _transport_window_vdiff_field(window, name)
        size(field) == expected ||
            throw(DimensionMismatch("window vdiff_$(name) has size $(size(field)), expected $(expected)"))
    end
    return nothing
end

function _transport_validate_optional_structured_deltas(window, Nx::Int, Ny::Int, nlevel::Int)
    if _transport_window_has_value(window, :dam)
        size(window.dam) == (Nx + 1, Ny, nlevel) ||
            throw(DimensionMismatch("window dam has size $(size(window.dam)), expected $((Nx + 1, Ny, nlevel))"))
    end
    if _transport_window_has_value(window, :dbm)
        size(window.dbm) == (Nx, Ny + 1, nlevel) ||
            throw(DimensionMismatch("window dbm has size $(size(window.dbm)), expected $((Nx, Ny + 1, nlevel))"))
    end
    if _transport_window_has_value(window, :dcm)
        size(window.dcm) == (Nx, Ny, nlevel + 1) ||
            throw(DimensionMismatch("window dcm has size $(size(window.dcm)), expected $((Nx, Ny, nlevel + 1))"))
    end
    if _transport_window_has_value(window, :dm)
        size(window.dm) == (Nx, Ny, nlevel) ||
            throw(DimensionMismatch("window dm has size $(size(window.dm)), expected $((Nx, Ny, nlevel))"))
    end
    return nothing
end

function _transport_validate_optional_faceindexed_deltas(window, ncell::Int, nface_h::Int, nlevel::Int)
    if _transport_window_has_value(window, :dhflux)
        size(window.dhflux) == (nface_h, nlevel) ||
            throw(DimensionMismatch("window dhflux has size $(size(window.dhflux)), expected $((nface_h, nlevel))"))
    end
    if _transport_window_has_value(window, :dcm)
        size(window.dcm) == (ncell, nlevel + 1) ||
            throw(DimensionMismatch("window dcm has size $(size(window.dcm)), expected $((ncell, nlevel + 1))"))
    end
    if _transport_window_has_value(window, :dm)
        size(window.dm) == (ncell, nlevel) ||
            throw(DimensionMismatch("window dm has size $(size(window.dm)), expected $((ncell, nlevel))"))
    end
    return nothing
end

function _transport_validate_structured_window(window,
                                               Nx::Int, Ny::Int, nlevel::Int,
                                               basis_sym::Symbol)
    m = _transport_window_mass(window)
    am = _transport_window_am(window)
    bm = _transport_window_bm(window)
    cm = _transport_window_cm(window)
    ps = _transport_window_ps(window)

    size(m) == (Nx, Ny, nlevel) ||
        throw(DimensionMismatch("window m has size $(size(m)), expected ($(Nx), $(Ny), $(nlevel))"))
    size(am) == (Nx + 1, Ny, nlevel) ||
        throw(DimensionMismatch("window am has size $(size(am)), expected ($(Nx + 1), $(Ny), $(nlevel))"))
    size(bm) == (Nx, Ny + 1, nlevel) ||
        throw(DimensionMismatch("window bm has size $(size(bm)), expected ($(Nx), $(Ny + 1), $(nlevel))"))
    size(cm) == (Nx, Ny, nlevel + 1) ||
        throw(DimensionMismatch("window cm has size $(size(cm)), expected ($(Nx), $(Ny), $(nlevel + 1))"))
    size(ps) == (Nx, Ny) ||
        throw(DimensionMismatch("window ps has size $(size(ps)), expected ($(Nx), $(Ny))"))

    _transport_validate_humidity_endpoints(window, (Nx, Ny, nlevel))
    _transport_validate_optional_structured_deltas(window, Nx, Ny, nlevel)
    _transport_validate_optional_surface(window, (Nx, Ny))
    _transport_validate_optional_vdiff(window, (Nx, Ny, nlevel))
    _transport_validate_tm5_pbl(window, size(m), size(ps))
    _transport_validate_basis(window, basis_sym)
    return nothing
end

function _transport_validate_reduced_window(window,
                                            ncell::Int, nface_h::Int, nlevel::Int,
                                            basis_sym::Symbol)
    m = _transport_window_mass(window)
    hflux = _transport_window_hflux(window)
    cm = _transport_window_cm(window)
    ps = _transport_window_ps(window)

    size(m) == (ncell, nlevel) ||
        throw(DimensionMismatch("window m has size $(size(m)), expected ($(ncell), $(nlevel))"))
    size(hflux) == (nface_h, nlevel) ||
        throw(DimensionMismatch("window hflux has size $(size(hflux)), expected ($(nface_h), $(nlevel))"))
    size(cm) == (ncell, nlevel + 1) ||
        throw(DimensionMismatch("window cm has size $(size(cm)), expected ($(ncell), $(nlevel + 1))"))
    size(ps) == (ncell,) ||
        throw(DimensionMismatch("window ps has size $(size(ps)), expected ($(ncell),)"))

    _transport_validate_humidity_endpoints(window, (ncell, nlevel))
    _transport_validate_optional_faceindexed_deltas(window, ncell, nface_h, nlevel)
    _transport_validate_optional_surface(window, (ncell,))
    _transport_validate_optional_vdiff(window, (ncell, nlevel))
    _transport_validate_tm5_pbl(window, size(m), size(ps))
    _transport_validate_basis(window, basis_sym)
    return nothing
end

function _transport_validate_tm5_pbl(window, cell_shape, surface_shape)
    (!haskey(window, :tm5_pbl) || window.tm5_pbl === nothing) && return nothing
    return validate_tm5_pbl_inputs(window.tm5_pbl,cell_shape,surface_shape)
end

"Validate complete live-Calc_Kzz input shapes, finite values, humidity, land fraction and land roughness."
function validate_tm5_pbl_inputs(pbl,cell_shape,surface_shape)
    length(propertynames(pbl)) == length(TM5_PBL_FIELD_NAMES) &&
        all(name -> hasproperty(pbl,name),TM5_PBL_FIELD_NAMES) || throw(ArgumentError(
        "tm5_pbl must supply the complete field group $(TM5_PBL_FIELD_NAMES)"))
    for name in TM5_PBL_FIELD_NAMES
        a = getproperty(pbl, name)
        expected = name in (:t, :q) ? cell_shape : surface_shape
        size(a) == expected || throw(DimensionMismatch("tm5_pbl.$name expected $expected, got $(size(a))"))
        all(isfinite, a) || throw(ArgumentError("tm5_pbl.$name contains non-finite values"))
    end
    all(>(0), pbl.t) || throw(ArgumentError("tm5_pbl.t must be positive Kelvin"))
    all(x -> 0 <= x < 1, pbl.q) || throw(ArgumentError("tm5_pbl.q must be in [0,1)"))
    all(x -> 0 <= x <= 1, pbl.lsm) || throw(ArgumentError("tm5_pbl.lsm must be a fraction in [0,1]"))
    for i in eachindex(pbl.lsm)
        if pbl.lsm[i] > 0
            pbl.sr[i] > 0 && 0 <= pbl.fsr[i] < 10 || throw(ArgumentError(
                "TM5 land roughness requires sr > 0 and 0 <= fsr < 10 m"))
        end
    end
    return nothing
end
