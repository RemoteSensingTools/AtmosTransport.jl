# Read-only capability inspection for every version-4 geometry.
#
# `binary_capabilities(reader)` returns a NamedTuple describing what
# operators this binary can drive, so the CLI + physics-recipe validator
# can give precise errors ("config requested `tm5` but binary lacks
# entu/detu/entd/detd") instead of silently failing at the first step.
#
# `inspect_binary(path)` opens a `TransportBinaryReader` (header-contract and
# file-size checks), prints a report, and returns the capability summary. It
# does not run the load-time replay gate; `TransportBinaryDriver` does.
# `scripts/diagnostics/inspect_transport_binary.jl` calls it and then
# constructs a `TransportBinaryDriver` to probe runtime compatibility.

_required_advection_sections(::LatLonBinaryGeometry) = (:m, :am, :bm, :cm)
_required_advection_sections(::ReducedGaussianBinaryGeometry) = (:m, :hflux, :cm)
_required_advection_sections(::CubedSphereBinaryGeometry) = (:m, :am, :bm, :cm)

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

"""
    binary_dry_mass_target(reader) -> NamedTuple

Return the binary's global-mean dry-mass target in pascals and its provenance.
Pinned current binaries record `mass_fix_enabled=true` together with
`mass_fix_target_ps_dry_pa` in the header. Historical binaries may instead use
the source-specific `global_mass_pin_enabled` or
`geos_global_mass_pin_enabled` flag and omit the pressure target; those files
fall back to their carried window-1 dry mass with `pinned=true`. Unpinned and
keyless legacy dry-basis binaries use the same zero-copy fallback,
`sum(m_k) g / (4 pi R_header^2)`, on the radius recorded by the binary.

For moist-basis payloads the fallback computes
`sum(m_k * (1-qv_k)) g / (4 pi R_header^2)` from `qv_start`. Without endpoint
humidity the dry target is physically unclassifiable and `ps_dry_pa=nothing`;
total-air mass is never reported as dry-air mass.
"""
function _dry_mass_pin_enabled(raw::AbstractDict, label::AbstractString)
    enabled = nothing
    enabled_key = nothing
    for key in ("mass_fix_enabled", "global_mass_pin_enabled",
                "geos_global_mass_pin_enabled")
        haskey(raw, key) || continue
        value = raw[key]
        value isa Bool || throw(ArgumentError(
            "$(label) has non-Boolean $(key)=$(repr(value))"))
        if enabled === nothing
            enabled = value
            enabled_key = key
        elseif value != enabled
            throw(ArgumentError(
                "$(label) has conflicting dry-mass pin flags: " *
                "$(enabled_key)=$(enabled) but $(key)=$(value)"))
        end
    end
    return enabled
end

function binary_dry_mass_target(reader::TransportBinaryReader)
    raw = reader.header.raw_header
    label = basename(reader.path)
    enabled = _dry_mass_pin_enabled(raw, label)
    recorded = get(raw, "mass_fix_target_ps_dry_pa", nothing)
    if enabled === true && recorded !== nothing
        recorded isa Real && !(recorded isa Bool) || throw(ArgumentError(
            "$(label) records a non-numeric mass_fix_target_ps_dry_pa"))
        target = Float64(recorded)
        isfinite(target) && target > 0.0 || throw(ArgumentError(
            "$(label) has invalid mass_fix_target_ps_dry_pa=$(recorded)"))
        return (ps_dry_pa = target, source = :header, pinned = true)
    end

    m = _transport_section_view(reader, 1, :m)
    dry_mass = if binary_mass_basis(reader) === :dry
        sum(Float64, m)
    elseif has_qv_endpoints(reader)
        qv = _transport_section_view(reader, 1, :qv_start)
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

"""
    binary_capabilities(reader) -> NamedTuple

Summarise what operators this binary can drive. Geometry-specific advection
requirements are selected through the reader's geometry type. Fields:

- `advection :: Bool` — all geometry-required advection sections present
  (`m, am, bm, cm` on LL/CS; `m, hflux, cm` on RG).
- `replay_gate :: Bool` — any flux-delta section (dam/dbm/dcm/dm/dhflux) present.
- `tm5_convection :: Bool` — entu/detu/entd/detd all present.
- `tm5_convection_mass_basis :: Union{Nothing, AbstractMassBasis}` — typed
  basis of the TM5 sections (`DryBasis()`/`MoistBasis()`); `nothing` without
  TM5 sections or when the
  binary predates the record (moist fluxes, see `TM5_CONVECTION_MASS_BASIS_KEY`).
- `tm5_convection_provenance :: Vector{Pair{String, String}}` — the recorded
  source of the TM5 sections (`TM5_CONVECTION_SOURCE_KEY`,
  `TM5_EC2TM_CONVENTION_KEY`, `convection_attachment`), empty when none is
  recorded or the binary has no TM5 sections.
- `cmfmc_convection :: Bool` — cmfmc present (CS only; LL/RG returns false).
- `pbl_diffusion :: Bool` — complete runnable PBL forcing (CS only).
- `gchp_vdiff :: Bool` — complete runnable GCHP VDIFF forcing (CS only).
- `gchp_nonlocal_vdiff :: Bool` — GCHP VDIFF forcing plus latent heat flux
  `pbl_eflux`, as GEOS-Chem's non-local PBL scheme needs (CS only).
- `cmfmc_cloud_base :: Bool` — convective cloud base present.
- `surface_pressure :: Bool` — ps present.
- `humidity :: Bool` — qv_start/qv_end present.
- `mass_basis :: Symbol` — `:dry` or `:moist`.
- `surface_pressure_basis :: AbstractSurfacePressureBasis` — typed meaning of
  the stored `ps` payload.
- `grid_type :: Symbol` — `:latlon` / `:reduced_gaussian` / `:cubed_sphere`.
- `nlevel`, `steps_per_window` — header values.
- `variable_step_schedule :: Bool` — per-window substep counts differ.
- `flux_kind :: Symbol` — stored mass-flux normalization contract.
- `preprocessor_contract`, `vertical_Nz_output`, `adaptive_substeps` — raw
  header values, `nothing` when absent.
- `dry_mass_target_ps_pa`, `dry_mass_target_source`, `dry_mass_pinned` — the
  pinned header target or carried window-1 mass, its provenance, and pin state
  (`nothing` for legacy files whose state is not recorded).
- `payload_sections :: Vector{Symbol}` — raw set for debugging.
"""
function binary_capabilities(reader::TransportBinaryReader)
    hdr = reader.header
    raw = hdr.raw_header
    required_advection = _required_advection_sections(hdr.geometry)
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
        nlevel           = hdr.nlevel,
        nwindow          = hdr.nwindow,
        window_seconds   = hdr.dt_met_seconds,
        steps_per_window = hdr.steps_per_window,
        variable_step_schedule = _has_variable_step_schedule(hdr.steps_per_window_by_window),
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

"""
    inspect_binary(path; io = stdout) -> NamedTuple

Open the binary at `path`, print
a capability-augmented report to `io`, and return the
`binary_capabilities` NamedTuple for programmatic consumption (tests,
CLI capability-intersection, folder-level validation).

Obsolete `format_version < TRANSPORT_BINARY_FORMAT_VERSION` files are rejected
here just like runtime drivers; inspect the raw JSON header with external tools
if a stale file must be audited.
"""
function inspect_binary(path::AbstractString; io::IO = stdout)
    isfile(path) || throw(ArgumentError("binary not found: $(path)"))
    reader = TransportBinaryReader(String(path); FT = Float64)
    try
        caps = binary_capabilities(reader)
        println(io, reader)
        println(io)
        _print_capability_rows(io, caps)
        return caps
    finally
        close(reader)
    end
end

function _print_capability_rows(io::IO, caps)
    println(io, "Capabilities:")
    _print_cap(io, caps.advection,        "advection",        "(m, am, bm, cm)")
    _print_cap(io, caps.replay_gate,      "replay gate",      "(dam, dbm, dcm, dm)")
    _print_cap(io, caps.tm5_convection,   "TM5 convection",   "(entu, detu, entd, detd)")
    _print_cap(io, caps.cmfmc_convection, "CMFMC convection", "(cmfmc)")
    _print_cap(io, caps.cmfmc_cloud_base, "conv. cloud base", "(cmfmc_cloud_base)")
    _print_cap(io, caps.pbl_diffusion,    "PBL diffusion",    "(pblh, ustar, pbl_hflux, t2m)")
    _print_cap(io, caps.gchp_nonlocal_vdiff, "GCHP non-local VDIFF", "(+ vdiff_u/v/t/qv, pbl_eflux)")
    _print_cap(io, caps.surface_pressure, "surface pressure", "(ps)")
    _print_cap(io, caps.humidity,         "humidity",         "(qv_start, qv_end)")
    println(io, "  mass_basis       = ", caps.mass_basis)
    println(io, "  surface_pressure_basis = ",
            _transport_surface_pressure_basis_label(caps.surface_pressure_basis))
    if caps.tm5_convection
        println(io, "  tm5_convection_mass_basis = ",
                caps.tm5_convection_mass_basis === nothing ? "not recorded" :
                    basis_symbol(caps.tm5_convection_mass_basis))
        caps.tm5_convection_mass_basis === nothing && caps.mass_basis === :dry &&
            println(io, "  ⚠ TM5 convection fluxes predate the dry-air conversion: moist fluxes " *
                        "on dry air mass, convective rates about 1 % too large")
        isempty(caps.tm5_convection_provenance) &&
            println(io, "  TM5 convection source not recorded")
        for (key, value) in caps.tm5_convection_provenance
            println(io, "  ", key, " = ", value)
        end
    end
    println(io, "  grid_type        = ", caps.grid_type)
    source_note = caps.dry_mass_target_source === :header ? "pinned header" :
                  caps.dry_mass_target_source === :unpinned_window1 ? "unpinned window-1 mass" :
                  "legacy window-1 fallback"
    if caps.dry_mass_target_ps_pa === nothing
        println(io, "  dry-mass target  = unclassifiable (moist mass without qv_start)")
    else
        @printf(io, "  dry-mass target  = %.6f Pa (%s)%s\n",
                caps.dry_mass_target_ps_pa, source_note,
                caps.dry_mass_target_is_current ? "" :
                    @sprintf("  ⚠ current constant is %.1f Pa", GLOBAL_MEAN_DRY_SURFACE_PRESSURE))
    end
    caps.write_replay_check || println(io, "  ⚠ written with the write-time replay check disabled; " *
                                           "not for production runs")
end

@inline function _print_cap(io::IO, present::Bool, label::AbstractString, ingredients::AbstractString)
    mark = present ? "✓" : "✗"
    @printf(io, "  %s %-16s %s\n", mark, label, ingredients)
end
