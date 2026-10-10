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
    binary_capabilities(reader) -> NamedTuple

Summarise what operators this binary can drive. Geometry-specific advection
requirements are selected through the reader's geometry type. Fields:

- `advection :: Bool` — all geometry-required advection sections present
  (`m, am, bm, cm` on LL/CS; `m, hflux, cm` on RG).
- `replay_gate :: Bool` — any flux-delta section (dam/dbm/dcm/dm/dhflux) present.
- `tm5_convection :: Bool` — entu/detu/entd/detd all present.
- `cmfmc_convection :: Bool` — cmfmc present (CS only; LL/RG returns false).
- `pbl_diffusion :: Bool` — complete runnable PBL forcing (CS only).
- `gchp_vdiff :: Bool` — complete runnable GCHP VDIFF forcing (CS only).
- `gchp_nonlocal_vdiff :: Bool` — GCHP VDIFF forcing plus latent heat flux
  `pbl_eflux`, as GEOS-Chem's non-local PBL scheme needs (CS only).
- `cmfmc_cloud_base :: Bool` — convective cloud base present.
- `surface_pressure :: Bool` — ps present.
- `humidity :: Bool` — qv_start/qv_end present.
- `mass_basis :: Symbol` — `:dry` or `:moist`.
- `grid_type :: Symbol` — `:latlon` / `:reduced_gaussian` / `:cubed_sphere`.
- `nlevel`, `steps_per_window` — header values.
- `variable_step_schedule :: Bool` — per-window substep counts differ.
- `flux_kind :: Symbol` — stored mass-flux normalization contract.
- `preprocessor_contract`, `vertical_Nz_output`, `adaptive_substeps` — raw
  header values, `nothing` when absent.
- `payload_sections :: Vector{Symbol}` — raw set for debugging.
"""
function binary_capabilities(reader::TransportBinaryReader)
    hdr = reader.header
    raw = hdr.raw_header
    required_advection = _required_advection_sections(hdr.geometry)
    return (
        advection        = all(s in hdr.payload_sections for s in required_advection),
        replay_gate      = has_flux_delta(reader),
        tm5_convection   = has_tm5_convection(reader),
        cmfmc_convection = has_cmfmc(reader),
        pbl_diffusion    = _supports_pbl_diffusion(reader),
        gchp_vdiff       = _supports_gchp_vdiff(reader),
        gchp_nonlocal_vdiff = _supports_gchp_nonlocal_vdiff(reader),
        cmfmc_cloud_base = has_cmfmc_cloud_base(reader),
        surface_pressure = :ps in hdr.payload_sections,
        humidity         = has_qv_endpoints(reader),
        mass_basis       = hdr.mass_basis,
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
        println(io, reader)
        println(io)
        _print_capability_rows(io, reader)
        return binary_capabilities(reader)
    finally
        close(reader)
    end
end

function _print_capability_rows(io::IO, reader)
    caps = binary_capabilities(reader)
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
    println(io, "  grid_type        = ", caps.grid_type)
end

@inline function _print_cap(io::IO, present::Bool, label::AbstractString, ingredients::AbstractString)
    mark = present ? "✓" : "✗"
    @printf(io, "  %s %-16s %s\n", mark, label, ingredients)
end
