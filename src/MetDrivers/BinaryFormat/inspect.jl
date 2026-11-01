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
- `format_version :: Int` — header format version (4 or 5).
- `flux_construction :: AbstractFluxConstructionHeader` — recorded
  flux-construction identity; `NoFluxConstructionHeader()` for version 4.
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
function binary_capabilities end
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
function binary_dry_mass_target end
"""
    inspect_binary(path; io = stdout) -> NamedTuple

Open the binary at `path`, print
a capability-augmented report to `io`, and return the
`binary_capabilities` NamedTuple for programmatic consumption (tests,
CLI capability-intersection, folder-level validation).

Files whose `format_version` is not in `TRANSPORT_BINARY_FORMAT_VERSIONS` are
rejected here just like runtime drivers; inspect the raw JSON header with
external tools if a stale file must be audited.
"""
function inspect_binary end

# Read-only capability inspection for every version-4/5 geometry.
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

required_advection_sections(::LatLonBinaryGeometry) = (:m, :am, :bm, :cm)
required_advection_sections(::ReducedGaussianBinaryGeometry) = (:m, :hflux, :cm)
required_advection_sections(::CubedSphereBinaryGeometry) = (:m, :am, :bm, :cm)

function dry_mass_pin_enabled(raw::AbstractDict, label::AbstractString)
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

function print_capability_rows(io::IO, caps)
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
            transport_surface_pressure_basis_label(caps.surface_pressure_basis))
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
    println(io, "  format_version   = ", caps.format_version)
    println(io, "  flux_construction = ", flux_construction_summary(caps.flux_construction))
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
