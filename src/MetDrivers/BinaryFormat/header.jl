# Typed geometry metadata, common header schema, and version-4/5 parsing.

const PBL_SURFACE_PAYLOAD_SECTIONS = (:pblh, :ustar, :pbl_hflux, :t2m)
const _PBL_SURFACE_FIELD_NAMES = (:pblh, :ustar, :hflux, :t2m)
const GCHP_VDIFF_PAYLOAD_SECTIONS = (:vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv)
const _GCHP_VDIFF_FIELD_NAMES = (:u, :v, :t, :qv)
"Optional v5 live Calc_Kzz input group; fractional lsm and ECMWF downward-positive heat fluxes."
const TM5_PBL_FIELD_NAMES = (:t, :q, :lsm, :sr, :fsr, :u10, :v10, :ewss, :nsss, :sshf, :slhf, :oro)
"All-or-nothing v5 payload sections for live TM5 Calc_Kzz."
const TM5_PBL_PAYLOAD_SECTIONS = map(n -> Symbol(:tm5_, n), TM5_PBL_FIELD_NAMES)

# Optional 2-D cubed-sphere sections: surface latent heat flux for the GCHP
# non-local PBL scheme, and the convective cloud-base layer (top-down index).
const CS_OPTIONAL_2D_SECTIONS = (:pbl_eflux, :cmfmc_cloud_base)

# Sections stored as one (Nc, Nc) field per panel.
@inline _is_cs_2d_section(section::Symbol) =
    section === :ps || _is_pbl_surface_payload_section(section) || section in CS_OPTIONAL_2D_SECTIONS
const TM5_CONVECTION_PAYLOAD_SECTIONS = (:entu, :detu, :entd, :detd)
const FLUX_DELTA_PAYLOAD_SECTIONS = (:dam, :dbm, :dcm, :dm, :dhflux)

"""
Transport-binary format version of a header that records no version-5 item.
Every writer emits it unless it is given a flux-construction identity, so all
existing outputs stay version 4.
"""
const TRANSPORT_BINARY_FORMAT_VERSION = 4

"""
Transport-binary format version of a header that records the flux-construction
identity (`flux_construction` and its keys, see
[`AbstractFluxConstructionHeader`](@ref)). Version 5 is version 4 plus these
keys: payload layout, timing and sampling are unchanged.
"""
const FLUX_CONSTRUCTION_FORMAT_VERSION = 5

"""Every transport-binary `format_version` this runtime reads."""
const TRANSPORT_BINARY_FORMAT_VERSIONS =
    (TRANSPORT_BINARY_FORMAT_VERSION, FLUX_CONSTRUCTION_FORMAT_VERSION)

# Reject a format version this runtime cannot read: older files are obsolete
# and must be regenerated; newer ones need a newer runtime.
function _check_transport_format_version(format_version::Integer)
    format_version in TRANSPORT_BINARY_FORMAT_VERSIONS && return Int(format_version)
    supported = join(TRANSPORT_BINARY_FORMAT_VERSIONS, " and ")
    format_version < first(TRANSPORT_BINARY_FORMAT_VERSIONS) && throw(ArgumentError(
        "Obsolete transport binary format_version=$(format_version); this runtime reads " *
        "format_version $(supported). Regenerate this file with the current preprocessor " *
        "so the header carries the per-window substep schedule and runtime contract."))
    throw(ArgumentError(
        "Unsupported transport binary format_version=$(format_version); this runtime reads " *
        "format_version $(supported). The file was written by a newer preprocessor."))
end

@inline _is_pbl_surface_payload_section(section::Symbol) =
    section in PBL_SURFACE_PAYLOAD_SECTIONS
@inline _pbl_surface_field_name(section::Symbol) =
    section === :pbl_hflux ? :hflux : section
@inline _is_gchp_vdiff_payload_section(section::Symbol) =
    section in GCHP_VDIFF_PAYLOAD_SECTIONS
@inline function _gchp_vdiff_field_name(section::Symbol)
    section === :vdiff_u  && return :u
    section === :vdiff_v  && return :v
    section === :vdiff_t  && return :t
    section === :vdiff_qv && return :qv
    throw(ArgumentError("not a GCHP VDIFF payload section: $(section)"))
end

"""
    AbstractTransportBinaryGeometry

Geometry metadata embedded in a transport binary (format versions 4 and 5).
Concrete geometry types carry only the fields meaningful for their topology;
reader and grid construction code dispatches on this type instead of branching
on string tags.
"""
abstract type AbstractTransportBinaryGeometry end

"""
Header-key prefixes of the write-time certificate (flux and OMEGA fidelity,
balance correction, column closure). The certificate is one block: writers
patch all of it at close, and a caller that supplies a certificate through
`extra_header` (e.g. the TM5 convection attachment copying its source's)
replaces the writer's default block as a whole.
"""
const FLUX_CERTIFICATE_KEY_PREFIXES =
    ("flux_fidelity_", "omega_fidelity_", "balance_correction_", "column_closure_")

is_certificate_key(key::AbstractString) =
    any(prefix -> startswith(key, prefix), FLUX_CERTIFICATE_KEY_PREFIXES)

# Write-time flux-fidelity certification is metadata, but it is deliberately
# parsed into nominals rather than Symbols.  This keeps unknown/new spellings
# fail-closed and lets downstream code dispatch on the result without string or
# Symbol switches.
abstract type AbstractFidelityStatus end
struct FidelityPassed        <: AbstractFidelityStatus end
struct FidelityFailed        <: AbstractFidelityStatus end
struct FidelityNotApplicable <: AbstractFidelityStatus end
struct FidelityUnavailable   <: AbstractFidelityStatus end

abstract type AbstractFluxFidelityKind end
struct WindRederivedFidelity  <: AbstractFluxFidelityKind end
struct NativePassthroughFidelity <: AbstractFluxFidelityKind end
struct NoFluxFidelityKind     <: AbstractFluxFidelityKind end

"""On-disk spelling of a typed fidelity status or flux kind."""
fidelity_tag(::FidelityPassed) = "passed"
fidelity_tag(::FidelityFailed) = "failed"
fidelity_tag(::FidelityNotApplicable) = "not_applicable"
fidelity_tag(::FidelityUnavailable) = "unavailable"
fidelity_tag(::WindRederivedFidelity) = "wind_rederived"
fidelity_tag(::NativePassthroughFidelity) = "native_passthrough"
fidelity_tag(::NoFluxFidelityKind) = "none"

"Deterministic flux-certificate sample seed; part of the binary contract."
const FLUX_FIDELITY_SAMPLE_SEED = UInt64(0x6a09e667f3bcc909)

struct FidelityReason
    code::String
    function FidelityReason(code::AbstractString)
        isempty(code) && throw(ArgumentError("fidelity reason must not be empty"))
        new(String(code))
    end
end

"""
Typed horizontal-flux certificate decoded from the flat binary header.

`status` and `kind` are nominal types. A `FidelityPassed` certificate is
always `WindRederivedFidelity` and carries every numeric metric (finite);
`NativePassthroughFidelity` (GEOS native fluxes) is `FidelityNotApplicable`
and carries only the balance-correction metrics; `NoFluxFidelityKind` has no
samples. `time_sampling` is the windows' common effective sampling or
`"mixed"`, in which case `time_sampling_by_window` lists every window's.
The balance-correction metrics measure the column-balance correction
`Δ = F_balanced − F` of the sampled pre-balance amounts `F`, flux-weighted:
`balance_correction_rms_relative = RMS(Δ)/RMS(F)` over all samples, and the
largest per-level value with its level (`balance_correction_max_level`,
1 = top).
"""
struct FluxFidelityHeader{S<:AbstractFidelityStatus,K<:AbstractFluxFidelityKind}
    schema::Int
    status::S
    kind::K
    reason::Union{Nothing,FidelityReason}
    sample_rule::String
    sample_seed::String
    samples_per_window::Int
    sample_count::Int
    windows::Int
    method::String
    time_sampling::String
    time_sampling_by_window::Vector{String}
    amount_seconds_min::Union{Nothing,Float64}
    amount_seconds_max::Union{Nothing,Float64}
    max_abs_kg::Union{Nothing,Float64}
    rms_abs_kg::Union{Nothing,Float64}
    max_rel::Union{Nothing,Float64}
    rms_rel::Union{Nothing,Float64}
    max_ulp::Union{Nothing,UInt64}
    abs_tol_kg::Union{Nothing,Float64}
    rel_tol::Union{Nothing,Float64}
    scale_floor_kg::Union{Nothing,Float64}
    worst_window::Union{Nothing,Int}
    worst_face::Union{Nothing,Int}
    worst_level::Union{Nothing,Int}
    balance_correction_max_abs_kg::Union{Nothing,Float64}
    balance_correction_rms_relative::Union{Nothing,Float64}
    balance_correction_max_level_rms_relative::Union{Nothing,Float64}
    balance_correction_max_level::Union{Nothing,Int}
end

"""
Typed source-OMEGA certificate decoded from the flat binary header. A
`FidelityPassed` certificate has samples, finite metrics, `passed == true` and
satisfies its own gates: `nrmse < nrmse_limit`, `correlation >
correlation_min`, `slope_min < slope < slope_max`.
"""
struct OmegaFidelityHeader{S<:AbstractFidelityStatus}
    status::S
    reason::Union{Nothing,FidelityReason}
    source::String
    sampling::String
    basis::String
    sample_count::Int
    bias_pa_s::Union{Nothing,Float64}
    rmse_pa_s::Union{Nothing,Float64}
    nrmse::Union{Nothing,Float64}
    max_abs_pa_s::Union{Nothing,Float64}
    slope::Union{Nothing,Float64}
    correlation::Union{Nothing,Float64}
    bottom_residual_pa_s::Union{Nothing,Float64}
    pressure_bands_hpa::Vector{String}
    band_nrmse::Vector{Float64}
    baseline_id::String
    nrmse_limit::Union{Nothing,Float64}
    correlation_min::Union{Nothing,Float64}
    slope_min::Union{Nothing,Float64}
    slope_max::Union{Nothing,Float64}
    passed::Union{Nothing,Bool}
end

"""
Column closure of a cubed-sphere transport binary, decoded from its
`column_closure_*` header keys: `ColumnClosureHeader` when the writer
column-balanced its fluxes and recorded the closure, `NoColumnClosureHeader`
when the keys are absent (no cubed-sphere column balance ran, or the binary
predates the record).
"""
abstract type AbstractColumnClosureHeader end
struct NoColumnClosureHeader <: AbstractColumnClosureHeader end

"""
Recorded column closure (`docs/src/concepts/binary_format.md`): the rule of
the per-cell tolerance `τ_c`, the windows covered, and the maxima over windows
of the projected column residual `|r_c^⊥|` [kg], its ratio to `τ_c`, the
residual folded into `cm` [kg] and its ratio to its tolerance. Both ratios are
at most 1 in a binary that loads.
"""
struct ColumnClosureHeader <: AbstractColumnClosureHeader
    rule::String
    windows::Int
    projected_residual_max_kg::Float64
    residual_to_tolerance_max::Float64
    booked_residual_max_kg::Float64
    cm_residual_to_tolerance_max::Float64
end

"""
Rule of the recorded column closure (header `column_closure_rule`): per-cell
`τ_c = max(COLUMN_CLOSURE_TOL_KG, 8 ε_FT Σ_{f∂c} Σ_k |F_{f,k}|)` over the
cell's incident faces.
"""
const COLUMN_CLOSURE_RULE = "per_cell_8eps_incident_flux_v1"

"""
Version of the flux-certificate key block (header `flux_fidelity_schema`).
Schema 2 carries the flux-weighted `balance_correction_*_relative` metrics and
the two-sided OMEGA slope bound; schema 1 headers came only from unreleased
development builds and are rejected.
"""
const FLUX_FIDELITY_SCHEMA = 2

const _COLUMN_CLOSURE_KEYS = (
    :column_closure_rule, :column_closure_windows,
    :column_closure_projected_residual_max_kg, :column_closure_residual_to_tolerance_max,
    :column_closure_booked_residual_max_kg, :column_closure_cm_residual_to_tolerance_max)

"""
    AbstractFluxConstructionHeader

Flux-construction identity of a transport binary: the method the preprocessor
used to build the horizontal mass fluxes. A version-5 header records it in the
key `flux_construction` (spelled by [`flux_construction_tag`](@ref)) plus the
keys of the construction; a version-4 header records none and parses as
[`NoFluxConstructionHeader`](@ref). Writers take it as the typed
`flux_construction` keyword and write version 5 exactly when it is recorded.
See `docs/src/concepts/binary_format.md`.
"""
abstract type AbstractFluxConstructionHeader end

"""Version-4 header: the flux construction is not recorded."""
struct NoFluxConstructionHeader <: AbstractFluxConstructionHeader end

"""
`flux_construction = "point_sampled_spectral"`: the ERA5 spectral lat-lon and
reduced-Gaussian construction (winds synthesized at the target truncation and
sampled on faces and centres, times the face pressure thickness). It has no
further keys.
"""
struct PointSampledSpectralFluxesHeader <: AbstractFluxConstructionHeader end

"""
    TM5SpectralFluxesHeader(revision, spectral_truncation_used,
                            quadrature_refinement_steps, vod2uv_radius_m)

`flux_construction = "tm5_spectral"`: TM5-style spectral mass fluxes, integrated
over the target cell faces from the retained spectrum
(`docs/src/preprocessing/tm5_spectral_fluxes.md`). Header keys:

- `flux_construction_revision`: revision of the construction (≥ 1);
- `spectral_truncation_used`: triangular truncation T of the spectrum the
  fluxes were integrated from (≥ 1);
- `quadrature_refinement_steps`: `nstep` of TM5's `ShRefinement` rule (≥ 1);
- `vod2uv_radius_m`: sphere radius of the vorticity/divergence-to-wind
  conversion [m] (finite, positive). The radius of the target mesh stays in
  `planet_radius_m`.
"""
struct TM5SpectralFluxesHeader <: AbstractFluxConstructionHeader
    revision                    :: Int
    spectral_truncation_used    :: Int
    quadrature_refinement_steps :: Int
    vod2uv_radius_m             :: Float64

    function TM5SpectralFluxesHeader(revision::Integer, spectral_truncation_used::Integer,
                                     quadrature_refinement_steps::Integer,
                                     vod2uv_radius_m::Real)
        for (key, value) in (("flux_construction_revision", revision),
                             ("spectral_truncation_used", spectral_truncation_used),
                             ("quadrature_refinement_steps", quadrature_refinement_steps))
            !(value isa Bool) && value >= 1 ||
                throw(ArgumentError("tm5_spectral `$key` must be an integer ≥ 1; got $value"))
        end
        radius = Float64(vod2uv_radius_m)
        !(vod2uv_radius_m isa Bool) && isfinite(radius) && radius > 0 || throw(ArgumentError(
            "tm5_spectral `vod2uv_radius_m` must be finite and positive; got $vod2uv_radius_m"))
        return new(Int(revision), Int(spectral_truncation_used),
                   Int(quadrature_refinement_steps), radius)
    end
end

"""On-disk spelling (`flux_construction` value) of a flux-construction identity."""
flux_construction_tag(::Type{PointSampledSpectralFluxesHeader}) = "point_sampled_spectral"
flux_construction_tag(::Type{TM5SpectralFluxesHeader}) = "tm5_spectral"
flux_construction_tag(construction::AbstractFluxConstructionHeader) =
    flux_construction_tag(typeof(construction))

const _FLUX_CONSTRUCTION_KEY = "flux_construction"
const _FLUX_CONSTRUCTION_TYPES = (PointSampledSpectralFluxesHeader, TM5SpectralFluxesHeader)

# The header keys of a construction, one per field and in field order: the one
# list from which writers record, readers decode and validators reject keys.
_flux_construction_keys(::Type{PointSampledSpectralFluxesHeader}) = ()
_flux_construction_keys(::Type{TM5SpectralFluxesHeader}) =
    ("flux_construction_revision", "spectral_truncation_used",
     "quadrature_refinement_steps", "vod2uv_radius_m")

# Every version-5-only header key.
const _FLUX_CONSTRUCTION_HEADER_KEYS = (_FLUX_CONSTRUCTION_KEY,
    Iterators.flatten(map(_flux_construction_keys, _FLUX_CONSTRUCTION_TYPES))...)

# Header entries a writer records for a construction; `format_version` is set
# by `_set_flux_construction!`.
function flux_construction_header_entries(construction::T) where {T <: AbstractFluxConstructionHeader}
    fields = ntuple(i -> _flux_construction_keys(T)[i] => getfield(construction, i),
                    fieldcount(T))
    return (_FLUX_CONSTRUCTION_KEY => flux_construction_tag(T), fields...)
end

# JSON3 reads an integral JSON number (also `639.0`) as an Int.
function _construction_value(header::AbstractDict, key::String, ::Type{Int})
    value = header[key]
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "flux-construction key `$key` must be an integer; got $(repr(value))"))
    return Int(value)
end

function _construction_value(header::AbstractDict, key::String, ::Type{Float64})
    value = header[key]
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "flux-construction key `$key` must be a number; got $(repr(value))"))
    return Float64(value)
end

_flux_construction_from_header(::Type{T}, header) where {T <: AbstractFluxConstructionHeader} =
    T(ntuple(i -> _construction_value(header, _flux_construction_keys(T)[i], fieldtype(T, i)),
             fieldcount(T))...)

"""
    parse_flux_construction_header(header, format_version) -> AbstractFluxConstructionHeader

Decode and check the flux-construction identity of a String-keyed header. A
version-4 header must carry none of the version-5 keys and returns
[`NoFluxConstructionHeader`](@ref). A version-5 header must carry
`flux_construction` with a known value, every key of that construction with a
valid value, and no key of another construction.
"""
function parse_flux_construction_header(header::AbstractDict, format_version::Integer)
    present = [key for key in _FLUX_CONSTRUCTION_HEADER_KEYS if haskey(header, key)]
    if format_version == TRANSPORT_BINARY_FORMAT_VERSION
        isempty(present) || throw(ArgumentError(
            "format_version=$(format_version) header carries the version-5 key(s) " *
            "$(join(("`$k`" for k in present), ", ")); a header that records the flux " *
            "construction must declare format_version=$(FLUX_CONSTRUCTION_FORMAT_VERSION)"))
        return NoFluxConstructionHeader()
    end
    format_version == FLUX_CONSTRUCTION_FORMAT_VERSION || throw(ArgumentError(
        "no flux-construction identity is defined for format_version=$(format_version)"))
    haskey(header, _FLUX_CONSTRUCTION_KEY) || throw(ArgumentError(
        "format_version=$(format_version) requires `$(_FLUX_CONSTRUCTION_KEY)`"))
    raw_tag = header[_FLUX_CONSTRUCTION_KEY]
    raw_tag isa AbstractString || throw(ArgumentError(
        "`$(_FLUX_CONSTRUCTION_KEY)` must be a string; got $(repr(raw_tag))"))
    index = findfirst(T -> flux_construction_tag(T) == raw_tag, _FLUX_CONSTRUCTION_TYPES)
    index === nothing && throw(ArgumentError(
        "unsupported $(_FLUX_CONSTRUCTION_KEY)=$(repr(raw_tag)); expected one of " *
        join((repr(flux_construction_tag(T)) for T in _FLUX_CONSTRUCTION_TYPES), ", ")))
    construction_type = _FLUX_CONSTRUCTION_TYPES[index]
    own_keys = _flux_construction_keys(construction_type)
    missing_keys = [key for key in own_keys if !haskey(header, key)]
    isempty(missing_keys) || throw(ArgumentError(
        "$(_FLUX_CONSTRUCTION_KEY)=$(repr(raw_tag)) requires " *
        join(("`$k`" for k in missing_keys), ", ")))
    foreign = [key for key in present if key != _FLUX_CONSTRUCTION_KEY && !(key in own_keys)]
    isempty(foreign) || throw(ArgumentError(
        "$(_FLUX_CONSTRUCTION_KEY)=$(repr(raw_tag)) does not use " *
        join(("`$k`" for k in foreign), ", ")))
    return _flux_construction_from_header(construction_type, header)
end

# Writers: record a construction (version 5) or leave the header untouched
# (version 4, byte-identical to a writer without the keyword).
_set_flux_construction!(header::Dict{String, Any}, ::NoFluxConstructionHeader) = header
function _set_flux_construction!(header::Dict{String, Any},
                                 construction::AbstractFluxConstructionHeader)
    header["format_version"] = FLUX_CONSTRUCTION_FORMAT_VERSION
    for (key, value) in flux_construction_header_entries(construction)
        header[key] = value
    end
    return header
end

"""One-line description of a flux-construction identity for reports."""
flux_construction_summary(::NoFluxConstructionHeader) = "not recorded (format v4)"
flux_construction_summary(::PointSampledSpectralFluxesHeader) =
    flux_construction_tag(PointSampledSpectralFluxesHeader)
flux_construction_summary(c::TM5SpectralFluxesHeader) = string(
    flux_construction_tag(c), " (revision ", c.revision,
    ", spectral_truncation_used=", c.spectral_truncation_used,
    ", quadrature_refinement_steps=", c.quadrature_refinement_steps,
    ", vod2uv_radius_m=", c.vod2uv_radius_m, ")")

"""Geometry metadata for a regular longitude-latitude binary."""
struct LatLonBinaryGeometry <: AbstractTransportBinaryGeometry
    Nx                 :: Int
    Ny                 :: Int
    longitudes         :: Vector{Float64}
    latitudes          :: Vector{Float64}
    longitude_interval :: Vector{Float64}
    latitude_interval  :: Vector{Float64}
end

"""Geometry metadata for a face-indexed reduced-Gaussian binary."""
struct ReducedGaussianBinaryGeometry <: AbstractTransportBinaryGeometry
    latitudes     :: Vector{Float64}
    nlon_per_ring :: Vector{Int}
end

"""Geometry metadata for a six-panel cubed-sphere binary."""
struct CubedSphereBinaryGeometry <: AbstractTransportBinaryGeometry
    Nc                   :: Int
    npanel               :: Int
    panel_convention     :: Symbol
    definition           :: Symbol
    coordinate_law       :: Symbol
    center_law           :: Symbol
    longitude_offset_deg :: Float64
end

grid_type(::LatLonBinaryGeometry) = :latlon
grid_type(::ReducedGaussianBinaryGeometry) = :reduced_gaussian
grid_type(::CubedSphereBinaryGeometry) = :cubed_sphere

horizontal_topology(::LatLonBinaryGeometry) = :structureddirectional
horizontal_topology(::ReducedGaussianBinaryGeometry) = :faceindexed
horizontal_topology(::CubedSphereBinaryGeometry) = :structureddirectional

"""
    TransportBinaryHeader{G<:AbstractTransportBinaryGeometry}

Validated metadata for the current transport-binary format. `geometry` selects
the horizontal topology through dispatch, while the remaining fields describe
the common timing, vertical coordinate, mass basis, and payload contract.

`planet_radius_m` is the radius of the sphere on which the preprocessor computed
cell areas, and so the air masses from pressure thickness (`m = Δp A / g`); the
runtime builds its mesh with it. Binaries written before the key existed read as
`EARTH_RADIUS`, the radius the runtime used for them.

`A_ifc` and `B_ifc` define interface pressure as
`p_half[k] = A_ifc[k] + B_ifc[k] * surface_pressure`, with `k = 1` at the
top of atmosphere. `flux_kind` distinguishes mass per transport substep from a
full-window mass amount.

`flux_fidelity`, `omega_fidelity` and `column_closure` are the write-time
certificate and column-closure record, checked for consistency when the header
is parsed. `flux_construction` is the flux-construction identity of a version-5
header and `NoFluxConstructionHeader()` for version 4.
"""
struct TransportBinaryHeader{G <: AbstractTransportBinaryGeometry}
    format_version       :: Int
    header_bytes         :: Int
    on_disk_float_type   :: Symbol      # :Float32 or :Float64
    float_bytes          :: Int         # 4 or 8
    geometry             :: G
    planet_radius_m      :: Float64     # radius of the preprocessing mesh [m]
    ncell                :: Int         # total horizontal cells
    nface_h              :: Int         # total horizontal faces
    nlevel               :: Int         # vertical levels (k=1 TOA, k=nlevel surface)
    nwindow              :: Int         # windows per day (typically 24)
    dt_met_seconds       :: Float64     # met-data window interval [s] (3600 for hourly)
    half_dt_seconds      :: Float64     # half-step time [s] for flux scaling
    steps_per_window     :: Int         # scalar compatibility summary (constant value or max schedule)
    steps_per_window_by_window :: Vector{Int}
    source_flux_sampling :: Symbol
    air_mass_sampling    :: Symbol
    flux_sampling        :: Symbol      # :window_constant
    flux_kind            :: Symbol      # :substep_mass_amount or :full_window_mass_amount
    humidity_sampling    :: Symbol
    delta_semantics      :: Symbol
    poisson_balance_target_scale :: Float64
    poisson_balance_target_semantics :: String
    poisson_balance_target_scale_by_window :: Vector{Float64}
    A_ifc                :: Vector{Float64}  # hybrid A [Pa], length nlevel+1
    B_ifc                :: Vector{Float64}  # hybrid B [1],  length nlevel+1
    mass_basis           :: Symbol           # :moist or :dry
    surface_pressure_basis :: Union{TotalSurfacePressure, DrySurfacePressure}
    tm5_convection_mass_basis :: Union{Nothing, DryBasis, MoistBasis}
    payload_sections     :: Vector{Symbol}
    n_geometry_elems     :: Int
    elems_per_window     :: Int
    flux_fidelity        :: FluxFidelityHeader
    omega_fidelity       :: OmegaFidelityHeader
    column_closure       :: AbstractColumnClosureHeader
    flux_construction    :: AbstractFluxConstructionHeader
    raw_header           :: Dict{String, Any}
end

binary_geometry(h::TransportBinaryHeader) = h.geometry
surface_pressure_basis(h::TransportBinaryHeader) = h.surface_pressure_basis
grid_type(h::TransportBinaryHeader) = grid_type(h.geometry)
horizontal_topology(h::TransportBinaryHeader) = horizontal_topology(h.geometry)

function parse_fidelity_status(value)
    text = lowercase(String(value))
    text == fidelity_tag(FidelityPassed()) && return FidelityPassed()
    text == fidelity_tag(FidelityFailed()) && return FidelityFailed()
    text == fidelity_tag(FidelityNotApplicable()) && return FidelityNotApplicable()
    text == fidelity_tag(FidelityUnavailable()) && return FidelityUnavailable()
    throw(ArgumentError("unsupported fidelity status $(repr(text)); expected passed, failed, not_applicable, or unavailable"))
end

function parse_flux_fidelity_kind(value)
    text = lowercase(String(value))
    text == fidelity_tag(WindRederivedFidelity()) && return WindRederivedFidelity()
    text == fidelity_tag(NativePassthroughFidelity()) && return NativePassthroughFidelity()
    text == fidelity_tag(NoFluxFidelityKind()) && return NoFluxFidelityKind()
    throw(ArgumentError("unsupported flux_fidelity_kind $(repr(text)); expected wind_rederived, native_passthrough, or none"))
end

@inline _optional_float(hdr, key) = haskey(hdr, key) ? Float64(getproperty(hdr, key)) : nothing
@inline optional_header_int(hdr, key) = haskey(hdr, key) ? Int(getproperty(hdr, key)) : nothing

# Certificate keys that only a wind-derived (passed) certificate carries.
const _FLUX_FIDELITY_METRIC_KEYS = (
    :flux_fidelity_max_abs_kg, :flux_fidelity_rms_abs_kg, :flux_fidelity_max_rel,
    :flux_fidelity_rms_rel, :flux_fidelity_max_ulp, :flux_fidelity_abs_tol_kg,
    :flux_fidelity_rel_tol, :flux_fidelity_scale_floor_kg, :flux_fidelity_worst_window,
    :flux_fidelity_worst_face, :flux_fidelity_worst_level)
const _BALANCE_CORRECTION_KEYS = (
    :balance_correction_max_abs_kg, :balance_correction_rms_relative,
    :balance_correction_max_level_rms_relative, :balance_correction_max_level)
const _OMEGA_FIDELITY_METRIC_KEYS = (
    :omega_fidelity_bias_pa_s, :omega_fidelity_rmse_pa_s, :omega_fidelity_nrmse,
    :omega_fidelity_max_abs_pa_s, :omega_fidelity_slope, :omega_fidelity_correlation,
    :omega_fidelity_bottom_residual_pa_s, :omega_fidelity_nrmse_limit,
    :omega_fidelity_correlation_min, :omega_fidelity_slope_min, :omega_fidelity_slope_max)

function _require_finite_keys(hdr, keys, what::AbstractString)
    for key in keys
        haskey(hdr, key) || throw(ArgumentError("$what certificate lacks `$key`"))
        value = Float64(getproperty(hdr, key))
        isfinite(value) || throw(ArgumentError("$what certificate has non-finite `$key` = $value"))
    end
    return nothing
end

function _forbid_keys(hdr, keys, what::AbstractString)
    for key in keys
        haskey(hdr, key) && throw(ArgumentError("$what certificate must not carry `$key`"))
    end
    return nothing
end

# The level of the largest per-level balance correction is one of the binary's.
function _check_balance_correction_level(hdr, nlevel::Integer)
    level = Int(hdr.balance_correction_max_level)
    1 <= level <= nlevel || throw(ArgumentError(
        "balance_correction_max_level = $level is not a level of this binary (1..$nlevel)"))
    return nothing
end

# Sampled certificates (wind-derived or native) cover every window, sampled
# face and level of the binary.
function _check_flux_fidelity_coverage(hdr, nwindow::Integer, nlevel::Integer)
    windows = Int(get(hdr, :flux_fidelity_windows, 0))
    per_window = Int(get(hdr, :flux_fidelity_samples_per_window, 0))
    count = Int(get(hdr, :flux_fidelity_sample_count, 0))
    windows == nwindow || throw(ArgumentError(
        "flux-fidelity certificate covers $windows windows; the binary has nwindow=$nwindow"))
    per_window >= 1 || throw(ArgumentError(
        "flux-fidelity certificate has $per_window samples per window"))
    count == windows * per_window * nlevel || throw(ArgumentError(
        "flux-fidelity sample count $count != windows ($windows) × samples per window " *
        "($per_window) × nlevel ($nlevel)"))
    isempty(String(get(hdr, :flux_fidelity_method, ""))) && throw(ArgumentError(
        "flux-fidelity certificate lacks `flux_fidelity_method`"))
    _require_finite_keys(hdr, (:flux_fidelity_amount_seconds_min,
                               :flux_fidelity_amount_seconds_max), "flux-fidelity")
    return nothing
end

function _parse_time_sampling(hdr, windows::Integer)
    sampling = String(get(hdr, :flux_fidelity_time_sampling, ""))
    by_window = haskey(hdr, :flux_fidelity_time_sampling_by_window) ?
        String.(collect(hdr.flux_fidelity_time_sampling_by_window)) : String[]
    if sampling == "mixed"
        length(by_window) == windows || throw(ArgumentError(
            "mixed flux_fidelity_time_sampling needs one entry per window; got " *
            "$(length(by_window)) for $windows windows"))
    else
        isempty(by_window) || throw(ArgumentError(
            "flux_fidelity_time_sampling_by_window is only valid with \"mixed\" sampling"))
    end
    return sampling, by_window
end

_check_flux_kind(::FidelityPassed, ::WindRederivedFidelity, hdr, nwindow, nlevel) = begin
    _check_flux_fidelity_coverage(hdr, nwindow, nlevel)
    _require_finite_keys(hdr, (_FLUX_FIDELITY_METRIC_KEYS..., _BALANCE_CORRECTION_KEYS...),
                         "passed flux-fidelity")
    _check_balance_correction_level(hdr, nlevel)
    isempty(String(get(hdr, :flux_fidelity_time_sampling, ""))) && throw(ArgumentError(
        "passed flux-fidelity certificate lacks `flux_fidelity_time_sampling`"))
    nothing
end
_check_flux_kind(::FidelityNotApplicable, ::NativePassthroughFidelity, hdr, nwindow, nlevel) = begin
    _check_flux_fidelity_coverage(hdr, nwindow, nlevel)
    _require_finite_keys(hdr, _BALANCE_CORRECTION_KEYS, "native flux-fidelity")
    _check_balance_correction_level(hdr, nlevel)
    _forbid_keys(hdr, _FLUX_FIDELITY_METRIC_KEYS, "native flux-fidelity")
    nothing
end
function _check_flux_kind(::Union{FidelityNotApplicable, FidelityUnavailable, FidelityFailed},
                          ::NoFluxFidelityKind, hdr, _nwindow, _nlevel)
    for key in (:flux_fidelity_windows, :flux_fidelity_samples_per_window,
                :flux_fidelity_sample_count)
        Int(get(hdr, key, 0)) == 0 || throw(ArgumentError(
            "flux-fidelity certificate without samples has `$key` = $(getproperty(hdr, key))"))
    end
    _forbid_keys(hdr, (_FLUX_FIDELITY_METRIC_KEYS..., _BALANCE_CORRECTION_KEYS...),
                 "flux-fidelity certificate without samples")
    return nothing
end
_check_flux_kind(status, kind, _hdr, _nwindow, _nlevel) = throw(ArgumentError(
    "flux-fidelity status $(nameof(typeof(status))) is inconsistent with kind " *
    "$(nameof(typeof(kind)))"))

"""
    parse_flux_fidelity_header(hdr, nwindow, nlevel) -> FluxFidelityHeader

Decode and check the horizontal-flux certificate: the status/kind pair must be
one a writer emits, a sampled certificate must cover `nwindow` windows and
`nlevel` levels (`sample_count == windows × samples_per_window × nlevel`) with
finite metrics, and a non-passed certificate needs a reason. Binaries written
before the certificate existed read as `FidelityUnavailable`
(`legacy_header_missing`).
"""
function parse_flux_fidelity_header(hdr, nwindow::Integer, nlevel::Integer)
    # Pre-C6 v4 files stay readable, but their lack of certification is made
    # explicit in the typed result. Every current writer emits FLUX_FIDELITY_SCHEMA.
    if !haskey(hdr, :flux_fidelity_schema)
        return FluxFidelityHeader(0, FidelityUnavailable(), NoFluxFidelityKind(),
            FidelityReason("legacy_header_missing"), "", "", 0, 0, 0, "", "", String[],
            nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing)
    end
    schema = Int(hdr.flux_fidelity_schema)
    schema == FLUX_FIDELITY_SCHEMA || throw(ArgumentError(
        "unsupported flux_fidelity_schema=$(schema); expected $(FLUX_FIDELITY_SCHEMA) — " *
        "regenerate this binary with the current preprocessor"))
    status = parse_fidelity_status(hdr.flux_fidelity_status)
    kind = parse_flux_fidelity_kind(hdr.flux_fidelity_kind)
    reason = haskey(hdr, :flux_fidelity_reason) ? FidelityReason(String(hdr.flux_fidelity_reason)) : nothing
    status isa FidelityPassed && reason !== nothing && throw(ArgumentError(
        "passed flux-fidelity certificate must not carry flux_fidelity_reason"))
    !(status isa FidelityPassed) && reason === nothing && throw(ArgumentError(
        "non-passed flux-fidelity certificate requires flux_fidelity_reason"))
    _check_flux_kind(status, kind, hdr, nwindow, nlevel)
    windows = Int(get(hdr, :flux_fidelity_windows, 0))
    time_sampling, by_window = _parse_time_sampling(hdr, windows)
    max_ulp = haskey(hdr, :flux_fidelity_max_ulp) ? UInt64(hdr.flux_fidelity_max_ulp) : nothing
    return FluxFidelityHeader(
        schema, status, kind, reason,
        String(get(hdr, :flux_fidelity_sample_rule, "")),
        String(get(hdr, :flux_fidelity_sample_seed, "")),
        Int(get(hdr, :flux_fidelity_samples_per_window, 0)),
        Int(get(hdr, :flux_fidelity_sample_count, 0)),
        windows,
        String(get(hdr, :flux_fidelity_method, "")),
        time_sampling, by_window,
        _optional_float(hdr, :flux_fidelity_amount_seconds_min),
        _optional_float(hdr, :flux_fidelity_amount_seconds_max),
        _optional_float(hdr, :flux_fidelity_max_abs_kg),
        _optional_float(hdr, :flux_fidelity_rms_abs_kg),
        _optional_float(hdr, :flux_fidelity_max_rel),
        _optional_float(hdr, :flux_fidelity_rms_rel), max_ulp,
        _optional_float(hdr, :flux_fidelity_abs_tol_kg),
        _optional_float(hdr, :flux_fidelity_rel_tol),
        _optional_float(hdr, :flux_fidelity_scale_floor_kg),
        optional_header_int(hdr, :flux_fidelity_worst_window),
        optional_header_int(hdr, :flux_fidelity_worst_face),
        optional_header_int(hdr, :flux_fidelity_worst_level),
        _optional_float(hdr, :balance_correction_max_abs_kg),
        _optional_float(hdr, :balance_correction_rms_relative),
        _optional_float(hdr, :balance_correction_max_level_rms_relative),
        optional_header_int(hdr, :balance_correction_max_level))
end

# OMEGA certificate checks by status.
function _check_omega_status(::FidelityPassed, hdr)
    Int(get(hdr, :omega_fidelity_sample_count, 0)) > 0 || throw(ArgumentError(
        "passed OMEGA-fidelity certificate has no samples"))
    _require_finite_keys(hdr, _OMEGA_FIDELITY_METRIC_KEYS, "passed OMEGA-fidelity")
    get(hdr, :omega_fidelity_passed, false) === true || throw(ArgumentError(
        "passed OMEGA-fidelity certificate needs omega_fidelity_passed = true"))
    haskey(hdr, :omega_fidelity_pressure_bands_hpa) && haskey(hdr, :omega_fidelity_band_nrmse) &&
        !isempty(hdr.omega_fidelity_band_nrmse) &&
        all(x -> x isa Real && isfinite(Float64(x)), hdr.omega_fidelity_band_nrmse) ||
        throw(ArgumentError(
        "passed OMEGA-fidelity certificate needs finite pressure-band NRMSE values"))
    nrmse = Float64(hdr.omega_fidelity_nrmse)
    correlation = Float64(hdr.omega_fidelity_correlation)
    slope = Float64(hdr.omega_fidelity_slope)
    nrmse < Float64(hdr.omega_fidelity_nrmse_limit) &&
        correlation > Float64(hdr.omega_fidelity_correlation_min) &&
        Float64(hdr.omega_fidelity_slope_min) < slope < Float64(hdr.omega_fidelity_slope_max) ||
        throw(ArgumentError(
        "passed OMEGA-fidelity certificate violates its own gates: nrmse=$nrmse, " *
        "correlation=$correlation, slope=$slope"))
    return nothing
end
function _check_omega_status(::FidelityFailed, hdr)
    get(hdr, :omega_fidelity_passed, true) === false || throw(ArgumentError(
        "failed OMEGA-fidelity certificate needs omega_fidelity_passed = false"))
    return nothing
end
function _check_omega_status(::Union{FidelityNotApplicable, FidelityUnavailable}, hdr)
    Int(get(hdr, :omega_fidelity_sample_count, 0)) == 0 || throw(ArgumentError(
        "OMEGA-fidelity certificate without a comparison has samples"))
    _forbid_keys(hdr, (_OMEGA_FIDELITY_METRIC_KEYS..., :omega_fidelity_passed),
                 "OMEGA-fidelity certificate without a comparison")
    return nothing
end

"""
    parse_omega_fidelity_header(hdr) -> OmegaFidelityHeader

Decode and check the source-OMEGA certificate: a passed certificate has
samples, finite metrics, `omega_fidelity_passed = true` and satisfies its
recorded gates; a failed one has `omega_fidelity_passed = false`; a
not-applicable or unavailable one has no samples or metrics.
"""
function parse_omega_fidelity_header(hdr)
    if !haskey(hdr, :omega_fidelity_status)
        return OmegaFidelityHeader(FidelityUnavailable(), FidelityReason("legacy_header_missing"),
            "", "", "", 0, nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, String[], Float64[], "", nothing, nothing, nothing, nothing, nothing)
    end
    status = parse_fidelity_status(hdr.omega_fidelity_status)
    reason = haskey(hdr, :omega_fidelity_reason) ? FidelityReason(String(hdr.omega_fidelity_reason)) : nothing
    status isa FidelityPassed && reason !== nothing && throw(ArgumentError(
        "passed OMEGA-fidelity certificate must not carry omega_fidelity_reason"))
    !(status isa FidelityPassed) && reason === nothing && throw(ArgumentError(
        "non-passed OMEGA-fidelity certificate requires omega_fidelity_reason"))
    _check_omega_status(status, hdr)
    bands = haskey(hdr, :omega_fidelity_pressure_bands_hpa) ?
        String.(collect(hdr.omega_fidelity_pressure_bands_hpa)) : String[]
    band_nrmse = haskey(hdr, :omega_fidelity_band_nrmse) ?
        Float64.(collect(hdr.omega_fidelity_band_nrmse)) : Float64[]
    length(bands) == length(band_nrmse) || throw(ArgumentError(
        "omega_fidelity_pressure_bands_hpa and omega_fidelity_band_nrmse lengths differ"))
    return OmegaFidelityHeader(
        status, reason, String(get(hdr, :omega_fidelity_source, "")),
        String(get(hdr, :omega_fidelity_sampling, "")),
        String(get(hdr, :omega_fidelity_basis, "")),
        Int(get(hdr, :omega_fidelity_sample_count, 0)),
        _optional_float(hdr, :omega_fidelity_bias_pa_s),
        _optional_float(hdr, :omega_fidelity_rmse_pa_s),
        _optional_float(hdr, :omega_fidelity_nrmse),
        _optional_float(hdr, :omega_fidelity_max_abs_pa_s),
        _optional_float(hdr, :omega_fidelity_slope),
        _optional_float(hdr, :omega_fidelity_correlation),
        _optional_float(hdr, :omega_fidelity_bottom_residual_pa_s),
        bands, band_nrmse,
        String(get(hdr, :omega_fidelity_baseline_id, "")),
        _optional_float(hdr, :omega_fidelity_nrmse_limit),
        _optional_float(hdr, :omega_fidelity_correlation_min),
        _optional_float(hdr, :omega_fidelity_slope_min),
        _optional_float(hdr, :omega_fidelity_slope_max),
        haskey(hdr, :omega_fidelity_passed) ? Bool(hdr.omega_fidelity_passed) : nothing)
end

"""
    parse_column_closure_header(hdr, nwindow) -> AbstractColumnClosureHeader

Decode and check the column-closure record. Absent keys mean the closure was
not recorded (`NoColumnClosureHeader`). A record must be complete, name the
known rule, cover `nwindow` windows and hold finite, non-negative values with
both residual/tolerance ratios at most 1; otherwise the stored fluxes cannot be
trusted to close the column mass budget and the binary must be regenerated.
"""
function parse_column_closure_header(hdr, nwindow::Integer)
    present = [key for key in _COLUMN_CLOSURE_KEYS if haskey(hdr, key)]
    isempty(present) && return NoColumnClosureHeader()
    missing_keys = setdiff(_COLUMN_CLOSURE_KEYS, present)
    isempty(missing_keys) || throw(ArgumentError(
        "column-closure record lacks $(join(("`$k`" for k in missing_keys), ", ")); " *
        "regenerate this binary"))
    rule = String(hdr.column_closure_rule)
    rule == COLUMN_CLOSURE_RULE || throw(ArgumentError(
        "unsupported column_closure_rule $(repr(rule)); expected $(repr(COLUMN_CLOSURE_RULE))"))
    windows = Int(hdr.column_closure_windows)
    windows == nwindow || throw(ArgumentError(
        "column-closure record covers $windows windows; the binary has nwindow=$nwindow; " *
        "regenerate this binary"))
    number(key) = (value = getproperty(hdr, key);
                   value isa Real ? Float64(value) : throw(ArgumentError(
                       "column-closure record has non-numeric `$key` = $(repr(value)); " *
                       "regenerate this binary")))
    projected = number(:column_closure_projected_residual_max_kg)
    ratio = number(:column_closure_residual_to_tolerance_max)
    booked = number(:column_closure_booked_residual_max_kg)
    cm_ratio = number(:column_closure_cm_residual_to_tolerance_max)
    values = (projected, ratio, booked, cm_ratio)
    all(isfinite, values) && all(>=(0), values) || throw(ArgumentError(
        "column-closure record is not finite and non-negative: projected residual " *
        "$projected kg, ratio $ratio, cm residual $booked kg, cm ratio $cm_ratio; " *
        "regenerate this binary"))
    ratio <= 1 && cm_ratio <= 1 || throw(ArgumentError(
        "column-closure record exceeds its per-cell tolerance (residual/tolerance " *
        "ratio $ratio, cm residual ratio $cm_ratio; both must be at most 1): the stored " *
        "fluxes do not close the column mass budget; regenerate this binary with the " *
        "current preprocessor"))
    return ColumnClosureHeader(rule, windows, projected, ratio, booked, cm_ratio)
end

function _parse_steps_per_window_schedule(hdr, nwindow::Integer,
                                          steps_per_window::Integer)
    haskey(hdr, :steps_per_window_by_window) ||
        throw(ArgumentError("transport-binary format_version $(join(TRANSPORT_BINARY_FORMAT_VERSIONS, " and ")) " *
                            "requires `steps_per_window_by_window`; regenerate this binary"))
    schedule = Int.(collect(hdr.steps_per_window_by_window))
    length(schedule) == Int(nwindow) ||
        throw(ArgumentError("steps_per_window_by_window length $(length(schedule)) " *
                            "does not match nwindow=$(nwindow)"))
    all(>=(1), schedule) ||
        throw(ArgumentError("steps_per_window_by_window must contain only positive integers; got $(schedule)"))
    Int(steps_per_window) == maximum(schedule) ||
        throw(ArgumentError("steps_per_window must be maximum(steps_per_window_by_window); " *
                            "got steps_per_window=$(steps_per_window), schedule=$(schedule)"))
    return schedule
end

function _parse_poisson_scale_schedule(hdr, nwindow::Integer,
                                       scalar_scale::Real)
    haskey(hdr, :poisson_balance_target_scale_by_window) ||
        throw(ArgumentError("transport-binary format_version $(join(TRANSPORT_BINARY_FORMAT_VERSIONS, " and ")) " *
                            "requires `poisson_balance_target_scale_by_window`; regenerate this binary"))
    scales = Float64.(collect(hdr.poisson_balance_target_scale_by_window))
    length(scales) == Int(nwindow) ||
        throw(ArgumentError("poisson_balance_target_scale_by_window length $(length(scales)) " *
                            "does not match nwindow=$(nwindow)"))
    all(s -> isfinite(s) && s > 0, scales) ||
        throw(ArgumentError("poisson_balance_target_scale_by_window must contain only finite positive values"))
    isfinite(Float64(scalar_scale)) && Float64(scalar_scale) > 0 ||
        throw(ArgumentError("poisson_balance_target_scale must be finite and positive"))
    return scales
end

@inline has_variable_step_schedule(schedule::AbstractVector{<:Integer}) =
    !isempty(schedule) && any(!=(first(schedule)), schedule)

function steps_per_window_summary(steps::Integer,
                                   schedule::AbstractVector{<:Integer})
    if has_variable_step_schedule(schedule)
        return string(Int(steps), " max (per-window ",
                      minimum(schedule), ":", maximum(schedule), ")")
    end
    return string(Int(steps))
end

@inline transport_geometry_summary(g::LatLonBinaryGeometry, h::TransportBinaryHeader) =
    string(g.Nx, "×", g.Ny, " structured cells, ", h.nlevel, " levels")
@inline transport_geometry_summary(::ReducedGaussianBinaryGeometry, h::TransportBinaryHeader) =
    string(h.ncell, " cells, ", h.nface_h, " faces, ", h.nlevel, " levels")
@inline transport_geometry_summary(g::CubedSphereBinaryGeometry, h::TransportBinaryHeader) =
    string("C", g.Nc, ", panels=", g.npanel, ", ", h.nlevel, " levels")
@inline transport_geometry_summary(h::TransportBinaryHeader) =
    string(transport_geometry_summary(h.geometry, h), ", radius ", h.planet_radius_m, " m")

@inline function transport_qv_summary(h::TransportBinaryHeader)
    return (:qv_start in h.payload_sections && :qv_end in h.payload_sections) ?
           "qv_start/qv_end" : "none"
end

@inline function transport_semantics_summary(h::TransportBinaryHeader)
    return string(
        "air_mass=", h.air_mass_sampling,
        ", flux=", h.flux_sampling, "/", h.flux_kind,
        ", humidity=", h.humidity_sampling,
        ", delta=", h.delta_semantics,
        h.source_flux_sampling === :unknown ? "" : string(", source_flux=", h.source_flux_sampling)
    )
end

function Base.summary(h::TransportBinaryHeader)
    return string(
        "TransportBinaryHeader(v", h.format_version, ", ",
        grid_type(h), "/", horizontal_topology(h), ", ",
        h.nwindow, " windows)"
    )
end

function Base.show(io::IO, h::TransportBinaryHeader)
    print(io, summary(h), "\n",
          "├── geometry:      ", transport_geometry_summary(h), "\n",
          "├── storage:       ", h.on_disk_float_type, " on disk, basis=", h.mass_basis, "\n",
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=",
              steps_per_window_summary(h.steps_per_window, h.steps_per_window_by_window), "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", transport_qv_summary(h), "\n",
          "├── semantics:     ", transport_semantics_summary(h), "\n",
          "├── poisson:       ", string("scale=", h.poisson_balance_target_scale, ", ", h.poisson_balance_target_semantics), "\n",
          "├── construction:  ", flux_construction_summary(h.flux_construction), "\n",
          "└── header bytes:  ", h.header_bytes)
end

transport_is_structured(h::TransportBinaryHeader) = h.geometry isa LatLonBinaryGeometry
transport_is_faceindexed(h::TransportBinaryHeader) = h.geometry isa ReducedGaussianBinaryGeometry

function transport_disk_float_type(sym::Symbol)
    sym === :Float64 ? Float64 : Float32
end

function _transport_parse_on_disk_float_type(hdr)
    haskey(hdr, :float_type) ||
        throw(ArgumentError("transport binary header is missing float_type"))
    ft_str = string(hdr.float_type)
    if ft_str == "Float64"
        return :Float64, 8
    elseif ft_str == "Float32"
        return :Float32, 4
    end
    throw(ArgumentError("unsupported transport binary float_type=$(repr(ft_str)); expected Float32 or Float64"))
end

function _transport_parse_mass_basis(hdr)
    haskey(hdr, :mass_basis) ||
        throw(ArgumentError("transport binary header is missing mass_basis; regenerate the binary"))
    return basis_symbol(hdr.mass_basis)
end

@inline transport_surface_pressure_basis_label(::TotalSurfacePressure) = "total"
@inline transport_surface_pressure_basis_label(::DrySurfacePressure) = "dry"

function _transport_parse_surface_pressure_basis(value)
    basis = lowercase(String(value))
    basis == "total" && return TotalSurfacePressure()
    basis == "dry" && return DrySurfacePressure()
    throw(ArgumentError(
        "unsupported transport binary surface_pressure_basis=$(repr(basis)); " *
        "expected total or dry"))
end


function _transport_parse_tm5_convection_mass_basis(hdr, payload_sections)
    all(section in payload_sections for section in TM5_CONVECTION_PAYLOAD_SECTIONS) ||
        return nothing
    key = Symbol(TM5_CONVECTION_MASS_BASIS_KEY)
    haskey(hdr, key) || return nothing
    return mass_basis(getproperty(hdr, key))
end

@inline function _transport_normalize_symbol(value)
    return Symbol(replace(lowercase(String(value)), '-' => '_', ' ' => '_'))
end

const _TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS = (
    :window_start_endpoint,
    :window_end_endpoint,
    :window_mean,
    :interval_integrated,
)

@inline function _transport_validate_source_flux_sampling(value)
    sym = _transport_normalize_symbol(value)
    sym in _TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS || throw(ArgumentError(
        "unsupported source_flux_sampling=$(value); supported values are " *
        join(string.(Tuple(_TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS)), ", ")
    ))
    return sym
end

@inline function _transport_parse_symbol_key(hdr, key::Symbol)
    return _transport_normalize_symbol(getproperty(hdr, key))
end

@inline function _transport_default_humidity_sampling(payload_sections::AbstractVector{Symbol})
    if (:qv_start in payload_sections) || (:qv_end in payload_sections)
        return :window_endpoints
    else
        return :none
    end
end

@inline function _transport_default_delta_semantics(payload_sections::AbstractVector{Symbol})
    return any(section in FLUX_DELTA_PAYLOAD_SECTIONS for section in payload_sections) ?
           :forward_window_endpoint_difference : :none
end

# Accepted timing and sampling semantics. Writers declare these explicitly;
# readers reject missing or unsupported values.

const _TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS = (:window_start_endpoint,)
const _TRANSPORT_ALLOWED_FLUX_SAMPLINGS     = (:window_start_endpoint, :window_constant, :window_mean)
const _TRANSPORT_ALLOWED_FLUX_KINDS         = (:substep_mass_amount, :full_window_mass_amount)
const _TRANSPORT_ALLOWED_DELTA_SEMANTICS    = (:forward_window_endpoint_difference, :none)
const _TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS = (:window_endpoints, :none)

_transport_parse_grid_type(hdr) = Symbol(lowercase(string(hdr.grid_type)))
_transport_parse_topology(hdr) = Symbol(lowercase(string(hdr.horizontal_topology)))

function _transport_parse_sections(hdr)
    sections = collect(hdr.payload_sections)
    return Symbol.(lowercase.(String.(sections)))
end

function _transport_header_axis(hdr, n::Int, key::Symbol)
    haskey(hdr, key) && return Float64.(collect(getproperty(hdr, key)))
    n == 0 && return Float64[]
    error("Transport binary header missing $(key)")
end

function _transport_header_interval(hdr, key::Symbol)
    return haskey(hdr, key) ? Float64.(collect(getproperty(hdr, key))) : Float64[]
end

function _parse_transport_geometry(hdr, grid::Symbol, topology::Symbol)
    if grid === :latlon && topology === :structureddirectional
        Nx, Ny = Int(hdr.Nx), Int(hdr.Ny)
        return LatLonBinaryGeometry(
            Nx,
            Ny,
            _transport_header_axis(hdr, Nx, :lons),
            _transport_header_axis(hdr, Ny, :lats),
            _transport_header_interval(hdr, :longitude_interval),
            _transport_header_interval(hdr, :latitude_interval),
        )
    elseif grid === :reduced_gaussian && topology === :faceindexed
        return ReducedGaussianBinaryGeometry(
            Float64.(collect(hdr.latitudes)),
            Int.(collect(hdr.nlon_per_ring)),
        )
    elseif grid === :cubed_sphere && topology === :structureddirectional
        return CubedSphereBinaryGeometry(
            Int(hdr.Nc),
            Int(hdr.npanel),
            Symbol(lowercase(String(hdr.panel_convention))),
            Symbol(lowercase(String(hdr.cs_definition))),
            Symbol(lowercase(String(hdr.cs_coordinate_law))),
            Symbol(lowercase(String(hdr.cs_center_law))),
            Float64(hdr.longitude_offset_deg),
        )
    end
    throw(ArgumentError("unsupported transport-binary geometry $(grid)/$(topology)"))
end

function parse_transport_header(raw_bytes::Vector{UInt8};
                                 legacy_surface_pressure_basis::Union{TotalSurfacePressure,
                                                                      DrySurfacePressure} =
                                     TotalSurfacePressure())
    json_end = something(findfirst(==(0x00), raw_bytes), length(raw_bytes) + 1) - 1
    hdr = JSON3.read(String(raw_bytes[1:json_end]))

    haskey(hdr, :format_version) ||
        error("TransportBinaryReader requires the topology-generic binary family header (`format_version` missing)")

    # No silent defaults for missing contract fields.
    # `validate_transport_contract!` (called by `TransportBinaryReader`
    # before we get here) has already verified the 8 fields are present —
    # before parsing the typed header.
    format_version = _check_transport_format_version(Int(hdr.format_version))
    header_bytes = Int(hdr.header_bytes)
    disk_ft, float_bytes = _transport_parse_on_disk_float_type(hdr)
    grid_type = _transport_parse_grid_type(hdr)
    topology = _transport_parse_topology(hdr)
    geometry = _parse_transport_geometry(hdr, grid_type, topology)
    planet_radius_m = Float64(get(hdr, :planet_radius_m, EARTH_RADIUS))
    isfinite(planet_radius_m) && planet_radius_m > 0 ||
        throw(ArgumentError("planet_radius_m must be finite and positive; got $(planet_radius_m)"))
    ncell = Int(hdr.ncell)
    nface_h = Int(hdr.nface_h)
    nlevel = Int(hdr.nlevel)
    nwindow = Int(hdr.nwindow)
    A_ifc = Float64.(collect(hdr.A_ifc))
    B_ifc = Float64.(collect(hdr.B_ifc))
    surface_pressure_basis = haskey(hdr, :surface_pressure_basis) ?
        _transport_parse_surface_pressure_basis(hdr.surface_pressure_basis) :
        legacy_surface_pressure_basis
    payload_sections = _transport_parse_sections(hdr)
    source_flux_sampling = _transport_parse_symbol_key(hdr, :source_flux_sampling)
    air_mass_sampling    = _transport_parse_symbol_key(hdr, :air_mass_sampling)
    flux_sampling        = _transport_parse_symbol_key(hdr, :flux_sampling)
    flux_kind            = _transport_parse_symbol_key(hdr, :flux_kind)
    humidity_sampling    = _transport_parse_symbol_key(hdr, :humidity_sampling)
    delta_semantics      = _transport_parse_symbol_key(hdr, :delta_semantics)
    poisson_balance_target_scale = Float64(hdr.poisson_balance_target_scale)
    poisson_balance_target_semantics = String(hdr.poisson_balance_target_semantics)
    steps_per_window = Int(hdr.steps_per_window)
    steps_schedule = _parse_steps_per_window_schedule(hdr, nwindow, steps_per_window)
    poisson_scale_schedule = _parse_poisson_scale_schedule(
        hdr, nwindow, poisson_balance_target_scale)
    n_geometry_elems = Int(hdr.n_geometry_elems)
    elems_per_window = Int(hdr.elems_per_window)
    flux_fidelity = parse_flux_fidelity_header(hdr, nwindow, nlevel)
    omega_fidelity = parse_omega_fidelity_header(hdr)
    column_closure = parse_column_closure_header(hdr, nwindow)
    raw_header = Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr))
    flux_construction = parse_flux_construction_header(raw_header, format_version)

    return TransportBinaryHeader(
        format_version,
        header_bytes,
        disk_ft,
        float_bytes,
        geometry,
        planet_radius_m,
        ncell,
        nface_h,
        nlevel,
        nwindow,
        Float64(hdr.dt_met_seconds),
        Float64(hdr.half_dt_seconds),
        steps_per_window,
        steps_schedule,
        source_flux_sampling,
        air_mass_sampling,
        flux_sampling,
        flux_kind,
        humidity_sampling,
        delta_semantics,
        poisson_balance_target_scale,
        poisson_balance_target_semantics,
        poisson_scale_schedule,
        A_ifc,
        B_ifc,
        _transport_parse_mass_basis(hdr),
        surface_pressure_basis,
        _transport_parse_tm5_convection_mass_basis(hdr, payload_sections),
        payload_sections,
        n_geometry_elems,
        elems_per_window,
        flux_fidelity,
        omega_fidelity,
        column_closure,
        flux_construction,
        raw_header,
    )
end


function _transport_common_header(grid_type::String,
                                  horizontal_topology::String,
                                  ncell::Int,
                                  nface_h::Int,
                                  nlevel::Int,
                                  nwindow::Int,
                                  vc,
                                  payload_sections::Vector{Symbol},
                                  elems_per_window::Int;
                                  FT::Type{<:AbstractFloat},
                                  planet_radius::Real,
                                  header_bytes::Int,
                                  dt_met_seconds::Real,
                                  half_dt_seconds::Real,
                                  steps_per_window::Integer,
                                  mass_basis::Symbol,
                                  surface_pressure_basis::Union{TotalSurfacePressure,
                                                                DrySurfacePressure},
                                  source_flux_sampling::Symbol,
                                  air_mass_sampling::Symbol,
                                  flux_sampling::Symbol,
                                  flux_kind::Symbol,
                                  humidity_sampling::Symbol,
                                  delta_semantics::Symbol)
    n_qv_start = (:qv_start in payload_sections) ? ncell * nlevel : 0
    n_qv_end = (:qv_end in payload_sections) ? ncell * nlevel : 0
    n_surface = (:pblh in payload_sections) ? ncell : 0
    n_tm5 = (:entu in payload_sections) ? ncell * nlevel : 0
    n_vdiff = (:vdiff_u in payload_sections) ? ncell * nlevel : 0
    humidity_sampling = humidity_sampling === :auto ? _transport_default_humidity_sampling(payload_sections) : _transport_normalize_symbol(humidity_sampling)
    delta_semantics = delta_semantics === :auto ? _transport_default_delta_semantics(payload_sections) : _transport_normalize_symbol(delta_semantics)
    steps = Int(steps_per_window)
    steps > 0 || throw(ArgumentError("steps_per_window must be positive"))
    step_schedule = fill(steps, nwindow)
    target = poisson_balance_target(:substep_mass_amount, step_schedule; steps_per_window=steps)
    poisson_scale_schedule = target.by_window
    contract = TransportBinaryContract(
        source_flux_sampling = source_flux_sampling,
        air_mass_sampling = air_mass_sampling,
        flux_sampling = flux_sampling,
        flux_kind = flux_kind,
        delta_semantics = delta_semantics,
        humidity_sampling = humidity_sampling,
        poisson_balance_target_scale = target.scale,
        poisson_balance_target_semantics = target.semantics,
    )

    return Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => TRANSPORT_BINARY_FORMAT_VERSION,
        "header_bytes" => header_bytes,
        "float_type" => string(FT),
        "float_bytes" => sizeof(FT),
        "grid_type" => grid_type,
        "horizontal_topology" => horizontal_topology,
        "planet_radius_m" => Float64(planet_radius),
        "ncell" => ncell,
        "nface_h" => nface_h,
        "nlevel" => nlevel,
        "nwindow" => nwindow,
        "vertical_coordinate_type" => "hybrid_sigma_pressure",
        "A_ifc" => Float64.(vc.A),
        "B_ifc" => Float64.(vc.B),
        "dt_met_seconds" => Float64(dt_met_seconds),
        "half_dt_seconds" => Float64(half_dt_seconds),
        "steps_per_window" => steps,
        "steps_per_window_by_window" => step_schedule,
        "time_step_schedule" => "constant",
        "source_flux_sampling" => String(contract.source_flux_sampling),
        "air_mass_sampling" => String(contract.air_mass_sampling),
        "flux_sampling" => String(contract.flux_sampling),
        "flux_kind" => String(contract.flux_kind),
        "humidity_sampling" => String(contract.humidity_sampling),
        "delta_semantics" => String(contract.delta_semantics),
        "poisson_balance_target_scale" => contract.poisson_balance_target_scale,
        "poisson_balance_target_semantics" => contract.poisson_balance_target_semantics,
        "poisson_balance_target_scale_by_window" => poisson_scale_schedule,
        "mass_basis" => String(mass_basis),
        "surface_pressure_basis" =>
            transport_surface_pressure_basis_label(surface_pressure_basis),
        "payload_sections" => String.(payload_sections),
        "include_qv_endpoints" => (:qv_start in payload_sections) || (:qv_end in payload_sections),
        "include_flux_delta" => any(section in FLUX_DELTA_PAYLOAD_SECTIONS for section in payload_sections),
        "include_surface" => n_surface > 0,
        "surface_payload" => n_surface > 0 ? "pbl_raw_v2" : "none",
        "include_tm5conv" => n_tm5 > 0,
        "include_gchp_vdiff" => n_vdiff > 0,
        "gchp_vdiff_payload" => n_vdiff > 0 ? "u_v_t_qv_layer_center_v1" : "none",
        "n_qv_start" => n_qv_start,
        "n_qv_end" => n_qv_end,
        "n_pblh" => n_surface,
        "n_ustar" => n_surface,
        "n_pbl_hflux" => n_surface,
        "n_t2m" => n_surface,
        "n_entu" => n_tm5,
        "n_detu" => n_tm5,
        "n_entd" => n_tm5,
        "n_detd" => n_tm5,
        "n_vdiff_u" => n_vdiff,
        "n_vdiff_v" => n_vdiff,
        "n_vdiff_t" => n_vdiff,
        "n_vdiff_qv" => n_vdiff,
        "n_geometry_elems" => 0,
        "elems_per_window" => elems_per_window,
        "flux_fidelity_schema" => FLUX_FIDELITY_SCHEMA,
        "flux_fidelity_status" => fidelity_tag(FidelityUnavailable()),
        "flux_fidelity_kind" => fidelity_tag(NoFluxFidelityKind()),
        "flux_fidelity_reason" => "evidence_not_attached",
        "flux_fidelity_sample_rule" => "none",
        "flux_fidelity_sample_seed" => string(FLUX_FIDELITY_SAMPLE_SEED),
        "flux_fidelity_samples_per_window" => 0,
        "flux_fidelity_sample_count" => 0,
        "flux_fidelity_windows" => 0,
        "omega_fidelity_status" => fidelity_tag(FidelityUnavailable()),
        "omega_fidelity_reason" => "source_not_opened",
        "omega_fidelity_source" => "none",
        "omega_fidelity_sampling" => "none",
        "omega_fidelity_basis" => "none",
        "omega_fidelity_sample_count" => 0,
    )
end
