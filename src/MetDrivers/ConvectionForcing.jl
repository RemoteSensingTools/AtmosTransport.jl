# ---------------------------------------------------------------------------
# ConvectionForcing — per-window convective mass-flux container.
#
# Runtime data flow: convection forcing lives at TWO runtime slots:
#
#   1. On the transport window: `window.convection::Union{Nothing,
#      ConvectionForcing}`, populated by `load_transport_window!`
#      once per met window.
#   2. On the model: `model.convection_forcing::ConvectionForcing`,
#      refreshed every substep by `DrivenSimulation._refresh_forcing!`
#      via `copy_convection_forcing!`.
#
# Both slots carry the same struct type. Capability (which of
# `cmfmc`, `dtrain`, `tm5_fields` are non-nothing) is invariant for
# the lifetime of a run.
# ---------------------------------------------------------------------------

"""
    ConvectionForcing{CM, DT, TM, CB}

Container for one window (or one substep) of convective mass-flux
forcing. Four optional payload slots:

- `cmfmc` — cloud updraft mass flux at level interfaces. Supported
  layouts are structured `(Nx, Ny, Nz+1)`, face-indexed
  `(ncell, Nz+1)`, and cubed-sphere panel tuples
  `NTuple{6, <:AbstractArray{FT, 3}}` with per-panel shape
  `(Nc, Nc, Nz+1)`. GCHP / `CMFMCConvection` consumer.
- `dtrain` — detraining mass flux at layer centers. Supported layouts
  match `cmfmc`, with layer-center shape `(Nx, Ny, Nz)`,
  `(ncell, Nz)`, or `NTuple{6}` of `(Nc, Nc, Nz)`. When `nothing`,
  `CMFMCConvection` falls through to Tiedtke-style single-flux
  transport.
- `tm5_fields :: Union{Nothing, NamedTuple{(:entu, :detu, :entd, :detd)}}` —
  four-field entrainment/detrainment arrays at layer centers
  `(Nx, Ny, Nz)`. TM5 / `TM5Convection` consumer.
- `cloud_base` — convective cloud-base layer (top-down index stored as a
  float), 2-D like the surface fields (`NTuple{6}` of `(Nc, Nc)` on the cube),
  or `nothing`. GEOS-Chem takes it from DQRCU; `CMFMCConvection` otherwise
  derives it from the lowest non-zero CMFMC edge. Requires `cmfmc`.

# Invariants (enforced by the inner constructor)

- **DTRAIN requires CMFMC.** `dtrain !== nothing ⇒ cmfmc !== nothing`.
  DTRAIN without CMFMC is meaningless (DTRAIN detrains from the
  updraft mass flux; no mass flux, nothing to detrain).

- **Dual capability is allowed.**
  A binary may carry both CMFMC and TM5 payloads simultaneously; the
  sim selects which capability to consume based on the installed
  operator. The invariant here only enforces the load-bearing
  constraint above.

- **Capability is INVARIANT for the lifetime of a `DrivenSimulation`.**
  `copy_convection_forcing!` enforces strict tuple
  match between `dst` and `src` so a mid-run capability toggle
  raises an error — catches both "stale values" (dst has a field src
  doesn't) and "missing destination" (src has a field dst doesn't).

# Default construction

`ConvectionForcing()` produces an all-nothing placeholder. This is
the initial value of `TransportModel.convection_forcing`;
`DrivenSimulation` allocates real buffers at construction via
`allocate_convection_forcing_like`.

# See also

- [`has_convection_forcing`](@ref) — capability probe.
- [`copy_convection_forcing!`](@ref) — per-substep refresh copy.
- [`allocate_convection_forcing_like`](@ref) — sim-construction
  allocation.
"""
struct ConvectionForcing{CM, DT, TM, CB}
    cmfmc      :: CM
    dtrain     :: DT
    tm5_fields :: TM
    cloud_base :: CB

    function ConvectionForcing{CM, DT, TM, CB}(cmfmc::CM, dtrain::DT, tm5_fields::TM,
                                               cloud_base::CB) where {CM, DT, TM, CB}
        cloud_base !== nothing && cmfmc === nothing && throw(ArgumentError(
            "ConvectionForcing: cloud_base is populated but cmfmc is nothing."))
        if dtrain !== nothing && cmfmc === nothing
            throw(ArgumentError(
                "ConvectionForcing: dtrain is populated but cmfmc is nothing. " *
                "DTRAIN requires CMFMC (DTRAIN detrains from the updraft mass flux). " *
                "If you meant a Tiedtke-style fallback, set dtrain = nothing too."))
        end
        return new{CM, DT, TM, CB}(cmfmc, dtrain, tm5_fields, cloud_base)
    end
end

# Defining a validating inner constructor suppresses Julia's
# auto-generated outer constructors, so we provide them explicitly.
# - 3/4-arg outer: forwards to the validating inner (no cloud base by default).
# - 0-arg default: all-nothing placeholder.
ConvectionForcing(cmfmc::CM, dtrain::DT, tm5_fields::TM,
                  cloud_base::CB = nothing) where {CM, DT, TM, CB} =
    ConvectionForcing{CM, DT, TM, CB}(cmfmc, dtrain, tm5_fields, cloud_base)
ConvectionForcing() = ConvectionForcing(nothing, nothing, nothing)

# =========================================================================
# Capability probes
# =========================================================================

"""
    has_convection_forcing(forcing::ConvectionForcing) -> Bool

Whether `forcing` carries any non-nothing payload. Returns `false`
for the all-nothing placeholder. Used by `_refresh_forcing!` as a
gate so the copy path is skipped for models without an active
convection operator.

The window-level overload `has_convection_forcing(window) =
window.convection !== nothing` is defined in `transport_binary/window.jl`
alongside the window struct extensions.
"""
has_convection_forcing(forcing::ConvectionForcing) =
    forcing.cmfmc !== nothing ||
    forcing.dtrain !== nothing ||
    forcing.tm5_fields !== nothing

"""
    _cap(f::ConvectionForcing) -> NTuple{4, Bool}

Capability tuple `(has_cmfmc, has_dtrain, has_tm5_fields, has_cloud_base)`. Used by
`_check_capability_match` to enforce capability invariance —
`copy_convection_forcing!` requires exact capability agreement between
src and dst.
"""
@inline _cap(f::ConvectionForcing) =
    (f.cmfmc !== nothing, f.dtrain !== nothing, f.tm5_fields !== nothing,
     f.cloud_base !== nothing)

function _check_capability_match(dst::ConvectionForcing, src::ConvectionForcing)
    _cap(dst) == _cap(src) ||
        throw(ArgumentError(
            "ConvectionForcing capability mismatch " *
            "(dst: $(_cap(dst)), src: $(_cap(src))). " *
            "Per plan 18 v5.1 Decision 27, capability is invariant for the " *
            "lifetime of a DrivenSimulation. Check the preprocessing pipeline: " *
            "a binary must write a consistent set of convection blocks (cmfmc, " *
            "dtrain, tm5_fields) across all windows."))
    return nothing
end

# =========================================================================
# Per-substep copy (refresh window → model)
# =========================================================================

"""
    copy_convection_forcing!(dst::ConvectionForcing, src::ConvectionForcing) -> dst

Copy `src`'s arrays into `dst`'s preallocated buffers in place.
Preserves `===` identity of the destination's arrays — this is what
makes the per-substep refresh zero-allocation after the sim-construction
allocation step.

Enforces strict capability match first: both sides must
have identical `_cap(...)` tuples. Otherwise throws `ArgumentError`.
This catches both directions of mismatch (dst has a field src lacks,
or vice versa) — both are silent correctness hazards.

Used by `DrivenSimulation._refresh_forcing!` to populate
`sim.model.convection_forcing` from `sim.window.convection` each substep.
"""
function copy_convection_forcing!(dst::ConvectionForcing, src::ConvectionForcing)
    _check_capability_match(dst, src)
    if src.cmfmc !== nothing
        _copy_convection_payload!(dst.cmfmc, src.cmfmc)
    end
    if src.dtrain !== nothing
        _copy_convection_payload!(dst.dtrain, src.dtrain)
    end
    if src.tm5_fields !== nothing
        for name in (:entu, :detu, :entd, :detd)
            _copy_convection_payload!(getfield(dst.tm5_fields, name),
                                      getfield(src.tm5_fields, name))
        end
    end
    src.cloud_base === nothing || _copy_convection_payload!(dst.cloud_base, src.cloud_base)
    return dst
end

# =========================================================================
# Sim-construction allocation
# =========================================================================

# Small backend-adapter helper. The implementation delegates to
# Architectures so convection forcing follows the same CPU/CUDA/Metal
# residency rule as transport windows and surface sources.
@inline function _convection_backend_adapter(reference_array)
    return array_adapter_for(reference_array)
end

@inline _convection_backend_adapter(reference_array::NTuple{6}) =
    _convection_backend_adapter(reference_array[1])

@inline _copy_convection_payload!(dst, src) = copyto!(dst, src)
@inline function _copy_convection_payload!(dst::NTuple{N}, src::NTuple{N}) where N
    @inbounds for i in 1:N
        _copy_convection_payload!(dst[i], src[i])
    end
    return dst
end

@inline function _allocate_convection_payload_like(src, adaptor)
    return adaptor === Array ? similar(src) : Base.invokelatest(adaptor, similar(src))
end

@inline function _allocate_convection_payload_like(src::NTuple{N}, adaptor) where N
    return ntuple(i -> _allocate_convection_payload_like(src[i], adaptor), N)
end

"""
    allocate_convection_forcing_like(src::ConvectionForcing, backend_hint) -> ConvectionForcing

Build a destination `ConvectionForcing` whose array fields are
`similar(src_field)` — same shape, same element type, same backend
(inferred from `backend_hint`, typically `model.state.air_mass`).
Capability (which fields are non-nothing) exactly matches `src`.

Used by `DrivenSimulation` construction to seed
`model.convection_forcing` from the first loaded window. After
this step, `copy_convection_forcing!` reuses the same buffers across
all subsequent substeps.

The all-nothing placeholder `ConvectionForcing()` produces another
all-nothing placeholder when run through this helper.
"""
function allocate_convection_forcing_like(src::ConvectionForcing, backend_hint)
    adaptor = _convection_backend_adapter(backend_hint)

    cmfmc = src.cmfmc === nothing ? nothing : _allocate_convection_payload_like(src.cmfmc, adaptor)
    dtrain = src.dtrain === nothing ? nothing : _allocate_convection_payload_like(src.dtrain, adaptor)
    tm5_fields = src.tm5_fields === nothing ? nothing : (
        entu = _allocate_convection_payload_like(src.tm5_fields.entu, adaptor),
        detu = _allocate_convection_payload_like(src.tm5_fields.detu, adaptor),
        entd = _allocate_convection_payload_like(src.tm5_fields.entd, adaptor),
        detd = _allocate_convection_payload_like(src.tm5_fields.detd, adaptor),
    )
    cloud_base = src.cloud_base === nothing ? nothing :
        _allocate_convection_payload_like(src.cloud_base, adaptor)
    return ConvectionForcing(cmfmc, dtrain, tm5_fields, cloud_base)
end

# =========================================================================
# Adapt.adapt_structure — GPU transfer support
# =========================================================================

function Adapt.adapt_structure(to, f::ConvectionForcing)
    cmfmc = f.cmfmc === nothing ? nothing : Adapt.adapt(to, f.cmfmc)
    dtrain = f.dtrain === nothing ? nothing : Adapt.adapt(to, f.dtrain)
    tm5_fields = f.tm5_fields === nothing ? nothing : (
        entu = Adapt.adapt(to, f.tm5_fields.entu),
        detu = Adapt.adapt(to, f.tm5_fields.detu),
        entd = Adapt.adapt(to, f.tm5_fields.entd),
        detd = Adapt.adapt(to, f.tm5_fields.detd),
    )
    return ConvectionForcing(cmfmc, dtrain, tm5_fields, Adapt.adapt(to, f.cloud_base))
end

export ConvectionForcing, has_convection_forcing
export copy_convection_forcing!, allocate_convection_forcing_like
