# Typed runtime-physics configuration specs.
#
# Parse TOML values into typed specifications, then construct the requested
# operators by dispatch. Numerical kernels receive operators, not TOML tables.
#
#   TOML section Dict  --<family>_spec-->  <Spec>  --materialize(spec, style[, …])-->  <operator>
#
# Each family has three parts: a specification type, a section parser, and
# `materialize` methods. Parsers validate values and option combinations;
# materialization applies topology/capability gates and operator constructors.
# GPU solver eligibility is checked later, when precision and state depth are
# known (see Convection/TM5Convection.jl).
#
# Keep all physics `materialize` methods here. RuntimePhysicsRecipe.jl supplies
# the grid/driver context helpers and assembles the resulting operators.
#
# This file holds the convection, advection, chemistry, and diffusion spec families.

# --- Typed TOML-value accessors (clean errors at the boundary) -------------
# Report the section, key, and offending value when a TOML value has the wrong
# type, before attempting numeric conversion.
# `label` is the TOML table name (e.g. "[convection]") for the error message.

function _spec_bool(section, key::AbstractString, default::Bool, label::AbstractString)
    v = get(section, key, default)
    v isa Bool || throw(ArgumentError("$(label).$(key) must be true or false; got $(repr(v))"))
    return v
end
function _spec_int(section, key::AbstractString, default::Int, label::AbstractString)
    v = get(section, key, default)
    v isa Integer || throw(ArgumentError("$(label).$(key) must be an integer; got $(repr(v))"))
    return Int(v)
end
function _spec_float64(section, key::AbstractString, default::Float64, label::AbstractString)
    v = get(section, key, default)
    v isa Real || throw(ArgumentError("$(label).$(key) must be a number; got $(repr(v))"))
    return Float64(v)
end

# =========================================================================
# Convection
# =========================================================================

abstract type AbstractConvectionSpec end

# The two collaborative-LU kinds (`tm5`, `cmfmc_matrix`) share the same knobs and
# the same LU machinery — they differ only in which operator `materialize` builds.
abstract type AbstractCollabLUConvectionSpec <: AbstractConvectionSpec end

struct NoConvectionSpec    <: AbstractConvectionSpec end
# `clamp` opts the explicit GCHP CMFMC scheme into the positivity clamp +
# whole-column rescale (stable at few sub-steps for strong convection, still
# conservative). Default false = the pure conservative explicit scheme.
# `cloud_base = "dqrcu"` takes GEOS-Chem's convective cloud base from the binary.
struct CMFMCConvectionSpec{CB <: AbstractCloudBase} <: AbstractConvectionSpec
    clamp      :: Bool
    cloud_base :: CB
end

function _cloud_base_rule(section)
    raw = lowercase(String(get(section, "cloud_base", "cmfmc")))
    raw == "cmfmc" && return CMFMCEdgeCloudBase()
    raw == "dqrcu" && return ArchivedCloudBase()
    throw(ArgumentError("Unknown [convection] cloud_base: $(repr(raw)). Supported: cmfmc | dqrcu"))
end

# Configuration scalars are independent of tracer precision. In particular,
# TM5Convection's type parameter describes the workspace budget, not the arrays
# or arithmetic used by the solver.
#
#   tile_workspace_gib — legacy column-tile target in binary GiB; default 1.0.
#     Collaborative solves defer this global scratch until a fallback needs it.
#   use_collab_lu — request one GPU workgroup per column, sharing the LU factors
#     across tracer batches. Defaults to the legacy solve. Actual speed and
#     eligibility depend on backend, precision, and effective vertical depth.
#   lmax_conv — 0 uses all Nz levels; a positive value selects that many bottom
#     levels. Validate the retained region against the forcing's cloud depths.
#   n_merge — 1 keeps individual levels; larger values aggregate adjacent layers
#     and change vertical mixing. The effective matrix has fld(L, n_merge) rows;
#     a nondivisible active depth L also omits top layers from the fine span.
# Tracer batching requires neither truncation nor aggregation. These numerical
# approximations apply only when the collaborative solver actually engages.
struct TM5ConvectionSpec <: AbstractCollabLUConvectionSpec
    tile_workspace_gib :: Float64
    use_collab_lu      :: Bool
    lmax_conv          :: Int
    n_merge            :: Int
end

struct CMFMCMatrixConvectionSpec <: AbstractCollabLUConvectionSpec
    tile_workspace_gib :: Float64
    use_collab_lu      :: Bool
    lmax_conv          :: Int
    n_merge            :: Int
end

function _parse_convection_kind(section)
    raw = lowercase(String(get(section, "kind", "none")))
    raw == "none"         && return :none
    raw == "tm5"          && return :tm5
    raw == "cmfmc"        && return :cmfmc
    raw == "cmfmc_matrix" && return :cmfmc_matrix
    throw(ArgumentError(
        "Unknown [convection] kind: $(repr(raw)). Supported: none | tm5 | cmfmc | cmfmc_matrix"))
end

# Shared knob extraction + validation for the collaborative-LU kinds.
function _collab_lu_knobs(section)
    budget     = _spec_float64(section, "tile_workspace_gib", 1.0, "[convection]")
    use_collab = _spec_bool(section, "use_collab_lu", false, "[convection]")
    lmax_conv  = _spec_int(section, "lmax_conv", 0, "[convection]")
    n_merge    = _spec_int(section, "n_merge", 1, "[convection]")

    if (lmax_conv != 0 || n_merge != 1) && !use_collab
        throw(ArgumentError(
            "[convection] lmax_conv/n_merge only take effect with use_collab_lu = true " *
            "— they steer the collaborative-LU kernel, and the per-thread fallback ignores " *
            "them. Set use_collab_lu = true to request the GPU solver, or remove " *
            "lmax_conv/n_merge. Solver support depends on backend, precision, and " *
            "effective depth; see TM5Convection. Got lmax_conv=$(lmax_conv), n_merge=$(n_merge), " *
            "use_collab_lu=false."))
    end
    return (budget, use_collab, lmax_conv, n_merge)
end

"""
    convection_spec(section) -> AbstractConvectionSpec

Parse a `[convection]` TOML section into a typed spec, validating at the boundary.

Nondefault `lmax_conv`/`n_merge` require `use_collab_lu = true`; otherwise parsing
throws an `ArgumentError`. They affect the workgroup-collaborative solver only.
Eligibility is checked against the backend, tracer precision, and state depth
at execution time. The legacy fallback uses the full column without aggregation.
"""
function convection_spec(section)
    kind = _parse_convection_kind(section)
    kind !== :cmfmc && haskey(section, "cloud_base") && throw(ArgumentError(
        "[convection] cloud_base applies to kind = \"cmfmc\" only; got kind = $(repr(String(section["kind"])))"))
    kind === :none  && return NoConvectionSpec()
    kind === :cmfmc &&
        return CMFMCConvectionSpec(_spec_bool(section, "clamp", false, "[convection]"),
                                   _cloud_base_rule(section))
    knobs = _collab_lu_knobs(section)
    kind === :tm5 && return TM5ConvectionSpec(knobs...)
    return CMFMCMatrixConvectionSpec(knobs...)  # :cmfmc_matrix
end

# materialize — turn the spec into the runtime operator. Convection needs neither
# the run `FT` nor the driver context, so the clean form is `materialize(spec, style)`
# (no unused ceremony args). The two collaborative-LU specs share knobs but build
# DIFFERENT operators, so they dispatch on their concrete types.
materialize(::NoConvectionSpec, ::AbstractRuntimeRecipeStyle)    = NoConvection()
materialize(s::CMFMCConvectionSpec, ::AbstractRuntimeRecipeStyle) =
    CMFMCConvection(; clamp = s.clamp, cloud_base = s.cloud_base)
materialize(s::TM5ConvectionSpec, ::AbstractRuntimeRecipeStyle) =
    TM5Convection(; tile_workspace_gib = s.tile_workspace_gib, use_collab_lu = s.use_collab_lu,
                    lmax_conv = s.lmax_conv, n_merge = s.n_merge)
materialize(s::CMFMCMatrixConvectionSpec, ::AbstractRuntimeRecipeStyle) =
    CMFMCMatrixConvection(; tile_workspace_gib = s.tile_workspace_gib,
                            use_collab_lu = s.use_collab_lu, lmax_conv = s.lmax_conv,
                            n_merge = s.n_merge)

# The singleton specs print fine via the default `Base.summary`; only the
# knob-carrying specs need a custom one (shared across both collab-LU variants).
Base.summary(s::AbstractCollabLUConvectionSpec) =
    "$(nameof(typeof(s)))(use_collab_lu=$(s.use_collab_lu), lmax_conv=$(s.lmax_conv), n_merge=$(s.n_merge))"

# =========================================================================
# Advection
# =========================================================================

abstract type AbstractAdvectionSpec end

struct UpwindAdvectionSpec <: AbstractAdvectionSpec end
struct SlopesAdvectionSpec <: AbstractAdvectionSpec end
struct PPMAdvectionSpec    <: AbstractAdvectionSpec end
struct NoAdvectionSpec     <: AbstractAdvectionSpec end

# Lin–Rood is cubed-sphere only. `order` selects its PPM edge-value family.
struct LinRoodAdvectionSpec <: AbstractAdvectionSpec
    order :: Int
end

function _parse_advection_scheme(section)
    raw = lowercase(String(get(section, "scheme", "upwind")))
    raw == "upwind" && return :upwind
    raw == "slopes" && return :slopes
    raw == "ppm"    && return :ppm
    raw == "none"   && return :none
    raw == "linrood" && return :linrood
    throw(ArgumentError(
        "Unknown [advection] scheme: $(repr(raw)). Supported: upwind | slopes | ppm | linrood | none"))
end

"""
    advection_spec(section) -> AbstractAdvectionSpec

Parse an `[advection]` section into a typed spec. `ppm_order` is only meaningful
for `scheme = "linrood"`; pairing it with `scheme = "ppm"` is rejected (the split
PPM path takes no order knob). An omitted selector defaults to upwind; an
omitted Lin–Rood `ppm_order` defaults to 5.
"""
function advection_spec(section)
    kind = _parse_advection_scheme(section)
    kind === :upwind && return UpwindAdvectionSpec()
    kind === :slopes && return SlopesAdvectionSpec()
    kind === :none   && return NoAdvectionSpec()
    if kind === :ppm
        haskey(section, "ppm_order") && throw(ArgumentError(
            "[advection] `ppm_order` is only valid with `scheme = \"linrood\"`; " *
            "`scheme = \"ppm\"` selects the standard split `PPMScheme()` path."))
        return PPMAdvectionSpec()
    end
    return LinRoodAdvectionSpec(_spec_int(section, "ppm_order", 5, "[advection]"))  # :linrood
end

# Materialize with topology gates close to construction: RG currently accepts
# only Upwind/NoAdvection, while LinRood is cubed-sphere only.
materialize(::UpwindAdvectionSpec, ::AbstractRuntimeRecipeStyle) = UpwindScheme()
materialize(::SlopesAdvectionSpec, ::AbstractRuntimeRecipeStyle) = SlopesScheme()
materialize(::PPMAdvectionSpec,    ::AbstractRuntimeRecipeStyle) = PPMScheme()
materialize(::SlopesAdvectionSpec, ::ReducedGaussianRuntimeRecipeStyle) =
    throw(ArgumentError(
        "[advection] `scheme = \"slopes\"` is not implemented for reduced-Gaussian runs; " *
        "use `scheme = \"upwind\"` or `scheme = \"none\"`."))
materialize(::PPMAdvectionSpec, ::ReducedGaussianRuntimeRecipeStyle) =
    throw(ArgumentError(
        "[advection] `scheme = \"ppm\"` is not implemented for reduced-Gaussian runs; " *
        "use `scheme = \"upwind\"` or `scheme = \"none\"`."))
materialize(::NoAdvectionSpec,     ::AbstractRuntimeRecipeStyle) = NoAdvection()
materialize(s::LinRoodAdvectionSpec, ::CubedSphereRuntimeRecipeStyle) = LinRoodPPMScheme(s.order)
materialize(::LinRoodAdvectionSpec, ::AbstractStructuredRuntimeRecipeStyle) = throw(ArgumentError(
    "[advection] `scheme = \"linrood\"` is only available on cubed-sphere runs."))

Base.summary(s::LinRoodAdvectionSpec) = "LinRoodAdvectionSpec(order=$(s.order))"

# =========================================================================
# Chemistry
# =========================================================================
#
# Chemistry is topology-independent, so its `materialize` dispatches on the run
# `FT` (not a style) — `ExponentialDecay` converts each half-life → first-order
# rate `log(2)/T` at FT precision.

abstract type AbstractChemistrySpec end

struct NoChemistrySpec <: AbstractChemistrySpec end

# Carries the parsed (tracer-name → half-life-seconds) map as a NamedTuple of raw
# Float64 values. The Symbol keys are the `[tracers.<name>]` names the run carries;
# `ExponentialDecay.apply!` resolves them against `state.tracer_names` at call time.
struct DecayChemistrySpec{NT <: NamedTuple} <: AbstractChemistrySpec
    half_lives :: NT
end

function _parse_chemistry_kind(section)
    raw = lowercase(String(get(section, "kind", "none")))
    raw == "none"  && return :none
    raw == "decay" && return :decay
    throw(ArgumentError(
        "Unknown [chemistry] kind: $(repr(raw)). Supported: none | decay"))
end

# A single half-life must be a positive number. The TOML parser normally hands us
# a Real, but a config arriving from test/programmatic code (or a stray string)
# would otherwise fail with a cryptic `MethodError(Float64, …)`; a zero/negative
# value silently yields an Inf/negative decay rate. Validate at the boundary.
function _spec_half_life(name, v)
    # `Bool <: Real`, so guard against it explicitly — otherwise `rn222 = true`
    # would pass `> 0` and silently store a 1-second half-life.
    (v isa Real && !(v isa Bool)) || throw(ArgumentError(
        "[chemistry.half_lives_seconds].$(name) must be a number; got $(repr(v))"))
    v > 0 || throw(ArgumentError(
        "[chemistry.half_lives_seconds].$(name) must be positive; got $(v)"))
    return Float64(v)
end

"""
    chemistry_spec(section) -> AbstractChemistrySpec

Parse a `[chemistry]` section into a typed spec. `kind = "decay"` with an empty
(or absent) `half_lives_seconds` table reduces to `NoChemistrySpec`. Each
half-life must be numeric, non-Boolean, and positive. Values are stored as
Float64 seconds and converted to tracer precision during materialization.
"""
function chemistry_spec(section)
    kind = _parse_chemistry_kind(section)
    kind === :none && return NoChemistrySpec()
    hl = get(section, "half_lives_seconds", Dict{String, Any}())  # :decay
    isempty(hl) && return NoChemistrySpec()
    syms = Symbol[]
    vals = Float64[]
    for (k, v) in pairs(hl)
        s = Symbol(k)
        # Distinct TOML string keys can't collide, but a programmatic
        # Dict{Any,Any} could carry both "rn222" and :rn222 → same Symbol. The old
        # `NamedTuple{sym_keys}(vals)` threw on that; preserve the visible error
        # rather than silently dropping a tracer.
        s in syms && throw(ArgumentError(
            "[chemistry.half_lives_seconds] has duplicate tracer name $(repr(s)) after " *
            "symbolization (keys that differ only by string-vs-Symbol collide)."))
        push!(syms, s)
        push!(vals, _spec_half_life(k, v))
    end
    return DecayChemistrySpec(NamedTuple{Tuple(syms)}(Tuple(vals)))
end

# `FT` is unused for the no-op, but the signature must accept it so the
# `materialize(spec, FT)` call site dispatches uniformly across both specs.
materialize(::NoChemistrySpec, ::Type{FT}) where {FT} = NoChemistry()
# Convert each half-life to `FT` BEFORE handing it to `ExponentialDecay`, exactly
# as the old builder did (`Tuple(FT(v) …)`). `ExponentialDecay` then forms the rate
# as `FT(log(2) / FT(T))`; pre-converting keeps the Float32 rounding bit-identical
# (`log(2) / Float32(T)` ≠ `Float32(log(2) / Float64(T))` in the last ULP).
materialize(s::DecayChemistrySpec, ::Type{FT}) where {FT} =
    ExponentialDecay(FT; map(FT, s.half_lives)...)

Base.summary(s::DecayChemistrySpec) =
    "DecayChemistrySpec($(join(keys(s.half_lives), ", ")))"

# =========================================================================
# Diffusion
# =========================================================================
#
# The one family whose `materialize` needs all three of `style` (Kz-field rank /
# cubed-sphere-only gating), `FT` (precision), and the runtime `context`
# (driver/reader — for the Kz-cache shape + binary-capability gate). The spec stays
# context-free so it can be parsed before any driver exists; the context work is
# deferred to `materialize`, which calls the helpers in `RuntimePhysicsRecipe.jl`
# (`_constant_runtime_kz_field`, `_pbl_cache_shape`, `_runtime_has_*`) — those resolve
# concrete grid/reader/driver types and are stubbed by tests, so they stay there.

abstract type AbstractDiffusionSpec end

struct NoDiffusionSpec <: AbstractDiffusionSpec end

# Constant Kz everywhere. `value` stored raw (Float64); the run `FT` is applied in
# `materialize` (the field rank comes from the style).
struct ConstantDiffusionSpec <: AbstractDiffusionSpec
    value                 :: Float64
    surface_flux_boundary :: Bool
end

# The three cubed-sphere, context-dependent closures. They differ only in which Kz
# field `materialize` builds and which binary capability it gates on; all carry the
# same parsed surface-flux-boundary flag. `WindowPBLKz` is the TM5 Beljaars-Viterbo
# local-Kz path (`kind = "tm5_beljaars_viterbo_local_kz"`).
struct WindowPBLKzDiffusionSpec <: AbstractDiffusionSpec
    surface_flux_boundary :: Bool
end
struct HoltslagBovilleVdiffDiffusionSpec <: AbstractDiffusionSpec
    surface_flux_boundary :: Bool
end
struct TM5DkgDiffusionSpec <: AbstractDiffusionSpec
    surface_flux_boundary :: Bool
end
# GEOS-Chem's non-local VDIFF always adds emissions before one full solve.
struct GCHPNonlocalVdiffDiffusionSpec <: AbstractDiffusionSpec end

function _parse_diffusion_kind(section)
    raw = lowercase(String(get(section, "kind", "none")))
    raw == "none"     && return :none
    raw == "constant" && return :constant
    raw == "tm5_beljaars_viterbo_local_kz" && return :pbl
    raw == "geoschem_holtslag_boville_vdiff" && return :vdiff
    raw == "tm5_dkg" && return :tm5_dkg
    raw == "geoschem_nonlocal_vdiff" && return :nonlocal_vdiff
    throw(ArgumentError(
        "Unknown [diffusion] kind: $(repr(raw)). Supported: none | constant | " *
        "tm5_beljaars_viterbo_local_kz | geoschem_holtslag_boville_vdiff | " *
        "geoschem_nonlocal_vdiff | tm5_dkg"))
end

"""
    diffusion_spec(section) -> AbstractDiffusionSpec

Parse a `[diffusion]` section into a typed spec, validating at the boundary. An
empty/absent section or `kind = "none"` is explicit "no diffusion". The legacy
`type = "..."` schema is rejected (it used to silently fall through to
`NoDiffusion`, hiding configs that expected diffusion to run); a present section
with no `kind` is rejected too.
"""
function diffusion_spec(section)
    # Empty / absent section is explicit "no diffusion".
    isempty(section) && return NoDiffusionSpec()
    # Reject the legacy `type = "..."` schema rather than silently mapping it to
    # NoDiffusion. Configs that said `type = "pbl"`/`"nonlocal_pbl"` expected
    # diffusion to run; the silent fall-through hid that for months. Migrate to
    # `kind`. (Preserved from the old builder.)
    haskey(section, "type") && !haskey(section, "kind") &&
        throw(ArgumentError(
            "[diffusion] uses unsupported `type = \"$(section["type"])\"`; use " *
            "`kind = \"...\"`. Supported kinds: \"none\", \"constant\", " *
            "\"tm5_beljaars_viterbo_local_kz\", " *
            "\"geoschem_holtslag_boville_vdiff\", \"tm5_dkg\"."))
    haskey(section, "kind") ||
        throw(ArgumentError(
            "[diffusion] section is present but has no `kind` key. " *
            "Set `kind = \"none\"`, `kind = \"constant\"`, " *
            "`kind = \"tm5_beljaars_viterbo_local_kz\"`, " *
            "`kind = \"geoschem_holtslag_boville_vdiff\"`, or " *
            "`kind = \"tm5_dkg\"`."))
    kind = _parse_diffusion_kind(section)
    kind === :none && return NoDiffusionSpec()
    if kind === :nonlocal_vdiff
        _spec_bool(section, "surface_flux_boundary", true, "[diffusion]") || throw(ArgumentError(
            "[diffusion] kind = \"geoschem_nonlocal_vdiff\" adds emissions before one full " *
            "diffusion solve (GEOS-Chem VDIFF); remove `surface_flux_boundary = false`."))
        return GCHPNonlocalVdiffDiffusionSpec()
    end
    sfb = _spec_bool(section, "surface_flux_boundary", false, "[diffusion]")
    kind === :constant &&
        return ConstantDiffusionSpec(_spec_float64(section, "value", 1.0, "[diffusion]"), sfb)
    kind === :pbl   && return WindowPBLKzDiffusionSpec(sfb)
    kind === :vdiff && return HoltslagBovilleVdiffDiffusionSpec(sfb)
    return TM5DkgDiffusionSpec(sfb)
end

@inline _diffusion_surface_coupling(b::Bool, ::AbstractRuntimeRecipeStyle) =
    b ? DiffusiveSurfaceFluxBoundary() : SplitSurfaceFluxCoupling()

function _diffusion_surface_coupling(b::Bool, ::ReducedGaussianRuntimeRecipeStyle)
    b && throw(ArgumentError(
        "[diffusion] `surface_flux_boundary = true` is not implemented for " *
        "reduced-Gaussian runs; omit the key or set it to false to use " *
        "V(dt/2) -> S(dt) -> V(dt/2)."))
    return SplitSurfaceFluxCoupling()
end

# materialize — uniform `(spec, style, FT, context)` signature so the recipe calls
# every diffusion kind identically. `NoDiffusion`/`constant` ignore `context`; the
# three CS closures dispatch on `CubedSphereRuntimeRecipeStyle` (build) vs the
# structured fallback (throw), exactly like the old `Val`-dispatch builders.
materialize(::NoDiffusionSpec, ::AbstractRuntimeRecipeStyle, ::Type{FT}, _context) where {FT} =
    NoDiffusion()

materialize(s::ConstantDiffusionSpec, style::AbstractRuntimeRecipeStyle, ::Type{FT},
            _context) where {FT} =
    ImplicitVerticalDiffusion(;
        kz_field = _constant_runtime_kz_field(style, FT(s.value)),
        surface_flux_coupling = _diffusion_surface_coupling(s.surface_flux_boundary, style))

function materialize(s::WindowPBLKzDiffusionSpec, ::CubedSphereRuntimeRecipeStyle,
                     ::Type{FT}, context) where {FT}
    _runtime_has_surface(context) ||
        throw(ArgumentError(
            "[diffusion] kind = \"tm5_beljaars_viterbo_local_kz\" requires pblh/ustar/pbl_hflux/t2m sections " *
            "in the cubed-sphere transport binary. Regenerate the binary with " *
            "include_surface=true."))
    Nc1, Nc2, Nz = _pbl_cache_shape(context)
    host_cache = ntuple(_ -> zeros(FT, Nc1, Nc2, Nz), 6)
    return ImplicitVerticalDiffusion(;
        kz_field = WindowPBLKzField(host_cache),
        surface_flux_coupling = _diffusion_surface_coupling(
            s.surface_flux_boundary, CubedSphereRuntimeRecipeStyle()))
end
materialize(::WindowPBLKzDiffusionSpec, ::AbstractRuntimeRecipeStyle, ::Type{FT},
            _context) where {FT} =
    throw(ArgumentError(
        "[diffusion] kind = \"tm5_beljaars_viterbo_local_kz\" is implemented for cubed-sphere " *
        "runtime binaries with pblh/ustar/pbl_hflux/t2m sections."))

function materialize(s::HoltslagBovilleVdiffDiffusionSpec, ::CubedSphereRuntimeRecipeStyle,
                     ::Type{FT}, context) where {FT}
    _runtime_has_gchp_vdiff(context) ||
        throw(ArgumentError(
            "[diffusion] kind = \"geoschem_holtslag_boville_vdiff\" requires " *
            "pblh/ustar/pbl_hflux/t2m and vdiff_u/vdiff_v/vdiff_t/vdiff_qv " *
            "sections in the cubed-sphere transport binary. Regenerate with " *
            "include_surface=true and include_gchp_vdiff=true."))
    Nc1, Nc2, Nz = _pbl_cache_shape(context)
    host_cache = ntuple(_ -> zeros(FT, Nc1, Nc2, Nz), 6)
    return ImplicitVerticalDiffusion(;
        kz_field = LocalHoltslagBovilleKzField(host_cache),
        surface_flux_coupling = _diffusion_surface_coupling(
            s.surface_flux_boundary, CubedSphereRuntimeRecipeStyle()))
end
materialize(::HoltslagBovilleVdiffDiffusionSpec, ::AbstractRuntimeRecipeStyle, ::Type{FT},
            _context) where {FT} =
    throw(ArgumentError(
        "[diffusion] kind = \"geoschem_holtslag_boville_vdiff\" is currently " *
        "implemented for cubed-sphere runtime binaries with GCHP VDIFF payloads."))

function materialize(s::TM5DkgDiffusionSpec, ::CubedSphereRuntimeRecipeStyle,
                     ::Type{FT}, context) where {FT}
    _runtime_has_precomputed_dkg(context) ||
        throw(ArgumentError(
            "[diffusion] kind = \"tm5_dkg\" requires an exact `:dkg` section in the cubed-sphere " *
            "transport binary. Regenerate with include_tm5_diffusion=true."))
    Nc1, Nc2, Nz = _pbl_cache_shape(context)
    host_cache = ntuple(_ -> zeros(FT, Nc1, Nc2, Nz), 6)
    field = PrecomputedCSDkgField(host_cache)
    return ImplicitVerticalDiffusion(;
        kz_field = field,
        surface_flux_coupling = _diffusion_surface_coupling(
            s.surface_flux_boundary, CubedSphereRuntimeRecipeStyle()))
end
function materialize(::GCHPNonlocalVdiffDiffusionSpec, ::CubedSphereRuntimeRecipeStyle,
                     ::Type{FT}, context) where {FT}
    _runtime_has_gchp_nonlocal_vdiff(context) || throw(ArgumentError(
        "[diffusion] kind = \"geoschem_nonlocal_vdiff\" requires pblh/ustar/pbl_hflux/t2m, " *
        "pbl_eflux and vdiff_u/vdiff_v/vdiff_t/vdiff_qv sections in the cubed-sphere transport " *
        "binary (MERRA-2 GEOS-Chem archive with include_vdiff_fields = true)."))
    Nc, _, Nz = _pbl_cache_shape(context)
    return ImplicitVerticalDiffusion(; kz_field = GCHPNonlocalPBLField(Nc, Nz, FT),
                                     surface_flux_coupling = DiffusiveSurfaceFluxBoundary())
end
materialize(::GCHPNonlocalVdiffDiffusionSpec, ::AbstractRuntimeRecipeStyle, ::Type{FT},
            _context) where {FT} =
    throw(ArgumentError(
        "[diffusion] kind = \"geoschem_nonlocal_vdiff\" is implemented for cubed-sphere runtime binaries."))

materialize(::TM5DkgDiffusionSpec, ::AbstractRuntimeRecipeStyle, ::Type{FT},
            _context) where {FT} =
    throw(ArgumentError(
        "[diffusion] kind = \"tm5_dkg\" is implemented for cubed-sphere " *
        "runtime binaries carrying a `:dkg` payload."))

export AbstractConvectionSpec, AbstractCollabLUConvectionSpec, NoConvectionSpec,
       TM5ConvectionSpec, CMFMCConvectionSpec, CMFMCMatrixConvectionSpec
export AbstractAdvectionSpec, UpwindAdvectionSpec, SlopesAdvectionSpec,
       PPMAdvectionSpec, NoAdvectionSpec, LinRoodAdvectionSpec
export AbstractChemistrySpec, NoChemistrySpec, DecayChemistrySpec
export AbstractDiffusionSpec, NoDiffusionSpec, ConstantDiffusionSpec,
       WindowPBLKzDiffusionSpec, HoltslagBovilleVdiffDiffusionSpec,
       GCHPNonlocalVdiffDiffusionSpec,
       TM5DkgDiffusionSpec
export convection_spec, advection_spec, chemistry_spec, diffusion_spec, materialize
