"""
    RuntimePhysicsRecipe

Validated operator composition for runtime-driven transport runners.

The recipe layer separates:

- component selection from TOML (`build_runtime_advection`,
  `build_runtime_diffusion`, `build_runtime_convection`)
- topology-specific construction rules (lat-lon, reduced Gaussian,
  cubed sphere) via dispatch on a lightweight runtime-style trait
- capability checks against readers / drivers
  (`validate_runtime_physics_recipe`)

This keeps the CLI scripts thin and prevents topology-specific
`if/elseif` trees from growing in parallel.
"""

# The runtime-style traits (`AbstractRuntimeRecipeStyle` + the LatLon/RG/CS
# variants) now live in `RuntimeRecipeStyles.jl`, included before this file (the
# `RuntimePhysicsSpecs.jl` `materialize` methods dispatch on them). The
# `_runtime_recipe_style(grid/driver/reader)` resolvers stay below.

struct RuntimePhysicsRecipe{AdvT, DiffT, ConvT, ChemT}
    advection  :: AdvT
    diffusion  :: DiffT
    convection :: ConvT
    chemistry  :: ChemT
end

# The flat-411 `catrine_co2` stub is gone. CS tracers
# now flow through the same `build_initial_mixing_ratio` +
# `pack_initial_tracer_mass` pipeline as LL/RG; `kind = "catrine_co2"`
# loads the Catrine NetCDF and regrids + remaps it conservatively onto
# the CS grid. Historical flat-411 behaviour is now expressed as
# `kind = "uniform" background = 4.11e-4`.

function _advection_section(cfg)
    run = get(cfg, "run", Dict{String,Any}())
    if haskey(cfg, "advection")
        for key in ("scheme", "ppm_order", "vertical")
            haskey(run, key) && throw(ArgumentError(
                "Advection option `[run].$(key)` is ambiguous because `[advection]` " *
                "is present. Move `$(key)` into `[advection]`; legacy `[run].$(key)` " *
                "is only accepted when `[advection]` is absent."))
        end
        return cfg["advection"]
    end
    return run
end
@inline _diffusion_section(cfg) = get(cfg, "diffusion", Dict{String,Any}())
@inline _convection_section(cfg) = get(cfg, "convection", Dict{String,Any}())
@inline _chemistry_section(cfg) = get(cfg, "chemistry", Dict{String,Any}())

@inline _runtime_recipe_style(style::AbstractRuntimeRecipeStyle) = style
@inline _runtime_recipe_style(::AtmosGrid{<:LatLonMesh}) = LatLonRuntimeRecipeStyle()
@inline _runtime_recipe_style(::AtmosGrid{<:ReducedGaussianMesh}) = ReducedGaussianRuntimeRecipeStyle()
@inline _runtime_recipe_style(::AtmosGrid{<:CubedSphereMesh}) = CubedSphereRuntimeRecipeStyle()
@inline _runtime_recipe_style(driver::AbstractMetDriver) = _runtime_recipe_style(driver_grid(driver))
@inline _runtime_recipe_style(
    ::TransportBinaryReader{<:Any, <:Any, LatLonBinaryGeometry},
) = LatLonRuntimeRecipeStyle()
@inline _runtime_recipe_style(
    ::TransportBinaryReader{<:Any, <:Any, ReducedGaussianBinaryGeometry},
) = ReducedGaussianRuntimeRecipeStyle()
@inline _runtime_recipe_style(
    ::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = CubedSphereRuntimeRecipeStyle()

function _runtime_recipe_style(context)
    throw(ArgumentError(
        "No runtime recipe style is defined for context $(typeof(context))."))
end

function build_runtime_advection(cfg, context)
    return build_runtime_advection(cfg, _runtime_recipe_style(context))
end

# Thin wrapper: parse the `[advection]` section into a typed `AbstractAdvectionSpec`
# once (validated), then materialize the scheme. The structured-vs-CS split lives in
# `materialize` (LinRood is CS-only); spec types + parser live in `RuntimePhysicsSpecs.jl`.
build_runtime_advection(cfg, style::AbstractRuntimeRecipeStyle) =
    materialize(advection_spec(_advection_section(cfg)), style)

function build_runtime_diffusion(cfg, context, ::Type{FT}) where FT
    return build_runtime_diffusion(cfg, _runtime_recipe_style(context), FT, context)
end

# Thin wrapper: parse the `[diffusion]` section into a typed `AbstractDiffusionSpec`
# once (validating the legacy `type=`/missing-`kind`/unknown-kind cases), then
# materialize. Diffusion is the one family whose `materialize` needs `style` (Kz-field
# rank / CS-only gating), `FT` (precision), AND the runtime `context` (driver/reader,
# for the Kz-cache shape + binary-capability gate). The spec stays context-free; the
# context work happens in `materialize`, which calls the helpers below. Spec types +
# parser + `materialize` live in `RuntimePhysicsSpecs.jl`.
build_runtime_diffusion(cfg, style::AbstractRuntimeRecipeStyle, ::Type{FT},
                        context = nothing) where FT =
    materialize(diffusion_spec(_diffusion_section(cfg)), style, FT, context)

# --- Context helpers the diffusion `materialize` methods call ----------------
# Kept here (not in RuntimePhysicsSpecs.jl) because they resolve concrete
# grid/reader/driver types; tests stub them via `AtmosTransport.Models._runtime_has_*`
# and `AtmosTransport.Models._pbl_cache_shape`.

@inline _constant_runtime_kz_field(::LatLonRuntimeRecipeStyle, value::FT) where FT =
    ConstantField{FT, 3}(value)
@inline _constant_runtime_kz_field(::ReducedGaussianRuntimeRecipeStyle, value::FT) where FT =
    ConstantField{FT, 2}(value)
@inline _constant_runtime_kz_field(::CubedSphereRuntimeRecipeStyle, value::FT) where FT =
    CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(value), 6))

function _pbl_cache_shape(context)
    throw(ArgumentError(
        "[diffusion] kind = \"tm5_beljaars_viterbo_local_kz\" requires a cubed-sphere reader or driver " *
        "context so the Kz cache can be sized."))
end
_pbl_cache_shape(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = (reader.header.geometry.Nc, reader.header.geometry.Nc, reader.header.nlevel)
_pbl_cache_shape(
    driver::TransportBinaryDriver{FT, ReaderT},
) where {FT, ReaderT <: TransportBinaryReader{<:Any, <:Any,
                                               CubedSphereBinaryGeometry}} =
    (driver.reader.header.geometry.Nc,
     driver.reader.header.geometry.Nc,
     driver.reader.header.nlevel)

@inline _runtime_has_surface(_context) = false
_runtime_has_surface(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) = MetDrivers.has_surface(reader)
_runtime_has_surface(driver::TransportBinaryDriver) =
    _runtime_has_surface(driver.reader)

@inline _runtime_has_gchp_vdiff(_context) = false
_runtime_has_gchp_vdiff(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) =
    MetDrivers.has_surface(reader) && MetDrivers.has_vdiff_fields(reader)
_runtime_has_gchp_vdiff(driver::TransportBinaryDriver) =
    _runtime_has_gchp_vdiff(driver.reader)

_runtime_has_gchp_nonlocal_vdiff(context) =
    _runtime_has_gchp_vdiff(context) && _runtime_has_pbl_eflux(context)
_runtime_has_pbl_eflux(_context) = false
_runtime_has_pbl_eflux(reader::TransportBinaryReader) = MetDrivers.has_pbl_eflux(reader)
_runtime_has_pbl_eflux(driver::TransportBinaryDriver) = _runtime_has_pbl_eflux(driver.reader)

_runtime_has_precomputed_dkg(_context) = false
_runtime_has_precomputed_dkg(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) =
    :dkg in reader.header.payload_sections
_runtime_has_precomputed_dkg(driver::TransportBinaryDriver) =
    _runtime_has_precomputed_dkg(driver.reader)

function build_runtime_convection(cfg, context)
    return build_runtime_convection(cfg, _runtime_recipe_style(context))
end

# Thin wrapper: parse the `[convection]` section into a typed `AbstractConvectionSpec`
# (validated, incl. the lmax_conv/n_merge-needs-use_collab_lu guard) once, then
# materialize the operator. Spec types + parser + `materialize` live in
# `RuntimePhysicsSpecs.jl`. `style` is threaded for API uniformity with
# build_runtime_advection/diffusion; convection materialization is
# topology-independent, so the `materialize` methods ignore it.
build_runtime_convection(cfg, style::AbstractRuntimeRecipeStyle) =
    materialize(convection_spec(_convection_section(cfg)), style)

@inline validate_runtime_advection(::AbstractRuntimeRecipeStyle,
                                   ::AbstractAdvectionScheme,
                                   _context) = nothing
@inline validate_runtime_diffusion(::AbstractRuntimeRecipeStyle,
                                   ::AbstractDiffusion,
                                   _context) = nothing
@inline validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                    ::NoConvection,
                                    _context) = nothing

function validate_runtime_advection(::AbstractStructuredRuntimeRecipeStyle,
                                    ::LinRoodPPMScheme,
                                    _context)
    throw(ArgumentError(
        "LinRoodPPMScheme is only supported on cubed-sphere runtimes."))
end

function validate_runtime_advection(::ReducedGaussianRuntimeRecipeStyle,
                                    scheme::Union{SlopesScheme, PPMScheme},
                                    _context)
    throw(ArgumentError(
        "$(nameof(typeof(scheme))) is not implemented for reduced-Gaussian runs; " *
        "use UpwindScheme or NoAdvection."))
end

function validate_runtime_diffusion(::ReducedGaussianRuntimeRecipeStyle,
                                    op::ImplicitVerticalDiffusion,
                                    _context)
    uses_diffusive_surface_flux_boundary(op) || return nothing
    throw(ArgumentError(
        "DiffusiveSurfaceFluxBoundary is not implemented for reduced-Gaussian runs; " *
        "use SplitSurfaceFluxCoupling."))
end

@inline _runtime_has_tm5_convection(_context) = false
@inline _runtime_has_cmfmc(_context) = false
@inline _runtime_has_tm5_convection(reader::TransportBinaryReader) = MetDrivers.has_tm5_convection(reader)
@inline _runtime_has_tm5_convection(driver::TransportBinaryDriver) = MetDrivers.has_tm5_convection(driver.reader)
@inline _runtime_has_cmfmc(reader::TransportBinaryReader) = MetDrivers.has_cmfmc(reader)
@inline _runtime_has_cmfmc(driver::TransportBinaryDriver) = MetDrivers.has_cmfmc(driver.reader)
@inline _runtime_has_cmfmc_cloud_base(_context) = false
@inline _runtime_has_cmfmc_cloud_base(reader::TransportBinaryReader) =
    MetDrivers.has_cmfmc_cloud_base(reader)
@inline _runtime_has_cmfmc_cloud_base(driver::TransportBinaryDriver) =
    _runtime_has_cmfmc_cloud_base(driver.reader)

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     ::TM5Convection,
                                     context)
    _runtime_has_tm5_convection(context) ||
        throw(ArgumentError(
            "[convection] kind = \"tm5\" requires TM5 convection sections " *
            "(`entu`, `detu`, `entd`, `detd`) in the runtime forcing source."))
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    ::ImplicitVerticalDiffusion{FT, <:WindowPBLKzField},
                                    context) where FT
    _runtime_has_surface(context) ||
        throw(ArgumentError(
            "[diffusion] kind = \"tm5_beljaars_viterbo_local_kz\" requires pblh/ustar/pbl_hflux/t2m sections " *
            "in every cubed-sphere transport binary."))
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    op::ImplicitVerticalDiffusion{FT, <:LocalHoltslagBovilleKzField},
                                    context) where FT
    _runtime_has_gchp_vdiff(context) ||
        throw(ArgumentError(
            "[diffusion] kind = \"geoschem_holtslag_boville_vdiff\" requires " *
            "pblh/ustar/pbl_hflux/t2m and vdiff_u/vdiff_v/vdiff_t/vdiff_qv " *
            "sections in every cubed-sphere transport binary."))
    # The supported GCHP-style placement adds emissions before one full
    # diffusion solve: S(dt) → V(dt). The historical policy name
    # `DiffusiveSurfaceFluxBoundary` describes that ordering; it does not add
    # a Neumann source term to the Thomas system. The default split policy is
    # V(dt/2) → S(dt) → V(dt/2), so warn when a GCHP-oriented Kz field is used
    # with the other ordering.
    if !(op.surface_flux_coupling isa DiffusiveSurfaceFluxBoundary)
        @warn """
        [diffusion] kind = "geoschem_holtslag_boville_vdiff" was selected but
        the surface-flux coupling is $(typeof(op.surface_flux_coupling)).
        For GCHP-style VDIFF placement, surface emissions must be added before
        one full diffusion solve: S(dt) -> V(dt). Set
        `surface_flux_boundary = true` to select that ordering. Despite its
        historical `DiffusiveSurfaceFluxBoundary` name, this policy does not
        insert a literal boundary term into the tridiagonal system. Exact
        discrete GCHP VDIFF parity remains to be validated end to end.
        """
    end
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    op::ImplicitVerticalDiffusion{FT, <:GCHPNonlocalPBLField},
                                    context) where FT
    _runtime_has_gchp_nonlocal_vdiff(context) || throw(ArgumentError(
        "[diffusion] kind = \"geoschem_nonlocal_vdiff\" requires the GCHP VDIFF sections " *
        "and pbl_eflux in every cubed-sphere transport binary."))
    op.surface_flux_coupling isa DiffusiveSurfaceFluxBoundary || throw(ArgumentError(
        "GEOS-Chem non-local VDIFF spreads fresh emissions before one full diffusion solve; " *
        "it needs DiffusiveSurfaceFluxBoundary coupling."))
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    ::ImplicitVerticalDiffusion{FT, <:PrecomputedCSDkgField},
                                    context) where FT
    _runtime_has_precomputed_dkg(context) || throw(ArgumentError(
        "TM5 precomputed diffusion requires a `:dkg` section in every cubed-sphere transport binary."))
    return nothing
end

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     op::CMFMCConvection,
                                     context)
    _runtime_has_cmfmc(context) ||
        throw(ArgumentError(
            "[convection] kind = \"cmfmc\" requires CMFMC convection forcing " *
            "in the runtime forcing source."))
    op.cloud_base isa ArchivedCloudBase && !_runtime_has_cmfmc_cloud_base(context) &&
        throw(ArgumentError(
            "[convection] cloud_base = \"dqrcu\" requires a cmfmc_cloud_base section in " *
            "the cubed-sphere transport binary."))
    return nothing
end

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     ::CMFMCMatrixConvection,
                                     context)
    # The matrix variant requires both cmfmc AND dtrain — see the
    # capability check in `DrivenRunner._validate_capability_match` for
    # the detailed reason. Recipe-level validators don't have access to
    # `caps.payload_sections` so we can only check the cmfmc capability
    # here; DrivenRunner enforces the dtrain requirement directly.
    _runtime_has_cmfmc(context) ||
        throw(ArgumentError(
            "[convection] kind = \"cmfmc_matrix\" requires CMFMC convection " *
            "forcing (cmfmc + dtrain) in the runtime forcing source. The " *
            "matrix variant reads the same binary sections as kind=\"cmfmc\" " *
            "and derives entu/detu at runtime — no Tiedtke fallback."))
    return nothing
end

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     op::AbstractConvection,
                                     _context)
    throw(ArgumentError(
        "Runtime recipe validation does not support convection operator $(typeof(op)) yet."))
end

function validate_runtime_halo_width(scheme::AbstractAdvectionScheme, halo_width::Integer)
    min_hp = required_halo_width(scheme)
    halo_width >= min_hp ||
        throw(ArgumentError(
            "[run] halo padding Hp=$(halo_width) is too small for $(typeof(scheme)); " *
            "need Hp >= $(min_hp)."))
    return nothing
end

@inline validate_runtime_combination(::AbstractRuntimeRecipeStyle,
                                     ::AbstractAdvectionScheme,
                                     ::AbstractDiffusion,
                                     ::AbstractConvection,
                                     _context) = nothing

"""
    validate_runtime_physics_recipe(recipe, context; halo_width=nothing)

Validate topology support, binary capabilities, and halo requirements for a
materialized RuntimePhysicsRecipe. Returns recipe or throws ArgumentError
before model allocation.
"""
function validate_runtime_physics_recipe(recipe::RuntimePhysicsRecipe,
                                         context;
                                         halo_width::Union{Nothing, Integer} = nothing)
    style = _runtime_recipe_style(context)
    validate_runtime_advection(style, recipe.advection, context)
    validate_runtime_diffusion(style, recipe.diffusion, context)
    validate_runtime_convection(style, recipe.convection, context)
    validate_runtime_combination(style,
                                 recipe.advection,
                                 recipe.diffusion,
                                 recipe.convection,
                                 context)
    halo_width === nothing || validate_runtime_halo_width(recipe.advection, halo_width)
    return recipe
end

"""
    build_runtime_physics_recipe(cfg, context, FT; halo_width=nothing)

Parse typed advection, diffusion, convection, and chemistry specifications
from cfg, materialize them for context and floating-point type FT, then run
the complete runtime compatibility validation.
"""
function build_runtime_physics_recipe(cfg,
                                      context,
                                      ::Type{FT};
                                      halo_width::Union{Nothing, Integer} = nothing) where FT
    recipe = RuntimePhysicsRecipe(
        build_runtime_advection(cfg, context),
        build_runtime_diffusion(cfg, context, FT),
        build_runtime_convection(cfg, context),
        build_runtime_chemistry(cfg, FT),
    )
    return validate_runtime_physics_recipe(recipe, context; halo_width = halo_width)
end

"""
    build_runtime_chemistry(cfg, ::Type{FT}) -> AbstractChemistryOperator

Read the optional `[chemistry]` TOML section and produce the
corresponding chemistry operator.

Supported `kind` values:

- `"none"` (default) — `NoChemistry()`.
- `"decay"` — `ExponentialDecay(FT; ...)`. Half-lives are read from
  the `half_lives_seconds` table:

      [chemistry]
      kind = "decay"
      [chemistry.half_lives_seconds]
      rn222 = 330350.4   # 3.8235 days

  The keyword name must match the corresponding `[tracers.<name>]`
  symbol that the run is carrying (case-insensitive — the builder
  symbolizes the key as-is and `ExponentialDecay.apply!` resolves
  it against `state.tracer_names` at call time).

Thin wrapper: parse the `[chemistry]` section into a typed
`AbstractChemistrySpec` once (validated), then materialize at run precision
`FT`. Spec types + parser live in `RuntimePhysicsSpecs.jl`.
"""
build_runtime_chemistry(cfg, ::Type{FT}) where FT =
    materialize(chemistry_spec(_chemistry_section(cfg)), FT)

function configured_halo_width(cfg, scheme::AbstractAdvectionScheme)
    run_cfg = get(cfg, "run", Dict{String,Any}())
    default_hp = required_halo_width(scheme)

    if haskey(run_cfg, "Hp") && haskey(run_cfg, "halo_padding")
        hp = Int(run_cfg["Hp"])
        halo_padding = Int(run_cfg["halo_padding"])
        hp == halo_padding || throw(ArgumentError(
            "[run] `Hp` ($(hp)) and `halo_padding` ($(halo_padding)) disagree; use one value."))
    end

    return haskey(run_cfg, "Hp") ? Int(run_cfg["Hp"]) :
           Int(get(run_cfg, "halo_padding", default_hp))
end

# CS tracers flow through the unified pipeline:
#
#     vmr = build_initial_mixing_ratio(air_mass, grid, init_cfg)
#     rm  = pack_initial_tracer_mass(grid, air_mass, vmr;
#                                    mass_basis = DryBasis())
#
# See `src/Models/InitialConditionIO.jl`.

export RuntimePhysicsRecipe
export build_runtime_advection, build_runtime_diffusion, build_runtime_convection
export build_runtime_chemistry
export build_runtime_physics_recipe, validate_runtime_physics_recipe
export configured_halo_width
