#!/usr/bin/env julia
# Smoke test for the canonical runtime operator builders. Exercises the
# TOML→operator path without requiring a real CS binary (that's what the
# full end-to-end test in test_cubed_sphere_runtime.jl covers).

using Test

using AtmosTransport
using .AtmosTransport.Models:
    build_runtime_advection, build_runtime_diffusion, build_runtime_convection,
    build_runtime_physics_recipe, validate_runtime_physics_recipe,
    configured_halo_width, CubedSphereRuntimeRecipeStyle, LatLonRuntimeRecipeStyle,
    convection_spec, TM5ConvectionSpec, CMFMCMatrixConvectionSpec,
    advection_spec, UpwindAdvectionSpec, SlopesAdvectionSpec, PPMAdvectionSpec,
    NoAdvectionSpec, LinRoodAdvectionSpec,
    ForwardAdvectionContext, InversionAdvectionContext,
    diffusion_spec, NoDiffusionSpec, ConstantDiffusionSpec,
    WindowPBLKzDiffusionSpec, HoltslagBovilleVdiffDiffusionSpec,
    TM5DkgDiffusionSpec, materialize
using .AtmosTransport.State.Fields:
    CubedSphereField,
    PrecomputedCSDkgField, field_value, panel_field
using .AtmosTransport.Operators.Diffusion:
    WindowPBLKzField, LocalHoltslagBovilleKzField, uses_diffusive_surface_flux_boundary
using .AtmosTransport.Operators: MonotoneLimiter, SameAsHorizontal, FV3ScalarProfile,
    NoLimiter, CW84Limiter

const CS_STYLE = CubedSphereRuntimeRecipeStyle()
cs_advection(cfg) = build_runtime_advection(cfg, CS_STYLE)
cs_diffusion(cfg, FT) = build_runtime_diffusion(cfg, CS_STYLE, FT)
cs_convection(cfg) = build_runtime_convection(cfg, CS_STYLE)
cs_physics_recipe(cfg, context, FT; kwargs...) =
    build_runtime_physics_recipe(cfg, context, FT; kwargs...)
cs_halo_width(cfg, scheme) = configured_halo_width(cfg, scheme)

struct UnknownBasisReader end
AtmosTransport.MetDrivers.binary_mass_basis(::UnknownBasisReader) = :unknown

@testset "structured model rejects an unknown driver basis before loading data" begin
    grid = AtmosGrid(LatLonMesh(; Nx = 4, Ny = 2),
                     HybridSigmaPressure([0.0, 0.0], [0.0, 1.0]), CPU())
    reader = UnknownBasisReader()
    driver = AtmosTransport.TransportBinaryDriver{Float64, typeof(reader), typeof(grid)}(reader, grid)
    @test_throws ArgumentError AtmosTransport.Models.DrivenRunner._make_structured_model(
        driver; FT = Float64, recipe = nothing, tracer_specs = (), arch = CPU())
end

@testset "runtime physics rejects values that change physics silently" begin
    @test_throws ArgumentError convection_spec(Dict("cloud_base" => "dqrcu"))
    for kind in ("tm5", "cmfmc_matrix")
        @test_throws ArgumentError convection_spec(Dict(
            "kind" => kind, "use_collab_lu" => true, "lmax_conv" => -1))
        @test_throws ArgumentError convection_spec(Dict(
            "kind" => kind, "tile_workspace_gib" => true))
    end
    @test_throws ArgumentError AtmosTransport.Models.diffusion_spec(
        Dict("kind" => "constant", "value" => true))
    for key in ("Hp", "halo_padding"), value in (true, 2.0, 2.5, "3", 0, -1)
        @test_throws ArgumentError configured_halo_width(
            Dict("run" => Dict(key => value)), UpwindScheme())
    end
    @test configured_halo_width(Dict(), NoAdvection()) == 0
    @test configured_halo_width(Dict("run" => Dict("Hp" => 3)), UpwindScheme()) == 3
end

struct StubReader
    has_cmfmc :: Bool
    has_tm5   :: Bool
end
AtmosTransport.MetDrivers.has_cmfmc(r::StubReader) = r.has_cmfmc
AtmosTransport.MetDrivers.has_tm5_convection(r::StubReader) = r.has_tm5
AtmosTransport.Models.Physics._runtime_recipe_style(::StubReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models.Physics._runtime_has_tm5_convection(r::StubReader) = r.has_tm5
AtmosTransport.Models.Physics._runtime_has_cmfmc(r::StubReader) = r.has_cmfmc
AtmosTransport.Models.Physics._runtime_has_dtrain(r::StubReader) = r.has_cmfmc
AtmosTransport.Models.Physics._runtime_has_surface(::StubReader) = false

struct StubMatrixReader
    has_cmfmc :: Bool
    has_dtrain :: Bool
end
AtmosTransport.Models.Physics._runtime_recipe_style(::StubMatrixReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models.Physics._runtime_has_cmfmc(r::StubMatrixReader) = r.has_cmfmc
AtmosTransport.Models.Physics._runtime_has_dtrain(r::StubMatrixReader) = r.has_dtrain

struct StubPBLReader end
AtmosTransport.Models.Physics._runtime_recipe_style(::StubPBLReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models.Physics._runtime_has_tm5_convection(::StubPBLReader) = false
AtmosTransport.Models.Physics._runtime_has_cmfmc(::StubPBLReader) = false
AtmosTransport.Models.Physics._runtime_has_surface(::StubPBLReader) = true
AtmosTransport.Models.Physics._pbl_cache_shape(::StubPBLReader) = (4, 4, 2)

struct StubGCHPVDIFFReader end
AtmosTransport.Models.Physics._runtime_recipe_style(::StubGCHPVDIFFReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models.Physics._runtime_has_tm5_convection(::StubGCHPVDIFFReader) = false
AtmosTransport.Models.Physics._runtime_has_cmfmc(::StubGCHPVDIFFReader) = false
AtmosTransport.Models.Physics._runtime_has_surface(::StubGCHPVDIFFReader) = true
AtmosTransport.Models.Physics._runtime_has_gchp_vdiff(::StubGCHPVDIFFReader) = true
AtmosTransport.Models.Physics._pbl_cache_shape(::StubGCHPVDIFFReader) = (4, 4, 2)

struct StubDkgReader end
AtmosTransport.Models.Physics._runtime_recipe_style(::StubDkgReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models.Physics._runtime_has_tm5_convection(::StubDkgReader) = false
AtmosTransport.Models.Physics._runtime_has_cmfmc(::StubDkgReader) = false
AtmosTransport.Models.Physics._runtime_has_surface(::StubDkgReader) = false
AtmosTransport.Models.Physics._runtime_has_precomputed_dkg(::StubDkgReader) = true
AtmosTransport.Models.Physics._pbl_cache_shape(::StubDkgReader) = (4, 4, 2)

struct StubStructuredReader
    has_tm5 :: Bool
end
AtmosTransport.Models.Physics._runtime_recipe_style(::StubStructuredReader) =
    AtmosTransport.Models.LatLonRuntimeRecipeStyle()
AtmosTransport.Models.Physics._runtime_has_tm5_convection(r::StubStructuredReader) = r.has_tm5
AtmosTransport.Models.Physics._runtime_has_cmfmc(::StubStructuredReader) = false

@testset "runtime operator builders" begin

    latlon_grid = AtmosGrid(
        LatLonMesh(; FT = Float64, Nx = 2, Ny = 2),
        HybridSigmaPressure(Float64[0, 1000], Float64[0, 1]),
        CPU();
        FT = Float64,
    )
    reduced_grid = AtmosGrid(
        ReducedGaussianMesh(Float64[-45, 45], [4, 4]; FT = Float64),
        HybridSigmaPressure(Float64[0, 1000], Float64[0, 1]),
        CPU();
        FT = Float64,
    )

    @testset "build_runtime_advection dispatch" begin
        @test build_runtime_advection(Dict("advection" => Dict("scheme" => "upwind")), latlon_grid) isa UpwindScheme
        @test build_runtime_advection(Dict("advection" => Dict("scheme" => "slopes")), latlon_grid) isa SlopesScheme
        @test build_runtime_advection(Dict("advection" => Dict("scheme" => "ppm")), latlon_grid) isa PPMScheme
        @test build_runtime_advection(Dict("advection" => Dict("scheme" => "upwind")), reduced_grid) isa UpwindScheme
        @test_throws ArgumentError build_runtime_advection(
            Dict("advection" => Dict("scheme" => "slopes")), reduced_grid)
        @test_throws ArgumentError build_runtime_advection(
            Dict("advection" => Dict("scheme" => "ppm")), reduced_grid)
        @test_throws ArgumentError build_runtime_advection(
            Dict("advection" => Dict("scheme" => "linrood")), latlon_grid)
    end

    @testset "cs_advection dispatch" begin
        @test cs_advection(Dict("advection" => Dict("scheme" => "slopes"))) isa SlopesScheme
        @test cs_advection(Dict("advection" => Dict("scheme" => "ppm"))) isa PPMScheme
        @test cs_advection(Dict("advection" => Dict("scheme" => "linrood"))) isa LinRoodPPMScheme
        @test cs_advection(Dict("advection" => Dict("scheme" => "linrood", "ppm_order" => 7))) isa LinRoodPPMScheme
        @test cs_advection(Dict{String,Any}()) isa UpwindScheme
        @test_throws ArgumentError cs_advection(
            Dict("advection" => Dict("scheme" => "ppm", "ppm_order" => 7)))
        @test_throws ArgumentError cs_advection(Dict("advection" => Dict("scheme" => "xyz")))
        @test_throws ArgumentError cs_advection(
            Dict("run" => Dict("scheme" => "linrood"),
                 "advection" => Dict("ppm_order" => 7)))      # moved, also next to [advection]
    end

    @testset "AdvectionSpec parse + materialize" begin
        @test advection_spec(Dict("scheme" => "upwind")) isa UpwindAdvectionSpec
        @test advection_spec(Dict("scheme" => "slopes")) isa SlopesAdvectionSpec
        @test advection_spec(Dict("scheme" => "ppm"))    isa PPMAdvectionSpec
        @test advection_spec(Dict("scheme" => "none"))   isa NoAdvectionSpec
        @test advection_spec(Dict{String,Any}())         isa UpwindAdvectionSpec  # default
        # LinRood carries its reconstruction order in a typed spec.
        @test advection_spec(Dict("scheme" => "linrood")).order == 5  # default order
        @test advection_spec(Dict("scheme" => "linrood", "ppm_order" => 7)).order == 7
        # parse-time validation matches the old builder.
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "ppm_order" => 7))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "xyz"))
        # LinRood materializes only on cubed-sphere; structured throws.
        @test_throws ArgumentError advection_spec(Dict("scheme" => "linrood_ppm"))
        # PPM vertical reconstruction: default, FV3 kord = 8 (cubed sphere only).
        @test advection_spec(Dict("scheme" => "ppm")).vertical isa SameAsHorizontal
        fv3 = advection_spec(Dict("scheme" => "ppm", "vertical" => "fv3_kord8"))
        @test fv3.vertical isa FV3ScalarProfile
        @test materialize(fv3, CS_STYLE) isa PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}
        @test materialize(advection_spec(Dict("scheme" => "ppm")), CS_STYLE) == PPMScheme()
        @test_throws ArgumentError materialize(fv3, LatLonRuntimeRecipeStyle())
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "vertical" => "kord9"))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "slopes", "vertical" => "fv3_kord8"))
        @test advection_spec(Dict("scheme" => "ppm", "vertical" => "fv3_kord8_signed")).vertical ==
              FV3ScalarProfile(; positive_definite = false)
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "vertical" => 8))
        @test cs_advection(Dict("advection" => Dict("scheme" => "ppm", "vertical" => "fv3_kord8"))) isa
              PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}
        @test_throws ArgumentError build_runtime_advection(
            Dict("advection" => Dict("scheme" => "ppm", "vertical" => "fv3_kord8_signed")),
            LatLonRuntimeRecipeStyle())
        # Lin-Rood vertical sweeps: the GCHP-matched FV3 profile by default;
        # explicit upwind remains available. The programmatic one-argument
        # constructor stays upwind; only configured forward runs default FV3.
        @test LinRoodAdvectionSpec(5).vertical isa UpwindScheme
        lr_default = advection_spec(Dict("scheme" => "linrood"))
        @test lr_default.vertical isa PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}
        @test materialize(lr_default, CS_STYLE) isa
              LinRoodPPMScheme{5, PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}}
        @test advection_spec(Dict("scheme" => "linrood", "vertical" => "upwind")).vertical isa
              UpwindScheme
        lr = cs_advection(Dict("advection" => Dict("scheme" => "linrood", "ppm_order" => 7,
                                                    "vertical" => "fv3_kord8")))
        @test lr isa LinRoodPPMScheme{7, PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}}
        @test_throws ArgumentError advection_spec(Dict("scheme" => "linrood",
                                                       "vertical" => "same_as_horizontal"))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "slopes", "vertical" => "upwind"))
    end

    @testset "AdvectionSpec context: forward run vs inversion driver" begin
        # Forward runs (the default context) keep their limiter vocabulary and
        # upwind default; the unlimited PPM is accepted only by the inversion.
        forward, inversion = ForwardAdvectionContext(), InversionAdvectionContext()
        @test advection_spec(Dict{String,Any}(), forward) isa UpwindAdvectionSpec
        @test advection_spec(Dict("scheme" => "ppm", "limiter" => "cw84"), forward).limiter isa CW84Limiter
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "limiter" => "none"))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "limiter" => "none"), forward)
        # The inversion driver: same parser, `none` accepted, `scheme` required.
        @test advection_spec(Dict("scheme" => "ppm", "limiter" => "none"), inversion).limiter isa NoLimiter
        @test materialize(advection_spec(Dict("scheme" => "ppm", "limiter" => "none"), inversion),
                          CS_STYLE) == PPMScheme(NoLimiter())
        @test advection_spec(Dict("scheme" => "ppm"), inversion).limiter isa MonotoneLimiter
        # Lin–Rood's vertical default is the same in both contexts (OF-94).
        @test advection_spec(Dict("scheme" => "linrood"), inversion).vertical ==
              advection_spec(Dict("scheme" => "linrood"), forward).vertical == PPMScheme(; vertical = FV3ScalarProfile())
        @test_throws ArgumentError advection_spec(Dict{String,Any}(), inversion)
        @test_throws ArgumentError advection_spec(Dict("scheme" => "upwind", "limiter" => "none"), inversion)
        @test_throws ArgumentError advection_spec(Dict("scheme" => "ppm", "limiter" => "nope"), inversion)
    end


    @testset "cs_halo_width dispatch" begin
        @test cs_halo_width(Dict{String,Any}(), UpwindScheme()) == 1
        @test cs_halo_width(Dict("advection" => Dict("scheme" => "ppm")), PPMScheme()) == 3
        @test cs_halo_width(Dict("run" => Dict("halo_padding" => 5)), SlopesScheme()) == 5
        @test cs_halo_width(Dict("run" => Dict("Hp" => 4)), LinRoodPPMScheme()) == 4
        @test_throws ArgumentError cs_halo_width(
            Dict("run" => Dict("Hp" => 3, "halo_padding" => 4)), UpwindScheme())
    end

    @testset "cs_diffusion dispatch" begin
        # default (no section) → NoDiffusion
        @test cs_diffusion(Dict{String,Any}(), Float64) isa NoDiffusion

        # kind = "none" → NoDiffusion
        @test cs_diffusion(Dict("diffusion" => Dict("kind" => "none")), Float64) isa NoDiffusion

        # kind = "constant" → ImplicitVerticalDiffusion
        op = cs_diffusion(Dict("diffusion" => Dict("kind" => "constant",
                                                          "value" => 2.5)), Float64)
        @test op isa ImplicitVerticalDiffusion
        # kz_field is a CubedSphereField wrapping 6 ConstantField
        kz = op.kz_field
        @test kz isa CubedSphereField
        # Check the per-panel field types round-trip the value.
        @test all(field_value(panel_field(kz, p), (1, 1, 1)) == 2.5 for p in 1:6)

        # F32 propagates to the Kz value
        op32 = cs_diffusion(Dict("diffusion" => Dict("kind" => "constant",
                                                            "value" => 1.0)), Float32)
        @test op32 isa ImplicitVerticalDiffusion
        @test eltype(field_value(panel_field(op32.kz_field, 1), (1, 1, 1))) === Float32 ||
              field_value(panel_field(op32.kz_field, 1), (1, 1, 1)) isa Float32

        # Unknown kind → error
        @test_throws ArgumentError cs_diffusion(
            Dict("diffusion" => Dict("kind" => "magic")), Float64)

        # Local TM5 Kz needs a reader/driver context with raw surface sections.
        pbl_recipe = cs_physics_recipe(
            Dict("diffusion" => Dict("kind" => "tm5_beljaars_viterbo_local_kz")),
            StubPBLReader(),
            Float64,
        )
        @test pbl_recipe.diffusion isa ImplicitVerticalDiffusion
        @test pbl_recipe.diffusion.kz_field isa WindowPBLKzField
        @test !uses_diffusive_surface_flux_boundary(pbl_recipe.diffusion)

        named_pbl_recipe = cs_physics_recipe(
            Dict("diffusion" => Dict("kind" => "tm5_beljaars_viterbo_local_kz",
                                     "surface_flux_boundary" => true)),
            StubPBLReader(),
            Float64,
        )
        @test named_pbl_recipe.diffusion isa ImplicitVerticalDiffusion
        @test named_pbl_recipe.diffusion.kz_field isa WindowPBLKzField
        @test uses_diffusive_surface_flux_boundary(named_pbl_recipe.diffusion)

        @test_throws ArgumentError cs_physics_recipe(
            Dict("diffusion" => Dict("kind" => "tm5_beljaars_viterbo_local_kz")), StubReader(false, false), Float64)
        gchp_vdiff_recipe = cs_physics_recipe(
            Dict("diffusion" => Dict("kind" => "geoschem_holtslag_boville_vdiff")),
            StubGCHPVDIFFReader(),
            Float64,
        )
        @test gchp_vdiff_recipe.diffusion isa ImplicitVerticalDiffusion
        @test gchp_vdiff_recipe.diffusion.kz_field isa LocalHoltslagBovilleKzField
        @test_throws ArgumentError cs_physics_recipe(
            Dict("diffusion" => Dict("kind" => "geoschem_holtslag_boville_vdiff")),
            StubPBLReader(),
            Float64,
        )

        # Exact TM5 interface exchange requires a :dkg payload.
        dkg_recipe = cs_physics_recipe(
            Dict("diffusion" => Dict("kind" => "tm5_dkg")),
            StubDkgReader(),
            Float64,
        )
        @test dkg_recipe.diffusion isa ImplicitVerticalDiffusion
        @test dkg_recipe.diffusion.kz_field isa PrecomputedCSDkgField
        @test_throws ArgumentError cs_physics_recipe(
            Dict("diffusion" => Dict("kind" => "tm5_dkg")),
            StubPBLReader(),
            Float64,
        )

        # Section B (codex) P0: legacy `type = "..."` schema must NOT
        # silently fall through to NoDiffusion. It must error with a
        # migration hint so old configs that expected diffusion are caught.
        @test_throws ArgumentError cs_diffusion(
            Dict("diffusion" => Dict("type" => "pbl")), Float64)
        @test_throws ArgumentError cs_diffusion(
            Dict("diffusion" => Dict("type" => "nonlocal_pbl")), Float64)

        # `[diffusion]` present but neither `type` nor `kind` → error.
        @test_throws ArgumentError cs_diffusion(
            Dict("diffusion" => Dict("value" => 1.0)), Float64)
    end

    @testset "build_runtime_diffusion chooses layout-aware field rank" begin
        op_ll = build_runtime_diffusion(
            Dict("diffusion" => Dict("kind" => "constant", "value" => 2.5)),
            latlon_grid,
            Float64)
        @test op_ll isa ImplicitVerticalDiffusion
        @test field_value(op_ll.kz_field, (1, 1, 1)) == 2.5

        op_rg = build_runtime_diffusion(
            Dict("diffusion" => Dict("kind" => "constant", "value" => 1.5)),
            reduced_grid,
            Float64)
        @test op_rg isa ImplicitVerticalDiffusion
        @test field_value(op_rg.kz_field, (1, 1)) == 1.5

        # RG's face-indexed transport currently implements only midpoint
        # surface-flux splitting. Reject the other policy while materializing
        # the recipe instead of silently ignoring it during the first step.
        @test_throws ArgumentError build_runtime_physics_recipe(
            Dict("diffusion" => Dict(
                "kind" => "constant",
                "value" => 1.5,
                "surface_flux_boundary" => true,
            )),
            reduced_grid,
            Float64,
        )
    end

    @testset "DiffusionSpec parse + materialize" begin
        @test diffusion_spec(Dict{String,Any}())             isa NoDiffusionSpec  # empty
        @test diffusion_spec(Dict("kind" => "none"))         isa NoDiffusionSpec
        @test diffusion_spec(Dict("kind" => "constant", "value" => 2.5)) isa ConstantDiffusionSpec
        @test diffusion_spec(Dict("kind" => "constant", "value" => 2.5)).value == 2.5
        # Cubed-sphere diffusion closures.
        @test diffusion_spec(Dict("kind" => "tm5_beljaars_viterbo_local_kz")) isa WindowPBLKzDiffusionSpec
        @test diffusion_spec(Dict("kind" => "geoschem_holtslag_boville_vdiff")) isa HoltslagBovilleVdiffDiffusionSpec
        @test diffusion_spec(Dict("kind" => "tm5_dkg")) isa TM5DkgDiffusionSpec
        # surface_flux_boundary flows onto the spec.
        @test diffusion_spec(Dict("kind" => "tm5_beljaars_viterbo_local_kz", "surface_flux_boundary" => true)).surface_flux_boundary
        @test !diffusion_spec(Dict("kind" => "tm5_beljaars_viterbo_local_kz")).surface_flux_boundary
        # parse-time validation (matches the old builder).
        @test_throws ArgumentError diffusion_spec(Dict("kind" => "magic"))
        @test_throws ArgumentError diffusion_spec(Dict("type" => "pbl"))         # legacy schema
        @test_throws ArgumentError diffusion_spec(Dict("value" => 1.0))         # present, no kind
        @test_throws ArgumentError diffusion_spec(
            Dict("kind" => "constant", "surface_flux_boundary" => "yes"))       # non-bool
        # CS-only closures throw on structured styles at materialize time.
        @test_throws ArgumentError materialize(
            diffusion_spec(Dict("kind" => "tm5_beljaars_viterbo_local_kz")),
            AtmosTransport.Models.LatLonRuntimeRecipeStyle(), Float64, nothing)
    end

    @testset "cs_convection + recipe validation" begin
        no_conv   = StubReader(false, false)
        only_tm5  = StubReader(false, true)
        only_cmfmc = StubReader(true, false)
        full_conv = StubReader(true, true)

        # default (no section) → NoConvection
        @test cs_convection(Dict{String,Any}()) isa NoConvection

        # kind = "none" → NoConvection
        @test cs_convection(Dict("convection" => Dict("kind" => "none"))) isa NoConvection

        @test cs_convection(Dict("convection" => Dict("kind" => "tm5"))) isa TM5Convection
        @test cs_convection(Dict("convection" => Dict("kind" => "cmfmc"))) isa CMFMCConvection

        @test cs_physics_recipe(Dict("convection" => Dict("kind" => "tm5")), only_tm5, Float64).convection isa TM5Convection
        @test cs_physics_recipe(Dict("convection" => Dict("kind" => "cmfmc")), only_cmfmc, Float64).convection isa CMFMCConvection
        @test cs_physics_recipe(Dict("convection" => Dict("kind" => "cmfmc")), full_conv, Float64).convection isa CMFMCConvection
        matrix_cfg = Dict("convection" => Dict("kind" => "cmfmc_matrix"))
        matrix_recipe = cs_physics_recipe(
            matrix_cfg, StubMatrixReader(true, true), Float64)
        @test matrix_recipe.convection isa CMFMCMatrixConvection
        # The runner reuses the first binary's recipe and validates each later
        # driver. A later cmfmc binary without dtrain must still be rejected.
        @test_throws ArgumentError validate_runtime_physics_recipe(
            matrix_recipe, StubMatrixReader(true, false))
        @test_throws ArgumentError cs_physics_recipe(
            matrix_cfg, StubMatrixReader(true, false), Float64)

        @test_throws ArgumentError cs_physics_recipe(
            Dict("convection" => Dict("kind" => "tm5")), no_conv, Float64)
        @test_throws ArgumentError cs_physics_recipe(
            Dict("convection" => Dict("kind" => "cmfmc")), no_conv, Float64)
        @test_throws ArgumentError cs_convection(
            Dict("convection" => Dict("kind" => "ras")))
    end

    @testset "ConvectionSpec footgun + parse-time validation" begin
        # lmax_conv / n_merge only take effect with use_collab_lu=true; setting them
        # without it was a silent no-op and is now a hard error (the live footgun).
        @test_throws ArgumentError convection_spec(Dict("kind" => "tm5", "lmax_conv" => 75))
        @test_throws ArgumentError convection_spec(Dict("kind" => "tm5", "n_merge" => 3))
        @test_throws ArgumentError convection_spec(
            Dict("kind" => "cmfmc_matrix", "lmax_conv" => 75))
        @test_throws ArgumentError cs_convection(
            Dict("convection" => Dict("kind" => "tm5", "n_merge" => 3)))
        # n_merge = 2 is a valid merge with collab on: the historical mass
        # blow-up was a clipping bug (fixed), not n_merge=2 itself, so it is now
        # the production value rather than a parse-time error.
        s2 = convection_spec(Dict("kind" => "tm5", "use_collab_lu" => true, "n_merge" => 2))
        @test s2 isa TM5ConvectionSpec && s2.n_merge == 2 && s2.use_collab_lu
        # unknown kind throws at the parser.
        @test_throws ArgumentError convection_spec(Dict("kind" => "ras"))

        # Happy path: collab on → spec + operator carry the knobs (parse parity).
        s = convection_spec(Dict("kind" => "tm5", "use_collab_lu" => true,
                                 "lmax_conv" => 75, "n_merge" => 3))
        @test s isa TM5ConvectionSpec
        @test s.use_collab_lu && s.lmax_conv == 75 && s.n_merge == 3
        op = cs_convection(Dict("convection" => Dict(
            "kind" => "tm5", "use_collab_lu" => true, "lmax_conv" => 75, "n_merge" => 3)))
        @test op isa TM5Convection && op.lmax_conv == 75 && op.n_merge == 3 && op.use_collab_lu
        # cmfmc_matrix path materializes to the matrix operator with the knobs.
        @test convection_spec(Dict("kind" => "cmfmc_matrix", "use_collab_lu" => true,
                                   "lmax_conv" => 75, "n_merge" => 3)) isa CMFMCMatrixConvectionSpec
        @test cs_convection(Dict("convection" => Dict("kind" => "cmfmc_matrix",
            "use_collab_lu" => true, "lmax_conv" => 75, "n_merge" => 3))) isa CMFMCMatrixConvection
    end

    @testset "build_runtime_physics_recipe validates structured convection capabilities" begin
        tm5_reader = StubStructuredReader(true)
        dry_reader = StubStructuredReader(false)

        @test build_runtime_physics_recipe(
            Dict("convection" => Dict("kind" => "tm5")), tm5_reader, Float64).convection isa TM5Convection

        @test_throws ArgumentError build_runtime_physics_recipe(
            Dict("convection" => Dict("kind" => "tm5")), dry_reader, Float64)

        @test_throws ArgumentError build_runtime_physics_recipe(
            Dict("convection" => Dict("kind" => "cmfmc")), tm5_reader, Float64)
    end

    @testset "cs_physics_recipe validates halo width" begin
        reader = StubReader(false, false)

        recipe = cs_physics_recipe(
            Dict("advection" => Dict("scheme" => "linrood")), reader, Float64; halo_width = 3)
        @test recipe.advection isa LinRoodPPMScheme

        @test_throws ArgumentError cs_physics_recipe(
            Dict("advection" => Dict("scheme" => "linrood")), reader, Float64; halo_width = 2)
    end


end

@testset "NoAdvection runtime builder on every topology" begin
    M = AtmosTransport.Models
    cfg = Dict("advection" => Dict("scheme" => "none"))
    for style in (M.LatLonRuntimeRecipeStyle(), M.ReducedGaussianRuntimeRecipeStyle(),
                  M.CubedSphereRuntimeRecipeStyle())
        @test M.build_runtime_advection(cfg, style) isa NoAdvection
    end
end
