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
    diffusion_spec, NoDiffusionSpec, ConstantDiffusionSpec,
    WindowPBLKzDiffusionSpec, HoltslagBovilleVdiffDiffusionSpec,
    TM5DkgDiffusionSpec, materialize
using .AtmosTransport.State.Fields:
    CubedSphereField, WindowPBLKzField, LocalHoltslagBovilleKzField,
    PrecomputedCSDkgField, field_value, panel_field
using .AtmosTransport.Operators.Diffusion:
    uses_diffusive_surface_flux_boundary
using .AtmosTransport.Operators: MonotoneLimiter, SameAsHorizontal, FV3ScalarProfile

const CS_STYLE = CubedSphereRuntimeRecipeStyle()
cs_advection(cfg) = build_runtime_advection(cfg, CS_STYLE)
cs_diffusion(cfg, FT) = build_runtime_diffusion(cfg, CS_STYLE, FT)
cs_convection(cfg) = build_runtime_convection(cfg, CS_STYLE)
cs_physics_recipe(cfg, context, FT; kwargs...) =
    build_runtime_physics_recipe(cfg, context, FT; kwargs...)
cs_halo_width(cfg, scheme) = configured_halo_width(cfg, scheme)

struct StubReader
    has_cmfmc :: Bool
    has_tm5   :: Bool
end
AtmosTransport.MetDrivers.has_cmfmc(r::StubReader) = r.has_cmfmc
AtmosTransport.MetDrivers.has_tm5_convection(r::StubReader) = r.has_tm5
AtmosTransport.Models._runtime_recipe_style(::StubReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models._runtime_has_tm5_convection(r::StubReader) = r.has_tm5
AtmosTransport.Models._runtime_has_cmfmc(r::StubReader) = r.has_cmfmc
AtmosTransport.Models._runtime_has_surface(::StubReader) = false

struct StubPBLReader end
AtmosTransport.Models._runtime_recipe_style(::StubPBLReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models._runtime_has_tm5_convection(::StubPBLReader) = false
AtmosTransport.Models._runtime_has_cmfmc(::StubPBLReader) = false
AtmosTransport.Models._runtime_has_surface(::StubPBLReader) = true
AtmosTransport.Models._pbl_cache_shape(::StubPBLReader) = (4, 4, 2)

struct StubGCHPVDIFFReader end
AtmosTransport.Models._runtime_recipe_style(::StubGCHPVDIFFReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models._runtime_has_tm5_convection(::StubGCHPVDIFFReader) = false
AtmosTransport.Models._runtime_has_cmfmc(::StubGCHPVDIFFReader) = false
AtmosTransport.Models._runtime_has_surface(::StubGCHPVDIFFReader) = true
AtmosTransport.Models._runtime_has_gchp_vdiff(::StubGCHPVDIFFReader) = true
AtmosTransport.Models._pbl_cache_shape(::StubGCHPVDIFFReader) = (4, 4, 2)

struct StubDkgReader end
AtmosTransport.Models._runtime_recipe_style(::StubDkgReader) =
    AtmosTransport.Models.CubedSphereRuntimeRecipeStyle()
AtmosTransport.Models._runtime_has_tm5_convection(::StubDkgReader) = false
AtmosTransport.Models._runtime_has_cmfmc(::StubDkgReader) = false
AtmosTransport.Models._runtime_has_surface(::StubDkgReader) = false
AtmosTransport.Models._runtime_has_precomputed_dkg(::StubDkgReader) = true
AtmosTransport.Models._pbl_cache_shape(::StubDkgReader) = (4, 4, 2)

struct StubStructuredReader
    has_tm5 :: Bool
end
AtmosTransport.Models._runtime_recipe_style(::StubStructuredReader) =
    AtmosTransport.Models.LatLonRuntimeRecipeStyle()
AtmosTransport.Models._runtime_has_tm5_convection(r::StubStructuredReader) = r.has_tm5
AtmosTransport.Models._runtime_has_cmfmc(::StubStructuredReader) = false

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
        @test cs_advection(Dict("run" => Dict("scheme" => "upwind"))) isa UpwindScheme
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
                 "advection" => Dict("ppm_order" => 7)))
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
        # `[run].vertical` next to an `[advection]` table is ambiguous
        @test_throws ArgumentError cs_advection(Dict("run" => Dict("vertical" => "fv3_kord8"),
                                                     "advection" => Dict("scheme" => "ppm")))
        @test cs_advection(Dict("advection" => Dict("scheme" => "ppm", "vertical" => "fv3_kord8"))) isa
              PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}
        @test_throws ArgumentError build_runtime_advection(
            Dict("advection" => Dict("scheme" => "ppm", "vertical" => "fv3_kord8_signed")),
            LatLonRuntimeRecipeStyle())
        # Lin-Rood vertical sweeps: upwind (default) or the FV3 profile
        @test advection_spec(Dict("scheme" => "linrood")).vertical isa UpwindScheme
        lr = cs_advection(Dict("advection" => Dict("scheme" => "linrood", "ppm_order" => 7,
                                                    "vertical" => "fv3_kord8")))
        @test lr isa LinRoodPPMScheme{7, PPMScheme{MonotoneLimiter, FV3ScalarProfile{true}}}
        @test_throws ArgumentError advection_spec(Dict("scheme" => "linrood",
                                                       "vertical" => "same_as_horizontal"))
        @test_throws ArgumentError advection_spec(Dict("scheme" => "slopes", "vertical" => "upwind"))
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

    # Plan 40 Commit 2 removed `build_cs_tracer_panels` (it was a flat-411
    # stub). CS tracers now flow through the unified pipeline:
    # `build_initial_mixing_ratio` + `pack_initial_tracer_mass`. Those are
    # tested in detail in `test/test_initial_condition_io.jl` (plan 40
    # Commits 1b–1d). The canonical runner wires exactly that pipeline
    # through from TOML.
    @testset "CS tracer IC flows through unified pipeline" begin
        FT = Float64
        Nc = 4
        Hp = 1
        Nz = 3
        mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
        vertical = HybridSigmaPressure(FT[0, 50000, 0], FT[1, 0.5, 0])
        grid = AtmosGrid(mesh, vertical, CPU(); FT = FT)
        air_mass = ntuple(_ -> fill(FT(1e9), Nc + 2Hp, Nc + 2Hp, Nz), 6)

        # uniform IC path
        vmr = build_initial_mixing_ratio(air_mass,
                                         grid,
                                         Dict("kind" => "uniform",
                                              "background" => 1e-6))
        rm  = pack_initial_tracer_mass(grid, air_mass, vmr;
                                       mass_basis = DryBasis())
        @test length(rm) == 6
        @test size(rm[1]) == size(air_mass[1])
        # Interior should be vmr * air_mass = 1e-6 * 1e9 = 1e3
        for p in 1:6
            interior = @view rm[p][Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :]
            @test all(interior .≈ 1e3)
            # Halo stays zero (populated at runtime by halo exchanges)
            @test rm[p][1, 1, 1] == zero(FT)
        end

        # zero-filled fossil placeholder
        vmr_zero = build_initial_mixing_ratio(air_mass, grid,
                                              Dict("kind" => "uniform",
                                                   "background" => 0.0))
        rm_zero  = pack_initial_tracer_mass(grid, air_mass, vmr_zero;
                                            mass_basis = DryBasis())
        @test all(iszero, rm_zero[1])

        # CS gaussian blob path mirrors the quickstart configs and returns
        # halo-free panel interiors for packing.
        vmr_blob = build_initial_mixing_ratio(
            air_mass, grid,
            Dict("kind" => "gaussian_blob",
                 "background" => 4.0e-4,
                 "amplitude" => 8.0e-5,
                 "lon0_deg" => 0.0,
                 "lat0_deg" => 35.0,
                 "sigma_lon_deg" => 30.0,
                 "sigma_lat_deg" => 20.0))
        @test vmr_blob isa NTuple{6, Array{FT, 3}}
        @test all(size(vmr_blob[p]) == (Nc, Nc, Nz) for p in 1:6)
        blob_vals = vcat((vec(vmr_blob[p]) for p in 1:6)...)
        @test minimum(blob_vals) ≥ FT(4.0e-4)
        @test maximum(blob_vals) > FT(4.0e-4)

        # Unsupported kind errors
        @test_throws ArgumentError build_initial_mixing_ratio(
            air_mass, grid, Dict("kind" => "bl_enhanced"))
    end
end
