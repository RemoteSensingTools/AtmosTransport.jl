#!/usr/bin/env julia
# GEOS-Chem non-local VDIFF port: column exchange `dkg`, the counter-gradient
# emission profile, and the profile-weighted surface emission deposit.

using Test
using AtmosTransport
using AtmosTransport.Operators: CMFMCWorkspace, apply_convection!
using AtmosTransport.State: GCHPNonlocalPBLField, GCHPVdiffParameters, refresh_gchp_nonlocal_pbl!,
                            panel_field
using AtmosTransport.Operators.SurfaceFlux: SurfaceFluxSource, SurfaceFluxOperator,
                                            ProfileDeposit, apply_surface_flux!

include(joinpath(@__DIR__, "..", "helpers", "gchp_nonlocal_pbl.jl"))
const FT = Float64
const Nc, Nz, Hp = NONLOCAL_NC, NONLOCAL_NZ, NONLOCAL_HP
const R = 287.0
refreshed(T = FT; kw...) = refreshed_pbl_field(T; kw...)
synthetic_window(; kw...) = synthetic_pbl_window(; kw...)

@testset "GCHP non-local PBL column" begin
    field, col = refreshed(hflux = 250.0)
    dkg = panel_field(field, 1).data[1, 1, :]
    profile = field.emission_profile[1][1, 1, :]
    z = field.z_mid[1][1, 1, :]
    @test all(isfinite, dkg) && all(>=(0), dkg)
    @test dkg[Nz] == 0                                            # closed surface
    @test sum(profile) ≈ 1 atol = 1e-12                           # deposits all emissions
    @test profile[Nz] < 1                                         # unstable: lofted fraction
    @test all(iszero, profile[z .> 1500 .+ 300])                  # nothing above the PBL
    # Above the shear layer K is GEOS-Chem's floor, so dkg = A ρ_dry K_min / Δz.
    prm = GCHPVdiffParameters{FT}()
    k = findfirst(<(6000), z)                                     # edge between k-1 and k
    p_edge = (field.p_mid[1][1, 1, k - 1] + field.p_mid[1][1, 1, k]) / 2
    ρ_dry = p_edge / (R * (col.T[k - 1] + col.T[k]) / 2) * (1 - (col.qv[k - 1] + col.qv[k]) / 2)
    @test dkg[k - 1] ≈ 1e10 * ρ_dry * prm.kz_min / (z[k - 1] - z[k]) rtol = 1e-6
    # Boundary-layer exchange dwarfs the free-troposphere floor.
    @test maximum(dkg[z .< 1500]) > 100 * dkg[k - 1]

    # GEOS-Chem's npbl: bottom layers with reference pressure above 400 hPa.
    @test AtmosTransport.State.Fields.gchp_pbl_layer_count(col.vertical.A, col.vertical.B, prm) ==
          count(>(4e4), col.p_mid)

    stable, _ = refreshed(hflux = -30.0, eflux = 0.0)
    @test stable.emission_profile[1][1, 1, :] == [zeros(Nz - 1); 1]   # no counter-gradient

    # Float32: shear-free free troposphere stays finite (no denormal Ri).
    f32, _ = refreshed(Float32; hflux = 250.0)
    @test all(isfinite, panel_field(f32, 1).data) && all(isfinite, f32.emission_profile[1])
    @test sum(f32.emission_profile[1][1, 1, :]) ≈ 1 atol = 1e-5

    # Forcing without the latent heat flux or the VDIFF profiles is rejected.
    sfc, vdiff, m, areas, c = synthetic_window(hflux = 100.0)
    field = GCHPNonlocalPBLField(Nc, Nz, FT)
    @test_throws ArgumentError refresh_gchp_nonlocal_pbl!(field, merge(sfc, (; eflux = nothing)),
                                                          vdiff, m, areas, c.vertical; halo_width = Hp)
    @test_throws ArgumentError refresh_gchp_nonlocal_pbl!(field, sfc, nothing, m, areas, c.vertical;
                                                          halo_width = Hp)
end

@testset "Profile-weighted emission deposit" begin
    rate = ntuple(_ -> fill(FT(2.0), Nc, Nc), 6)                  # kg s⁻¹ per cell
    op = SurfaceFluxOperator(SurfaceFluxSource(:x, rate))
    profile = zeros(FT, Nc, Nc, Nz); profile[1, 1, Nz-2:Nz] .= (0.2, 0.3, 0.5); profile[2:end, :, Nz] .= 1
    profile[1, 2, Nz] = 1
    deposit = ntuple(_ -> ProfileDeposit(profile), 6)
    q = ntuple(_ -> fill(FT(100.0), Nc + 2Hp, Nc + 2Hp, Nz), 6)
    apply_surface_flux!(q, op, nothing, 10.0, nothing, nothing; tracer_names = (:x,),
                        halo_width = Hp, deposit)
    added = q[1][1 + Hp, 1 + Hp, :] .- 100
    @test added[Nz-2:Nz] ≈ [4.0, 6.0, 10.0]
    @test sum(added) ≈ 20.0                                       # mass exact
    @test q[1][2 + Hp, 1 + Hp, Nz] ≈ 120.0                        # surface-only column

    # A layer the counter-gradient part alone would turn negative keeps the
    # emission at the surface (GEOS-Chem's qmincg check).
    profile[1, 1, Nz-2:Nz] .= (0.2, -0.5, 1.3)
    q = ntuple(_ -> fill(FT(1.0), Nc + 2Hp, Nc + 2Hp, Nz), 6)
    apply_surface_flux!(q, op, nothing, 10.0, nothing, nothing; tracer_names = (:x,),
                        halo_width = Hp, deposit)
    @test q[1][1 + Hp, 1 + Hp, Nz] ≈ 21.0 && q[1][1 + Hp, 1 + Hp, Nz-1] == 1.0

    # Packed multi-tracer panels: only the emitting tracer's slab changes.
    profile[1, 1, Nz-2:Nz] .= (0.2, 0.3, 0.5)
    packed = ntuple(_ -> fill(FT(100), Nc + 2Hp, Nc + 2Hp, Nz, 2), 6)
    apply_surface_flux!(packed, op, nothing, 10.0, nothing, nothing; tracer_names = (:y, :x),
                        halo_width = Hp, deposit)
    @test packed[1][1 + Hp, 1 + Hp, Nz-2:Nz, 2] ≈ [104.0, 106.0, 110.0]
    @test all(==(100.0), packed[1][:, :, :, 1])
end

@testset "geoschem_nonlocal_vdiff config" begin
    spec(d) = AtmosTransport.Models.diffusion_spec(d)
    @test spec(Dict("kind" => "geoschem_nonlocal_vdiff")) isa
          AtmosTransport.Models.GCHPNonlocalVdiffDiffusionSpec
    @test_throws ArgumentError spec(Dict("kind" => "geoschem_nonlocal_vdiff",
                                         "surface_flux_boundary" => false))
    conv = AtmosTransport.Models.convection_spec(Dict("kind" => "cmfmc", "cloud_base" => "dqrcu"))
    @test conv.cloud_base isa ArchivedCloudBase
    @test_throws ArgumentError AtmosTransport.Models.convection_spec(
        Dict("kind" => "cmfmc", "cloud_base" => "lcl"))
end

@testset "CMFMC convection with GEOS-Chem's archived cloud base" begin
    Nc_, Hp_, Nz_ = 2, 1, 8
    N = Nc_ + 2Hp_
    mesh = CubedSphereMesh(; Nc = Nc_, Hp = Hp_, FT = FT)
    vc = HybridSigmaPressure(collect(range(0.0, 0.0; length = Nz_ + 1)), collect(range(0.0, 1.0; length = Nz_ + 1)))
    grid = AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT = FT)
    air = ntuple(_ -> fill(FT(1e15), N, N, Nz_), 6)
    areas = ntuple(_ -> fill(FT(1e12), Nc_, Nc_), 6)
    cmfmc = ntuple(_ -> zeros(FT, Nc_, Nc_, Nz_ + 1), 6)
    for p in 1:6
        cmfmc[p][:, :, 3:8] .= 0.02                 # updraft through edges 3…8; edge 8 = bottom of layer 7
    end
    dtrain = ntuple(_ -> zeros(FT, Nc_, Nc_, Nz_), 6)
    for p in 1:6
        dtrain[p][:, :, 2] .= 0.02                  # detrain at cloud top
    end
    archived = ntuple(_ -> fill(FT(5), Nc_, Nc_), 6)  # DQRCU cloud base at layer 5
    profile = FT[400, 400, 400, 400, 401, 402, 405, 410]
    function run(op)
        q = ntuple(_ -> repeat(reshape(profile .* 1e15, 1, 1, Nz_, 1), N, N, 1, 1), 6)
        ws = CMFMCWorkspace(q; cell_metrics = areas)
        apply_convection!(q, air, ConvectionForcing(cmfmc, dtrain, nothing, archived), op, FT(600), ws, grid)
        return q[1][1 + Hp_, 1 + Hp_, :, 1] ./ 1e15
    end
    q_edge = run(CMFMCConvection())
    q_arch = run(CMFMCConvection(cloud_base = ArchivedCloudBase()))
    @test sum(q_arch) ≈ sum(profile) rtol = 1e-14          # column mass conserved
    @test q_arch[6] ≈ q_arch[7] ≈ q_arch[8]                # layers below layer 5 well mixed
    @test !(q_edge[6] ≈ q_edge[8])                         # edge base (layer 7) leaves layer 6 alone
    # The archived rule needs the cloud base in the forcing.
    q0 = ntuple(_ -> zeros(FT, N, N, Nz_, 1), 6)
    @test_throws ArgumentError apply_convection!(q0, air, ConvectionForcing(cmfmc, dtrain, nothing),
                                                 CMFMCConvection(cloud_base = ArchivedCloudBase()), FT(600),
                                                 CMFMCWorkspace(q0; cell_metrics = areas), grid)
end
