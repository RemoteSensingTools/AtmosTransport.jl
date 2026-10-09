# The physical constants live in src/Parameters/PhysicalConstants.jl; the
# parameter objects and meshes default to them, in their own precision.

using Test
using AtmosTransport
using AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE,
                                 R_DRY_AIR, CP_OVER_R_DIATOMIC, CP_DRY_AIR, DRY_AIR_MOLAR_MASS,
                                 THETA_REFERENCE_PRESSURE, AVOGADRO, VIRTUAL_TEMPERATURE_FACTOR,
                                 SPECIES_MOLAR_MASS, TM5_CONSTANTS, GEOSCHEM_CONSTANTS

@testset "physical constants" begin
    @test (EARTH_RADIUS, IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE) == (6.371e6, 6.371229e6, 9.80665, 101325.0)
    @test (R_DRY_AIR, CP_DRY_AIR, DRY_AIR_MOLAR_MASS) == (287.04, 1004.64, 28.9644e-3)
    @test CP_OVER_R_DIATOMIC == 3.5
    @test CP_DRY_AIR ≈ CP_OVER_R_DIATOMIC * R_DRY_AIR        # the GEOS set is coherent (to one ulp)
    @test (THETA_REFERENCE_PRESSURE, AVOGADRO, VIRTUAL_TEMPERATURE_FACTOR) == (1e5, 6.02214076e23, 0.61)
    @test SPECIES_MOLAR_MASS == (co2 = 44.0095e-3, sf6 = 146.055e-3, rn222 = 222.0e-3)
    @test TM5_CONSTANTS == (gravity = 9.80665, cp_air = 1004.0, r_air = 287.307, r_vap = 461.51,
                            l_vap = 2.5e6, karman = 0.4, p_ref = 1.0e5)
    @test GEOSCHEM_CONSTANTS == (gravity = 9.80665, r_dry = 287.0, cp_dry = 1004.64, r_vap = 461.0,
                                 l_vap = 2.5104e6, karman = 0.4, p_ref = 1e5)
end

@testset "defaults take the constants in their own precision" begin
    for FT in (Float32, Float64)
        planet = AtmosTransport.PlanetParameters(; FT)
        @test (planet.radius, planet.gravity, planet.reference_pressure) ===
              (FT(EARTH_RADIUS), FT(STANDARD_GRAVITY), FT(STANDARD_PRESSURE))
        @test AtmosTransport.CubedSphereMesh(; Nc = 2, FT).radius === FT(EARTH_RADIUS)
        bl = AtmosTransport.Preprocessing.BLDiffConstants{FT}()
        @test (bl.grav, bl.cp_air, bl.r_air) === (FT(9.80665), FT(1004.0), FT(287.307))
        vdiff = AtmosTransport.State.Fields.GCHPVdiffParameters{FT}()
        @test (vdiff.g, vdiff.R_dry, vdiff.ε_virtual) === (FT(9.80665), FT(287.0), FT(461.0 / 287.0 - 1))
        pbl = AtmosTransport.State.Fields.PBLPhysicsParameters{FT}()
        @test (pbl.gravity, pbl.cp_dry) === (FT(STANDARD_GRAVITY), FT(CP_DRY_AIR))
    end
end

@testset "one dry-air gas constant, gravity and molar mass" begin
    D = AtmosTransport.Operators.Diffusion
    # Hydrostatic layer thickness of the diffusion operator, dz = R T / g · Δp / p.
    ps, ak, bk = fill(1e5, 1, 1), [0.0, 5e4, 0.0], [0.0, 0.0, 1.0]
    dz = zeros(1, 1, 2)
    D.fill_dz_hydrostatic_constT!(dz, ps, ak, bk; T_ref = 260)
    @test dz[1, 1, 2] ≈ R_DRY_AIR * 260 / STANDARD_GRAVITY * 5e4 / 7.5e4 rtol = 1e-14
    # Surface-flux mass conversion: kg tracer per mol-ratio of dry air.
    @test AtmosTransport.Models.InitialConditionIO.DRY_AIR_MOLAR_MASS === DRY_AIR_MOLAR_MASS
end
