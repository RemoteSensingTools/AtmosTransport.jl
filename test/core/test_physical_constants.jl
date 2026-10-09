# The physical constants live in src/Parameters/PhysicalConstants.jl; the
# parameter objects and meshes default to them, in their own precision.

using Test
using AtmosTransport
using AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE,
                                 CP_DRY_AIR, AVOGADRO, VIRTUAL_TEMPERATURE_FACTOR, SPECIES_MOLAR_MASS,
                                 TM5_CONSTANTS, GEOSCHEM_CONSTANTS

@testset "physical constants" begin
    @test (EARTH_RADIUS, IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE) == (6.371e6, 6.371229e6, 9.80665, 101325.0)
    @test (CP_DRY_AIR, AVOGADRO, VIRTUAL_TEMPERATURE_FACTOR) == (1004.64, 6.02214076e23, 0.61)
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
    end
end
