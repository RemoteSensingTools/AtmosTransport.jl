# The physical constants, numerical defaults and met-source constants live in
# src/Parameters/; the parameter objects and meshes default to them, in their
# own precision.

using Test
using AtmosTransport
using AtmosTransport.Parameters: EARTH_RADIUS, IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE,
                                 R_DRY_AIR, CP_OVER_R_DIATOMIC, CP_DRY_AIR, DRY_AIR_MOLAR_MASS,
                                 VON_KARMAN, STANDARD_SEA_LEVEL_AIR_DENSITY,
                                 GLOBAL_MEAN_DRY_SURFACE_PRESSURE, SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS,
                                 GEOSCHEM_SPECIES_MOLAR_MASS, RN222_HALF_LIFE,
                                 TM5_CONVECTION_MASS_BASIS_KEY

# Every exported constant of `Parameters`, with its value. The test compares with
# `===`, which pins the type as well: counts stay `Int`, everything else stays
# `Float64`. A constant added to src/Parameters must be added here too; the
# completeness test below fails otherwise.
const PINNED_PARAMETERS = (
    DEFAULT_OBSERVATION_LAYER_TEMPERATURE_KELVIN = 280.0,
    # PhysicalConstants.jl
    EARTH_RADIUS = 6.371e6,
    IFS_EARTH_RADIUS = 6.371229e6,
    STANDARD_GRAVITY = 9.80665,
    STANDARD_PRESSURE = 101325.0,
    R_DRY_AIR = 287.04,
    CP_OVER_R_DIATOMIC = 3.5,
    CP_DRY_AIR = 1004.64,
    DRY_AIR_MOLAR_MASS = 28.9644e-3,
    THETA_REFERENCE_PRESSURE = 1.0e5,
    AVOGADRO = 6.02214076e23,
    VIRTUAL_TEMPERATURE_FACTOR = 0.61,
    VON_KARMAN = 0.41,
    STANDARD_SEA_LEVEL_AIR_DENSITY = 1.225,
    STANDARD_LAPSE_RATE = 0.0065,
    STANDARD_TROPOPAUSE_TEMPERATURE = 216.65,
    HYDROSTATIC_REFERENCE_TEMPERATURE = 260.0,
    DAYS_PER_JULIAN_YEAR = 365.25,
    GLOBAL_MEAN_DRY_SURFACE_PRESSURE = 98325.0,
    GLOBAL_MEAN_SPECIFIC_HUMIDITY = 0.00247,
    SPECIES_MOLAR_MASS = (co2 = 44.0095e-3, sf6 = 146.055e-3, rn222 = 222.0e-3),
    GEOSCHEM_SPECIES_MOLAR_MASS = (co2 = 44.01e-3, sf6 = 146.06e-3, rn222 = 222.0e-3),
    RN222_HALF_LIFE = 330350.4,
    CARBON_MOLAR_MASS = 12.0107e-3,
    TM5_CONSTANTS = (gravity = 9.80665, cp_air = 1004.0, r_air = 287.307, r_vap = 461.51,
                     l_vap = 2.5e6, karman = 0.4, p_ref = 1.0e5),
    GEOSCHEM_CONSTANTS = (gravity = 9.80665, r_dry = 287.0, cp_dry = 1004.64, r_vap = 461.0,
                          l_vap = 2.5104e6, karman = 0.4, p_ref = 1.0e5),
    # NumericalDefaults.jl
    DEFAULT_CFL_LIMIT = 0.95,
    DEFAULT_POSITIVITY_CFL_LIMIT = 0.95,
    DEFAULT_SUBSTEP_CFL_TARGET = 0.95,
    ERA5_N320_SUBSTEP_CFL_TARGET = 0.85,
    COARSEN_SUBSTEP_CFL_TARGET = 0.90,
    DEFAULT_MET_INTERVAL_SECONDS = 3600.0,
    DEFAULT_BACKGROUND_VMR = 4.0e-4,
    DEFAULT_GEOS_SMOOTHING_SWEEPS = 8,
    DEFAULT_MAX_SUBCYCLES = 4096,
    DEFAULT_MAX_STEPS_PER_WINDOW = 4096,
    ADAPTIVE_SUBSTEP_MAX_REFINEMENTS = 8,
    DEFAULT_POISSON_BALANCE_TOL = 1.0e-14,
    DEFAULT_POISSON_BALANCE_MAX_ITER = 20000,
    DEFAULT_POISSON_PROJECT_EVERY = 50,
    HORIZONTAL_BALANCE_TAG = "column",
    HYBRID_LAYER_MIN_DB = 1.0e-6,
    COLUMN_CLOSURE_TOL_KG = 10.0,
    OMEGA_FIDELITY_NRMSE_LIMIT = 1.0,
    OMEGA_FIDELITY_CORRELATION_MIN = 0.5,
    OMEGA_FIDELITY_SLOPE_MIN = 0.5,
    OMEGA_FIDELITY_SLOPE_MAX = 1.5,
    REPLAY_TOLERANCE_FLOAT64 = 1.0e-12,
    REPLAY_TOLERANCE_FLOAT32 = 1.0e-5,
    DEFAULT_GEOS_SMOOTHING_WEIGHT = 0.5,
    DEFAULT_GAUSSIAN_BLOB_SIGMA_DEG = 10.0,
    DEFAULT_LINROOD_PPM_ORDER = 5,
    # MetSourceConstants.jl
    ERA5_NATIVE_LEVEL_COUNT = 137,
    ERA5_MET_INTERVAL_SECONDS = 3600.0,
    TM5_CONVECTION_MASS_BASIS_KEY = "tm5_convection_mass_basis",
    TM5_CONVECTION_SOURCE_KEY = "tm5_convection_source",
    TM5_EC2TM_CONVENTION_KEY = "tm5_ec2tm_convention",
    TM5_EC2TM_CONVENTION = "tm5_halflevel_v2",
    GEOS_MASS_FLUX_DT_SECONDS = 450.0,
    GEOS_MET_INTERVAL_SECONDS = 3600.0,
    GEOS_3HOURLY_INTERVAL_SECONDS = 10800.0,
    GMAO_CS_LONGITUDE_OFFSET_DEG = -10.0,
    SURFACE_AIR_TEMPERATURE_RANGE = (150.0, 350.0),
    SURFACE_HEAT_FLUX_MAX_ABS = 5000.0,
    # TM5 echlevs lists (TM5 half-level numbering, surface first).
    ECHLEVS_ML137_TROPO34 = (137, 134, 129, 124, 119, 114, 110, 105, 101, 97, 93, 88, 84, 81,
                             78, 76, 73, 70, 67, 65, 62, 59, 57, 54, 51, 46, 42, 37, 32, 27,
                             22, 17, 12, 7, 0),
    ECHLEVS_ML137_TROPO34A = (137, 134, 131, 126, 122, 117, 113, 109, 105, 102, 98, 96, 94, 92,
                              90, 88, 85, 83, 81, 78, 75, 73, 70, 67, 64, 61, 57, 54, 51, 44,
                              34, 26, 20, 14, 0),
    # AtmosTransport-native recipes (top first).
    ECHLEVS_ML137_66L = (0, 2:5:37..., 41, 44:3:77..., 79:2:107..., 108:137...),
    ECHLEVS_ML137_CFL94 = (0, 12, 24:4:52..., 53:137...),
    ECHLEVS_ML137_CFL85 = (0, 12, 24:4:52..., 53:2:71..., 72:137...),
    ECHLEVS_ML137_FULL = Tuple(0:137),
    ERA5_L137_ECHLEVS_PRESETS = (
        ml137_tropo34 = (137, 134, 129, 124, 119, 114, 110, 105, 101, 97, 93, 88, 84, 81, 78,
                         76, 73, 70, 67, 65, 62, 59, 57, 54, 51, 46, 42, 37, 32, 27, 22, 17,
                         12, 7, 0),
        ml137_tropo34a = (137, 134, 131, 126, 122, 117, 113, 109, 105, 102, 98, 96, 94, 92, 90,
                          88, 85, 83, 81, 78, 75, 73, 70, 67, 64, 61, 57, 54, 51, 44, 34, 26,
                          20, 14, 0),
        ml137_66L = (0, 2:5:37..., 41, 44:3:77..., 79:2:107..., 108:137...),
        ml137_cfl94 = (0, 12, 24:4:52..., 53:137...),
        ml137_cfl85 = (0, 12, 24:4:52..., 53:2:71..., 72:137...),
        ml137_full = Tuple(0:137),
    ),
)

@testset "every Parameters constant is pinned" begin
    P = AtmosTransport.Parameters
    @testset "$name" for name in keys(PINNED_PARAMETERS)
        @test getfield(P, name) === PINNED_PARAMETERS[name]
    end
    constants = setdiff(names(P), (:Parameters, :PlanetParameters, :earth_parameters))
    pinned = collect(keys(PINNED_PARAMETERS))
    @test setdiff(constants, pinned) == Symbol[]   # exported but not pinned
    @test setdiff(pinned, constants) == Symbol[]   # pinned but not exported
end

@testset "physical constants are coherent" begin
    @test CP_DRY_AIR ≈ CP_OVER_R_DIATOMIC * R_DRY_AIR        # the GEOS set is coherent (to one ulp)
    @test SPECIES_MOLAR_MASS.co2 ≈ CARBON_MOLAR_MASS + 2 * 15.9994e-3   # IUPAC 2007 C and O
    # Modern-reanalysis pressure target, cross-checked against T&S pressure.
    ts_dry_ps = 98550.0 * 5.1352 / 5.1480
    @test GLOBAL_MEAN_DRY_SURFACE_PRESSURE ≈ ts_dry_ps rtol = 5e-4
    @test GLOBAL_MEAN_DRY_SURFACE_PRESSURE / (1 - 0.00247) ≈ 98550.0 rtol = 5e-4
    # 3.8235 days in Float64 is the 330350.4 s of the run configs, bit for bit.
    @test RN222_HALF_LIFE === 3.8235 * 86400.0
    # GEOS-Chem's species molar masses differ from the model's by the amounts its docstring states.
    @test GEOSCHEM_SPECIES_MOLAR_MASS.co2 / SPECIES_MOLAR_MASS.co2 - 1 ≈ 1.14e-5 rtol = 0.01
    @test GEOSCHEM_SPECIES_MOLAR_MASS.sf6 / SPECIES_MOLAR_MASS.sf6 - 1 ≈ 3.42e-5 rtol = 0.01
    @test GEOSCHEM_SPECIES_MOLAR_MASS.rn222 === SPECIES_MOLAR_MASS.rn222
end

@testset "defaults take the constants in their own precision" begin
    for FT in (Float32, Float64)
        planet = AtmosTransport.PlanetParameters(; FT)
        @test (planet.radius, planet.gravity, planet.reference_pressure) ===
              (FT(EARTH_RADIUS), FT(STANDARD_GRAVITY), FT(STANDARD_PRESSURE))
        @test AtmosTransport.CubedSphereMesh(; Nc = 2, FT).radius === FT(EARTH_RADIUS)
        bl = AtmosTransport.Preprocessing.BLDiffConstants{FT}()
        @test (bl.grav, bl.cp_air, bl.r_air) === (FT(9.80665), FT(1004.0), FT(287.307))
        @test bl.ustar_min === FT(0.01)
        vdiff = AtmosTransport.State.Fields.GCHPVdiffParameters{FT}()
        @test (vdiff.g, vdiff.R_dry, vdiff.ε_virtual) === (FT(9.80665), FT(287.0), FT(461.0 / 287.0 - 1))
        pbl = AtmosTransport.State.Fields.PBLPhysicsParameters{FT}()
        @test (pbl.gravity, pbl.cp_dry, pbl.kappa_vk, pbl.rho_ref) ===
              (FT(STANDARD_GRAVITY), FT(CP_DRY_AIR), FT(VON_KARMAN), FT(STANDARD_SEA_LEVEL_AIR_DENSITY))
        # Surface-input floors and profile shape of the Beljaars-Viterbo Kz fields.
        @test (pbl.h_pbl_min, pbl.ustar_min, pbl.T_sfc_min) === (FT(100), FT(0.01), FT(200))
        @test (pbl.surface_layer_fraction, pbl.entrainment_zone_fraction) === (FT(0.1), FT(0.2))
        # The entrainment-zone top `(1 + fraction) h_pbl` equals the former `1.2 h_pbl` bit for bit.
        @test one(FT) + pbl.entrainment_zone_fraction === FT(1.2)
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
