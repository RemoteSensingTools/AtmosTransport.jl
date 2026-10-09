# =============================================================================
# Physical constants
# =============================================================================
#
# The one place for the constants of nature the model uses, in SI units, with
# their sources. `PlanetParameters` defaults to the Earth values here. A scheme
# that reproduces another model's code uses that model's constant set
# (`TM5_CONSTANTS`, `GEOSCHEM_CONSTANTS`) so that it matches the reference
# model; everything else uses the model's own values. Scheme tuning parameters
# (stability-function slopes, critical Richardson numbers, ...) stay with their
# schemes.
# =============================================================================

"Default Earth radius of meshes built without one [m] (mean radius, rounded)."
const EARTH_RADIUS = 6.371e6

"Earth radius of the ECMWF IFS [m]: ERA5 spectral transforms, TM5, and the preprocessing target meshes."
const IFS_EARTH_RADIUS = 6.371229e6

"Standard acceleration of gravity [m s⁻²] (3rd CGPM 1901; WMO)."
const STANDARD_GRAVITY = 9.80665

"Standard sea-level pressure [Pa]."
const STANDARD_PRESSURE = 101325.0

"Gas constant of dry air [J kg⁻¹ K⁻¹] (GEOS, MAPL `MAPL_RDRY`)."
const R_DRY_AIR = 287.04

"Ratio of the heat capacity at constant pressure to the gas constant of an ideal diatomic gas, 7/2."
const CP_OVER_R_DIATOMIC = 3.5

"Heat capacity of dry air at constant pressure [J kg⁻¹ K⁻¹] (GEOS; the decimal product 3.5 × 287.04)."
const CP_DRY_AIR = 1004.64

"Molar mass of dry air [kg mol⁻¹] (U.S. Standard Atmosphere 1976; GEOS-Chem `AIRMW`)."
const DRY_AIR_MOLAR_MASS = 28.9644e-3

"Reference pressure of the potential temperature, θ = T (p₀/p)^κ [Pa]."
const THETA_REFERENCE_PRESSURE = 1.0e5

"Avogadro constant [mol⁻¹] (SI 2019, exact)."
const AVOGADRO = 6.02214076e23

"""
Coefficient of the specific humidity in the virtual temperature,
`T_v = T (1 + 0.61 q)`: `R_vap / R_dry − 1 ≈ 0.608`, rounded as in most models.
"""
const VIRTUAL_TEMPERATURE_FACTOR = 0.61

"""
Molar masses of the transported species [kg mol⁻¹], for converting fluxes and
mixing ratios to and from mass: CO₂ (IUPAC atomic weights), SF₆ and ²²²Rn.
"""
const SPECIES_MOLAR_MASS = (co2 = 44.0095e-3, sf6 = 146.055e-3, rn222 = 222.0e-3)

"""
TM5's physical constants, used by the port of TM5's boundary-layer diffusion
(`tm5_bldiff.jl`): gravity [m s⁻²], dry-air heat capacity and gas constant
(`Rgas · 1000 / 28.94`, TM5 `binas.F90`) and water-vapour gas constant
[J kg⁻¹ K⁻¹], latent heat of vaporisation [J kg⁻¹], von Kármán constant, and the
potential-temperature reference pressure [Pa].
"""
const TM5_CONSTANTS = (gravity = 9.80665, cp_air = 1004.0, r_air = 287.307, r_vap = 461.51,
                       l_vap = 2.5e6, karman = 0.4, p_ref = 1.0e5)

"""
GEOS-Chem's constants, used by the GEOS-Chem non-local boundary-layer scheme:
gravity [m s⁻²], dry-air and water-vapour gas constants [J kg⁻¹ K⁻¹] and the von
Kármán constant from `Headers/physconstants.F90`; dry-air heat capacity
(`cpair`) [J kg⁻¹ K⁻¹] and latent heat of vaporisation (`latvap`) [J kg⁻¹] from
`GeosCore/vdiff_mod.F90`; the potential-temperature reference pressure [Pa] is
this model's convention.
"""
const GEOSCHEM_CONSTANTS = (gravity = 9.80665, r_dry = 287.0, cp_dry = 1004.64, r_vap = 461.0,
                            l_vap = 2.5104e6, karman = 0.4, p_ref = THETA_REFERENCE_PRESSURE)

export EARTH_RADIUS, IFS_EARTH_RADIUS, STANDARD_GRAVITY, STANDARD_PRESSURE, R_DRY_AIR, CP_OVER_R_DIATOMIC,
       CP_DRY_AIR, DRY_AIR_MOLAR_MASS, THETA_REFERENCE_PRESSURE, AVOGADRO,
       VIRTUAL_TEMPERATURE_FACTOR, SPECIES_MOLAR_MASS, TM5_CONSTANTS, GEOSCHEM_CONSTANTS
