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
Von Kármán constant of the model's own boundary-layer schemes (0.41, the value of
the legacy PBL diffusion they port). The TM5 and GEOS-Chem ports use their models'
0.4 (`TM5_CONSTANTS.karman`, `GEOSCHEM_CONSTANTS.karman`).
"""
const VON_KARMAN = 0.41

"Air density at sea level in the U.S. Standard Atmosphere 1976 [kg m⁻³]."
const STANDARD_SEA_LEVEL_AIR_DENSITY = 1.225

"Temperature lapse rate of the troposphere in the U.S. Standard Atmosphere 1976 [K m⁻¹]."
const STANDARD_LAPSE_RATE = 0.0065

"Tropopause temperature of the U.S. Standard Atmosphere 1976 [K]."
const STANDARD_TROPOPAUSE_TEMPERATURE = 216.65

"""
Temperature of an isothermal atmosphere used for hydrostatic layer thicknesses
when no temperature is available, `dz = R_dry T / g · Δp / p_mid` (layer pressure
thickness over its midpoint pressure) [K].
"""
const HYDROSTATIC_REFERENCE_TEMPERATURE = 260.0

"Days in a Julian year, for annual totals given as rates."
const DAYS_PER_JULIAN_YEAR = 365.25

"""
Global-mean dry-air surface pressure that the preprocessing dry-mass fix holds
[Pa]: over the IFS sphere (`IFS_EARTH_RADIUS`) with `STANDARD_GRAVITY` it is a dry
air mass of 5.1353 × 10¹⁸ kg, the value of Trenberth and Smith (2005, J. Climate
18, 864–875).
"""
const GLOBAL_MEAN_DRY_SURFACE_PRESSURE = 98726.0

"""
Global-mean specific humidity [kg kg⁻¹] that turns the dry surface-pressure
target into a total one where only total pressure is available:
`ps_total = GLOBAL_MEAN_DRY_SURFACE_PRESSURE / (1 − q)`.
"""
const GLOBAL_MEAN_SPECIFIC_HUMIDITY = 0.00247

"""
Molar masses of the transported species [kg mol⁻¹], for converting fluxes and
mixing ratios to and from mass: CO₂ (IUPAC atomic weights), SF₆ and ²²²Rn.
"""
const SPECIES_MOLAR_MASS = (co2 = 44.0095e-3, sf6 = 146.055e-3, rn222 = 222.0e-3)

"""
Molar mass of carbon [kg mol⁻¹] (IUPAC 2007 standard atomic weight, the value
behind `SPECIES_MOLAR_MASS.co2 = 12.0107 + 2 × 15.9994`). Carbon-mass fluxes
(kgC) become CO₂-mass fluxes by `SPECIES_MOLAR_MASS.co2 / CARBON_MOLAR_MASS`
≈ 3.66419, which conserves carbon atoms; GEOS-Chem's HEMCO uses the scale factor
`CTOCO2 = 3.6667` (44/12), 0.068 % more.
"""
const CARBON_MOLAR_MASS = 12.0107e-3

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
       VIRTUAL_TEMPERATURE_FACTOR, VON_KARMAN, STANDARD_SEA_LEVEL_AIR_DENSITY, STANDARD_LAPSE_RATE,
       STANDARD_TROPOPAUSE_TEMPERATURE, HYDROSTATIC_REFERENCE_TEMPERATURE, DAYS_PER_JULIAN_YEAR,
       GLOBAL_MEAN_DRY_SURFACE_PRESSURE, GLOBAL_MEAN_SPECIFIC_HUMIDITY,
       SPECIES_MOLAR_MASS, CARBON_MOLAR_MASS, TM5_CONSTANTS, GEOSCHEM_CONSTANTS
