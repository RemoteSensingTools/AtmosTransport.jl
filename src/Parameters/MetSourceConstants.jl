# =============================================================================
# Met-source constants
# =============================================================================
#
# Fixed properties of the external meteorological datasets that the readers and
# preprocessors rely on: level counts, archive cadences, accumulation periods
# and grid offsets. Values a run config can override still default to these.
# =============================================================================

"Model levels of the ECMWF IFS L137 grid on which ERA5 is archived."
const ERA5_NATIVE_LEVEL_COUNT = 137

"Interval of the ERA5 hourly analyses, and accumulation period of its hourly accumulated fields [s]."
const ERA5_MET_INTERVAL_SECONDS = 3600.0

"""
GEOS FV3 dynamics time step [s]. GEOS-FP and GEOS-IT accumulate the mass fluxes
`MFXC`/`MFYC` over this step, not over the hourly met interval.
"""
const GEOS_MASS_FLUX_DT_SECONDS = 450.0

"Interval of the GEOS-FP / GEOS-IT hourly CTM collections [s]."
const GEOS_MET_INTERVAL_SECONDS = 3600.0

"Interval of the GEOS and MERRA-2 three-hourly collections (A3, I3) [s]."
const GEOS_3HOURLY_INTERVAL_SECONDS = 10800.0

"Longitude by which the GMAO (GEOS) cubed sphere is rotated [degrees]."
const GMAO_CS_LONGITUDE_OFFSET_DEG = -10.0

"""
Plausible range of 2 m air temperature in met input [K], an open interval:
readers reject values outside it (or on its ends) as a sign of a wrong variable,
unit or fill value.
"""
const SURFACE_AIR_TEMPERATURE_RANGE = (150.0, 350.0)

"Bound on the magnitude of the surface sensible or latent heat flux in met input; readers reject values at or above it [W m⁻²]."
const SURFACE_HEAT_FLUX_MAX_ABS = 5000.0

export ERA5_NATIVE_LEVEL_COUNT, ERA5_MET_INTERVAL_SECONDS, GEOS_MASS_FLUX_DT_SECONDS,
       GEOS_MET_INTERVAL_SECONDS, GEOS_3HOURLY_INTERVAL_SECONDS, GMAO_CS_LONGITUDE_OFFSET_DEG,
       SURFACE_AIR_TEMPERATURE_RANGE, SURFACE_HEAT_FLUX_MAX_ABS
