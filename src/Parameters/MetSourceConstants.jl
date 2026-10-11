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

# Interface selections for the ERA5 L137 hybrid coordinate. Values use the
# ECMWF/TM5 half-level numbering: 0 is the model top and 137 is the surface.
# Tuples keep the process-wide recipes immutable; preprocessing returns a
# Vector copy to callers that need one.

"TM5-MP r1112 `levels/ml137/tropo34` interface list, exactly as written."
const ECHLEVS_ML137_TROPO34 = (
    137, 134, 129, 124, 119, 114, 110, 105, 101,  97,
     93,  88,  84,  81,  78,  76,  73,  70,  67,  65,
     62,  59,  57,  54,  51,  46,  42,  37,  32,  27,
     22,  17,  12,   7,   0,
)

"TM5 cy3-4dvar `levels/ml137/tropo34a` (CAMS73) interface list, exactly as written."
const ECHLEVS_ML137_TROPO34A = (
    137, 134, 131, 126, 122, 117, 113, 109, 105, 102,
     98,  96,  94,  92,  90,  88,  85,  83,  81,  78,
     75,  73,  70,  67,  64,  61,  57,  54,  51,  44,
     34,  26,  20,  14,   0,
)

# AtmosTransport-native recipes. These values are the `137 - e` transforms
# of the recipes predating TM5-numbered selection, so correcting the selector
# leaves their physical grids bit-identical.
const ECHLEVS_ML137_66L = (
      0,   2,   7,  12,  17,  22,  27,  32,  37,  41,
     44,  47,  50,  53,  56,  59,  62,  65,  68,  71,
     74,  77,  79,  81,  83,  85,  87,  89,  91,  93,
     95,  97,  99, 101, 103, 105, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137,
)

const ECHLEVS_ML137_CFL94 = (
      0,  12,  24,  28,  32,  36,  40,  44,  48,  52,
     53,  54,  55,  56,  57,  58,  59,  60,  61,  62,
     63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
     73,  74,  75,  76,  77,  78,  79,  80,  81,  82,
     83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
     93,  94,  95,  96,  97,  98,  99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
    133, 134, 135, 136, 137,
)

const ECHLEVS_ML137_CFL85 = (
      0,  12,  24,  28,  32,  36,  40,  44,  48,  52,
     53,  55,  57,  59,  61,  63,  65,  67,  69,  71,
     72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
     82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
     92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137,
)

const ECHLEVS_ML137_FULL = Tuple(0:ERA5_NATIVE_LEVEL_COUNT)

"Named ERA5 L137 interface selections accepted by preprocessing configs."
const ERA5_L137_ECHLEVS_PRESETS = (
    ml137_tropo34  = ECHLEVS_ML137_TROPO34,
    ml137_tropo34a = ECHLEVS_ML137_TROPO34A,
    ml137_66L      = ECHLEVS_ML137_66L,
    ml137_cfl94    = ECHLEVS_ML137_CFL94,
    ml137_cfl85    = ECHLEVS_ML137_CFL85,
    ml137_full     = ECHLEVS_ML137_FULL,
)

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
       SURFACE_AIR_TEMPERATURE_RANGE, SURFACE_HEAT_FLUX_MAX_ABS,
       ECHLEVS_ML137_TROPO34, ECHLEVS_ML137_TROPO34A, ECHLEVS_ML137_66L,
       ECHLEVS_ML137_CFL94, ECHLEVS_ML137_CFL85, ECHLEVS_ML137_FULL,
       ERA5_L137_ECHLEVS_PRESETS
