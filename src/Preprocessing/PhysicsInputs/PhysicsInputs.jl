module PhysicsInputs

using Dates
using NCDatasets
using Logging
using Printf: @sprintf
import ...expand_data_path
using ...Parameters: STANDARD_GRAVITY, TM5_CONSTANTS, R_DRY_AIR, VIRTUAL_TEMPERATURE_FACTOR,
    ERA5_MET_INTERVAL_SECONDS, SURFACE_AIR_TEMPERATURE_RANGE, SURFACE_HEAT_FLUX_MAX_ABS,
    TM5_CONVECTION_SOURCE_KEY, TM5_EC2TM_CONVENTION_KEY, TM5_EC2TM_CONVENTION
using ...Grids: LatLonMesh
using ...State: DryBasis, MoistBasis
using ...GriddedIO: source_latlon_mesh, SnappedFaces
using ..Spectral: normalize_lon_to_centered
export BLDiffColumnScratch,
    BLDiffConstants,
    BLDiffDiag,
    TM5CleanupStats,
    dz_hydrostatic_virtual!,
    ec2tm_from_rates!,
    ecmwf_convection_to_basis!,
    era5_surface_source_mesh,
    era5_tm5_convection_header,
    interface_specific_humidity,
    load_era5_surface_window,
    tm5_bldiff_center_kz_column!,
    tm5_bldiff_dkg_column!,
    tm5_bldiff_kvh_column!,
    tm5_convection_to_basis!

include("convection_mass_basis.jl")
include("tm5_convection_conversion.jl")
include("era5_surface_reader.jl")
include("tm5_bldiff.jl")

end
