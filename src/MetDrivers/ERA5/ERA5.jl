"""
    ERA5 met driver

ERA5-specific dry flux building from spectral, gridded, and native-GRIB fields.
"""
module ERA5

using ...Grids: ReducedGaussianMesh, b_diff
using ...Architectures: _kahan_add
using ...Parameters: EARTH_RADIUS

include("NativeGRIBGeometry.jl")
include("VerticalClosure.jl")

end # module ERA5
