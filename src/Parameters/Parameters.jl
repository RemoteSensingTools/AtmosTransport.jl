"""
    Parameters

Constants and small immutable parameter objects shared across the package: physical
constants, numerical defaults, met-source constants and planet parameters. Numbers
that several places use are defined here once.
"""
module Parameters

include("PhysicalConstants.jl")
include("NumericalDefaults.jl")
include("MetSourceConstants.jl")
include("PlanetParameters.jl")

end # module Parameters
