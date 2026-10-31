"""
    Physics

Runtime physics specs, topology traits, recipes and metadata. Independent of
model and simulation types; forcing refresh lives with the driven runtime.
"""
module Physics

using ...State
using ...Grids
using ...Operators
using ...MetDrivers
using ...ConfigChecks: config_bool as _config_bool, config_int
using ...Parameters: DEFAULT_LINROOD_PPM_ORDER

include("RuntimeRecipeStyles.jl")
include("RuntimePhysicsSpecs.jl")
include("RuntimePhysicsRecipe.jl")
include("RuntimePhysicsMetadata.jl")

end # module Physics
