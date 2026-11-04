"""
    Physics

Runtime physics specs, topology traits, recipes and metadata. Independent of
model and simulation types; forcing refresh lives with the driven runtime.
"""
module Physics

using ...State
using ...Grids
using ...Grids: cs_definition, cs_face_edge_lengths, panel_cell_local_tangent_basis
using ...Operators
import ...Operators.StepPlans
using ...Operators.Diffusion: TM5CalcKzzField, TM5CalcKzzCSField, TM5LLMetrics, TM5RGMetrics, TM5CSMetrics
using ...MetDrivers.BinaryFormat: TM5_PBL_PAYLOAD_SECTIONS, section_shape
using ...MetDrivers
using ...ConfigChecks: config_bool as _config_bool, config_int, config_float, parse_choice
using ...Parameters: DEFAULT_LINROOD_PPM_ORDER

include("RuntimeRecipeStyles.jl")
include("RuntimePhysicsSpecs.jl")
include("RuntimePhysicsRecipe.jl")
include("RuntimePhysicsMetadata.jl")

end # module Physics
