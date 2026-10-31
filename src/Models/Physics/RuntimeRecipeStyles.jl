# Runtime-style traits — a lightweight tag the runtime physics recipe dispatches
# on to pick topology-specific construction rules (lat-lon, reduced Gaussian,
# cubed sphere). Kept in their own file, included BEFORE `RuntimePhysicsSpecs.jl`,
# because the spec `materialize` methods dispatch on these types. The
# `_runtime_recipe_style(grid/driver/reader)` resolvers live in `RuntimePhysicsRecipe.jl`
# (they need the grid/driver types) — only the bare trait types live here.

abstract type AbstractRuntimeRecipeStyle end
abstract type AbstractStructuredRuntimeRecipeStyle <: AbstractRuntimeRecipeStyle end

struct LatLonRuntimeRecipeStyle <: AbstractStructuredRuntimeRecipeStyle end
struct ReducedGaussianRuntimeRecipeStyle <: AbstractStructuredRuntimeRecipeStyle end
struct CubedSphereRuntimeRecipeStyle <: AbstractRuntimeRecipeStyle end

# Runtime config/metadata spelling; keep the existing default value and type.
const DEFAULT_AIR_MASS_RESET_MODE = "preserve_tracer_mass"
