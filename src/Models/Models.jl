"""
    Models

Minimal standalone runtime layer for `src`.
"""
module Models

using Adapt
using ..Architectures
using ..SectionTimer
using ..State
using ..Grids
using ..Operators
using ..MetDrivers
# Bring Regridding + Preprocessing into Models' namespace
# so the nested `InitialConditionIO` submodule can `using ..Regridding` etc.
# Regridding + Preprocessing are loaded before Models in AtmosTransport.jl.
using ..Regridding
using ..Preprocessing
# Output writers (SnapshotFrame / capture_snapshot / write_snapshot_netcdf)
# loaded by AtmosTransport.jl before Models — pull into Models' namespace
# so the nested DrivenRunner submodule can `using ..Output: …`.
using ..Output

using ..ConfigChecks: config_bool as _config_bool

include("TransportModel.jl")
include("RuntimeRecipeStyles.jl")    # runtime-style traits (dispatched on by specs)
include("RuntimePhysicsSpecs.jl")    # typed config specs + materialize (Oceananigans-style)
include("RuntimePhysicsRecipe.jl")
include("InitialConditionIO.jl")
using .InitialConditionIO: FileInitialConditionSource,
                           build_initial_mixing_ratio,
                           pack_initial_tracer_mass,
                           FileSurfaceFluxField,
                           build_surface_flux_source,
                           build_surface_flux_sources
export FileInitialConditionSource, build_initial_mixing_ratio, pack_initial_tracer_mass
export FileSurfaceFluxField, build_surface_flux_source, build_surface_flux_sources
include("BinaryPathExpander.jl")  # `[input]` folder+date-range
using .BinaryPathExpander: expand_binary_paths
export expand_binary_paths
include("InputStaging.jl")        # rolling NVMe staging of the daily binaries
using .InputStaging: InputStager, staged_path_for!, cleanup_staging!
export InputStager, staged_path_for!, cleanup_staging!
include("Simulation.jl")
include("DrivenSimulation.jl")
include("driven_window_state.jl")
include("driven_physics_refresh.jl")
include("driven_stepping.jl")
include("DrivenRunner.jl")        # library-level driven runner
using .DrivenRunner: run_driven_simulation, validate_config, TransportTracerSpec
export run_driven_simulation, validate_config, TransportTracerSpec

end # module Models
