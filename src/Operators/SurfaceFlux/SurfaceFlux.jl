"""
    SurfaceFlux

Surface emission operators.

Ships the data types and helpers needed to apply per-tracer surface
sources to a `CellState`:

- `SurfaceFluxSource{RateT}` — single-tracer source plus a rate array in
  model-storage units per second per cell, also available from the top-level
  `AtmosTransport` module. Physical inventories in kg species/s are converted
  to these units by the runtime's basis-aware input builders.
- `PerTracerFluxMap{S}` — NTuple-backed map of
  `SurfaceFluxSource`s, keyed by `tracer_name`. Ships with the
  `flux_for(map, :name)` lookup helper. Storage-bits-stable on GPU.

The `AbstractSurfaceFluxOperator` hierarchy (`NoSurfaceFlux`,
`SurfaceFluxOperator`), the `_surface_flux_kernel!` KA kernel, and the
`apply!` / `apply_surface_flux!` entry points live alongside these data
types.

# Surface layer convention

All kernels here assume `k = Nz` is the surface.
This matches `src/Models/DrivenSimulation.jl` and the LatLon
grid storage layout. A future `AbstractLayerOrdering{TopDown, BottomUp}`
refactor can generalise this.
"""
module SurfaceFlux

using Adapt
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState, CubedSphereState, get_tracer, tracer_index, eachtracer,
                GCHPNonlocalPBLField
using ..Diffusion: ImplicitVerticalDiffusion
using ...MetDrivers: current_time
using ...Architectures: _two_sum, launch!
import ..AbstractOperator, ..apply!

export SurfaceFluxSource, AbstractSurfaceFluxSource, TimeVaryingSurfaceFluxSource
export AbstractFluxTemporalScheme, StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux
export flux_temporal_scheme
export PerTracerFluxMap, flux_for
export AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator
export apply_surface_flux!, emitting_tracer_indices
export SurfaceLayerDeposit, ProfileDeposit, emission_deposit

include("sources.jl")
include("PerTracerFluxMap.jl")
include("surface_flux_kernels.jl")
include("operators.jl")

end # module SurfaceFlux
