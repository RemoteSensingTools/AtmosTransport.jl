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

A `TimeVaryingSurfaceFluxSource` keeps its rate series in host memory. Its
device copy is a small window buffer (`FluxWindowBuffer`) holding only the
slices that the current met window reads; `stage_surface_flux_window!` fills it
when a window becomes current, and the kernels read the rates from it.

Emission totals for logs and budgets come from the operator's own temporal
reconstruction: `total_emission_rate(src, t)` (global storage rate at an
instant), `emission_segments(src, t, dt)` (the pieces one step applies) and
`integrated_emission(src, t0, t1, dt)` (the mass added over a run's steps).

# Surface layer convention

All kernels here assume `k = Nz` is the surface.
This matches `src/Models/DrivenSimulation.jl` and the LatLon
grid storage layout. A future `AbstractLayerOrdering{TopDown, BottomUp}`
refactor can generalise this.
"""
module SurfaceFlux

using Adapt
using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState, CubedSphereState, get_tracer, tracer_index, eachtracer,
                GCHPNonlocalPBLField
using ..Diffusion: ImplicitVerticalDiffusion, SplitSurfaceFluxCoupling,
                   DiffusiveSurfaceFluxBoundary
using ...MetDrivers: current_time
using ...Architectures: _two_sum, _neumaier_add, launch!, select_panel
import ..AbstractOperator, ..apply!

export SurfaceFluxSource, AbstractSurfaceFluxSource, TimeVaryingSurfaceFluxSource
export AbstractFluxTemporalScheme, StepwiseFlux, LinearInterpFlux, ConservativeMeanFlux
export flux_temporal_scheme
export PerTracerFluxMap, flux_for
export AbstractSurfaceFluxOperator, NoSurfaceFlux, SurfaceFluxOperator
export apply_surface_flux!, emitting_tracer_indices
export SurfaceLayerDeposit, ProfileDeposit, emission_deposit
export total_emission_rate, emission_segments, integrated_emission

include("sources.jl")
include("PerTracerFluxMap.jl")
include("surface_flux_kernels.jl")
include("operators.jl")
include("integrals.jl")

end # module SurfaceFlux
