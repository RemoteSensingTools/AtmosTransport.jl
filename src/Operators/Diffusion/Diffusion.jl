"""
    Diffusion

Vertical-diffusion operator hierarchy and solver infrastructure.

Public surface:

- [`NoDiffusion`](@ref) — identity no-op; the default when no
  `[diffusion]` config section is present.
- [`ImplicitVerticalDiffusion`](@ref) — Backward-Euler implicit
  diffusion driven by an `AbstractTimeVaryingField` Kz. Wired into
  the Strang palindrome via [`apply_vertical_diffusion_vmr!`](@ref) and
  installed into `TransportModel.diffusion` by the runtime recipe
  when `[diffusion] kind = "constant"`.

Both subtype the global `AbstractDiffusion` declared in
`src/Operators/AbstractOperators.jl`; concrete operator structs live in
`operators.jl`. The column-level Thomas solve (`solve_tridiagonal!`) is
exposed as the numerical reference. Generic Kz kernels name the tridiagonal
coefficients `(a, b, c)` explicitly. Precomputed Dkg mass transport uses
column-conservative bidiagonal factors in `conservative_dkg.jl`; both matching
transposes live in `src/Adjoints/DiffusionAdjoint.jl`.
"""
module Diffusion

using Adapt
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...State: CellState, CubedSphereState,
                AbstractTimeVaryingField, AbstractCubedSphereField,
                AbstractCSDkgField, PrecomputedCSDkgField,
                field_value, update_field!, panel_field, eachtracer, ntracers
using ...MetDrivers: current_time
using ...Architectures: _two_sum, _neumaier_add, _ledger_residual
using ...Parameters: VIRTUAL_TEMPERATURE_FACTOR
import ..apply!
import ..AbstractDiffusion                # global root from src/Operators/AbstractOperators.jl

export solve_tridiagonal!
export AbstractDiffusion, NoDiffusion, ImplicitVerticalDiffusion
export DiffusionWorkspace
export AbstractSurfaceFluxCoupling, SplitSurfaceFluxCoupling,
       DiffusiveSurfaceFluxBoundary, uses_diffusive_surface_flux_boundary
export apply_vertical_diffusion_vmr!
export fill_dz_hydrostatic_constT!, fill_dz_hydrostatic_virtualT!

include("thomas_solve.jl")
include("diffusion_kernels.jl")
include("conservative_dkg.jl")
include("dz_helpers.jl")
include("workspace.jl")
include("operators.jl")

end # module Diffusion
