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
`operators.jl`. Generic Kz kernels name the tridiagonal
coefficients `(a, b, c)` explicitly. Precomputed Dkg mass transport uses
column-conservative bidiagonal factors in `conservative_dkg.jl`; both matching
transposes live in `adjoint_kernels.jl`.
"""
module Diffusion

using ...Architectures: host_array, zeros_like
using Adapt
using KernelAbstractions: @kernel, @index, @Const, @atomic, get_backend, synchronize
using ...State: CellState, CubedSphereState,
                AbstractTimeVaryingField, AbstractCubedSphereField,
                AbstractCSDkgField, PrecomputedCSDkgField, GCHPNonlocalPBLField, GCHPVdiffParameters,
                ConstantField, PreComputedKzField, ntracers
import ...State: field_value, update_field!, panel_field
using ...MetDrivers: current_time
using ...State: DryBasis, MoistBasis
using ...Architectures: _two_sum, _neumaier_add, _kahan_add, _ledger_residual, _ledger_deposit,
                       _load_ledger_carry, _store_ledger_carry!, launch!, select_panel
using ...Parameters: VIRTUAL_TEMPERATURE_FACTOR, R_DRY_AIR, STANDARD_GRAVITY,
                     HYDROSTATIC_REFERENCE_TEMPERATURE, CP_DRY_AIR, CP_OVER_R_DIATOMIC,
                     THETA_REFERENCE_PRESSURE, GEOSCHEM_CONSTANTS, VON_KARMAN,
                     STANDARD_SEA_LEVEL_AIR_DENSITY, TM5_BLDIFF_CONSTANTS, TM5_GPH_CONSTANTS,
                     TM5_PBL_REFERENCE_PRESSURE, TM5_PBL_CUTOFF_PRESSURE
import ..apply!
import ..AbstractDiffusion                # global root from src/Operators/AbstractOperators.jl

export AbstractDiffusion, NoDiffusion, ImplicitVerticalDiffusion
export DiffusionWorkspace
export AbstractSurfaceFluxCoupling, SplitSurfaceFluxCoupling,
       DiffusiveSurfaceFluxBoundary, uses_diffusive_surface_flux_boundary
export apply_vertical_diffusion_vmr!
export HybridCoefficientWorkspace, stage_hybrid_coefficients!
export fill_dz_hydrostatic_constT_staged!, fill_dz_hydrostatic_virtualT_staged!
export recover_total_surface_pressure_staged!
export check_diffusion_workspace, preflight_diffusion

export TM5CalcKzzField, TM5BLDiffParameters, refresh_tm5_calc_kzz!, tm5_calc_kzz_column!
export refresh_gchp_nonlocal_pbl!
export WindowPBLKzField, LocalHoltslagBovilleKzField, PBLPhysicsParameters
export refresh_pbl_kz_cache!, refresh_local_holtslag_boville_kz_cache!

include("pbl_physics.jl")
include("WindowPBLKzField.jl")
include("LocalHoltslagBovilleKzField.jl")
include("window_kz_cache.jl")
include("tm5_calc_kzz.jl")
include("TM5CalcKzzField.jl")
include("gchp_nonlocal_pbl.jl")

include("diffusion_kernels.jl")
include("conservative_dkg.jl")
include("workspace.jl")
include("dz_helpers.jl")
include("operators.jl")
include("workspace_validation.jl")
include("adjoint_kernels.jl")

end # module Diffusion
