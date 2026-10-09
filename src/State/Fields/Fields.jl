"""
    Fields

Time-varying field abstraction.

`AbstractTimeVaryingField{FT, N}` is the common interface presented
to physics operators for rate-like and field-like inputs (chemistry
decay rates, Kz diffusion fields, ...). Concrete types back different
sources — constant, stepwise from file, derived from meteorology —
with the same kernel code downstream.

Required interface (every concrete type):
- `field_value(f, idx::NTuple{N, Int}) -> FT` — kernel-safe read
- `update_field!(f, t::Real) -> f` — refresh caches for time `t`

Concrete types:
- `ConstantField{FT, N}` — one scalar presented as a rank-N field.
- `ProfileKzField{FT}` — rank-2/3 vertical profile, horizontally uniform.
- `PreComputedKzField{FT, A}` — rank-2/3 wrapper over a spatial array.
- `DerivedKzField{FT, ...}` — Beljaars-Viterbo Kz from surface fields.
- `WindowPBLKzField{FT, ...}` — cubed-sphere per-window Beljaars-Viterbo Kz.
- `LocalHoltslagBovilleKzField{FT, ...}` — cubed-sphere GEOS VDIFF local-Kz cache.
- `GCHPNonlocalPBLField{FT, ...}` — cubed-sphere GEOS-Chem non-local PBL exchange.
- `StepwiseField{FT, N, A, B, W}` — piecewise-constant in time at any rank.

`PBLPhysicsParameters{FT}` carries the physical constants for
`DerivedKzField`.
"""
module Fields

using Adapt
using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
using ...Parameters: STANDARD_GRAVITY, CP_DRY_AIR, GEOSCHEM_CONSTANTS, VIRTUAL_TEMPERATURE_FACTOR

export AbstractTimeVaryingField, AbstractCubedSphereField
export ConstantField, ProfileKzField, PreComputedKzField, CubedSphereField
export DerivedKzField, WindowPBLKzField, LocalHoltslagBovilleKzField
export AbstractCSDkgField, PrecomputedCSDkgField
export GCHPNonlocalPBLField, GCHPVdiffParameters, refresh_gchp_nonlocal_pbl!
export PBLPhysicsParameters, StepwiseField
export field_value, update_field!, integral_between, panel_field
export refresh_pbl_kz_cache!, refresh_local_holtslag_boville_kz_cache!,
       refresh_precomputed_cs_dkg_cache!

# =========================================================================
# Abstract type
# =========================================================================

"""
    AbstractTimeVaryingField{FT, N}

Common interface for rate-like / field-like inputs to physics
operators. `FT <: AbstractFloat` is the element type; `N :: Int` is
the spatial rank (0 = scalar, 2 = surface, 3 = volume).

See [`field_value`](@ref) and [`update_field!`](@ref) for the
required methods.
"""
abstract type AbstractTimeVaryingField{FT, N} end

"""
    field_value(f::AbstractTimeVaryingField{FT, N}, idx::NTuple{N, Int}) -> FT

Return the field's current value at spatial index `idx`.

**Contract (kernel-safe):** allocation-free, type-stable, pure with
respect to `f`. Called from inside KernelAbstractions kernels. For
`N = 0`, `idx` is the empty tuple `()`.

Every concrete subtype of `AbstractTimeVaryingField` must implement
a method of `field_value`.
"""
function field_value end

"""
    update_field!(f::AbstractTimeVaryingField, t::Real) -> f

Refresh any caches so that subsequent `field_value` calls return the
field's value at simulation time `t`. Runs on the host (not kernel-
safe itself); may be expensive for derived fields. Called once per
`apply!` by an operator, before any kernel launch that reads the
field.

For time-independent fields (e.g. `ConstantField`), this is a no-op.
Returns `f` for chaining.
"""
function update_field! end

# =========================================================================
# ConstantField — the minimum concrete type
# =========================================================================

"""
    ConstantField{FT, N}(value)

A scalar `value` presented as an `AbstractTimeVaryingField{FT, N}`.
`field_value` ignores its index and returns `value`. `update_field!`
is a no-op. Storage is one scalar; backend-agnostic by construction.

# Examples
```julia
rate    = ConstantField{Float64, 0}(2.098e-6)   # chemistry decay rate
kz_test = ConstantField{Float32, 3}(1.0f0)      # idealized diffusion Kz
```
"""
struct ConstantField{FT <: AbstractFloat, N} <: AbstractTimeVaryingField{FT, N}
    value :: FT
end

@inline field_value(f::ConstantField{FT, N}, ::NTuple{N, Int}) where {FT, N} = f.value

update_field!(f::ConstantField, ::Real) = f

# =========================================================================
# Rank-3 concrete types
# =========================================================================

include("ProfileKzField.jl")
include("PreComputedKzField.jl")
include("CubedSphereField.jl")
include("DerivedKzField.jl")
include("WindowPBLKzField.jl")
include("LocalHoltslagBovilleKzField.jl")
include("PrecomputedCSDkgField.jl")
include("GCHPNonlocalPBLField.jl")
include("StepwiseField.jl")

end # module Fields
