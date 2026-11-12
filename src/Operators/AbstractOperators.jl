# ---------------------------------------------------------------------------
# Abstract operator hierarchy for the basis-explicit transport architecture.
#
# All physics operators dispatch on these abstract types and share the mutating
# function `apply!`. Its second positional argument is family-specific: face
# fluxes for advection, `ConvectionForcing` for convection, and meteorology or
# clock data for diffusion, surface fluxes, and chemistry. There is therefore
# no universal positional forcing type beyond the operator root itself.
#
# Concrete state, flux storage, and kernel strategies are selected by dispatch
# on the state and grid types. Raw meteorology is interpreted at the driver or
# family boundary rather than inside transport kernels.
#
# Diffusion and convection roots are declared here because those modules
# extend them directly. Advection, chemistry, and surface-flux modules declare
# their own family roots as subtypes of AbstractOperator. Each family has one
# root and one apply! protocol.
# ---------------------------------------------------------------------------

"""
    AbstractOperator

Root type for all physics operators in the transport model.
"""
abstract type AbstractOperator end

"""
    AbstractDiffusion <: AbstractOperator

Root type for vertical diffusion operators. Concrete subtypes live in
`src/Operators/Diffusion/operators.jl` (`NoDiffusion`,
`ImplicitVerticalDiffusion`, …).
"""
abstract type AbstractDiffusion <: AbstractOperator end

"""
    AbstractConvection <: AbstractOperator

Root type for convective-transport operators. Concrete subtypes live in
`src/Operators/Convection/` (`NoConvection`, `CMFMCConvection`,
`CMFMCMatrixConvection`, `TM5Convection`, `IFSConvection`).
"""
abstract type AbstractConvection <: AbstractOperator end

# ---------------------------------------------------------------------------
# Default error stubs
# ---------------------------------------------------------------------------

function apply! end

export AbstractOperator
export AbstractDiffusion, AbstractConvection
export apply!
