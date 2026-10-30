# ---------------------------------------------------------------------------
# Change of variables between preconditioned χ-space
# and physical x-space for the CS 4D-Var surface-flux path.
#
# Two `optim_type` flavors:
#
#   Linear:     x = x_b + B^(1/2) χ
#                                      (the textbook 4D-Var ansatz —
#                                      x is unbounded.)
#
#   LogNormal:  x = x_b ⊙ exp(B^(1/2) χ)
#                                      (positivity-preserving — used
#                                      when the physical control must
#                                      remain strictly positive, e.g.
#                                      emission rates.)
#
# The preconditioner exposes two primitives:
#
#   * `apply_preconditioner!(x, prec, χ)`        — forward T(χ).
#   * `apply_preconditioner_adjoint!(g_χ, prec, x, g_phys)`
#                                                  — adjoint g_χ =
#                                                    T'(χ)^T · g_phys
#                                                    at base x. The
#                                                    `χ` part of the
#                                                    cost gradient
#                                                    (from `0.5 ‖χ‖²`)
#                                                    is added by the
#                                                    caller.
#
# The tangent/adjoint pair satisfies
#     ⟨T'(χ) δχ, g_phys⟩ = ⟨δχ, T'(χ)^T g_phys⟩
# which is the adjoint-identity test.
#
# For LogNormal the chain rule reads
#     ∂x_i/∂χ_j = x_i · L_{ij}
# so the tangent is `δx = x ⊙ (L δχ)` and the adjoint is
# `g_χ = L^T (x ⊙ g_phys)`. The base point `x` (not `χ`) is the
# natural argument because the adjoint is consumed in the cost
# pass after `x` is already known.
# ---------------------------------------------------------------------------

"""
    AbstractCSOptimType

Supertype for control-variable parameterizations used by the
preconditioner. Concrete subtypes:

- [`LinearOptimType`](@ref) — `x = x_b + B^(1/2) χ`. Unbounded.
- [`LogNormalOptimType`](@ref) — `x = x_b ⊙ exp(B^(1/2) χ)`.
  Strictly positive; requires `x_b > 0`.
"""
abstract type AbstractCSOptimType end

"""
    LinearOptimType

Linear additive parameterization: `x = x_b + B^(1/2) χ`. The physical
control `x` is unbounded; the only nonlinearity is in the cost via
the observation operator `H(x)`.
"""
struct LinearOptimType <: AbstractCSOptimType end

"""
    LogNormalOptimType

Log-normal multiplicative parameterization:
`x = x_b ⊙ exp(B^(1/2) χ)`. The physical control is guaranteed
strictly positive for any finite χ, at the cost of the background
`x_b` having to be strictly positive everywhere. Suitable for
emissions, mole fractions, and other non-negative atmospheric
quantities.
"""
struct LogNormalOptimType <: AbstractCSOptimType end

"""
    CSSurfaceFluxPreconditioner(covariance, background, optim_type)

Bundle of `(covariance, background, optim_type)` that performs the
change of variables between preconditioned χ-space and physical
x-space for one CS surface-flux control. `background` is the prior
`x_b` shaped as `NTuple{6, Matrix{FT}}` (matching the covariance's
sigma shape and a single `CSSurfaceFluxControl.value`).

Constructor validates `background` shapes against the covariance
and — for `LogNormalOptimType` — requires every `background` entry
to be strictly positive (else `log(x ./ x_b)` is undefined and the
log-normal transform is meaningless). It also pre-allocates a
panel-tuple scratch buffer reused by
the LogNormal variant of
`apply_preconditioner_adjoint!`. **The preconditioner is therefore
not thread-safe** — build one instance per worker thread if you
intend to call these methods concurrently.
"""
struct CSSurfaceFluxPreconditioner{FT, A, O <: AbstractCSOptimType,
                                   C <: AbstractCSSurfaceFluxCovariance{FT, A}}
    covariance::C
    background::NTuple{6, A}
    optim_type::O
    # Pre-allocated panel-tuple scratch reused by the LogNormal adjoint.
    # Avoids the per-call `ntuple(p -> ..., 6)` allocation that dominated
    # the χ-space gradient evaluation cost.
    panel_scratch::NTuple{6, A}
end

function CSSurfaceFluxPreconditioner(
        covariance::AbstractCSSurfaceFluxCovariance{FT, A},
        background::NTuple{6, A},
        optim_type::AbstractCSOptimType) where {FT, A}
    Nc = covariance.Nc
    @inbounds for p in 1:6
        size(background[p]) == (Nc, Nc) || throw(DimensionMismatch(
            "CSSurfaceFluxPreconditioner background panel $p has shape " *
            "$(size(background[p])); expected $((Nc, Nc)) to match the " *
            "covariance's `Nc`"))
        all(isfinite, background[p]) || throw(ArgumentError(
            "CSSurfaceFluxPreconditioner background panel $p has " *
            "non-finite entries"))
    end
    if optim_type isa LogNormalOptimType
        @inbounds for p in 1:6
            all(>(0), background[p]) || throw(ArgumentError(
                "CSSurfaceFluxPreconditioner with LogNormalOptimType " *
                "requires strictly positive background; panel $p " *
                "has a non-positive entry"))
        end
    end
    panel_scratch = ntuple(p -> similar(background[p]), 6)
    return CSSurfaceFluxPreconditioner{FT, A, typeof(optim_type),
                                       typeof(covariance)}(
        covariance, background, optim_type, panel_scratch)
end

# ---------------------------------------------------------------------------
# Forward: χ → x
# ---------------------------------------------------------------------------

"""
    apply_preconditioner!(x, prec, chi) -> x

Forward change of variables. With Linear `optim_type`,
`x = x_b + B^(1/2) χ`. With LogNormal, `x = x_b ⊙ exp(B^(1/2) χ)`.
`x`, `chi` are `NTuple{6, AbstractMatrix}`.
"""
function apply_preconditioner!(x::NTuple{6, A},
                                prec::CSSurfaceFluxPreconditioner{FT, A, LinearOptimType},
                                chi::NTuple{6, A}) where {FT, A}
    apply_B_half!(x, prec.covariance, chi)
    @inbounds for p in 1:6
        @. x[p] += prec.background[p]
    end
    return x
end

function apply_preconditioner!(x::NTuple{6, A},
                                prec::CSSurfaceFluxPreconditioner{FT, A, LogNormalOptimType},
                                chi::NTuple{6, A}) where {FT, A}
    apply_B_half!(x, prec.covariance, chi)
    @inbounds for p in 1:6
        @. x[p] = prec.background[p] * exp(x[p])
    end
    return x
end

# ---------------------------------------------------------------------------
# Adjoint: g_phys → g_chi at base x
# ---------------------------------------------------------------------------

"""
    apply_preconditioner_adjoint!(g_chi, prec, x, g_phys) -> g_chi

Adjoint of the tangent at base `x`. Linear:
`g_χ = (B^(1/2))^T g_phys`. LogNormal: `g_χ = (B^(1/2))^T (x ⊙ g_phys)`.

This is the observation part of the χ-space gradient under the
chain rule of `T`. The caller adds `χ` from the `0.5 ‖χ‖²` term
of the preconditioned cost separately.
"""
function apply_preconditioner_adjoint!(g_chi::NTuple{6, A},
                                       prec::CSSurfaceFluxPreconditioner{FT, A, LinearOptimType},
                                       x::NTuple{6, A},
                                       g_phys::NTuple{6, A}) where {FT, A}
    apply_B_half_adjoint!(g_chi, prec.covariance, g_phys)
    return g_chi
end

function apply_preconditioner_adjoint!(g_chi::NTuple{6, A},
                                       prec::CSSurfaceFluxPreconditioner{FT, A, LogNormalOptimType},
                                       x::NTuple{6, A},
                                       g_phys::NTuple{6, A}) where {FT, A}
    @inbounds for p in 1:6
        @. prec.panel_scratch[p] = x[p] * g_phys[p]
    end
    apply_B_half_adjoint!(g_chi, prec.covariance, prec.panel_scratch)
    return g_chi
end
