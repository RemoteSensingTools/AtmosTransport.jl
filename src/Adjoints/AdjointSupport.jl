# ---------------------------------------------------------------------------
# The adjoint setup guard.
#
# `adjoint_support(x)` says whether the cubed-sphere reverse pass has a record
# and an adjoint for a stage of the step plan, an operator or a scheme:
# `nothing` when it does, otherwise the reason it does not. Every adjoint
# entry point checks its whole step plan with `require_adjoint_support`
# before it records anything, so a configuration with an operator the tape
# cannot transpose is an error that names the operator, never a silent
# approximation. The fallbacks are "unsupported": a new stage, operator or
# scheme has no adjoint until it adds a method here (and the record and
# reverse kernel it needs).
# ---------------------------------------------------------------------------

"""
    adjoint_support(x) -> Union{Nothing, String}

`nothing` if the cubed-sphere adjoint records and transposes `x` (a step-plan
stage, an advection scheme or a diffusion, surface-flux, convection or
chemistry operator), otherwise the reason it does not. See
[`require_adjoint_support`](@ref).
"""
adjoint_support(x) = "$(nameof(typeof(x))) has no cubed-sphere adjoint."

# Stages of the step plan.
adjoint_support(::TransportStage) = nothing
adjoint_support(::ConvectionStage) = nothing
adjoint_support(::ChemistryBeforeStage) = nothing
adjoint_support(::ChemistryAfterStage) = nothing
adjoint_support(stage::AirMassResetStage) =
    stage.mode in (:preserve_tracer_mass, :preserve_vmr) ? nothing :
    "air-mass reset mode $(repr(stage.mode)) has no adjoint."
adjoint_support(stage::AbstractStepStage) =
    "step-plan stage $(nameof(typeof(stage))) has no adjoint record."

# Advection.
adjoint_support(::CSAdjointSupportedScheme) = nothing
adjoint_support(scheme::AbstractAdvectionScheme) = _adjoint_gap(scheme)

# Diffusion.
adjoint_support(::NoDiffusion) = nothing
function adjoint_support(op::ImplicitVerticalDiffusion)
    field = op.kz_field
    field isa GCHPNonlocalPBLField && return "ImplicitVerticalDiffusion with the GEOS-Chem " *
        "non-local PBL field has no adjoint (its counter-gradient emission profile is not transposed)."
    field isa AbstractCubedSphereField || field isa AbstractCSDkgField ||
        return "ImplicitVerticalDiffusion with a $(nameof(typeof(field))) Kz field has no " *
               "cubed-sphere adjoint; it needs a CubedSphereField Kz or a Dkg field."
    return nothing
end

# Surface fluxes: the emissions enter at the palindrome center, the footprint
# reads the adjoint state there.
adjoint_support(::NoSurfaceFlux) = nothing
adjoint_support(::SurfaceFluxOperator) = nothing

# Convection.
adjoint_support(::NoConvection) = nothing
function adjoint_support(op::CMFMCConvection)
    op.clamp && return "CMFMCConvection(clamp = true) has no adjoint: the positivity clamp " *
        "is nonlinear and its branch decisions are not taped."
    op.cloud_base isa CMFMCEdgeCloudBase ||
        return "CMFMCConvection with an archived cloud base has no adjoint (the adjoint " *
               "kernel derives the cloud base from CMFMC)."
    return nothing
end
function adjoint_support(op::TM5Convection)
    op.use_collab_lu || op.lmax_conv != 0 || op.n_merge != 1 || return nothing
    return "TM5Convection has an adjoint only for the full-column, unmerged solve " *
        "(use_collab_lu = false, lmax_conv = 0, n_merge = 1); got use_collab_lu = " *
        "$(op.use_collab_lu), lmax_conv = $(op.lmax_conv), n_merge = $(op.n_merge)."
end
adjoint_support(op::CMFMCMatrixConvection) = adjoint_support(op.inner)

# Chemistry.
adjoint_support(::NoChemistry) = nothing
adjoint_support(op::AbstractChemistryOperator) =
    "chemistry $(nameof(typeof(op))) has no cubed-sphere adjoint."

# The operators a stage applies.
_stage_operators(::TransportStage, ops::StepOperators) = (ops.advection, ops.diffusion, ops.emissions)
_stage_operators(::ConvectionStage, ops::StepOperators) = (ops.convection,)
_stage_operators(::Union{ChemistryBeforeStage, ChemistryAfterStage}, ops::StepOperators) =
    (ops.chemistry,)
_stage_operators(_stage, ::StepOperators) = ()

"""
    require_adjoint_support(plan::StepPlan, ops::StepOperators...) -> plan

Check that the cubed-sphere adjoint records and transposes every stage of
`plan` and every operator those stages apply (one `ops` per distinct operator
set, e.g. per step when the diffusion operator changes). Throws an
`ArgumentError` naming every unsupported piece. Every adjoint entry point
calls it before recording; `DrivenSimulation` and `TransportModel` setups can
call it directly (`require_adjoint_support(sim)`).
"""
function require_adjoint_support(plan::StepPlan, ops::StepOperators...)
    gaps = String[]
    for stage in plan_stages(plan)
        reason = adjoint_support(stage)
        reason === nothing || push!(gaps, reason)
        for o in ops, op in _stage_operators(stage, o)
            reason = adjoint_support(op)
            reason === nothing || reason in gaps || push!(gaps, reason)
        end
    end
    isempty(gaps) || throw(ArgumentError(
        "the cubed-sphere adjoint cannot replay this step plan: " * join(gaps, " ")))
    return plan
end

# The replay of an adjoint entry point: its plan with the operators of every
# step (the diffusion operator may change per step).
function require_adjoint_support(replay::_CSReplay)
    ops = [_replay_operators(replay, s, _sequence_at(replay.diffusion_op, s))
           for s in _distinct_diffusion_steps(replay)]
    require_adjoint_support(replay.plan, ops...)
    return replay
end

function _distinct_diffusion_steps(replay::_CSReplay)
    replay.diffusion_op isa AbstractVector || return 1:1
    seen = IdDict{Any, Int}()
    for s in 1:_replay_nsteps(replay)
        get!(seen, replay.diffusion_op[s], s)
    end
    return sort!(collect(values(seen)))
end
