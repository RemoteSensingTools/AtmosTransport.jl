# Common diagnostics; RG keeps its different outgoing-budget label and scan.

"""Construct a typed worst-window record with `N` location coordinates."""
init_positivity_accumulator(::Val{N}) where N =
    (ratio=0.0, direction=:none, win=0, location=ntuple(_ -> 0, N))

"""Fold a diagnostic into a worst-window record, preserving first-worst ties."""
function update_positivity_accumulator(worst::W, diag::NamedTuple,
                                       win_idx::Integer) where {W<:NamedTuple}
    diag.ratio > worst.ratio || return worst
    return W((Float64(diag.ratio),
              diag.direction === nothing ? :none : Symbol(diag.direction),
              Int(win_idx),
              typeof(worst.location)(diag.location)))
end

"""Log the worst-window outcome and enforce its configured positivity policy."""
function summarize_positivity_status(worst; budget_label::AbstractString,
                                      cfl_limit::Real,
                                      steps_per_window::Integer,
                                      require_substep_positivity::Bool=true,
                                      quarantine_path::Union{Nothing, AbstractString}=nothing)
    msg = @sprintf("max %s=%.3f dir=%s win=%d cell=%s (limit=%.2f)",
                   budget_label, worst.ratio, worst.direction, worst.win,
                   worst.location, cfl_limit)
    if worst.ratio <= cfl_limit
        @info "  Per-substep positivity gate: $msg"
        return nothing
    end

    max_safe_factor       = typemax(Int) ÷ max(Int(steps_per_window), 1)
    useful_recommendation = isfinite(worst.ratio) &&
                             isfinite(cfl_limit) && cfl_limit > 0 &&
                             worst.ratio / cfl_limit <= max_safe_factor

    detail = if useful_recommendation
        recommended = max(Int(steps_per_window),
                          ceil(Int, worst.ratio / cfl_limit) * Int(steps_per_window))
        "Per-substep positivity contract violated: $msg. " *
        "The binary stores fluxes that violate positivity at the recorded " *
        "`steps_per_window=$(steps_per_window)`. Re-run with " *
        "`steps_per_window=$(recommended)` (or higher) on the source " *
        "preprocessing config, or set `[numerics].require_substep_positivity = false` " *
        "to suppress."
    else
        "Per-substep positivity contract violated, and no representable " *
        "`steps_per_window` can rescue it: $msg. " *
        "The observed ratio is `Inf`/`NaN` (m<=0, NaN-mass/flux, or Inf-flux) " *
        "or finite but pathologically large; the configured `cfl_limit` may " *
        "also be non-positive. Either the source data has been corrupted " *
        "upstream of preprocessing, the pressure-fixer / endpoint-balance " *
        "pass drove a cell negative, or `[numerics].positivity_cfl_limit` is " *
        "misconfigured (must satisfy `0 < cfl_limit <= 1`). Investigate the " *
        "source field at the reported cell location, or set " *
        "`[numerics].require_substep_positivity = false` to record the " *
        "violation as a warning and keep the binary for diagnostic inspection."
    end
    if require_substep_positivity
        quarantine_path === nothing || (isfile(quarantine_path) && rm(quarantine_path; force = true))
        error(detail)
    else
        @warn detail
    end
    return nothing
end
