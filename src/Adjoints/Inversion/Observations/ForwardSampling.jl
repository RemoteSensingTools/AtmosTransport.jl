# One forward replay sampling `observations` after each step of their time
# sampling (`CSTimeSampling`): their simulated values `H = Σ_s ω_s h(r_s)`,
# their stencils (`CSObservationStencil`: steps, weights and the level read at
# each step) and, with `build_functionals = true`, their linear functionals:
# each sampled step's `_linear_functional` (weights from the replayed air mass
# of that state) scaled by `ω_s`. Placed point observations choose their layer
# at each sampled state from `column_met` (`_sample_objective`), and
# averaging-kernel columns build their weights there; a column whose
# time-weighted surface-pressure mismatch exceeds its limit is an error or is
# marked not assimilated in its stencil (`_screen_surface_pressure`). The
# replay stops after the last sampled step.
#
# An instantaneous observation (one step, ω = 1) gets exactly the value and
# functional it had before time sampling existed: `0 + 1·h` and `1·w` are
# exact.
function _observe_cs_forward(panels_rm0, panels_m0,
                             panels_am_steps,
                             panels_bm_steps,
                             panels_cm_steps,
                             mesh::CubedSphereMesh,
                             observations;
                             scheme = PPMScheme(NoLimiter()),
                             column_met = nothing,
                             build_functionals::Bool,
                             kwargs...)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    Nz = size(panels_rm0[1], 3)
    @inbounds for obs in observations
        last(obs.sampling.steps) <= nsteps || throw(ArgumentError(
            "CSObservation samples step $(last(obs.sampling.steps)), outside 1:$nsteps"))
        _validate_objective(obs.objective, mesh, Nz)
    end
    _validate_column_met(observations, column_met, mesh, Nz, nsteps)

    # (observation, sample) pairs by step, in observation order.
    samples_by_step = [Tuple{Int, Int}[] for _ in 1:nsteps]
    @inbounds for idx in eachindex(observations), (n, step) in enumerate(observations[idx].sampling.steps)
        push!(samples_by_step[step], (idx, n))
    end
    FT = eltype(panels_rm0[1])
    nobs = length(observations)
    simulated = zeros(FT, nobs)
    levels = [zeros(Int, length(obs.sampling.steps)) for obs in observations]
    # Functional entries per observation (steps, panels, i, j, k, weights) and
    # its offset, the time-sampling-weighted sum of the steps' offsets.
    entries = build_functionals ?
        [(Int[], Int[], Int[], Int[], Int[], FT[]) for _ in 1:nobs] : nothing
    offsets = zeros(FT, nobs)
    # Averaging-kernel columns: time-weighted surface-pressure mismatch and
    # rebinned profile (NaN and empty for other observations).
    mismatch = fill(NaN, nobs)
    profiles = [Float64[] for _ in 1:nobs]
    last_step = maximum(obs -> obs.step, observations; init = 0)

    if last_step > 0
        replay, _ = _cs_replay_for_call(panels_rm0, panels_m0, panels_am_steps, panels_bm_steps,
                                        panels_cm_steps, mesh; scheme, nsteps = last_step,
                                        kwargs...)
        sample! = (step, _state) -> begin
            panels_rm = _replay_rm(replay)
            panels_m = _replay_m(replay)
            for (obs_idx, n) in samples_by_step[step]
                obs = observations[obs_idx]
                weight = FT(obs.sampling.weights[n])
                value, f, level, sample = _sample_objective(obs.objective, step, panels_rm, panels_m,
                                                            mesh, column_met, build_functionals)
                simulated[obs_idx] += weight * value
                levels[obs_idx][n] = level
                _accumulate_retrieval_sample!(mismatch, profiles, obs_idx, sample,
                                              obs.sampling.weights[n])
                if build_functionals
                    _append_entries!(entries[obs_idx], f, weight)
                    offsets[obs_idx] += weight * f.offset
                end
            end
            return nothing
        end
        _replay_steps!(replay, 1:last_step; on_step_end = sample!)
    end

    assimilated = _screen_surface_pressure(observations, mismatch)
    stencils = [_observation_stencil(observations[i], levels[i], mismatch[i], profiles[i], assimilated[i])
                for i in 1:nobs]
    functionals = build_functionals ?
        [CSLinearFunctional{FT}(entries[i]..., offsets[i]) for i in 1:nobs] : CSLinearFunctional{FT}[]
    return (; simulated, functionals, stencils)
end

# Append one sampled step's functional entries, their weights scaled by the
# sample's time-sampling weight.
function _append_entries!(entries, f::CSLinearFunctional{FT}, weight::FT) where {FT}
    steps, panel, i, j, k, w = entries
    append!(steps, f.steps)
    append!(panel, f.panel)
    append!(i, f.i)
    append!(j, f.j)
    append!(k, f.k)
    append!(w, weight .* f.weight)
    return entries
end
