# Independent reference for the one mixed-observation single-sweep and FD
# gate. Included after cs_observation_columns.jl and the AT/ADJ/O aliases.
# Layer columns use the TM5 transcription; level columns use independent hat
# integration. Reverse seeds call the production footprint-from-seed API.
include(joinpath(@__DIR__, "oco2_mip_fixture.jl"))

function _tm5_oco2(model_profile, tm5_pres_levels, levels, ak, prior; h = nothing)
    nmod = length(model_profile)
    noco = length(ak)
    m = (levels[1] - levels[noco + 1]) / (tm5_pres_levels[1] - tm5_pres_levels[nmod + 1])
    c1 = levels[1] - m * tm5_pres_levels[1]
    c2 = levels[noco + 1] - m * tm5_pres_levels[nmod + 1]
    c = 0.5 * (c1 + c2)
    in_x = m .* tm5_pres_levels .+ c
    out_x = vcat(in_x[1], levels[2:noco], in_x[end])
    R = zeros(noco, nmod)
    for i in 1:noco, j in 1:nmod
        ((in_x[j + 1] >= out_x[i]) || (out_x[i + 1] >= in_x[j])) && continue
        ub = min(out_x[i], in_x[j])
        lb = max(in_x[j + 1], out_x[i + 1])
        R[i, j] = (ub - lb) / (out_x[i] - out_x[i + 1])
    end
    pres_weights = h === nothing ? (levels[1:noco] .- levels[2:(noco + 1)]) ./ levels[1] : h
    W = pres_weights .* ak
    squeezed = R * model_profile
    model_tc = dot(W, squeezed) + dot(pres_weights .- W, prior)
    return (; model_tc, H = vec(W' * R), R, squeezed, stretched = in_x)
end

function _level(ps; tilt = 0.0, sigma = Float64[], mapping = O.HatLevelProjection(), prior_column = NaN)
    r = oco2_fixture_retrieval(ps / 100; tilt)
    column = O.RetrievalColumn(; pressure_pa = reverse(OCO2_SIGMA .* ps), pressure_weight = reverse(r.pw),
                               averaging_kernel = reverse(r.ak), prior_profile = reverse(r.prior) .* 1e-6,
                               profile_sigma = sigma, mapping = O.HatLevelProjection(), prior_column)
    return O.oco2_level_form(mapping, column)
end

function _replayed_states(P, controls; scheme, physics)
    rates = ADJ.Inversion.Solver._surface_rates_from_controls(controls, P.nsteps, P.mesh, P.rm[1])
    replay, _ = ADJ.Footprint._cs_replay_for_call(P.rm, P.m, P.am, P.bm, P.cm, P.mesh; scheme, dt = DT,
                                        emission_rates = rates, physics(P)...)
    states = []
    ADJ.Footprint._replay_steps!(replay, 1:P.nsteps; on_step_end = (s, _) -> push!(states,
        (m = map(copy, ADJ.Footprint._replay_m(replay)), rm = map(copy, ADJ.Footprint._replay_rm(replay)))))
    return states, rates
end

# Per observation and sampled step: the weights w_k on the tracer mass of its
# cell (an averaging-kernel column from the TM5 transcription on the replayed
# total interfaces, a column mean 1/Σm, a layer 1/max(m, ε)), the value, and the
# footprint of that seed on a run truncated to the step.
function _reference(P, observations, stencils, controls; scheme, physics)
    FT = P.FT
    states, rates = _replayed_states(P, controls; scheme, physics)
    Hp = P.mesh.Hp
    floor = eps(FT)
    gradients = [ntuple(_ -> zeros(FT, P.Nc, P.Nc), 6) for _ in controls]
    simulated = FT[]
    mismatch = Float64[]
    profiles = Vector{Float64}[]
    for (obs, stencil) in zip(observations, stencils)
        o = obs.objective
        parts = map(zip(stencil.steps, stencil.weights, stencil.levels)) do (s, ω, level)
            m = states[s].m[o.panel][Hp + o.i, Hp + o.j, :]
            rm = states[s].rm[o.panel][Hp + o.i, Hp + o.j, :]
            dps, profile = NaN, Float64[]
            w, offset = if o isa AT.CSRetrievalColumnObjective
                p_half = O.total_interface_pressures!(zeros(P.Nz + 1), m, COLUMN_Q, o.area, COLUMN_GRAVITY, P.p_top)
                rc = o.retrieval
                x = Float64.(rm) ./ max.(Float64.(m), Float64(floor))
                H, profile = if rc.mapping isa O.HatLevelProjection
                    # Independent hat integration, never production retrieval_profile!.
                    proj = hat_projection_reference(p_half, Float64.(m), rc.pressure_pa).P
                    (vec((rc.pressure_weight .* rc.averaging_kernel)' * proj), proj * x)
                else
                    # Independent TM5 rebin, including OCO-2's 19-layer mode.
                    t = _tm5_oco2(reverse(x), reverse(p_half), rc.pressure_pa,
                                  rc.averaging_kernel, rc.prior_profile; h = rc.pressure_weight)
                    (reverse(t.H), t.squeezed)
                end
                dps = p_half[end] - rc.surface_pressure_pa
                (FT.(H ./ max.(Float64.(m), Float64(floor))), FT(O.retrieval_prior_offset(rc)))
            elseif o isa AT.CSColumnMeanObjective
                (fill(one(FT) / sum(m), P.Nz), zero(FT))
            else
                ([k == level ? one(FT) / max(m[k], floor) : zero(FT) for k in 1:P.Nz], zero(FT))
            end
            seed = ntuple(q -> zeros(FT, size(P.rm[q])), 6)
            seed[o.panel][Hp + o.i, Hp + o.j, :] .= w
            footprint = AT.cs_surface_emission_footprint_from_seed(seed, P.m, P.am[1:s], P.bm[1:s], P.cm[1:s],
                                                                   P.mesh; scheme, dt = DT, base_panels_rm0 = P.rm,
                                                                   base_emission_rates = rates[1:s], physics(P)...)
            (; s, ω, value = offset + sum(w .* rm), footprint, dps, profile)
        end
        H = sum(part.ω * part.value for part in parts)
        push!(simulated, H)
        push!(mismatch, sum(part.ω * part.dps for part in parts))
        push!(profiles, o isa AT.CSRetrievalColumnObjective ? sum(part.ω .* part.profile for part in parts) : Float64[])
        obs.assimilate || continue
        scale = (H - obs.value) / obs.sigma^2
        for part in parts, (c, control) in enumerate(controls)
            for (idx, step) in enumerate(control.window.steps)
                step <= part.s || continue
                weight = FT(scale) * FT(part.ω) * FT(control.window.weights[idx])
                for q in 1:6
                    gradients[c][q] .+= weight .* part.footprint.footprints[step][q]
                end
            end
        end
    end
    return (; simulated, gradients, mismatch, profiles)
end

function _mixed_observations(P; kwargs...)
    a, b = _site(P), _site(P; panel = 5, i = 3, j = 1)
    records = [_record(1; a..., t = 100.0, value = 402e-6, retrieval = _retrieval(97000.0)),
               _record(2; a..., t = 1000.0, time_window_s = 1500.0, value = 399e-6,
                       retrieval = _retrieval(101500.0; top = 2000.0, n = 8, sigma = fill(3e-6, 8))),
               _record(3; b..., t = 700.0, time_window_s = 900.0, value = 404e-6,
                       retrieval = _retrieval(99500.0; top = 50.0, n = 4, ak = [0.5, 0.9, 1.1, 1.3])),
               _record(4; b..., t = 1500.0, value = 405e-6),
               _record(5; a..., t = 1700.0, kind = "point", intake_height_m = 10.0, value = 430e-6)]
    append!(records, [
        _record(6; a..., kind = "point", t = 100.0, intake_height_m = 10.0, value = 420e-6),
        _record(7; a..., kind = "point", t = 1000.0, altitude_m = 5200.0, elevation_m = 300.0, value = 401e-6),
        _record(8; b..., kind = "point", t = 700.0, pressure_pa = 70000.0, time_window_s = 1500.0, value = 399e-6),
        _record(9; b..., t = 1500.0, time_window_s = 900.0, value = 404e-6, model_sigma = 0.5e-6),
        _record(10; a..., kind = "point", t = 1700.0, intake_height_m = 10.0, time_window_s = 600.0, value = 430e-6),
        _record(11; a..., t = 100.0, value = 402e-6, retrieval = _level(97000.0; prior_column = 404.2e-6)),
        _record(12; a..., t = 1000.0, time_window_s = 1500.0, value = 399e-6,
                retrieval = _level(101500.0; tilt = 0.3, sigma = fill(1e-6, 20))),
        _record(13; b..., t = 700.0, time_window_s = 900.0, value = 404e-6, retrieval = _level(99500.0; tilt = -0.2)),
        _record(14; b..., t = 1500.0, value = 405e-6, retrieval = _level(98500.0; mapping = O.LayerRebin())),
        _record(15; a..., t = 1700.0, value = 403e-6),
    ])
    return _bind(P, records; kwargs...)
end
