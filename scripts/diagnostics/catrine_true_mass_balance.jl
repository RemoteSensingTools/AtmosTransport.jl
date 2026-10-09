#!/usr/bin/env julia
# =============================================================================
# catrine_true_mass_balance.jl — true global mass balance of a multi-day run
# =============================================================================
#
# For every output time t and tracer, the residual
#
#     r(t) = B(t) − B(t₀) − ∫_{t₀}^{t} E dt
#
# is the mass the model gained (r > 0) or lost (r < 0) beyond what its surface
# sources put in. Transport only moves mass, so r is the true conservation
# error of the whole model (advection, convection, diffusion, mass fixers).
# Surface sources and radioactive decay are the only non-transport terms
# accounted for; runs with deposition are rejected.
#
#   * B(t) is the run's own compensated Float64 global storage mass,
#     `<tracer>_total_mass` = Σ VMR_dry · m_dry (kg dry-air equivalent).
#   * E is the run's own applied source: the sources are rebuilt with the
#     model's `build_surface_flux_sources` from the run's config (same files,
#     regridding, unit and storage scaling) and integrated between output
#     times with the model's own `_flux_temporal_segments`. In storage units
#     the molar masses cancel, so no inventory rate or unit factor enters.
#   * Decaying tracers (Rn-222): the model emits at every advection substep
#     and applies the decay e^{−λT} once at the end of each met window T. The
#     reference follows that order, B̂ ← (B̂ + ∫_window E dt) e^{−λT}, so r is
#     again the conservation error. The model's departure from the exact
#     solution of B' = E − λB (about −λT/2 of the burden: fresh emissions decay
#     for a whole window) is reported separately as the splitting bias.
#
# r is reported in kg-storage, relative to the cumulative source, and as a
# global-mean dry mole fraction r / m_dry (mol/mol; ×1e6 for ppm, ×1e12 for ppt),
# with m_dry from the output's dry `column_air_mass_per_area`. The output `time`
# axis is taken as hours since [input].start_date 00 UTC, the origin of the
# model's source times. Cubed-sphere runs with stepwise or conservative-mean
# sources only.
#
#   julia --project=. scripts/diagnostics/catrine_true_mass_balance.jl \
#       <run.toml> '<directory>/<file glob>' <out.csv> [start_hours]
#
# start_hours (default 3) is the first output time used as t₀; 3 h matches the
# first GEOS-Chem CATRINE output (see catrine_true_mass_balance_gc.py).
# =============================================================================

using AtmosTransport, NCDatasets, TOML, Printf

const AT   = AtmosTransport
const DR   = AT.Models.DrivenRunner
const ICIO = AT.Models.InitialConditionIO
const SF   = AT.Operators.SurfaceFlux

# Global source rate (kg-storage/s): per time slice (time-varying) or constant (static); 0 without a source.
function global_rates(s::AT.Operators.TimeVaryingSurfaceFluxSource)
    s.cell_mass_rate_series isa Tuple ||
        error("$(s.tracer_name): only cubed-sphere (per-panel) source series are supported")
    s.scheme isa SF.LinearInterpFlux &&
        error("$(s.tracer_name): LinearInterpFlux samples per substep; one integral per output interval would not match")
    return [sum(p -> sum(Float64, @view p[:, :, k]), s.cell_mass_rate_series) for k in eachindex(s.times)]
end
function global_rates(s::AT.Operators.SurfaceFluxSource)
    s.cell_mass_rate isa Tuple || error("$(s.tracer_name): only cubed-sphere (per-panel) sources are supported")
    return sum(p -> sum(Float64, p), s.cell_mass_rate)
end
global_rates(::Nothing) = 0.0

# (rate, length) segments of the source over [t, t + Δ], as the surface operator applies them.
segments(s::AT.Operators.TimeVaryingSurfaceFluxSource, (R, times), t, Δ) =
    [(w0 * R[i0] + w1 * R[i1], Δ * f) for (i0, i1, w0, w1, f) in SF._flux_temporal_segments(s.scheme, times, t, Δ)]
segments(::AT.Operators.SurfaceFluxSource, (R, _), t, Δ) = ((R, Δ),)
segments(::Nothing, _, t, Δ) = ((0.0, Δ),)

source_times(s::AT.Operators.TimeVaryingSurfaceFluxSource) = Float64.(s.times)
source_times(_) = nothing

emitted(segs) = sum(r * ℓ for (r, ℓ) in segs; init = 0.0)

function main(cfg_path, pattern, out_csv, start_hours = 3.0)
    cfg = TOML.parsefile(cfg_path)
    haskey(cfg, "deposition") && error("deposition is not accounted for; the residual would include it")
    FT = DR._cfg_float_type(cfg)
    driver = AT.MetDrivers.TransportBinaryDriver(first(AT.expand_binary_paths(cfg["input"])); FT, arch = AT.CPU(), Hp = 1)
    specs = DR._parse_tracer_specs(cfg)
    sources = ICIO.build_surface_flux_sources(AT.MetDrivers.driver_grid(driver), specs, FT;
                                              reference_time = DR._run_reference_time(cfg))
    source = Dict(s.tracer_name => s for s in sources)
    window = Float64(AT.MetDrivers.window_dt(driver))      # met window T (s); decay is applied once per window
    chem = get(cfg, "chemistry", Dict{String, Any}())
    λ = get(chem, "kind", "") == "decay" ?
        Dict(Symbol(n) => log(2) / Float64(h) for (n, h) in get(chem, "half_lives_seconds", Dict())) :
        Dict{Symbol, Float64}()
    tracers = [s.name for s in specs]

    # Output series; daily files share their boundary frame, which is read once.
    dir, glob = splitdir(pattern)
    dir = isempty(dir) ? "." : dir
    isdir(dir) || error("no directory $dir for pattern $pattern")
    # '*' → '.*', the literal parts escaped (a shell-style glob without a Glob.jl dependency)
    escaped = [replace(p, r"[\\^$.|?*+()\[\]{}]" => s"\\\0") for p in split(glob, "*")]
    pattern_re = Regex("^" * join(escaped, ".*") * "\$")
    files = joinpath.(dir, sort(filter(f -> occursin(pattern_re, f), readdir(dir))))
    isempty(files) && error("no files match $pattern")
    hours, B, air = Float64[], Dict(n => Float64[] for n in tracers), Float64[]
    for f in files
        NCDataset(f) do ds
            area = Float64.(ds["cell_area"][:, :, :])
            for (k, h) in enumerate(Float64.(ds["time"][:]))
                (h < start_hours || (!isempty(hours) && h == hours[end])) && continue
                push!(hours, h)
                foreach(n -> push!(B[n], Float64(ds["$(n)_total_mass"][k])), tracers)
                push!(air, sum(Float64.(ds["column_air_mass_per_area"][:, :, :, k]) .* area))
            end
        end
    end
    issorted(hours) || error("output times are not increasing; check the file glob")
    t = hours .* 3600

    # Residual series per tracer: cumulative source, r(t) = B(t) − B̂(t) and, for
    # decaying tracers, the splitting bias B̂_model − B̂_exact.
    cum = Dict{Symbol, Vector{Float64}}(); res = Dict{Symbol, Vector{Float64}}(); bias = Dict{Symbol, Float64}()
    for n in tracers
        src = get(source, n, nothing)
        rates = (global_rates(src), source_times(src))
        l = get(λ, n, 0.0)
        c, ref, exact = 0.0, B[n][1], B[n][1]
        cum[n], res[n] = zeros(length(t)), zeros(length(t))
        for k in 2:length(t)
            if l > 0      # window by window: emit, then decay (model order); exact ODE alongside
                nw = round(Int, (t[k] - t[k - 1]) / window)
                nw * window ≈ t[k] - t[k - 1] ||
                    error("output interval $(t[k] - t[k - 1]) s is not a multiple of the met window $window s")
                for a in t[k - 1] .+ window .* (0:nw - 1)
                    segs = segments(src, rates, a, window)
                    e = emitted(segs)
                    c += e
                    ref = (ref + e) * exp(-l * window)
                    for (r, ℓ) in segs   # exact solution of B' = r − λB over ℓ
                        exact = exact * exp(-l * ℓ) - r * expm1(-l * ℓ) / l
                    end
                end
            else
                e = emitted(segments(src, rates, t[k - 1], t[k] - t[k - 1]))
                c += e; ref += e
            end
            cum[n][k], res[n][k] = c, B[n][k] - ref
        end
        l > 0 && exact > 0 && (bias[n] = (ref - exact) / exact)
    end

    open(out_csv, "w") do io
        println(io, "hours,dry_air_kg,", join(("$(n)_burden,$(n)_source_cum,$(n)_residual" for n in tracers), ","))
        for k in eachindex(t)
            println(io, hours[k], ",", air[k], ",", join((x for n in tracers for x in (B[n][k], cum[n][k], res[n][k])), ","))
        end
    end

    println("true mass balance from t₀ = $(start_hours) h to $(hours[end]) h ($(length(t)) times) → $out_csv")
    println("  residual r = B(t) − B(t₀) − ∫E dt (decay: B − B̂ with the model's emit-then-decay order); mole fraction = r / dry-air mass")
    for (n, b) in bias
        @printf("  %-24s splitting bias of the model's decay vs the exact solution: %+.3e of the burden\n", n, b)
    end
    for n in tracers
        r, c = res[n][end], cum[n][end]
        @printf("  %-24s r = %+.4e kg-storage   r/∫E = %+.3e   r/air = %+.4e mol/mol   max|r|/air = %.4e\n",
                n, r, c == 0 ? NaN : r / c, r / air[end], maximum(abs, res[n] ./ air))
    end
end

length(ARGS) in 3:4 || error("usage: catrine_true_mass_balance.jl <run.toml> '<dir>/<file glob>' <out.csv> [start_hours]")
main(ARGS[1], expanduser(ARGS[2]), ARGS[3], parse(Float64, get(ARGS, 4, "3")))
