#!/usr/bin/env julia
# =============================================================================
# tracer_budget_closure.jl — global tracer mass-budget closure diagnostic
# =============================================================================
#
# Two modes:
#
# * `series` (daily-series mode): the budget of every tracer of a finished run
#   over all of its snapshot files, from `<tracer>_total_mass` and the run's
#   own surface sources, through `AtmosTransport.Analysis.budget_closure`.
#   Tracers, carriers, output files and the run start come from the run
#   config. A tracer passes when |Δburden − emission| ≤ ρ·|emission| +
#   k·eps(FT(carrier))·air_mass for the run float type FT (strict: ρ = 1e-4, k = 1; operational:
#   ρ = 1e-3, k = 4), its totals and column means are finite and its carrier
#   is never exhausted. Writes a JSON report and exits 1 when the gated level
#   fails, so campaign controllers can gate on it. This mode replaces
#   `check_trendy_npp_rh_run.py`.
#
#       julia --project=. scripts/diagnostics/tracer_budget_closure.jl series <run.toml> \
#           [--output-glob GLOB] [--report report.json] [--gate strict|operational] \
#           [--strict REL] [--operational REL] [--tracers a,b] [--allow-incomplete]
#
# * single-file mode (below): the hourly budget of one output file, with the
#   decay sink of rn222 and extra diagnostics (first-window injection ratio,
#   linear-fit burden rate).
#
#       julia --project=. scripts/diagnostics/tracer_budget_closure.jl [<output.nc> <config.toml>]
#
# The rest of this header describes the single-file mode.
#
# Verifies, per tracer, the fundamental conservation law of the offline
# transport model:
#
#     burden(t_end) - burden(t_0)  ==  ∫ emission dt   [ - ∫ decay dt ]
#     "change in atmospheric burden  ==  sum of surface fluxes (minus sinks)"
#
# to the Float32 roundoff floor (~1e-6 .. 1e-5 relative). Transport
# (advection + convection + diffusion) only *moves* mass between cells, so
# it must drop out of the GLOBAL budget exactly; the only global source is
# the surface emission, and the only global sink (rn222) is radioactive
# decay. A residual far above the Float32 floor would indicate a genuine
# leak, an emission unit/scale error, or a temporal-scheme mismatch.
#
# --- WHY THIS IS A SELF-CONSISTENT, INTERNAL CHECK ---
# We work in the model's NATIVE conserved quantity: dry-air-equivalent
# storage mass = Σ VMR_dry · air_mass_dry, which is exactly what
# `total_mass(state, tracer)` accumulates and what the surface kernels add
# to (the per-cell source rate is pre-multiplied by the storage scale
# M_dryair/M_tracer inside the loader). Computing both the burden and the
# emission integral in these storage units makes the molar-mass conversion
# cancel EXACTLY — no dependence on the precise M values — so any residual
# is a true transport/accumulation residual, not a unit mismatch.
# We also report physical kg (using the model's own molar masses) for
# interpretability.
#
# --- HOW THE EMISSION IS OBTAINED (bit-identical to the run) ---
# We do NOT re-implement any inventory reader. We open the same transport
# binary the run used, build the same grid, parse the same tracer specs,
# and call the model's own `build_surface_flux_sources(...)` with the same
# `reference_time = DateTime(start_date)`. That returns, per tracer, either
#   * `SurfaceFluxSource` (static): per-cell kg-storage/s rate (constant in
#     time) — emission = total_rate · (t_end - t_0); or
#   * `TimeVaryingSurfaceFluxSource` (co2_natural): a per-cell rate SERIES +
#     slice times + temporal scheme. `integrated_emission(source, t0, tend,
#     tend - t0)` integrates it with the operator's own temporal
#     reconstruction (one step split at every slice time, which is exact for
#     the stepwise and conservative schemes), so the post-hoc integral matches
#     the run's accumulated mass.
#
# The +3h CAMS interval-start phase shift, the M(CO2)/M(C) kgC→kgCO2 factor, the
# `scale` key, the conservative LL→CS regrid, and the storage scale are all
# applied inside `build_surface_flux_sources` — we inherit them for free.
#
# --- DECAY (rn222 only) ---
# d(burden)/dt = emission - λ·burden, λ = ln2 / half_life. The model applies
# EXACT exponential decay per substep. Post-hoc we can only integrate the
# decay sink λ∫burden dt from the HOURLY output burden series (trapezoid /
# Simpson). A residual at the ~1e-3 level for rn222 is therefore the
# trapezoidal integration error of the decay term against an hourly series,
# NOT a model leak. The clean conservation statement for rn222 is:
# "same conservative transport as fossil (proven below) + exact exponential
# decay  ⇒  conserves by construction."
#
# Usage:
#   julia --project=. scripts/diagnostics/tracer_budget_closure.jl \
#       <output.nc> <config.toml>
#
# Defaults (if args omitted) point at the Dec 1-2 2021 validation run.
# =============================================================================

using AtmosTransport
using AtmosTransport: TransportTracerSpec
using AtmosTransport.Analysis: RunContext, BudgetTolerance, budget_closure, report_dict
using AtmosTransport.AtomicFiles: atomic_publish
using ArgParse
using JSON3
using NCDatasets
using TOML
using Dates
using Printf

# The run's grid, tracer specs and reference time come from `Analysis.RunContext`,
# which builds them with the code paths `run_driven_simulation` takes; the
# sources from the runtime's `build_surface_flux_sources`. Emission totals come
# from the operator's public `total_emission_rate` / `integrated_emission`.
const _ATM = AtmosTransport
const _DR  = _ATM.Models.DrivenRunner          # _parse_tracer_specs, _run_reference_time
const _ICIO = _ATM.Models.InitialConditionIO   # build_surface_flux_sources, surface_flux_species

const _ICIO = _ATM.Models.InitialConditionIO   # _tracer_molar_mass_kg_mol
const _OPS = _ATM.Operators                     # source types, emission totals
using AtmosTransport.Operators: total_emission_rate, integrated_emission

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
const DEFAULT_NC  = expanduser("~/data/AtmosTransport/output/campaign_winter2021/geosit_omega_4tracer_val_dec1-2.nc")
const DEFAULT_CFG = abspath(joinpath(@__DIR__, "..", "..",
    "config", "runs", "campaign_winter2021", "geosit_omega_4tracer_val_dec1-2.toml"))

# Model molar masses (for physical-kg reporting only; the dimensionless
# closure is computed in storage units and is independent of these).
const M_DRYAIR = _ATM.Parameters.DRY_AIR_MOLAR_MASS          # kg/mol

# Molar mass of a tracer's surface-flux species, as the source loader resolves
# it (`surface_flux.species`, else the species the tracer name implies), from
# `Parameters.SPECIES_MOLAR_MASS`; NaN when it has none, in which case no
# physical-kg line is printed.
function tracer_molar_mass(spec)
    species = try
        _ICIO.surface_flux_species(spec.name, spec.surface_flux_cfg)
    catch err
        err isa ArgumentError || rethrow()
        return NaN
    end
    return getproperty(_ATM.Parameters.SPECIES_MOLAR_MASS, species)
end

# ---------------------------------------------------------------------------
# Burden time series from the output NetCDF (model storage units = Σ VMR·air_mass)
# ---------------------------------------------------------------------------
"""
    burden_series(ds, tracer, cell_area) -> Vector{Float64}

Storage-unit burden (= the model's conserved tracer mass `rm` = Σ VMR·air_mass)
at every output time, in kg dry-air-equivalent.

We PREFER the `<tracer>_column_mass_per_area` output variable, because the
writer computes it directly as `rm / cell_area` in Float64 from the model's
internal mass field (`column_mass_per_area`, src/Output/diagnostics.jl) — so
`Σ column_mass_per_area · cell_area` reconstructs the EXACT conserved mass with
no VMR↔air_mass round-trip. (Reconstructing from the Float32 3D VMR ×
air_mass agrees to the Float32 floor, but the column-mass path is cleaner and
~72× cheaper in IO.) Falls back to VMR·air_mass if the column var is absent.
"""
function burden_series(ds, tracer::AbstractString, cell_area::Array{Float64})
    colvar = tracer * "_column_mass_per_area"
    if haskey(ds, colvar)
        cm = ds[colvar]
        nt = size(cm)[end]
        out = Vector{Float64}(undef, nt)
        for t in 1:nt
            out[t] = sum(Float64.(cm[:, :, :, t]) .* cell_area)
        end
        return out
    end
    vmr_var = ds[tracer]
    am_var  = ds["air_mass"]
    nt = size(vmr_var)[end]
    out = Vector{Float64}(undef, nt)
    for t in 1:nt
        vmr = Float64.(vmr_var[:, :, :, :, t])
        am  = Float64.(am_var[:, :, :, :, t])
        out[t] = sum(vmr .* am)
    end
    return out
end

"""
    output_seconds(ds) -> Vector{Float64}

Output times in seconds since the run start (the origin of the flux slice
times). AtmosTransport snapshots store `time` as hours since the run start:
current files with the bare units `"hours"`, older ones labelled
`"hours since 2000-01-01 00:00:00"` (a fixed label; the values still count
from the run start, 0 at the initial state). Both are read as stored; other
units are an error.
"""
function output_seconds(ds)
    tvar = ds["time"]
    units = strip(String(get(tvar.attrib, "units", "")))
    (units == "hours" || startswith(units, "hours since")) || error(
        "output `time` has units \"$(units)\"; expected AtmosTransport snapshot hours since the run start")
    return 3600 .* Float64.(collect(tvar.var[:]))
end

# Least-squares slope of `b` vs `t_sec` (kg-storage / s), with the residual
# scatter (the noise floor). For a constant-rate / clean-linear burden this
# slope is robust to per-snapshot Float32 quantization; the residual std
# reports that quantization. Returns (slope, intercept, resid_std).
function linfit_rate(b::Vector{Float64}, t_sec::Vector{Float64})
    n = length(b)
    t̄ = sum(t_sec) / n; b̄ = sum(b) / n
    sxx = sum((t_sec .- t̄) .^ 2)
    sxy = sum((t_sec .- t̄) .* (b .- b̄))
    slope = sxy / sxx
    intercept = b̄ - slope * t̄
    res = b .- (intercept .+ slope .* t_sec)
    resid_std = sqrt(sum(res .^ 2) / max(n - 2, 1))
    return slope, intercept, resid_std
end

# ---------------------------------------------------------------------------
# Decay sink integral (rn222): λ ∫ burden dt via composite trapezoid over the
# hourly burden series.
# ---------------------------------------------------------------------------
function decay_sink_trapz(burden::Vector{Float64}, t_sec::Vector{Float64}, λ::Float64)
    s = 0.0
    @inbounds for k in 1:(length(burden) - 1)
        dt = t_sec[k + 1] - t_sec[k]
        s += 0.5 * (burden[k] + burden[k + 1]) * dt
    end
    return λ * s
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main(nc_path::AbstractString, cfg_path::AbstractString)
    isfile(nc_path)  || error("output NetCDF not found: $nc_path")
    isfile(cfg_path) || error("config TOML not found: $cfg_path")

    @info "Tracer mass-budget closure diagnostic" nc_path cfg_path

    # The SAME grid (first binary, CPU), float type, tracer specs and reference
    # time the run used.
    ctx = RunContext(cfg_path)
    cfg = ctx.config
    FT = _ATM.floattype(ctx)
    grid = ctx.grid
    tracer_specs = ctx.tracer_specs
    reference_time = ctx.reference_time
    @info "Using transport binary for grid + flux regrid" first(ctx.binary_paths) FT
    @info "Reference time (run start; flux slice-time origin)" reference_time

    # Build the model's EXACT surface-flux sources (inherits +3h shift,
    # M(CO2)/M(C), scale, conservative regrid, storage scale).
    sources = _ATM.build_surface_flux_sources(grid, tracer_specs, FT;
                                              reference_time = reference_time)
    src_by_tracer = Dict(s.tracer_name => s for s in sources)

    # Output time axis in seconds since the run start (`output_seconds`).
    ds = NCDataset(nc_path)
    cell_area = Float64.(ds["cell_area"][:, :, :])    # (Xdim, Ydim, nf), m²
    t_sec = output_seconds(ds)
    t0, tend = first(t_sec), last(t_sec)
    tend > t0 || error("the budget needs at least two output times; $nc_path has $(length(t_sec))")
    @info "Run window" t0_sec=t0 tend_sec=tend hours=(tend - t0) / 3600 nsnapshots=length(t_sec)

    # Decay constants (rn222) from [chemistry].
    decay_lambda = Dict{Symbol, Float64}()
    chem = get(cfg, "chemistry", Dict{String, Any}())
    if get(chem, "kind", "") == "decay"
        for (name, hl) in get(chem, "half_lives_seconds", Dict{String, Any}())
            decay_lambda[Symbol(name)] = log(2.0) / Float64(hl)
        end
    end

    # Iterate tracers in config order.
    tracer_order = [spec.name for spec in tracer_specs]
    spec_by_tracer = Dict(spec.name => spec for spec in tracer_specs)

    println()
    println("="^118)
    @printf("%-13s %16s %16s %16s %16s %12s %10s\n",
            "tracer", "burden(t0)", "burden(tend)", "Δburden", "emission[+sink]",
            "rel_close", "verdict")
    println("(storage units = Σ VMR·air_mass, kg dry-air-equivalent; rel_close = (Δburden - emission [+ decay]) / |emission|)")
    println("-"^118)

    results = Dict{Symbol, NamedTuple}()
    for name in tracer_order
        skey = String(name)
        haskey(ds, skey) || (@warn "tracer $skey not in output; skipping"; continue)
        b = burden_series(ds, skey, cell_area)
        b0, bend = b[1], b[end]
        Δb = bend - b0

        src = get(src_by_tracer, name, nothing)
        emit = 0.0
        rate_info = ""
        if src === nothing
            @warn "no surface source for $skey; emission = 0"
        else
            # One step over [t0, tend]: equal to the run's substep sum for static,
            # stepwise and conservative-mean sources (the step is split at every
            # slice time); only approximate for LinearInterpFlux, which samples
            # its rate at every substep start.
            emit = integrated_emission(src, t0, tend, tend - t0)
            rate_info = if src isa _OPS.TimeVaryingSurfaceFluxSource
                @sprintf("time-varying %s; mean global rate over the run %.6e kg-stor/s (%d slices)",
                         typeof(src.scheme).name.name, emit / (tend - t0), length(src.times))
            else
                @sprintf("static; global rate %.6e kg-stor/s", total_emission_rate(src, t0))
            end
        end

        decay = 0.0
        has_decay = haskey(decay_lambda, name)
        analytic_bend = NaN
        if has_decay
            λ = decay_lambda[name]
            decay = decay_sink_trapz(b, t_sec, λ)
            # Quadrature-free analytic check for a CONSTANT-rate emission +
            # exact decay starting from b(t0): b(tend) = b0·e^{-λΔt} +
            # (E/λ)·(1-e^{-λΔt}), with E the constant storage emission rate.
            Δt = tend - t0
            E_rate = Δt > 0 ? emit / Δt : 0.0
            if src isa _OPS.SurfaceFluxSource     # only meaningful for constant E
                analytic_bend = b0 * exp(-λ * Δt) + (E_rate / λ) * (1 - exp(-λ * Δt))
            end
        end

        # Closure: Δburden should equal emission - decay_sink.
        expected = emit - decay
        rel = abs(emit) > 0 ? (Δb - expected) / abs(emit) : (Δb - expected)

        # Physical kg via the model's own molar masses (storage_scale = M_dry/M_tracer).
        Mt = tracer_molar_mass(spec_by_tracer[name])
        phys_scale = isnan(Mt) ? NaN : Mt / M_DRYAIR     # storage → physical kg

        # Robust diagnostics that separate a REAL gap from Float32 noise:
        #  * first-window injection ratio: Δburden over the first output step
        #    vs source·dt — before transport / decay / cancellation matter;
        #  * linear-fit burden rate vs source rate (static tracers): the LS
        #    slope is robust to per-snapshot quantization (resid_std reports it).
        inj_ratio = NaN
        if src !== nothing && length(b) >= 2
            inst = total_emission_rate(src, t0)    # the scheme's global rate at t0
            dtfh = t_sec[2] - t_sec[1]
            inj_ratio = (b[2] - b[1]) / (inst * dtfh)
        end
        fit_slope, _, fit_resid = linfit_rate(b, t_sec)

        results[name] = (b0 = b0, bend = bend, Δb = Δb, emit = emit, decay = decay,
                         rel = rel, has_decay = has_decay, rate_info = rate_info,
                         phys_scale = phys_scale, burden = b, analytic_bend = analytic_bend,
                         inj_ratio = inj_ratio, fit_slope = fit_slope, fit_resid = fit_resid)

        # Float32 floor: a global burden is Σ over ~3e7 Float32 cells, so the
        # achievable closure is a few ×1e-5 (the IC=0 fossil control lands at
        # ~2.5e-5 = the floor). Flag CLOSES ≤5e-5; decay/inj for the rn222
        # decay caveat; GAP otherwise.
        floor_ok = abs(rel) <= 5e-5
        verdict = floor_ok ? "CLOSES" : (has_decay && abs(rel) <= 5e-3 ? "decay/inj" : "GAP!")
        @printf("%-13s %16.8e %16.8e %16.8e %16.8e %12.3e %10s\n",
                skey, b0, bend, Δb, expected, rel, verdict)
    end
    println("="^118)
    println()

    # Detail block: physical kg + per-tracer notes.
    for name in tracer_order
        haskey(results, name) || continue
        r = results[name]
        println("── $name ", "─"^(60 - length(String(name))))
        @printf("   storage units:  burden(t0)=%.8e  burden(tend)=%.8e  Δ=%.8e  emission=%.8e\n",
                r.b0, r.bend, r.Δb, r.emit)
        if !isnan(r.phys_scale)
            @printf("   physical kg:    Δburden=%.6e kg   emission=%.6e kg\n",
                    r.Δb * r.phys_scale, r.emit * r.phys_scale)
        end
        if r.has_decay
            @printf("   decay sink λ∫burden dt (trapz over hourly burden): %.8e kg-stor\n", r.decay)
            println("   NOTE: rn222 = constant emission + EXACT per-substep exponential decay.")
            println("         The trapezoidal/Simpson integral of λ∫burden dt over the HOURLY")
            println("         output series carries an O((Δt)²) quadrature error.")
            if !isnan(r.analytic_bend)
                @printf("   ANALYTIC cross-check  b(tend)=E/λ·(1-e^{-λΔt}) [const E + exact decay]:\n")
                @printf("     model b(tend)=%.8e   analytic=%.8e   rel=%.4e\n",
                        r.bend, r.analytic_bend, (r.bend - r.analytic_bend) / r.analytic_bend)
                println("     (this rel is QUADRATURE-FREE; a residual here beyond the Float32")
                println("      floor is a real injection/decay effect, not an integration artifact.)")
            end
        end
        @printf("   relative closure = (Δburden - emission[+decay]) / |emission| = %.4e\n", r.rel)
        if !isnan(r.inj_ratio)
            @printf("   first-window injection ratio (Δburden_step / source·dt) = %.6f  (1.0 = exact)\n",
                    r.inj_ratio)
        end
        @printf("   linear-fit burden rate = %.6e kg-stor/s  (resid_std = %.3e = Float32 noise floor)\n",
                r.fit_slope, r.fit_resid)
        isempty(r.rate_info) || println("   source: ", r.rate_info)
        println()
    end

    close(ds)
    return results
end

# ---------------------------------------------------------------------------
# Daily-series mode
# ---------------------------------------------------------------------------
function _series_settings()
    s = ArgParseSettings(prog = "tracer_budget_closure.jl series",
        description = "Global budget of every tracer of a finished run over its snapshot files " *
                      "(Analysis.budget_closure). Exit status 1 when the gated level fails.")
    @add_arg_table! s begin
        "config"
            help = "run config TOML (as passed to scripts/run_transport.jl)"
            required = true
        "--output-glob"
            help = "snapshot files to read (wildcards in the file name), instead of the files [output].path names"
        "--report"
            help = "write the JSON report here"
        "--gate"
            help = "level that sets the exit status: strict or operational"
            default = "strict"
        "--strict"
            help = "strict relative tolerance (plus 1 run-precision ULP of each carrier)"
            arg_type = Float64
            default = 1e-4
        "--operational"
            help = "operational relative tolerance (plus 4 run-precision ULPs of each carrier)"
            arg_type = Float64
            default = 1e-3
        "--tracers"
            help = "comma-separated tracer names (default: every tracer of the run)"
        "--allow-incomplete"
            help = "check the frames present instead of requiring the run's full snapshot schedule"
            action = :store_true
    end
    return s
end

function main_series(args)
    opts = parse_args(args, _series_settings())
    gate = opts["gate"]
    gate in ("strict", "operational") || error("--gate must be strict or operational, got $(gate)")
    tracers = opts["tracers"] === nothing ? nothing : Symbol.(strip.(split(opts["tracers"], ',')))
    report = budget_closure(expanduser(opts["config"]);
                            output_glob = opts["output-glob"],
                            tolerance = BudgetTolerance(strict = opts["strict"], operational = opts["operational"]),
                            tracers, require_complete = !opts["allow-incomplete"])

    println("="^112)
    @printf("%-24s %16s %16s %12s %11s %9s %7s %7s\n", "tracer", "expected [kg]", "actual [kg]",
            "rel. error", "error [ppm]", "min col", "strict", "operat.")
    println("-"^112)
    for b in report.tracers
        @printf("%-24s %16.8e %16.8e %12.4e %11.3e %9.3g %7s %7s\n", b.tracer, b.expected_change,
                b.actual_change, b.relative_error, 1e6 * b.error_vmr, b.minimum_column_mean,
                b.strict_passed ? "pass" : "FAIL", b.operational_passed ? "pass" : "FAIL")
    end
    println("="^112)
    for (tracer, reason) in report.skipped
        println("skipped ", tracer, ": ", reason)
    end
    @printf("%d tracers, %s to %s (%d frames in %d files); max relative error %.6g\n",
            length(report.tracers), report.start, report.stop, report.frames, length(report.files),
            report.max_relative_error)
    println("strict ", report.passed ? "passed" : "FAILED", "; operational ",
            report.operational_passed ? "passed" : "FAILED", "; gate: ", gate)
    if opts["report"] !== nothing
        path = abspath(expanduser(opts["report"]))
        atomic_publish(path) do tmp
            open(io -> (JSON3.pretty(io, report_dict(report)); println(io)), tmp, "w")
        end
        println("report: ", path)
    end
    return gate == "strict" ? report.passed : report.operational_passed
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "series"
        exit(main_series(ARGS[2:end]) ? 0 : 1)
    end
    nc_path  = length(ARGS) >= 1 ? expanduser(ARGS[1]) : DEFAULT_NC
    cfg_path = length(ARGS) >= 2 ? expanduser(ARGS[2]) : DEFAULT_CFG
    main(nc_path, cfg_path)
end
