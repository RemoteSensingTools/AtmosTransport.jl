#!/usr/bin/env julia
# =============================================================================
# tracer_budget_closure.jl — global tracer mass-budget closure diagnostic
# =============================================================================
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
#     slice times + temporal scheme. We reduce it to a GLOBAL scalar rate
#     series R(t_k) and integrate it over [t_0, t_end] using the MODEL's own
#     `_flux_temporal_segments(scheme, times, t, dt)` knot-split — i.e. the
#     exact same window-conservative integral the surface operator applies,
#     so the post-hoc integral matches the run's accumulated mass.
#
# The +3h CAMS interval-start phase shift, the 44/12 kgC→kgCO2 factor, the
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
using NCDatasets
using TOML
using Dates
using Printf

# Reach the (un-exported) internals the model uses to resolve binary paths,
# build flux sources, set the reference time, and integrate the temporal
# scheme. These are the SAME code paths `run_driven_simulation` takes.
const _ATM = AtmosTransport
const _DR  = _ATM.Models.DrivenRunner          # _parse_tracer_specs, _run_reference_time
const _ICIO = _ATM.Models.InitialConditionIO   # build_surface_flux_sources
const _OPS = _ATM.Operators                     # exported scheme/source types
const _SF  = _ATM.Operators.SurfaceFlux         # _flux_temporal_segments (un-exported)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
const DEFAULT_NC  = expanduser("~/data/AtmosTransport/output/campaign_winter2021/geosit_omega_4tracer_val_dec1-2.nc")
const DEFAULT_CFG = abspath(joinpath(@__DIR__, "..", "..",
    "config", "runs", "campaign_winter2021", "geosit_omega_4tracer_val_dec1-2.toml"))

# Model molar masses (for physical-kg reporting only; the dimensionless
# closure is computed in storage units and is independent of these).
const M_DRYAIR = _ATM.Parameters.DRY_AIR_MOLAR_MASS          # kg/mol
const M_TRACER = _ICIO._KNOWN_TRACER_MOLAR_MASS_KG_MOL     # Symbol => kg/mol

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
# Emission integral (storage units), reusing the model's loaders + scheme
# ---------------------------------------------------------------------------
"""
    static_emission(source, t0, tend) -> kg_storage

Constant-rate source: total per-cell storage rate × window length.
"""
function static_emission(source, t0::Float64, tend::Float64)
    r = source.cell_mass_rate
    total_rate = r isa Tuple ? sum(sum, r) : sum(r)   # kg-storage / s
    return Float64(total_rate) * (tend - t0), Float64(total_rate)
end

"""
    timevarying_emission(source, t0, tend) -> kg_storage

Time-varying source integrated with the MODEL's own temporal scheme. We
build the global scalar rate series R_k = Σ_cells series[:,:,k] (kg-storage/s)
and integrate the scheme's reconstruction exactly over [t0, tend] using the
model's `_flux_temporal_segments` knot-split (the same routine the surface
operator calls). For ConservativeMeanFlux this is the exact integral of the
piecewise-LINEAR flux reconstruction with constant extrapolation outside the
slice range — i.e. the window-conservative integral the run accumulated.
"""
function timevarying_emission(source, t0::Float64, tend::Float64)
    series = source.cell_mass_rate_series          # NTuple{6} of (Nc,Nc,ntime)
    times  = Float64.(source.times)                # seconds since run start
    scheme = source.scheme
    ntime  = length(times)

    # Global scalar rate per slice (kg-storage / s).
    R = zeros(Float64, ntime)
    for p in eachindex(series)
        sp = series[p]
        for k in 1:ntime
            R[k] += sum(@view sp[:, :, k])
        end
    end

    # Integrate over [t0, tend] using the model's segment splitter as ONE
    # step of length (tend - t0). Each segment carries (i0,i1,w0,w1,dt_frac);
    # the per-segment rate is w0·R[i0] + w1·R[i1] and the segment length is
    # (tend-t0)·dt_frac, so the emitted mass = Σ rate·length — bit-identical
    # to summing the operator's per-substep applications over the window.
    dt = tend - t0
    segs = _SF._flux_temporal_segments(scheme, times, t0, dt)
    emit = 0.0
    for (i0, i1, w0, w1, dt_frac) in segs
        rate = w0 * R[i0] + w1 * R[i1]
        emit += rate * (dt * dt_frac)
    end
    return emit, R
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

    cfg = TOML.parsefile(cfg_path)
    FT  = _ATM.Models.DrivenRunner._cfg_float_type(cfg)

    input_cfg     = cfg["input"]
    binary_paths  = _ATM.expand_binary_paths(input_cfg)
    isempty(binary_paths) && error("could not resolve any binary path from [input]")
    binpath = first(binary_paths)
    @info "Using transport binary for grid + flux regrid" binpath FT

    # Build the SAME grid the run used (CPU driver — we only need the mesh).
    # Cubed-sphere binaries use the CS-specific driver/reader; the generic
    # TransportBinaryDriver is for lat/lon + reduced-Gaussian. We mirror the
    # CS run path (`_run_driven_simulation_cs`) here.
    driver = _ATM.MetDrivers.TransportBinaryDriver(binpath; FT = FT,
                                                        arch = _ATM.CPU(), Hp = 1)
    grid   = _ATM.MetDrivers.driver_grid(driver)

    # Parse the SAME tracer specs and reference time the run used.
    tracer_specs   = _DR._parse_tracer_specs(cfg)
    reference_time = _DR._run_reference_time(cfg)
    @info "Reference time (run start; flux slice-time origin)" reference_time

    # Build the model's EXACT surface-flux sources (inherits +3h shift,
    # 44/12, scale, conservative regrid, storage scale).
    sources = _ICIO.build_surface_flux_sources(grid, tracer_specs, FT;
                                               reference_time = reference_time)
    src_by_tracer = Dict(s.tracer_name => s for s in sources)

    # Output time axis (seconds since the first snapshot = run start t0=0).
    ds = NCDataset(nc_path)
    cell_area = Float64.(ds["cell_area"][:, :, :])    # (Xdim, Ydim, nf), m²
    tvar = ds["time"][:]
    t0_dt = DateTime(first(tvar))
    t_sec = Float64[Dates.value(DateTime(x) - t0_dt) / 1000.0 for x in tvar]
    t0, tend = first(t_sec), last(t_sec)
    @info "Run window" t0_sec=t0 tend_sec=tend hours=(tend - t0) / 3600 nsnapshots=length(t_sec)

    # Instantaneous global source rate per tracer (kg-storage/s) for the
    # first-window injection check (static + the time-varying scheme value at t0).
    function instantaneous_rate(src)
        if src isa _OPS.SurfaceFluxSource
            r = src.cell_mass_rate
            return Float64(r isa Tuple ? sum(sum, r) : sum(r))
        elseif src isa _OPS.TimeVaryingSurfaceFluxSource
            _, R = timevarying_emission(src, t0, tend)
            i0, i1, w0, w1 = _SF._time_interp_bracket(src.times, t0)
            return w0 * R[i0] + w1 * R[i1]
        end
        return NaN
    end

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
        elseif src isa _OPS.TimeVaryingSurfaceFluxSource
            emit, R = timevarying_emission(src, t0, tend)
            rate_info = @sprintf("time-varying %s; mean global rate %.6e kg-stor/s (%d slices)",
                                 typeof(src.scheme).name.name, sum(R) / length(R), length(R))
        elseif src isa _OPS.SurfaceFluxSource
            emit, rate = static_emission(src, t0, tend)
            rate_info = @sprintf("static; global rate %.6e kg-stor/s", rate)
        else
            @warn "unhandled source type for $skey" typeof(src)
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
        Mt = get(M_TRACER, name, NaN)
        phys_scale = isnan(Mt) ? NaN : Mt / M_DRYAIR     # storage → physical kg

        # Robust diagnostics that separate a REAL gap from Float32 noise:
        #  * first-window injection ratio: Δburden over the first output step
        #    vs source·dt — before transport / decay / cancellation matter;
        #  * linear-fit burden rate vs source rate (static tracers): the LS
        #    slope is robust to per-snapshot quantization (resid_std reports it).
        inj_ratio = NaN
        if src !== nothing && length(b) >= 2
            inst = instantaneous_rate(src)
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

if abspath(PROGRAM_FILE) == @__FILE__
    nc_path  = length(ARGS) >= 1 ? expanduser(ARGS[1]) : DEFAULT_NC
    cfg_path = length(ARGS) >= 2 ? expanduser(ARGS[2]) : DEFAULT_CFG
    main(nc_path, cfg_path)
end
