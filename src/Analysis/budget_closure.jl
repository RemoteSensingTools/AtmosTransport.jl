# ===========================================================================
# Global budget closure of a finished run, and the carrier drawdown of a
# planned one.
#
# The runtime conserves tracer storage, the dry-air-equivalent mass
#
#     S = Σ_c χ_c m_c          (dry mixing ratio × carrier air mass, kg),
#
# and writes its global total as `<tracer>_total_mass` in every snapshot (a
# compensated Float64 sum taken before any output precision conversion).
# Transport only moves storage between cells, so it drops out of the global
# sum; the only global source is the tracer's surface flux. Between two
# snapshots at t₀ and t₁
#
#     ΔS = S(t₁) − S(t₀) = E(t₀, t₁),
#
# where E is the storage the surface-flux operator added: the per-cell rates
# r_c = F_c A_c M_air / M_tracer (flux density × mesh cell area × storage
# scale) summed over cells and integrated with the operator's temporal
# reconstruction (`Operators.integrated_emission`). Building E from the
# runtime's own sources makes the cell areas, unit factors, storage scale and
# slice timing those of the run.
#
# A run stores χ (or the per-cell storage) in its configured float type FT. A
# tracer that starts from a uniform carrier χ₀ cannot resolve increments below
# one FT spacing of χ₀ per cell; summed over the air mass M this is the carrier
# term εᶠᵀ(χ₀) M of the tolerance, with εᶠᵀ(x) = eps(FT(x)).
# ===========================================================================

"""
    BudgetTolerance(; strict = 1e-4, operational = 1e-3,
                      strict_carrier_ulps = 1, operational_carrier_ulps = 4)

Tolerances of [`budget_closure`](@ref). A tracer with expected storage change
``E``, actual change ``\\Delta S``, initial carrier ``\\chi_0`` and air mass
``M`` passes a level with relative tolerance ``\\rho`` and ``k`` carrier ulps
when

```math
|\\Delta S - E| \\le \\rho\\,|E| + k\\,\\varepsilon_{FT}(\\chi_0)\\,M,
\\qquad \\varepsilon_{FT}(x) = \\mathrm{eps}(\\mathrm{FT}(x)).
```

The strict level (``10^{-4}`` plus one carrier ulp) is the check of the
retired `check_trendy_npp_rh_run.py`, which gated the NPP/Rh campaign
(`scripts/run_trendy_npp_rh_campaign.py` still gates on it); the operational
level (``10^{-3}`` plus four ulps) is the guard of
`scripts/run_transcom_campaign.py`. Every value must be finite and
non-negative.
"""
struct BudgetTolerance
    strict                   :: Float64
    operational              :: Float64
    strict_carrier_ulps      :: Float64
    operational_carrier_ulps :: Float64

    function BudgetTolerance(strict::Real, operational::Real,
                             strict_carrier_ulps::Real, operational_carrier_ulps::Real)
        for (name, value) in (("strict", strict), ("operational", operational),
                              ("strict_carrier_ulps", strict_carrier_ulps),
                              ("operational_carrier_ulps", operational_carrier_ulps))
            isfinite(value) && value >= 0 || throw(ArgumentError(
                "BudgetTolerance: $(name) must be finite and non-negative; got $(value)"))
        end
        return new(strict, operational, strict_carrier_ulps, operational_carrier_ulps)
    end
end

BudgetTolerance(; strict::Real = 1e-4, operational::Real = 1e-3,
                strict_carrier_ulps::Real = 1, operational_carrier_ulps::Real = 4) =
    BudgetTolerance(strict, operational, strict_carrier_ulps, operational_carrier_ulps)

"""
    TracerBudget

Budget of one tracer in a [`BudgetReport`](@ref). Masses are tracer storage in
kg (mixing ratio × carrier air mass, the unit of `<tracer>_total_mass`).

- `tracer`, `carrier` (initial carrier mixing ratio ``\\chi_0``: the
  `background` of a `uniform` initial condition, otherwise the initial mean
  ``S(t_0)/M``), `air_mass` (``M``, [`RunContext`](@ref));
- `initial_storage` ``S(t_0)`` and `final_storage` ``S(t_1)``, the first and
  last snapshot totals;
- `expected_change` ``E`` and `actual_change` ``\\Delta S = S(t_1) - S(t_0)``;
- `relative_error` ``= |\\Delta S - E| / \\max(|E|, 1\\,\\mathrm{kg})`` and
  `error_vmr` ``= (\\Delta S - E)/M``, the error as a mixing ratio;
- `carrier_ulp` ``= \\varepsilon_{FT}(\\chi_0)\\,M`` for the run's float type,
  and the absolute `strict_tolerance` and `operational_tolerance` of
  [`BudgetTolerance`](@ref);
- `minimum_column_mean`: the smallest column-mean mixing ratio in any snapshot
  (NaN when the files have no `<tracer>_column_mean`), `column_checked`:
  whether the files have it, and `finite`: whether every total and column
  mean is finite;
- `strict_passed`, `operational_passed`: the closure holds at that level, the
  values are finite and, for a tracer with a positive carrier, every column
  mean stays positive (the carrier is not exhausted). A tracer with a positive
  carrier whose column means were not written cannot show that and fails.
"""
struct TracerBudget
    tracer                :: Symbol
    carrier               :: Float64
    air_mass              :: Float64
    initial_storage       :: Float64
    final_storage         :: Float64
    expected_change       :: Float64
    actual_change         :: Float64
    relative_error        :: Float64
    error_vmr             :: Float64
    carrier_ulp           :: Float64
    strict_tolerance      :: Float64
    operational_tolerance :: Float64
    minimum_column_mean   :: Float64
    column_checked        :: Bool
    finite                :: Bool
    strict_passed         :: Bool
    operational_passed    :: Bool
end

function TracerBudget(tracer::Symbol, carrier::Real, air_mass::Real, initial::Real, final::Real,
                      expected::Real, carrier_ulp::Real, minimum_column::Real, finite::Bool,
                      tolerance::BudgetTolerance)
    actual = Float64(final) - Float64(initial)
    error = actual - Float64(expected)
    ulp = Float64(carrier_ulp)
    strict = tolerance.strict * abs(expected) + tolerance.strict_carrier_ulps * ulp
    operational = tolerance.operational * abs(expected) + tolerance.operational_carrier_ulps * ulp
    finite_all = finite && isfinite(initial) && isfinite(final) && isfinite(expected)
    # A positive carrier must survive everywhere (and must be shown to); a tracer
    # that starts from zero may go negative. NaN marks column means not written.
    column_checked = !isnan(minimum_column)
    carrier_kept = !(carrier > 0) || (column_checked && minimum_column > 0)
    healthy = finite_all && carrier_kept
    return TracerBudget(tracer, carrier, air_mass, initial, final, expected, actual,
                        abs(error) / max(abs(expected), 1.0), error / air_mass, ulp,
                        strict, operational, minimum_column, column_checked, finite_all,
                        healthy && abs(error) <= strict, healthy && abs(error) <= operational)
end

"""
    BudgetReport

Result of [`budget_closure`](@ref):

- `config_path` (empty when the config was passed as a `Dict`) and `files`,
  the snapshot files read, in time order;
- `start` and `stop`: the times of the first and last snapshot, between which
  the budgets are formed; `frames`: the number of distinct snapshot times;
- `tolerance` ([`BudgetTolerance`](@ref)) and `emission`, the method of the
  expected changes (`repr` of an [`AbstractExpectedEmission`](@ref));
- `tracers`: one [`TracerBudget`](@ref) per checked tracer, in the order of
  the run's tracer specifications;
- `skipped`: tracers not checked, with the reason (written to no snapshot, or
  a chemistry sink that snapshots do not resolve);
- `passed` and `operational_passed`: every checked tracer passes that level;
  `max_relative_error`: the largest `relative_error`.

[`report_dict`](@ref) turns it into the JSON report of
`scripts/diagnostics/tracer_budget_closure.jl`.
"""
struct BudgetReport
    config_path        :: String
    files              :: Vector{String}
    start              :: DateTime
    stop               :: DateTime
    frames             :: Int
    tolerance          :: BudgetTolerance
    emission           :: String
    tracers            :: Vector{TracerBudget}
    skipped            :: Vector{Pair{Symbol, String}}
    passed             :: Bool
    operational_passed :: Bool
    max_relative_error :: Float64
end

function BudgetReport(config_path, files, start, stop, frames, tolerance, emission, tracers, skipped)
    return BudgetReport(config_path, files, start, stop, frames, tolerance, emission, tracers, skipped,
                        all(b -> b.strict_passed, tracers), all(b -> b.operational_passed, tracers),
                        isempty(tracers) ? NaN : maximum(b -> b.relative_error, tracers))
end

# ---------------------------------------------------------------------------
# Expected storage change
# ---------------------------------------------------------------------------

"""
    AbstractExpectedEmission

How [`budget_closure`](@ref) obtains the expected storage change of a tracer.
A subtype implements
[`expected_storage_change(method, ctx, spec, t0, t1)`](@ref expected_storage_change).
The default, [`RuntimeSurfaceSources`](@ref), integrates the run's own
surface-flux sources; a validation harness can plug in another method, for
example a reimplementation of a retired script with its constants.
"""
abstract type AbstractExpectedEmission end

"""
    RuntimeSurfaceSources()

Expected storage change from the run's own surface-flux sources: the tracer's
source is built by `build_surface_flux_sources` on the run's grid, in the
run's float type, with the run start as reference time (so with the run's cell
areas, unit factors, storage scale ``M_{air}/M_{tracer}`` and slice times), and
integrated over the run's advection steps (`ctx.windows`) with
`Operators.integrated_emission` and `Operators.emission_segments`, so
``[t_0, t_1]`` must consist of whole met windows, as the snapshot times do.
The runtime starts step ``j`` of a window of length ``L`` and ``n`` steps at
``t_w + L\\,j/n`` and applies the source for a step ``\\Delta t`` =
`FT(L) / FT(n)` in the run's float type. Static, stepwise and conservative-mean
sources are integrated exactly piece by piece, so over windows whose steps
tile them (``n\\,\\Delta t = L``) one exact integral gives the step sum; in
the other windows (a Float32 run with seven steps per window, for example)
the steps are summed one by one. A linearly interpolated source is sampled at
every step start: window by window with `integrated_emission` and the
window's step where the steps tile it (its starts ``t_w + j\\,\\Delta t`` equal
the runtime's to Float64 rounding), step by step elsewhere. The source holds
the slices the runner loads for the interval (through 1 s past ``t_1``).
Piece durations are Float64; the runtime rounds the duration
``\\Delta t \\cdot f`` of a piece of a step split at a slice time to the
float type (`dt_FT * FT(frac)`, with its blend weights in the float type
too), a difference of a few float-type roundings of that piece's emission,
the order of the rounding of each per-cell increment the operator adds; it is
not reproduced here.
"""
struct RuntimeSurfaceSources <: AbstractExpectedEmission end

"""
    expected_storage_change(method::AbstractExpectedEmission, ctx::RunContext,
                            spec::TransportTracerSpec, t0, t1) -> Float64

Tracer storage in kg that the tracer of `spec` gains from its sources between
`t0` and `t1` (seconds since `ctx.reference_time`), by `method`. A tracer
without a surface flux gains 0.
"""
function expected_storage_change end

function expected_storage_change(::RuntimeSurfaceSources, ctx::RunContext,
                                 spec::TransportTracerSpec, t0::Real, t1::Real)
    src = _runtime_source(ctx, spec, _flux_span(Float64(t0), Float64(t1)))
    src === nothing && return 0.0
    return _run_integral(src, ctx, Float64(t0), Float64(t1))
end

# Flux slices for an interval [t0, t1] as the runner loads them for its run:
# through 1 s past the end (`_run_time_span` of the runner), so that a
# float-type step reaching past t1 sees the slices it sees at run time.
_flux_span(t0::Float64, t1::Float64) = (t0, last(_run_time_span(t1 / 3600)))

# The tracer's runtime source with the slices needed for `span`, or `nothing`.
function _runtime_source(ctx::RunContext, spec::TransportTracerSpec, span::Tuple{Float64, Float64})
    sources = build_surface_flux_sources(ctx.grid, (spec,), floattype(ctx);
                                         reference_time = ctx.reference_time, run_span = span)
    return isempty(sources) ? nothing : only(sources)
end

# Whether the operator's integral over an interval depends on the step: static,
# stepwise and conservative-mean sources are integrated exactly piece by piece;
# a linearly interpolated source is sampled at every step start.
_step_independent(::SurfaceFluxSource) = true
_step_independent(src::TimeVaryingSurfaceFluxSource) = _step_independent(src.scheme)
_step_independent(::Union{StepwiseFlux, ConservativeMeanFlux}) = true
_step_independent(::AbstractFluxTemporalScheme) = false

# Whether the rate is constant between slice times (static and stepwise
# sources), so that the cumulative storage change is linear there and its
# extremes lie at slice times and stretch ends.
_constant_between_slices(::SurfaceFluxSource) = true
_constant_between_slices(src::TimeVaryingSurfaceFluxSource) = _constant_between_slices(src.scheme)
_constant_between_slices(::StepwiseFlux) = true
_constant_between_slices(::AbstractFluxTemporalScheme) = false

# Whether the runtime's steps of a window add up to the window exactly.
_steps_tile(w::RunWindow) = w.steps * w.step == w.length

# Start of step `j` (0-based) of window `w` on the run clock, as `_clock_time`
# of `DrivenSimulation` computes it.
_step_start(w::RunWindow, j::Integer) = w.start + w.length * (j / w.steps)

# Storage the source adds over [t0, t1] as the run applies it (`_run_pieces`).
function _run_integral(src, ctx::RunContext, t0::Float64, t1::Float64)
    total = compensation = 0.0
    for (start, stop, w) in _run_pieces(ctx, t0, t1, _step_independent(src))
        piece = w === nothing ? integrated_emission(src, start, stop, stop - start) : _step_sum(src, w)
        total, compensation = _neumaier_add(total, compensation, piece)   # Architectures
    end
    return total + compensation
end

# The intervals in which the run adds a source over [t0, t1], in time order,
# as `(start, stop, w)`: with `merge`, `w === nothing` for a stretch of whole
# windows whose steps tile them, which the caller takes in one piece;
# otherwise the one window `w`, whose steps (`_step_start`, `w.step`) are taken
# one by one.
function _run_pieces(ctx::RunContext, t0::Float64, t1::Float64, merge::Bool)
    pieces = Tuple{Float64, Float64, Union{Nothing, RunWindow}}[]
    for w in _windows_between(ctx, t0, t1)
        if merge && _steps_tile(w)
            if !isempty(pieces) && last(pieces)[3] === nothing
                pieces[end] = (first(last(pieces)), w.start + w.length, nothing)
            else
                push!(pieces, (w.start, w.start + w.length, nothing))
            end
        else
            push!(pieces, (w.start, w.start + w.length, w))
        end
    end
    return pieces
end

# Sum over the steps of window `w` of the storage each adds. When the steps tile
# the window, `integrated_emission` samples the same step starts (to rounding).
function _step_sum(src, w::RunWindow)
    _steps_tile(w) && return integrated_emission(src, w.start, w.start + w.length, w.step)
    total = compensation = 0.0
    for j in 0:(w.steps - 1), (rate, duration) in emission_segments(src, _step_start(w, j), w.step)
        total, compensation = _neumaier_add(total, compensation, rate * duration)
    end
    return total + compensation
end

# The run's windows that make up [t0, t1] (whole windows, to 1 ms).
function _windows_between(ctx::RunContext, t0::Float64, t1::Float64)
    tol = 1e-3
    windows = filter(w -> w.start >= t0 - tol && w.start + w.length <= t1 + tol, ctx.windows)
    (!isempty(windows) && abs(first(windows).start - t0) <= tol &&
     abs(last(windows).start + last(windows).length - t1) <= tol) || throw(ArgumentError(
        "[$(t0), $(t1)] s is not a whole number of the run's met windows"))
    return windows
end

# Pieces (rate, duration) in which the run adds the source over [t0, t1], in
# time order (`Operators.emission_segments`): whole stretches of tiled windows
# split at slice times when the rate is constant between them, otherwise every
# step of every window (a conservative-mean rate varies between slice times,
# and the run adds it step by step).
function _foreach_emission_piece(f, src, ctx::RunContext, t0::Float64, t1::Float64)
    for (start, stop, w) in _run_pieces(ctx, t0, t1, _constant_between_slices(src))
        if w === nothing
            foreach(piece -> f(piece...), emission_segments(src, start, stop - start))
        else
            for j in 0:(w.steps - 1), piece in emission_segments(src, _step_start(w, j), w.step)
                f(piece...)
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Budget closure
# ---------------------------------------------------------------------------

"""
    budget_closure(run; output_glob = nothing, tolerance = BudgetTolerance(),
                   emission = RuntimeSurfaceSources(), tracers = nothing,
                   require_complete = true) -> BudgetReport

Check the global budget of every tracer of a finished run: the change of its
storage between the first and last snapshot against the storage its sources
added over the same interval.

`run` is the run config (a TOML path or the parsed `Dict`, as passed to
`run_driven_simulation`) or a [`RunContext`](@ref). For each tracer:

- the actual change ``\\Delta S = S(t_1) - S(t_0)`` comes from
  `<tracer>_total_mass` in the run's snapshot files, read with
  [`daily_snapshot_series`](@ref) from the files the run names in
  `[output].path` ([`run_snapshot_paths`](@ref)), or from the files matching
  `output_glob` (sorted), for a run whose output was moved;
- the expected change ``E`` is
  [`expected_storage_change`](@ref)`(emission, ctx, spec, t0, t1)`;
- the tolerance is ``\\rho |E| + k\\,\\varepsilon_{FT}(\\chi_0)\\,M``
  ([`BudgetTolerance`](@ref)), with ``FT`` the run's float type, ``\\chi_0``
  the tracer's initial carrier and ``M`` the run's initial air mass;
- every total and `<tracer>_column_mean` value must be finite, and a tracer
  with a positive carrier must keep every column mean positive (so its column
  means must be in the snapshots, as in `check_trendy_npp_rh_run.py`).

With `require_complete = true` the snapshot times must be exactly those the
run writes (the `[output]` schedule through the end of the run, hour 0
included when scheduled), so a missing file or an unfinished run is an error;
with `false` the budget covers whatever frames the files hold. `tracers`
restricts the check to some tracer names. Tracers missing from the snapshots
and tracers with a `[chemistry]` sink (decay) are listed in `skipped`: a
daily total does not resolve the decay integral.

The budget is in tracer storage, so molar masses cancel; a residual is a
transport or accumulation error, not a unit error. Replaces
`check_trendy_npp_rh_run.py`, whose expected change used the flux file's
`cell_area`, `M_air = 0.02896546` and whole-day sums; this one uses the
runtime's sources, so it is valid for any run and any temporal scheme.
"""
function budget_closure(run; output_glob::Union{Nothing, AbstractString} = nothing,
                        tolerance::BudgetTolerance = BudgetTolerance(),
                        emission::AbstractExpectedEmission = RuntimeSurfaceSources(),
                        tracers = nothing, require_complete::Bool = true)
    ctx = run isa RunContext ? run : RunContext(run)
    paths = output_glob === nothing ? run_snapshot_paths(ctx) : _glob_paths(output_glob)
    missing_files = filter(!isfile, paths)
    isempty(missing_files) || throw(ArgumentError(
        "$(length(missing_files)) of $(length(paths)) snapshot files are missing, the first " *
        "$(first(missing_files)); the run is unfinished or its output moved (use `output_glob`)"))

    specs = _selected_specs(ctx, tracers)
    present_totals, present_columns = NCDataset(first(paths), "r") do ds
        Set(name for name in keys(ds) if endswith(name, "_total_mass")),
        Set(name for name in keys(ds) if endswith(name, "_column_mean"))
    end
    skipped = Pair{Symbol, String}[]
    checked = TransportTracerSpec[]
    for spec in specs
        if _decays(ctx, spec.name)
            push!(skipped, spec.name => "decays ([chemistry]); daily totals do not resolve the decay sink")
        elseif !(_total_name(spec.name) in present_totals)
            push!(skipped, spec.name => "no $(_total_name(spec.name)) in $(first(paths))")
        else
            push!(checked, spec)
        end
    end

    isempty(checked) && throw(ArgumentError(
        "no tracer to check: " * join(("$(k) ($(v))" for (k, v) in skipped), ", ")))
    series = daily_snapshot_series(paths, [_total_name(spec.name) for spec in checked];
                                   origin = ctx.reference_time)
    hours = [Dates.value(t - ctx.reference_time) / 3_600_000 for t in series.times]
    length(hours) >= 2 || throw(ArgumentError(
        "the snapshot files hold $(length(hours)) distinct time(s); a budget needs two"))
    require_complete && _check_complete_frames(ctx, hours)

    column_names = [_column_name(spec.name) for spec in checked if _column_name(spec.name) in present_columns]
    columns = _column_extrema(paths, column_names)

    t0, t1 = 3600 * first(hours), 3600 * last(hours)
    budgets = TracerBudget[]
    for spec in checked
        total = series.values[_total_name(spec.name)]
        initial, final = Float64(first(total)), Float64(last(total))
        carrier = _initial_carrier(spec, initial, ctx.air_mass)
        carrier_ulp = Float64(eps(floattype(ctx)(carrier))) * ctx.air_mass
        minimum_column, column_finite = get(columns, _column_name(spec.name), (NaN, true))
        push!(budgets, TracerBudget(spec.name, carrier, ctx.air_mass, initial, final,
                                    expected_storage_change(emission, ctx, spec, t0, t1),
                                    carrier_ulp, minimum_column,
                                    column_finite && all(isfinite, total), tolerance))
    end
    return BudgetReport(ctx.config_path, paths, first(series.times), last(series.times),
                        length(hours), tolerance, repr(emission), budgets, skipped)
end

_total_name(tracer::Symbol) = string(tracer, "_total_mass")
_column_name(tracer::Symbol) = string(tracer, "_column_mean")

function _selected_specs(ctx::RunContext, tracers)
    tracers === nothing && return ctx.tracer_specs
    wanted = Set(Symbol.(collect(tracers)))
    unknown = setdiff(wanted, Set(spec.name for spec in ctx.tracer_specs))
    isempty(unknown) || throw(ArgumentError(
        "tracers $(sort!(collect(unknown))) are not tracers of the run"))
    return filter(spec -> spec.name in wanted, ctx.tracer_specs)
end

# Files matching a glob pattern (`*` and `?` in the file name), sorted.
function _glob_paths(pattern::AbstractString)
    dir, name = splitdir(expanduser(pattern))
    dir = isempty(dir) ? "." : dir
    occursin('/', name) && throw(ArgumentError("output_glob may use wildcards only in the file name"))
    regex = Regex("^" * replace(name, r"[.^$|+()\[\]{}\\]" => s"\\\0", "*" => ".*", "?" => ".") * "\$")
    isdir(dir) || throw(ArgumentError("output_glob directory $(dir) does not exist"))
    paths = sort!([joinpath(dir, f) for f in readdir(dir) if occursin(regex, f)])
    isempty(paths) && throw(ArgumentError("no file matches output_glob $(pattern)"))
    return paths
end

# The frames must be the ones the runner writes: the scheduled hours through
# the end of the run.
function _check_complete_frames(ctx::RunContext, hours::Vector{Float64})
    scheduled = _scheduled_snapshot_hours(ctx)
    absent = [h for h in scheduled if !any(x -> abs(x - h) <= _HOUR_MATCH, hours)]
    extra = [h for h in hours if !any(x -> abs(x - h) <= _HOUR_MATCH, scheduled)]
    isempty(absent) && isempty(extra) && return nothing
    show_hours(v) = join(first(v, 5), ", ") * (length(v) > 5 ? ", … ($(length(v)) in all)" : "")
    throw(ArgumentError(
        "the snapshot times do not match the run's schedule (through hour $(ctx.run_seconds / 3600))" *
        (isempty(absent) ? "" : "; missing hours $(show_hours(absent))") *
        (isempty(extra) ? "" : "; unscheduled hours $(show_hours(extra))") *
        "; pass `require_complete = false` to check the frames present"))
end

# Smallest column mean of each variable over every frame of every file, and
# whether all of them are finite (fill values count as non-finite).
function _column_extrema(paths, names)
    lowest = Dict{String, Float64}(name => Inf for name in names)
    finite = Dict{String, Bool}(name => true for name in names)
    isempty(names) && return Dict{String, Tuple{Float64, Bool}}()
    for path in paths
        NCDataset(path, "r") do ds
            for name in names
                haskey(ds, name) || throw(ArgumentError("$(path) has no `$(name)`"))
                var = ds[name]
                T = nonmissingtype(eltype(var))
                low, all_finite = _finite_minimum(nomissing(var[ntuple(_ -> Colon(), ndims(var))...], T(NaN)))
                lowest[name] = min(lowest[name], low)
                finite[name] &= all_finite
            end
        end
    end
    return Dict(name => (lowest[name], finite[name]) for name in names)
end

# Smallest finite value (Inf when none) and whether every value is finite.
function _finite_minimum(values::AbstractArray{<:Real})
    low = Inf
    all_finite = true
    for v in values
        if isfinite(v)
            low = min(low, Float64(v))
        else
            all_finite = false
        end
    end
    return low, all_finite
end

# χ₀: the background of a uniform initial condition (the runner's default for a
# missing background), otherwise the initial mean mixing ratio S(t₀)/M.
function _initial_carrier(spec::TransportTracerSpec, initial_storage::Real, air_mass::Real)
    _init_kind(spec.init_cfg) === :uniform &&
        return Float64(get(spec.init_cfg, "background", DEFAULT_BACKGROUND_VMR))
    return Float64(initial_storage) / Float64(air_mass)
end

"""
    report_dict(report::BudgetReport) -> Dict{String, Any}

The JSON report of a budget check. The keys of `check_trendy_npp_rh_run.py`
are kept (`days`, `tracers`, `max_relative_budget_error`, `tolerance`,
`passed`, and per tracer under `checks`: `expected_storage_change_kg`,
`actual_storage_change_kg`, `relative_error`, `absolute_tolerance_kg` (the
strict tolerance) and `passed`); the other keys add the operational level,
the carrier, the error as ppm of the carrier air, the column check and the
skipped tracers. `days` is the budget interval in days.
"""
function report_dict(report::BudgetReport)
    tol = report.tolerance
    checks = Dict{String, Any}()
    for b in report.tracers
        checks[String(b.tracer)] = Dict{String, Any}(
            "carrier_vmr" => _json_number(b.carrier),
            "air_mass_kg" => _json_number(b.air_mass),
            "initial_storage_kg" => _json_number(b.initial_storage),
            "final_storage_kg" => _json_number(b.final_storage),
            "expected_storage_change_kg" => _json_number(b.expected_change),
            "actual_storage_change_kg" => _json_number(b.actual_change),
            "relative_error" => _json_number(b.relative_error),
            "error_ppm" => _json_number(1e6 * b.error_vmr),
            "carrier_ulp_kg" => _json_number(b.carrier_ulp),
            "absolute_tolerance_kg" => _json_number(b.strict_tolerance),
            "operational_tolerance_kg" => _json_number(b.operational_tolerance),
            "minimum_column_mean" => _json_number(b.minimum_column_mean),
            "column_checked" => b.column_checked,
            "finite" => b.finite,
            "passed" => b.strict_passed,
            "operational_passed" => b.operational_passed)
    end
    return Dict{String, Any}(
        "config" => report.config_path,
        "files" => length(report.files),
        "first_file" => first(report.files),
        "last_file" => last(report.files),
        "start" => string(report.start),
        "end" => string(report.stop),
        "days" => Dates.value(report.stop - report.start) / 86_400_000,
        "frames" => report.frames,
        "tracers" => length(report.tracers),
        "emission" => report.emission,
        "max_relative_budget_error" => _json_number(report.max_relative_error),
        "tolerance" => _tolerance_text(tol),
        "strict_relative_tolerance" => tol.strict,
        "strict_carrier_ulps" => tol.strict_carrier_ulps,
        "operational_relative_tolerance" => tol.operational,
        "operational_carrier_ulps" => tol.operational_carrier_ulps,
        "passed" => report.passed,
        "operational_passed" => report.operational_passed,
        "skipped" => Dict{String, Any}(String(k) => v for (k, v) in report.skipped),
        "checks" => checks)
end

# JSON has no NaN or Inf: a missing or non-finite number is written as null.
_json_number(x::Real) = isfinite(x) ? Float64(x) : nothing

_tolerance_text(tol::BudgetTolerance) =
    "strict: $(tol.strict) relative plus $(_ulps(tol.strict_carrier_ulps)) run-precision ULP of each " *
    "initial carrier; operational: $(tol.operational) relative plus $(_ulps(tol.operational_carrier_ulps))"
_ulps(k) = isinteger(k) ? string(Int(k)) : string(k)

function Base.show(io::IO, ::MIME"text/plain", report::BudgetReport)
    println(io, "BudgetReport: ", length(report.tracers), " tracers, ", report.start, " to ",
            report.stop, " (", report.frames, " frames in ", length(report.files), " files)")
    println(io, "  strict ", report.passed ? "passed" : "FAILED", ", operational ",
            report.operational_passed ? "passed" : "FAILED", ", max relative error ",
            report.max_relative_error)
    isempty(report.skipped) || print(io, "  skipped: ", join(first.(report.skipped), ", "))
end

# ---------------------------------------------------------------------------
# Carrier drawdown
# ---------------------------------------------------------------------------

"""
    carrier_drawdown(run) -> Dict{Symbol, Float64}

Largest cumulative loss of global mixing ratio that each tracer's sources
cause over the run, for checking before a run that a carrier (the initial
`background`) is large enough to absorb a net sink such as NPP uptake:

```math
D = \\max\\Big(0,\\; -\\min_{t_0 \\le t \\le T} \\frac{E(t_0, t)}{M}\\Big),
```

with ``E(t_0, t)`` the storage the tracer's runtime source adds from the run
start ``t_0 = 0`` to ``t`` ([`RuntimeSurfaceSources`](@ref)), ``T`` the run
length and ``M`` the run's initial air mass (the air mass of the first window
of the first binary, [`RunContext`](@ref)). ``E`` is accumulated piece by
piece of `Operators.emission_segments`: at every slice time for static and
stepwise sources, whose rate is constant between slice times, and in addition
at every advection step for conservative-mean and linearly interpolated
sources and in windows whose float-type steps do not tile them (the steps of
[`RuntimeSurfaceSources`](@ref)). The minimum is taken at the ends of the
pieces, so it is the run's minimum at its step ends, or lower where a slice
time falls inside a step. A carrier ``\\chi_0``
covers the run when ``D < \\chi_0``. Tracers without a surface flux have
``D = 0``. `run` is a run config (path or `Dict`) or a [`RunContext`](@ref);
the run needs no output.

Replaces the `npp_max_global_drawdown_ppm` audit of
`prepare_trendy_s3_npp_rh.py`, which used whole-day sums of the flux file,
`M_air = 0.02896546` and a fixed dry-air mass of 5.1353e18 kg.
"""
function carrier_drawdown(run)
    ctx = run isa RunContext ? run : RunContext(run)
    drawdown = Dict{Symbol, Float64}()
    for spec in ctx.tracer_specs
        src = _runtime_source(ctx, spec, _flux_span(0.0, ctx.run_seconds))
        drawdown[spec.name] = src === nothing ? 0.0 : _maximum_storage_loss(src, ctx) / ctx.air_mass
    end
    return drawdown
end

# max(0, −min_t E(0, t)) over the ends of the emission pieces of the run.
function _maximum_storage_loss(src, ctx::RunContext)
    cumulative = Ref(0.0)
    lowest = Ref(0.0)
    _foreach_emission_piece(src, ctx, 0.0, ctx.run_seconds) do rate, duration
        cumulative[] += rate * duration
        lowest[] = min(lowest[], cumulative[])
    end
    return lowest[] < 0 ? -lowest[] : 0.0
end
