#!/usr/bin/env julia
# ===========================================================================
# budget_closure_parity.jl: compare `Analysis.budget_closure` with a report of
# the retired `check_trendy_npp_rh_run.py` (design memo 2026-10-10, §4.3)
#
# Runs the budget of one run three ways on the same snapshot files and writes
# one JSON report:
#
# 1. legacy: the expected changes of the Python script
#    (`LegacyDailyFluxFileEmission`: flux-file `cell_area`, whole-day sums,
#    M_air = 0.02896546). Expected and actual changes must equal the Python
#    report to about 1e-12 (summation order only), with the same pass flags.
# 2. runtime: the library default (`RuntimeSurfaceSources`): the run's own
#    sources with the current constants (DRY_AIR_MOLAR_MASS = 0.0289644).
# 3. run_constant: the runtime changes rescaled by 0.02896546 / DRY_AIR_MOLAR_MASS,
#    the storage scale a run made before commit d552d3e0 applied. Against
#    the legacy numbers this isolates the cell-area and integration
#    differences; against the runtime numbers, the constant.
#
# The carrier term of the tolerance uses the run's initial air mass from the
# binary; the Python script used total_mass(t0) / carrier. Their difference
# is reported (`tolerance_max_relative_difference`).
#
# Usage:
#   julia --project=. scripts/validation/flux_port_parity/budget_closure_parity.jl \
#       --config <run.toml> --reference <validation.json> --json <report.json> \
#       [--output-glob <glob>]
#
# Exit status 0 when the legacy run reproduces the reference (expected and
# actual changes within 1e-10 relative, identical pass flags), 1 otherwise.
# ===========================================================================

using AtmosTransport
using AtmosTransport.Analysis
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS
using AtmosTransport.AtomicFiles: atomic_publish
using JSON3, Logging, Printf

include(joinpath(@__DIR__, "emulators", "legacy_budget_emission.jl"))
using .LegacyBudgetEmission

const PARITY_RTOL = 1e-10

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        key in ("--config", "--reference", "--json", "--output-glob") ||
            error("unknown argument $(key)")
        i < length(args) || error("$(key) needs a value")
        opts[key[3:end]] = args[i + 1]
        i += 2
    end
    for key in ("config", "reference", "json")
        haskey(opts, key) || error("--$(key) is required")
    end
    return opts
end

reldiff(a, b) = a == b ? 0.0 : abs(a - b) / max(abs(a), abs(b))
# JSON has no NaN or Inf.
num(x) = isfinite(x) ? Float64(x) : nothing

function compare_with_reference(report::BudgetReport, ref)
    rows = Dict{String, Any}()
    max_expected = max_actual = max_tolerance = max_relative = 0.0
    flags_equal = true
    for b in report.tracers
        c = ref.checks[b.tracer]
        row = Dict{String, Any}(
            "expected_relative_difference" => reldiff(b.expected_change, c.expected_storage_change_kg),
            "actual_relative_difference" => reldiff(b.actual_change, c.actual_storage_change_kg),
            "tolerance_relative_difference" => reldiff(b.strict_tolerance, c.absolute_tolerance_kg),
            "relative_error" => num(b.relative_error),
            "reference_relative_error" => c.relative_error,
            "passed" => b.strict_passed,
            "reference_passed" => c.passed)
        max_expected = max(max_expected, row["expected_relative_difference"])
        max_actual = max(max_actual, row["actual_relative_difference"])
        max_tolerance = max(max_tolerance, row["tolerance_relative_difference"])
        max_relative = max(max_relative, abs(b.relative_error - c.relative_error))
        flags_equal &= b.strict_passed == c.passed
        rows[String(b.tracer)] = row
    end
    length(report.tracers) == length(ref.checks) || (flags_equal = false)
    for row in values(rows), key in ("expected_relative_difference", "actual_relative_difference",
                                     "tolerance_relative_difference")
        row[key] = num(row[key])
    end
    return Dict{String, Any}(
        "expected_max_relative_difference" => num(max_expected),
        "actual_max_relative_difference" => num(max_actual),
        "tolerance_max_relative_difference" => num(max_tolerance),
        "relative_error_max_abs_difference" => num(max_relative),
        "pass_flags_equal" => flags_equal,
        "per_tracer" => rows)
end

summary(report::BudgetReport) = Dict{String, Any}(
    "emission" => report.emission,
    "passed" => report.passed,
    "operational_passed" => report.operational_passed,
    "tracers" => length(report.tracers),
    "strict_passing" => count(b -> b.strict_passed, report.tracers),
    "max_relative_budget_error" => num(report.max_relative_error),
    "worst_tracer" => String(report.tracers[argmax([isnan(b.relative_error) ? Inf : b.relative_error
                                                    for b in report.tracers])].tracer),
    "failed" => [String(b.tracer) for b in report.tracers if !b.strict_passed],
    "relative_error" => Dict(String(b.tracer) => num(b.relative_error) for b in report.tracers),
    "signed_relative_error" => Dict(String(b.tracer) =>
        num((b.actual_change - b.expected_change) / max(abs(b.expected_change), 1.0)) for b in report.tracers))

function main(args)
    opts = parse_args(args)
    ref = JSON3.read(read(opts["reference"], String))
    glob = get(opts, "output-glob", nothing)
    ctx = RunContext(opts["config"])
    quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))
    timed(f) = (t = time(); r = quiet(f); (r, time() - t))

    legacy, t_legacy = timed(() -> budget_closure(ctx; output_glob = glob, emission = LegacyDailyFluxFileEmission()))
    runtime, t_runtime = timed(() -> budget_closure(ctx; output_glob = glob))
    factor = LegacyBudgetEmission.LEGACY_DRY_AIR_MOLAR_MASS / DRY_AIR_MOLAR_MASS
    tabulated = TabulatedEmission(Dict(b.tracer => b.expected_change for b in runtime.tracers))
    run_constant, t_run = timed(() -> budget_closure(ctx; output_glob = glob,
                                                     emission = ScaledEmission(tabulated, factor)))

    parity = compare_with_reference(legacy, ref)
    within(x) = x isa Real && x <= PARITY_RTOL       # `nothing` marks a non-finite difference
    ok = within(parity["expected_max_relative_difference"]) &&
         within(parity["actual_max_relative_difference"]) && parity["pass_flags_equal"]
    area_effect = Dict(String(b.tracer) => reldiff(b.expected_change, l.expected_change)
                       for (b, l) in zip(run_constant.tracers, legacy.tracers))
    out = Dict{String, Any}(
        "config" => abspath(opts["config"]),
        "reference" => abspath(opts["reference"]),
        "files" => length(legacy.files),
        "days" => report_dict(legacy)["days"],
        "reference_summary" => Dict(k => ref[k] for k in keys(ref) if k != :checks),
        "legacy" => merge(summary(legacy), Dict("seconds" => t_legacy)),
        "legacy_vs_reference" => parity,
        "runtime" => merge(summary(runtime), Dict("seconds" => t_runtime)),
        "run_constant" => merge(summary(run_constant), Dict("seconds" => t_run,
                                                            "factor" => factor)),
        "run_constant_vs_legacy_expected_max_relative_difference" => num(maximum(values(area_effect))),
        "run_constant_vs_legacy_expected" => Dict(k => num(v) for (k, v) in area_effect),
        "parity_passed" => ok,
        "runtime_report" => report_dict(runtime))
    mkpath(dirname(abspath(opts["json"])))
    atomic_publish(abspath(opts["json"])) do tmp
        open(io -> JSON3.pretty(io, out), tmp, "w")
    end
    @printf("legacy:       %d/%d strict pass, max relative error %.6g (%s); reference %.6g, passed %s\n",
            count(b -> b.strict_passed, legacy.tracers), length(legacy.tracers), legacy.max_relative_error,
            summary(legacy)["worst_tracer"], ref.max_relative_budget_error, ref.passed)
    @printf("  vs Python:  expected %.3e, actual %.3e, tolerance %.3e (max relative differences); pass flags equal %s\n",
            something(parity["expected_max_relative_difference"], NaN),
            something(parity["actual_max_relative_difference"], NaN),
            something(parity["tolerance_max_relative_difference"], NaN), parity["pass_flags_equal"])
    @printf("runtime:      %d/%d strict pass, max relative error %.6g (%s)\n",
            count(b -> b.strict_passed, runtime.tracers), length(runtime.tracers), runtime.max_relative_error,
            summary(runtime)["worst_tracer"])
    @printf("run constant: %d/%d strict pass, max relative error %.6g (%s); expected vs legacy max %.3e\n",
            count(b -> b.strict_passed, run_constant.tracers), length(run_constant.tracers),
            run_constant.max_relative_error, summary(run_constant)["worst_tracer"], maximum(values(area_effect)))
    println("report: ", abspath(opts["json"]), "; parity ", ok ? "PASSED" : "FAILED")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
