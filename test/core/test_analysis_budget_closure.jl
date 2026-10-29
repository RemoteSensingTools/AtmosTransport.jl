"""
`AtmosTransport.Diagnostics.budget_closure` and `carrier_drawdown`
(`src/Diagnostics/budget_closure.jl`, `src/Diagnostics/run_context.jl`) on a tiny
synthetic run written into `mktempdir()`.

The run: two C3 transport binaries of one 24 h window each (4 advection
steps in the first, 3 in the second) with zero mass fluxes (so transport moves
nothing) and a uniform air mass of 1e16 kg per cell and level; a daily
`cs_native` flux file; four tracers, run in Float64 by `run_driven_simulation`
with daily snapshot files:

- `npp`: carrier 1e-3, a uniform flux of -2e-8 kg m⁻² s⁻¹ on day 1 and
  +1e-8 on day 2 (its global integral is known analytically from 4πR²);
- `rh`: carrier 1e-5, a positive flux that varies from cell to cell;
- `lin`: carrier 1e-5, the `rh` flux linearly interpolated between the daily
  stamps 0, 24 and 48 h (`temporal_scheme = "linear"`, sampled at every step
  start);
- `passive`: carrier 4e-4, no flux.

Asserts:
- `RunContext` reads the run as the runner does: Float64, start, 2 binaries,
  the windows with their own step counts and step lengths, 48 h, the air mass
  1e16 × 54 cells × 2 levels; `run_snapshot_paths` names the daily files of
  the binaries that take a snapshot; it rejects the window ranges the runner
  rejects (a later start on a cubed-sphere grid, a partial range over several
  binaries, a range outside a binary's windows) and a cubed-sphere run without
  `[tracers]`; a lat-lon run of the legacy single-tracer form (`[run].tracer_name`
  and `[init]`) that starts at window 2 has the runner's single tracer and the
  air mass of window 1, on which the runner packs the initial state;
- in a Float32 run with seven steps per 24 h window the step is
  Float32(86400)/Float32(7), applied from the run clock's step starts
  86400 j/7, so the steps add 86400.001953125 s: the expected change of the
  stepwise `npp` source counts the last step's 2.8e-4 s in the day-2 slice
  (1e-14 against the source's slice totals), the linear source is summed at those step starts, and
  the drawdown stops at the end of the day-1 part;
- the expected change of `npp` equals the analytic
  86400 × 4πR² × (q₁ + q₂) × M_air / M_CO2 (1e-12), that of `rh` the direct
  sum over cells and days (1e-13), and both equal the actual change of the
  snapshot totals (1e-12); `lin` closes only with each window's own step
  (its step-start sum with the first window's step throughout differs);
  `passive` has zero expected change and closes;
- the carrier term is eps(FT(carrier)) × air mass for the run's float type;
  a Float64 error between one and four ULPs fails strict and passes operational;
  the tolerances follow `BudgetTolerance`; `report_dict` keeps the keys of the retired
  Python report and writes valid JSON;
- `carrier_drawdown` is the day-1 loss of `npp` divided by the air mass
  (1e-12), with 0 for the tracers that only gain; for the same flux read as a
  conservative-mean source (linear between the stamps) it is the loss at the
  step end nearest the rate's zero (18 h, analytic, 1e-12), not at the slice
  time (24 h);
- failures: a perturbed final total fails the strict and operational levels;
  an exhausted carrier (a negative column mean), a non-finite value and a
  positive carrier without column means in the snapshots fail;
  a missing daily file and an incomplete or unscheduled frame list are errors
  unless `require_complete = false`; unknown tracers and invalid tolerances
  are errors; a decaying tracer and a tracer absent from the snapshots are
  skipped;
- `scripts/diagnostics/tracer_budget_closure.jl`: the `series` mode writes
  the JSON report and returns false (exit 1) when the gated level fails; the
  single-file mode reads the snapshot `time` axis as hours since the run start
  and closes the budget of one daily file;
- an `AbstractExpectedEmission` subtype plugs in by dispatch, and the
  harness emulator of the retired Python arithmetic
  (`scripts/validation/flux_port_parity/emulators/legacy_budget_emission.jl`,
  deleted with the harness in P11) gives the runtime change times
  0.02896546 / DRY_AIR_MOLAR_MASS when the file areas are the mesh areas.
"""

using Test
using Dates
using JSON3
using Logging
using NCDatasets
using AtmosTransport
using AtmosTransport: HybridSigmaPressure
using AtmosTransport.Diagnostics
using AtmosTransport.Parameters: DRY_AIR_MOLAR_MASS, SPECIES_MOLAR_MASS
using TOML
const MD = AtmosTransport.MetDrivers

module BudgetClosureScript
include(joinpath(@__DIR__, "..", "..", "scripts", "diagnostics", "tracer_budget_closure.jl"))
end

include(joinpath(@__DIR__, "..", "..", "scripts", "validation", "flux_port_parity",
                 "emulators", "legacy_budget_emission.jl"))
using .LegacyBudgetEmission

const NC, NZ = 3, 2
const AIR_PER_CELL = 1e16
const NPP_DAILY = (-2e-8, 1e-8)
const RADIUS = AtmosTransport.Parameters.EARTH_RADIUS

# One 24 h window per entry of `air` (air mass per cell and level), zero mass fluxes.
function write_binary(path, steps; air = (AIR_PER_CELL,))
    vc = HybridSigmaPressure(fill(100.0, NZ + 1), collect(range(0.0, 1.0; length = NZ + 1)))
    writer = MD.open_streaming_cs_transport_binary(path, NC, 6, NZ, length(air), vc; planet_radius = RADIUS,
                                                   FT = Float64, dt_met_seconds = 86400.0,
                                                   steps_per_window = steps, mass_basis = :dry,
                                                   surface_pressure_basis = TotalSurfacePressure())
    try
        for m in air
            window = (; m = ntuple(_ -> fill(m, NC, NC, NZ), 6),
                      am = ntuple(_ -> zeros(NC + 1, NC, NZ), 6),
                      bm = ntuple(_ -> zeros(NC, NC + 1, NZ), 6),
                      cm = ntuple(_ -> zeros(NC, NC, NZ + 1), 6),
                      ps = ntuple(_ -> fill(1e5, NC, NC), 6))
            MD.write_streaming_cs_window!(writer, window, NC, 6)
        end
    finally
        MD.close_streaming_transport_binary!(writer)
    end
    return path
end

# Slice k (stamp 24(k-1) h) of RH_FLUX; a stepwise source uses slices 1 and 2 over the two days.
rh_density(i, j, p, k) = 1e-9 * (1 + 0.1i + 0.01j + 0.001p) * (1.0, 1.5, 2.5)[k]

function write_flux(path, areas)
    NCDataset(path, "c") do ds
        defDim(ds, "Xdim", NC); defDim(ds, "Ydim", NC); defDim(ds, "nf", 6); defDim(ds, "time", 3)
        defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours since 2021-01-01 00:00:00",
                             "calendar" => "proleptic_gregorian"))[:] = [0.0, 24.0, 48.0]
        attrib = Dict("units" => "kg m-2 s-1")
        npp = defVar(ds, "NPP_FLUX", Float64, ("Xdim", "Ydim", "nf", "time"); attrib)
        rh = defVar(ds, "RH_FLUX", Float64, ("Xdim", "Ydim", "nf", "time"); attrib)
        for (k, day) in enumerate((1, 2, 2))
            npp[:, :, :, k] = fill(NPP_DAILY[day], NC, NC, 6)
            rh[:, :, :, k] = [rh_density(i, j, p, k) for i in 1:NC, j in 1:NC, p in 1:6]
        end
        defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"))[:, :, :] = areas
    end
    return path
end

flux_cfg(file, variable; scheme = "stepwise") =
    Dict{String, Any}("kind" => "cs_native", "file" => file, "variable" => variable,
                      "species" => "co2",
                      "time_varying" => true, "temporal_scheme" => scheme)

function run_config(dir, binaries, flux)
    return Dict{String, Any}(
        "input" => Dict{String, Any}("binary_paths" => binaries, "start_date" => "2021-01-01"),
        "architecture" => Dict{String, Any}("use_gpu" => false),
        "numerics" => Dict{String, Any}("float_type" => "Float64"),
        "advection" => Dict{String, Any}("scheme" => "upwind"),
        "diffusion" => Dict{String, Any}("kind" => "none"),
        "convection" => Dict{String, Any}("kind" => "none"),
        "tracers" => Dict{String, Any}(
            "npp" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 1e-3),
                                       "surface_flux" => flux_cfg(flux, "NPP_FLUX")),
            "rh" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 1e-5),
                                      "surface_flux" => flux_cfg(flux, "RH_FLUX")),
            "lin" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 1e-5),
                                       "surface_flux" => flux_cfg(flux, "RH_FLUX"; scheme = "linear")),
            "passive" => Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform", "background" => 4e-4))),
        "output" => Dict{String, Any}("format" => "netcdf", "split" => "daily",
                                      "snapshot_interval_hours" => 24,
                                      "path" => joinpath(dir, "output", "tiny_{YYYYMMDD}.nc")))
end

quietly(f) = with_logger(f, NullLogger())

# Copy the run's snapshot files into `dir` and apply `edit!(ds, file_index)` to each copy.
function edited_copies(paths, dir, edit!)
    mkpath(dir)
    copies = String[]
    for (k, path) in enumerate(paths)
        copy = joinpath(dir, basename(path))
        cp(path, copy; force = true)
        NCDataset(ds -> edit!(ds, k), copy, "a")
        push!(copies, copy)
    end
    return joinpath(dir, "tiny_*.nc")
end

struct FixedEmission <: AbstractExpectedEmission
    value::Float64
end
Diagnostics.expected_storage_change(m::FixedEmission, ::RunContext, spec, t0, t1) =
    spec.name === :passive ? 0.0 : m.value

@testset "budget closure of a synthetic run" begin
    dir = mktempdir()
    binaries = [write_binary(joinpath(dir, "tiny_20210101.bin"), 4),
                write_binary(joinpath(dir, "tiny_20210102.bin"), 3)]
    mesh = MD.driver_grid(MD.TransportBinaryDriver(binaries[1]; FT = Float64)).horizontal
    areas = repeat(Float64.(mesh.cell_areas), 1, 1, 6)
    flux = write_flux(joinpath(dir, "flux.nc"), areas)
    cfg = run_config(dir, binaries, flux)
    quietly() do
        redirect_stdout(devnull) do
            redirect_stderr(() -> AtmosTransport.run_driven_simulation(cfg), devnull)
        end
    end

    # The run context is the runner's view of the run.
    ctx = RunContext(cfg)
    air = AIR_PER_CELL * NC * NC * 6 * NZ
    @test AtmosTransport.floattype(ctx) === Float64
    @test ctx.reference_time == DateTime(2021, 1, 1)
    @test ctx.binary_paths == binaries
    @test ctx.run_seconds == 2 * 86400.0
    @test ctx.windows == [(; binary = 1, start = 0.0, length = 86400.0, steps = 4, step = 21600.0),
                          (; binary = 2, start = 86400.0, length = 86400.0, steps = 3, step = 28800.0)]
    @test ctx.air_mass == air
    @test [s.name for s in ctx.tracer_specs] == [:lin, :npp, :passive, :rh]
    paths = run_snapshot_paths(ctx)
    @test paths == [joinpath(dir, "output", "tiny_20210101.nc"), joinpath(dir, "output", "tiny_20210102.nc")]
    @test all(isfile, paths)
    # Daily files exist only for binaries that take a snapshot.
    for (hours, owners) in (([0.0], [1]), ([48.0], [2]), ([0.0, 48.0], [1, 2]), ([24.0], [1]))
        scheduled = deepcopy(cfg)
        delete!(scheduled["output"], "snapshot_interval_hours")
        scheduled["output"]["hours"] = hours
        @test run_snapshot_paths(RunContext(scheduled)) == paths[owners]
    end

    report = quietly(() -> budget_closure(cfg))
    @test report.files == paths
    @test (report.start, report.stop, report.frames) == (DateTime(2021, 1, 1), DateTime(2021, 1, 3), 3)
    @test report.passed && report.operational_passed
    @test isempty(report.skipped)
    b = Dict(t.tracer => t for t in report.tracers)
    @test [t.tracer for t in report.tracers] == [:lin, :npp, :passive, :rh]

    scale = DRY_AIR_MOLAR_MASS / SPECIES_MOLAR_MASS.co2
    npp_analytic = 86400 * 4π * RADIUS^2 * sum(NPP_DAILY) * scale
    rh_direct = 86400 * scale * sum(rh_density(i, j, p, d) * mesh.cell_areas[i, j]
                                    for i in 1:NC, j in 1:NC, p in 1:6, d in 1:2)
    @test b[:npp].expected_change ≈ npp_analytic rtol = 1e-12
    @test b[:rh].expected_change ≈ rh_direct rtol = 1e-13
    @test b[:passive].expected_change == 0
    for t in (:npp, :rh, :lin)
        @test b[t].actual_change ≈ b[t].expected_change rtol = 1e-12
        @test b[t].relative_error < 1e-12
    end
    # The linear source is sampled at each step start: 4 steps on day 1, 3 on day 2.
    lin_src = only(AtmosTransport.build_surface_flux_sources(ctx.grid, (ctx.tracer_specs[1],), Float64;
                                                             reference_time = ctx.reference_time))
    by_window = AtmosTransport.Operators.integrated_emission(lin_src, 0.0, 86400.0, 21600.0) +
                AtmosTransport.Operators.integrated_emission(lin_src, 86400.0, 172800.0, 28800.0)
    first_step = AtmosTransport.Operators.integrated_emission(lin_src, 0.0, 172800.0, 21600.0)
    @test b[:lin].expected_change ≈ by_window rtol = 1e-14
    @test abs(first_step / b[:lin].actual_change - 1) > 1e-3
    @test abs(b[:passive].actual_change) <= 1e-12 * b[:passive].initial_storage
    @test b[:npp].initial_storage ≈ 1e-3 * air rtol = 1e-14
    @test b[:npp].carrier == 1e-3 && b[:rh].carrier == 1e-5 && b[:passive].carrier == 4e-4
    @test b[:npp].air_mass == air
    @test b[:npp].carrier_ulp == eps(Float64(1e-3)) * air
    @test b[:npp].strict_tolerance == 1e-4 * abs(b[:npp].expected_change) + b[:npp].carrier_ulp
    @test b[:npp].operational_tolerance == 1e-3 * abs(b[:npp].expected_change) + 4 * b[:npp].carrier_ulp
    @test b[:npp].error_vmr == (b[:npp].actual_change - b[:npp].expected_change) / air
    @test 0 < b[:npp].minimum_column_mean < 1e-3 && b[:npp].finite

    # The carrier allowance follows the Float64 run, not the much wider
    # Float32 spacing. An error of two Float64 carrier ULPs is outside strict
    # (one ULP) but inside operational (four ULPs).
    ulp64 = eps(Float64(b[:npp].carrier)) * air
    ulp_gate = BudgetTolerance(strict = 0, operational = 0)
    f64_gate = quietly(() -> budget_closure(ctx; tracers = [:npp],
        emission = FixedEmission(b[:npp].actual_change + 2ulp64), tolerance = ulp_gate))
    gated = only(f64_gate.tracers)
    @test gated.carrier_ulp == ulp64
    @test !gated.strict_passed && gated.operational_passed

    # JSON report: the Python keys plus the operational level.
    d = report_dict(report)
    # The defining module moved; persisted reports keep their method label.
    @test d["emission"] == "AtmosTransport.Analysis.RuntimeSurfaceSources()"
    for key in ("days", "tracers", "max_relative_budget_error", "tolerance", "passed", "checks")
        @test haskey(d, key)
    end
    @test d["days"] == 2.0 && d["tracers"] == 4 && d["frames"] == 3
    for key in ("expected_storage_change_kg", "actual_storage_change_kg", "relative_error",
                "absolute_tolerance_kg", "passed", "operational_passed", "error_ppm", "carrier_ulp_kg")
        @test haskey(d["checks"]["npp"], key)
    end
    @test d["checks"]["npp"]["absolute_tolerance_kg"] == b[:npp].strict_tolerance
    @test JSON3.read(JSON3.write(d))["checks"]["rh"]["passed"] == true

    # Drawdown: the npp loss of day 1 (the cumulative minimum) over the air mass.
    drawdown = quietly(() -> carrier_drawdown(ctx))
    @test drawdown[:npp] ≈ -86400 * 4π * RADIUS^2 * NPP_DAILY[1] * scale / air rtol = 1e-12
    @test drawdown[:rh] == 0 && drawdown[:lin] == 0 && drawdown[:passive] == 0
    @test !signbit(drawdown[:rh])

    # A perturbed final total fails both levels (1 % ≫ 1e-3).
    glob = edited_copies(paths, joinpath(dir, "perturbed"), (ds, k) -> begin
        k == 2 && (ds["npp_total_mass"][end] = ds["npp_total_mass"][end] + 0.01 * abs(b[:npp].expected_change))
    end)
    bad = quietly(() -> budget_closure(cfg; output_glob = glob))
    @test !bad.passed && !bad.operational_passed
    @test !only(t for t in bad.tracers if t.tracer === :npp).strict_passed
    @test only(t for t in bad.tracers if t.tracer === :rh).strict_passed
    @test bad.max_relative_error ≈ 0.01 rtol = 1e-9
    # A looser tolerance passes it operationally.
    loose = quietly(() -> budget_closure(cfg; output_glob = glob, tolerance = BudgetTolerance(operational = 0.02)))
    @test !loose.passed && loose.operational_passed

    # An exhausted carrier (one negative column mean) and a non-finite value fail.
    glob = edited_copies(paths, joinpath(dir, "exhausted"), (ds, k) -> begin
        k == 2 && (ds["npp_column_mean"][1, 1, 1, end] = -1f-6)
        k == 2 && (ds["rh_column_mean"][2, 2, 2, end] = NaN32)
    end)
    sick = quietly(() -> budget_closure(cfg; output_glob = glob))
    s = Dict(t.tracer => t for t in sick.tracers)
    @test s[:npp].minimum_column_mean ≈ -1e-6 rtol = 1e-6
    @test !s[:npp].strict_passed && !s[:npp].operational_passed && s[:npp].finite
    @test !s[:rh].finite && !s[:rh].strict_passed
    @test s[:passive].strict_passed
    @test JSON3.write(report_dict(sick)) isa String
    # Without column means a positive carrier cannot be shown to survive.
    glob = edited_copies(paths, joinpath(dir, "no_column"), (ds, k) -> renameVar(ds, "npp_column_mean", "npp_cm"))
    unseen = Dict(t.tracer => t for t in quietly(() -> budget_closure(cfg; output_glob = glob)).tracers)
    @test !unseen[:npp].column_checked && isnan(unseen[:npp].minimum_column_mean)
    @test !unseen[:npp].strict_passed && unseen[:rh].strict_passed && unseen[:rh].column_checked

    # Missing files, incomplete and unscheduled frames.
    partial = joinpath(dir, "partial")
    mkpath(partial)
    cp(paths[1], joinpath(partial, basename(paths[1])))
    @test_throws ArgumentError quietly(() -> budget_closure(cfg; output_glob = joinpath(partial, "tiny_*.nc")))
    first_day = quietly(() -> budget_closure(cfg; output_glob = joinpath(partial, "tiny_*.nc"),
                                             require_complete = false))
    f = Dict(t.tracer => t for t in first_day.tracers)
    @test first_day.stop == DateTime(2021, 1, 2) && first_day.frames == 2
    @test f[:npp].expected_change ≈ 86400 * 4π * RADIUS^2 * NPP_DAILY[1] * scale rtol = 1e-12
    @test f[:npp].actual_change ≈ f[:npp].expected_change rtol = 1e-12
    moved = deepcopy(cfg)
    moved["output"]["path"] = joinpath(dir, "nowhere", "tiny_{YYYYMMDD}.nc")
    @test_throws ArgumentError quietly(() -> budget_closure(moved))
    hourly = deepcopy(cfg)
    hourly["output"]["snapshot_interval_hours"] = 12      # schedules frames the files lack
    @test_throws ArgumentError quietly(() -> budget_closure(hourly))
    @test_throws ArgumentError quietly(() -> budget_closure(cfg; output_glob = joinpath(dir, "output", "none_*.nc")))

    # Selection and skipping.
    only_rh = quietly(() -> budget_closure(cfg; tracers = ["rh"]))
    @test [t.tracer for t in only_rh.tracers] == [:rh]
    @test_throws ArgumentError quietly(() -> budget_closure(cfg; tracers = [:co2]))
    decaying = deepcopy(cfg)
    decaying["chemistry"] = Dict{String, Any}("kind" => "decay",
                                              "half_lives_seconds" => Dict{String, Any}("passive" => 1e5))
    dr = quietly(() -> budget_closure(decaying))
    @test [t.tracer for t in dr.tracers] == [:lin, :npp, :rh]
    @test first(only(dr.skipped)) === :passive
    extra = deepcopy(cfg)
    extra["tracers"]["unwritten"] = Dict{String, Any}("init" => Dict{String, Any}("kind" => "uniform"))
    er = quietly(() -> budget_closure(extra))
    @test first.(er.skipped) == [:unwritten] && er.passed
    @test haskey(report_dict(er)["skipped"], "unwritten")
    @test_throws ArgumentError quietly(() -> budget_closure(decaying; tracers = [:passive]))

    # Tolerances.
    @test_throws ArgumentError BudgetTolerance(strict = -1e-4)
    @test_throws ArgumentError BudgetTolerance(operational_carrier_ulps = Inf)
    @test BudgetTolerance().strict == 1e-4 && BudgetTolerance().operational_carrier_ulps == 4

    # Other expected-emission methods plug in by dispatch.
    fixed = quietly(() -> budget_closure(ctx; emission = FixedEmission(b[:npp].actual_change)))
    fx = Dict(t.tracer => t for t in fixed.tracers)
    @test fx[:npp].expected_change == b[:npp].actual_change
    @test fx[:rh].expected_change == b[:npp].actual_change && !fx[:rh].strict_passed
    @test occursin("FixedEmission", fixed.emission)

    # Harness emulator of the Python arithmetic: file areas = mesh areas here.
    legacy = quietly(() -> budget_closure(ctx; emission = LegacyDailyFluxFileEmission()))
    factor = LegacyBudgetEmission.LEGACY_DRY_AIR_MOLAR_MASS / DRY_AIR_MOLAR_MASS
    for t in legacy.tracers
        t.tracer === :lin && continue        # the emulator knows daily stepwise sums only
        @test t.expected_change ≈ factor * b[t.tracer].expected_change rtol = 1e-13
        @test t.actual_change == b[t.tracer].actual_change
    end
    scaled = quietly(() -> budget_closure(ctx; emission = ScaledEmission(RuntimeSurfaceSources(), factor)))
    @test [t.expected_change for t in scaled.tracers] == factor .* [t.expected_change for t in report.tracers]

    # A Float32 run steps by Float32(L)/Float32(n) from the clock's starts 86400 j / 7:
    # seven steps of a 24 h window add 86400.001953125 s, and the last one reaches
    # δ = step − 86400/7 ≈ 2.8e-4 s into the day-2 slice.
    seven = write_binary(joinpath(dir, "seven_20210101.bin"), 7)
    cfg32 = deepcopy(cfg)
    cfg32["input"]["binary_paths"] = [seven]
    cfg32["numerics"]["float_type"] = "Float32"
    ctx32 = quietly(() -> RunContext(cfg32))
    step32 = Float64(Float32(86400) / Float32(7))
    @test AtmosTransport.floattype(ctx32) === Float32
    @test ctx32.windows == [(; binary = 1, start = 0.0, length = 86400.0, steps = 7, step = step32)]
    @test 7 * step32 - 86400 == 0.001953125
    spec32 = Dict(s.name => s for s in ctx32.tracer_specs)
    source32(name) = only(AtmosTransport.build_surface_flux_sources(ctx32.grid, (spec32[name],), Float32;
                                                                    reference_time = ctx32.reference_time))
    # Global rates of the Float32 source's day-1 and day-2 slices (Float32 cell
    # rates, so not the analytic 4πR² q M_air/M_CO2 beyond about 1e-8).
    rate1, rate2 = (AtmosTransport.Operators.total_emission_rate(source32(:npp), t) for t in (0.0, 86400.0))
    npp32 = quietly(() -> expected_storage_change(RuntimeSurfaceSources(), ctx32, spec32[:npp], 0.0, 86400.0))
    day2 = step32 - 86400 / 7
    day1 = 7 * step32 - day2
    @test npp32 ≈ rate1 * day1 + rate2 * day2 rtol = 1e-14
    @test abs(npp32 / (rate1 * 86400) - 1) > 1e-8          # not the exact day
    # The linear source at the run clock's step starts 86400 j / 7, each applied for step32.
    lin32 = source32(:lin)
    lin_steps = sum(r * τ for j in 0:6
                    for (r, τ) in AtmosTransport.Operators.emission_segments(lin32, 86400 * (j / 7), step32))
    @test quietly(() -> expected_storage_change(RuntimeSurfaceSources(), ctx32, spec32[:lin], 0.0, 86400.0)) ≈
          lin_steps rtol = 1e-14
    # The cumulative minimum: the end of the day-1 part of the last step.
    @test quietly(() -> carrier_drawdown(ctx32))[:npp] ≈ -rate1 * day1 / ctx32.air_mass rtol = 1e-14
    # A conservative-mean source changes rate between slice times and the run adds it
    # step by step: npp's linear reconstruction runs from -2e-8 (0 h) to +1e-8 (24 h),
    # so the cumulative storage is lowest at the step end nearest its zero (16 h):
    # 18 h, the end of day-1 step 3 (the day's net, at 24 h, is less negative).
    cons_cfg = deepcopy(cfg)
    cons_cfg["tracers"] = Dict{String, Any}("cons" => Dict{String, Any}(
        "init" => Dict{String, Any}("kind" => "uniform", "background" => 1e-3),
        "surface_flux" => flux_cfg(flux, "NPP_FLUX"; scheme = "conservative")))
    cumulative(t) = 4π * RADIUS^2 * scale * (NPP_DAILY[1] * t + (NPP_DAILY[2] - NPP_DAILY[1]) * t^2 / (2 * 86400))
    cons_drawdown = quietly(() -> carrier_drawdown(RunContext(cons_cfg)))[:cons]
    @test cons_drawdown ≈ -cumulative(64800.0) / air rtol = 1e-12
    @test cons_drawdown > -cumulative(86400.0) / air * (1 + 1e-3)

    # Window ranges the runner rejects: a later start on a cubed-sphere grid, a
    # partial range over several binaries.
    two = write_binary(joinpath(dir, "two_20210101.bin"), 4; air = (AIR_PER_CELL, 2AIR_PER_CELL))
    late_cs = deepcopy(cfg)
    late_cs["input"]["binary_paths"] = [two]
    late_cs["run"] = Dict{String, Any}("start_window" => 2)
    @test_throws ArgumentError quietly(() -> RunContext(late_cs))
    partial_range = deepcopy(cfg)
    partial_range["run"] = Dict{String, Any}("start_window" => 2)
    @test_throws ArgumentError quietly(() -> RunContext(partial_range))
    uneven = deepcopy(cfg)                       # stop_window covers binary 1, not binary 2
    uneven["input"]["binary_paths"] = [binaries[1], two]
    uneven["run"] = Dict{String, Any}("stop_window" => 1)
    @test_throws ArgumentError quietly(() -> RunContext(uneven))
    no_tracers = deepcopy(cfg)                   # cubed-sphere runs need [tracers]
    delete!(no_tracers, "tracers")
    @test_throws ArgumentError quietly(() -> RunContext(no_tracers))

    # A lat-lon run of the legacy single-tracer form ([run].tracer_name and [init],
    # no [tracers]) that starts at window 2: the runner packs its initial state on
    # window 1, so the air mass is window 1's.
    ll_grid = AtmosGrid(LatLonMesh(; FT = Float64, Nx = 4, Ny = 2),
                        HybridSigmaPressure(Float64[0, 1], Float64[0, 1]), CPU(); FT = Float64)
    ll_windows = [(; m = fill(m, 4, 2, 1), am = zeros(5, 2, 1), bm = zeros(4, 3, 1),
                   cm = zeros(4, 2, 2), ps = fill(95000.0, 4, 2)) for m in (2e15, 3e15)]
    ll_bin = joinpath(dir, "ll_20210101.bin")
    write_transport_binary(ll_bin, ll_grid, ll_windows; FT = Float64, dt_met_seconds = 86400.0,
                           half_dt_seconds = 43200.0, steps_per_window = 2, mass_basis = :dry,
                           surface_pressure_basis = TotalSurfacePressure(),
                           source_flux_sampling = :window_start_endpoint, flux_sampling = :window_constant)
    legacy_cfg = Dict{String, Any}(
        "input" => Dict{String, Any}("binary_paths" => [ll_bin], "start_date" => "2021-01-01"),
        "numerics" => Dict{String, Any}("float_type" => "Float64"),
        "run" => Dict{String, Any}("tracer_name" => "co2", "start_window" => 2),
        "init" => Dict{String, Any}("kind" => "uniform", "background" => 4e-4))
    ctx_ll = quietly(() -> RunContext(legacy_cfg))
    @test ctx_ll.air_mass == 2e15 * 8
    @test only(ctx_ll.tracer_specs).init_spec isa AtmosTransport.Models.InitialConditionIO.UniformInitialConditionSpec
    @test [(s.name, s.init_cfg["background"], isempty(s.surface_flux_cfg)) for s in ctx_ll.tracer_specs] ==
          [(:co2, 4e-4, true)]
    nonuniform = TransportTracerSpec(:blob,
        Dict{String, Any}("kind" => "gaussian_blob"), Dict{String, Any}())
    @test nonuniform.init_spec isa AtmosTransport.Models.InitialConditionIO.GaussianBlobInitialConditionSpec
    @test Diagnostics._initial_carrier(nonuniform, 6.0, 3.0) == 2.0
    @test ctx_ll.windows == [(; binary = 1, start = 0.0, length = 86400.0, steps = 2, step = 43200.0)]
    @test ctx_ll.run_seconds == 86400.0
    for (start, stop) in ((3, nothing), (1, 3), (0, 1), (2, 1),   # outside 1 ≤ start ≤ stop ≤ 2
                          (true, nothing), (1, 2.0))              # not integers
        bad_range = deepcopy(legacy_cfg)
        bad_range["run"]["start_window"] = start
        stop === nothing || (bad_range["run"]["stop_window"] = stop)
        @test_throws ArgumentError quietly(() -> RunContext(bad_range))
    end

    # Command line: series mode (gate and JSON report) and single-file mode.
    cfg_path = joinpath(dir, "run.toml")
    open(io -> TOML.print(io, cfg), cfg_path, "w")
    json = joinpath(dir, "report.json")
    script(f) = quietly(() -> redirect_stdout(f, devnull))
    @test script(() -> BudgetClosureScript.main_series([cfg_path, "--report", json]))
    written = JSON3.read(read(json, String))
    @test written.passed && written.tracers == 4
    @test written.checks.npp.expected_storage_change_kg == b[:npp].expected_change
    perturbed = joinpath(dir, "perturbed", "tiny_*.nc")
    @test !script(() -> BudgetClosureScript.main_series([cfg_path, "--output-glob", perturbed]))
    @test script(() -> BudgetClosureScript.main_series([cfg_path, "--output-glob", perturbed,
                                                        "--gate", "operational", "--operational", "0.02"]))
    @test script(() -> BudgetClosureScript.main_series([cfg_path, "--output-glob", perturbed,
                                                        "--tracers", "rh,passive"]))
    @test NCDataset(BudgetClosureScript.output_seconds, paths[2]) == [48 * 3600.0]
    single = script(() -> BudgetClosureScript.main(paths[1], cfg_path))
    @test single[:npp].Δb ≈ 86400 * 4π * RADIUS^2 * NPP_DAILY[1] * scale rtol = 1e-10
    @test abs(single[:npp].rel) < 1e-10 && abs(single[:rh].rel) < 1e-10
    @test single[:npp].inj_ratio ≈ 1 rtol = 1e-10
end
