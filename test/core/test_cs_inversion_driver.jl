#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.C4 — end-to-end inversion driver smoke test.
#
# Loads `config/inversions/example_synthetic.toml`, runs the
# driver's `run_inversion`, and verifies the result. This is the
# Plan 26 Phase C "wraps" gate — every B + C component (covariance,
# preconditioner, optimizer, iteration log, preconditioned 4D-Var
# entrypoint) participates.
# ---------------------------------------------------------------------------

using Test
using Logging: NullLogger, with_logger

# The driver is a script, not a module — include it so `run_inversion`
# is exposed at the top level.
include(joinpath(@__DIR__, "..", "..", "scripts", "inversions", "cs_4dvar.jl"))

const CONFIG = joinpath(@__DIR__, "..", "..", "config", "inversions",
                         "example_synthetic.toml")
const SHIPPED_ADVECTION = "[advection]\nscheme = \"ppm\"\nlimiter = \"none\"\n"

# Run the driver on a config given as TOML text.
function _run_config_text(text::AbstractString)
    mktempdir() do dir
        path = joinpath(dir, "config.toml")
        write(path, text)
        return run_inversion(path)
    end
end

# The `ArgumentError` message a config raises ("" if it runs).
function _config_error(text::AbstractString)
    try
        _run_config_text(text)
        return ""
    catch e
        e isa ArgumentError || rethrow()
        return e.msg
    end
end

@testset "cs_4dvar.jl driver — synthetic smoke test" begin
    @test isfile(CONFIG)

    result = run_inversion(CONFIG)
    @test result isa AT.CS4DVarSolveResult

    # The synthetic problem has zero air mass flux (constant flow),
    # so the observation residual is whatever the initial mass field
    # produces minus the configured observation value. The optimizer
    # should reduce the cost from initial.
    initial_cost = result.cost_history[1]
    @test result.last.cost <= initial_cost
    @test result.last.cost < initial_cost  # strict — the example is
                                            # not at a local min at χ=0.

    # Cost-decomposition invariant: total cost == obs + background.
    @test result.last.cost ≈ result.last.observation_cost +
                              result.last.background_cost atol = 1e-9
    # Preconditioned mode: at χ = 0 the background term is exactly 0
    # (0.5 ‖0‖² = 0). The initial cost recorded in cost_history[1]
    # therefore equals the initial observation cost.
    @test result.cost_history[1] >= 0

    # Iteration log was requested via `log = true` in the TOML.
    @test result.log isa AT.CSIterationLog
    @test !isempty(result.log)
    @test result.log[1].iteration == 0

    # Cost-decomposition invariant on every log row.
    for entry in result.log
        @test entry.cost ≈ entry.observation_cost + entry.background_cost atol = 1e-9
    end

    # L-BFGS produces a monotone non-increasing cost trajectory.
    @test issorted(result.cost_history, rev = true) ||
          all(diff(result.cost_history) .<= 1e-12)
end

@testset "cs_4dvar.jl driver — config dispatch coverage" begin
    # Each TOML knob the driver dispatches on must produce a working
    # solve. We override the default config in-place via a tempdir
    # copy.
    base = read(CONFIG, String)

    function _run_with_override(override::AbstractString)
        mktempdir() do dir
            path = joinpath(dir, "config.toml")
            write(path, override)
            return run_inversion(path)
        end
    end

    # Switch optimizer to gradient descent.
    gd_cfg = replace(base,
        "kind = \"lbfgs\"" => "kind = \"gradient_descent\"\ninitial_step = 0.25")
    result_gd = _run_with_override(gd_cfg)
    @test result_gd isa AT.CS4DVarSolveResult

    # Switch covariance to diagonal.
    diag_cfg = replace(base,
        "kind = \"isotropic_gaussian\"" => "kind = \"diagonal\"")
    result_diag = _run_with_override(diag_cfg)
    @test result_diag isa AT.CS4DVarSolveResult

    # Unconditioned mode (preconditioner.enabled = false).
    uncond_cfg = replace(base,
        "enabled = true" => "enabled = false")
    result_uncond = _run_with_override(uncond_cfg)
    @test result_uncond isa AT.CS4DVarSolveResult
end

# ---------------------------------------------------------------------------
# Regression: control.initial = "background" in preconditioned mode must
# set χ = 0 (so the physical starting point is x_b). The previous build
# copied prec.background into the χ slot, which seeded χ = x_b and
# leaked a nonzero 0.5‖χ‖² prior cost on iteration 0.
# ---------------------------------------------------------------------------

@testset "cs_4dvar.jl driver — initial='background' in preconditioned mode → χ=0" begin
    base = read(CONFIG, String)
    # Force a non-zero `preconditioner.background_value` so seeding
    # χ from `prec.background` (the buggy behavior) would produce a
    # detectably nonzero initial 0.5‖χ‖² cost.
    cfg = replace(base,
        "background_value = 0.0" => "background_value = 0.7",
        "initial = \"zeros\""    => "initial = \"background\"")
    mktempdir() do dir
        path = joinpath(dir, "bg.toml")
        write(path, cfg)
        result = run_inversion(path)
        # χ = 0 at iteration 0 ⇒ background term 0.5‖χ‖² = 0. The
        # pre-fix copy of `x_b = 0.7` into the χ slot would have
        # produced 0.5 · 6 · (Nc²) · 0.7² > 0 as the initial bg
        # cost.
        @test result.log isa AT.CSIterationLog
        @test result.log[1].iteration == 0
        @test result.log[1].background_cost == 0.0
        # And the iteration-0 total cost equals the observation
        # cost since the background term is exactly zero.
        @test result.log[1].cost ≈ result.log[1].observation_cost atol = 1e-12
    end
end

# ---------------------------------------------------------------------------
# Regression: [preconditioner].enabled = true with no [covariance]
# section used to silently degrade to unconditioned mode (prec became
# nothing). Now throws a clear ArgumentError.
# ---------------------------------------------------------------------------

@testset "cs_4dvar.jl driver — preconditioner.enabled w/o covariance rejected" begin
    cfg = """
    [mesh]
    Nc = 3
    Hp = 3
    float_type = "Float64"

    [time]
    nsteps = 2
    dt_seconds = 2.0

    [meteo]
    source = "synthetic_constant"

    [advection]
    scheme = "ppm"
    limiter = "none"

    [[observations.entries]]
    step = 2
    objective = "layer_mean"
    panel = 1
    i = 2
    j = 2
    level = 3
    value = 0.05
    sigma = 0.2

    [control]
    name = "both_steps"
    steps = [1, 2]
    normalize = true
    initial = "zeros"

    [preconditioner]
    enabled = true
    optim_type = "linear"
    background_value = 0.0

    [optimizer]
    kind = "lbfgs"
    iterations = 2
    """
    @test occursin("requires a [covariance] section", _config_error(cfg))
end

# ---------------------------------------------------------------------------
# Issue #16: `[advection]` used to be ignored (the driver hard-coded
# `PPMScheme(NoLimiter())`). It now goes through the forward run's
# `advection_spec` + `materialize` in the inversion context, and schemes
# without a cubed-sphere adjoint are rejected.
# ---------------------------------------------------------------------------

@testset "cs_4dvar.jl driver — [advection] selects the scheme (#16)" begin
    base = read(CONFIG, String)
    @test occursin(SHIPPED_ADVECTION, base)
    with_advection(block) = replace(base, SHIPPED_ADVECTION => block)
    scheme_of(block) = _build_advection_scheme(TOML.parse(block)["advection"])

    @test scheme_of(SHIPPED_ADVECTION) == AT.PPMScheme(AT.NoLimiter())
    @test scheme_of("[advection]\nscheme = \"ppm\"\n") == AT.PPMScheme()
    @test scheme_of("[advection]\nscheme = \"ppm\"\nlimiter = \"cw84\"\n") ==
          AT.PPMScheme(AT.CW84Limiter())
    @test scheme_of("[advection]\nscheme = \"upwind\"\n") == AT.UpwindScheme()
    @test scheme_of("[advection]\nscheme = \"linrood\"\nvertical = \"upwind\"\n") isa
          AT.LinRoodPPMScheme{5, AT.UpwindScheme}
    @test scheme_of("[advection]\nscheme = \"linrood\"\nppm_order = 7\nvertical = \"upwind\"\n") isa
          AT.LinRoodPPMScheme{7, AT.UpwindScheme}
    # The FV3 vertical profile has an adjoint (OF-94); an omitted `vertical`
    # selects it, as in a forward run with the same table.
    fv3_linrood = AT.LinRoodPPMScheme{5, AT.PPMScheme{AT.MonotoneLimiter, AT.FV3ScalarProfile{true}}}
    @test scheme_of("[advection]\nscheme = \"linrood\"\nvertical = \"fv3_kord8\"\n") isa fv3_linrood
    @test scheme_of("[advection]\nscheme = \"linrood\"\n") isa fv3_linrood
    @test scheme_of("[advection]\nscheme = \"linrood\"\nvertical = \"fv3_kord8_signed\"\n") isa
          AT.LinRoodPPMScheme{5, AT.PPMScheme{AT.MonotoneLimiter, AT.FV3ScalarProfile{false}}}

    # A Lin–Rood inversion runs end to end through the driver.
    lr = _run_config_text(with_advection(
        "[advection]\nscheme = \"linrood\"\nvertical = \"upwind\"\n"))
    @test lr isa AT.CS4DVarSolveResult
    @test lr.last.cost < lr.cost_history[1]

    # Invalid selections and schemes without an adjoint fail before any transport.
    @test occursin("Unknown [advection] scheme",
                   _config_error(with_advection("[advection]\nscheme = \"no_such_scheme\"\n")))
    @test occursin("`scheme` is required", _config_error(with_advection("[advection]\n")))
    @test occursin("[advection] table is required", _config_error(with_advection("")))
    @test occursin("SlopesScheme with MonotoneLimiter has no adjoint",
                   _config_error(with_advection("[advection]\nscheme = \"slopes\"\n")))
    @test occursin("FV3 vertical profile", _config_error(with_advection(
        "[advection]\nscheme = \"ppm\"\nvertical = \"fv3_kord8\"\n")))
    # The synthetic meteorology has three layers; the FV3 profile needs four.
    @test occursin("at least 4 layers", _config_error(with_advection(
        "[advection]\nscheme = \"linrood\"\n")))
    @test occursin("NoAdvection", _config_error(with_advection("[advection]\nscheme = \"none\"\n")))
    @test occursin("Unknown [advection] limiter", _config_error(with_advection(
        "[advection]\nscheme = \"ppm\"\nlimiter = \"minmod\"\n")))
end

@testset "require_adjoint_support — runtime [advection] choices (#16)" begin
    style = AT.Models.CubedSphereRuntimeRecipeStyle()
    context = AT.Models.InversionAdvectionContext()
    scheme_of(section) = AT.Models.materialize(AT.Models.advection_spec(section, context), style)
    for section in (Dict("scheme" => "upwind"),
                    Dict("scheme" => "ppm"),
                    Dict("scheme" => "ppm", "limiter" => "cw84"),
                    Dict("scheme" => "ppm", "limiter" => "none"),
                    Dict("scheme" => "linrood", "vertical" => "upwind"),
                    Dict("scheme" => "linrood", "ppm_order" => 7,
                         "vertical" => "upwind"),
                    Dict("scheme" => "linrood"),
                    Dict("scheme" => "linrood", "vertical" => "fv3_kord8"),
                    Dict("scheme" => "linrood", "ppm_order" => 7,
                         "vertical" => "fv3_kord8_signed"))
        scheme = scheme_of(section)
        @test AT.require_adjoint_support(scheme) === scheme
    end
    for section in (Dict("scheme" => "slopes"),
                    Dict("scheme" => "ppm", "vertical" => "fv3_kord8"),
                    Dict("scheme" => "ppm", "vertical" => "fv3_kord8_signed"),
                    Dict("scheme" => "none"))
        @test_throws ArgumentError AT.require_adjoint_support(scheme_of(section))
    end
    # Schemes outside the runtime vocabulary follow the same rule.
    @test AT.require_adjoint_support(AT.SlopesScheme(AT.NoLimiter())) == AT.SlopesScheme(AT.NoLimiter())
    @test_throws ArgumentError AT.require_adjoint_support(AT.PPMScheme(AT.Operators.PositivityLimiter()))
end

@testset "cs_4dvar.jl driver — unknown tables and keys are rejected (#16)" begin
    base = read(CONFIG, String)
    @test occursin("[diffusion] is not supported",
                   _config_error(base * "\n[diffusion]\nkind = \"constant\"\nvalue = 1.0\n"))
    @test occursin("[convection] is not supported",
                   _config_error(base * "\n[convection]\nkind = \"tm5\"\n"))
    @test occursin("Unknown top-level inversion config option(s): output",
                   _config_error(base * "\n[output]\npath = \"result.toml\"\n"))
    @test occursin("Unknown [mesh] option(s): Hpp",
                   _config_error(replace(base, "Hp = 3" => "Hpp = 3")))
    @test occursin("Unknown [advection] option(s): check_binary_cfl",
                   _config_error(replace(base, SHIPPED_ADVECTION =>
                                               SHIPPED_ADVECTION * "check_binary_cfl = true\n")))
    @test occursin("Unknown [[observations.entries]] option(s): sigmaa",
                   _config_error(replace(base, "sigma = 0.2" => "sigmaa = 0.2")))
    @test occursin("Unknown [optimizer] option(s): iteration",
                   _config_error(replace(base, "iterations = 8" => "iteration = 8")))
    # Known keys that the selected kind does not read are warnings.
    @test_logs (:warn, r"initial_step ignored with kind = \"lbfgs\"") match_mode = :any begin
        _run_config_text(replace(base, "m = 5" => "m = 5\ninitial_step = 0.25"))
    end
end

@testset "cs_4dvar.jl driver — rejects bad TOML" begin
    # Missing required fields, unknown enum values, etc.
    base = read(CONFIG, String)

    function _expect_throw(override::AbstractString)
        mktempdir() do dir
            path = joinpath(dir, "bad.toml")
            write(path, override)
            @test_throws Exception run_inversion(path)
        end
    end

    _expect_throw(replace(base,
        "kind = \"isotropic_gaussian\"" => "kind = \"mystery_kernel\""))
    _expect_throw(replace(base,
        "kind = \"lbfgs\"" => "kind = \"newton_raphson\""))
    _expect_throw(replace(base,
        "objective = \"layer_mean\"" => "objective = \"point_sample\""))
    _expect_throw(replace(base,
        "optim_type = \"linear\"" => "optim_type = \"sigmoid\""))

    # Missing config file.
    @test_throws ArgumentError run_inversion(joinpath(tempdir(),
        "does_not_exist_$(rand(UInt64)).toml"))
end

# ---------------------------------------------------------------------------
# Issue #20: an observation file (schema v2) bound with `bind_to_mesh`, point
# records placed in the `synthetic_hydrostatic` column by
# `[observations.placement]` (the forward sampler's parser and rule).
# ---------------------------------------------------------------------------

const HYDROSTATIC_BLOCK = """
[meteo]
source = "synthetic_hydrostatic"
interface_pressures_pa = [1000.0, 10000.0, 20000.0, 30000.0, 45000.0, 60000.0, 75000.0, 85000.0, 92000.0, 97000.0, 100000.0]
specific_humidity = [1e-5, 1e-5, 5e-5, 2e-4, 6e-4, 1.5e-3, 3e-3, 6e-3, 1e-2, 1.4e-2]
"""

function _hydrostatic_config(obs_path; meteo = HYDROSTATIC_BLOCK, observations_extra = "", time_extra = "start = \"2024-06-15T00:00:00\"\n")
    return """
    [mesh]
    Nc = 3
    Hp = 3
    float_type = "Float64"

    [time]
    nsteps = 3
    dt_seconds = 600.0
    $(time_extra)
    $(meteo)

    [advection]
    scheme = "upwind"

    [observations]
    file = "$(obs_path)"
    $(observations_extra)

    [control]
    name = "all_steps"
    steps = [1, 2, 3]
    normalize = true
    initial = "zeros"

    [optimizer]
    kind = "lbfgs"
    iterations = 3
    """
end

function _write_driver_observations(path; extra = AT.CSObservationRecord[])
    mesh = AT.CubedSphereMesh(Nc = 3, Hp = 3, FT = Float64)
    lons, lats = AT.panel_cell_center_lonlat(mesh, 2)
    record(id; kwargs...) = AT.CSObservationRecord(; id, date_components = (2024, 6, 15, 0, 15, 0),
        lat = lats[2, 2], lon = lons[2, 2], value = 1e-7, value_sigma = 1e-8, instrument_type = "test",
        tracer = "CO2", kwargs...)
    records = [record(1; intake_height_m = 10.0),
               record(2; altitude_m = 5000.0, elevation_m = 0.0),
               record(3; pressure_pa = 70000.0, time_window_s = 1200.0),
               record(4; kind = "column", time_window_s = 3600.0),
               record(5; intake_height_m = 10.0, tracer = "CH4"),
               extra...]
    AT.write_observations(path, AT.CSObservationSet(records; time_origin = "2024-06-15 00:00:00"))
    return path
end

@testset "cs_4dvar.jl driver — observation file + synthetic_hydrostatic (#20)" begin
    mktempdir() do dir
        obs_path = _write_driver_observations(joinpath(dir, "obs.nc"))
        cfg = _hydrostatic_config(obs_path; observations_extra = "tracer = \"CO2\"\n")
        result = _run_config_text(cfg)
        @test result isa AT.CS4DVarSolveResult
        last = result.last
        @test [o.id for o in last.observations] == [1, 2, 3, 4]       # the CH4 record is filtered out
        @test last.stencils[1].levels == [10]                          # 10 m intake: surface layer
        @test last.stencils[2].levels == [5]                           # 5 km aircraft: about 540 hPa
        @test last.stencils[3].levels == fill(6, 3) && last.stencils[3].steps == [1, 2, 3]   # 700 hPa: 600-750 hPa
        @test last.stencils[4].levels == zeros(Int, 3)                 # column mean over the window
        @test last.cost < result.cost_history[1]

        # The hydrostatic air mass: m = (1 − q)(p_{k+1} − p_k) A / g.
        parsed = TOML.parse(cfg)
        mesh = AT.CubedSphereMesh(Nc = 3, Hp = 3, FT = Float64)
        meteo = _build_meteo(parsed, mesh, 3, Float64)
        p, q = parsed["meteo"]["interface_pressures_pa"], parsed["meteo"]["specific_humidity"]
        @test meteo.panels_m[2][3 + 1, 3 + 2, 10] ≈
              (1 - q[10]) * (p[11] - p[10]) * AT.cell_area(mesh, 1, 2) / AT.Parameters.STANDARD_GRAVITY rtol = 1e-14
        @test meteo.column_met.p_top == 1000.0 && meteo.column_met.humidity !== nothing

        # Pressure placement needs the humidity unless the basis is dry.
        dry_meteo = replace(HYDROSTATIC_BLOCK, r"specific_humidity = .*\n" => "")
        @test occursin("pressure_basis = \"dry\"", _config_error(_hydrostatic_config(obs_path; meteo = dry_meteo)))
        dry = _run_config_text(_hydrostatic_config(obs_path; meteo = dry_meteo,
            observations_extra = "[observations.placement]\npressure_basis = \"dry\"\n"))
        @test dry isa AT.CS4DVarSolveResult
        # Height first, then pressure; a layer-temperature profile.
        profile = HYDROSTATIC_BLOCK * "temperature_kelvin = [220.0, 220.0, 225.0, 235.0, 250.0, 262.0, 272.0, 280.0, 285.0, 288.0]\n"
        cold = _run_config_text(_hydrostatic_config(obs_path; meteo = profile,
                                                    observations_extra = "tracer = \"CO2\"\n"))
        @test cold.last.stencils[2].levels == [5]

        # Configuration errors.
        no_start = _hydrostatic_config(obs_path; time_extra = "")
        @test occursin("needs [time].start", _config_error(no_start))
        constant = replace(_hydrostatic_config(obs_path), HYDROSTATIC_BLOCK => "[meteo]\nsource = \"synthetic_constant\"\n")
        @test occursin("synthetic_hydrostatic", _config_error(constant))
        @test occursin("apply to source = \"synthetic_hydrostatic\" only",
                       _config_error(replace(read(CONFIG, String), "source = \"synthetic_constant\"" =>
                           "source = \"synthetic_constant\"\nspecific_humidity = [0.0, 0.0, 0.0]")))
        @test occursin("exactly one of", _config_error(replace(_hydrostatic_config(obs_path),
            "[control]" => "[[observations.entries]]\nstep = 1\nobjective = \"column_mean\"\npanel = 1\ni = 1\nj = 1\nvalue = 0.0\nsigma = 1.0\n\n[control]")))
        @test occursin("apply to an observation `file`", _config_error(replace(read(CONFIG, String),
            "[[observations.entries]]" => "[observations]\ntracer = \"CO2\"\n\n[[observations.entries]]")))
        @test occursin("Unknown [observations.placement] option(s): coordinates",
                       _config_error(_hydrostatic_config(obs_path;
                           observations_extra = "[observations.placement]\ncoordinates = [\"pressure\"]\n")))
        @test occursin("has no effect with [meteo].temperature_kelvin",
                       _config_error(_hydrostatic_config(obs_path; meteo = profile,
                           observations_extra = "[observations.placement]\nlayer_height_temperature_kelvin = 270.0\n")))
        @test occursin("[observations.placement].vertical_coordinates",
                       _config_error(_hydrostatic_config(obs_path;
                           observations_extra = "[observations.placement]\nvertical_coordinates = [\"altitude\"]\n")))
        @test occursin("interface_pressures_pa", _config_error(_hydrostatic_config(obs_path;
            meteo = replace(HYDROSTATIC_BLOCK, "[1000.0, 10000.0" => "[10000.0, 1000.0"))))
        @test occursin("[meteo].interface_pressures_pa is required", _config_error(_hydrostatic_config(obs_path;
            meteo = replace(HYDROSTATIC_BLOCK, r"interface_pressures_pa = .*\n" => ""))))
        # A point record without a vertical coordinate: error, or skip and count.
        mktempdir() do sub
            path = _write_driver_observations(joinpath(sub, "obs.nc");
                extra = [AT.CSObservationRecord(id = 9, date_components = (2024, 6, 15, 0, 15, 0), lat = 10.0,
                                                lon = 20.0, value = 1e-7, value_sigma = 1e-8,
                                                instrument_type = "test", tracer = "CO2")])
            @test occursin("cannot place every request", _config_error(_hydrostatic_config(path)))
            skipped = _run_config_text(_hydrostatic_config(path;
                observations_extra = "tracer = \"CO2\"\n[observations.placement]\nunplaced_records = \"skip\"\n"))
            @test [o.id for o in skipped.last.observations] == [1, 2, 3, 4]
        end
        # Out-of-range records follow out_of_range_policy.
        late = joinpath(dir, "late.nc")
        mesh3 = AT.CubedSphereMesh(Nc = 3, Hp = 3, FT = Float64)
        AT.write_observations(late, AT.CSObservationSet([AT.CSObservationRecord(id = 1,
            date_components = (2024, 6, 16, 0, 0, 0), lat = 0.0, lon = 0.0, kind = "column", value = 1e-7,
            value_sigma = 1e-8, instrument_type = "test", tracer = "CO2")]))
        @test occursin("outside [1, 3]", _config_error(_hydrostatic_config(late)))
        @test occursin("no observation is left",
                       _config_error(_hydrostatic_config(late; observations_extra = "out_of_range_policy = \"skip\"\n")))
    end
end

# Issue #20, part 4: column records with a retrieval block are averaging-kernel
# columns; `[observations.placement].max_surface_pressure_mismatch_pa` bounds
# the model minus retrieval surface pressure (the column's is 1000 hPa).
@testset "cs_4dvar.jl driver — averaging-kernel columns, max_surface_pressure_mismatch_pa (#20)" begin
    mktempdir() do dir
        mesh = AT.CubedSphereMesh(Nc = 3, Hp = 3, FT = Float64)
        lons, lats = AT.panel_cell_center_lonlat(mesh, 2)
        retrieval(ps) = AT.Observations.RetrievalColumn(; pressure_pa = collect(range(ps, 0.0; length = 6)),
                                                  pressure_weight = fill(0.2, 5),
                                                  averaging_kernel = [0.6, 0.8, 1.0, 1.1, 1.2],
                                                  prior_profile = fill(1.2e-7, 5))
        column(id, ps) = AT.CSObservationRecord(; id, date_components = (2024, 6, 15, 0, 15, 0), lat = lats[2, 2],
                                                lon = lons[2, 2], kind = "column", value = 1e-7, value_sigma = 1e-8,
                                                instrument_type = "test", tracer = "CO2", retrieval = retrieval(ps))
        path = joinpath(dir, "ak.nc")
        AT.write_observations(path, AT.CSObservationSet([column(1, 99000.0), column(2, 96000.0)]))
        free = _run_config_text(_hydrostatic_config(path))
        @test all(o -> o.objective isa AT.CSRetrievalColumnObjective, free.last.observations)
        @test all(s -> s.assimilated, free.last.stencils)
        @test [s.surface_pressure_mismatch_pa for s in free.last.stencils] ≈ [1000.0, 4000.0] rtol = 1e-9
        limit = "[observations.placement]\nmax_surface_pressure_mismatch_pa = 2000.0\n"
        @test occursin("max_surface_pressure_mismatch_pa", _config_error(_hydrostatic_config(path;
                                                                                observations_extra = limit)))
        skipped = with_logger(NullLogger()) do
            _run_config_text(_hydrostatic_config(path; observations_extra = limit * "unplaced_records = \"skip\"\n"))
        end
        @test [s.assimilated for s in skipped.last.stencils] == [true, false]
        @test occursin("must be a positive pressure", _config_error(_hydrostatic_config(path;
            observations_extra = "[observations.placement]\nmax_surface_pressure_mismatch_pa = -1.0\n")))
        # A column-only inversion on a dry column sets pressure_basis = "dry"
        # without placing points by pressure; the key applies to the layer
        # heights too, so no table rejects it without pressure placement
        # (OF-117).
        dry_meteo = replace(HYDROSTATIC_BLOCK, r"specific_humidity = .*\n" => "")
        height_only = "[observations.placement]\nvertical_coordinates = [\"height_above_ground\"]\n"
        @test occursin("pressure_basis = \"dry\"", _config_error(_hydrostatic_config(path; meteo = dry_meteo,
                                                                                    observations_extra = height_only)))
        dry = _run_config_text(_hydrostatic_config(path; meteo = dry_meteo,
                                                   observations_extra = height_only * "pressure_basis = \"dry\"\n"))
        @test all(o -> o.objective.pressure_basis isa AT.Observations.DryInterfaces, dry.last.observations)
        keys = Dict{String, Any}("vertical_coordinates" => ["height_above_ground"], "pressure_basis" => "dry")
        for label in ("[output.observations]", "[observations.placement]")
            @test AT.Observations.parse_vertical_placement(keys, label).pressure_basis isa AT.Observations.DryInterfaces
        end
    end
end

# Issue #20 follow-up (D30): level columns (OCO-2's hat projection), co-sampled
# flagged records and cell averaging through the driver's TOML keys.
@testset "cs_4dvar.jl driver — level columns, flagged_records, cell_averaging (D30)" begin
    mktempdir() do dir
        mesh = AT.CubedSphereMesh(Nc = 3, Hp = 3, FT = Float64)
        lons, lats = AT.panel_cell_center_lonlat(mesh, 2)
        σ = vcat(1e-4, collect(1:19) ./ 19)
        level(ps) = AT.Observations.RetrievalColumn(; pressure_pa = reverse(σ .* ps), pressure_weight = fill(0.05, 20),
                                              averaging_kernel = collect(range(1.0, 0.4; length = 20)),
                                              prior_profile = fill(1.2e-7, 20), mapping = AT.Observations.HatLevelProjection())
        record(id, sec, flag) = AT.CSObservationRecord(; id, date_components = (2024, 6, 15, 0, 15, sec), flag,
                                                       lat = lats[2, 2], lon = lons[2, 2], kind = "column",
                                                       value = 1e-7, value_sigma = 1e-8, model_sigma = 0.5e-8,
                                                       instrument_type = "OCO2_OG", tracer = "CO2",
                                                       retrieval = level(99000.0))
        path = joinpath(dir, "levels.nc")
        AT.write_observations(path, AT.CSObservationSet([record(1, 0, 0), record(2, 10, 0), record(3, 20, 1)]))
        quiet(f) = with_logger(f, NullLogger())
        base = quiet(() -> _run_config_text(_hydrostatic_config(path)))
        @test [o.id for o in base.last.observations] == [1, 2]
        @test all(o -> o.objective.retrieval.mapping isa AT.Observations.HatLevelProjection, base.last.observations)
        cosampled = quiet(() -> _run_config_text(_hydrostatic_config(path;
            observations_extra = "flagged_records = \"cosample\"\n")))
        @test [o.assimilate for o in cosampled.last.observations] == [true, true, false]
        # The co-sample changes nothing the optimizer sees: bit for bit
        # single-threaded; with threads the atomics of the halo and Lin-Rood
        # adjoint kernels make the gradient reproducible to rounding only.
        if Threads.nthreads() == 1
            @test cosampled.cost_history == base.cost_history
            @test cosampled.last.gradients[1][2] == base.last.gradients[1][2]
        else
            @test cosampled.cost_history ≈ base.cost_history rtol = 1e-13
            @test cosampled.last.gradients[1][2] ≈ base.last.gradients[1][2] rtol = 1e-13
        end
        averaged = quiet(() -> _run_config_text(_hydrostatic_config(path;
            observations_extra = "[observations.cell_averaging]\nmode = \"average\"\ntime_tolerance_s = 60.0\n")))
        @test [o.assimilate for o in averaged.last.observations] == [false, false, true]
        @test averaged.last.observations[end].objective isa AT.CSCellAverageObjective
        @test averaged.last.observations[end].sigma ≈ sqrt(1e-16 / 2 + 0.25e-16) rtol = 1e-12
        inflated = quiet(() -> _run_config_text(_hydrostatic_config(path;
            observations_extra = "[observations.cell_averaging]\nmode = \"inflate_model_error\"\n")))
        @test [o.sigma for o in inflated.last.observations] ≈ fill(sqrt(1e-16 + 2 * 0.25e-16), 2) rtol = 1e-12
        @test occursin("[observations.cell_averaging].mode", _config_error(_hydrostatic_config(path;
            observations_extra = "[observations.cell_averaging]\nmode = \"median\"\n")))
        @test occursin("flagged_records", _config_error(_hydrostatic_config(path;
            observations_extra = "flagged_records = \"keep\"\n")))
    end
end
