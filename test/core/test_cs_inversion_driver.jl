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
    @test occursin("forward Lin–Rood runs default", _config_error(with_advection(
        "[advection]\nscheme = \"linrood\"\nvertical = \"fv3_kord8\"\n")))
    @test occursin("`vertical` is required", _config_error(with_advection(
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
                         "vertical" => "upwind"))
        scheme = scheme_of(section)
        @test AT.require_adjoint_support(scheme) === scheme
    end
    for section in (Dict("scheme" => "slopes"),
                    Dict("scheme" => "ppm", "vertical" => "fv3_kord8"),
                    Dict("scheme" => "ppm", "vertical" => "fv3_kord8_signed"),
                    Dict("scheme" => "linrood", "vertical" => "fv3_kord8"),
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
